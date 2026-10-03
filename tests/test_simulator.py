"""Verification of the simulator: cost model, invariants, queueing theory, behaviour.

The tests are layered the way a pre-silicon verification plan would be:

1. unit      – the cost model against hand-calculated numbers
2. invariant – properties that must hold for *any* run (conservation, ordering)
3. analytic  – the event engine against closed-form queueing results
4. behaviour – the qualitative effects the model exists to show
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from disagg_sim import (ACCELERATORS, LINKS, MODELS, CostModel, LengthDist, ModelSpec,
                        SimConfig, poisson_workload, simulate, summarise)
from disagg_sim.workload import dump_workload, load_workload

H100 = ACCELERATORS["h100"]
L8B, L70B = MODELS["llama3-8b"], MODELS["llama3-70b"]


def small_run(**kw):
    cfg = SimConfig(**{k: v for k, v in kw.items() if k not in ("rate", "n", "seed")})
    wl = poisson_workload(kw.get("rate", 3.0), kw.get("n", 300), LengthDist(2048, 0.5),
                          LengthDist(128, 0.5), seed=kw.get("seed", 7))
    return wl, simulate(cfg, wl)


# ── 1. unit: cost model ──────────────────────────────────────────────────
def test_parameter_counts_match_published_sizes():
    assert L8B.params == pytest.approx(8.03e9, rel=0.01)
    assert L70B.params == pytest.approx(70.6e9, rel=0.01)


def test_kv_bytes_per_token():
    # 2 (K,V) x layers x kv_heads x head_dim x 2 bytes
    assert L8B.kv_bytes_per_token == 2 * 32 * 8 * 128 * 2 == 131072
    assert L70B.kv_bytes_per_token == 2 * 80 * 8 * 128 * 2 == 327680


def test_decode_is_memory_bound_and_prefill_compute_bound():
    cm = CostModel(L70B, H100, n_devices=4)
    assert cm.decode([2048]).bound == "memory"
    assert cm.decode([2048] * 16).bound == "memory"
    assert cm.prefill([2048]).bound == "compute"


def test_decode_step_is_roughly_weights_over_bandwidth():
    cm = CostModel(L8B, H100, step_overhead=0.0)
    t = cm.decode([1]).time
    assert t == pytest.approx(L8B.weight_bytes_read(1) / (H100.mem_bw * H100.bw_eff), rel=0.01)


def test_step_weight_traffic_reads_embedding_rows_not_the_table():
    # Layers + LM head are read in full every step; the embedding table only by the rows looked up.
    per_layer = 2 * 4096 * 4096 + 2 * 4096 * 1024 + 3 * 4096 * 14336
    streamed = (32 * per_layer + 128256 * 4096) * 2
    assert L8B.weight_bytes_streamed == streamed == 15_009_316_864
    assert L8B.weight_bytes_read(16) == streamed + 16 * 4096 * 2
    # Residency still holds both tables: the difference is the input embedding, 1.05 GB.
    assert L8B.weight_bytes_total - L8B.weight_bytes_streamed == 128256 * 4096 * 2 == 1_050_673_152
    cm = CostModel(L8B, H100)
    assert cm.decode([1000, 2000]).bytes == streamed + 2 * 8192 + (3000 + 2) * 131072
    assert cm.prefill([300, 200]).bytes == streamed + 500 * 8192 + 500 * 131072


def test_decode_attention_includes_the_new_token_itself():
    cm = CostModel(L8B, H100)
    # Each new token attends to its cached context and to itself: ctx + batch positions.
    assert cm.decode([1000, 2000]).flops == 2 * L8B.matmul_params * 2 + 4 * 32 * 4096 * (3000 + 2)
    # Prefill's s(s+1) already includes the diagonal; decode of one token after a prompt of s
    # therefore adds exactly the attention prefill would add going from s to s+1 tokens.
    att = lambda s: 2 * 32 * 4096 * s * (s + 1)
    assert cm.decode([512]).flops - 2 * L8B.matmul_params == att(513) - att(512)


def test_cost_model_matches_traced_llama3_8b():
    """Pin the closed form against an operator trace of the real model (Torch_Sim_Frontend's recorded run)."""
    fx = json.loads((Path(__file__).parent / "fixtures" / "simfront_llama3_8b.json").read_text())
    d = fx["decode"]
    cm = CostModel(L8B, H100)
    step = cm.decode([d["context"]] * d["batch"])
    assert step.flops + d["rotary_flops"] == d["matmul_flops"]
    assert L8B.weight_bytes_read(d["batch"]) + d["norm_weight_bytes"] == d["weight_bytes"]
    assert 2 * L8B.matmul_params * fx["prefill"]["tokens"] == fx["prefill"]["weight_matmul_flops"]


def test_model_that_does_not_fit_is_rejected():
    with pytest.raises(ValueError):
        CostModel(L70B, H100, n_devices=1).kv_capacity_tokens


# ── 2. invariants ────────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["disagg", "colocated"])
def test_every_request_completes_with_all_its_tokens(mode):
    wl, res = small_run(mode=mode)
    assert all(r.finish is not None for r in wl)
    assert all(r.tokens_out == r.output_len for r in wl)
    assert all(len(r.itls) == r.output_len - 1 for r in wl)


@pytest.mark.parametrize("mode", ["disagg", "colocated"])
def test_timestamps_are_ordered_and_stages_sum_to_e2e(mode):
    wl, _ = small_run(mode=mode)
    for r in wl:
        assert r.arrival <= r.prefill_start <= r.first_token <= r.finish
        if mode == "disagg":
            assert r.first_token <= r.kv_start <= r.kv_ready <= r.decode_start
        assert sum(r.stages().values()) == pytest.approx(r.e2e, abs=1e-9)


def test_kv_memory_is_released():
    _, res = small_run(mode="disagg")
    assert all(i.kv_used == 0 for i in res.instances)


def test_same_seed_same_answer():
    a = summarise(small_run(seed=3)[1])
    b = summarise(small_run(seed=3)[1])
    assert a["latency_s"] == b["latency_s"]


def test_littles_law_holds():
    _, res = small_run(mode="disagg", n=500)
    ll = summarise(res)["littles_law"]
    assert ll["L_measured"] == pytest.approx(ll["lambda_W"], rel=1e-6)


def test_oversized_request_is_rejected_not_hung():
    cfg = SimConfig(model=L8B, devices_per_instance=1)
    cap = CostModel(L8B, H100).kv_capacity_tokens
    wl = poisson_workload(1.0, 3, LengthDist(cap, hi=cap), LengthDist(16))
    res = simulate(cfg, wl)
    assert len(res.rejected) == 3


def test_workload_round_trip(tmp_path):
    wl = poisson_workload(2.0, 20, LengthDist(500, 1.0), LengthDist(50, 1.0), seed=1)
    dump_workload(wl, tmp_path / "w.json")
    back = load_workload(tmp_path / "w.json")
    assert [(r.prompt_len, r.output_len) for r in back] == [(r.prompt_len, r.output_len) for r in wl]


# ── 3. analytic: the KV link is an M/D/1 queue ───────────────────────────
def test_kv_link_waiting_time_matches_pollaczek_khinchine():
    """Make prefill (almost) free so the link sees the Poisson arrival stream,
    give every request the same prompt so service is deterministic, then compare
    the mean wait with the M/D/1 formula  Wq = λ S² / (2 (1 - ρ))."""
    tiny = ModelSpec("tiny", n_layers=1, d_model=64, n_heads=1, n_kv_heads=1, d_ff=64, vocab=8,
                     kv_bytes=100_000.0)        # 12.8 MB of KV per token: the link dominates
    link = replace(LINKS["ib-ndr"], latency=0.0)
    cfg = SimConfig(model=tiny, devices_per_instance=1, n_prefill=1, n_decode=1, link=link,
                    step_overhead=0.0, max_prefill_tokens=1)
    S = link.transfer_time(4 * tiny.kv_bytes_per_token)
    lam = 0.5 / S                                # ρ = 0.5
    wl = poisson_workload(lam, 20000, LengthDist(4), LengthDist(2), seed=11)
    res = simulate(cfg, wl)
    wq_sim = res.link.wait / res.link.transfers
    wq_theory = lam * S * S / (2 * (1 - lam * S))
    assert wq_sim == pytest.approx(wq_theory, rel=0.08)


# ── 4. behaviour ─────────────────────────────────────────────────────────
def test_disaggregation_removes_prefill_decode_interference():
    kw = dict(rate=4.0, n=400)
    co = summarise(small_run(mode="colocated", n_colocated=2, **kw)[1])
    dis = summarise(small_run(mode="disagg", n_prefill=1, n_decode=1, **kw)[1])
    # Colocated decodes stall behind whole prefills: a long ITL tail.
    assert co["latency_s"]["itl"]["p99"] > 4 * dis["latency_s"]["itl"]["p99"]


def test_slow_link_becomes_the_hotspot():
    m = summarise(small_run(mode="disagg", n_prefill=2, rate=6.0,
                            link=LINKS["eth-25g"])[1])
    assert m["hotspots"]["stage"] in ("kv_wait", "kv_transfer")
    assert m["hotspots"]["resource"] == "kv-link"


def test_more_prefill_instances_cut_ttft():
    one = summarise(small_run(n_prefill=1, rate=6.0)[1])
    two = summarise(small_run(n_prefill=2, rate=6.0)[1])
    assert two["latency_s"]["ttft"]["p99"] < one["latency_s"]["ttft"]["p99"]


def test_trace_export_is_chrome_trace_format():
    _, res = small_run(n=30, trace=True)
    ev = res.trace["traceEvents"]
    names = {e["args"]["name"] for e in ev if e["ph"] == "M"}
    assert {"prefill-0", "decode-0", "kv-link"} <= names
    spans = [e for e in ev if e["ph"] == "X"]
    assert spans and all(e["dur"] > 0 for e in spans)


# ── 5. acceleration: faster must mean *identical*, not just close ────────
@pytest.mark.parametrize("extra", [dict(), dict(n_prefill=2, n_decode=2),
                                   dict(n_prefill=2, link=replace(LINKS["eth-25g"], channels=2))])
def test_fast_forward_is_exact(extra):
    a, _ = small_run(rate=5.0, n=400, **extra)
    b, _ = small_run(rate=5.0, n=400, fast_forward=True, **extra)
    for x, y in zip(a, b):
        assert (x.first_token, x.decode_start, x.finish) == (y.first_token, y.decode_start, y.finish)
        assert x.itls == y.itls
        assert x.tokens_out == y.tokens_out


def test_fast_forward_metrics_match():
    m1 = summarise(small_run(rate=4.0, n=300)[1])
    m2 = summarise(small_run(rate=4.0, n=300, fast_forward=True)[1])
    assert m1["latency_s"] == m2["latency_s"]
    assert m1["utilisation"] == pytest.approx(m2["utilisation"], rel=1e-12)


def test_analytic_ceiling_bounds_the_simulated_knee():
    from disagg_sim.search import Workload, analytic_capacity, max_sustainable_rate, sweep
    cfg = SimConfig(fast_forward=True)
    wl = Workload(LengthDist(2048, 0.5), LengthDist(128, 0.5), n=300, seed=7)
    cap = analytic_capacity(cfg, wl)
    found = max_sustainable_rate(cfg, wl, target=0.9, rel_tol=0.05)
    assert 0 < found["rate"] <= cap["ceiling"]
    # the bisection answer agrees with a coarse grid to within its resolution
    grid = sweep(cfg, wl, [found["rate"] * f for f in (0.8, 1.25)])
    assert grid[0][1] >= 0.9 and grid[1][1] < 0.9


def test_javascript_port_matches_python():
    """The browser simulator in the deck must reproduce this package exactly."""
    import json
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    engine = Path(__file__).parent.parent / "web" / "sim_engine.js"
    cases = []
    for mode, kw in [("disagg", dict(n_prefill=1, n_decode=1)),
                     ("disagg", dict(n_prefill=2, n_decode=2, link=replace(LINKS["eth-25g"], channels=2))),
                     ("colocated", dict(n_colocated=2)),
                     ("disagg", dict(n_prefill=1, n_decode=1, power_cap_w=350.0, dvfs=True)),
                     ("colocated", dict(n_colocated=2, power_cap_w=300.0))]:
        wl = poisson_workload(5.0, 300, LengthDist(2048, 0.6), LengthDist(128, 0.6), seed=5)
        rows = [[r.arrival, r.prompt_len, r.output_len] for r in wl]
        cfg = SimConfig(mode=mode, **kw)
        simulate(cfg, wl)
        cases.append({"cfg": {"model": "llama3-70b", "device": "h100", "devicesPerInstance": 4,
                              "mode": mode, "nPrefill": cfg.n_prefill, "nDecode": cfg.n_decode,
                              "nColocated": cfg.n_colocated, "link": {50e9: "ib-ndr", 3.125e9: "eth-25g"}[cfg.link.bandwidth],
                              "linkChannels": cfg.link.channels, "powerCap": cfg.power_cap_w,
                              "dvfs": cfg.dvfs},
                      "rows": rows, "finish": [r.finish for r in wl]})
    script = (f"require({json.dumps(str(engine))});"
              "const S=globalThis.DisaggSim, cases=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify(cases.map(c=>{const r=S.simulate(c.cfg,c.rows);"
              "return Math.max(...r.reqs.map((q,i)=>Math.abs(q.finish-c.finish[i])));})));")
    out = subprocess.run([node, "-e", script], input=json.dumps(cases), capture_output=True,
                         text=True, check=True)
    diffs = json.loads(out.stdout)
    assert all(d == 0 for d in diffs[:3])            # no power cap: bit-identical
    assert all(d < 1e-9 for d in diffs[3:])          # cube roots may differ by an ulp across libms


def test_probes_are_passive():
    """Switching the sampler and tracer on or off must not change any result."""
    quiet = summarise(small_run(sample_dt=1e9, trace=False)[1])
    noisy = summarise(small_run(sample_dt=0.001, trace=True)[1])
    assert quiet["latency_s"] == noisy["latency_s"]
    assert quiet["utilisation"] == noisy["utilisation"]



# ── 6. power and energy ──────────────────────────────────────────────────
def test_energy_accounts_add_up():
    _, res = small_run()
    e = summarise(res)["energy"]
    assert sum(e["breakdown"].values()) == pytest.approx(1.0)
    statics = sum(i.cost.idle_w * res.horizon for i in res.instances)
    dynamic = sum(i.compute_j + i.memory_j for i in res.instances)
    assert e["total_J"] == pytest.approx(statics + dynamic + res.link.energy)


def test_power_cap_is_respected_and_costs_prefill_time():
    free_res = small_run(rate=3.0)[1]
    free = summarise(free_res)
    for i in free_res.instances:                      # the board limit (TDP) always applies
        assert i.peak_power <= i.cost.device.tdp_w * i.cost.n_devices * (1 + 1e-9)
    capped_res = small_run(rate=3.0, power_cap_w=400.0)[1]
    capped = summarise(capped_res)
    for i in capped_res.instances:
        assert i.peak_power <= 400.0 * i.cost.n_devices * (1 + 1e-9)
    assert capped["latency_s"]["ttft"]["p50"] > free["latency_s"]["ttft"]["p50"]     # prefill throttled
    assert capped["latency_s"]["tpot"]["p50"] == pytest.approx(free["latency_s"]["tpot"]["p50"], rel=0.02)
    assert capped["energy"]["J_per_output_token"] < free["energy"]["J_per_output_token"]


def test_dvfs_saves_energy_without_slowing_memory_bound_decode():
    a = summarise(small_run(rate=3.0)[1])
    b = summarise(small_run(rate=3.0, dvfs=True)[1])
    assert b["latency_s"] == a["latency_s"]
    assert b["energy"]["total_J"] < a["energy"]["total_J"]


def test_fast_forward_exact_under_power_cap_and_dvfs():
    a, _ = small_run(rate=5.0, n=300, power_cap_w=350.0, dvfs=True)
    b, _ = small_run(rate=5.0, n=300, power_cap_w=350.0, dvfs=True, fast_forward=True)
    assert [x.finish for x in a] == [y.finish for y in b]


def test_cap_below_idle_is_an_error():
    with pytest.raises(ValueError):
        CostModel(L70B, H100, 4, power_cap_w=50.0).prefill([128])
