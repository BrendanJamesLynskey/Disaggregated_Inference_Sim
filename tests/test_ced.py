"""The Causal Encoder-Decoder (CED) option (brief 11, 2026-10-05).

DeepSeek-V4.1-Flash (arXiv:2609.19969, section 2.2) splits its layers into a causal encoder and a
decoder whose global KV is projected from the last encoder hidden state, so prefill runs only the
encoder and the decoder's K/V projections; the last n_win prompt tokens are replayed through the
decoder (section 3.2.2). The SGLang RFC #39963 moves that replay to the decode instance's first
step so a prefill instance holds only the encoder. These tests pin the cost model, the two
replay placements, the simulator's invariants, that the option is inert when off, and the
JavaScript port (bit-exact).
"""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from disagg_sim.cli import build_parser, config_from_args
from disagg_sim.hardware import (A100_SXM, H100_SXM, LINKS, LLAMA3_8B, LLAMA3_8B_CED, LLAMA3_8B_HYENA,
                                 LLAMA3_70B, LLAMA3_70B_CED, MODELS, KV_PRESETS, CostModel, KVTransit, ModelSpec)
from disagg_sim.metrics import summarise
from disagg_sim.sim import SimConfig, simulate
from disagg_sim.workload import LengthDist, poisson_workload

ON_DECODE = replace(LLAMA3_70B_CED, ced_replay_on="decode")


def wl(n=200, rate=2.0, prompt=4096, output=96, seed=3, pcv=0.6, ocv=0.6, singles=False):
    reqs = poisson_workload(rate, n, LengthDist(prompt, pcv), LengthDist(output, ocv), seed=seed)
    if singles:                              # every 7th request wants a single token
        for r in reqs[::7]:
            r.output_len = 1
    return reqs


def stamps(reqs):
    return [(r.prefill_start, r.first_token, r.prefill_done, r.kv_start, r.kv_ready, r.decode_start, r.finish,
             tuple(r.itls)) for r in reqs]


# ── the option is inert when off ────────────────────────────────────────
def test_off_by_default_and_presets_are_the_only_ced_models():
    assert not any(m.is_ced for k, m in MODELS.items() if not k.endswith("-ced"))
    assert LLAMA3_70B_CED.ced_encoder_layers == 40 and LLAMA3_8B_CED.ced_encoder_layers == 16


@pytest.mark.parametrize("mode", ["disagg", "colocated"])
def test_replay_settings_are_ignored_when_ced_is_off(mode):
    """W and the replay placement change nothing unless ced_encoder_layers > 0: bit-identical runs."""
    a, b = wl(), wl()
    simulate(SimConfig(mode=mode), a)
    simulate(SimConfig(mode=mode, model=replace(LLAMA3_70B, ced_replay=7, ced_replay_on="decode")), b)
    assert stamps(a) == stamps(b)


def test_prefill_only_flag_changes_nothing_for_other_models():
    for m in (LLAMA3_70B, LLAMA3_8B_HYENA, LLAMA3_70B_CED):
        n = 4 if m.n_layers == 80 else 1
        x, y = CostModel(m, H100_SXM, n), CostModel(m, H100_SXM, n, prefill_only=True)
        assert x.kv_capacity_tokens == y.kv_capacity_tokens
        assert x.prefill([2048, 512]) == y.prefill([2048, 512])


# ── the cost model ──────────────────────────────────────────────────────
def test_prompt_token_touches_half_the_parameters():
    """The paper's 8B prefill vs 16B decode activated parameters is a 1:2 ratio; the 40+40 proxy
    gives the same ratio to within the decoder's K/V projections."""
    m = LLAMA3_70B_CED
    d, kv = m.d_model, m.n_kv_heads * m.head_dim
    assert m.ced_prompt_params == 40 * m.params_per_layer + 40 * 2 * d * kv
    assert 0.50 < m.ced_prompt_params / m.matmul_params < 0.51


def test_ced_prefill_flops_closed_form():
    m, s = LLAMA3_70B_CED, 8192
    E = D = 40
    w = m.ced_replay
    want = (2 * m.ced_prompt_params * s + 2 * E * m.d_model * s * (s + 1)
            + 2 * D * m.params_per_layer * w + 2 * D * m.d_model * w * (2 * s - w + 1)
            + 2 * m.vocab * m.d_model * w)
    assert CostModel(m, H100_SXM, 4).prefill([s]).flops == want
    # replay attention is the tail of the causal sum: sum over c = s-w+1..s of 4 d c
    assert m.ced_replay_attention(1, s, w) == sum(4 * m.d_model * c for c in range(s - w + 1, s + 1))


def test_full_replay_costs_the_decoder_only_model_plus_the_extra_kv_projections():
    """With W >= the prompt, every token runs every layer: the baseline's FLOPs, plus the decoder's
    K/V projections from the encoder state on top of the replayed layers' own."""
    s = 1024
    base = CostModel(LLAMA3_70B, H100_SXM, 4).prefill([s])
    ced = CostModel(replace(LLAMA3_70B_CED, ced_replay=4096), H100_SXM, 4).prefill([s])
    assert ced.flops == base.flops + LLAMA3_70B_CED.ced_kv_proj_params * 2 * s
    assert ced.bytes == base.bytes


@pytest.mark.parametrize("s", [8192, 32768])
def test_long_prompts_cost_about_half(s):
    base = CostModel(LLAMA3_70B, H100_SXM, 4).prefill([s]).time
    ced = CostModel(LLAMA3_70B_CED, H100_SXM, 4).prefill([s]).time
    enc = CostModel(ON_DECODE, H100_SXM, 4, prefill_only=True).prefill([s]).time
    assert 0.50 < ced / base < 0.52 and 0.50 < enc / base < ced / base


def test_encoder_only_prefill_reads_only_the_encoder():
    m, s = ON_DECODE, 4096
    c = CostModel(m, H100_SXM, 4, prefill_only=True).prefill([s])
    assert c.flops == 2 * m.ced_prompt_params * s + 2 * 40 * m.d_model * s * (s + 1)
    assert c.bytes == m.ced_prompt_params * 2.0 + s * m.embedding_row_bytes + s * m.kv_bytes_per_token
    # the same model on a decode or colocated instance prefills with the replay included
    assert CostModel(m, H100_SXM, 4).prefill([s]) == CostModel(LLAMA3_70B_CED, H100_SXM, 4).prefill([s])


def test_a_prefill_only_instance_holds_about_half_the_weights():
    m = ON_DECODE
    assert m.ced_prefill_resident_params / m.params == pytest.approx(0.51, abs=0.01)
    assert CostModel(m, H100_SXM, 2, prefill_only=True).kv_capacity_tokens > 0
    with pytest.raises(ValueError):          # the whole model does not fit on one H100 ...
        CostModel(LLAMA3_70B_CED, H100_SXM, 1).kv_capacity_tokens
    # the encoder alone only just fits on one (room for a few hundred tokens of KV); on two it has
    # 25x the KV room the whole model has
    assert CostModel(m, H100_SXM, 1, prefill_only=True).kv_capacity_tokens < 1000
    assert (CostModel(m, H100_SXM, 2, prefill_only=True).kv_capacity_tokens
            > 20 * CostModel(LLAMA3_70B_CED, H100_SXM, 2).kv_capacity_tokens)


def test_ced_step_without_replay_is_a_decode_step():
    cm = CostModel(ON_DECODE, H100_SXM, 4)
    assert cm.ced_step(30000, 12, []) == cm.decode_sum(30000, 12)
    # one replayed request: the whole model over its last W tokens, on top of the decode
    r = cm.ced_step(30000, 12, [(4096, 128)])
    m = ON_DECODE
    assert r.flops - cm.decode_sum(30000, 12).flops == (2 * m.layer_params * 128 + 2 * m.vocab * m.d_model * 128
                                                         + 2 * 80 * m.d_model * 128 * (2 * 4096 - 128 + 1))


def test_decode_is_unchanged_by_ced():
    for b in (1, 16):
        assert (CostModel(LLAMA3_70B_CED, H100_SXM, 4).decode([2300] * b)
                == CostModel(LLAMA3_70B, H100_SXM, 4).decode([2300] * b))


def test_invalid_ced_models_are_rejected():
    with pytest.raises(ValueError):
        replace(LLAMA3_8B, ced_encoder_layers=32)
    with pytest.raises(ValueError):
        replace(LLAMA3_8B_HYENA, ced_encoder_layers=16)
    with pytest.raises(ValueError):
        replace(LLAMA3_8B_CED, ced_replay=0)
    with pytest.raises(ValueError):
        replace(LLAMA3_8B_CED, ced_replay_on="link")
    with pytest.raises(ValueError):
        simulate(SimConfig(model=ON_DECODE, mode="colocated"), wl(n=10))
    with pytest.raises(ValueError):
        simulate(SimConfig(model=ON_DECODE, fast_forward=True), wl(n=10))


# ── the simulator ───────────────────────────────────────────────────────
CED_CONFIGS = {
    "replay on prefill": SimConfig(model=LLAMA3_70B_CED),
    "replay on decode": SimConfig(model=ON_DECODE),
    "replay on decode, 2-GPU prefill": SimConfig(model=ON_DECODE, prefill_devices_per_instance=2),
    "colocated, replay on prefill": SimConfig(model=LLAMA3_70B_CED, mode="colocated"),
}


@pytest.mark.parametrize("name", CED_CONFIGS)
def test_every_request_completes_and_stages_sum(name):
    reqs = wl(output=48, ocv=1.0, singles=True)
    assert any(r.output_len == 1 for r in reqs)
    res = simulate(CED_CONFIGS[name], reqs)
    assert not res.rejected
    for r in reqs:
        assert r.tokens_out == r.output_len and len(r.itls) == r.output_len - 1
        assert r.arrival <= r.prefill_start <= r.handoff_start <= r.finish
        assert r.first_token <= r.finish
        assert sum(r.stages().values()) == pytest.approx(r.e2e, abs=1e-9)
    for inst in res.instances:
        assert inst.kv_used == 0


def test_replay_on_decode_emits_the_first_token_on_the_decode_pool():
    reqs = wl()
    simulate(CED_CONFIGS["replay on decode"], reqs)
    for r in reqs:
        assert r.prefill_done is not None and r.prefill_done <= r.kv_start <= r.kv_ready
        assert r.kv_ready <= r.decode_start < r.first_token        # TTFT includes hand-off and replay


def test_fast_forward_stays_exact_with_replay_on_prefill():
    a, b = wl(n=300, rate=3.0), wl(n=300, rate=3.0)
    ra = simulate(SimConfig(model=LLAMA3_70B_CED), a)
    rb = simulate(SimConfig(model=LLAMA3_70B_CED, fast_forward=True), b)
    for x, y in zip(a, b):
        assert x.finish == pytest.approx(y.finish, rel=1e-12, abs=1e-12)
        assert x.first_token == y.first_token
    assert summarise(ra)["throughput"]["slo_attainment"] == summarise(rb)["throughput"]["slo_attainment"]


def test_ced_relieves_a_prefill_bound_pool():
    """Agent-like load (long prompts, short outputs) that swamps one decoder-only prefill instance."""
    out = {}
    for name, m in (("base", LLAMA3_70B), ("ced", LLAMA3_70B_CED), ("ced-d", ON_DECODE)):
        out[name] = summarise(simulate(SimConfig(model=m, ttft_slo=2.0), wl(n=300, rate=2.5, prompt=8192, output=128)))
    base, ced = out["base"]["latency_s"]["ttft"]["p99"], out["ced"]["latency_s"]["ttft"]["p99"]
    assert ced < base / 5
    assert out["ced"]["throughput"]["slo_attainment"] > out["base"]["throughput"]["slo_attainment"]
    assert out["ced-d"]["throughput"]["slo_attainment"] > out["base"]["throughput"]["slo_attainment"]


def test_kv_wait_is_measured_from_the_end_of_prefill():
    reqs = wl(n=120, rate=3.0, prompt=8192, output=64)
    res = simulate(replace(CED_CONFIGS["replay on decode"], link=LINKS["eth-25g"]), reqs)
    assert res.link.wait == pytest.approx(sum(r.kv_start - r.prefill_done for r in reqs), rel=1e-12)
    assert res.link.wait > 0


def test_cli_ced_flags():
    a = build_parser().parse_args(["--model", "llama3-70b", "--ced-encoder-layers", "30", "--ced-replay", "64",
                                   "--ced-replay-on", "decode", "--prefill-devices-per-instance", "2"])
    cfg = config_from_args(a)
    assert (cfg.model.ced_encoder_layers, cfg.model.ced_replay, cfg.model.ced_replay_on) == (30, 64, "decode")
    assert config_from_args(build_parser().parse_args([])).model == LLAMA3_70B


# ── the JavaScript port ─────────────────────────────────────────────────
JS_CASES = {
    "replay on prefill": dict(model="llama3-70b-ced"),
    "replay on decode": dict(model="llama3-70b-ced", cedReplayOn="decode"),
    "replay on decode, 2-GPU prefill pool, 2P2D": dict(model="llama3-70b-ced", cedReplayOn="decode",
                                                       prefillDevicesPerInstance=2, nPrefill=2, nDecode=2),
    "8B, W=32, LM head last": dict(model="llama3-8b-ced", devicesPerInstance=1, cedReplay=32, lmHead="last"),
    "8B, encoder 8 layers, replay on decode, A100 decode": dict(model="llama3-8b", cedEncoderLayers=8,
                                                               cedReplayOn="decode", devicesPerInstance=1,
                                                               decodeDevice="a100"),
    "colocated, replay on prefill": dict(model="llama3-70b-ced", mode="colocated"),
    "replay on decode, fp8 at the GPU, 25 GbE": dict(model="llama3-70b-ced", cedReplayOn="decode", link="eth-25g",
                                                     kvCompress="fp8", kvCompressAt="endpoint"),
    "replay on decode, fp4 in transit": dict(model="llama3-70b-ced", cedReplayOn="decode", link="eth-25g",
                                             kvCompress="fp4-block"),
    "replay on decode, power cap + DVFS": dict(model="llama3-70b-ced", cedReplayOn="decode", powerCap=350.0, dvfs=True),
    "decoder-only baseline (unchanged path)": dict(model="llama3-70b"),
}


def py_config(c: dict) -> SimConfig:
    model = MODELS[c["model"]]
    kw = {k2: c[k1] for k1, k2 in (("cedEncoderLayers", "ced_encoder_layers"), ("cedReplay", "ced_replay"),
                                    ("cedReplayOn", "ced_replay_on"), ("lmHead", "prefill_lm_head")) if k1 in c}
    if kw:
        model = replace(model, **kw)
    tr = None
    if "kvCompress" in c:
        tr = KVTransit(KV_PRESETS[c["kvCompress"]], where=c.get("kvCompressAt", "transit"))
    dev = {"h100": H100_SXM, "a100": A100_SXM}
    return SimConfig(model=model, device=H100_SXM, devices_per_instance=c.get("devicesPerInstance", 4),
                     mode=c.get("mode", "disagg"), n_prefill=c.get("nPrefill", 1), n_decode=c.get("nDecode", 1),
                     n_colocated=2, link=LINKS[c.get("link", "ib-ndr")], power_cap_w=c.get("powerCap"),
                     dvfs=c.get("dvfs", False), prefill_devices_per_instance=c.get("prefillDevicesPerInstance"),
                     decode_device=dev.get(c.get("decodeDevice")), kv_transit=tr)


def test_javascript_port_matches_python_for_ced():
    """Every timestamp (including prefill_done), every instance's energy accounts, the link's, the energy
    total and the stage means. Bit-exact, except runs with power-bound steps (cube roots: 1e-9) and the
    stage means (Python's fmean is compensated, the port's mean is a plain sum: 1e-12)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    engine = Path(__file__).parent.parent / "web" / "sim_engine.js"
    cases = []
    for name, c in JS_CASES.items():
        reqs = poisson_workload(2.5, 150, LengthDist(4096, 0.3), LengthDist(48, 1.0), seed=17)
        for r in reqs[::7]:
            r.output_len = 1
        rows = [[r.arrival, r.prompt_len, r.output_len] for r in reqs]
        res = simulate(py_config(c), reqs)
        m = summarise(res)
        cases.append({"name": name, "cfg": {"device": "h100", "devicesPerInstance": 4, "mode": "disagg", "nPrefill": 1,
                                            "nDecode": 1, "nColocated": 2, "link": "ib-ndr", **c},
                      "rows": rows,
                      "stamps": [[r.prefill_start, r.first_token, r.prefill_done, r.kv_start, r.kv_ready, r.decode_start,
                                  r.finish] for r in reqs],
                      "inst": [[i.compute_j, i.memory_j, i.busy, i.peak_power, i.steps, i.batch_sum] for i in res.instances],
                      "link": [res.link.energy, res.link.bytes, res.link.wait, res.link.transit_j],
                      "total": m["energy"]["total_J"], "stages": [m["stage_breakdown_s"][k] for k in sorted(m["stage_breakdown_s"])],
                      "capped": any(i.power_bound_time > 0 for i in res.instances)})
    script = (f"require({json.dumps(str(engine))});"
              "const S=globalThis.DisaggSim, cases=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify(cases.map(c=>{const r=S.simulate(c.cfg,c.rows), m=S.summarise(r);"
              "return {stamps:r.reqs.map(q=>[q.prefillStart,q.firstToken,q.prefillDone,q.kvStart,q.kvReady,q.decodeStart,q.finish]),"
              "inst:r.insts.map(i=>[i.ec,i.em,i.busy,i.peakW,i.steps,i.batchSum]),"
              "link:[r.link.energy,r.link.bytes,r.link.wait,r.link.transitJ],"
              "total:m.energy.totalJ, stages:Object.keys(m.stages).sort().map(k=>m.stages[k])};})));")
    out = subprocess.run([node, "-e", script], input=json.dumps(cases), capture_output=True, text=True, check=True)
    # prompts well above the ridge point keep most steps off the power roof, so most cases are bit-exact
    assert sum(not c["capped"] for c in cases) >= 8
    for c, js in zip(cases, json.loads(out.stdout)):
        for key in ("stamps", "inst", "link", "stages"):
            tol = 1e-9 if c["capped"] else (1e-12 if key == "stages" else 0.0)
            rows_py = c[key] if key in ("stamps", "inst") else [c[key]]
            rows_js = js[key] if key in ("stamps", "inst") else [js[key]]
            assert len(rows_py) == len(rows_js), (c["name"], key)
            for a, b in zip(rows_py, rows_js):
                for x, y in zip(a, b):
                    assert (x is None and y is None) or abs(x - y) <= tol * max(1.0, abs(x)), (c["name"], key, a, b)
        tol = 1e-9 if c["capped"] else 0.0
        assert abs(c["total"] - js["total"]) <= tol * c["total"], c["name"]
