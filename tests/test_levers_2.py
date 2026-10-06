"""Simulator levers, part II (brief 20A2, 2026-10-06).

The brief-20A1 levers in disaggregated pools (a prefill pool with prefix caching and chunking; a decode pool with
paged KV and preemption that pulls each hand-off once it has room); speculative decoding (Leviathan et al.,
arXiv:2211.17192); tensor, pipeline and expert parallelism inside an instance (Megatron-LM arXiv:1909.08053, GPipe
arXiv:1811.06965, GShard arXiv:2006.16668); weight and KV storage formats and native compute formats; an MoE shape.
These tests pin: every lever off by default and the recorded behaviour unchanged (the disaggregated pools at default
levers reproduce the old ones bit for bit); the closed forms; memory never over-committed in either pool; the
combinations that are not modelled raising clearly; each lever's direction; and the JavaScript port bit-exact.
"""

import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from disagg_sim.cli import build_parser, config_from_args, workload_from_args
from disagg_sim.hardware import (A100_40G, A100_SXM, ACCELERATORS, B200, H100_SXM, H200_SXM, KV_PRESETS, LINKS,
                                 LLAMA32_1B, LLAMA3_70B, LLAMA3_70B_CED, LLAMA3_8B, LLAMA3_8B_HYENA, MIXTRAL_8X7B, MODELS,
                                 OPT_13B, OPTICAL_FFT, QUANT_FORMATS, CostModel, KVTransit, Parallel, ipow)
from disagg_sim.metrics import format_report, summarise
from disagg_sim.sim import (ScheduledDecodeInstance, ScheduledPrefillInstance, SimConfig, Simulation, simulate)
from disagg_sim.speculative import (Speculative, draw_accepted, expected_operations, expected_speedup, expected_tokens,
                                    mulberry32, seed_state)
from disagg_sim.workload import LengthDist, chat_sessions, poisson_workload, workload_rows

D8 = SimConfig(model=LLAMA3_8B, devices_per_instance=1, n_prefill=1, n_decode=1)
OPTD = SimConfig(model=OPT_13B, device=A100_40G, devices_per_instance=1, n_prefill=1, n_decode=1,
                 host_link=LINKS["pcie4"])
FORCED = dict(max_num_batched_tokens=8192)


def wl(n=300, rate=6.0, prompt=2048, output=128, seed=2, pcv=0.6, ocv=0.6, singles=False):
    reqs = poisson_workload(rate, n, LengthDist(prompt, pcv), LengthDist(output, ocv), seed=seed)
    if singles:
        for r in reqs[::7]:
            r.output_len = 1
    return reqs


def opt_wl(n=400, rate=6.0, seed=1):
    return poisson_workload(rate, n, LengthDist(161, 1.0, hi=1024), LengthDist(338, 1.0, hi=1024), seed=seed)


def chat(n=240, rate=1.0, out=(64, 32, 96), seed=3, turns=4, think=1.0, sys_len=1024, sys_n=4):
    return chat_sessions(rate, n, LengthDist(384, 0.19, lo=256, hi=512), LengthDist(out[0], 0.3, lo=out[1], hi=out[2]),
                         seed=seed, turns=turns, think=think, system_prompts=sys_n, system_len=sys_len)


def stamps(reqs):
    return [(r.prefill_start, r.first_token, r.kv_start, r.kv_ready, r.decode_start, r.finish, tuple(r.itls))
            for r in reqs]


def run(cfg, reqs):
    res = simulate(cfg, reqs)
    return res, summarise(res)


# ── off by default; the old disaggregated pools reproduced ──────────────
def test_every_new_lever_is_off_by_default():
    c = SimConfig()
    assert (c.parallel, c.prefill_parallel, c.decode_parallel, c.speculative) == (None, None, None, None)
    assert (c.weight_format, c.kv_format, c.compute_format) == ("bf16", "bf16", "bf16")
    assert not c.scheduled and not c.quantised and c.lever_summary is None
    m = summarise(simulate(D8, wl(n=60)))
    assert not {"scheduler", "parallel", "formats"} & set(m) and "scale_up" not in m["energy"]["breakdown"]


@pytest.mark.parametrize("n_pre, singles, cap, link", [(1, False, None, "ib-ndr"), (2, True, None, "ib-ndr"),
                                                        (2, False, 350.0, "eth-25g"), (3, True, None, "nvlink4")])
def test_scheduled_pools_at_default_levers_equal_the_old_pools(n_pre, singles, cap, link):
    """``max_num_batched_tokens = max_prefill_tokens`` routes a disaggregated run through the scheduled pools with
    nothing else on: with one decode instance and room in both pools, every timestamp, every instance's energy and
    every summary number is unchanged (the decode pool pulls each hand-off at once, as the old link did)."""
    cfg = replace(D8, n_prefill=n_pre, power_cap_w=cap, link=LINKS[link])
    a, b = wl(singles=singles), wl(singles=singles)
    ra, rb = simulate(cfg, a), simulate(replace(cfg, **FORCED), b)
    assert isinstance(rb.instances[0], ScheduledPrefillInstance) and isinstance(rb.instances[-1], ScheduledDecodeInstance)
    assert stamps(a) == stamps(b)
    assert [(i.compute_j, i.memory_j, i.busy, i.steps, i.batch_sum) for i in ra.instances] == \
           [(i.compute_j, i.memory_j, i.busy, i.steps, i.batch_sum) for i in rb.instances]
    ma, mb = summarise(ra), summarise(rb)
    mb.pop("scheduler")
    assert json.dumps(ma, sort_keys=True, default=str) == json.dumps(mb, sort_keys=True, default=str)


def test_cost_model_levers_at_their_defaults_change_nothing():
    """A one-stage, one-way 'parallel' layout is not the default (it adds nothing but a check), so the defaults are
    None; compute_speedup = 1 and no draft leave the cost model's numbers bit-identical."""
    cm = CostModel(LLAMA3_70B, H100_SXM, 4)
    same = CostModel(LLAMA3_70B, H100_SXM, 4, compute_speedup=1.0, parallel=None, draft=None)
    for f in (lambda c: c.prefill([3000, 100]), lambda c: c.decode_sum(90000, 40),
              lambda c: c.step_mixed(500, 3, [(0, 700, True)])):
        assert f(cm) == f(same)
    assert cm.kv_capacity_tokens == same.kv_capacity_tokens


# ── disaggregated pools with the levers ─────────────────────────────────
@pytest.mark.parametrize("kw", [dict(kv_policy="paged"), dict(kv_policy="paged", preemption="swap"),
                                dict(kv_policy="paged", batch_policy="chunked", max_num_batched_tokens=512),
                                dict(kv_policy="max", **FORCED), dict(kv_policy="pow2", batch_policy="decode-priority"),
                                dict(kv_policy="paged", speculative=Speculative("mtp", 3, 0.7))])
def test_decode_pool_memory_is_never_over_committed_and_every_request_completes(kw):
    res, m = run(replace(OPTD, sample_dt=0.01, **kw), opt_wl(n=300))
    assert m["requests"]["completed"] == 300
    for row in res.samples:
        assert row["decode-0.kv"] <= 1.0 and row["prefill-0.kv"] <= 1.0
    for r in res.requests:
        assert r.tokens_out == r.output_len and len(r.itls) == r.output_len - 1
        assert r.arrival <= r.prefill_start <= r.first_token <= r.kv_start <= r.kv_ready <= r.decode_start <= r.finish
        assert abs(sum(r.stages().values()) - r.e2e) < 1e-9
    for inst in res.instances:
        assert inst.kv_used == 0 and not inst.running and not inst.queue and not inst.swapped
    assert not res.instances[0].held and not res.instances[-1].inbox and not res.instances[-1].landed


def test_paged_decode_pool_preempts_and_batches_more_than_reservation():
    _, oracle = run(replace(OPTD, **FORCED), opt_wl())
    _, rec = run(replace(OPTD, kv_policy="paged"), opt_wl())
    _, swp = run(replace(OPTD, kv_policy="paged", preemption="swap"), opt_wl())
    assert rec["scheduler"]["mean_running"] > 1.3 * oracle["scheduler"]["mean_running"]
    assert rec["scheduler"]["preemptions"] > 0 and rec["scheduler"]["recompute_tokens"] > 0
    assert swp["scheduler"]["preemptions"] > 0 and swp["scheduler"]["swap"]["out_GB"] > 0
    assert rec["throughput"]["output_tok_per_s"] > 1.15 * oracle["throughput"]["output_tok_per_s"]


def test_hand_off_waits_for_decode_blocks():
    """With the decode pool short of memory, prefilled requests wait for blocks before their transfer starts: the
    KV-wait stage grows, the link stays idle meanwhile, and the prefill pool holds their KV."""
    small = replace(OPTD, n_prefill=2, **FORCED)
    res, m = run(small, opt_wl(rate=8.0))
    assert m["stage_breakdown_s"]["kv_wait"] > 0.5
    assert m["kv_link"]["mean_wait_ms"] > 500


def test_prefix_cache_in_the_prefill_pool():
    """Hits skip prefill; the reply is not cached on the prefill pool (it was generated on decode), so the hit rate
    is below the colocated one for the same sessions, and every hit still crosses the link in full."""
    _, off = run(replace(D8, **FORCED), chat())
    res, on = run(replace(D8, prefix_caching=True), chat())
    pc = on["scheduler"]["prefix_cache"]
    assert pc["hit_rate"] > 0.4
    assert on["latency_s"]["ttft"]["p50"] < 0.6 * off["latency_s"]["ttft"]["p50"]
    assert on["kv_link"]["bytes_GB"] == off["kv_link"]["bytes_GB"]
    _, col = run(SimConfig(model=LLAMA3_8B, devices_per_instance=1, mode="colocated", n_colocated=1, prefix_caching=True),
                 chat())
    assert col["scheduler"]["prefix_cache"]["hit_rate"] > pc["hit_rate"]


def test_chunked_prefill_pool_runs_budgeted_steps():
    res, m = run(replace(D8, batch_policy="chunked", max_num_batched_tokens=1024), wl(n=120))
    assert m["requests"]["completed"] == 120
    pre = res.instances[0]
    assert pre.computed_tokens == sum(r.prompt_len for r in res.requests)
    assert pre.steps >= sum(r.prompt_len for r in res.requests) / 1024


def test_levers_with_heterogeneous_pools_and_hand_off_compression():
    """Supported combinations: other devices per pool, and compressing the hand-off in transit or at the GPU."""
    het = replace(D8, prefill_device=H100_SXM, decode_device=A100_SXM, kv_policy="paged")
    _, m = run(het, wl(n=150))
    assert m["requests"]["completed"] == 150 and m["pools"]["decode"] == "1x A100-SXM"
    for where in ("transit", "endpoint"):
        cfg = replace(D8, link=LINKS["eth-25g"], kv_policy="paged", kv_transit=KVTransit(KV_PRESETS["fp8"], where=where))
        _, m = run(cfg, wl(n=150, rate=2.0))
        assert m["kv_link"]["transit"]["ratio"] == 2.0 and m["requests"]["completed"] == 150


@pytest.mark.parametrize("kw, msg", [
    (dict(model=LLAMA3_70B_CED, kv_policy="paged"), "without CED"),
    (dict(model=LLAMA3_8B_HYENA, prefill_device=OPTICAL_FFT, kv_policy="paged"), "without CED"),
    (dict(kv_policy="paged", fast_forward=True), "fast_forward"),
    (dict(speculative=Speculative(), kv_transit=KVTransit(KV_PRESETS["fp8"])), "compression"),
    (dict(prefix_caching=True, kv_policy="paged", preemption="swap"), "swap"),
    (dict(kv_format="fp8", kv_transit=KVTransit(KV_PRESETS["fp8"])), "BF16 KV"),
    (dict(weight_format="fp6"), "weight_format"),
    (dict(weight_format="int4", compute_format="fp8"), "compute_format"),
    (dict(device=A100_SXM, weight_format="fp8", compute_format="fp8"), "no fp8 units"),
    (dict(model=LLAMA3_8B_HYENA, weight_format="fp8"), "attention models"),
    (dict(speculative=Speculative(gamma=0)), "gamma"),
    (dict(speculative=Speculative(draft="llama3.2-1b"), model=OPT_13B, device=A100_40G), "vocabulary"),
    (dict(speculative=Speculative(draft="gpt-9")), "draft model"),
    (dict(parallel=Parallel(tp=2)), "needs 2 devices"),
    (dict(parallel=Parallel(tp=1, ep=2)), "ep must be"),
    (dict(parallel=Parallel(tp=2, ep=2), devices_per_instance=2), "mixture-of-experts"),
    (dict(model=LLAMA3_70B, parallel=Parallel(tp=1, pp=3), devices_per_instance=3), "not divisible"),
    (dict(model=LLAMA3_70B_CED, parallel=Parallel(tp=4), devices_per_instance=4), "without CED"),
    (dict(model=LLAMA3_70B, device=A100_40G, devices_per_instance=4, parallel=Parallel(tp=1, pp=4)), "does not fit"),
    (dict(parallel=Parallel(tp=1, pp=2), devices_per_instance=2, speculative=Speculative()), "pipeline"),
    (dict(model=MIXTRAL_8X7B, devices_per_instance=2, fast_forward=True), "fast_forward"),
])
def test_unsupported_combinations_raise_clearly(kw, msg):
    with pytest.raises(ValueError, match=msg):
        Simulation(replace(D8, **kw), wl(n=5))


# ── speculative decoding (Leviathan et al., arXiv:2211.17192) ────────────
@pytest.mark.parametrize("alpha, gamma", [(0.0, 3), (0.5, 1), (0.6, 4), (0.8, 5), (0.9, 8), (1.0, 4)])
def test_expected_tokens_is_the_closed_form(alpha, gamma):
    """E[tokens per verify] = (1 - a^(g+1)) / (1 - a) (their equation 1) = sum_{i=0..g} a^i; g + 1 at a = 1."""
    e = expected_tokens(alpha, gamma)
    assert e == pytest.approx(sum(alpha ** i for i in range(gamma + 1)), rel=1e-12)
    assert expected_speedup(alpha, gamma, 0.0) == e
    assert expected_speedup(alpha, gamma, 0.1) == pytest.approx(e / (gamma * 0.1 + 1))


@pytest.mark.parametrize("alpha, gamma, ops, speed", [(0.6, 2, 1.53, 1.96), (0.7, 3, 1.58, 2.53), (0.8, 2, 1.23, 2.44),
                                                     (0.8, 5, 1.63, 3.69), (0.9, 2, 1.11, 2.71), (0.9, 10, 1.60, 6.86)])
def test_closed_forms_reproduce_leviathan_table_1(alpha, gamma, ops, speed):
    """Their Table 1 (c = c^ = 0): operations factor (Theorem 3.11) and speed (Theorem 3.8), to the 2 decimals printed."""
    assert round(expected_operations(alpha, gamma, 0.0), 2) == ops
    assert round(expected_speedup(alpha, gamma, 0.0), 2) == speed


def test_acceptance_draws_converge_to_the_closed_form():
    """The simulator's per-row draws (accept while u < alpha, then one more token): their mean matches the closed
    form to within 4 standard errors, for several (alpha, gamma)."""
    for alpha, gamma in ((0.5, 2), (0.7, 3), (0.85, 6)):
        st, xs = seed_state(7, 11), []
        for _ in range(40000):
            st, k = draw_accepted(st, alpha, gamma)
            xs.append(k + 1)
        mean = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
        assert abs(mean - expected_tokens(alpha, gamma)) < 4 * sd / math.sqrt(len(xs)), (alpha, gamma, mean)


def test_mulberry32_matches_the_javascript_generator():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    st, py = 12345, []
    for _ in range(6):
        st, u = mulberry32(st)
        py.append(u)
    js = subprocess.run([node, "-e", "function m(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);"
                         "t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}const r=m(12345);"
                         "console.log(JSON.stringify([r(),r(),r(),r(),r(),r()]))"], capture_output=True, text=True, check=True)
    assert py == json.loads(js.stdout)


def test_simulated_tokens_per_verify_match_the_closed_form():
    sp = Speculative("mtp", gamma=4, alpha=0.75)
    _, m = run(SimConfig(model=LLAMA3_8B, devices_per_instance=1, mode="colocated", n_colocated=1, speculative=sp),
               wl(n=200, rate=0.5, prompt=512, output=1024, ocv=0.0))
    s = m["scheduler"]["speculative"]
    assert s["closed_form"] == expected_tokens(0.75, 4)
    assert abs(s["tokens_per_verify"] - s["closed_form"]) < 0.03 * s["closed_form"]     # caps at the output end only


def test_speculation_helps_at_low_batch_and_hurts_when_compute_bound():
    """Memory-bound decode (few rows): verifying gamma + 1 positions costs about one step, so tokens per second per
    request rise by close to Leviathan's Theorem 3.8 factor. Compute-bound (a large batch of long prompts on a
    slow device): every rejected position is wasted compute, and throughput falls."""
    base = SimConfig(model=LLAMA3_8B, devices_per_instance=1, mode="colocated", n_colocated=1)
    sp = Speculative("mtp", gamma=3, alpha=0.8)
    lo = [run(replace(base, speculative=s), wl(n=60, rate=0.2, prompt=512, output=256))[1] for s in (None, sp)]
    gain = lo[0]["latency_s"]["tpot"]["p50"] / lo[1]["latency_s"]["tpot"]["p50"]
    cm, dm = CostModel(LLAMA3_8B, H100_SXM, 1), CostModel(replace(LLAMA3_8B, n_layers=1), H100_SXM, 1, 0.0)
    c = dm.decode_sum(768, 1).time / cm.decode_sum(768, 1).time
    assert gain > 1.8 and gain == pytest.approx(expected_speedup(0.8, 3, c), rel=0.15)
    hi_cfg = replace(base, device=A100_SXM, max_decode_batch=512)
    hi = [run(replace(hi_cfg, speculative=s), wl(n=600, rate=200.0, prompt=1024, output=256, ocv=0.0))[1]
          for s in (None, Speculative("mtp", gamma=3, alpha=0.5))]
    assert hi[1]["throughput"]["output_tok_per_s"] < hi[0]["throughput"]["output_tok_per_s"]


def test_speculation_in_the_decode_pool_moves_the_draft_kv():
    sp = Speculative("llama3.2-1b", gamma=3, alpha=0.7)
    res, m = run(replace(D8, speculative=sp), wl(n=80, rate=1.0))
    _, base = run(replace(D8, **FORCED), wl(n=80, rate=1.0))
    ratio = (LLAMA3_8B.kv_bytes_per_token + LLAMA32_1B.kv_bytes_per_token) / LLAMA3_8B.kv_bytes_per_token
    assert m["kv_link"]["bytes_GB"] == pytest.approx(base["kv_link"]["bytes_GB"] * ratio, rel=1e-12)
    assert m["scheduler"]["speculative"]["draft_time_frac"] > 0.05
    assert m["latency_s"]["tpot"]["p50"] < base["latency_s"]["tpot"]["p50"]


def test_speculative_verify_step_closed_form():
    m, cm = LLAMA3_8B, CostModel(LLAMA3_8B, H100_SXM, 1)
    rows, g = [700, 1500, 64], 3
    s = cm.step_spec(sum(rows), len(rows), g + 1)
    flops = sum(2 * m.matmul_params * (g + 1) + sum(4 * m.n_layers * m.d_model * j for j in range(p + 1, p + g + 2))
                for p in rows)
    assert s.flops == flops
    assert s.bytes == m.weight_bytes_read(len(rows) * (g + 1)) + (sum(rows) + len(rows) * (g + 1)) * m.kv_bytes_per_token
    one = cm.step_spec(sum(rows), len(rows), 1)          # gamma = 0 is a decode step
    assert one.flops == cm.decode_sum(sum(rows), len(rows)).flops and one.bytes == cm.decode_sum(sum(rows), len(rows)).bytes


# ── parallelism (Megatron-LM, GPipe, GShard) ─────────────────────────────
def test_tensor_parallel_step_adds_the_ring_all_reduces():
    m, b, ctx = LLAMA3_70B, 32, 32 * 2000
    cm = CostModel(m, H100_SXM, 4, parallel=Parallel(tp=4))
    plain = CostModel(m, H100_SXM, 4)
    s, q = cm.decode_sum(ctx, b), plain.decode_sum(ctx, b)
    link = LINKS["nvlink4"]
    act = b * m.d_model * 2.0
    ar = 2 * 3 / 4 * act / link.bandwidth + 2 * 3 * link.latency
    assert s.comm_time == pytest.approx(2 * m.n_layers * ar, rel=1e-12)
    assert s.time == pytest.approx(q.time + s.comm_time, rel=1e-12)       # one stage, one micro-batch: compute as before
    assert s.comm_j == pytest.approx(2 * m.n_layers * 2 * 3 * act * 8 * link.pj_per_bit * 1e-12, rel=1e-12)


def test_pipeline_bubbles_and_weight_rereads():
    """GPipe: m micro-batches over p stages take m + p - 1 slots; each re-reads its stage's weights, so micro-batches
    shrink a compute-bound prefill's bubble and lengthen a memory-bound decode."""
    m = LLAMA3_70B
    st = CostModel(m, H100_SXM, 2, step_overhead=0.0)
    for mb in (1, 2, 4):
        cm = CostModel(m, H100_SXM, 4, parallel=Parallel(tp=2, pp=2, microbatches=mb))
        s = cm.prefill([4096] * 4)
        tokens = 4 * 4096
        f = 2 * m.matmul_params * tokens + 4 * (2 * m.n_layers * m.d_model * 4096 * 4097)
        slot = st.step_time_raw(f / (mb * 2), m.weight_bytes_read(tokens) / 2 + tokens * m.kv_bytes_per_token / (mb * 2))[0]
        assert s.time - s.comm_time == pytest.approx((mb + 1) * slot + cm.step_overhead, rel=1e-9)
    pre = [CostModel(m, H100_SXM, 4, parallel=Parallel(tp=2, pp=2, microbatches=k)).prefill([4096] * 4).time for k in (1, 4)]
    dec = [CostModel(m, H100_SXM, 4, parallel=Parallel(tp=2, pp=2, microbatches=k)).decode_sum(64 * 2000, 64).time
           for k in (1, 4)]
    assert pre[1] < pre[0] and dec[1] > dec[0]


def test_memory_is_checked_per_gpu_stage_by_stage():
    m = LLAMA3_70B
    with pytest.raises(ValueError, match="per GPU"):
        CostModel(m, A100_40G, 4, parallel=Parallel(tp=1, pp=4)).kv_capacity_tokens     # 36.4 GB a GPU, 36 usable
    cm = CostModel(m, H100_SXM, 4, parallel=Parallel(tp=2, pp=2))
    room = 80e9 * 0.9
    stage0 = (40 * m.params_per_layer + m.vocab * m.d_model) * 2.0
    assert cm.kv_capacity_tokens == int((room * 2 - stage0) // (m.kv_bytes_per_token / 2))
    # each stage holds half the weights and half of every token's KV: as many tokens as TP over the same 4 GPUs
    assert cm.kv_capacity_tokens == CostModel(m, H100_SXM, 4).kv_capacity_tokens


def test_moe_shapes_and_expected_experts():
    m = MIXTRAL_8X7B
    active = m.matmul_params + m.vocab * m.d_model          # with the input embedding, as the paper counts
    assert 46.6e9 < m.params < 46.8e9 and 12.8e9 < active < 13.0e9          # 46.7B total, 12.9B active
    assert m.experts_touched(1) == pytest.approx(2.0) and m.experts_touched(8192) == pytest.approx(8.0)
    assert m.experts_touched(4) == pytest.approx(8 * (1 - 0.75 ** 4), rel=1e-15)
    assert ipow(0.75, 13) == 0.75 ** 13 or ipow(0.75, 13) == pytest.approx(0.75 ** 13, rel=1e-15)
    cm = CostModel(m, H100_SXM, 2)
    assert cm.decode_sum(1000, 1).bytes < cm.decode_sum(1000, 64).bytes < m.weight_bytes_total
    dense = CostModel(replace(m, n_experts=0, top_k=0), H100_SXM, 2)
    assert cm.decode_sum(1000, 1).flops == dense.decode_sum(1000, 1).flops + 2 * 32 * (4096 * 8 + 3 * 4096 * 14336)


def test_expert_parallel_all_to_all_and_imbalance():
    m = MIXTRAL_8X7B
    tp = CostModel(m, H100_SXM, 2, parallel=Parallel(tp=2)).prefill([2048] * 2)
    ep = CostModel(m, H100_SXM, 2, parallel=Parallel(tp=2, ep=2)).prefill([2048] * 2)
    skew = CostModel(m, H100_SXM, 2, parallel=Parallel(tp=2, ep=2, expert_imbalance=1.5)).prefill([2048] * 2)
    link, tokens = LINKS["nvlink4"], 4096
    a2a = 2 * tokens * 2 * 4096 * 2.0 * 1 / 2
    ar = 2 * 1 / 2 * tokens * 4096 * 2.0 / link.bandwidth + 2 * link.latency
    assert ep.comm_time == pytest.approx(32 * ar + 32 * (a2a / link.bandwidth + 2 * link.latency), rel=1e-12)
    assert tp.comm_time == pytest.approx(64 * ar, rel=1e-12)
    assert skew.time > ep.time and skew.compute_j == pytest.approx(ep.compute_j, rel=1e-12)


# ── quantisation ────────────────────────────────────────────────────────
def test_formats_change_bytes_capacity_and_compute_rate():
    assert QUANT_FORMATS["int4"] == 0.515625 and QUANT_FORMATS["fp4"] == 0.53125
    base = SimConfig(model=LLAMA3_70B, mode="colocated", n_colocated=1)
    sims = {k: Simulation(replace(base, **kw), wl(n=5)) for k, kw in {
        "bf16": {}, "w8": dict(weight_format="fp8"), "w8a8": dict(weight_format="fp8", compute_format="fp8"),
        "kv8": dict(kv_format="fp8"), "w4": dict(weight_format="int4")}.items()}
    cost = {k: s.colocated[0].cost for k, s in sims.items()}
    assert cost["w8"].decode_sum(10000, 8).bytes < 0.52 * cost["bf16"].decode_sum(10000, 8).bytes
    assert cost["w8a8"].prefill([8192]).time < 0.55 * cost["bf16"].prefill([8192]).time
    assert cost["w8"].prefill([8192]).time > 0.9 * cost["bf16"].prefill([8192]).time       # weight-only: BF16 rate
    assert abs(cost["kv8"].kv_capacity_tokens - 2 * cost["bf16"].kv_capacity_tokens) <= 1
    assert cost["w4"].kv_capacity_tokens > 1.6 * cost["bf16"].kv_capacity_tokens      # 252 GB free against 147
    assert cost["w8a8"].joules_per_flop == cost["bf16"].joules_per_flop / 2
    assert ACCELERATORS["b200"].format_speedup("fp4") == 4.0 and H100_SXM.format_speedup("fp4") is None
    assert H200_SXM.mem_capacity == 141e9 and B200.mem_bw == 8e12


def test_cli_flags_reach_the_config():
    a = build_parser().parse_args(["--tp", "2", "--pp", "2", "--microbatches", "4", "--weight-format", "fp8",
                                   "--compute-format", "fp8", "--kv-format", "fp8", "--speculative", "mtp", "--gamma", "5",
                                   "--alpha", "0.6", "--decode-tp", "4", "--workload", "chat"])
    c = config_from_args(a)
    assert c.parallel == Parallel(tp=2, pp=2, microbatches=4) and c.decode_parallel == Parallel(tp=4, microbatches=4)
    assert (c.weight_format, c.compute_format, c.kv_format) == ("fp8", "fp8", "fp8")
    assert c.speculative == Speculative("mtp", 5, 0.6) and c.ttft_slo == 1.0
    assert any(r.prefix for r in workload_from_args(a, 2.0))
    m = summarise(simulate(replace(c, mode="colocated", decode_parallel=None, speculative=None,
                                   model=LLAMA3_8B), wl(n=40, rate=1.0)))
    assert "parallel     colocated tp 2 pp 2" in format_report(m) and "accuracy not simulated" in format_report(m)


# ── the JavaScript port ─────────────────────────────────────────────────
OPT_JS = dict(model="opt-13b", device="a100-40g", devicesPerInstance=1, hostLink="pcie4")
D8_JS = dict(model="llama3-8b", devicesPerInstance=1, mode="disagg", nPrefill=1, nDecode=1)
JS_CASES = {
    "disagg forced scheduler, 2P1D": (dict(D8_JS, nPrefill=2, maxNumBatchedTokens=8192), "poisson"),
    "disagg paged recompute": (dict(OPT_JS, mode="disagg", nPrefill=1, nDecode=1, kvPolicy="paged"), "opt"),
    "disagg paged swap, chunked, 2P2D": (dict(OPT_JS, mode="disagg", nPrefill=2, nDecode=2, kvPolicy="paged", preemption="swap",
                                              batchPolicy="chunked", maxNumBatchedTokens=512), "opt"),
    "disagg oracle, slow link, decode-priority": (dict(OPT_JS, mode="disagg", nPrefill=2, nDecode=1, link="eth-25g",
                                                       batchPolicy="decode-priority"), "opt"),
    "disagg prefix cache, paged": (dict(D8_JS, prefixCaching=True, kvPolicy="paged"), "chat"),
    "disagg prefix cache, chunked, hetero": (dict(D8_JS, prefixCaching=True, batchPolicy="chunked", maxNumBatchedTokens=1024,
                                                  decodeDevice="a100", prefillDevice="h100"), "chat"),
    "disagg prefix cache evicting, OPT, 25 GbE": (dict(OPT_JS, mode="disagg", nPrefill=1, nDecode=1, prefixCaching=True,
                                                     kvPolicy="paged", link="eth-25g"), "chat-long"),
    "disagg endpoint fp8, paged": (dict(D8_JS, link="eth-25g", kvCompress="fp8", kvCompressAt="endpoint", kvPolicy="paged"), "poisson"),
    "disagg transit fp4, max": (dict(D8_JS, link="eth-25g", kvCompress="fp4-block", kvPolicy="max"), "poisson"),
    "colocated spec mtp": (dict(model="llama3-8b", devicesPerInstance=1, mode="colocated", nColocated=2,
                                speculative=dict(draft="mtp", gamma=3, alpha=0.7, seed=0)), "poisson"),
    "colocated spec 1B draft, chunked, paged": (dict(model="llama3-8b", devicesPerInstance=1, mode="colocated", nColocated=1,
                                                     batchPolicy="chunked", maxNumBatchedTokens=512, kvPolicy="paged",
                                                     speculative=dict(draft="llama3.2-1b", gamma=4, alpha=0.8, seed=3)), "poisson"),
    "disagg spec, paged, power cap": (dict(D8_JS, kvPolicy="paged", powerCap=400.0, dvfs=True,
                                           speculative=dict(draft="mtp", gamma=2, alpha=0.6, seed=1)), "poisson"),
    "70B tp4 colocated": (dict(model="llama3-70b", mode="colocated", nColocated=1, parallel=dict(tp=4, pp=1)), "poisson"),
    "70B tp2 pp2 mb3 disagg": (dict(model="llama3-70b", mode="disagg", nPrefill=1, nDecode=1,
                                    parallel=dict(tp=2, pp=2, microbatches=3)), "poisson"),
    "70B pool parallel + paged": (dict(model="llama3-70b", mode="disagg", nPrefill=1, nDecode=1, prefillDevicesPerInstance=2,
                                       prefillParallel=dict(tp=2, pp=1), decodeParallel=dict(tp=1, pp=4), kvPolicy="paged"), "poisson"),
    "mixtral ep2 imbalance": (dict(model="mixtral-8x7b", devicesPerInstance=2, mode="colocated", nColocated=1,
                                   parallel=dict(tp=2, pp=1, ep=2, expertImbalance=1.3)), "poisson"),
    "mixtral tp2 paged, B200 nvlink3": (dict(model="mixtral-8x7b", device="b200", devicesPerInstance=2, mode="colocated", nColocated=1,
                                             kvPolicy="paged", parallel=dict(tp=2, pp=1, link="nvlink3")), "poisson"),
    "mixtral plain": (dict(model="mixtral-8x7b", devicesPerInstance=2, mode="colocated", nColocated=2), "poisson"),
    "70B fp8 w8a8 kv8": (dict(model="llama3-70b", mode="colocated", nColocated=1, weightFormat="fp8", computeFormat="fp8",
                              kvFormat="fp8"), "poisson"),
    "70B int4 weight-only on h200, tp2": (dict(model="llama3-70b", device="h200", devicesPerInstance=2, mode="disagg", nPrefill=1,
                                              nDecode=1, weightFormat="int4", parallel=dict(tp=2, pp=1)), "poisson"),
    "8B fp4 w4a4 on b200 + spec": (dict(model="llama3-8b", device="b200", devicesPerInstance=1, mode="colocated", nColocated=1,
                                        weightFormat="fp4", computeFormat="fp4", kvFormat="int4",
                                        speculative=dict(draft="mtp", gamma=3, alpha=0.75, seed=2)), "poisson"),
    "a100 int8 w8a8 kv int8, disagg paged": (dict(model="llama3-8b", device="a100", devicesPerInstance=1, mode="disagg", nPrefill=1,
                                                  nDecode=1, weightFormat="int8", computeFormat="int8", kvFormat="int8",
                                                  kvPolicy="paged"), "poisson"),
}


def py_config(c: dict) -> SimConfig:
    def par(x):
        if x is None:
            return None
        return Parallel(tp=x["tp"], pp=x.get("pp", 1), ep=x.get("ep", 1), microbatches=x.get("microbatches"),
                        expert_imbalance=x.get("expertImbalance", 1.0), link=x.get("link"))
    sp = c.get("speculative")
    tr = None
    if "kvCompress" in c:
        tr = KVTransit(KV_PRESETS[c["kvCompress"]], where=c.get("kvCompressAt", "transit"))
    dev = ACCELERATORS
    return SimConfig(model=MODELS[c["model"]], device=dev[c.get("device", "h100")],
                     devices_per_instance=c.get("devicesPerInstance", 4), mode=c.get("mode", "disagg"),
                     n_prefill=c.get("nPrefill", 1), n_decode=c.get("nDecode", 1), n_colocated=c.get("nColocated", 2),
                     link=LINKS[c.get("link", "ib-ndr")], power_cap_w=c.get("powerCap"), dvfs=c.get("dvfs", False),
                     prefill_device=dev[c["prefillDevice"]] if "prefillDevice" in c else None,
                     decode_device=dev[c["decodeDevice"]] if "decodeDevice" in c else None,
                     prefill_devices_per_instance=c.get("prefillDevicesPerInstance"),
                     decode_devices_per_instance=c.get("decodeDevicesPerInstance"), kv_transit=tr,
                     batch_policy=c.get("batchPolicy", "prefill-priority"),
                     max_num_batched_tokens=c.get("maxNumBatchedTokens"), kv_policy=c.get("kvPolicy", "oracle"),
                     kv_block_size=c.get("kvBlockSize", 16), preemption=c.get("preemption", "recompute"),
                     host_link=LINKS[c.get("hostLink", "pcie5")], prefix_caching=c.get("prefixCaching", False),
                     parallel=par(c.get("parallel")), prefill_parallel=par(c.get("prefillParallel")),
                     decode_parallel=par(c.get("decodeParallel")), weight_format=c.get("weightFormat", "bf16"),
                     kv_format=c.get("kvFormat", "bf16"), compute_format=c.get("computeFormat", "bf16"),
                     speculative=Speculative(sp["draft"], sp["gamma"], sp["alpha"], sp["seed"]) if sp else None)


def js_workload(kind: str):
    if kind == "opt":
        return opt_wl(n=300)
    if kind == "chat":
        return chat(n=200, rate=2.0, out=(120, 60, 200), think=1.0, sys_len=512)
    if kind == "chat-long":     # the prefill pool's cache fills and evicts while hand-offs hold their segments
        return chat(n=200, rate=2.0, out=(120, 60, 200), think=1.0, sys_len=2048, sys_n=6)
    return wl(n=150, rate=5.0, singles=True)


def test_javascript_port_matches_python_for_the_levers_2():
    """Every timestamp, every instance's energy accounts, the scale-up link and draft time, the energy total and
    the scheduler / parallel summaries. Bit-exact, except runs with power-bound steps (cube roots: 1e-9)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    engine = Path(__file__).parent.parent / "web" / "sim_engine.js"
    cases = []
    for name, (c, kind) in JS_CASES.items():
        reqs = js_workload(kind)
        rows = workload_rows(reqs)
        res = simulate(py_config(c), reqs)
        m = summarise(res)
        s = m.get("scheduler", {})
        sc = [s.get("mean_running", 0.0), s.get("kv_token_frac", 0.0), s.get("computed_prefill_tokens", 0),
              s.get("preemptions", 0), s.get("recompute_tokens", 0), s.get("swap", {}).get("seconds", 0.0),
              s.get("prefix_cache", {}).get("hit_tokens", 0), s.get("speculative", {}).get("tokens_per_verify", 0.0),
              s.get("speculative", {}).get("draft_time_frac", 0.0), s.get("prefix_cache", {}).get("evicted_tokens", 0)]
        par = [[v["comm_s"], v["comm_J"], v["comm_frac"]] for v in m.get("parallel", {}).values()]
        cases.append({"name": name, "cfg": {"device": "h100", "devicesPerInstance": 4, "link": "ib-ndr", "ttftSlo": 1.0,
                                            "tpotSlo": 0.025, **c},
                      "rows": rows,
                      "stamps": [[r.arrival, r.prefill_start, r.first_token, r.kv_start, r.kv_ready, r.decode_start, r.finish]
                                 for r in res.requests],
                      "itls": sum(len(r.itls) for r in res.requests),
                      "inst": [[i.compute_j, i.memory_j, i.busy, i.peak_power, i.steps, i.batch_sum, i.comm_time, i.comm_j,
                                i.draft_time] for i in res.instances],
                      "total": m["energy"]["total_J"], "sched": sc, "par": par,
                      "capped": any(i.power_bound_time > 0 for i in res.instances)})
    script = (f"require({json.dumps(str(engine))});"
              "const S=globalThis.DisaggSim, cases=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify(cases.map(c=>{const r=S.simulate(c.cfg,c.rows), m=S.summarise(r), s=m.scheduler||{};"
              "const sp=s.speculative||{};"
              "return {stamps:r.reqs.map(q=>[q.arrival,q.prefillStart,q.firstToken,q.kvStart,q.kvReady,q.decodeStart,q.finish]),"
              "itls:r.reqs.reduce((a,q)=>a+q.itls.length,0),"
              "inst:r.insts.map(i=>[i.ec,i.em,i.busy,i.peakW,i.steps,i.batchSum,i.commT,i.commJ,i.draftT]), total:m.energy.totalJ,"
              "sched:[s.meanRunning??0,s.kvTokenFrac??0,s.computedPrefillTokens??0,s.preemptions??0,s.recomputeTokens??0,"
              "s.swap?s.swap.seconds:0,s.prefixCache?s.prefixCache.hitTokens:0,sp.tokensPerVerify??0,sp.draftTimeFrac??0,"
              "s.prefixCache?s.prefixCache.evictedTokens:0],"
              "par:Object.values(m.parallel||{}).map(v=>[v.commS,v.commJ,v.commFrac])};})));")
    out = subprocess.run([node, "-e", script], input=json.dumps(cases), capture_output=True, text=True, check=True)
    exercised = {"preempt": 0, "swap": 0, "hit": 0, "spec": 0, "comm": 0, "evict": 0}
    exact = 0
    for c, js in zip(cases, json.loads(out.stdout)):
        exact += all(c[k] == js[k] for k in ("stamps", "inst", "total", "sched", "par"))
        tol = 1e-9 if c["capped"] else 0.0
        assert c["itls"] == js["itls"], c["name"]
        for key in ("stamps", "inst", "par"):
            assert len(c[key]) == len(js[key]), (c["name"], key)
            for a, b in zip(c[key], js[key]):
                for x, y in zip(a, b):
                    assert (x is None and y is None) or abs(x - y) <= tol * max(1.0, abs(x)), (c["name"], key, a, b)
        assert abs(c["total"] - js["total"]) <= tol * c["total"], c["name"]
        for x, y in zip(c["sched"], js["sched"]):
            assert abs(x - y) <= tol * max(1.0, abs(x)), (c["name"], c["sched"], js["sched"])
        exercised["preempt"] += c["sched"][3] > 0
        exercised["swap"] += c["sched"][5] > 0
        exercised["hit"] += c["sched"][6] > 0
        exercised["spec"] += c["sched"][7] > 0
        exercised["comm"] += any(p[0] > 0 for p in c["par"])
        exercised["evict"] += c["sched"][9] > 0
    assert all(v >= 1 for v in exercised.values()), exercised
    assert exact >= len(JS_CASES) - 2, exact
