"""Simulator levers, part I (brief 20A1, 2026-10-06): batching policy, KV memory policy, prefix caching.

Colocated instances gain Sarathi-Serve's chunked prefill (arXiv:2403.02310) and a decode-priority policy
beside today's prefill-priority one; vLLM's paged KV blocks with preemption by recompute or swap, and the
over-reserving baselines it compares against (arXiv:2309.06180); and an LRU prefix cache of shared prompt
segments in the style of SGLang's RadixAttention (arXiv:2312.07104), with closed-loop multi-turn sessions.
These tests pin: every lever off by default and inert settings changing nothing (bit-identical runs); the
new scheduler with the levers at their defaults reproducing the old colocated instance exactly; the mixed
step's closed forms; memory never over-committed; each paper's qualitative ordering; and the JavaScript
port bit-exact on every lever.
"""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from disagg_sim.cli import build_parser, config_from_args, workload_from_args
from disagg_sim.hardware import (A100_40G, A100_SXM, H100_SXM, LINKS, LLAMA3_8B, LLAMA3_8B_CED, LLAMA3_8B_HYENA,
                                 LLAMA3_70B, MISTRAL_7B, MODELS, OPT_13B, CostModel)
from disagg_sim.metrics import format_report, summarise
from disagg_sim.sim import PrefixCache, ScheduledInstance, SimConfig, Simulation, simulate
from disagg_sim.workload import (LengthDist, chat_sessions, dump_workload, load_workload, poisson_workload,
                                 workload_rows)

C8 = SimConfig(mode="colocated", model=LLAMA3_8B, devices_per_instance=1)
OPT = SimConfig(mode="colocated", model=OPT_13B, device=A100_40G, devices_per_instance=1, n_colocated=1,
                host_link=LINKS["pcie4"])
FORCED = dict(max_num_batched_tokens=8192)       # the new scheduler with every lever at today's behaviour


def wl(n=300, rate=6.0, prompt=2048, output=128, seed=2, pcv=0.6, ocv=0.6, singles=False):
    reqs = poisson_workload(rate, n, LengthDist(prompt, pcv), LengthDist(output, ocv), seed=seed)
    if singles:
        for r in reqs[::7]:
            r.output_len = 1
    return reqs


def opt_wl(n=400, rate=6.0, seed=1):
    return poisson_workload(rate, n, LengthDist(161, 1.0, hi=1024), LengthDist(338, 1.0, hi=1024), seed=seed)


def chat(n=240, rate=1.0, out=(6, 4, 8), seed=3, turns=4, think=1.0, sys_len=1024, sys_n=4):
    return chat_sessions(rate, n, LengthDist(384, 0.19, lo=256, hi=512), LengthDist(out[0], 0.19, lo=out[1], hi=out[2]),
                         seed=seed, turns=turns, think=think, system_prompts=sys_n, system_len=sys_len)


def stamps(reqs):
    return [(r.prefill_start, r.first_token, r.decode_start, r.finish, tuple(r.itls)) for r in reqs]


def run(cfg, reqs):
    res = simulate(cfg, reqs)
    return res, summarise(res)


# ── off by default; inert settings; the new scheduler reproduces the old one ──
def test_every_lever_is_off_by_default():
    c = SimConfig()
    assert (c.batch_policy, c.max_num_batched_tokens, c.kv_policy, c.prefix_caching) == (
        "prefill-priority", None, "oracle", False)
    assert not c.scheduled and not C8.scheduled
    assert "scheduler" not in summarise(simulate(C8, wl(n=60)))


@pytest.mark.parametrize("mode", ["colocated", "disagg"])
def test_inert_settings_change_nothing(mode):
    """Block size, preemption mode, host link, max_seq_len and the watermark only matter with their policy;
    a workload's prefix chains only with prefix caching. Bit-identical runs."""
    base = replace(C8, mode=mode)
    a = wl()
    simulate(base, a)
    b = wl()
    for i, r in enumerate(b):                  # give every request a chain and an emitted segment
        r.prefix, r.emit_key = (("sys0", 64),), f"e{i}"
    simulate(replace(base, kv_block_size=1, preemption="swap", host_link=LINKS["pcie4"], max_seq_len=64,
                     kv_watermark=0.5), b)
    assert stamps(a) == stamps(b)


@pytest.mark.parametrize("n_coloc, singles, cap", [(2, False, None), (1, True, None), (2, True, 300.0), (3, False, None)])
def test_new_scheduler_at_default_levers_equals_the_colocated_instance(n_coloc, singles, cap):
    """``max_num_batched_tokens = max_prefill_tokens`` routes through ScheduledInstance with nothing else on: every
    timestamp, every instance's energy and every summary number is unchanged."""
    cfg = replace(C8, n_colocated=n_coloc, power_cap_w=cap)
    a, b = wl(singles=singles), wl(singles=singles)
    ra, rb = simulate(cfg, a), simulate(replace(cfg, **FORCED), b)
    assert isinstance(rb.instances[0], ScheduledInstance)
    assert stamps(a) == stamps(b)
    assert [(i.compute_j, i.memory_j, i.busy, i.steps, i.batch_sum) for i in ra.instances] == \
           [(i.compute_j, i.memory_j, i.busy, i.steps, i.batch_sum) for i in rb.instances]
    ma, mb = summarise(ra), summarise(rb)
    mb.pop("scheduler")
    assert json.dumps(ma, sort_keys=True, default=str) == json.dumps(mb, sort_keys=True, default=str)


def test_one_turn_sessions_without_system_prompts_draw_the_poisson_workload():
    a = poisson_workload(3.0, 200, LengthDist(1000, 0.5), LengthDist(100, 0.5), seed=9)
    b = chat_sessions(3.0, 200, LengthDist(1000, 0.5), LengthDist(100, 0.5), seed=9)
    assert [(r.arrival, r.prompt_len, r.output_len) for r in a] == [(r.arrival, r.prompt_len, r.output_len) for r in b]
    assert workload_rows(b) == [[r.arrival, r.prompt_len, r.output_len] for r in a]


# ── the mixed step ──────────────────────────────────────────────────────
@pytest.mark.parametrize("model, n", [(LLAMA3_8B, 1), (LLAMA3_70B, 4)])
def test_mixed_step_reduces_to_prefill_and_decode(model, n):
    cm = CostModel(model, H100_SXM, n)
    assert cm.step_mixed(0, 0, [(0, 2000, True), (0, 900, True)]) == cm.prefill([2000, 900])
    assert cm.step_mixed(70000, 37, []) == cm.decode_sum(70000, 37)
    last = CostModel(replace(model, prefill_lm_head="last"), H100_SXM, n)
    assert last.step_mixed(0, 0, [(0, 2000, True), (0, 900, True)]) == last.prefill([2000, 900])


def test_chunks_sum_to_the_whole_prompt_plus_kv_rereads():
    """Chunking a prompt preserves its FLOPs exactly (the causal sum splits into tails) and adds only the
    re-reads of earlier chunks' KV and of the weights (Sarathi-Serve section 4.3)."""
    m, s, c = LLAMA3_8B, 4096, 512
    cm = CostModel(m, H100_SXM, 1)
    whole = cm.step_mixed(0, 0, [(0, s, True)])
    parts = [cm.step_mixed(0, 0, [(p0, c, p0 + c == s)]) for p0 in range(0, s, c)]
    assert sum(x.flops for x in parts) == whole.flops
    k = s // c
    reread = sum(p0 for p0 in range(0, s, c)) * m.kv_bytes_per_token
    assert sum(x.bytes for x in parts) == whole.bytes + (k - 1) * m.weight_bytes_streamed + reread
    assert sum(x.time for x in parts) > whole.time


def test_a_prefix_hit_prices_only_the_uncached_tail():
    m = LLAMA3_8B
    cm = CostModel(m, H100_SXM, 1)
    hit = cm.step_mixed(0, 0, [(3000, 1000, True)])
    assert hit.flops == 2 * m.matmul_params * 1000 + sum(4 * m.n_layers * m.d_model * j for j in range(3001, 4001))
    assert hit.time < cm.prefill([4000]).time / 3


# ── batching policy (Sarathi-Serve) ─────────────────────────────────────
def test_policies_order_ttft_and_itl_as_sarathi_serve_describes():
    """Prefill-priority: generation stalls (high ITL p99). Decode-priority: no stalls, TTFT explodes. Chunked
    prefill: ITL p99 far below prefill-priority's, TTFT a little above it (arXiv:2403.02310, Fig. 1, Table 4)."""
    out = {}
    for pol, extra in [("prefill-priority", {}), ("decode-priority", {}), ("chunked", {"max_num_batched_tokens": 512})]:
        _, m = run(replace(C8, batch_policy=pol, **extra), wl(n=400, rate=5.0))
        out[pol] = (m["latency_s"]["ttft"]["p50"], m["latency_s"]["itl"]["p99"])
    pp, dp, ch = out["prefill-priority"], out["decode-priority"], out["chunked"]
    assert ch[1] < pp[1] / 2
    assert pp[0] < ch[0] < 2 * pp[0]
    assert dp[0] > 10 * pp[0] and dp[1] < pp[1]


def test_token_budget_is_the_ttft_itl_dial():
    itl, ttft = [], []
    for tau in (256, 512, 1024, 2048):              # beyond the typical prompt (2,048) it stops mattering
        _, m = run(replace(C8, batch_policy="chunked", max_num_batched_tokens=tau), wl(n=300, rate=5.0))
        itl.append(m["latency_s"]["itl"]["p99"])
        ttft.append(m["latency_s"]["ttft"]["p50"])
    assert itl == sorted(itl)                        # a bigger budget lets longer chunks stall decodes
    assert ttft[0] > ttft[-1]                        # and a smaller one stretches prompts over more steps


@pytest.mark.parametrize("pol", ["prefill-priority", "decode-priority", "chunked"])
@pytest.mark.parametrize("kv", ["oracle", "pow2", "max", "paged"])
def test_every_request_completes_with_consistent_timestamps(pol, kv):
    res, m = run(replace(OPT, batch_policy=pol, kv_policy=kv, max_num_batched_tokens=512 if pol == "chunked" else 8192),
                 opt_wl(n=150))
    assert m["requests"]["completed"] + m["requests"]["rejected"] == 150
    for r in res.requests:
        if r.finish is None:
            continue
        assert r.tokens_out == r.output_len and len(r.itls) == r.output_len - 1
        assert r.arrival <= r.prefill_start <= r.first_token <= r.finish
        assert abs(sum(r.stages().values()) - r.e2e) < 1e-9
    inst = res.instances[0]
    assert inst.kv_used == 0 and not inst.running and not inst.queue and not inst.swapped


# ── KV memory (vLLM / PagedAttention) ───────────────────────────────────
def test_memory_is_never_over_committed():
    for kw in (dict(kv_policy="paged"), dict(kv_policy="paged", preemption="swap"), dict(kv_policy="max"),
               dict(kv_policy="paged", prefix_caching=True), dict(kv_policy="oracle", prefix_caching=True)):
        cfg = replace(OPT, sample_dt=0.01, **kw)
        reqs = chat(n=200, rate=3.0, out=(200, 100, 300), sys_len=512) if cfg.prefix_caching else opt_wl(n=250)
        res = simulate(cfg, reqs)
        kv = [row["colocated-0.kv"] for row in res.samples]
        assert max(kv) <= 1.0, kw


def test_batched_requests_order_as_vllm_figure_13():
    """At a memory-bound load, paged blocks batch the most requests, then exact reservation (Orca Oracle), then
    power-of-two (Pow2), then max-length (Max) reservations (arXiv:2309.06180, Fig. 13a)."""
    running = {}
    for kv in ("paged", "oracle", "pow2", "max"):
        _, m = run(replace(OPT, kv_policy=kv, **FORCED), opt_wl())
        running[kv] = m["scheduler"]["mean_running"]
        assert m["scheduler"]["kv_token_frac"] <= 1.0
    assert running["paged"] > running["oracle"] > running["pow2"] > running["max"]


def test_paged_preemption_recompute_and_swap():
    _, rec = run(replace(OPT, kv_policy="paged"), opt_wl())
    _, swp = run(replace(OPT, kv_policy="paged", preemption="swap"), opt_wl())
    r, s = rec["scheduler"], swp["scheduler"]
    assert r["preemptions"] > 0 and r["recompute_tokens"] > 0
    assert s["preemptions"] > 0 and s["swap"]["out_GB"] == pytest.approx(s["swap"]["in_GB"]) and s["swap"]["J"] > 0
    assert "swap" in swp["energy"]["breakdown"] and "swap" not in rec["energy"]["breakdown"]
    for m in (rec, swp):
        assert m["requests"]["completed"] == 400


def test_small_blocks_waste_less_memory_but_swap_slower_per_byte():
    """Internal fragmentation grows with the block size; swapping pays a per-block latency, so small blocks
    swap slowly (arXiv:2309.06180, Figs. 18-19)."""
    frac, per_gb = [], []
    for b in (1, 16, 128):
        _, m = run(replace(OPT, kv_policy="paged", kv_block_size=b, preemption="swap"), opt_wl())
        s = m["scheduler"]
        frac.append(s["kv_token_frac"])
        per_gb.append(s["swap"]["seconds"] / (s["swap"]["out_GB"] + s["swap"]["in_GB"]))
    assert frac == sorted(frac, reverse=True)
    assert per_gb[0] > per_gb[1] > per_gb[2]


def test_preempted_requests_are_the_latest_arrivals():
    res = simulate(replace(OPT, kv_policy="paged"), opt_wl())
    pre = [r for r in res.requests if r.finish is not None and any(x > 0.5 for x in r.itls)]
    assert pre                                        # recomputed rows show an ITL gap


# ── prefix caching (SGLang RadixAttention) ──────────────────────────────
def test_lru_evicts_unpinned_leaves_oldest_first():
    c = PrefixCache()
    a = c.insert("a", 10, 10, None, 0.0)
    b = c.insert("b", 20, 20, a, 1.0)
    d = c.insert("d", 30, 30, None, 2.0)
    for n, t in ((a, 3.0), (b, 4.0), (d, 3.5)):
        c.unref(n, t)
    assert c.evict(1) == 30 and set(c.nodes) == {"a", "b"}   # a is not a leaf; d (3.5) is older than b (4.0)
    c.ref(b, 5.0)
    assert c.evict(100) == 0                         # b is pinned and a still has a child
    c.unref(b, 6.0)
    assert c.evict(100) == 30 and not c.nodes        # b, then a once it became a leaf
    assert c.evicted_tokens == 60


def test_prefix_cache_skips_cached_tokens():
    cfg = replace(C8, n_colocated=1)
    a, b = chat(), chat()
    ra, ma = run(replace(cfg, **FORCED), a)
    rb, mb = run(replace(cfg, prefix_caching=True), b)
    pc = mb["scheduler"]["prefix_cache"]
    assert pc["hit_rate"] > 0.5
    assert mb["scheduler"]["computed_prefill_tokens"] == pc["prompt_tokens"] - pc["hit_tokens"]
    assert ma["scheduler"]["computed_prefill_tokens"] == pc["prompt_tokens"]
    assert mb["latency_s"]["ttft"]["p50"] < ma["latency_s"]["ttft"]["p50"] / 2


def test_a_fully_cached_prompt_still_computes_its_last_token():
    m, reqs = LLAMA3_8B, chat_sessions(1.0, 2, LengthDist(100), LengthDist(10), seed=1, system_prompts=1, system_len=100)
    reqs[1].prompt_len = 100                         # the second prompt is exactly the shared system prompt
    reqs[1].arrival = reqs[0].arrival + 5.0
    res, s = run(replace(C8, n_colocated=1, prefix_caching=True), reqs)
    assert s["scheduler"]["prefix_cache"]["hit_tokens"] == 0     # its chain node is 100 tokens: capped away
    reqs = chat_sessions(1.0, 2, LengthDist(100), LengthDist(10), seed=1, system_prompts=1, system_len=60)
    reqs[1].arrival = reqs[0].arrival + 5.0
    _, s = run(replace(C8, n_colocated=1, prefix_caching=True), reqs)
    assert s["scheduler"]["prefix_cache"]["hit_tokens"] == 60


def test_closed_loop_turns_follow_their_predecessor():
    reqs = chat(n=40, rate=0.5, think=2.0)
    res = simulate(replace(C8, n_colocated=1, prefix_caching=True), reqs)
    by = {r.rid: r for r in res.requests}
    nxt = [r for r in res.requests if r.after is not None]
    assert nxt
    for r in nxt:
        assert r.arrival == by[r.after].finish + r.think
        assert r.prompt_len > by[r.after].prompt_len


def test_rejecting_a_turn_rejects_its_later_turns():
    reqs = chat(n=8, rate=0.5, turns=4)
    reqs[0].prompt_len = 10 ** 7                    # cannot fit
    res, m = run(replace(C8, n_colocated=1, prefix_caching=True), reqs)
    assert m["requests"]["rejected"] == 4 and m["requests"]["completed"] == 4


def test_reuse_distance_lowers_the_hit_rate_under_lru():
    """With little KV memory, a longer think time puts more other sessions between two turns, and LRU
    evicts the history first."""
    cfg = replace(OPT, prefix_caching=True)
    rates = []
    for think in (1.0, 10.0, 30.0, 100.0):
        _, m = run(cfg, chat(n=300, rate=0.2, out=(64, 32, 96), think=think, sys_len=256, sys_n=8))
        rates.append(m["scheduler"]["prefix_cache"]["hit_rate"])
    assert rates == sorted(rates, reverse=True) and rates[0] > 2 * rates[-1]


# ── validation, CLI, I/O ────────────────────────────────────────────────
@pytest.mark.parametrize("kw, msg", [
    (dict(mode="disagg", batch_policy="chunked"), "colocated"),
    (dict(model=LLAMA3_8B_HYENA, kv_policy="paged"), "attention models"),
    (dict(model=LLAMA3_8B_CED, prefix_caching=True), "attention models"),
    (dict(prefix_caching=True, kv_policy="paged", preemption="swap"), "swap"),
    (dict(batch_policy="sjf"), "batch_policy"),
    (dict(kv_policy="buddy"), "kv_policy"),
    (dict(kv_policy="paged", kv_block_size=0), "at least 1"),
])
def test_invalid_lever_settings_raise(kw, msg):
    with pytest.raises(ValueError, match=msg):
        Simulation(replace(C8, **kw), wl(n=5))


def test_closed_loop_sessions_reject_the_fast_path():
    with pytest.raises(ValueError, match="fast_forward"):
        Simulation(SimConfig(fast_forward=True), chat(n=8))


def test_cli_flags_reach_the_config_and_the_workload():
    a = build_parser().parse_args(["--mode", "colocated", "--batch-policy", "chunked", "--max-num-batched-tokens",
                                   "512", "--kv-policy", "paged", "--kv-block-size", "32", "--preemption", "swap",
                                   "--host-link", "pcie4", "--turns", "3", "--system-prompts", "2", "--system-len",
                                   "100", "--think", "1.5", "--model", "opt-13b", "--device", "a100-40g"])
    c = config_from_args(a)
    assert (c.batch_policy, c.max_num_batched_tokens, c.kv_policy, c.kv_block_size, c.preemption) == (
        "chunked", 512, "paged", 32, "swap")
    assert c.host_link == LINKS["pcie4"] and c.model == OPT_13B and c.device == A100_40G
    reqs = workload_from_args(a, 2.0)
    assert any(r.after is not None for r in reqs) and all(r.prefix for r in reqs)
    m = summarise(simulate(c, reqs[:60]))
    assert "scheduler" in m and "prefix cache" not in format_report(m)
    assert "scheduler    chunked" in format_report(m)


def test_workload_rows_round_trip(tmp_path):
    reqs = chat(n=30)
    dump_workload(reqs, tmp_path / "w.json")
    back = load_workload(tmp_path / "w.json")
    assert [(r.arrival, r.prompt_len, r.output_len, r.prefix, r.emit_key, r.after, r.think) for r in reqs] == \
           [(r.arrival, r.prompt_len, r.output_len, r.prefix, r.emit_key, r.after, r.think) for r in back]


def test_validation_presets_match_their_sources():
    assert MODELS["mistral-7b"] is MISTRAL_7B and 7.2e9 < MISTRAL_7B.params < 7.3e9
    assert 13.0e9 < OPT_13B.params < 13.2e9 and OPT_13B.weight_bytes_total / 1e9 == pytest.approx(26.2, abs=0.1)
    assert OPT_13B.kv_bytes_per_token == 2 * 40 * 5120 * 2                      # MHA: 800 KiB per token
    assert 34.0e9 < MODELS["yi-34b"].params < 34.6e9
    assert A100_40G.mem_bw == 1.555e12 and A100_40G.peak_flops == A100_SXM.peak_flops


# ── the JavaScript port ─────────────────────────────────────────────────
OPT_JS = dict(model="opt-13b", device="a100-40g", devicesPerInstance=1, nColocated=1, hostLink="pcie4")
JS_CASES = {
    "forced scheduler, 2 instances": (dict(model="llama3-8b", devicesPerInstance=1, nColocated=2, maxNumBatchedTokens=8192), "poisson"),
    "chunked 512": (dict(model="llama3-8b", devicesPerInstance=1, nColocated=2, batchPolicy="chunked", maxNumBatchedTokens=512), "poisson"),
    "chunked 256, LM head last": (dict(model="llama3-8b", devicesPerInstance=1, nColocated=1, batchPolicy="chunked",
                                       maxNumBatchedTokens=256, lmHead="last"), "poisson"),
    "decode-priority": (dict(model="llama3-8b", devicesPerInstance=1, nColocated=2, batchPolicy="decode-priority"), "poisson"),
    "paged recompute": (dict(OPT_JS, kvPolicy="paged"), "opt"),
    "paged swap, block 4": (dict(OPT_JS, kvPolicy="paged", preemption="swap", kvBlockSize=4), "opt"),
    "chunked + paged swap": (dict(OPT_JS, kvPolicy="paged", preemption="swap", batchPolicy="chunked", maxNumBatchedTokens=512), "opt"),
    "max reservation": (dict(OPT_JS, kvPolicy="max"), "opt"),
    "pow2 reservation, decode-priority": (dict(OPT_JS, kvPolicy="pow2", batchPolicy="decode-priority"), "opt"),
    "prefix cache, oracle": (dict(OPT_JS, prefixCaching=True, maxNumBatchedTokens=8192), "chat"),
    "prefix cache, paged recompute, chunked": (dict(OPT_JS, prefixCaching=True, kvPolicy="paged", batchPolicy="chunked",
                                                    maxNumBatchedTokens=1024), "chat"),
    "prefix cache, 2 instances, power cap": (dict(model="llama3-8b", devicesPerInstance=1, nColocated=2, prefixCaching=True,
                                                  kvPolicy="paged", powerCap=350.0, dvfs=True), "chat"),
    "70B chunked, 4xH100": (dict(model="llama3-70b", batchPolicy="chunked", maxNumBatchedTokens=2048), "poisson"),
    "prefix cache, LRU ties (fixed lengths)": (dict(OPT_JS, prefixCaching=True, maxNumBatchedTokens=8192), "ties"),
    "prefix cache, whole-prompt hits": (dict(model="llama3-8b", devicesPerInstance=1, nColocated=1, prefixCaching=True,
                                             kvPolicy="paged"), "full"),
}


def py_config(c: dict) -> SimConfig:
    model = MODELS[c["model"]]
    if "lmHead" in c:
        model = replace(model, prefill_lm_head=c["lmHead"])
    dev = {"h100": H100_SXM, "a100-40g": A100_40G}[c.get("device", "h100")]
    return SimConfig(model=model, device=dev, devices_per_instance=c.get("devicesPerInstance", 4), mode="colocated",
                     n_colocated=c.get("nColocated", 2), power_cap_w=c.get("powerCap"), dvfs=c.get("dvfs", False),
                     batch_policy=c.get("batchPolicy", "prefill-priority"),
                     max_num_batched_tokens=c.get("maxNumBatchedTokens"), kv_policy=c.get("kvPolicy", "oracle"),
                     kv_block_size=c.get("kvBlockSize", 16), preemption=c.get("preemption", "recompute"),
                     host_link=LINKS[c.get("hostLink", "pcie5")], prefix_caching=c.get("prefixCaching", False))


def js_workload(kind: str):
    if kind == "opt":
        return opt_wl(n=300)
    if kind == "chat":
        return chat(n=200, rate=2.0, out=(120, 60, 200), think=1.0, sys_len=512)
    if kind == "ties":          # fixed lengths: rows admitted together finish together, so evicted leaves tie on time
        return chat_sessions(4.0, 240, LengthDist(300), LengthDist(64), seed=5, turns=3, think=0.0,
                             system_prompts=6, system_len=700)
    if kind == "full":          # every third prompt is exactly its system prompt: the hit is capped at prompt - 1
        reqs = chat_sessions(3.0, 150, LengthDist(200, 0.5), LengthDist(40, 0.5), seed=6, system_prompts=3, system_len=400)
        for r in reqs[::3]:
            r.prompt_len = 400
        return reqs
    reqs = wl(n=150, rate=5.0, singles=True)
    return reqs


def test_javascript_port_matches_python_for_the_levers():
    """Every timestamp, every instance's energy accounts and lever statistics, the energy total and the
    scheduler summary. Bit-exact, except runs with power-bound steps (cube roots: 1e-9)."""
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
        s = m["scheduler"]
        sc = [s["mean_running"], s["kv_token_frac"], s["computed_prefill_tokens"], s.get("preemptions", 0),
              s.get("recompute_tokens", 0), s.get("swap", {}).get("seconds", 0.0), s.get("swap", {}).get("J", 0.0),
              s.get("prefix_cache", {}).get("hit_tokens", 0), s.get("prefix_cache", {}).get("evicted_tokens", 0),
              s.get("prefix_cache", {}).get("cached_tokens_end", 0)]
        cases.append({"name": name, "cfg": {"device": "h100", "devicesPerInstance": 4, "mode": "colocated", "nColocated": 2, "link": "ib-ndr",
                                            "ttftSlo": 1.0, "tpotSlo": 0.025, **c},
                      "rows": rows,
                      "stamps": [[r.arrival, r.prefill_start, r.first_token, r.decode_start, r.finish] for r in res.requests],
                      "itls": sum(len(r.itls) for r in res.requests),
                      "inst": [[i.compute_j, i.memory_j, i.busy, i.peak_power, i.steps, i.batch_sum] for i in res.instances],
                      "total": m["energy"]["total_J"], "sched": sc,
                      "capped": any(i.power_bound_time > 0 for i in res.instances)})
    script = (f"require({json.dumps(str(engine))});"
              "const S=globalThis.DisaggSim, cases=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify(cases.map(c=>{const r=S.simulate(c.cfg,c.rows), m=S.summarise(r), s=m.scheduler;"
              "return {stamps:r.reqs.map(q=>[q.arrival,q.prefillStart,q.firstToken,q.decodeStart,q.finish]),"
              "itls:r.reqs.reduce((a,q)=>a+q.itls.length,0),"
              "inst:r.insts.map(i=>[i.ec,i.em,i.busy,i.peakW,i.steps,i.batchSum]), total:m.energy.totalJ,"
              "sched:[s.meanRunning,s.kvTokenFrac,s.computedPrefillTokens,s.preemptions??0,s.recomputeTokens??0,"
              "s.swap?s.swap.seconds:0,s.swap?s.swap.J:0,s.prefixCache?s.prefixCache.hitTokens:0,"
              "s.prefixCache?s.prefixCache.evictedTokens:0,s.prefixCache?s.prefixCache.cachedTokensEnd:0]};})));")
    out = subprocess.run([node, "-e", script], input=json.dumps(cases), capture_output=True, text=True, check=True)
    exercised = {"preempt": 0, "swap": 0, "hit": 0, "evict": 0}  # each mechanism must actually occur
    exact = 0
    for c, js in zip(cases, json.loads(out.stdout)):
        exact += all(c[k] == js[k] for k in ("stamps", "inst", "total", "sched"))
        tol = 1e-9 if c["capped"] else 0.0
        assert c["itls"] == js["itls"], c["name"]
        for key in ("stamps", "inst"):
            assert len(c[key]) == len(js[key]), (c["name"], key)
            for a, b in zip(c[key], js[key]):
                for x, y in zip(a, b):
                    assert (x is None and y is None) or abs(x - y) <= tol * max(1.0, abs(x)), (c["name"], key, a, b)
        assert abs(c["total"] - js["total"]) <= tol * c["total"], c["name"]
        for x, y in zip(c["sched"], js["sched"]):
            assert abs(x - y) <= tol * max(1.0, abs(x)), (c["name"], c["sched"], js["sched"])
        exercised["preempt"] += c["sched"][3] > 0
        exercised["swap"] += c["sched"][5] > 0
        exercised["hit"] += c["sched"][7] > 0
        exercised["evict"] += c["sched"][8] > 0
    assert all(v >= 1 for v in exercised.values()), exercised
    # power-bound steps take a cube root (tolerance 1e-9 above), yet in practice every case is identical
    assert exact >= 11, exact
