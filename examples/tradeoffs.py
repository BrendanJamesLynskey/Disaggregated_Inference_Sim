"""The trade-off sweep (brief 20A2): levers x workloads x hardware, every metric, Pareto flags.

    python examples/tradeoffs.py                    # the full grid -> examples/tradeoffs.json (about 20 min, 2 workers)
    python examples/tradeoffs.py --workers 1 --log /tmp/tradeoffs.log
    python examples/tradeoffs.py --quick            # a small grid, for testing the script (not recorded)

Every configuration is a cluster of the same 8 GPUs serving Llama-3-70B: the baseline is two colocated instances of
four GPUs with tensor parallelism (all-reduces priced on NVLink), prefill-priority batching, whole-sequence KV
reservation and BF16. Each lever changes one thing (or, in the "modern" rows, a few) from there. For every
(workload, hardware, lever) point the sweep measures

* capacity: the highest arrival rate whose SLO attainment is at least 90% (DistServe's goodput, arXiv:2401.09670),
  found by doubling then bisection to 5%; requests the configuration rejects (KV that can never fit) count as misses.
  Attainment is measured over the requests that arrive after the first 10% and before the last session starts (the
  ramp-up and the drain are left out). Multi-turn workloads: the rate is of sessions, and a run has at least 200
  sessions and enough to keep them starting for four session lengths (estimated from the SLOs), so the closed loop
  reaches a steady state; their request rate is the session rate times the turns. Offline batch is measured
  saturated (every request at once): its capacity is the throughput of a 400-request batch;
* at that capacity: requests and output tokens per second, per GPU; cost per million output tokens (8 GPUs at the
  illustrative price per GPU-hour below); joules per output token (the power model, static power included);
* at one reference load per workload (half the baseline's capacity on H100, the same offered load for every lever):
  TTFT, TPOT and ITL p50 / p99, SLO attainment, peak and mean KV occupancy, and each lever's own statistics;
* memory: weights per GPU.

Pareto flags are computed per workload across every lever and hardware, for each pair of the headline metrics and
for all of them at once. ``effects`` gives each lever's relative change from the baseline on the same hardware.
The JSON is the single number source of the trade-offs site (brief 20B); results.py renders its summary into
results.md. Prices and the speculative acceptance rate are parameters, not measurements; accuracy is not simulated.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from disagg_sim.hardware import ACCELERATORS, LLAMA3_70B, Parallel  # noqa: E402
from disagg_sim.metrics import summarise  # noqa: E402
from disagg_sim.sim import SimConfig, Simulation, simulate  # noqa: E402
from disagg_sim.speculative import Speculative  # noqa: E402
from disagg_sim.workload import WORKLOADS  # noqa: E402

OUT = ROOT / "examples" / "tradeoffs.json"
GPUS = 8
N_REQUESTS = 400        # single-turn workloads
SESSIONS_MIN = 200      # multi-turn workloads: at least this many sessions per run ...
WINDOW_SESSIONS = 4     # ... and enough that sessions keep starting for 4 session lengths (a steady state)
SEED = 1
SLO_TARGET = 0.9
CAP_TOL = 0.05
ALPHA = 0.7          # speculative acceptance rate per drafted token: a parameter (it depends on draft, target, text)
# Illustrative prices per GPU-hour: round numbers of the order of 2026 cloud list prices, not quotes. Cost per token
# scales with them linearly, so the ranking of levers on one device does not depend on them.
USD_PER_GPU_HOUR = {"h100": 3.00, "h200": 3.50, "b200": 5.00}
HARDWARE = ["h100", "h200", "b200"]

TP4 = Parallel(tp=4)
FP8 = dict(weight_format="fp8", compute_format="fp8")
MODERN = dict(batch_policy="chunked", max_num_batched_tokens=2048, kv_policy="paged", prefix_caching=True,
              kv_format="fp8", **FP8)

# (key, family, label, overrides of the baseline, hardware it applies to (None: all))
LEVERS = [
    ("baseline", "baseline", "2 x TP4 colocated, prefill-priority, reserved KV, BF16", {}, None),
    ("decode-priority", "batching", "Decode-priority batching", dict(batch_policy="decode-priority"), None),
    ("chunked-512", "batching", "Chunked prefill, 512-token budget",
     dict(batch_policy="chunked", max_num_batched_tokens=512), None),
    ("chunked-2048", "batching", "Chunked prefill, 2,048-token budget",
     dict(batch_policy="chunked", max_num_batched_tokens=2048), None),
    ("paged", "kv-memory", "Paged KV, preempt by recompute", dict(kv_policy="paged"), None),
    ("paged-swap", "kv-memory", "Paged KV, preempt by swap to host", dict(kv_policy="paged", preemption="swap"), None),
    ("prefix-cache", "prefix-caching", "Prefix caching (paged KV)", dict(kv_policy="paged", prefix_caching=True), None),
    ("disagg-1p1d", "disaggregation", "Disaggregated 1P1D (TP4 each)", dict(mode="disagg", n_prefill=1, n_decode=1),
     None),
    ("disagg-2p1d", "disaggregation", "Disaggregated 2P (TP2) + 1D (TP4)",
     dict(mode="disagg", n_prefill=2, n_decode=1, prefill_devices_per_instance=2, prefill_parallel=Parallel(tp=2)),
     None),
    ("disagg-levers", "disaggregation", "Disaggregated 1P1D, paged decode, prefix-cached prefill",
     dict(mode="disagg", n_prefill=1, n_decode=1, kv_policy="paged", prefix_caching=True), None),
    ("tp8", "parallelism", "1 x TP8", dict(n_colocated=1, devices_per_instance=8, parallel=Parallel(tp=8)), None),
    ("tp2x4", "parallelism", "4 x TP2", dict(n_colocated=4, devices_per_instance=2, parallel=Parallel(tp=2)), None),
    ("tp2pp2", "parallelism", "2 x (TP2, PP2), 2 micro-batches", dict(parallel=Parallel(tp=2, pp=2)), None),
    ("tp2pp2-mb1", "parallelism", "2 x (TP2, PP2), no micro-batching", dict(parallel=Parallel(tp=2, pp=2, microbatches=1)),
     None),
    ("w8a8-fp8", "quantisation", "FP8 weights and matmuls (W8A8)", dict(FP8), None),
    ("w4-int4", "quantisation", "INT4 weights, BF16 matmuls (W4A16)", dict(weight_format="int4"), None),
    ("kv-fp8", "quantisation", "FP8 KV cache", dict(kv_format="fp8"), None),
    ("fp8-all", "quantisation", "FP8 weights, matmuls and KV", dict(kv_format="fp8", **FP8), None),
    ("w4a4-fp4", "quantisation", "FP4 weights and matmuls (W4A4)", dict(weight_format="fp4", compute_format="fp4"),
     ["b200"]),
    ("spec-mtp", "speculative", f"Speculative, MTP head, gamma 3, alpha {ALPHA}",
     dict(speculative=Speculative("mtp", 3, ALPHA)), None),
    ("spec-1b", "speculative", f"Speculative, Llama-3.2-1B draft, gamma 4, alpha {ALPHA}",
     dict(speculative=Speculative("llama3.2-1b", 4, ALPHA)), None),
    ("modern-colocated", "combined", "Chunked 2048 + paged + prefix cache + FP8 (W8A8, KV)", dict(MODERN), None),
    ("modern-disagg", "combined", "Disaggregated 1P1D + paged + prefix cache + FP8",
     dict(mode="disagg", n_prefill=1, n_decode=1, kv_policy="paged", prefix_caching=True, kv_format="fp8", **FP8), None),
    ("modern-spec", "combined", "Modern colocated + speculative (MTP)",
     dict(MODERN, speculative=Speculative("mtp", 3, ALPHA)), None),
]
QUICK = ["baseline", "chunked-512", "disagg-levers", "w8a8-fp8", "spec-mtp"]
OFFLINE = {"offline-batch"}    # measured saturated: every request arrives at once
SATURATED = 1e6

# headline metrics, with the direction that is better
OBJECTIVES = {"goodput_req_s_per_gpu": "max", "usd_per_mtok": "min", "j_per_tok": "min", "ttft_p99": "min",
              "tpot_p99": "min", "itl_p99": "min"}


def config(hw: str, workload: str, overrides: dict) -> SimConfig:
    w = WORKLOADS[workload]
    base = SimConfig(model=LLAMA3_70B, device=ACCELERATORS[hw], mode="colocated", n_colocated=2, devices_per_instance=4,
                     parallel=TP4, ttft_slo=w.ttft_slo, tpot_slo=w.tpot_slo)
    return replace(base, **overrides)


def js_config(cfg: SimConfig, hw: str) -> dict:
    """The same configuration in the JavaScript engine's keys (web/sim_engine.js), for the site's live simulator."""
    def par(p):
        if p is None:
            return None
        return {k: v for k, v in (("tp", p.tp), ("pp", p.pp), ("ep", p.ep), ("microbatches", p.microbatches),
                                  ("expertImbalance", p.expert_imbalance), ("link", p.link)) if v is not None}
    out = {"model": "llama3-70b", "device": hw, "devicesPerInstance": cfg.devices_per_instance, "mode": cfg.mode,
           "nPrefill": cfg.n_prefill, "nDecode": cfg.n_decode, "nColocated": cfg.n_colocated, "link": "ib-ndr",
           "ttftSlo": cfg.ttft_slo, "tpotSlo": cfg.tpot_slo, "batchPolicy": cfg.batch_policy,
           "maxNumBatchedTokens": cfg.max_num_batched_tokens, "kvPolicy": cfg.kv_policy, "preemption": cfg.preemption,
           "hostLink": "pcie5", "prefixCaching": cfg.prefix_caching, "parallel": par(cfg.parallel),
           "prefillParallel": par(cfg.prefill_parallel), "decodeParallel": par(cfg.decode_parallel),
           "prefillDevicesPerInstance": cfg.prefill_devices_per_instance,
           "decodeDevicesPerInstance": cfg.decode_devices_per_instance, "weightFormat": cfg.weight_format,
           "kvFormat": cfg.kv_format, "computeFormat": cfg.compute_format,
           "speculative": asdict(cfg.speculative) if cfg.speculative is not None else None}
    return {k: v for k, v in out.items() if v is not None}


def session_seconds(workload: str) -> float:
    """A generous estimate of one session's length: each turn's think time plus its SLO-limited response."""
    w = WORKLOADS[workload]
    return w.turns * (w.think + w.ttft_slo + w.output.mean * w.tpot_slo)


def n_requests(workload: str, rate: float) -> int:
    w = WORKLOADS[workload]
    if w.turns == 1:
        return N_REQUESTS
    return w.turns * max(SESSIONS_MIN, math.ceil(rate * WINDOW_SESSIONS * session_seconds(workload)))


def measure(cfg: SimConfig, workload: str, rate: float) -> dict:
    w = WORKLOADS[workload]
    res = simulate(cfg, w.generate(rate, n_requests(workload, rate), seed=SEED))
    m = summarise(res)
    every = sorted(res.requests, key=lambda r: r.arrival)
    last_start = max(r.arrival for r in res.requests if r.after is None)
    window = [r for r in every[int(len(every) * cfg.warmup_frac):] if r.arrival <= last_start]
    span = window[-1].arrival - window[0].arrival if len(window) > 1 else 0.0
    met = 0
    for r in window:                          # a rejected request (never finished) is a miss
        if r.finish is not None and r.ttft <= cfg.ttft_slo and (r.tpot is None or r.tpot <= cfg.tpot_slo):
            met += 1
    lat = m["latency_s"]
    kv = [v for row in res.samples for k, v in row.items() if k.endswith(".kv")]
    if rate < SATURATED / 2 and span > 0:     # steady state: what arrives in the window is what is served
        req_s, tok_s = (len(window) - 1) / span, sum(r.output_len for r in window[:-1]) / span
    else:                                      # saturated: the batch's throughput
        req_s, tok_s = m["throughput"]["req_per_s"], m["throughput"]["output_tok_per_s"]
    out = {"rate": rate, "attainment": met / len(window) if window else 0.0, "requests": len(res.requests),
           "req_s": req_s, "tok_s": tok_s,
           "ttft_p50": lat["ttft"]["p50"], "ttft_p99": lat["ttft"]["p99"], "tpot_p50": lat["tpot"]["p50"],
           "tpot_p99": lat["tpot"]["p99"], "itl_p50": lat["itl"]["p50"], "itl_p99": lat["itl"]["p99"],
           "e2e_p50": lat["e2e"]["p50"], "j_per_tok": m["energy"]["J_per_output_token"],
           "avg_power_w": m["energy"]["avg_power_W"], "completed": m["requests"]["completed"],
           "rejected": m["requests"]["rejected"], "kv_peak_frac": max(kv) if kv else 0.0,
           "kv_mean_frac": sum(kv) / len(kv) if kv else 0.0, "sim_time_s": m["sim_time_s"]}
    s = m.get("scheduler", {})
    for k in ("mean_running", "preemptions"):
        if k in s:
            out[k] = s[k]
    if "prefix_cache" in s:
        out["prefix_hit_rate"] = s["prefix_cache"]["hit_rate"]
    if "speculative" in s:
        out["tokens_per_verify"] = s["speculative"]["tokens_per_verify"]
        out["draft_time_frac"] = s["speculative"]["draft_time_frac"]
    if "parallel" in m:
        busy = comm = 0.0
        for v in m["parallel"].values():
            busy, comm = busy + v["busy_s"], comm + v["comm_s"]
        out["comm_frac"] = comm / busy if busy else 0.0
    return out


def capacity(cfg: SimConfig, workload: str, start: float) -> tuple[dict | None, int]:
    """The highest rate with attainment >= SLO_TARGET (doubling, then bisection to CAP_TOL); its measurement."""
    runs, best = 0, None

    def ok(rate):
        nonlocal runs, best
        runs += 1
        r = measure(cfg, workload, rate)
        good = r["attainment"] >= SLO_TARGET
        if good and (best is None or rate > best["rate"]):
            best = r
        return good
    lo, hi = 0.0, start
    if ok(hi):
        lo = hi
        while hi < 1e4:              # (a saturated start, 1e6, is not raised: every request already arrives at once)
            hi = hi * 2
            if not ok(hi):
                break
            lo = hi
    else:
        while hi > start / 64:
            hi = hi / 2
            if ok(hi):
                lo = hi
                hi = hi * 2
                break
        if lo == 0.0:
            return None, runs
    while hi - lo > CAP_TOL * hi:
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return best, runs


def point(job: tuple) -> dict:
    workload, hw, key, ref_rate, start = job
    fam, label, over = next((f, lab, o) for k, f, lab, o, _ in LEVERS if k == key)
    t0 = time.perf_counter()
    cfg = config(hw, workload, over)
    try:
        cap, runs = capacity(cfg, workload, start)
        at = measure(cfg, workload, ref_rate) if ref_rate else None
    except ValueError as e:           # a combination this hardware cannot hold: recorded, not hidden
        return {"workload": workload, "hardware": hw, "lever": key, "family": fam, "label": label,
                "error": str(e), "js_cfg": js_config(cfg, hw)}
    inst = cfg.n_colocated if cfg.mode == "colocated" else cfg.n_prefill + cfg.n_decode
    gpus = 0
    for role, n in (("colocated", cfg.n_colocated),) if cfg.mode == "colocated" else (("prefill", cfg.n_prefill),
                                                                                       ("decode", cfg.n_decode)):
        gpus += n * cfg.pool(role)[1]
    sim = Simulation(cfg, [])
    wpg = max(i.cost.model.weight_bytes_total / i.cost.n_devices + i.cost.draft_resident_bytes / i.cost.n_devices
              for i in sim.instances)
    price = USD_PER_GPU_HOUR[hw] * gpus
    metrics = {"gpus": gpus, "instances": inst, "weights_gb_per_gpu": wpg / 1e9, "capacity_runs": runs}
    if cap is not None:
        metrics.update({"capacity_rate": cap["rate"], "goodput_req_s": cap["req_s"],
                        "goodput_req_s_per_gpu": cap["req_s"] / gpus, "tok_s_at_capacity": cap["tok_s"],
                        "tok_s_per_gpu": cap["tok_s"] / gpus,
                        "usd_per_mtok": price / 3600 / cap["tok_s"] * 1e6 if cap["tok_s"] else None,
                        "j_per_tok": cap["j_per_tok"], "avg_power_w_at_capacity": cap["avg_power_w"]})
    else:
        metrics.update({"capacity_rate": 0.0, "goodput_req_s": 0.0, "goodput_req_s_per_gpu": 0.0,
                        "tok_s_at_capacity": 0.0, "tok_s_per_gpu": 0.0, "usd_per_mtok": None, "j_per_tok": None})
    if at is not None:
        for k in ("ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99", "itl_p50", "itl_p99", "e2e_p50", "attainment",
                  "kv_peak_frac", "kv_mean_frac", "rejected", "mean_running", "preemptions", "prefix_hit_rate",
                  "tokens_per_verify", "draft_time_frac", "comm_frac"):
            if k in at:
                metrics[k if k not in ("attainment", "rejected") else f"{k}_at_load"] = at[k]
        metrics["j_per_tok_at_load"] = at["j_per_tok"]
    return {"workload": workload, "hardware": hw, "lever": key, "family": fam, "label": label, "metrics": metrics,
            "js_cfg": js_config(cfg, hw), "seconds": time.perf_counter() - t0}


def clean(x):
    """NaN and infinity -> None, so the JSON is strict."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def pareto(points: list[dict]) -> None:
    """Flag each point non-dominated within its workload, for each pair of objectives and for all of them."""
    def val(p, k):
        v = p["metrics"].get(k) if "metrics" in p else None
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return math.inf if OBJECTIVES[k] == "min" else -math.inf
        return v if OBJECTIVES[k] == "min" else -v           # smaller is better throughout
    keys = list(OBJECTIVES)
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
    for wl in {p["workload"] for p in points}:
        pts = [p for p in points if p["workload"] == wl and "metrics" in p]
        for group in [tuple(pr) for pr in pairs] + [tuple(keys)]:
            vecs = [[val(p, k) for k in group] for p in pts]
            for i, p in enumerate(pts):
                dom = any(all(x <= y for x, y in zip(vecs[j], vecs[i])) and any(x < y for x, y in zip(vecs[j], vecs[i]))
                          for j in range(len(pts)) if j != i)
                ok = not dom and all(math.isfinite(x) for x in vecs[i])
                p.setdefault("pareto", {})["all" if len(group) > 2 else f"{group[0]}|{group[1]}"] = ok


def effects(points: list[dict]) -> dict:
    """Relative change of every metric from the baseline on the same workload and hardware."""
    base = {(p["workload"], p["hardware"]): p for p in points if p["lever"] == "baseline" and "metrics" in p}
    out: dict = {}
    for p in points:
        b = base.get((p["workload"], p["hardware"]))
        if b is None or "metrics" not in p or p["lever"] == "baseline":
            continue
        d = {}
        for k in list(OBJECTIVES) + ["tok_s_per_gpu", "ttft_p50", "tpot_p50", "kv_peak_frac"]:
            x, y = p["metrics"].get(k), b["metrics"].get(k)
            if x is None or y is None or not y:
                d[k] = None
            else:
                d[k] = (x - y) / y
        out.setdefault(p["workload"], {}).setdefault(p["hardware"], {})[p["lever"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log", default="tradeoffs.log", help="progress log (one line per point)")
    ap.add_argument("--quick", action="store_true", help="a small grid to test the script; writes tradeoffs_quick.json")
    a = ap.parse_args()
    levers = [lv for lv in LEVERS if not a.quick or lv[0] in QUICK]
    hardware = ["h100"] if a.quick else HARDWARE
    log = open(a.log, "a", buffering=1)
    t0 = time.perf_counter()
    log.write(f"{date.today()} start: {len(levers)} levers x {len(WORKLOADS)} workloads x {len(hardware)} hardware\n")
    # 1. the baseline's capacity on H100 sets each workload's reference load (half of it) and the search start
    ref = {}
    for wl in WORKLOADS:
        cap, runs = capacity(config("h100", wl, {}), wl, SATURATED if wl in OFFLINE else 1.0)
        ref[wl] = 0.5 * cap["rate"] if cap else 0.25
        log.write(f"reference {wl}: baseline H100 capacity {cap['rate'] if cap else 0:.3f}/s ({runs} runs), "
                  f"reference load {ref[wl]:.3f}/s\n")
    jobs = [(wl, hw, key, ref[wl], SATURATED if wl in OFFLINE else 2 * ref[wl]) for wl in WORKLOADS for hw in hardware
            for key, _, _, _, only in levers if only is None or hw in only]
    points = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, p in enumerate(ex.map(point, jobs), 1):
            points.append(p)
            m = p.get("metrics", {})
            log.write(f"[{i}/{len(jobs)}] {time.perf_counter() - t0:7.0f}s {p['workload']:13s} {p['hardware']} "
                      f"{p['lever']:17s} " + (f"ERROR {p['error']}" if "error" in p else
                                              f"goodput/GPU {m['goodput_req_s_per_gpu']:.3f} $/Mtok {m['usd_per_mtok']} "
                                              f"runs {m['capacity_runs']} ({p['seconds']:.0f}s)") + "\n")
    pareto(points)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src"], capture_output=True,
                                text=True).stdout.strip())
    out = {
        "meta": {"generated": str(date.today()), "simulator_commit": commit + ("+dirty" if dirty else ""),
                 "model": LLAMA3_70B.name, "gpus": GPUS, "requests_per_run": N_REQUESTS, "sessions_min": SESSIONS_MIN,
                 "window_sessions": WINDOW_SESSIONS, "seed": SEED,
                 "slo_target": SLO_TARGET, "capacity_tolerance": CAP_TOL, "speculative_alpha": ALPHA,
                 "usd_per_gpu_hour": USD_PER_GPU_HOUR, "objectives": OBJECTIVES,
                 "machine": f"{platform.machine()} / Python {platform.python_version()}", "workers": a.workers,
                 "wall_s": round(time.perf_counter() - t0, 1), "quick": a.quick,
                 "notes": ["Prices per GPU-hour are illustrative; accuracy is not simulated (see the Numerics site).",
                           "Roofline cost model with illustrative efficiencies and power coefficients (see README).",
                           "Capacity: highest arrival rate with >= 90% of requests meeting both SLOs; rejected requests "
                           "count as misses. Multi-turn workloads: the rate is of sessions.",
                           "Latency columns are at the workload's reference load (half the H100 baseline capacity)."]},
        "workloads": {k: {"label": w.label, "ttft_slo": w.ttft_slo, "tpot_slo": w.tpot_slo, "turns": w.turns,
                          "session_s_estimate": session_seconds(k) if w.turns > 1 else None,
                          "think": w.think, "system_prompts": w.system_prompts, "system_len": w.system_len,
                          "prompt_mean": w.prompt.mean, "prompt_cv": w.prompt.cv, "output_mean": w.output.mean,
                          "output_cv": w.output.cv, "reference_rate": ref[k], "rationale": w.rationale}
                      for k, w in WORKLOADS.items()},
        "hardware": {h: {"name": ACCELERATORS[h].name, "usd_per_gpu_hour": USD_PER_GPU_HOUR[h],
                         "bf16_tflops": ACCELERATORS[h].peak_flops / 1e12, "hbm_tb_s": ACCELERATORS[h].mem_bw / 1e12,
                         "hbm_gb": ACCELERATORS[h].mem_capacity / 1e9,
                         "native_formats": dict(ACCELERATORS[h].native_formats)} for h in hardware},
        "levers": {k: {"family": f, "label": lab, "hardware": only} for k, f, lab, _, only in levers},
        "points": sorted(points, key=lambda p: (p["workload"], p["hardware"], p["lever"])),
        "effects": effects(points),
    }
    dest = OUT.with_name("tradeoffs_quick.json") if a.quick else OUT
    dest.write_text(json.dumps(clean(out), indent=1, allow_nan=False))
    log.write(f"wrote {dest} ({len(points)} points) in {time.perf_counter() - t0:.0f} s\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
