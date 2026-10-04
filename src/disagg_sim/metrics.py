"""Turning a finished simulation into numbers an architect can act on.

Three families of metric, mirroring what a hardware-architecture simulator
is expected to produce:

* **latency**     – TTFT, TPOT, inter-token latency, end-to-end; p50/p90/p99
* **utilisation** – busy fraction of every instance and of the KV link
* **hot-spots**   – where the average request spends its time, and which
                    resource is closest to saturation
* **power**       – average and peak power, energy per token, and where the
                    joules go (static, compute, memory, interconnect)
"""

from __future__ import annotations

import math
from statistics import fmean

from .sim import SimResult

STAGES = ["prefill_queue", "prefill", "kv_wait", "kv_transfer", "decode_queue", "decode"]

# Which resource "owns" each stage -- the one to upgrade if that stage dominates.
STAGE_OWNER = {"prefill_queue": "prefill", "prefill": "prefill", "kv_wait": "kv-link",
               "kv_transfer": "kv-link", "decode_queue": "decode", "decode": "decode"}


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile (same convention as numpy's default)."""
    return _percentile_sorted(sorted(xs), p)


def _percentile_sorted(s: list[float], p: float) -> float:
    if not s:
        return math.nan
    k = (len(s) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _dist(xs: list[float]) -> dict:
    # Sort once for all three percentiles (profiling showed the repeated sorts dominating
    # summarise). The mean stays fmean of the data, so every value is unchanged.
    s = sorted(xs)
    return {"mean": fmean(xs) if xs else math.nan, "p50": _percentile_sorted(s, 50),
            "p90": _percentile_sorted(s, 90), "p99": _percentile_sorted(s, 99),
            "max": s[-1] if s else math.nan}


def summarise(res: SimResult) -> dict:
    cfg = res.cfg
    done = sorted((r for r in res.requests if r.finish is not None), key=lambda r: r.arrival)
    steady = done[int(len(done) * cfg.warmup_frac):]

    ttft = [r.ttft for r in steady]
    tpot = [r.tpot for r in steady if r.tpot is not None]
    itl = [x for r in steady for x in r.itls]
    e2e = [r.e2e for r in steady]
    met = [r for r in steady
           if r.ttft <= cfg.ttft_slo and (r.tpot is None or r.tpot <= cfg.tpot_slo)]

    window = (steady[-1].arrival - steady[0].arrival) if len(steady) > 1 else math.nan
    out_tokens = sum(r.output_len for r in done)

    util = {i.name: i.busy / res.horizon for i in res.instances}
    link_util = res.link.busy / (cfg.link.channels * res.horizon)
    util["kv-link"] = link_util

    stage_means = {s: fmean(r.stages()[s] for r in steady) for s in STAGES} if steady else {}
    mean_e2e = fmean(e2e) if e2e else math.nan

    # Little's law: L = λ W, using the exact time-average population.
    lam = len(done) / res.horizon
    w = fmean(r.e2e for r in done) if done else math.nan
    little_L = res.in_system_area / res.horizon

    # The hot-spot is the stage where the average request waits longest, excluding
    # the irreducible decode time itself, mapped to the resource that owns it.
    # (Busy fraction alone is a poor guide: a continuous-batching decode instance is
    # "busy" whenever any sequence is running, even at batch size 1.)
    waits = {s: v for s, v in stage_means.items() if s != "decode"}
    hottest_stage = max(waits, key=waits.get) if waits else None
    owner = STAGE_OWNER.get(hottest_stage, "")
    if cfg.mode == "colocated" and owner != "kv-link":
        owner = "colocated"
    pool = {k: v for k, v in util.items() if k.startswith(owner)}
    hottest_resource = max(pool, key=pool.get) if pool else max(util, key=util.get)

    energy = energy_report(res, out_tokens, len(met))

    out = {"mode": cfg.mode}
    if cfg.heterogeneous:
        out["pools"] = pools(cfg)
    out.update({
        "requests": {"completed": len(done), "rejected": len(res.rejected),
                     "measured": len(steady)},
        "latency_s": {"ttft": _dist(ttft), "tpot": _dist(tpot), "itl": _dist(itl),
                      "e2e": _dist(e2e)},
        "throughput": {
            "output_tok_per_s": out_tokens / res.horizon,
            "req_per_s": lam,
            "goodput_req_per_s": len(met) / window if window and window > 0 else math.nan,
            "slo_attainment": len(met) / len(steady) if steady else math.nan,
        },
        "utilisation": util,
        "kv_link": {"bytes_GB": res.link.bytes / 1e9, "transfers": res.link.transfers,
                    "mean_wait_ms": 1e3 * res.link.wait / max(1, res.link.transfers),
                    "achieved_GBps": res.link.bytes / res.horizon / 1e9},
        "efficiency": {i.name: {
            "mfu": i.flops / (res.horizon * i.cost.device.peak_flops * i.cost.n_devices),
            "mbu": i.bytes / (res.horizon * i.cost.device.mem_bw * i.cost.n_devices),
            "mean_batch": i.batch_sum / i.steps if i.steps else 0.0,
            "steps": i.steps} for i in res.instances},
        "compute_bound_fraction": {i.name: (i.compute_bound_time / i.busy if i.busy else 0.0)
                                   for i in res.instances},
        "stage_breakdown_s": stage_means,
        "stage_share": {s: v / mean_e2e for s, v in stage_means.items()} if stage_means else {},
        "hotspots": {"stage": hottest_stage, "resource": hottest_resource,
                     "resource_util": util[hottest_resource],
                     "busy_over_90pct": [k for k, v in util.items() if v > 0.9]},
        "littles_law": {"L_measured": little_L, "lambda_W": lam * w},
        "energy": energy,
        "sim_time_s": res.horizon,
    })
    if any(i.cost.device.transform is not None for i in res.instances):
        out["optical"] = {i.name: {
            "optical_bound_frac": i.optical_bound_time / i.busy if i.busy else 0.0,
            "optical_flops": i.optical_flops, "conversion_J": i.optical_j,
            "static_J": i.cost.optical_static_w * res.horizon} for i in res.instances}
    tr = cfg.kv_transit
    if tr is not None:
        lk = res.link
        out["kv_link"]["transit"] = {
            "preset": tr.compression.name, "where": tr.where, "ratio": tr.compression.ratio,
            "handoff_GB": lk.handoff_bytes / 1e9,
            "transit_bound_frac": lk.transit_bound / lk.transfers if lk.transfers else 0.0,
            "transit_J": lk.transit_j}
    return out


def pools(cfg) -> dict:
    """Which device, and how many per instance, each pool runs on."""
    out = {}
    for role in ("prefill", "decode"):
        dev, n = cfg.pool(role)
        out[role] = f"{n}x {dev.name}"
    return out


def energy_report(res: SimResult, out_tokens: int, slo_met: int) -> dict:
    """Energy = static power x wall time + dynamic energy of the work done + link energy.

    Static power is charged for the *whole* run, busy or not: an idle accelerator
    still burns it, which is why utilisation matters for energy as much as for cost.
    """
    H = res.horizon
    per = {}
    static = compute = memory = 0.0
    o_static = o_conv = 0.0                  # transform engine: lasers + tuning, conversions
    optical = any(i.cost.device.transform is not None for i in res.instances)
    for i in res.instances:
        cm = i.cost
        s_j = cm.idle_w * H
        c_j, m_j = i.compute_j, i.memory_j
        static, compute, memory = static + s_j, compute + c_j, memory + m_j
        if optical:
            os_j = cm.optical_static_w * H
            o_static, o_conv = o_static + os_j, o_conv + i.optical_j
            avg = (s_j + c_j + m_j + os_j + i.optical_j) / H
        else:
            avg = (s_j + c_j + m_j) / H
        per[i.name] = {"avg_w": avg, "peak_step_w": i.peak_power,
                       "power_bound_frac": i.power_bound_time / i.busy if i.busy else 0.0}
    link = res.link.energy
    total = static + compute + memory + link
    transit = res.cfg.kv_transit is not None
    if optical:
        total = total + o_static + o_conv
    if transit:
        total = total + res.link.transit_j
    breakdown = {"static": static / total, "compute": compute / total,
                 "memory": memory / total, "link": link / total}
    if optical:
        breakdown["optical_static"] = o_static / total
        breakdown["optical_conversions"] = o_conv / total
    if transit:
        breakdown["transit"] = res.link.transit_j / total
    return {
        "total_J": total,
        "avg_power_W": total / H,
        "J_per_output_token": total / out_tokens if out_tokens else math.nan,
        "output_tokens_per_J": out_tokens / total if total else math.nan,
        "J_per_slo_met_request": total / slo_met if slo_met else math.inf,
        "breakdown": breakdown,
        "per_instance": per,
    }


def format_report(m: dict) -> str:
    ms = lambda x: f"{1e3 * x:8.1f}"
    lines = [f"── {m['mode']} ── {m['requests']['completed']} done, "
             f"{m['requests']['rejected']} rejected, sim {m['sim_time_s']:.1f}s"]
    if "pools" in m:
        lines.append(f"pools        prefill {m['pools']['prefill']}   decode {m['pools']['decode']}")
    lines.append("latency (ms)       mean      p50      p90      p99")
    for k in ("ttft", "tpot", "itl", "e2e"):
        d = m["latency_s"][k]
        lines.append(f"  {k:<10} {ms(d['mean'])} {ms(d['p50'])} {ms(d['p90'])} {ms(d['p99'])}")
    t = m["throughput"]
    lines.append(f"throughput   {t['output_tok_per_s']:9.0f} tok/s   {t['req_per_s']:.2f} req/s   "
                 f"goodput {t['goodput_req_per_s']:.2f} req/s   SLO met {100 * t['slo_attainment']:.1f}%")
    lines.append("utilisation  " + "  ".join(f"{k} {100 * v:.0f}%" for k, v in m["utilisation"].items()))
    lines.append("efficiency   " + "  ".join(
        f"{k} MFU {100 * v['mfu']:.0f}% MBU {100 * v['mbu']:.0f}% batch {v['mean_batch']:.1f}"
        for k, v in m["efficiency"].items()))
    lines.append("where time goes  " + "  ".join(
        f"{k} {100 * v:.0f}%" for k, v in m["stage_share"].items() if v >= 0.005))
    e = m["energy"]
    lines.append(f"power        avg {e['avg_power_W']:,.0f} W   {e['J_per_output_token']:.2f} J/token   "
                 f"{e['output_tokens_per_J']:.2f} tok/J   energy: " + "  ".join(
                     f"{k} {100 * v:.0f}%" for k, v in e["breakdown"].items()))
    if "optical" in m:
        lines.append("optical      " + "  ".join(
            f"{k} optical-bound {100 * v['optical_bound_frac']:.0f}% of busy time, conversions {v['conversion_J']:,.0f} J, "
            f"lasers+tuning {v['static_J']:,.0f} J" for k, v in m["optical"].items() if v["static_J"]))
    if "transit" in m["kv_link"]:
        t = m["kv_link"]["transit"]
        lines.append(f"kv hand-off  {t['preset']} at {t['where']} (ratio {t['ratio']:.2f}): "
                     f"{t['handoff_GB']:.1f} GB -> {m['kv_link']['bytes_GB']:.1f} GB on the link, "
                     f"transit-bound {100 * t['transit_bound_frac']:.0f}% of transfers, stage {t['transit_J']:,.1f} J")
    capped = {k: v["power_bound_frac"] for k, v in e["per_instance"].items() if v["power_bound_frac"] > 0}
    if capped:
        lines.append("power-capped " + "  ".join(f"{k} {100 * v:.0f}% of busy time" for k, v in capped.items()))
    h = m["hotspots"]
    lines.append(f"hot-spot     stage={h['stage']} -> {h['resource']} "
                 f"(busy {100 * h['resource_util']:.0f}%)"
                 + (f"   busy>90%: {', '.join(h['busy_over_90pct'])}" if h["busy_over_90pct"] else ""))
    return "\n".join(lines)
