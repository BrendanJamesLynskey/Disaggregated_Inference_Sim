"""Regenerate examples/results.md: every number quoted in the README and the decks.

    python examples/results.py                  # deterministic runs + timings (run on an idle machine)
    python examples/results.py --no-timings     # deterministic runs only
    python examples/results.py --keep-timings   # reuse section 9 from the current results.md

The cost model was corrected on 2026-10-03 (per-step weight traffic now reads only
the embedding rows looked up, not the whole table; decode attention includes the
new token's attention to itself). Section 1 and the before/after columns rerun the
same experiments on the last commit before the correction (``LEGACY_REF``), unpacked
with ``git archive`` into a temporary directory, so both sides come from real runs.

Timings are wall-clock on whatever machine runs this; the header records which one.

Sections 10-15 (added 2026-10-04, FOptInf deck 03) run only on the current code: heterogeneous
pools, FFT-mixing models on an optical transform engine, the KV hand-off links, and compressing
the hand-off in transit or at the GPU. Their coefficients are illustrative (see each section).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "results.md"
LEGACY_REF = "f2d8207"          # last commit with the uncorrected cost model

# The power-and-energy table: (label, CLI flags). Every row is `disagg-sim` with these flags.
POWER_ROWS = [
    ("Colocated, 2 instances", ["--mode", "colocated"]),
    ("Disaggregated 1P1D", []),
    ("... --dvfs", ["--dvfs"]),
    ("... --dvfs --decode-power-cap 250", ["--dvfs", "--decode-power-cap", "250"]),
    ("... --dvfs --decode-power-cap 200", ["--dvfs", "--decode-power-cap", "200"]),
    ("1P1D --power-cap 400 --dvfs", ["--power-cap", "400", "--dvfs"]),
    ("1P1D --power-cap 300 --dvfs", ["--power-cap", "300", "--dvfs"]),
    ("2P1D --power-cap 400 --dvfs", ["--prefill", "2", "--power-cap", "400", "--dvfs"]),
]
SWEEP_RATES = [2, 3, 4, 5, 6, 8]
SWEEP_2P1D = [2, 4, 6, 8, 10]
PROPORTIONALITY_RATES = [0.5, 1, 2, 4, 6]


# ───────────────────────────────────────────── deterministic collection ──
def collect() -> dict:
    """Every deterministic number, from whichever ``disagg_sim`` is importable."""
    from disagg_sim.cli import build_parser, config_from_args
    from disagg_sim.hardware import H100_SXM, LLAMA3_8B, LLAMA3_70B, CostModel
    from disagg_sim.metrics import format_report, summarise
    from disagg_sim.search import Workload, analytic_capacity
    from disagg_sim.sim import simulate
    from disagg_sim.workload import LengthDist, poisson_workload

    def run(flags, rate=None):
        a = build_parser().parse_args(flags + (["--rate", str(rate)] if rate is not None else []))
        cfg = config_from_args(a)
        wl = poisson_workload(a.rate, a.requests, LengthDist(a.prompt, a.prompt_cv),
                              LengthDist(a.output, a.output_cv), seed=a.seed)
        res = simulate(cfg, wl)
        return summarise(res), res

    out: dict = {}

    # 1. closed-form steps
    steps = {}
    cm8, cm70 = CostModel(LLAMA3_8B, H100_SXM), CostModel(LLAMA3_70B, H100_SXM, 4)
    for key, cm, kind, arg in [("8B decode b=1 ctx 2048, 1xH100", cm8, "decode", [2048]),
                               ("8B decode b=64 ctx 2048, 1xH100", cm8, "decode", [2048] * 64),
                               ("70B decode b=16 ctx 2300, 4xH100", cm70, "decode", [2300] * 16),
                               ("70B decode b=1 ctx 2048, 4xH100", cm70, "decode", [2048]),
                               ("70B prefill 2048, 4xH100", cm70, "prefill", [2048]),
                               ("8B prefill 2048, 1xH100", cm8, "prefill", [2048])]:
        s = getattr(cm, kind)(arg)
        steps[key] = {"flops": s.flops, "bytes": s.bytes, "time": s.time, "bound": s.bound,
                      "intensity": s.flops / s.bytes, "ridge": cm.device.ridge_point}
    out["steps"] = steps
    floor = {}
    for name, m, n in [("Llama-3-8B", LLAMA3_8B, 1), ("Llama-3-70B", LLAMA3_70B, 4)]:
        cm = CostModel(m, H100_SXM, n)
        read = m.weight_bytes_read(1) if hasattr(m, "weight_bytes_read") else m.weight_bytes_total
        floor[name] = {"resident": m.weight_bytes_total, "read_b1": read, "devices": n,
                       "floor_ms": 1e3 * read / cm.byte_rate}
    out["weights"] = floor

    # 2. per-step power (deck 07 slide 04): power per device includes static power
    power_steps = {}
    for dvfs in (False, True):
        cm = CostModel(LLAMA3_70B, H100_SXM, 4, dvfs=dvfs)
        for key, s in [("prefill 2048", cm.prefill([2048])), ("decode b=16 ctx 2300", cm.decode([2300] * 16))]:
            power_steps[f"{key}{' dvfs' if dvfs else ''}"] = {
                "bound": s.bound, "time": s.time, "energy": s.energy,
                "w_per_device": (cm.idle_w + s.energy / s.time) / cm.n_devices}
    out["power_steps"] = power_steps

    def row(m):
        e = m["energy"]
        return {"ttft_p99": m["latency_s"]["ttft"]["p99"], "tpot_p99": m["latency_s"]["tpot"]["p99"],
                "itl_p50": m["latency_s"]["itl"]["p50"], "itl_p99": m["latency_s"]["itl"]["p99"],
                "itl_max": m["latency_s"]["itl"]["max"],
                "slo": m["throughput"]["slo_attainment"], "goodput": m["throughput"]["goodput_req_per_s"],
                "avg_w": e["avg_power_W"], "j_tok": e["J_per_output_token"],
                "static": e["breakdown"]["static"],
                "power_bound": {k: v["power_bound_frac"] for k, v in e["per_instance"].items()},
                "hotspot": f"{m['hotspots']['stage']} -> {m['hotspots']['resource']}",
                "efficiency": m["efficiency"], "utilisation": m["utilisation"],
                "compute_bound": m["compute_bound_fraction"]}

    # 3. power and energy table
    out["power_table"] = {label: row(run(flags)[0]) for label, flags in POWER_ROWS}

    # 4. load sweeps
    out["sweep"] = {mode: {r: row(run(["--mode", mode], r)[0]) for r in SWEEP_RATES}
                    for mode in ("colocated", "disagg")}
    out["sweep_2p1d"] = {r: row(run(["--prefill", "2"], r)[0]) for r in SWEEP_2P1D}

    # 6. energy proportionality, and the static-heavy optical part
    out["proportionality"] = {r: row(run([], r)[0]) for r in PROPORTIONALITY_RATES}
    out["optical"] = row(run(["--device", "optical"])[0])

    # 7. the example report
    m, _ = run(["--prefill", "2", "--rate", "6", "--link", "eth-25g"])
    out["report_link"] = format_report(m)
    out["report_link_row"] = row(m)

    # 8. analytic capacity bounds for the defaults
    a = build_parser().parse_args([])
    cfg = config_from_args(a)
    cap = analytic_capacity(cfg, Workload(LengthDist(2048, 0.5), LengthDist(256, 0.5), n=800))
    out["capacity"] = {k: v for k, v in cap.items()}
    return out


def collect_legacy(ref: str) -> dict:
    """Run ``collect`` against the cost model as it was at ``ref``."""
    with tempfile.TemporaryDirectory() as tmp:
        arch = subprocess.run(["git", "-C", str(ROOT), "archive", ref, "src"], check=True,
                              capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=arch, check=True)
        env = {**os.environ, "PYTHONPATH": str(Path(tmp) / "src")}
        code = ("import json, sys; sys.path.insert(0, %r); import results; "
                "import disagg_sim; assert disagg_sim.__file__.startswith(%r); "
                "print(json.dumps(results.collect()))") % (str(Path(__file__).parent), tmp)
        out = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
        return json.loads(out.stdout)


# ─────────────────────────────────────────────────────────── rendering ──
def ms(x):
    return f"{1e3 * x:,.1f} ms" if x < 1 else f"{1e3 * x:,.0f} ms"


def pct(x):
    return f"{100 * x:.1f}%"


def table(lines, head, rows):
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    lines.append("")


def render(new: dict, old: dict, timings: str | None) -> str:
    L: list[str] = []
    p = L.append
    p("# Recorded results")
    p("")
    p(f"Generated by `examples/results.py` on {date.today().isoformat()}: {cpu_name()},"
      f" Python {platform.python_version()}, {os.cpu_count()} CPUs. Every number in the README and in the LLM Inference")
    p("Simulators decks comes from this file. *Before* columns rerun the same experiment on commit"
      f" `{LEGACY_REF}`, the last one with the uncorrected cost model.")
    p("")

    # 1
    p("## 1. Cost-model correction (2026-10-03): closed-form steps")
    p("")
    p("An operator trace of the real Llama-3-8B (Torch_Sim_Frontend) found two errors in the closed form:")
    p("")
    p("* **Weight traffic.** Every step was charged `weight_bytes_total`, which includes the input embedding table;"
      " a lookup reads only one row per token. Each step now reads the layers and the LM head in full plus the rows"
      " looked up. Residency (KV capacity) still counts both tables.")
    p("* **Decode attention.** The new token attends to its context *and to itself*: `ctx + batch` positions, not `ctx`.")
    p("* Prefill is unchanged in FLOPs: its LM head is charged for every prompt token, as the traced forward pass"
      " computes (`logits` of shape (T, vocab)); engines that keep only the last position do less.")
    p("")
    rows = []
    for k in new["steps"]:
        a, b = old["steps"][k], new["steps"][k]
        rows.append([k, f"{a['bytes'] / 1e9:.3f}", f"{b['bytes'] / 1e9:.3f}", f"{(b['bytes'] - a['bytes']) / a['bytes']:+.2%}",
                     f"{int(b['flops'] - a['flops']):,}", f"{1e3 * a['time']:.3f}", f"{1e3 * b['time']:.3f}",
                     f"{b['intensity']:.1f} / {b['ridge']:.0f}", b["bound"]])
    table(L, ["Step", "GB before", "GB after", "Bytes change", "FLOPs added", "ms before", "ms after",
              "Intensity / ridge (FLOP/B)", "Bound after"], rows)
    k = "8B decode b=1 ctx 2048, 1xH100"
    p(f"* For Llama-3-8B at batch 1 the step reads {(old['steps'][k]['bytes'] - new['steps'][k]['bytes']) / 1e9:.2f} GB"
      f" less ({(old['steps'][k]['bytes'] - new['steps'][k]['bytes']) / old['steps'][k]['bytes']:.1%} of the old figure)"
      f" and adds {int(new['steps'][k]['flops'] - old['steps'][k]['flops']):,} FLOPs.")
    p("* Every decode step stays memory-bound: arithmetic intensity stays far below the ridge point.")
    p("")
    p("Weights resident against weights read by a batch-1 step, and the step-time floor that weight read sets"
      " (H100 at 80% of peak bandwidth):")
    p("")
    rows = []
    for k, b in new["weights"].items():
        a = old["weights"][k]
        rows.append([f"{k}, {b['devices']}xH100", f"{b['resident'] / 1e9:.2f}", f"{a['read_b1'] / 1e9:.2f}",
                     f"{b['read_b1'] / 1e9:.2f}", f"{a['floor_ms']:.2f}", f"{b['floor_ms']:.2f}"])
    table(L, ["Model", "Resident GB", "Read per step GB before", "after", "Floor ms before", "after"], rows)

    # 2
    p("## 2. Per-step power, Llama-3-70B on 4xH100")
    p("")
    p("Power per device = (static power + dynamic energy / step time) / devices; step time includes the 0.5 ms overhead.")
    p("")
    rows = []
    for k, b in new["power_steps"].items():
        a = old["power_steps"][k]
        rows.append([k, b["bound"], f"{1e3 * a['time']:.1f} ms", f"{1e3 * b['time']:.1f} ms",
                     f"{a['w_per_device']:.0f} W", f"{b['w_per_device']:.0f} W", f"{a['energy']:.1f} J", f"{b['energy']:.1f} J"])
    table(L, ["Step", "Bound", "Time before", "Time after", "W/device before", "W/device after", "Dynamic J before",
              "Dynamic J after"], rows)
    d0, d1 = new["power_steps"]["decode b=16 ctx 2300"], new["power_steps"]["decode b=16 ctx 2300 dvfs"]
    p(f"* DVFS saves {1 - d1['energy'] / d0['energy']:.1%} of the decode step's dynamic energy at the same step time.")
    p("")

    # 3
    p("## 3. Power and energy (Llama-3-70B, 4xH100 per instance, 4 req/s, 800 requests)")
    p("")
    rows = []
    for label, b in new["power_table"].items():
        a = old["power_table"][label]
        pb = max(b["power_bound"].values())
        rows.append([label, ms(b["ttft_p99"]), ms(b["tpot_p99"]), pct(b["slo"]), f"{b['avg_w']:,.0f} W",
                     f"{b['j_tok']:.2f}", f"{pb:.0%}", b["hotspot"]])
    table(L, ["Configuration", "TTFT p99", "TPOT p99", "SLO met", "Avg power", "J / token", "Max power-bound share",
              "Hot-spot"], rows)
    p("Before and after the correction:")
    p("")
    rows = []
    for label, b in new["power_table"].items():
        a = old["power_table"][label]
        rows.append([label, ms(a["tpot_p99"]), ms(b["tpot_p99"]), pct(a["slo"]), pct(b["slo"]),
                     f"{a['j_tok']:.2f}", f"{b['j_tok']:.2f}", f"{a['avg_w']:,.0f} W", f"{b['avg_w']:,.0f} W"])
    table(L, ["Configuration", "TPOT p99 before", "TPOT p99 after", "SLO before", "SLO after", "J/token before",
              "J/token after", "Avg power before", "Avg power after"], rows)
    rank = lambda d: [k for k, _ in sorted(d.items(), key=lambda kv: kv[1]["j_tok"])]
    same = rank(new["power_table"]) == rank(old["power_table"])
    p(f"* J/token ranking of the eight configurations unchanged: {same}.")
    pb_old = {k: max(v["power_bound"].values()) > 0 for k, v in old["power_table"].items()}
    pb_new = {k: max(v["power_bound"].values()) > 0 for k, v in new["power_table"].items()}
    pool = lambda h: h.rsplit("-", 1)[0]          # "prefill -> colocated-1" -> "prefill -> colocated"
    p(f"* Power-bound flags unchanged: {pb_old == pb_new}. Hot-spot stage and resource pool unchanged:"
      f" {all(pool(old['power_table'][k]['hotspot']) == pool(v['hotspot']) for k, v in new['power_table'].items())}.")
    flips = [f"{k} ({old['power_table'][k]['hotspot']} → {v['hotspot']})" for k, v in new["power_table"].items()
             if old["power_table"][k]["hotspot"] != v["hotspot"]]
    if flips:
        p(f"* Which of two identical instances is busiest changed in: {'; '.join(flips)}.")
    p("")

    # 4
    p("## 4. Load sweep (`disagg-sim --compare --sweep-rate 2 3 4 5 6 8`)")
    p("")
    rows = []
    for r in SWEEP_RATES:
        c, d = get(new["sweep"]["colocated"], r), get(new["sweep"]["disagg"], r)
        rows.append([r, ms(c["tpot_p99"]), pct(c["slo"]), ms(d["tpot_p99"]), ms(d["ttft_p99"]), pct(d["slo"])])
    table(L, ["Offered load (req/s)", "Colocated TPOT p99", "Colocated SLO met", "Disagg 1P1D TPOT p99",
              "Disagg 1P1D TTFT p99", "Disagg 1P1D SLO met"], rows)
    rows = []
    for r in SWEEP_RATES:
        a, b = get(old["sweep"]["disagg"], r), get(new["sweep"]["disagg"], r)
        ac, bc = get(old["sweep"]["colocated"], r), get(new["sweep"]["colocated"], r)
        rows.append([r, pct(ac["slo"]), pct(bc["slo"]), pct(a["slo"]), pct(b["slo"]), ms(a["tpot_p99"]), ms(b["tpot_p99"])])
    table(L, ["req/s", "Colocated SLO before", "after", "1P1D SLO before", "after", "1P1D TPOT p99 before", "after"], rows)
    two = [get(new["sweep_2p1d"], r) for r in SWEEP_2P1D]
    p(f"* 2P1D, rates {', '.join(map(str, SWEEP_2P1D))} req/s: SLO met {', '.join(pct(x['slo']) for x in two)};"
      f" TPOT p99 {', '.join(ms(x['tpot_p99']) for x in two)}.")
    d4 = get(new["sweep"]["disagg"], 4)
    e = d4["efficiency"]["decode-0"]
    p(f"* At 4 req/s (1P1D) the decode instance runs at a mean batch of {e['mean_batch']:.1f} with MBU {e['mbu']:.0%}"
      f" and MFU {e['mfu']:.0%}; it is compute-bound for {d4['compute_bound']['decode-0']:.1%} of its busy time.")
    c4 = get(new["sweep"]["colocated"], 4)
    p(f"* Inter-token latency at 4 req/s: colocated p50 {ms(c4['itl_p50'])}, p99 {ms(c4['itl_p99'])}, max"
      f" {ms(c4['itl_max'])}; disaggregated p50 {ms(d4['itl_p50'])}, p99 {ms(d4['itl_p99'])}, max {ms(d4['itl_max'])}."
      f" Disaggregation cuts p99 ITL {c4['itl_p99'] / d4['itl_p99']:.1f}x.")
    p("")

    # 5
    p("## 5. Utilisation and efficiency (defaults, 4 req/s)")
    p("")
    rows = []
    for name, u in d4["utilisation"].items():
        ef = d4["efficiency"].get(name)
        rows.append([name, f"{u:.0%}"] + ([f"{ef['mfu']:.0%}", f"{ef['mbu']:.0%}", f"{ef['mean_batch']:.1f}"]
                                          if ef else ["—", "—", "—"]))
    table(L, ["Resource", "Busy", "MFU", "MBU", "Mean batch"], rows)

    # 6
    p("## 6. Energy proportionality (1P1D)")
    p("")
    rows = []
    for r in PROPORTIONALITY_RATES:
        a, b = get(old["proportionality"], r), get(new["proportionality"], r)
        rows.append([f"{r:g} req/s", f"{b['avg_w']:,.0f} W", f"{b['j_tok']:.1f}", f"{b['static']:.0%}",
                     f"{a['j_tok']:.2f} → {b['j_tok']:.2f}"])
    table(L, ["Offered load", "Avg power", "J / output token", "Static share of energy", "J/token before → after"], rows)
    o, oo = new["optical"], old["optical"]
    p(f"* Hypothetical optical part, 4 req/s: static share {o['static']:.0%}, {o['j_tok']:.2f} J/token"
      f" (before: {oo['static']:.0%}, {oo['j_tok']:.2f}); TTFT p99 {ms(o['ttft_p99'])}, TPOT p99 {ms(o['tpot_p99'])}.")
    cap = new["power_table"]["1P1D --power-cap 400 --dvfs"]["j_tok"]
    p(f"* For comparison, the GPU under a 400 W cap with DVFS: {cap:.2f} J/token.")
    p("")

    # 7
    p("## 7. Example report (`disagg-sim --prefill 2 --rate 6 --link eth-25g`)")
    p("")
    p("```")
    p(new["report_link"])
    p("```")
    p("")

    # 8
    c, co = new["capacity"], old["capacity"]
    p("## 8. Analytic capacity bounds (defaults)")
    p("")
    p(f"* Prefill {c['prefill']:.2f} req/s, decode {c['decode']:.1f}, KV link {c['kv-link']:.1f}: bottleneck"
      f" {c['bottleneck']} (before: prefill {co['prefill']:.2f}, decode {co['decode']:.1f}, link {co['kv-link']:.1f}).")
    p("")

    if timings:
        p("## 9. Acceleration (`examples/benchmark_acceleration.py`, wall clock)")
        p("")
        p("```")
        p(timings.rstrip())
        p("```")
        p("")
    return "\n".join(L) + "\n"


# ───────────────────────────────────── sections 10-15: FOptInf phase B ──
OPT_RATE, OPT_N = 8.0, 800          # Llama-3-8B shape, 1 device per instance, 1P1D unless stated
R16 = 1 / 16                        # illustrative GPU FFT efficiency (FOptInf A§3)
VARIANTS = [("Transformer (Llama-3-8B)", "llama3-8b", "all"), ("Hyena-2", "llama3-8b-hyena", "all"),
            ("Hybrid 1:3", "llama3-8b-hybrid", "all"), ("Hyena-2, distilled decode", "llama3-8b-hyena-dist", "all"),
            ("Hyena-2 + circulant 256", "llama3-8b-hyena-circ", "all"),
            ("Hyena-2 + circulant 256, last-token head", "llama3-8b-hyena-circ", "last")]
LINK_KEYS = ["nvlink4", "ib-ndr", "cpo-optical", "eth-100g", "eth-25g"]
HANDOFFS = [("GQA KV cache", "llama3-8b"), ("Hyena direct cache", "llama3-8b-hyena"),
            ("Hyena distilled state", "llama3-8b-hyena-dist")]
PRESETS = ["none", "fp8", "fp4-block", "freq-keep-k"]
TRANSIT_RATES = [8, 10, 12, 14]


def collect_optical() -> dict:
    from dataclasses import replace

    from disagg_sim.hardware import (A100_SXM, H100_SXM, KV_PRESETS, LINKS, MODELS, OPTICAL_FFT, CostModel,
                                     KVTransit)
    from disagg_sim.metrics import percentile, summarise
    from disagg_sim.ppa import device_area, ppa_report
    from disagg_sim.sim import SimConfig, simulate
    from disagg_sim.workload import LengthDist, poisson_workload

    def model(key, head="all", **kw):
        return replace(MODELS[key], prefill_lm_head=head, **kw)

    def gpu(r=1.0, dev=H100_SXM):
        return dev if r == 1.0 else replace(dev, fft_efficiency=r)

    def engine(enob=8, mask=1031.0, overlap=False, static=20.0, detection="coherent"):
        return replace(OPTICAL_FFT, transform=replace(OPTICAL_FFT.transform, enob=enob, mask_rate_hz=mask,
                                                      overlap=overlap, laser_w=static / 2, tuning_w=static / 2,
                                                      detection=detection))

    DEFAULT, OPTIMISTIC = engine(), engine(enob=11, mask=20000.0, overlap=True)

    def go(cfg, rate=OPT_RATE):
        """One full run; the hand-off latency is first token -> KV on the decode pool (DistServe's
        TPOT includes it, TTFT does not: the first token is emitted at the end of prefill)."""
        wl = poisson_workload(rate, OPT_N, LengthDist(2048, 0.5), LengthDist(256, 0.5), seed=1)
        res = simulate(replace(cfg, fast_forward=True), wl)
        m = summarise(res)
        e, lat, ss = m["energy"], m["latency_s"], m["stage_share"]
        p = ppa_report(cfg, m)
        opt = m.get("optical", {}).get("prefill-0", {})
        lk = m["kv_link"]
        return {"ttft_p99": lat["ttft"]["p99"], "tpot_p99": lat["tpot"]["p99"], "slo": m["throughput"]["slo_attainment"],
                "goodput": m["throughput"]["goodput_req_per_s"], "tok_s": m["throughput"]["output_tok_per_s"],
                "j_tok": e["J_per_output_token"], "avg_w": e["avg_power_W"], "tok_j": e["output_tokens_per_J"],
                "breakdown": e["breakdown"], "kv_share": ss.get("kv_wait", 0.0) + ss.get("kv_transfer", 0.0),
                "link_util": m["utilisation"].get("kv-link", 0.0), "link_GB": lk["bytes_GB"],
                "link_j_req": (res.link.energy + res.link.transit_j) / max(1, res.link.transfers),
                "transit_bound": lk.get("transit", {}).get("transit_bound_frac", 0.0),
                "optical_bound": opt.get("optical_bound_frac", 0.0),
                "usd": p["total_usd"], "tok_s_kusd": p["perf_per_kusd"], "tok_s_mm2": p["perf_per_mm2"],
                "handoff_p99": percentile([r.kv_ready - r.first_token for r in res.requests
                                           if r.kv_ready is not None], 99) if cfg.mode == "disagg" else 0.0,
                "stamps": [(r.first_token, r.finish) for r in res.requests]}

    out: dict = {}
    L8 = model("llama3-8b")
    base = SimConfig(model=L8, devices_per_instance=1)

    # 10. heterogeneous pools
    homo = go(base)
    same = go(replace(base, prefill_device=H100_SXM, decode_device=H100_SXM, prefill_devices_per_instance=1,
                      decode_devices_per_instance=1))
    rows = {"H100 prefill + H100 decode (--device h100)": homo,
            "H100 prefill + A100 decode": go(replace(base, prefill_device=H100_SXM, decode_device=A100_SXM)),
            "A100 prefill + H100 decode": go(replace(base, prefill_device=A100_SXM, decode_device=H100_SXM)),
            "A100 prefill + A100 decode": go(replace(base, device=A100_SXM)),
            "2x A100 prefill + H100 decode": go(replace(base, prefill_device=A100_SXM, decode_device=H100_SXM,
                                                        n_prefill=2))}
    out["hetero"] = {"rows": rows, "identical": same["stamps"] == homo["stamps"]
                     and {k: v for k, v in same.items() if k != "stamps"} == {k: v for k, v in homo.items() if k != "stamps"}}

    # 11. baselines against an optical prefill pool
    t11 = {}
    for label, key, head in VARIANTS:
        m = model(key, head)
        b = SimConfig(model=m, devices_per_instance=1)
        t11[label] = {"Colocated x2, H100": go(replace(b, mode="colocated", n_colocated=2)),
                      "1P1D H100": go(b),
                      "1P1D H100, GPU FFT at 1/16": go(replace(b, device=gpu(R16))),
                      "optical-fft prefill (defaults) + H100 decode": go(replace(b, prefill_device=DEFAULT)),
                      "optical-fft prefill (optimistic) + H100 decode": go(replace(b, prefill_device=OPTIMISTIC))}
    out["baselines"] = t11

    # 12. sweeps
    def step(m, dev):
        c = CostModel(m, dev).prefill([2048])
        return {"ms": 1e3 * c.time, "bound": c.bound, "dyn_j": c.energy, "opt_j": c.optical_j}

    shares = {}
    for label, key, head, blk in [("Hyena-2", "llama3-8b-hyena", "all", 0), ("Hybrid 1:3", "llama3-8b-hybrid", "all", 0),
                                  ("circulant 256, all-token head", "llama3-8b-hyena-circ", "all", 256),
                                  ("circulant 1024, last-token head", "llama3-8b-hyena-circ", "last", 1024),
                                  ("circulant 256, last-token head", "llama3-8b-hyena-circ", "last", 256),
                                  ("circulant 64, last-token head", "llama3-8b-hyena-circ", "last", 64)]:
        m = model(key, head, **({"circulant_block": blk} if blk else {}))
        o = m.prefill_ops([2048])
        shares[label] = {"share": o.optical / o.total, "gflop": o.total / 1e9, "gpu": step(m, gpu()),
                         "gpu16": step(m, gpu(R16)), "default": step(m, DEFAULT), "optimistic": step(m, OPTIMISTIC),
                         "mask_values": m.mask_values([2048]), "pairs": m.conversion_pairs_per_token}
    CIRC = model("llama3-8b-hyena-circ", "last")
    enob = {}
    for e in range(6, 15):
        dev = engine(enob=e, mask=20000.0, overlap=True)
        k, conv, rew, t_opt = CostModel(CIRC, dev).optical_terms([2048], 2048)
        enob[e] = {"passes": k, "conversions": conv, "t_opt_ms": 1e3 * t_opt, **step(CIRC, dev),
                   "pj_pair": dev.transform.pj_per_pair}
    mask = {}
    for hz in (30.0, 1031.0, 20000.0, 1e5, 1e6):
        dev = engine(enob=11, mask=hz, overlap=True)
        k, conv, rew, t_opt = CostModel(CIRC, dev).optical_terms([2048], 2048)
        mask[hz] = {"rewrites": rew, "t_mask_ms": 1e3 * rew / hz, **step(CIRC, dev),
                    "run": go(SimConfig(model=CIRC, devices_per_instance=1, prefill_device=dev))}
    cb = SimConfig(model=CIRC, devices_per_instance=1)
    static = {w: go(replace(cb, prefill_device=engine(enob=11, mask=20000.0, overlap=True, static=w)))
              for w in (0.0, 10.0, 20.0, 50.0, 100.0, 200.0)}
    out["sweeps"] = {"shares": shares, "enob": enob, "mask": mask, "static": static,
                     "gpu_ref": {"r1": go(cb), "r16": go(replace(cb, device=gpu(R16)))}}

    # 13. break-even, by bisection on the simulator (as search.py does for load)
    def bisect(f, lo, hi, iters=40, log=False):
        """Largest x in [lo, hi] with f(x) true, f monotone (true then false)."""
        import math as _m
        for _ in range(iters):
            mid = _m.sqrt(lo * hi) if log else (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return lo

    be = {}
    for rlabel, r in [("GPU FFT at 1x", 1.0), ("GPU FFT at 1/4", 0.25), ("GPU FFT at 1/16", R16)]:
        ref = go(replace(cb, device=gpu(r)))
        opt = lambda **kw: go(replace(cb, device=gpu(r), prefill_device=engine(**{"enob": 11, "mask": 20000.0,
                                                                                  "overlap": True, **kw})))
        row = {"gpu": ref, "optical": opt()}
        cheaper = lambda w: opt(static=w)["j_tok"] <= ref["j_tok"]
        row["static_w"] = (None if not cheaper(0.0) else (">1000" if cheaper(1000.0) else bisect(cheaper, 0.0, 1000.0, 20)))
        faster = lambda e: opt(enob=e)["ttft_p99"] <= ref["ttft_p99"]
        row["enob"] = next((e for e in range(6, 17) if faster(e)), None)
        fmask = lambda hz: opt(mask=hz)["ttft_p99"] <= ref["ttft_p99"]
        row["mask_hz"] = (None if not fmask(1e9) else bisect(lambda hz: not fmask(hz), 1.0, 1e9, 30, log=True))
        be[rlabel] = row
    # optical wins at low GPU FFT efficiency r and loses at high r: the largest r where it still wins
    fr = lambda r: go(replace(cb, device=gpu(r), prefill_device=OPTIMISTIC))["ttft_p99"] <= go(replace(cb, device=gpu(r)))["ttft_p99"]
    out["breakeven"] = {"rows": be, "gpu_fft_eff": (None if not fr(1 / 64) else (">=1" if fr(1.0) else bisect(fr, 1 / 64, 1.0, 20, log=True)))}

    # 14. the KV hand-off: link x what is handed off
    out["links"] = {h: {lk: go(SimConfig(model=model(key), devices_per_instance=1, link=LINKS[lk])) for lk in LINK_KEYS}
                    for h, key in HANDOFFS}
    out["link_specs"] = {k: {"name": LINKS[k].name, "GBps": LINKS[k].bandwidth / 1e9, "us": LINKS[k].latency * 1e6,
                             "pj": LINKS[k].pj_per_bit} for k in LINK_KEYS}
    out["handoff_MB"] = {h: MODELS[key].handoff_bytes(2048) / 1e6 for h, key in HANDOFFS}

    # 15. compute in transit vs the same compression at the GPU
    cit = {}
    for h, key in HANDOFFS:
        for lk in LINK_KEYS:
            for pr in PRESETS:
                if pr == "freq-keep-k" and key.endswith("dist"):
                    continue                       # a recurrence state has no token axis to transform
                for where in ("transit", "endpoint"):
                    cfg = SimConfig(model=model(key), devices_per_instance=1, link=LINKS[lk],
                                    kv_transit=KVTransit(KV_PRESETS[pr], where=where))
                    cit[f"{h}|{lk}|{pr}|{where}"] = go(cfg)
    out["transit"] = cit
    # load sweep where the GQA KV hand-off saturates 25 GbE (one transfer of a 2,048-token prompt: 86 ms)
    sweep = {}
    for rate in TRANSIT_RATES:
        for pr in PRESETS:
            for where in ("transit", "endpoint"):
                if pr == "none" and where == "endpoint":
                    continue
                cfg = SimConfig(model=L8, devices_per_instance=1, link=LINKS["eth-25g"],
                                kv_transit=None if pr == "none" else KVTransit(KV_PRESETS[pr], where=where))
                sweep[f"{rate}|{pr}|{where}"] = go(cfg, rate)
    out["transit_sweep"] = sweep
    out["presets"] = {k: {"ratio": v.ratio, "ops": v.ops_per_value, "fft": v.fft} for k, v in KV_PRESETS.items()}
    tr = KVTransit(KV_PRESETS["none"])
    out["transit_stage"] = {"ops_per_byte": tr.ops_per_byte, "pj_per_bit": tr.pj_per_bit, "latency_us": tr.latency * 1e6}
    out["areas"] = {d.name: device_area(d) for d in (H100_SXM, A100_SXM, OPTICAL_FFT)}
    for sec in out.values():                       # drop per-request stamps before rendering
        _strip(sec)
    return out


def _strip(x):
    if isinstance(x, dict):
        x.pop("stamps", None)
        for v in x.values():
            _strip(v)


def render_optical(o: dict) -> str:
    L: list[str] = []
    p = L.append
    ms2 = lambda x: f"{1e3 * x:,.1f} ms" if x < 10 else f"{x:,.1f} s"
    usd = lambda x: "—" if x is None else f"${x:,.0f}"
    kusd = lambda x: "—" if x is None else f"{x:,.0f}"
    pp = lambda x: f"{100 * x:.1f}%"

    def latrow(r):
        return [ms2(r["ttft_p99"]), ms(r["tpot_p99"]), f"{r['goodput']:.2f}", pp(r["slo"]), f"{r['j_tok']:.3f}",
                f"{r['avg_w']:,.0f} W"]

    common = (f"Workload for sections 10-15: Llama-3-8B shape, 1 device per instance, 1 prefill + 1 decode instance unless"
              f" stated, Poisson arrivals at {OPT_RATE:g} req/s, {OPT_N} requests, prompts 2,048 tokens (cv 0.5), outputs 256"
              " (cv 0.5), seed 1, InfiniBand NDR unless stated; SLOs TTFT 1 s and TPOT 25 ms. Exact fast path on.")

    # 10
    p("## 10. Heterogeneous pools (the Splitwise idea, arXiv:2311.18677)")
    p("")
    p(common)
    p("")
    h = o["hetero"]
    rows = [[k] + latrow(r) + [usd(r["usd"]), kusd(r["tok_s_kusd"])] for k, r in h["rows"].items()]
    table(L, ["Pools", "TTFT p99", "TPOT p99", "Goodput (req/s)", "SLO met", "J / token", "Avg power",
              "Silicon $ (illustrative)", "tok/s per $1000"], rows)
    p(f"* Setting `--prefill-device h100 --decode-device h100` explicitly reproduces the `--device h100` run"
      f" bit-identically (every request's timestamps and every summary number): {h['identical']}.")
    p("")
    rows = [[k, f"{v['die_mm2']:,.0f}", f"{v['photonic_mm2']:,.0f}", f"{v['total_mm2']:,.0f}", f"${v['usd']:,.0f}"]
            for k, v in o["areas"].items()]
    table(L, ["Device", "Digital die mm²", "Photonic die mm²", "Total mm²", "Silicon $ per good device"], rows)
    p("* Silicon $ = GPU dies only (GH100 814 mm², GA100 826 mm², NVIDIA's architecture posts), Murphy yield at"
      " D0 = 0.1/cm², $10,000 per wafer: brief 03's illustrative method, not a price.")
    p("")

    # 11
    p("## 11. All-GPU baselines against an optical prefill pool")
    p("")
    p("Optical prefill = `optical-fft` (a Fourier-optical transform engine co-packaged with an H100-class part) in"
      " the prefill pool, H100 decode. *Defaults*: ENOB 8, 8-bit DMD mask at 1,031 Hz, coherent detection, optical and"
      " digital time added, 20 W of lasers and tuning per device. *Optimistic*: ENOB 11 (one pass for BF16), 20 kHz"
      " mask (the 1-bit DMD rate), times overlapped. *GPU FFT at 1/16*: the GPU runs FFT and Fourier-domain work at"
      " 1/16 of its matmul rate (illustrative, FOptInf A§3). All coefficients are illustrative.")
    p("")
    for model_label, cfgs in o["baselines"].items():
        p(f"**{model_label}**")
        p("")
        rows = [[k] + latrow(r) + [pp(r["optical_bound"]), f"{r['tok_j']:.2f}", kusd(r["tok_s_kusd"])]
                for k, r in cfgs.items()]
        table(L, ["Configuration", "TTFT p99", "TPOT p99", "Goodput (req/s)", "SLO met", "J / token", "Avg power",
                  "Prefill optical-bound", "tok/J", "tok/s per $1000"], rows)

    # 12
    s = o["sweeps"]
    p("## 12. Sweeps: transform share, ENOB, mask rate and static power")
    p("")
    p("One 2,048-token prefill step (closed form). Optical share = FFT plus Fourier-plane FLOPs over all FLOPs"
      " (FOptInf A§2).")
    p("")
    rows = [[k, pp(v["share"]), f"{v['gflop']:,.1f}", f"{v['gpu']['ms']:.2f} ms", f"{v['gpu16']['ms']:.2f} ms",
             f"{v['default']['ms']:,.1f} ms", f"{v['optimistic']['ms']:.2f} ms", f"{v['mask_values']:,}"]
            for k, v in s["shares"].items()]
    table(L, ["Variant", "Optical share", "GFLOP", "H100", "H100 FFT at 1/16", "optical-fft defaults",
              "optical-fft optimistic", "Mask values"], rows)
    p("ENOB (circulant 256, last-token head; 20 kHz mask, overlapped). Passes = 4^(11 - ENOB) for BF16; conversion"
      " energy per pair = Walden FoM x 2^ENOB (10 + 20 fJ):")
    p("")
    rows = [[e, f"{v['passes']:,}", f"{v['conversions']:,}", f"{v['pj_pair']:.2f}", f"{v['t_opt_ms']:,.2f} ms",
             f"{v['ms']:,.2f} ms", f"{v['opt_j']:.3f} J"] for e, v in s["enob"].items()]
    table(L, ["ENOB", "Passes", "Conversion pairs", "pJ per pair", "Optical time", "Step time", "Conversion energy"], rows)
    p("Mask rewrite rate (same model, ENOB 11, overlapped). 30 Hz is a liquid-crystal SLM, 1,031 Hz an 8-bit DMD,"
      " 20 kHz a 1-bit DMD (Miscuglio et al., Optica 2020); 100 kHz and 1 MHz are hypothetical:")
    p("")
    rows = [[f"{float(hz):,.0f} Hz", f"{v['rewrites']:,}", f"{v['t_mask_ms']:,.2f} ms", f"{v['ms']:,.2f} ms",
             ms2(v["run"]["ttft_p99"]), pp(v["run"]["slo"])] for hz, v in s["mask"].items()]
    table(L, ["Mask rate", "Rewrites per step", "Mask time", "Step time", "TTFT p99 (full run)", "SLO met"], rows)
    g1, g16 = s["gpu_ref"]["r1"], s["gpu_ref"]["r16"]
    p(f"Static power of the engine (lasers + tuning per device; optimistic engine; full runs). All-GPU 1P1D for"
      f" comparison: {g1['j_tok']:.3f} J/token (GPU FFT at 1x), {g16['j_tok']:.3f} J/token (at 1/16).")
    p("")
    rows = [[f"{float(w):g} W", f"{r['j_tok']:.3f}", pp(r["breakdown"].get("optical_static", 0.0)),
             pp(r["breakdown"].get("optical_conversions", 0.0)), ms2(r["ttft_p99"])] for w, r in s["static"].items()]
    table(L, ["Lasers + tuning", "J / token", "Optical static share", "Conversion share", "TTFT p99"], rows)

    # 13
    b = o["breakeven"]
    p("## 13. Break-even: where an optical prefill pool stops paying")
    p("")
    p("Circulant 256 with the last-token LM head (the most transform-heavy variant), optimistic engine unless the"
      " row varies it. Each break-even point is found by bisection on full simulator runs (the J/token and TTFT p99"
      " of the whole cluster). *Never* = the optical pool is worse at every value tried: static power 0-1,000 W,"
      " ENOB 6-16, mask rate 1 Hz-1 GHz.")
    p("")
    f = lambda x, unit, fmt: "never" if x is None else (x if isinstance(x, str) else f"{x:{fmt}} {unit}")
    rows = []
    for k, r in b["rows"].items():
        rows.append([k, ms2(r["gpu"]["ttft_p99"]), ms2(r["optical"]["ttft_p99"]), f"{r['gpu']['j_tok']:.3f}",
                     f"{r['optical']['j_tok']:.3f}", f(r["static_w"], "W", ".1f"),
                     "never" if r["enob"] is None else f"ENOB {r['enob']}", f(r["mask_hz"], "Hz", ",.0f")])
    table(L, ["GPU baseline", "GPU TTFT p99", "Optical TTFT p99", "GPU J/token", "Optical J/token",
              "Static power at equal J/token", "Lowest ENOB with TTFT p99 <= GPU", "Mask rate at equal TTFT p99"], rows)
    ge = b["gpu_fft_eff"]
    p(f"* GPU FFT efficiency below which the optimistic optical pool wins TTFT p99: "
      + ("none in [1/64, 1]" if ge is None else (f"{ge}" if isinstance(ge, str) else f"{ge:.3f} (1/{1 / ge:.1f})")) + ".")
    p("")

    # 14
    p("## 14. The KV hand-off: which link, and what is handed off")
    p("")
    p("Photonic interconnect is *not* Fourier optics: `cpo-optical` is an illustrative co-packaged-optics link (round"
      " numbers; optical I/O chiplets reach multi-Tb/s, Wade et al., IEEE Micro 2020). Hand-off for a 2,048-token"
      " prompt: " + "; ".join(f"{h} {mb:,.1f} MB" for h, mb in o["handoff_MB"].items()) + ". All-H100 pools.")
    p("")
    rows = [[k, v["name"], f"{v['GBps']:,.1f}", f"{v['us']:g}", f"{v['pj']:g}"] for k, v in o["link_specs"].items()]
    table(L, ["Link", "Name", "GB/s", "Latency (us)", "pJ/bit"], rows)
    for hlabel, links in o["links"].items():
        p(f"**{hlabel}**")
        p("")
        rows = [[lk, pp(r["kv_share"]), pp(r["link_util"]), ms2(r["ttft_p99"]), ms(r["tpot_p99"]), pp(r["slo"]),
                 f"{r['link_j_req']:.3f}", f"{r['j_tok']:.3f}"] for lk, r in links.items()]
        table(L, ["Link", "KV wait + transfer share of E2E", "Link busy", "TTFT p99", "TPOT p99", "SLO met",
                  "Link J / request", "J / token"], rows)

    # 15
    st = o["transit_stage"]
    p("## 15. Compute in the transport: compressing the hand-off in transit or at the GPU")
    p("")
    p(f"An in-transit stage compresses the hand-off as it crosses the link: no GPU time, {st['pj_per_bit']:g} pJ per input"
      f" bit, {st['latency_us']:g} us added, and a compute budget of {st['ops_per_byte']:g} operations per line byte"
      " (illustrative, representative of published 2026 compute-in-transit prototypes; the mapping to KV hand-off is"
      " speculation). Its transforms are passive (an optical FFT costs no budget). If its work does not fit the"
      " budget, its compute time sets the transfer time (*transit-bound*). *Endpoint* = the same compression on the"
      " prefill GPU: an elementwise pass (read the KV, write the compressed copy, plus the FLOPs, FFT included).")
    p("")
    rows = [[k, f"{v['ratio']:.3f}", f"{v['ops']:g}", "yes" if v["fft"] else "no"] for k, v in o["presets"].items()]
    table(L, ["Preset", "Compression ratio", "Ops per value (besides any FFT)", "FFT along tokens"], rows)
    p("Bit widths: 8- and 4-bit KV are within what independent KV-quantisation work reports as tolerable (KIVI,"
      " arXiv:2402.02750, 2-bit; KVQuant, arXiv:2401.18079, 3-bit with < 0.1 perplexity loss); keeping half the"
      " frequency components is FreqKV's default (arXiv:2505.00570, which fine-tunes lightly). The simulator does not"
      " model accuracy, nor the decode side's handling of compressed KV.")
    p("")
    cit = o["transit"]
    p("Hand-off p99 = first token to the KV landing on the decode pool (queueing for the link plus the transfer), the"
      " 99th percentile over all requests. TTFT does not include it (the first token leaves the prefill pool); TPOT does"
      " (DistServe's definition), so TPOT and SLO are where compressing the hand-off can show.")
    p("")
    for hlabel, _ in HANDOFFS:
        p(f"**{hlabel}**")
        p("")
        rows = []
        for lk in LINK_KEYS:
            for pr in PRESETS:
                t, e = cit.get(f"{hlabel}|{lk}|{pr}|transit"), cit.get(f"{hlabel}|{lk}|{pr}|endpoint")
                if t is None:
                    continue
                rows.append([lk, pr, ms2(t["handoff_p99"]), ms2(e["handoff_p99"]), ms(t["tpot_p99"]), ms(e["tpot_p99"]),
                             pp(t["slo"]), pp(e["slo"]), f"{t['j_tok']:.3f}", f"{e['j_tok']:.3f}", pp(t["transit_bound"])])
        table(L, ["Link", "Preset", "Hand-off p99 in transit", "Hand-off p99 at GPU", "TPOT p99 in transit",
                  "TPOT p99 at GPU", "SLO in transit", "SLO at GPU", "J/token in transit", "J/token at GPU",
                  "Transit-bound transfers"], rows)
    sw = o["transit_sweep"]
    p("**Load sweep: GQA KV over 25 GbE** (the link saturates near 11.6 req/s uncompressed: 268.4 MB per 2,048-token"
      " prompt at 3.125 GB/s). Same workload otherwise.")
    p("")
    rows = []
    for rate in TRANSIT_RATES:
        for pr in PRESETS:
            t = sw.get(f"{rate}|{pr}|transit")
            e = sw.get(f"{rate}|{pr}|endpoint", t)
            rows.append([f"{rate} req/s", pr, pp(t["link_util"]), ms2(t["handoff_p99"]), ms2(e["handoff_p99"]),
                         ms(t["tpot_p99"]), ms(e["tpot_p99"]), pp(t["slo"]), pp(e["slo"]), f"{t['j_tok']:.3f}",
                         f"{e['j_tok']:.3f}", pp(t["transit_bound"])])
    table(L, ["Load", "Preset", "Link busy (transit)", "Hand-off p99 in transit", "Hand-off p99 at GPU",
              "TPOT p99 in transit", "TPOT p99 at GPU", "SLO in transit", "SLO at GPU", "J/token in transit",
              "J/token at GPU", "Transit-bound transfers"], rows)
    # headline: the transport-bound regime
    p("Where serving is transport-bound (the uncompressed hand-off is at least 10% of end-to-end time), the most"
      " in-transit compression buys and the same compression at the GPU:")
    p("")
    rows = []
    cases = [(hlabel, lk, OPT_RATE, o["links"][hlabel][lk], lambda pr, w, h=hlabel, l=lk: cit.get(f"{h}|{l}|{pr}|{w}"))
             for hlabel, _ in HANDOFFS for lk in LINK_KEYS]
    cases += [("GQA KV cache", "eth-25g", rate, sw[f"{rate}|none|transit"],
               lambda pr, w, r=rate: sw.get(f"{r}|{pr}|{w}")) for rate in TRANSIT_RATES]
    seen = set()
    for hlabel, lk, rate, none, get in cases:
        if none["kv_share"] < 0.10 or (hlabel, lk, rate) in seen:
            continue
        seen.add((hlabel, lk, rate))
        best = min((x for x in PRESETS if x != "none" and get(x, "transit") is not None),
                   key=lambda x: get(x, "transit")["handoff_p99"])
        t, e = get(best, "transit"), get(best, "endpoint")
        rows.append([hlabel, lk, f"{rate:g} req/s", pp(none["kv_share"]), best, ms2(none["handoff_p99"]),
                     ms2(t["handoff_p99"]), ms2(e["handoff_p99"]), f"{none['handoff_p99'] / t['handoff_p99']:.1f}x",
                     ms(none["tpot_p99"]), ms(t["tpot_p99"]), ms(e["tpot_p99"]), pp(none["slo"]), pp(t["slo"]),
                     pp(e["slo"])])
    table(L, ["Hand-off", "Link", "Load", "Hand-off share uncompressed", "Best preset in transit",
              "Hand-off p99 uncompressed", "in transit", "at GPU", "Hand-off gain in transit",
              "TPOT p99 uncompressed", "in transit", "at GPU", "SLO uncompressed", "SLO in transit", "SLO at GPU"], rows)
    return "\n".join(L) + "\n"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def get(d, r):
    """JSON round trips turn numeric keys into strings."""
    return d[r] if r in d else d[str(r)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-timings", action="store_true")
    ap.add_argument("--keep-timings", action="store_true", help="reuse section 9 from the current results.md")
    ap.add_argument("--optical-from-json", action="store_true",
                    help="render sections 10-15 from examples/results_optical.json instead of rerunning them")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    new = json.loads(json.dumps(collect()))          # same key types as the legacy side
    old = collect_legacy(LEGACY_REF)
    timings = None
    if a.keep_timings:
        import re
        timings = re.search(r"## 9\..*?```\n(.*?)```", OUT.read_text(), re.S).group(1)
    elif not a.no_timings:
        timings = subprocess.run([sys.executable, str(ROOT / "examples" / "benchmark_acceleration.py")],
                                 check=True, capture_output=True, text=True).stdout
    oj = ROOT / "examples" / "results_optical.json"
    if a.optical_from_json:
        optical = json.loads(oj.read_text())
    else:
        optical = json.loads(json.dumps(collect_optical()))
        oj.write_text(json.dumps(optical, indent=1))
    OUT.write_text(render(new, old, timings) + render_optical(optical))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
