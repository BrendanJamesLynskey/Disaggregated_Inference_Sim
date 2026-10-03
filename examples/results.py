"""Regenerate examples/results.md: every number quoted in the README and the decks.

    python examples/results.py                  # deterministic runs + timings (run on an idle machine)
    python examples/results.py --no-timings     # deterministic runs only

The cost model was corrected on 2026-10-03 (per-step weight traffic now reads only
the embedding rows looked up, not the whole table; decode attention includes the
new token's attention to itself). Section 1 and the before/after columns rerun the
same experiments on the last commit before the correction (``LEGACY_REF``), unpacked
with ``git archive`` into a temporary directory, so both sides come from real runs.

Timings are wall-clock on whatever machine runs this; the header records which one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import replace
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
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    new = json.loads(json.dumps(collect()))          # same key types as the legacy side
    old = collect_legacy(LEGACY_REF)
    timings = None
    if not a.no_timings:
        timings = subprocess.run([sys.executable, str(ROOT / "examples" / "benchmark_acceleration.py")],
                                 check=True, capture_output=True, text=True).stdout
    OUT.write_text(render(new, old, timings))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
