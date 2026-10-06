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

Sections 16-18 (added 2026-10-05, brief 11) measure the Causal Encoder-Decoder option (DeepSeek-V4.1-Flash,
arXiv:2609.19969) on a dense Llama-3-70B-shape proxy: prefill cost, pool split and capacity across prompt:output
ratios. ``--ced-from-json`` re-renders them from examples/results_ced.json; with ``--keep-timings
--optical-from-json --ced-from-json`` nothing is rerun except sections 1-8.

Sections 19-21 (added 2026-10-06, brief 20A1) validate the simulator levers against their papers: batching policy
and chunked prefill (Sarathi-Serve, arXiv:2403.02310), paged KV and preemption (vLLM, arXiv:2309.06180) and prefix
caching (SGLang, arXiv:2312.07104). ``--levers-from-json`` re-renders them from examples/results_levers.json.

Sections 23-25 (added 2026-10-06, brief 20A2): the levers in disaggregated pools, parallelism / MoE / storage formats,
and speculative decoding against Leviathan et al. (arXiv:2211.17192); ``--levers2-from-json`` re-renders them from
examples/results_levers2.json. Section 26 is rendered from examples/tradeoffs.json (written by examples/tradeoffs.py,
the trade-off sweep), never rerun here.
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


# ───────────────────── sections 16-18: the Causal Encoder-Decoder (brief 11) ──
# DeepSeek-V4.1-Flash (arXiv:2609.19969): prefill runs the causal encoder only (section 2.2). The
# proxies are dense Llama-3-70B shapes split 40+40, so every number here is the simulator's, not
# the paper's. The prompt:output ratios stand in for agent loops (illustrative, not measured).
CED_WORKLOADS = [(2048, 512), (4096, 256), (8192, 128), (16384, 64)]   # mean prompt, mean output
CED_CLUSTER = 6                       # instances of 4xH100 (24 GPUs), split between the pools
CED_N, CED_SEED, CED_TARGET, CED_TOL = 1000, 1, 0.9, 0.02
CED_TTFT_X = 5.0                      # TTFT SLO = 5x the decoder-only model's unloaded prefill of the mean prompt
CED_GPU_BUDGET = 24
CED_SMALL_PREFILL = [(2, 5), (4, 4), (6, 3), (8, 2)]   # (prefill, decode) instances: 2P x 2 GPUs + D x 4 GPUs = 24


def collect_ced() -> dict:
    from dataclasses import replace
    from disagg_sim.hardware import H100_SXM, LLAMA3_70B, LLAMA3_70B_CED, CostModel
    from disagg_sim.search import Workload, max_sustainable_rate
    from disagg_sim.sim import SimConfig
    from disagg_sim.workload import LengthDist

    base, ced = LLAMA3_70B, LLAMA3_70B_CED
    enc = replace(ced, ced_replay_on="decode")
    variants = {"decoder-only": base, "CED, replay on prefill": ced, "CED, replay on decode": enc}
    out: dict = {}

    # 16a. activated parameters per token
    out["params"] = {"decoder_only_token": base.matmul_params, "ced_prompt_token": ced.ced_prompt_params,
                     "ced_decode_token": ced.matmul_params, "ced_kv_proj": ced.ced_kv_proj_params,
                     "layer_params": base.layer_params, "lm_head": base.vocab * base.d_model,
                     "replay": ced.ced_replay}
    # 16b. one prefill step, one prompt, 4xH100
    steps = {}
    for s in (2048, 8192, 32768):
        b = CostModel(base, H100_SXM, 4).prefill([s])
        c = CostModel(ced, H100_SXM, 4).prefill([s])
        e = CostModel(enc, H100_SXM, 4, prefill_only=True).prefill([s])
        steps[s] = {k: {"flops": x.flops, "time": x.time, "bound": x.bound} for k, x in
                    (("base", b), ("ced", c), ("enc", e))}
    out["steps"] = steps
    # 16c. what a prefill instance holds, and the KV room left
    room = {}
    for n in (1, 2, 4):
        row = {}
        for k, m, po in (("base", base, False), ("enc", enc, True)):
            cm = CostModel(m, H100_SXM, n, prefill_only=po)
            try:
                kv = cm.kv_capacity_tokens
            except ValueError:
                kv = None
            row[k] = {"weights": cm.resident_weight_bytes, "kv_tokens": kv}
        room[n] = row
    out["room"] = room

    def wl_of(p, o):
        return Workload(LengthDist(p, 0.5), LengthDist(o, 0.5), n=CED_N, seed=CED_SEED)

    def best_rate(cfg, w):
        return max_sustainable_rate(cfg, w, target=CED_TARGET, rel_tol=CED_TOL)["rate"]

    # 17. pool split at a fixed 6-instance cluster, per prompt:output ratio
    split = {}
    for p, o in CED_WORKLOADS:
        slo = CED_TTFT_X * CostModel(base, H100_SXM, 4).prefill([p]).time
        w = wl_of(p, o)
        rows = {}
        for name, m in variants.items():
            rates = {}
            for np_ in range(1, CED_CLUSTER):
                cfg = SimConfig(model=m, n_prefill=np_, n_decode=CED_CLUSTER - np_, ttft_slo=slo,
                                fast_forward=not (m.is_ced and m.ced_replay_on == "decode"))
                rates[np_] = best_rate(cfg, w)
            rows[name] = rates
        split[f"{p}:{o}"] = {"ttft_slo": slo, "rates": rows}
    out["split"] = split

    # 18. two-GPU prefill instances under a 24-GPU budget (the 8192:128 workload)
    p, o = 8192, 128
    slo = CED_TTFT_X * CostModel(base, H100_SXM, 4).prefill([p]).time
    small = {}
    for name, m in variants.items():
        rates = {}
        for np_, nd in CED_SMALL_PREFILL:
            cfg = SimConfig(model=m, n_prefill=np_, n_decode=nd, prefill_devices_per_instance=2, ttft_slo=slo,
                            fast_forward=not (m.is_ced and m.ced_replay_on == "decode"))
            rates[f"{np_}x2+{nd}x4"] = best_rate(cfg, wl_of(p, o))
        small[name] = rates
    out["small_prefill"] = {"workload": f"{p}:{o}", "ttft_slo": slo, "rates": small}
    return out


def render_ced(c: dict) -> str:
    L: list[str] = []
    p = L.append
    pa = c["params"]
    b = lambda x: f"{x / 1e9:.2f}B"
    p("## 16. Causal Encoder-Decoder (CED): the cost model (Llama-3-70B shape, 4xH100)")
    p("")
    p("DeepSeek-V4.1-Flash (arXiv:2609.19969, section 2.2) splits its 40 layers into a 20-layer causal encoder and a"
      " 20-layer decoder whose global KV is projected from the last encoder hidden state, so a prompt token runs only"
      " the encoder (and the decoder's K/V projections). The proxy here is the dense Llama-3-70B shape split 40 + 40"
      f" (`llama3-70b-ced`); the last {pa['replay']} prompt tokens are replayed through the decoder (the paper's"
      " n_win = 128), on the prefill instance (`--ced-replay-on prefill`, the paper) or as the decode instance's first"
      " step (`--ced-replay-on decode`, SGLang RFC #39963, where a prefill instance holds only the encoder). Decode"
      " always runs the whole model. Illustrative: the paper's model is a 552B MoE with compressed sparse attention.")
    p("")
    table(L, ["Matmul parameters a token touches", "Parameters", "vs a decode token"], [
        ["Decoder-only, any token (layers + LM head)", b(pa["decoder_only_token"]), "1.000"],
        ["CED, prompt token (encoder + decoder K/V projections)", b(pa["ced_prompt_token"]),
         f"{pa['ced_prompt_token'] / pa['ced_decode_token']:.3f}"],
        ["CED, decode token (whole model)", b(pa["ced_decode_token"]), "1.000"],
        ["... of which the decoder's K/V projections", b(pa["ced_kv_proj"]),
         f"{pa['ced_kv_proj'] / pa['ced_decode_token']:.3f}"],
    ])
    p("The paper's own figures are 8B activated per token at prefill and 16B at decode (abstract; section 4.2.1).")
    p("")
    rows = []
    for s, r in c["steps"].items():
        rows.append([f"{int(s):,}", f"{r['base']['flops'] / 1e15:.3f}", ms(r["base"]["time"]),
                     f"{r['ced']['flops'] / 1e15:.3f}", ms(r["ced"]["time"]),
                     f"{r['ced']['time'] / r['base']['time']:.3f}",
                     f"{r['enc']['flops'] / 1e15:.3f}", ms(r["enc"]["time"]),
                     f"{r['enc']['time'] / r['base']['time']:.3f}"])
    table(L, ["Prompt", "Decoder-only PFLOP", "time", "CED replay on prefill PFLOP", "time", "vs decoder-only",
              "CED encoder only PFLOP", "time", "vs decoder-only"], rows)
    p("One prompt per step; every step here is compute-bound. The paper: CED \"effectively halv[es] the overall"
      " computation\" of prefill for N >> n_win (section 2.2).")
    p("")
    rows = []
    for n, r in c["room"].items():
        kv = lambda x: "does not fit" if x is None else f"{x:,}"
        rows.append([f"{n}x H100", f"{r['base']['weights'] / 1e9:.1f} GB", kv(r["base"]["kv_tokens"]),
                     f"{r['enc']['weights'] / 1e9:.1f} GB", kv(r["enc"]["kv_tokens"])])
    table(L, ["Prefill instance", "Decoder-only (or CED replay on prefill): weights", "KV room, tokens",
              "CED replay on decode: weights", "KV room, tokens"], rows)
    p("KV room = 90% of HBM minus the resident weights, in tokens of BF16 KV for all 80 layers (327,680 bytes each).")
    p("")

    p("## 17. CED: pool split and capacity across prompt:output ratios (6 instances of 4xH100)")
    p("")
    p(f"Highest Poisson rate with at least {CED_TARGET:.0%} of requests inside both SLOs (bisection on full runs to"
      f" {CED_TOL:.0%}, as `search.max_sustainable_rate`), for every split of {CED_CLUSTER} instances into prefill and"
      f" decode pools. {CED_N} requests, seed {CED_SEED}, prompt and output cv 0.5, InfiniBand NDR, TPOT SLO 25 ms;"
      f" TTFT SLO = {CED_TTFT_X:g}x the decoder-only model's unloaded prefill of the mean prompt (shown). The"
      " prompt:output ratios stand in for agent loops (illustrative). Gains above 2x are queueing, not FLOPs: with"
      " a fixed TTFT SLO, halving the prefill service time cuts the queueing delay by more than half.")
    p("")
    rows = []
    for wk, d in c["split"].items():
        base_best = max(d["rates"]["decoder-only"].values())
        for name, rates in d["rates"].items():
            rs = {int(k): v for k, v in rates.items()}
            bp = max(rs, key=rs.get)
            rows.append([wk.replace(":", " : "), ms(d["ttft_slo"]) if name == "decoder-only" else "", name]
                        + [f"{rs[k]:.2f}" for k in sorted(rs)]
                        + [f"{bp}P{CED_CLUSTER - bp}D", f"{rs[bp]:.2f}", f"{rs[bp] / CED_GPU_BUDGET:.3f}",
                           f"{rs[bp] / base_best:.2f}x"])
    table(L, ["Prompt : output", "TTFT SLO", "Model"] + [f"{k}P{CED_CLUSTER - k}D req/s" for k in range(1, CED_CLUSTER)]
          + ["Best split", "Best req/s", "req/s per GPU", "vs decoder-only best"], rows)

    p("## 18. CED: two-GPU prefill instances (8,192 : 128, 24 GPUs)")
    p("")
    sp = c["small_prefill"]
    p("Prefill instances on 2 H100s (decode instances keep 4), every split that uses all 24 GPUs; otherwise as"
      f" section 17 (TTFT SLO {ms(sp['ttft_slo'])}). A whole Llama-3-70B shape only just fits on two H100s (section 16:"
      " KV room for one 8,192-token batch); the CED encoder fits with 25x the room. The roofline charges no extra"
      " cost for a tight fit, so this compares capacity, not feasibility. The decoder-only row is limited by the"
      " prefill router, not by compute: the router (unchanged) sends each request to the instance with the fewest"
      " queued prompt tokens and ignores the batch in flight, so with 1.1-second steps it piles work onto busy"
      " instances while others idle. The same router applies in section 17, where it slightly favours the model with"
      " the shorter prefill steps.")
    p("")
    rows = []
    for name, rates in sp["rates"].items():
        bk = max(rates, key=rates.get)
        rows.append([name] + [f"{rates[k]:.2f}" for k in rates] + [bk, f"{rates[bk]:.2f}"])
    keys = list(next(iter(sp["rates"].values())))
    table(L, ["Model"] + [f"{k} req/s" for k in keys] + ["Best", "Best req/s"], rows)
    return "\n".join(L) + "\n"


# ───────────────────────────────────── brief 20A1: simulator levers I (sections 19-21) ──
LV_N, LV_SEED, LV_TOL = 500, 1, 0.02
SCHED_DELAY_CAP = 2.0          # Sarathi-Serve: median scheduling delay at most 2 s (section 5.1)
Z90 = 1.2815515655446004       # standard normal 90th percentile


def lognormal_fit(median: float, p90: float) -> tuple[float, float]:
    """(mean, cv) of the lognormal with this median and 90th percentile."""
    import math
    sigma = math.log(p90 / median) / Z90
    return median * math.exp(sigma * sigma / 2), math.sqrt(math.exp(sigma * sigma) - 1)


# Sarathi-Serve Table 2 (median, P90) and its length filters (total <= 8,192 / 16,384 tokens), as lognormal
# fits clipped so that prompt + output stays within the filter.
SARATHI_DATA = {"openchat_sharegpt4": ((1730, 5696), (415, 834), 7168, 1024),
                "arxiv_summarization": ((7059, 12985), (208, 371), 15360, 1024)}
SARATHI_MODELS = {"Mistral-7B, 1xA100": ("mistral-7b", 1, 0.1, 0.5), "Yi-34B, 2xA100": ("yi-34b", 2, 0.2, 1.0)}


def collect_levers() -> dict:
    import math
    from dataclasses import replace
    from statistics import median
    from disagg_sim.hardware import A100_40G, A100_SXM, LINKS, MODELS, CostModel
    from disagg_sim.metrics import summarise
    from disagg_sim.sim import SimConfig, simulate
    from disagg_sim.workload import LengthDist, chat_sessions, poisson_workload

    out: dict = {}

    def dists(name):
        (pm, p9), (om, o9), phi, ohi = SARATHI_DATA[name]
        (pmean, pcv), (omean, ocv) = lognormal_fit(pm, p9), lognormal_fit(om, o9)
        return LengthDist(pmean, pcv, hi=phi), LengthDist(omean, ocv, hi=ohi)

    def measure(cfg, reqs):
        res = simulate(cfg, reqs)
        m = summarise(res)
        done = sorted((r for r in res.requests if r.finish is not None), key=lambda r: r.arrival)
        steady = done[int(len(done) * cfg.warmup_frac):]
        lat, s = m["latency_s"], m.get("scheduler", {})
        row = {"ttft_p50": lat["ttft"]["p50"], "ttft_p99": lat["ttft"]["p99"], "itl_p50": lat["itl"]["p50"],
               "itl_p99": lat["itl"]["p99"], "tpot_p99": lat["tpot"]["p99"],
               "sched_delay_p50": median(r.prefill_start - r.arrival for r in steady),
               "norm_latency": sum(r.e2e / r.output_len for r in steady) / len(steady),
               "tok_s": m["throughput"]["output_tok_per_s"], "req_s": m["throughput"]["req_per_s"],
               "j_tok": m["energy"]["J_per_output_token"], "completed": m["requests"]["completed"],
               "rejected": m["requests"]["rejected"], "sim_time": m["sim_time_s"]}
        for k in ("mean_running", "kv_token_frac", "preemptions", "recompute_tokens", "computed_prefill_tokens",
                  "kv_capacity_units"):
            if k in s:
                row[k] = s[k]
        if "swap" in s:
            row["swap_s"], row["swap_gb"] = s["swap"]["seconds"], s["swap"]["out_GB"] + s["swap"]["in_GB"]
        if "prefix_cache" in s:
            row["hit_rate"] = s["prefix_cache"]["hit_rate"]
            row["evicted_tokens"] = s["prefix_cache"]["evicted_tokens"]
        return row

    def bisect_rate(ok, lo=0.0, hi=0.5):
        """Highest rate where ok(rate) holds, to LV_TOL (ok is assumed true below and false above)."""
        while ok(hi):
            lo, hi = hi, hi * 2
        while hi - lo > LV_TOL * hi:
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        return lo

    # 19. batching policy (Sarathi-Serve, arXiv:2403.02310)
    sar = {}
    for label, (mkey, n, strict, relaxed) in SARATHI_MODELS.items():
        base = SimConfig(mode="colocated", model=MODELS[mkey], device=A100_SXM, devices_per_instance=n, n_colocated=1,
                         max_decode_batch=128)
        for data in (("openchat_sharegpt4", "arxiv_summarization") if mkey == "mistral-7b" else ("openchat_sharegpt4",)):
            pd, od = dists(data)

            def wl(rate):
                return poisson_workload(rate, LV_N, pd, od, seed=LV_SEED)

            caps = {}
            for slo_name, slo in (("strict", strict), ("relaxed", relaxed)):
                for pol, cfg in (("prefill-priority (vLLM)", base),
                                 ("chunked 512", replace(base, batch_policy="chunked", max_num_batched_tokens=512)),
                                 ("chunked 2048", replace(base, batch_policy="chunked", max_num_batched_tokens=2048))):
                    def ok(rate, cfg=cfg, slo=slo):
                        r = measure(cfg, wl(rate))
                        return r["itl_p99"] <= slo and r["sched_delay_p50"] <= SCHED_DELAY_CAP
                    caps[f"{slo_name}|{pol}"] = bisect_rate(ok)
            sar[f"{label}|{data}"] = {"slo": {"strict": strict, "relaxed": relaxed}, "capacity": caps,
                                      "prompt_mean": pd.mean, "prompt_cv": pd.cv, "output_mean": od.mean, "output_cv": od.cv}
    out["sarathi_capacity"] = sar

    # 19a. mechanism at one load (Mistral-7B, sharegpt): every policy
    pd, od = dists("openchat_sharegpt4")
    base = SimConfig(mode="colocated", model=MODELS["mistral-7b"], device=A100_SXM, devices_per_instance=1, n_colocated=1,
                     max_decode_batch=128)
    rate = sar["Mistral-7B, 1xA100|openchat_sharegpt4"]["capacity"]["relaxed|prefill-priority (vLLM)"] * 0.8
    pol_rows = {}
    for pol, kw in (("prefill-priority", {}), ("decode-priority", {"batch_policy": "decode-priority"}),
                    ("chunked 256", {"batch_policy": "chunked", "max_num_batched_tokens": 256}),
                    ("chunked 512", {"batch_policy": "chunked", "max_num_batched_tokens": 512}),
                    ("chunked 1024", {"batch_policy": "chunked", "max_num_batched_tokens": 1024}),
                    ("chunked 2048", {"batch_policy": "chunked", "max_num_batched_tokens": 2048}),
                    ("chunked 4096", {"batch_policy": "chunked", "max_num_batched_tokens": 4096})):
        pol_rows[pol] = measure(replace(base, max_num_batched_tokens=8192, **kw) if not kw else replace(base, **kw),
                                poisson_workload(rate, LV_N, pd, od, seed=LV_SEED))
    out["policies"] = {"rate": rate, "rows": pol_rows}

    # 19c. chunking overhead on one prompt (Yi-34B on 2xA100, Sarathi-Serve Fig. 14's setting)
    cm = CostModel(MODELS["yi-34b"], A100_SXM, 2)
    over = {}
    for s in (2048, 4096, 8192, 16384):
        whole = cm.step_mixed(0, 0, [(0, s, True)]).time
        over[s] = {c: sum(cm.step_mixed(0, 0, [(p0, min(c, s - p0), p0 + c >= s)]).time for p0 in range(0, s, c)) / whole - 1
                   for c in (512, 1024, 2048)}
    out["chunk_overhead"] = over

    # 20. KV memory (vLLM, arXiv:2309.06180): OPT-13B on one A100-40GB, ShareGPT lengths (means 161 / 338)
    kv_base = SimConfig(mode="colocated", model=MODELS["opt-13b"], device=A100_40G, devices_per_instance=1,
                        n_colocated=1, host_link=LINKS["pcie4"], max_num_batched_tokens=8192)
    p_d, o_d = LengthDist(161.31, 1.0, hi=1024), LengthDist(337.99, 1.0, hi=1024)

    def kv_wl(rate, n=1000):
        return poisson_workload(rate, n, p_d, o_d, seed=LV_SEED)

    sample = kv_wl(1.0)
    out["vllm_workload"] = {"prompt_mean": sum(r.prompt_len for r in sample) / len(sample),
                            "output_mean": sum(r.output_len for r in sample) / len(sample),
                            "kv_room_tokens": CostModel(MODELS["opt-13b"], A100_40G, 1).kv_capacity_tokens}
    pols = {"Orca (Max)": {"kv_policy": "max"}, "Orca (Pow2)": {"kv_policy": "pow2"}, "Orca (Oracle)": {},
            "paged, 16-token blocks": {"kv_policy": "paged"}}
    out["vllm_batch"] = {str(rate): {k: measure(replace(kv_base, **kw), kv_wl(rate)) for k, kw in pols.items()}
                         for rate in (2.0, 6.0)}
    thr = 0.5      # s/token: "normalized latency" stays low (vLLM Fig. 12 plots 0 to 1 s/token)

    def kv_ok(cfg):
        return lambda rate: measure(cfg, kv_wl(rate, 600))["norm_latency"] <= thr
    out["vllm_capacity"] = {"threshold": thr,
                            "rates": {k: bisect_rate(kv_ok(replace(kv_base, **kw))) for k, kw in pols.items()}}
    # block size, recompute against swap, at the memory-bound load
    blocks = {}
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        blocks[b] = {mode: measure(replace(kv_base, kv_policy="paged", kv_block_size=b, preemption=mode), kv_wl(6.0))
                     for mode in ("recompute", "swap")}
    out["vllm_blocks"] = blocks

    # 21. prefix caching (SGLang, arXiv:2312.07104)
    pc_base = SimConfig(mode="colocated", model=MODELS["mistral-7b"], device=A100_SXM, devices_per_instance=1,
                        n_colocated=1, max_num_batched_tokens=8192)
    turn_in = LengthDist(384, 0.19, lo=256, hi=512)        # SGLang's multi-turn chat: inputs 256-512 tokens

    def mt(out_d, rate, n=800, think=0.0, **kw):
        return chat_sessions(rate, n, turn_in, out_d, seed=LV_SEED, turns=4, think=think, **kw)
    multi = {}
    for name, od_ in (("short (4-8 tokens)", LengthDist(6, 0.19, lo=4, hi=8)),
                      ("long (256-512 tokens)", LengthDist(384, 0.19, lo=256, hi=512))):
        multi[name] = {str(pc): measure(replace(pc_base, prefix_caching=pc), mt(od_, 50.0)) for pc in (False, True)}
    out["sglang_multiturn"] = multi
    # hit rate against latency and throughput: a shared system prompt of growing length, one-turn requests
    hits = {}
    od_ = LengthDist(128, 0.5)
    for sys_len in (0, 256, 1024, 2048, 4096, 8192):
        def one(pc, rate):
            reqs = chat_sessions(rate, 600, LengthDist(512, 0.5), od_, seed=LV_SEED, system_prompts=4 if sys_len else 0,
                                 system_len=sys_len)
            return measure(replace(pc_base, prefix_caching=pc), reqs)
        sat = {str(pc): one(pc, 200.0)["req_s"] for pc in (False, True)}
        rate = 0.5 * sat["False"]            # latency at half the cache-off capacity, so neither side is overloaded
        hits[sys_len] = {str(pc): one(pc, rate) for pc in (False, True)}
        hits[sys_len]["saturated"], hits[sys_len]["rate"] = sat, rate
    out["sglang_hits"] = hits
    # reuse distance under LRU: think time against hit rate, little KV memory (OPT-13B on an A100-40GB)
    reuse = {}
    lr_base = replace(kv_base, prefix_caching=True)
    for think in (1.0, 10.0, 30.0, 100.0, 300.0):
        reuse[think] = measure(lr_base, chat_sessions(0.2, 400, LengthDist(384, 0.19, lo=256, hi=512),
                                                      LengthDist(64, 0.5), seed=LV_SEED, turns=4, think=think,
                                                      system_prompts=8, system_len=256))
    out["reuse"] = reuse
    return out


def render_levers(v: dict) -> str:
    L: list[str] = []
    p = L.append
    f2 = lambda x: f"{x:.2f}"
    sar = v["sarathi_capacity"]
    p("## 19. Batching policy: prefill-priority, decode-priority and chunked prefill (Sarathi-Serve)")
    p("")
    p("Brief 20A1. Colocated instances, `--batch-policy`: `prefill-priority` (today's default, vLLM v0), `decode-priority`"
      " (admit new prompts only when the running batch has drained) and `chunked` (Sarathi-Serve's stall-free batching,"
      " arXiv:2403.02310 Algorithm 3: every decode row, then unfinished prefill chunks, then new prompts, cut to the"
      " `--max-num-batched-tokens` budget, in one forward pass). Model shapes from the Hugging Face configs; the A100 is"
      " the 80 GB part; a 2xA100 instance is treated as one device (no tensor-parallel all-reduce). Workloads: lognormal"
      " fits to the paper's Table 2 medians and 90th percentiles, clipped to its length filters (shown below);"
      f" {LV_N} Poisson requests, seed {LV_SEED}; max batch 128 (the paper's largest vLLM setting).")
    p("")
    pol = v["policies"]
    p(f"**Mechanism** (Mistral-7B on 1xA100, openchat_sharegpt4 lengths, {pol['rate']:.2f} req/s = 0.8x the"
      " prefill-priority capacity under the relaxed SLO below):")
    p("")
    rows = []
    for k, r in pol["rows"].items():
        rows.append([k, ms(r["ttft_p50"]), ms(r["ttft_p99"]), ms(r["itl_p50"]), ms(r["itl_p99"]), ms(r["tpot_p99"]),
                     ms(r["sched_delay_p50"]), f"{r['tok_s']:,.0f}", f"{r['mean_running']:.1f}", f"{r['j_tok']:.3f}"])
    table(L, ["Policy", "TTFT p50", "TTFT p99", "ITL p50", "ITL p99", "TPOT p99", "Scheduling delay p50", "Output tok/s",
              "Mean running", "J/token"], rows)
    p("")
    p(f"**Capacity** (the paper's section 5.1 definition: the highest Poisson rate with ITL p99 inside the SLO and a"
      f" median scheduling delay of at most {SCHED_DELAY_CAP:g} s; bisection to {LV_TOL:.0%}). SLOs are the paper's"
      " Table 3 (strict / relaxed P99 TBT).")
    p("")
    rows = []
    for key, d in sar.items():
        label, data = key.split("|")
        c = d["capacity"]
        for slo in ("strict", "relaxed"):
            base = c[f"{slo}|prefill-priority (vLLM)"]
            best = max(c[f"{slo}|chunked 512"], c[f"{slo}|chunked 2048"])
            rows.append([label, data, f"{slo} ({d['slo'][slo]:g} s)", f2(base), f2(c[f"{slo}|chunked 512"]),
                         f2(c[f"{slo}|chunked 2048"]), f"{best / base:.2f}x" if base else "inf"])
    table(L, ["Model", "Dataset", "SLO", "Prefill-priority req/s", "Chunked 512 req/s", "Chunked 2048 req/s",
              "Best chunked vs prefill-priority"], rows)
    fits = []
    for key, d in sar.items():
        fits.append(f"{key.split('|')[1]}: prompt mean {d['prompt_mean']:,.0f} cv {d['prompt_cv']:.2f},"
                    f" output mean {d['output_mean']:,.0f} cv {d['output_cv']:.2f}")
    p("Length fits: " + "; ".join(dict.fromkeys(fits)) + ".")
    p("")
    p("**Chunking overhead** on one prompt (Yi-34B on 2xA100, the setting of the paper's Fig. 14): total time of the"
      " chunks over one whole-prompt step, minus 1.")
    p("")
    rows = [[f"{int(s_):,}"] + [pct(x) for x in r.values()] for s_, r in v["chunk_overhead"].items()]
    table(L, ["Prompt", "Chunks of 512", "Chunks of 1,024", "Chunks of 2,048"], rows)

    p("## 20. KV memory: reservation against paged blocks, and preemption (vLLM / PagedAttention)")
    p("")
    w = v["vllm_workload"]
    p("`--kv-policy`: `oracle` reserves prompt + output at admission (today's behaviour; vLLM's \"Orca (Oracle)\"), `pow2`"
      " the prompt plus the output rounded up to a power of two (\"Orca (Pow2)\"), `max` the model's 2,048-token maximum"
      " (\"Orca (Max)\"), and `paged` allocates 16-token blocks as tokens arrive, keeps 1% free at admission and, when"
      " no block is left, preempts the latest-arrived request (recompute or swap to host memory; arXiv:2309.06180"
      " sections 4.2-4.5 and 6.1). Setting of the paper's Figs. 2 and 13a: OPT-13B on one A100-40GB, ShareGPT lengths"
      f" (lognormal, cv 1.0, each clipped to 1,024; sampled means {w['prompt_mean']:.1f} / {w['output_mean']:.1f} against"
      f" the paper's 161.31 / 337.99), prefill-priority scheduling. KV room here: {w['kv_room_tokens']:,} tokens (90% of"
      " 40 GB less 26.2 GB of weights) against the paper's 15.7K slots (12 GB). The Orca baselines here have no"
      " allocator fragmentation (the paper's use a buddy allocator).")
    p("")
    rows = []
    for rate, d in v["vllm_batch"].items():
        for k, r in d.items():
            rows.append([f"{float(rate):g}", k, f"{r['mean_running']:.2f}", pct(r["kv_token_frac"]),
                         f"{r['norm_latency']:.3f}", f2(r["req_s"]), str(r.get("preemptions", 0))])
    table(L, ["Rate req/s", "KV policy", "Mean running requests", "Allocated KV holding tokens",
              "Normalised latency s/token", "Completed req/s", "Preemptions"], rows)
    cap = v["vllm_capacity"]
    rows = [[k, f2(x), f"{cap['rates']['paged, 16-token blocks'] / x:.2f}x"] for k, x in cap["rates"].items()]
    p(f"Highest rate whose mean normalised latency (end-to-end time over output tokens, the paper's metric) stays at or"
      f" below {cap['threshold']:g} s/token (600 requests, bisection to {LV_TOL:.0%}):")
    p("")
    table(L, ["KV policy", "Rate req/s", "Paged vs this"], rows)
    p("**Block size and preemption mode** at 6 req/s (memory-bound), swap over PCIe Gen4 x16 (32 GB/s, 5 us per"
      " block transfer):")
    p("")
    rows = []
    for b, d in v["vllm_blocks"].items():
        rc, sw = d["recompute"], d["swap"]
        rows.append([str(b), pct(rc["kv_token_frac"]), f"{rc['mean_running']:.2f}", f"{rc['norm_latency']:.3f}",
                     str(rc["preemptions"]), f"{sw['norm_latency']:.3f}", str(sw["preemptions"]),
                     f"{sw['swap_s']:.2f}", f"{sw['swap_s'] / sw['swap_gb']:.3f}" if sw.get("swap_gb") else "-"])
    table(L, ["Block tokens", "KV holding tokens", "Mean running", "Recompute: norm. latency", "preemptions",
              "Swap: norm. latency", "preemptions", "swap seconds", "s per GB swapped"], rows)

    p("## 21. Prefix caching (SGLang RadixAttention)")
    p("")
    p("`--prefix-caching`: an LRU cache of shared prompt segments (a radix tree whose edges are whole segments: system"
      " prompts and earlier turns). A prompt skips its longest cached prefix (never its last token); segments are"
      " inserted when its prefill ends and its own turn when it finishes; unpinned leaves are evicted least recently used"
      " first (arXiv:2312.07104 section 3). Sessions (`chat_sessions`): turns are closed-loop, released a think time"
      " after the previous turn finishes. Mistral-7B on 1xA100 unless stated; hit rate = cached prompt tokens / prompt"
      " tokens (the paper's definition).")
    p("")
    p("**Multi-turn chat** (the paper's benchmark: 4 turns, inputs 256-512 tokens, outputs 4-8 or 256-512; here"
      " lognormal cv 0.19 clipped to those ranges, no think time, sessions arriving at 50/s so the server is saturated):")
    p("")
    rows = []
    for name, d in v["sglang_multiturn"].items():
        off, on = d["False"], d["True"]
        rows.append([name, f2(off["req_s"]), f2(on["req_s"]), f"{on['req_s'] / off['req_s']:.2f}x", pct(on["hit_rate"]),
                     ms(off["ttft_p50"]), ms(on["ttft_p50"]), f"{off['mean_running']:.1f}", f"{on['mean_running']:.1f}"])
    table(L, ["Outputs", "Throughput off req/s", "on req/s", "Gain", "Hit rate", "TTFT p50 off", "TTFT p50 on",
              "Mean running off", "on"], rows)
    p("**Hit rate against latency and throughput** (one-turn requests, 4 shared system prompts of the length shown,"
      " user input lognormal mean 512 cv 0.5, output mean 128 cv 0.5, 600 requests; throughput with arrivals at 200 req/s"
      " (saturated); latency at half the cache-off throughput, shown, for both):")
    p("")
    rows = []
    for sl, d in v["sglang_hits"].items():
        off, on = d["False"], d["True"]
        sat = d["saturated"]
        rows.append([f"{int(sl):,}", pct(on["hit_rate"]), f2(d["rate"]), ms(off["ttft_p50"]), ms(on["ttft_p50"]),
                     f"{off['ttft_p50'] / on['ttft_p50']:.2f}x", f2(sat["False"]), f2(sat["True"]),
                     f"{sat['True'] / sat['False']:.2f}x"])
    table(L, ["System prompt tokens", "Hit rate", "Latency at req/s", "TTFT p50 off", "TTFT p50 on", "TTFT gain", "Saturated req/s off",
              "on", "Throughput gain"], rows)
    p("**Reuse distance under LRU** (OPT-13B on one A100-40GB, little KV memory; 4-turn sessions at 0.2 sessions/s,"
      " 8 system prompts of 256 tokens, inputs 256-512, outputs mean 64): a longer think time puts more other"
      " sessions between two turns of one session.")
    p("")
    rows = [[f"{float(t):g} s", pct(r["hit_rate"]), f"{r['evicted_tokens']:,}", ms(r["ttft_p50"])] for t, r in v["reuse"].items()]
    table(L, ["Mean think time", "Hit rate", "Tokens evicted", "TTFT p50"], rows)
    L.extend(render_validation(v))
    return "\n".join(L) + "\n"


def render_validation(v: dict) -> list[str]:
    """Section 22: each paper's figure against this simulator's, with a verdict. The papers' numbers are quoted
    from their text (arXiv versions read 2026-10-06: 2403.02310v3, 2309.06180v1, 2312.07104v2)."""
    L: list[str] = []
    sar = v["sarathi_capacity"]
    cap = lambda k, slo, pol: sar[k]["capacity"][f"{slo}|{pol}"]
    ms_s, yi = "Mistral-7B, 1xA100|openchat_sharegpt4", "Yi-34B, 2xA100|openchat_sharegpt4"
    ratio = lambda k, slo: max(cap(k, slo, "chunked 512"), cap(k, slo, "chunked 2048")) / cap(k, slo, "prefill-priority (vLLM)")
    best = lambda k, slo: "512" if cap(k, slo, "chunked 512") >= cap(k, slo, "chunked 2048") else "2048"
    ov = v["chunk_overhead"]
    b6 = v["vllm_batch"]["6.0"]
    b2 = v["vllm_batch"]["2.0"]
    run = lambda d, k: d[k]["mean_running"]
    pg = "paged, 16-token blocks"
    vc = v["vllm_capacity"]["rates"]
    blk = v["vllm_blocks"]
    s_small = blk["1"]["swap"]["swap_s"] / blk["1"]["swap"]["swap_gb"]
    s_big = blk["64"]["swap"]["swap_s"] / blk["64"]["swap"]["swap_gb"]
    swap_wins = sum(d["swap"]["norm_latency"] < d["recompute"]["norm_latency"] for d in blk.values())
    mt = v["sglang_multiturn"]
    g = lambda name: mt[name]["True"]["req_s"] / mt[name]["False"]["req_s"]
    hits = v["sglang_hits"]
    hr = [(d["True"]["hit_rate"], d["False"]["ttft_p50"] / d["True"]["ttft_p50"],
           d["saturated"]["True"] / d["saturated"]["False"]) for d in hits.values()]
    mono = all(a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2] for a, b in zip(hr, hr[1:]))
    tok = lambda d, k: 100 * d[k]["kv_token_frac"]
    rows = [
        ["Sarathi-Serve", "Capacity gain over vLLM, strict SLO: Mistral-7B \"2.6x\" (abstract), \"3.5x\" at 100 ms with a"
         " 512 budget (section 5.2)", f"{ratio(ms_s, 'strict'):.2f}x", "Reproduces (inside the paper's range)"],
        ["Sarathi-Serve", "Strict SLO, Yi-34B on 2 A100s: \"up to 3.7x\" (section 5.1)", f"{ratio(yi, 'strict'):.2f}x",
         "Same direction, smaller gain"],
        ["Sarathi-Serve", "Relaxed SLO: smaller gains, \"1.65x\" for Yi-34B at 1 s (section 5.2)",
         f"Yi-34B {ratio(yi, 'relaxed'):.2f}x, Mistral-7B {ratio(ms_s, 'relaxed'):.2f}x", "Reproduces the ordering (strict > relaxed); Yi-34B lower"],
        ["Sarathi-Serve", "Token budget 512 for strict SLOs, 2,048 for relaxed (section 5.1)",
         f"best budget strict: {best(ms_s, 'strict')} (Mistral), {best(yi, 'strict')} (Yi); relaxed: {best(ms_s, 'relaxed')} (Mistral),"
         f" {best(yi, 'relaxed')} (Yi)", "Strict reproduces; relaxed does not: chunking costs almost nothing here, so a small budget never loses"],
        ["Sarathi-Serve", "Chunking overhead at most ~25% with 512-token chunks, negligible at 2,048 (section 5.4.1, Fig. 14)",
         f"{100 * max(r['512'] for r in ov.values()):.1f}% / {100 * max(r['2048'] for r in ov.values()):.1f}%",
         "Does not reproduce: the roofline hides re-read weights under compute and has no tile-quantisation or kernel cost"],
        ["Sarathi-Serve", "Prefill-priority stalls decodes (ITL tail); decode-priority starves prefills (TTFT); chunked keeps ITL"
         " low at a small TTFT cost (sections 3.2, 4.2; Table 4)",
         "ITL p99 {:.0f} / {:.0f} / {:.0f} ms, TTFT p50 {:.0f} / {:,.0f} / {:.0f} ms (prefill-priority / decode-priority / chunked 512)".format(
             *[1e3 * v["policies"]["rows"][k]["itl_p99"] for k in ("prefill-priority", "decode-priority", "chunked 512")],
             *[1e3 * v["policies"]["rows"][k]["ttft_p50"] for k in ("prefill-priority", "decode-priority", "chunked 512")]),
         "Reproduces (qualitative)"],
        ["vLLM", "KV memory holding token states: Orca Max 20.4%, Pow2 26.8%, Oracle 38.2%, vLLM 96.3% (Fig. 2)",
         f"{tok(b6, 'Orca (Max)'):.1f}% / {tok(b6, 'Orca (Pow2)'):.1f}% / {tok(b6, 'Orca (Oracle)'):.1f}% / {tok(b6, pg):.1f}%",
         "Max and paged reproduce; Pow2 and Oracle do not (the paper's baselines also lose memory to a buddy"
         " allocator's fragmentation, which is not modelled here)"],
        ["vLLM", "Batched requests at 2 req/s: 7.00 / 9.81 / 13.62 / 30.42, paged 2.2x Oracle and 4.3x Max (Fig. 13a)",
         f"2 req/s: {run(b2, 'Orca (Max)'):.2f} / {run(b2, 'Orca (Pow2)'):.2f} / {run(b2, 'Orca (Oracle)'):.2f} / {run(b2, pg):.2f};"
         f" 6 req/s: {run(b6, 'Orca (Max)'):.2f} / {run(b6, 'Orca (Pow2)'):.2f} / {run(b6, 'Orca (Oracle)'):.2f} / {run(b6, pg):.2f}"
         f" ({run(b6, pg) / run(b6, 'Orca (Oracle)'):.2f}x, {run(b6, pg) / run(b6, 'Orca (Max)'):.2f}x)",
         "Ordering reproduces; at 2 req/s this faster roofline is not memory-bound, at 6 req/s the ratios are 1.6x / 5.4x against 2.2x / 4.3x"],
        ["vLLM", "Sustainable rate on ShareGPT: 1.7-2.7x Orca (Oracle), 2.7-8x Orca (Max) (section 6.2)",
         f"{vc[pg] / vc['Orca (Oracle)']:.2f}x / {vc[pg] / vc['Orca (Max)']:.2f}x",
         "Oracle reproduces; Max does not (larger here: 11,969 tokens of KV room hold five 2,048-token reservations,"
         " the paper's 15.7K slots seven)"],
        ["vLLM", "Swapping is slow with small blocks; recompute is better at small blocks, swap at large, comparable at 16-64"
         " (section 7.3, Fig. 19)", f"swap {s_small:.3f} s/GB at 1-token blocks, {s_big:.3f} at 64; swap has the lower normalised"
         f" latency at {swap_wins} of {len(blk)} block sizes", "Direction of the swap cost reproduces; the crossover does not (5 us per block is"
         " too small to make swapping lose here)"],
        ["SGLang", "Multi-turn chat: a noticeable speed-up with short outputs, almost none with long ones (section 6.2)",
         f"short {g('short (4-8 tokens)'):.2f}x, long {g('long (256-512 tokens)'):.2f}x", "Ordering reproduces; the long-output gain is larger here"],
        ["SGLang", "A higher cache hit rate gives a larger batch, higher throughput and lower latency (Fig. 8a-b)",
         "monotone in every column" if mono else "not monotone", "Reproduces" if mono else "Does not reproduce"],
        ["SGLang", "74.1% hit rate in production cut first-token latency 1.7x on average (Vicuna-33B, section 6.2)",
         f"{100 * hr[2][0]:.1f}% gives {hr[2][1]:.2f}x, {100 * hr[3][0]:.1f}% gives {hr[3][1]:.2f}x (Mistral-7B, half load)",
         "Not comparable: different model, traffic and load; the direction holds"],
    ]
    L.append("## 22. Validation against the papers: what reproduces and what does not")
    L.append("")
    L.append("Quoted figures are from the papers' text (arXiv 2403.02310, 2309.06180, 2312.07104, each checked at"
             " export.arxiv.org). Settings differ where the simulator cannot follow them, as stated in sections 19-21;"
             " \"reproduces\" means the measured value lies in the paper's range or has its ordering, nothing more.")
    L.append("")
    table(L, ["Paper", "Paper's figure", "This simulator", "Verdict"], rows)
    return L


# ──────────────────────────────────────────── brief 20A2: levers, part II ──
L2_N = 600           # requests per run in sections 23-25
L2_SEED = 1


def collect_levers2() -> dict:
    """Sections 23-25: the levers in disaggregated pools, parallelism / MoE / formats, speculative decoding."""
    from dataclasses import replace
    from disagg_sim.hardware import (A100_40G, A100_SXM, ACCELERATORS, H100_SXM, KV_PRESETS, LINKS, MODELS,
                                     CostModel, KVTransit, Parallel, QUANT_FORMATS)
    from disagg_sim.metrics import summarise
    from disagg_sim.sim import SimConfig, Simulation, simulate
    from disagg_sim.speculative import (Speculative, expected_operations, expected_speedup, expected_tokens)
    from disagg_sim.workload import LengthDist, chat_sessions, poisson_workload

    out: dict = {}

    def row(cfg, reqs):
        res = simulate(cfg, reqs)
        m = summarise(res)
        lat, s = m["latency_s"], m.get("scheduler", {})
        r = {"ttft_p50": lat["ttft"]["p50"], "ttft_p99": lat["ttft"]["p99"], "tpot_p50": lat["tpot"]["p50"],
             "tpot_p99": lat["tpot"]["p99"], "itl_p99": lat["itl"]["p99"], "tok_s": m["throughput"]["output_tok_per_s"],
             "req_s": m["throughput"]["req_per_s"], "j_tok": m["energy"]["J_per_output_token"],
             "kv_wait": m["stage_breakdown_s"].get("kv_wait", 0.0), "completed": m["requests"]["completed"],
             "link_GB": m["kv_link"]["bytes_GB"]}
        for k in ("mean_running", "preemptions", "computed_prefill_tokens"):
            if k in s:
                r[k] = s[k]
        if "prefix_cache" in s:
            r["hit_rate"] = s["prefix_cache"]["hit_rate"]
        if "speculative" in s:
            r["tokens_per_verify"] = s["speculative"]["tokens_per_verify"]
            r["draft_time_frac"] = s["speculative"]["draft_time_frac"]
        flops = 0.0
        for i in res.instances:
            flops += i.flops
        r["flops"] = flops
        r["out_tokens"] = sum(q.output_len for q in res.requests if q.finish is not None)
        return r

    # 23a. paged KV in the decode pool against colocated (vLLM's setting: OPT-13B on A100-40GB, ShareGPT lengths)
    opt = SimConfig(model=MODELS["opt-13b"], device=A100_40G, devices_per_instance=1, n_prefill=1, n_decode=1,
                    n_colocated=2, host_link=LINKS["pcie4"], max_num_batched_tokens=8192)
    p_d, o_d = LengthDist(161.31, 1.0, hi=1024), LengthDist(337.99, 1.0, hi=1024)
    mem = {}
    for rate in (2.0, 4.0):
        for label, kw in (("colocated x2, reserved", dict(mode="colocated")),
                          ("colocated x2, paged", dict(mode="colocated", kv_policy="paged")),
                          ("1P1D, reserved", dict(mode="disagg")),
                          ("1P1D, paged, recompute", dict(mode="disagg", kv_policy="paged")),
                          ("1P1D, paged, swap", dict(mode="disagg", kv_policy="paged", preemption="swap")),
                          ("1P1D, paged, chunked 512", dict(mode="disagg", kv_policy="paged", batch_policy="chunked",
                                                             max_num_batched_tokens=512))):
            mem[f"{rate:g}|{label}"] = row(replace(opt, **kw), poisson_workload(rate, L2_N, p_d, o_d, seed=L2_SEED))
    out["disagg_memory"] = mem

    # 23b. prefix caching: prefill pool against colocated (Llama-3-8B on 2 H100s; multi-turn sessions)
    pc = SimConfig(model=MODELS["llama3-8b"], devices_per_instance=1, n_prefill=1, n_decode=1, n_colocated=2,
                   max_num_batched_tokens=8192)
    pref = {}
    for outs, od in (("short (4-8)", LengthDist(6, 0.19, lo=4, hi=8)), ("medium (64)", LengthDist(64, 0.3, lo=32, hi=96))):
        for label, kw in (("colocated x2, off", dict(mode="colocated")), ("colocated x2, on", dict(mode="colocated", prefix_caching=True)),
                          ("1P1D, off", dict(mode="disagg")), ("1P1D, on", dict(mode="disagg", prefix_caching=True))):
            reqs = chat_sessions(2.0, L2_N, LengthDist(384, 0.19, lo=256, hi=512), od, seed=L2_SEED, turns=4, think=1.0,
                                 system_prompts=4, system_len=1024)
            pref[f"{outs}|{label}"] = row(replace(pc, **kw), reqs)
    out["disagg_prefix"] = pref

    # 23c. interactions with the other options
    inter = {}
    base8 = SimConfig(model=MODELS["llama3-8b"], devices_per_instance=1, n_prefill=1, n_decode=1)
    wl8 = lambda rate=4.0: poisson_workload(rate, L2_N, LengthDist(2048, 0.6), LengthDist(256, 0.6), seed=L2_SEED)
    for label, kw in (("H100 prefill, H100 decode, reserved", dict(max_num_batched_tokens=8192)),
                      ("H100 prefill, A100 decode, reserved", dict(decode_device=A100_SXM, max_num_batched_tokens=8192)),
                      ("H100 prefill, A100 decode, paged", dict(decode_device=A100_SXM, kv_policy="paged")),
                      ("25 GbE link, reserved", dict(link=LINKS["eth-25g"], max_num_batched_tokens=8192)),
                      ("25 GbE link, fp8 in transit, paged", dict(link=LINKS["eth-25g"], kv_policy="paged",
                                                                  kv_transit=KVTransit(KV_PRESETS["fp8"]))),
                      ("25 GbE link, fp8 at the GPU, paged", dict(link=LINKS["eth-25g"], kv_policy="paged",
                                                                  kv_transit=KVTransit(KV_PRESETS["fp8"], where="endpoint")))):
        inter[label] = row(replace(base8, **kw), wl8(2.0 if "25 GbE" in label else 4.0))
    rejected = {}
    for label, kw in (("CED model (llama3-70b-ced)", dict(model=MODELS["llama3-70b-ced"], devices_per_instance=4)),
                      ("Optical prefill pool (Hyena model on optical-fft)", dict(model=MODELS["llama3-8b-hyena"],
                                                                                prefill_device=ACCELERATORS["optical-fft"])),
                      ("Prefix caching + swap preemption", dict(prefix_caching=True, preemption="swap")),
                      ("fast_forward decode", dict(fast_forward=True))):
        try:
            Simulation(replace(base8, kv_policy="paged", **kw), wl8()[:5])
            rejected[label] = "runs"
        except ValueError as e:
            rejected[label] = str(e)
    try:
        Simulation(replace(base8, speculative=Speculative(), kv_transit=KVTransit(KV_PRESETS["fp8"])), wl8()[:5])
        rejected["Speculative decoding + hand-off compression"] = "runs"
    except ValueError as e:
        rejected["Speculative decoding + hand-off compression"] = str(e)
    out["interactions"], out["rejected"] = inter, rejected

    # 24a. tensor parallelism: Llama-3-70B, one instance of tp H100s
    m70 = MODELS["llama3-70b"]
    tp = {}
    for n in (2, 4, 8):
        cm = CostModel(m70, H100_SXM, n, parallel=Parallel(tp=n))
        plain = CostModel(m70, H100_SXM, n)
        d1, d64, pf = cm.decode_sum(2048, 1), cm.decode_sum(64 * 2048, 64), cm.prefill([8192])
        tp[n] = {"decode1": d1.time, "decode1_comm": d1.comm_time, "decode64": d64.time, "decode64_comm": d64.comm_time,
                 "prefill8k": pf.time, "prefill8k_comm": pf.comm_time, "decode64_plain": plain.decode_sum(64 * 2048, 64).time,
                 "kv_tokens": cm.kv_capacity_tokens, "tok_s_gpu_64": 64 / d64.time / n}
    out["tp"] = tp
    # 24b. pipeline against tensor parallelism on 4 GPUs
    pp = {}
    for label, par in (("TP4", Parallel(tp=4)), ("TP2 x PP2, 1 micro-batch", Parallel(tp=2, pp=2, microbatches=1)),
                       ("TP2 x PP2, 2 micro-batches", Parallel(tp=2, pp=2)),
                       ("TP2 x PP2, 4 micro-batches", Parallel(tp=2, pp=2, microbatches=4)),
                       ("PP4, 4 micro-batches", Parallel(tp=1, pp=4))):
        cm = CostModel(m70, H100_SXM, 4, parallel=par)
        pf, d64 = cm.prefill([4096] * 4), cm.decode_sum(64 * 2048, 64)
        mb, stages = min(par.microbatches or par.pp, 64), par.pp
        pp[label] = {"prefill": pf.time, "decode64": d64.time, "comm_prefill": pf.comm_time,
                     "bubble": (stages - 1) / (mb + stages - 1), "kv_tokens": cm.kv_capacity_tokens}
    out["pp"] = pp
    # 24c. mixture of experts: Mixtral-8x7B on 2 H100s
    mx = MODELS["mixtral-8x7b"]
    moe = {"touched": {b: mx.experts_touched(b) for b in (1, 2, 4, 8, 16, 64)},
           "decode_bytes": {b: CostModel(mx, H100_SXM, 2).decode_sum(b * 2048, b).bytes for b in (1, 4, 16, 64)},
           "params": mx.params, "active": mx.matmul_params + mx.vocab * mx.d_model, "layouts": {}}
    for label, par in (("TP2", Parallel(tp=2)), ("EP2", Parallel(tp=2, ep=2)),
                       ("EP2, imbalance 1.25", Parallel(tp=2, ep=2, expert_imbalance=1.25)),
                       ("EP2, imbalance 1.5", Parallel(tp=2, ep=2, expert_imbalance=1.5))):
        cm = CostModel(mx, H100_SXM, 2, parallel=par)
        pf, d32 = cm.prefill([2048] * 4), cm.decode_sum(32 * 2048, 32)
        moe["layouts"][label] = {"prefill": pf.time, "prefill_comm": pf.comm_time, "decode32": d32.time,
                                 "decode32_comm": d32.comm_time, "prefill_J": pf.compute_j + pf.memory_j}
    out["moe"] = moe
    # 24d. formats: Llama-3-70B on 4 GPUs (no parallel comm, to isolate the format)
    fm = {}
    for dev, label, w, kvf, c in (("h100", "BF16", "bf16", "bf16", "bf16"), ("h100", "FP8 weights only (W8A16)", "fp8", "bf16", "bf16"),
                                  ("h100", "FP8 W8A8", "fp8", "bf16", "fp8"), ("h100", "INT8 W8A8", "int8", "bf16", "int8"),
                                  ("h100", "INT4 weights only (W4A16)", "int4", "bf16", "bf16"),
                                  ("h100", "FP4 weights only (W4A16)", "fp4", "bf16", "bf16"),
                                  ("h100", "FP8 KV cache", "bf16", "fp8", "bf16"), ("h100", "INT4 KV cache", "bf16", "int4", "bf16"),
                                  ("h100", "FP8 W8A8 + FP8 KV", "fp8", "fp8", "fp8"),
                                  ("b200", "BF16", "bf16", "bf16", "bf16"), ("b200", "FP8 W8A8", "fp8", "bf16", "fp8"),
                                  ("b200", "FP4 W4A4", "fp4", "bf16", "fp4"), ("b200", "FP4 W4A4 + FP8 KV", "fp4", "fp8", "fp4")):
        cfg = SimConfig(model=m70, device=ACCELERATORS[dev], mode="colocated", n_colocated=1, weight_format=w, kv_format=kvf,
                        compute_format=c)
        cm = Simulation(cfg, []).colocated[0].cost
        d1, d64, pf = cm.decode_sum(2048, 1), cm.decode_sum(64 * 2048, 64), cm.prefill([8192])
        fm[f"{dev}|{label}"] = {"weights_GB": cm.model.weight_bytes_total / 1e9, "kv_tokens": cm.kv_capacity_tokens,
                                "decode1": d1.time, "decode64": d64.time, "prefill8k": pf.time,
                                "j_tok64": (d64.compute_j + d64.memory_j + cm.idle_w * d64.time) / 64,
                                "bound64": d64.bound, "boundpf": pf.bound}
    out["formats"] = fm

    # 25a/b. speculative decoding: closed forms and the simulator's draws
    out["lev_table1"] = [{"alpha": a, "gamma": g, "ops": expected_operations(a, g, 0.0), "speed": expected_speedup(a, g, 0.0)}
                         for a, g in ((0.6, 2), (0.7, 3), (0.8, 2), (0.8, 5), (0.9, 2), (0.9, 10))]
    m8 = MODELS["llama3-8b"]
    one = SimConfig(model=m8, devices_per_instance=1, mode="colocated", n_colocated=1)
    lo = lambda: poisson_workload(0.2, 150, LengthDist(512, 0.5), LengthDist(512, 0.5), seed=L2_SEED)
    # tiny prompts: decode is nearly all the work, so FLOPs per output token compare the decode schemes alone
    tiny = lambda: poisson_workload(0.2, 150, LengthDist(16), LengthDist(512, 0.5), seed=L2_SEED)
    draws = {}
    for a in (0.6, 0.8):
        for g in (2, 4, 6):
            r = row(replace(one, speculative=Speculative("mtp", g, a)), lo())
            draws[f"{a}|{g}"] = {"sim": r["tokens_per_verify"], "closed": expected_tokens(a, g)}
    out["spec_draws"] = draws
    # 25c. Theorem 3.8 (wall time) and 3.11 (operations) against the simulator at low load
    th = {}
    for mk, n in (("llama3-8b", 1), ("llama3-70b", 4)):
        base = SimConfig(model=MODELS[mk], devices_per_instance=n, mode="colocated", n_colocated=1)
        r0, f0 = row(base, lo()), row(base, tiny())
        target = CostModel(MODELS[mk], H100_SXM, n)
        for draft in ("mtp", "llama3.2-1b"):
            for a, g in ((0.7, 3), (0.8, 5)):
                sp = Speculative(draft, g, a)
                r1, f1 = row(replace(base, speculative=sp), lo()), row(replace(base, speculative=sp), tiny())
                dm = (replace(MODELS[mk], n_layers=1) if draft == "mtp" else MODELS[draft])
                dcm = CostModel(dm, H100_SXM, n, 0.0)
                c = dcm.decode_sum(1024, 1).time / target.decode_sum(1024, 1).time
                c_hat = dm.matmul_params / MODELS[mk].matmul_params
                th[f"{mk}|{draft}|{a}|{g}"] = {"c": c, "c_hat": c_hat, "expected": expected_speedup(a, g, c),
                                               "simulated": r0["tpot_p50"] / r1["tpot_p50"],
                                               "ops_expected": expected_operations(a, g, c_hat),
                                               "ops_simulated": (f1["flops"] / f1["out_tokens"]) / (f0["flops"] / f0["out_tokens"])}
    out["spec_theorems"] = th
    # 25d. where it helps and where it hurts: time per output token against batch, from the cost model
    hh = {}
    for dev in ("h100", "a100"):
        cm = CostModel(m8, ACCELERATORS[dev], 1)
        dcm = CostModel(replace(m8, n_layers=1), ACCELERATORS[dev], 1, 0.0)
        for c_len, b in [(c, b) for c in (512, 2048) for b in (1, 32, 128, 512)]:
            ctx = b * c_len
            plain = cm.decode_sum(ctx, b)
            g, a = 3, 0.7
            verify = cm.step_spec(ctx, b, g + 1)
            draft = 0.0
            for j in range(g):
                draft += dcm.decode_sum(ctx + j * b, b).time
            per_tok_spec = (verify.time + draft) / (b * expected_tokens(a, g))
            hh[f"{dev}|{c_len}|{b}"] = {"plain": plain.time / b, "spec": per_tok_spec, "bound_plain": plain.bound,
                                "bound_verify": verify.bound}
    out["spec_batch"] = hh
    # ... and in the simulator: an 8B colocated instance from light to saturated load
    load = {}
    for rate in (0.5, 4.0, 16.0, 64.0):
        for label, sp in (("off", None), ("on", Speculative("mtp", 3, 0.7))):
            cfg = replace(one, device=A100_SXM, speculative=sp, max_decode_batch=512)
            r = row(cfg, poisson_workload(rate, L2_N, LengthDist(1024, 0.5), LengthDist(256, 0.5), seed=L2_SEED))
            load[f"{rate:g}|{label}"] = r
    out["spec_load"] = load
    return out


def render_levers2(v: dict) -> str:
    L: list[str] = []
    p = L.append
    f2 = lambda x: f"{x:.2f}"
    p("## 23. The levers in disaggregated pools")
    p("")
    p("Brief 20A2 (owner decision 2026-10-06). With any scheduling lever on, a disaggregated run uses"
      " `ScheduledPrefillInstance` and `ScheduledDecodeInstance`. The **prefill pool** holds prompt KV only, from"
      " admission until its hand-off has landed, plus the prefix cache; it batches whole prompts to the budget, or"
      " chunks (no decode rows to piggy-back). Its cache holds prompt segments, not replies (they are generated on the"
      " decode pool), so a session's next turn recomputes the previous reply. The **decode pool** pulls each hand-off"
      " only once it can allocate its KV (the policy's reservation, or the prompt's blocks plus the 1% watermark when"
      " paged), so a transfer never lands without room; then it runs the colocated scheduler's paged growth and"
      " latest-arrival preemption (recompute here, or swap to host). With every lever at its default the scheduled pools"
      " reproduce the old ones bit for bit (tested with one decode instance; with several, the hand-off is routed when"
      " prefill ends rather than when the KV lands).")
    p("")
    p(f"**Paged KV in the decode pool** (vLLM's setting: OPT-13B, one A100-40GB per instance, ShareGPT lengths as in"
      f" section 20; {L2_N} Poisson requests; reserved KV rows use the scheduled pools at default levers):")
    p("")
    rows = []
    for k, r in v["disagg_memory"].items():
        rate, label = k.split("|")
        rows.append([rate, label, f"{r['tok_s']:,.0f}", f"{r.get('mean_running', 0):.1f}", str(r.get("preemptions", "-")),
                     ms(r["ttft_p50"]), ms(r["ttft_p99"]), ms(r["tpot_p50"]), ms(r["tpot_p99"]), ms(r["kv_wait"]),
                     f"{r['j_tok']:.3f}"])
    table(L, ["Rate req/s", "Configuration", "Output tok/s", "Mean running (decode)", "Preemptions", "TTFT p50", "TTFT p99",
              "TPOT p50", "TPOT p99", "Mean KV wait (ms)", "J/token"], rows)
    p("**Prefix caching in the prefill pool** (Llama-3-8B, one H100 per instance, 2 GPUs either way; 4-turn sessions at"
      " 2/s, 1 s think time, 4 system prompts of 1,024 tokens, turn inputs 256-512):")
    p("")
    rows = []
    for k, r in v["disagg_prefix"].items():
        outs, label = k.split("|")
        rows.append([outs, label, pct(r.get("hit_rate", 0.0)), f"{r['computed_prefill_tokens']:,}", ms(r["ttft_p50"]),
                     ms(r["ttft_p99"]), ms(r["tpot_p50"]), f"{r['link_GB']:.1f}"])
    table(L, ["Outputs", "Configuration", "Hit rate", "Prefill tokens computed", "TTFT p50", "TTFT p99", "TPOT p50",
              "Hand-off GB"], rows)
    p("**Interactions with the other options** (Llama-3-8B, 1P1D, one GPU per instance, prompts 2,048 / outputs 256;"
      " 4 req/s, 2 req/s on 25 GbE):")
    p("")
    rows = [[k, f"{r['tok_s']:,.0f}", ms(r["ttft_p50"]), ms(r["tpot_p50"]), ms(r["kv_wait"]), f"{r['link_GB']:.1f}",
             str(r.get("preemptions", "-"))] for k, r in v["interactions"].items()]
    table(L, ["Configuration", "Output tok/s", "TTFT p50", "TPOT p50", "Mean KV wait (ms)", "GB on the link", "Preemptions"], rows)
    p("Combinations that are **not modelled** and are rejected with these errors:")
    p("")
    for k, e in v["rejected"].items():
        p(f"- {k}: \"{e}\"")
    p("")
    p("So heterogeneous pools and in-transit or endpoint hand-off compression combine with the levers. The CED option"
      " and the FFT-mixing models (the only ones an optical prefill pool speeds up) do not: the levers are modelled for"
      " attention models without CED.")
    p("")

    p("## 24. Parallelism, mixture of experts and storage formats")
    p("")
    p("`hardware.Parallel` (first-order, illustrative): ring all-reduces after attention and after the MLP"
      " (Megatron-LM, arXiv:1909.08053), 2(n-1)/n S / B + 2(n-1) x link latency each, on the device's NVLink, not"
      " overlapped with compute; GPipe micro-batches (arXiv:1811.06965), m + p - 1 slots, each slot re-reading its"
      " stage's weights; MoE all-to-all dispatch and combine (GShard, arXiv:2006.16668). Memory is checked per GPU, stage"
      " by stage. Cost-model steps below (no queueing).")
    p("")
    p("**Tensor parallelism**, Llama-3-70B on H100s (decode contexts 2,048 tokens):")
    p("")
    rows = []
    for n, r in v["tp"].items():
        rows.append([str(n), ms(r["decode1"]), pct(r["decode1_comm"] / r["decode1"]), ms(r["decode64"]),
                     pct(r["decode64_comm"] / r["decode64"]), ms(r["decode64_plain"]), ms(r["prefill8k"]),
                     pct(r["prefill8k_comm"] / r["prefill8k"]), f"{r['tok_s_gpu_64']:,.0f}", f"{r['kv_tokens']:,}"])
    table(L, ["TP GPUs", "Decode b=1", "of it all-reduce", "Decode b=64", "of it all-reduce", "b=64 without the all-reduce",
              "Prefill 8,192", "of it all-reduce", "Decode tok/s per GPU at b=64", "KV tokens"], rows)
    p("**Pipeline against tensor parallelism** on 4 H100s (prefill: 4 prompts of 4,096; decode: 64 rows of 2,048):")
    p("")
    rows = [[k, ms(r["prefill"]), ms(r["decode64"]), pct(r["bubble"]), f"{r['kv_tokens']:,}"] for k, r in v["pp"].items()]
    table(L, ["Layout", "Prefill step", "Decode step", "GPipe bubble (p-1)/(m+p-1)", "KV tokens"], rows)
    p("Micro-batches shrink the bubble of a compute-bound prefill but re-read every stage's weights, so they lengthen a"
      " memory-bound decode step. Successive steps are not overlapped across stages here (no cross-step pipelining, as"
      " serving engines do with several batches in flight), so pipeline parallelism shows its cost, not its"
      " throughput: a known gap.")
    p("")
    mo = v["moe"]
    p(f"**Mixture of experts**, Mixtral-8x7B ({mo['params'] / 1e9:.1f}B parameters, {mo['active'] / 1e9:.1f}B active with"
      " the input embedding; 8 experts, top-2) on 2 H100s. Expected experts a layer touches, E(1 - (1 - k/E)^tokens), and"
      " the bytes a decode step reads:")
    p("")
    table(L, ["Tokens in the step"] + [str(b) for b in mo["touched"]], [["Experts touched (of 8)"] + [f"{x:.2f}" for x in mo["touched"].values()]])
    table(L, ["Decode batch"] + [str(b) for b in mo["decode_bytes"]], [["GB read per step"] + [f"{x / 1e9:.1f}" for x in mo["decode_bytes"].values()]])
    rows = [[k, ms(r["prefill"]), pct(r["prefill_comm"] / r["prefill"]), ms(r["decode32"]), pct(r["decode32_comm"] / r["decode32"]),
             f"{r['prefill_J']:.0f}"] for k, r in mo["layouts"].items()]
    table(L, ["Layout (2 GPUs)", "Prefill 4 x 2,048", "of it link", "Decode b=32", "of it link", "Prefill dynamic J"], rows)
    p("**Storage and compute formats**, Llama-3-70B on 4 GPUs (no all-reduce, to isolate the format; decode contexts"
      " 2,048). Bytes per value include block scales (INT4: groups of 128 with an FP16 scale, AWQ arXiv:2306.00978; FP4:"
      " MXFP4, 32 values per 8-bit scale, arXiv:2310.10537). A compute format runs at the datasheet's dense rate when the"
      " device has units for it (H100: FP8, INT8; B200: FP8, FP4); otherwise weights are dequantised to BF16 before the"
      " multiply (weight-only), at no modelled cost. **Accuracy is not simulated**: see the Numerics site for what each"
      " format does to accuracy.")
    p("")
    rows = []
    for k, r in v["formats"].items():
        dev, label = k.split("|")
        rows.append([dev.upper(), label, f"{r['weights_GB']:.1f}", f"{r['kv_tokens']:,}", ms(r["decode1"]), ms(r["decode64"]),
                     f"{r['bound64']}", ms(r["prefill8k"]), f"{r['boundpf']}", f"{r['j_tok64']:.3f}"])
    table(L, ["Device", "Format", "Weights GB", "KV tokens", "Decode b=1", "Decode b=64", "bound", "Prefill 8,192", "bound",
              "J/token at b=64 (incl. idle)"], rows)

    p("## 25. Speculative decoding against Leviathan et al.")
    p("")
    p("`--speculative DRAFT --gamma G --alpha A`: each decode step drafts G tokens per row (G draft passes over the batch),"
      " verifies them in one target pass over G + 1 positions per row (`CostModel.step_spec`), and keeps the accepted run"
      " plus one token, drawn per request with acceptance rate A (i.i.d., the paper's assumption, section 3.1). Drafts:"
      " `mtp` (one extra layer of the target sharing its embedding and LM head, DeepSeek-V3's multi-token prediction"
      " module, arXiv:2412.19437) or `llama3.2-1b`. The draft's weights and KV share the instance's memory and its KV"
      " crosses the hand-off in disaggregated mode. Closed forms from arXiv:2211.17192 (read 2026-10-06, v2).")
    p("")
    p("**The paper's Table 1** (c = c^ = 0), from the closed forms in `speculative.py`:")
    p("")
    paper = {(0.6, 2): (1.53, 1.96), (0.7, 3): (1.58, 2.53), (0.8, 2): (1.23, 2.44), (0.8, 5): (1.63, 3.69),
             (0.9, 2): (1.11, 2.71), (0.9, 10): (1.60, 6.86)}
    rows = [[f"{r['alpha']:g}", str(r["gamma"]), f"{paper[(r['alpha'], r['gamma'])][0]:.2f}x", f"{r['ops']:.2f}x",
             f"{paper[(r['alpha'], r['gamma'])][1]:.2f}x", f"{r['speed']:.2f}x"] for r in v["lev_table1"]]
    table(L, ["alpha", "gamma", "Operations (paper)", "Operations (Theorem 3.11 here)", "Speed (paper)", "Speed (Theorem 3.8 here)"], rows)
    p("**Tokens per verify pass**: the simulator's draws against equation (1) (Llama-3-8B, one H100, MTP draft, 150"
      " requests at 0.2/s, outputs mean 512; the last pass of a request is capped at its remaining tokens):")
    p("")
    rows = [[k.split("|")[0], k.split("|")[1], f"{r['closed']:.3f}", f"{r['sim']:.3f}", pct(r["sim"] / r["closed"] - 1)]
            for k, r in v["spec_draws"].items()]
    table(L, ["alpha", "gamma", "Equation (1)", "Simulated", "Difference"], rows)
    p("**Wall time and operations at low load** (one instance, 0.2 req/s, outputs mean 512): Theorem 3.8 with"
      " c = one draft decode step over one target decode step (batch 1, from the cost model), against the simulated"
      " TPOT p50 ratio (prompts mean 512); Theorem 3.11 with c^ = draft over target matmul parameters per token, against"
      " the simulated FLOPs per output token ratio (16-token prompts, so decode is nearly all the work; attention FLOPs"
      " are not in c^):")
    p("")
    rows = []
    for k, r in v["spec_theorems"].items():
        mk, dr, a, g = k.split("|")
        rows.append([mk, dr, a, g, f"{r['c']:.3f}", f"{r['expected']:.2f}x", f"{r['simulated']:.2f}x", f"{r['ops_expected']:.2f}x",
                     f"{r['ops_simulated']:.2f}x"])
    table(L, ["Target", "Draft", "alpha", "gamma", "c", "Speed-up, Theorem 3.8", "Simulated", "Operations, Theorem 3.11",
              "Simulated"], rows)
    p("**Where it helps and where it hurts**: time per output token from the cost model (one step, no queueing),"
      " Llama-3-8B with an MTP draft, gamma 3, alpha 0.7 (2.53 tokens per pass), against context and batch size:")
    p("")
    rows = []
    for k, r in v["spec_batch"].items():
        dev, c_len, b = k.split("|")
        rows.append([dev.upper(), f"{int(c_len):,}", b, f"{1e3 * r['plain']:.3f}", r["bound_plain"], f"{1e3 * r['spec']:.3f}",
                     r["bound_verify"], f"{r['plain'] / r['spec']:.2f}x"])
    table(L, ["Device", "Context", "Batch", "ms/token plain", "bound", "ms/token speculative", "verify bound", "Speed-up"], rows)
    p("In the simulator (Llama-3-8B on one A100, prompts 1,024 / outputs 256, max batch 512), from light load to"
      " saturation:")
    p("")
    rows = []
    for rate in dict.fromkeys(k.split("|")[0] for k in v["spec_load"]):
        off, on = v["spec_load"][f"{rate}|off"], v["spec_load"][f"{rate}|on"]
        rows.append([rate, f"{off['tok_s']:,.0f}", f"{on['tok_s']:,.0f}", f"{on['tok_s'] / off['tok_s']:.2f}x", ms(off["tpot_p50"]),
                     ms(on["tpot_p50"]), f"{off.get('mean_running', 0):.0f}" if 'mean_running' in off else "-",
                     f"{on['mean_running']:.0f}", pct(on["draft_time_frac"])])
    table(L, ["Rate req/s", "Throughput off", "on", "Ratio", "TPOT p50 off", "on", "Mean running off", "on",
              "Draft share of busy time"], rows)
    return "\n".join(L) + "\n"


def render_tradeoffs(t: dict) -> str:
    """Section 26: the sweep's summary, rendered from examples/tradeoffs.json (written by examples/tradeoffs.py)."""
    L: list[str] = []
    p = L.append
    meta = t["meta"]
    p("## 26. The trade-off sweep (`examples/tradeoffs.py` -> `examples/tradeoffs.json`)")
    p("")
    p(f"Every point is the same {meta['gpus']} GPUs serving {meta['model']}; the baseline is 2 colocated instances of TP4"
      " (all-reduces priced), prefill-priority batching, reserved KV and BF16, and each lever changes that. Capacity:"
      f" the highest arrival rate with at least {meta['slo_target']:.0%} of requests meeting both SLOs (DistServe's"
      " goodput, arXiv:2401.09670; rejected requests count as misses), by doubling and bisection to"
      f" {meta['capacity_tolerance']:.0%}. Cost per million output tokens at capacity uses illustrative prices per"
      " GPU-hour (" + ", ".join(f"{k.upper()} ${v:.2f}" for k, v in meta["usd_per_gpu_hour"].items()) + "); energy"
      " per token is the power model's (illustrative coefficients, static power included). Latencies are at each"
      f" workload's reference load. Speculative rows use alpha {meta['speculative_alpha']} (a parameter). Simulator"
      f" {meta['simulator_commit']}, {meta['generated']}, {meta['wall_s'] / 60:.0f} min on {meta['workers']} workers."
      " Accuracy is not simulated.")
    p("")
    rows = []
    for k, w in t["workloads"].items():
        rows.append([w["label"], f"{w['prompt_mean']:,.0f} (cv {w['prompt_cv']:g})", f"{w['output_mean']:,.0f}",
                     str(w["turns"]), f"{w['system_prompts']} x {w['system_len']:,}" if w["system_prompts"] else "-",
                     f"{w['ttft_slo']:g} s / {1e3 * w['tpot_slo']:g} ms",
                     "saturated" if w["reference_rate"] > 1e4 else f"{w['reference_rate']:.3g}"])
    table(L, ["Workload", "New input tokens", "Output tokens", "Turns", "Shared prefixes", "SLO TTFT / TPOT",
              "Reference load (/s)"], rows)
    pts = {(q["workload"], q["hardware"], q["lever"]): q for q in t["points"]}
    eff = t["effects"]
    levers = t["levers"]
    hdr = ["Lever", "Goodput/GPU", "$/M tok", "J/tok", "TTFT p99", "TPOT p99", "ITL p99"]
    keys = ["goodput_req_s_per_gpu", "usd_per_mtok", "j_per_tok", "ttft_p99", "tpot_p99", "itl_p99"]

    def cell(x):
        return "-" if x is None else f"{100 * x:+.0f}%"
    for wk, w in t["workloads"].items():
        b = pts.get((wk, "h100", "baseline"))
        if b is None or "metrics" not in b:
            continue
        bm = b["metrics"]
        p(f"**{w['label']}** on H100. Baseline: goodput {bm['goodput_req_s_per_gpu']:.3f} req/s per GPU, "
          + (f"${bm['usd_per_mtok']:.2f} per M output tokens, {bm['j_per_tok']:.2f} J/token" if bm.get("usd_per_mtok") else "no capacity")
          + f"; at {'saturation' if w['reference_rate'] > 1e4 else format(w['reference_rate'], '.3g') + '/s'} TTFT p99"
            f" {ms(bm['ttft_p99']).strip()}, TPOT p99 {ms(bm['tpot_p99']).strip()}."
          " Relative change per lever (Pareto-optimal over all six metrics marked *):")
        p("")
        rows = []
        for lk, info in levers.items():
            if lk == "baseline":
                continue
            q = pts.get((wk, "h100", lk))
            if q is None:
                continue
            if "error" in q:
                rows.append([info["label"], "error: " + q["error"][:60]] + [""] * 5)
                continue
            e = eff[wk]["h100"].get(lk, {})
            star = "*" if q.get("pareto", {}).get("all") else ""
            rows.append([info["label"] + star] + [cell(e.get(k)) for k in keys])
        table(L, hdr, rows)
    # sign flips across workloads
    p("**Levers whose effect on goodput per GPU changes sign between workloads** (H100; +/- is more/less goodput"
      " than the baseline by more than 2%):")
    p("")
    rows = []
    for lk, info in levers.items():
        if lk == "baseline":
            continue
        signs = {}
        for wk in t["workloads"]:
            x = eff.get(wk, {}).get("h100", {}).get(lk, {}).get("goodput_req_s_per_gpu")
            if x is not None:
                signs[wk] = "+" if x > 0.02 else "-" if x < -0.02 else "0"
        if len({s for s in signs.values() if s != "0"}) > 1:
            rows.append([info["label"]] + [signs.get(wk, "") for wk in t["workloads"]])
    table(L, ["Lever"] + [w["label"] for w in t["workloads"].values()], rows)
    # hardware
    p("**Hardware**: goodput per GPU and cost per M output tokens of the baseline and of the best lever on each device"
      " (best = highest goodput per GPU):")
    p("")
    rows = []
    for wk, w in t["workloads"].items():
        for hw in t["hardware"]:
            b = pts.get((wk, hw, "baseline"))
            cand = [q for q in t["points"] if q["workload"] == wk and q["hardware"] == hw and "metrics" in q]
            if b is None or not cand:
                continue
            best = max(cand, key=lambda q: q["metrics"]["goodput_req_s_per_gpu"])
            bm, xm = b["metrics"], best["metrics"]
            usd = lambda m: f"{m['usd_per_mtok']:.2f}" if m.get("usd_per_mtok") else "-"
            rows.append([w["label"], hw.upper(), f"{bm['goodput_req_s_per_gpu']:.3f}", usd(bm), levers[best["lever"]]["label"],
                         f"{xm['goodput_req_s_per_gpu']:.3f}", usd(xm)])
    table(L, ["Workload", "Device", "Baseline goodput/GPU", "Baseline $/M tok", "Best lever", "Its goodput/GPU", "Its $/M tok"], rows)
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
    ap.add_argument("--ced-from-json", action="store_true",
                    help="render sections 16-18 from examples/results_ced.json instead of rerunning them")
    ap.add_argument("--levers-from-json", action="store_true",
                    help="render sections 19-21 from examples/results_levers.json instead of rerunning them")
    ap.add_argument("--levers2-from-json", action="store_true",
                    help="render sections 23-25 from examples/results_levers2.json instead of rerunning them")
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
    cj = ROOT / "examples" / "results_ced.json"
    if a.ced_from_json:
        ced = json.loads(cj.read_text())
    else:
        ced = json.loads(json.dumps(collect_ced()))
        cj.write_text(json.dumps(ced, indent=1))
    lj = ROOT / "examples" / "results_levers.json"
    if a.levers_from_json:
        levers = json.loads(lj.read_text())
    else:
        levers = json.loads(json.dumps(collect_levers()))
        lj.write_text(json.dumps(levers, indent=1))
    l2j = ROOT / "examples" / "results_levers2.json"
    if a.levers2_from_json:
        levers2 = json.loads(l2j.read_text())
    else:
        levers2 = json.loads(json.dumps(collect_levers2()))
        l2j.write_text(json.dumps(levers2, indent=1))
    tail = "\n" + render_levers2(levers2)
    tj = ROOT / "examples" / "tradeoffs.json"
    if tj.exists():
        tail += "\n" + render_tradeoffs(json.loads(tj.read_text()))
    OUT.write_text(render(new, old, timings) + render_optical(optical) + "\n" + render_ced(ced) + "\n"
                   + render_levers(levers) + tail)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
