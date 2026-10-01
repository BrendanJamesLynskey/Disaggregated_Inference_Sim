"""Command-line front end.

    disagg-sim                                   # one disaggregated run, report to stdout
    disagg-sim --compare                         # colocated vs disaggregated, same GPUs
    disagg-sim --sweep-rate 2 4 8 16             # load sweep: where does goodput collapse?
    disagg-sim --trace run.json                  # Perfetto / chrome://tracing timeline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .hardware import ACCELERATORS, LINKS, MODELS
from .metrics import format_report, summarise
from .sim import SimConfig, simulate
from .trace import write_trace
from .workload import LengthDist, poisson_workload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="disagg-sim", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="llama3-70b")
    p.add_argument("--device", choices=ACCELERATORS, default="h100")
    p.add_argument("--devices-per-instance", type=int, default=4)
    p.add_argument("--mode", choices=["disagg", "colocated"], default="disagg")
    p.add_argument("--prefill", type=int, default=1, help="prefill instances (disagg)")
    p.add_argument("--decode", type=int, default=1, help="decode instances (disagg)")
    p.add_argument("--link", choices=LINKS, default="ib-ndr")
    p.add_argument("--link-channels", type=int, default=1)
    p.add_argument("--rate", type=float, default=4.0, help="Poisson arrival rate, req/s")
    p.add_argument("--requests", type=int, default=800)
    p.add_argument("--prompt", type=float, default=2048, help="mean prompt length")
    p.add_argument("--prompt-cv", type=float, default=0.5)
    p.add_argument("--output", type=float, default=256, help="mean output length")
    p.add_argument("--output-cv", type=float, default=0.5)
    p.add_argument("--ttft-slo", type=float, default=1.0)
    p.add_argument("--tpot-slo", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--compare", action="store_true",
                   help="also run colocated on prefill+decode instances")
    p.add_argument("--sweep-rate", type=float, nargs="+", metavar="RATE")
    p.add_argument("--trace", metavar="FILE", help="write a Chrome trace-event JSON")
    p.add_argument("--json", action="store_true", help="print metrics as JSON")
    p.add_argument("--power-cap", type=float, metavar="W",
                   help="per-device power cap in watts (steps throttle to respect it)")
    p.add_argument("--prefill-power-cap", type=float, metavar="W")
    p.add_argument("--decode-power-cap", type=float, metavar="W")
    p.add_argument("--dvfs", action="store_true",
                   help="lower compute clocks on memory-bound steps (saves energy, not time)")
    p.add_argument("--fast", action="store_true",
                   help="exact accelerated decode path (macro-steps + lazy bookkeeping)")
    return p


def config_from_args(a) -> SimConfig:
    link = replace(LINKS[a.link], channels=a.link_channels)
    return SimConfig(model=MODELS[a.model], device=ACCELERATORS[a.device],
                     devices_per_instance=a.devices_per_instance, mode=a.mode,
                     n_prefill=a.prefill, n_decode=a.decode,
                     n_colocated=a.prefill + a.decode, link=link,
                     ttft_slo=a.ttft_slo, tpot_slo=a.tpot_slo, trace=bool(a.trace),
                     fast_forward=a.fast, power_cap_w=a.power_cap,
                     prefill_power_cap_w=a.prefill_power_cap, decode_power_cap_w=a.decode_power_cap,
                     dvfs=a.dvfs)


def run_once(cfg: SimConfig, a, rate: float) -> dict:
    wl = poisson_workload(rate, a.requests, LengthDist(a.prompt, a.prompt_cv),
                          LengthDist(a.output, a.output_cv), seed=a.seed)
    res = simulate(cfg, wl)
    if a.trace and res.trace:
        write_trace(res.trace, a.trace)
    return summarise(res)


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    cfg = config_from_args(a)

    if a.sweep_rate:
        modes = ["colocated", "disagg"] if a.compare else [cfg.mode]
        print(f"{'mode':<10} {'rate':>6} {'TTFT p99':>9} {'TPOT p99':>9} {'goodput':>8} "
              f"{'SLO%':>6}  hot-spot (stage -> resource)")
        for mode in modes:
            for rate in a.sweep_rate:
                m = run_once(replace(cfg, mode=mode, trace=False), a, rate)
                lat, t, h = m["latency_s"], m["throughput"], m["hotspots"]
                print(f"{mode:<10} {rate:6.1f} {1e3 * lat['ttft']['p99']:8.0f}m "
                      f"{1e3 * lat['tpot']['p99']:8.1f}m {t['goodput_req_per_s']:8.2f} "
                      f"{100 * t['slo_attainment']:6.1f}  {h['stage']} -> {h['resource']}")
        return

    runs = [replace(cfg, mode="colocated"), replace(cfg, mode="disagg")] if a.compare else [cfg]
    results = [run_once(c, a, a.rate) for c in runs]
    if a.json:
        print(json.dumps(results if a.compare else results[0], indent=2))
    else:
        print("\n\n".join(format_report(m) for m in results))


if __name__ == "__main__":
    main()
