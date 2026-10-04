"""Command-line front end.

    disagg-sim                                   # one disaggregated run, report to stdout
    disagg-sim --compare                         # colocated vs disaggregated, same GPUs
    disagg-sim --sweep-rate 2 4 8 16             # load sweep: where does goodput collapse?
    disagg-sim --trace run.json                  # Perfetto / chrome://tracing timeline
    disagg-sim --model llama3-8b --devices-per-instance 1 --prefill-device h100 --decode-device a100
    disagg-sim --model llama3-8b-hyena-circ --devices-per-instance 1 --prefill-device optical-fft --decode-device h100
    disagg-sim --model llama3-8b --devices-per-instance 1 --link eth-25g --kv-compress fp8 --kv-compress-at transit
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .hardware import ACCELERATORS, KV_PRESETS, LINKS, MODELS, KVTransit
from .metrics import format_report, summarise
from .ppa import ppa_report
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
    g = p.add_argument_group("heterogeneous pools (default: --device for both)")
    g.add_argument("--prefill-device", choices=ACCELERATORS)
    g.add_argument("--decode-device", choices=ACCELERATORS)
    g.add_argument("--prefill-devices-per-instance", type=int)
    g.add_argument("--decode-devices-per-instance", type=int)
    g.add_argument("--prefill-lm-head", choices=["all", "last"],
                   help="LM head on every prompt token (default) or the last one only")
    g = p.add_argument_group("optical transform engine (devices optical-fft*; illustrative)")
    g.add_argument("--ft-enob", type=int, help="effective bits (passes = 4^(required - ENOB))")
    g.add_argument("--ft-static-w", type=float, help="lasers + thermal tuning, W per device (split evenly)")
    g.add_argument("--ft-mask-rate", type=float, help="Fourier-plane mask rewrites per second")
    g.add_argument("--ft-samples-per-s", type=float, help="converter samples per second")
    g.add_argument("--ft-detection", choices=["coherent", "intensity"])
    g.add_argument("--ft-overlap", action="store_true", help="overlap optical and digital time")
    g.add_argument("--fft-efficiency", type=float,
                   help="fraction of the matmul rate digital devices reach on FFT work (default 1.0)")
    g = p.add_argument_group("KV hand-off compression (illustrative; speculative in-transit stage)")
    g.add_argument("--kv-compress", choices=KV_PRESETS)
    g.add_argument("--kv-compress-at", choices=["transit", "endpoint"], default="transit")
    g.add_argument("--kv-keep", type=float, help="freq-keep-k: fraction of frequency bins kept (default 0.5)")
    g.add_argument("--transit-ops-per-byte", type=float, help="in-transit compute budget per line byte")
    g.add_argument("--transit-pj-per-bit", type=float, help="in-transit stage energy per input bit")
    p.add_argument("--ppa", action="store_true", help="also report area, silicon cost and perf/W, /mm², /$")
    return p


def engine_overrides(a) -> dict:
    over = {}
    if a.ft_enob is not None:
        over["enob"] = a.ft_enob
    if a.ft_static_w is not None:
        over["laser_w"] = over["tuning_w"] = a.ft_static_w / 2
    if a.ft_mask_rate is not None:
        over["mask_rate_hz"] = a.ft_mask_rate
    if a.ft_samples_per_s is not None:
        over["samples_per_s"] = a.ft_samples_per_s
    if a.ft_detection is not None:
        over["detection"] = a.ft_detection
    if a.ft_overlap:
        over["overlap"] = True
    return over


def pick_device(key: str | None, over: dict, fft_eff: float | None = None):
    if key is None:
        return None
    dev = ACCELERATORS[key]
    if over and dev.transform is not None:
        dev = replace(dev, transform=replace(dev.transform, **over))
    if fft_eff is not None:
        dev = replace(dev, fft_efficiency=fft_eff)
    return dev


def kv_transit_from_args(a):
    if a.kv_compress is None:
        return None
    c = KV_PRESETS[a.kv_compress]
    if a.kv_keep is not None:
        if not c.fft:
            raise SystemExit("--kv-keep applies to freq-keep-k only")
        c = replace(c, ratio=1 / a.kv_keep)
    kw = {}
    if a.transit_ops_per_byte is not None:
        kw["ops_per_byte"] = a.transit_ops_per_byte
    if a.transit_pj_per_bit is not None:
        kw["pj_per_bit"] = a.transit_pj_per_bit
    return KVTransit(c, where=a.kv_compress_at, **kw)


def config_from_args(a) -> SimConfig:
    link = replace(LINKS[a.link], channels=a.link_channels)
    over = engine_overrides(a)
    model = MODELS[a.model]
    if a.prefill_lm_head is not None:
        model = replace(model, prefill_lm_head=a.prefill_lm_head)
    fe = a.fft_efficiency
    return SimConfig(model=model, device=pick_device(a.device, over, fe),
                     prefill_device=pick_device(a.prefill_device, over, fe),
                     decode_device=pick_device(a.decode_device, over, fe),
                     prefill_devices_per_instance=a.prefill_devices_per_instance,
                     decode_devices_per_instance=a.decode_devices_per_instance,
                     kv_transit=kv_transit_from_args(a),
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
    m = summarise(res)
    if getattr(a, "ppa", False):
        m["ppa"] = ppa_report(cfg, m)
    return m


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
        print("\n\n".join(format_report(m) + (format_ppa(m["ppa"]) if "ppa" in m else "") for m in results))


def format_ppa(p: dict) -> str:
    lines = []
    for role, q in p["pools"].items():
        if q["mm2_per_instance"] is None:
            lines.append(f"\nppa          {role}: {q['device']} has no area model")
        else:
            lines.append(f"\nppa          {role}: {q['instances']}x ({q['devices_per_instance']}x {q['device']}) "
                         f"{q['mm2_per_instance']:,.0f} mm², ${q['usd_per_instance']:,.0f} of silicon per instance")
    if p["total_mm2"] is not None:
        lines.append(f"\n             cluster {p['total_mm2']:,.0f} mm², ${p['total_usd']:,.0f}: "
                     f"{p['perf_per_W']:.2f} tok/J, {p['perf_per_mm2']:.3f} tok/s per mm², "
                     f"{p['perf_per_kusd']:.2f} tok/s per $1000 (illustrative; silicon only)")
    return "".join(lines)


if __name__ == "__main__":
    main()
