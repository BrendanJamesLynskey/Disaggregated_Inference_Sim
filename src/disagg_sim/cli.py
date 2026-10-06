"""Command-line front end.

    disagg-sim                                   # one disaggregated run, report to stdout
    disagg-sim --compare                         # colocated vs disaggregated, same GPUs
    disagg-sim --sweep-rate 2 4 8 16             # load sweep: where does goodput collapse?
    disagg-sim --trace run.json                  # Perfetto / chrome://tracing timeline
    disagg-sim --model llama3-8b --devices-per-instance 1 --prefill-device h100 --decode-device a100
    disagg-sim --model llama3-8b-hyena-circ --devices-per-instance 1 --prefill-device optical-fft --decode-device h100
    disagg-sim --model llama3-8b --devices-per-instance 1 --link eth-25g --kv-compress fp8 --kv-compress-at transit
    disagg-sim --model llama3-70b-ced --ced-replay-on decode --prefill-devices-per-instance 2
    disagg-sim --mode colocated --model llama3-8b --devices-per-instance 1 --batch-policy chunked --max-num-batched-tokens 512
    disagg-sim --mode colocated --model opt-13b --device a100-40g --devices-per-instance 1 --kv-policy paged --preemption swap
    disagg-sim --mode colocated --model llama3-8b --devices-per-instance 1 --prefix-caching --turns 4 --think 2 --system-prompts 4 --system-len 1024
    disagg-sim --model opt-13b --device a100-40g --devices-per-instance 1 --kv-policy paged --preemption swap
    disagg-sim --mode colocated --model llama3-70b --tp 4 --weight-format fp8 --compute-format fp8 --workload chat
    disagg-sim --mode colocated --model llama3-8b --devices-per-instance 1 --speculative mtp --gamma 3 --alpha 0.7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .hardware import ACCELERATORS, KV_PRESETS, LINKS, MODELS, QUANT_FORMATS, KVTransit, Parallel
from .metrics import format_report, summarise
from .ppa import ppa_report
from .sim import SimConfig, simulate
from .speculative import Speculative
from .trace import write_trace
from .workload import WORKLOADS, LengthDist, chat_sessions, poisson_workload


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
    p.add_argument("--ttft-slo", type=float, help="seconds (default 1.0, or the --workload preset's)")
    p.add_argument("--tpot-slo", type=float, help="seconds per token (default 0.025, or the --workload preset's)")
    p.add_argument("--workload", choices=WORKLOADS,
                   help="a named workload: its lengths, sessions and SLOs replace --prompt/--output/--turns/...")
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
    g = p.add_argument_group("Causal Encoder-Decoder (DeepSeek-V4.1-Flash, arXiv:2609.19969; *-ced models)")
    g.add_argument("--ced-encoder-layers", type=int, metavar="E",
                   help="make the bottom E layers a causal encoder (0: off; the *-ced models use half)")
    g.add_argument("--ced-replay", type=int, metavar="W", help="prompt tokens replayed through the decoder (default 128)")
    g.add_argument("--ced-replay-on", choices=["prefill", "decode"],
                   help="replay on the prefill instance (the paper) or as decode's first step (SGLang RFC #39963)")
    g = p.add_argument_group("scheduling and KV memory (colocated mode; brief 20A1)")
    g.add_argument("--batch-policy", choices=["prefill-priority", "decode-priority", "chunked"],
                   default="prefill-priority", help="chunked = Sarathi-Serve stall-free batching (arXiv:2403.02310)")
    g.add_argument("--max-num-batched-tokens", type=int, metavar="N",
                   help="token budget of one step (default: the prefill budget, 8192)")
    g.add_argument("--kv-policy", choices=["oracle", "pow2", "max", "paged"], default="oracle",
                   help="reserve prompt + output (default), over-reserve (pow2, max) or paged blocks (vLLM, arXiv:2309.06180)")
    g.add_argument("--kv-block-size", type=int, default=16)
    g.add_argument("--max-seq-len", type=int, default=2048, help="the 'max' policy's reservation")
    g.add_argument("--preemption", choices=["recompute", "swap"], default="recompute")
    g.add_argument("--host-link", choices=LINKS, default="pcie5", help="swap path to host memory")
    g.add_argument("--prefix-caching", action="store_true",
                   help="LRU cache of shared prompt segments (SGLang RadixAttention, arXiv:2312.07104)")
    g = p.add_argument_group("sessions with shared prefixes (the workload; --prompt is each turn's new input)")
    g.add_argument("--turns", type=int, default=1, help="turns per session (closed loop)")
    g.add_argument("--think", type=float, default=0.0, help="mean think time between turns, seconds")
    g.add_argument("--system-prompts", type=int, default=0, help="distinct shared system prompts")
    g.add_argument("--system-len", type=int, default=0, help="system prompt length, tokens")
    g.add_argument("--prefix-share", type=float, default=1.0, help="fraction of sessions with a system prompt")
    g = p.add_argument_group("parallelism inside an instance (brief 20A2; devices per instance = tp x pp)")
    g.add_argument("--tp", type=int, help="tensor-parallel GPUs per pipeline stage")
    g.add_argument("--pp", type=int, help="pipeline stages")
    g.add_argument("--ep", type=int, help="expert parallelism, 1 or tp (MoE models)")
    g.add_argument("--microbatches", type=int, help="pipeline micro-batches per step (default pp)")
    g.add_argument("--expert-imbalance", type=float, help="busiest GPU's expert load over the mean (default 1)")
    g.add_argument("--scale-up-link", choices=LINKS, help="link inside an instance (default: the device's NVLink)")
    for pool in ("prefill", "decode"):
        g.add_argument(f"--{pool}-tp", type=int, help=f"tensor parallelism of the {pool} pool")
        g.add_argument(f"--{pool}-pp", type=int, help=f"pipeline stages of the {pool} pool")
    g = p.add_argument_group("quantisation as a performance lever (brief 20A2; accuracy is not simulated)")
    g.add_argument("--weight-format", choices=QUANT_FORMATS, default="bf16")
    g.add_argument("--kv-format", choices=QUANT_FORMATS, default="bf16")
    g.add_argument("--compute-format", choices=QUANT_FORMATS, default="bf16",
                   help="bf16 (weight-only) or the weight format on a device with units for it (W8A8, W4A4)")
    g = p.add_argument_group("speculative decoding (brief 20A2; Leviathan et al., arXiv:2211.17192)")
    g.add_argument("--speculative", metavar="DRAFT", choices=["mtp", *MODELS],
                   help="draft model: 'mtp' (one extra target layer) or a model with the target's vocabulary")
    g.add_argument("--gamma", type=int, default=3, help="drafted tokens per verify pass")
    g.add_argument("--alpha", type=float, default=0.7, help="acceptance rate per drafted token")
    g.add_argument("--spec-seed", type=int, default=0)
    p.add_argument("--ppa", action="store_true", help="also report area, silicon cost and perf/W, /mm², /$")
    return p


def parallel_from_args(a, tp, pp):
    if tp is None and pp is None and (a is None or (a.ep is None and a.microbatches is None)):
        return None
    return Parallel(tp=tp or 1, pp=pp or 1, ep=(a.ep or 1) if a is not None else 1,
                    microbatches=a.microbatches if a is not None else None,
                    expert_imbalance=(a.expert_imbalance or 1.0) if a is not None else 1.0,
                    link=a.scale_up_link if a is not None else None)


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
    ced = {k: v for k, v in (("ced_encoder_layers", a.ced_encoder_layers), ("ced_replay", a.ced_replay),
                             ("ced_replay_on", a.ced_replay_on)) if v is not None}
    if ced:
        model = replace(model, **ced)
    fe = a.fft_efficiency
    wp = WORKLOADS[a.workload] if getattr(a, "workload", None) else None
    ttft = a.ttft_slo if a.ttft_slo is not None else (wp.ttft_slo if wp else 1.0)
    tpot = a.tpot_slo if a.tpot_slo is not None else (wp.tpot_slo if wp else 0.025)
    par = parallel_from_args(a, a.tp, a.pp)
    pools = {}
    for pool in ("prefill", "decode"):
        t, q = getattr(a, f"{pool}_tp"), getattr(a, f"{pool}_pp")
        if t is not None or q is not None:
            pools[f"{pool}_parallel"] = Parallel(tp=t or 1, pp=q or 1, ep=a.ep or 1, microbatches=a.microbatches,
                                                 expert_imbalance=a.expert_imbalance or 1.0, link=a.scale_up_link)
    spec = Speculative(a.speculative, a.gamma, a.alpha, a.spec_seed) if a.speculative else None
    return SimConfig(model=model, device=pick_device(a.device, over, fe),
                     prefill_device=pick_device(a.prefill_device, over, fe),
                     decode_device=pick_device(a.decode_device, over, fe),
                     prefill_devices_per_instance=a.prefill_devices_per_instance,
                     decode_devices_per_instance=a.decode_devices_per_instance,
                     kv_transit=kv_transit_from_args(a),
                     devices_per_instance=a.devices_per_instance, mode=a.mode,
                     n_prefill=a.prefill, n_decode=a.decode,
                     n_colocated=a.prefill + a.decode, link=link,
                     ttft_slo=ttft, tpot_slo=tpot, trace=bool(a.trace),
                     fast_forward=a.fast, power_cap_w=a.power_cap,
                     prefill_power_cap_w=a.prefill_power_cap, decode_power_cap_w=a.decode_power_cap,
                     dvfs=a.dvfs, batch_policy=a.batch_policy, max_num_batched_tokens=a.max_num_batched_tokens,
                     kv_policy=a.kv_policy, kv_block_size=a.kv_block_size, max_seq_len=a.max_seq_len,
                     preemption=a.preemption, host_link=LINKS[a.host_link], prefix_caching=a.prefix_caching,
                     parallel=par, weight_format=a.weight_format, kv_format=a.kv_format,
                     compute_format=a.compute_format, speculative=spec, **pools)


def workload_from_args(a, rate: float):
    if getattr(a, "workload", None):
        return WORKLOADS[a.workload].generate(rate, a.requests, seed=a.seed)
    prompt, output = LengthDist(a.prompt, a.prompt_cv), LengthDist(a.output, a.output_cv)
    if getattr(a, "turns", 1) > 1 or getattr(a, "system_prompts", 0):
        return chat_sessions(rate, a.requests, prompt, output, seed=a.seed, turns=a.turns, think=a.think,
                             system_prompts=a.system_prompts, system_len=a.system_len, share=a.prefix_share)
    return poisson_workload(rate, a.requests, prompt, output, seed=a.seed)


def run_once(cfg: SimConfig, a, rate: float) -> dict:
    res = simulate(cfg, workload_from_args(a, rate))
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
