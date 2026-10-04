# Disaggregated_Inference_Sim

A [SimPy](https://simpy.readthedocs.io/) discrete-event simulator of
**prefill/decode-disaggregated LLM serving**, built as the companion code for the
[LLM Inference Simulators](https://github.com/BrendanJamesLynskey/LLM_Hub_Inference_Simulators)
presentation series.

The goal is **learnability**: a small, readable, fully tested simulator that
shows how a serious architecture simulator is structured. It has a pluggable cost
model, a discrete-event engine, passive probes, metrics with hot-spot attribution,
Chrome traces, an exact accelerated path, and a test ladder from unit tests to
queueing theory.

* Roofline cost model for prefill and decode (Llama-3-8B/70B; H100, A100 and a
  hypothetical optical-MAC part; NVLink, PCIe, InfiniBand and Ethernet links).
* Prefill pool → shared KV-transfer link → decode pool with continuous batching
  and KV-capacity admission; a colocated (vLLM-v0 style) baseline on the same GPUs.
* TTFT / TPOT / ITL / E2E percentiles, goodput and SLO attainment, utilisation,
  MFU/MBU, a per-stage latency breakdown, and **hot-spot attribution**.
* **Power and energy**: static power plus pJ per FLOP, per HBM byte and per link bit;
  DVFS and per-pool power caps as a third "power roof"; average and peak power,
  joules per token and the static/compute/memory/link energy split.
* Chrome trace-event export for [Perfetto](https://ui.perfetto.dev).
* An **exact fast path** (2.0–2.5× faster, bit-identical results) and experiment
  accelerators: analytic capacity bounds, bisection search, and parallel sweeps.
* **Heterogeneous pools** (added 2026-10-04): a device and device count per pool
  (`--prefill-device h100 --decode-device a100`), the Splitwise idea.
* **FFT-mixing models, an optical transform engine and KV hand-off compression**
  (added 2026-10-04, for the [Fourier Optics for Inference](https://github.com/BrendanJamesLynskey/LLM_Hub_Fourier_Optics_Inference)
  series): Hyena-2, a 1:3 hybrid, block-circulant weights and distilled decode;
  `optical-fft`, a Fourier-optical transform engine co-packaged with a digital part;
  fp8 / fp4 / frequency-domain compression of the KV hand-off, in the link or on the GPU;
  PPA (area, silicon cost, perf/W, perf/mm², perf/$) with FHE_Accelerator_Sim's method.
* A JavaScript port (`web/sim_engine.js`) that runs live in
  [deck 05](https://brendanjameslynskey.github.io/InfSim_05_Disaggregated_Inference/) and
  [FOptInf deck 03](https://brendanjameslynskey.github.io/FOptInf_03_Optical_Prefill_Pools/),
  and is tested to match the Python exactly, every new feature included.
* 130 tests: unit, invariant, analytic (M/D/1, Little's law), behavioural,
  property-based (Hypothesis), power and energy, differential (fast path vs
  baseline, JS vs Python), the cost model pinned against an operator trace
  of the real Llama-3-8B, and (94, added 2026-10-04) the new features: identical pools
  reproduce the homogeneous run bit-exactly, the FFT-mixing op ledger equals the FOptInf
  analysis to the FLOP, a transformer on the transform device equals its digital part,
  the averaging-pass rule, monotone energy in static power, and JS parity on 21 new
  configurations.
* Every number below comes from [`examples/results.md`](examples/results.md),
  written by `examples/results.py`.

> **Cost model corrected on 2026-10-03.** An operator trace of the real model, in
> [Torch_Sim_Frontend](https://github.com/BrendanJamesLynskey/Torch_Sim_Frontend), found two
> errors. Every step was charged the whole input-embedding table (1.05 GB too much per step
> for Llama-3-8B, 6.4% of a batch-1 decode step), where a lookup reads only one row per token.
> And decode attention left out each new token's attention to itself (`ctx + batch`
> positions, not `ctx`). Both are fixed in Python, in the JavaScript port and in the
> [Rust port](https://github.com/BrendanJamesLynskey/Rust_DES_Kernel). Every table here was
> regenerated: decode steps are 1–6% shorter, decode is still memory-bound, and no hot-spot,
> SLO verdict, power-bound flag or J/token ranking changed. `examples/results.md` has the
> before/after figures, rerun on the pre-fix commit.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest                                   # 130 tests, about a minute

disagg-sim                               # one disaggregated run (70B, 4xH100 per instance)
disagg-sim --compare                     # colocated vs disaggregated, same request stream
disagg-sim --compare --sweep-rate 2 3 4 5 6 8
disagg-sim --prefill 2 --rate 6 --link eth-25g   # watch the hot-spot move to the KV link
disagg-sim --trace run.json              # open in https://ui.perfetto.dev
disagg-sim --fast                        # exact accelerated decode path
disagg-sim --dvfs --decode-power-cap 250 # energy savings that cost (almost) no latency
disagg-sim --power-cap 400 --dvfs        # cap everything: watch TTFT pay for it

# heterogeneous pools, FFT-mixing models, the optical transform engine, KV compression
disagg-sim --model llama3-8b --devices-per-instance 1 --prefill-device h100 --decode-device a100 --ppa
disagg-sim --model llama3-8b-hyena-circ --prefill-lm-head last --devices-per-instance 1 \
           --prefill-device optical-fft --ft-enob 11 --ft-mask-rate 20000 --ft-overlap --fft-efficiency 0.0625
disagg-sim --model llama3-8b --devices-per-instance 1 --link eth-25g --rate 14 --kv-compress fp8 --kv-compress-at transit

python examples/benchmark_acceleration.py   # measure every acceleration technique
python examples/results.py               # regenerate examples/results.md (every quoted number)
```

Example report (`disagg-sim --prefill 2 --rate 6 --link eth-25g`):

```
utilisation  prefill-0 47%  prefill-1 13%  decode-0 100%  kv-link 96%
where time goes  prefill 1%  kv_wait 81%  kv_transfer 1%  decode 18%
hot-spot     stage=kv_wait -> kv-link (busy 96%)   busy>90%: decode-0, kv-link
```

---

## What's in the box

| Module | Role |
|--------|------|
| `src/disagg_sim/hardware.py` | `ModelSpec`, `Accelerator`, `Link`, and the roofline `CostModel` (the only place that knows about time) |
| `src/disagg_sim/workload.py` | `Request` and all its timestamps; lognormal lengths; Poisson arrivals; JSON replay |
| `src/disagg_sim/sim.py` | SimPy processes for prefill, decode and colocated instances; the KV link as a `simpy.Resource`; routers; `FastDecodeInstance` |
| `src/disagg_sim/metrics.py` | Percentiles, goodput, utilisation, MFU/MBU, stage breakdown, hot-spot attribution, Little's law |
| `src/disagg_sim/ppa.py` | Area, silicon cost and perf/W, perf/mm², perf/$ per pool (FHE_Accelerator_Sim's method, illustrative) |
| `src/disagg_sim/trace.py` | Chrome trace-event export |
| `src/disagg_sim/search.py` | Analytic capacity bounds, parallel sweeps, bisection for the maximum sustainable load |
| `src/disagg_sim/cli.py` | The `disagg-sim` command |
| `web/sim_engine.js` | The browser port used in the deck |
| `examples/benchmark_acceleration.py` | Speed and exactness of each acceleration technique |
| `examples/results.py` | Regenerates `examples/results.md`, the source of every quoted number (with before/after for the 2026-10-03 correction) |
| `tests/` | The verification ladder |

### Measured acceleration (i7-3770, 8 threads, Python 3.12, SimPy 4.1, power model on)

| Technique | Result |
|-----------|--------|
| One event per batch step (not per token / per layer) | 37.1k events instead of 506k / 3.0M |
| Fast path: incremental state + lazy bookkeeping + macro-steps | 2.0–2.5× faster, bit-identical |
| Probe sampling 5 ms → off | 1.9× faster |
| Parallel sweep, 8 processes | 1.6× (short runs; overhead-bound) |
| Analytic bracket + bisection vs 40-point grid | 7 runs instead of 40, 6.5× less time |

### Power and energy (Llama-3-70B, 4×H100 per instance, 4 req/s)

| Configuration | TTFT p99 | TPOT p99 | SLO met | Avg power | J / token |
|---------------|----------|----------|---------|-----------|-----------|
| Colocated, 2 instances | 630 ms | 25.9 ms | 97.8% | 2,993 W | 3.09 |
| Disaggregated 1P1D | 854 ms | 15.2 ms | 99.7% | 2,687 W | 2.77 |
| … `--dvfs` | 854 ms | 15.2 ms | 99.7% | 2,570 W | 2.65 |
| … `--dvfs --decode-power-cap 250` | 854 ms | 17.2 ms | 99.7% | 2,503 W | 2.59 |
| 1P1D `--power-cap 400 --dvfs` | 1,240 ms | 15.2 ms | 95.3% | 2,190 W | 2.26 |
| 2P1D `--power-cap 400 --dvfs` | 689 ms | 15.1 ms | 100% | 2,593 W | 2.68 |

The power coefficients in `hardware.py` are **illustrative**, chosen so prefill runs
near the H100's 700 W TDP and decode near 300 W. Calibrate them for real hardware by
regressing measured power (DCGM) on FLOP and byte rates.

### Optical prefill pools and heterogeneous pools (results.md sections 10–15)

Llama-3-8B shape, one device per instance, 1P1D, 8 req/s, 800 requests. Every coefficient of the
optical engine is **illustrative**, and the block-circulant model is **speculative**:

| Configuration | TTFT p99 | TPOT p99 | SLO met | J / token |
|---------------|----------|----------|---------|-----------|
| H100 prefill + H100 decode | 353.6 ms | 8.8 ms | 100.0% | 0.326 |
| H100 prefill + A100 decode | 353.6 ms | 16.8 ms | 100.0% | 0.299 |
| Hyena-2 on H100s | 374.9 ms | 114.4 ms | 0.0% | 0.371 |
| Hyena-2, `optical-fft` prefill (defaults: ENOB 8, 1,031 Hz mask) | 172.0 s | 12.5 ms | 0.0% | 0.660 |
| Circulant, last-token head, on H100s | 5.4 ms | 13.5 ms | 100.0% | 0.185 |
| … same, GPU FFTs at 1/16 of the matmul rate | 40.0 ms | 13.5 ms | 100.0% | 0.210 |
| … `optical-fft` prefill (optimistic: ENOB 11, 20 kHz, overlapped) | 90.5 ms | 13.5 ms | 100.0% | 0.194 |

An optical transform engine does not pay for attention or Hyena models; for the circulant variant the
optimistic engine wins TTFT only if GPU FFTs run below 1/30.9 of the matmul rate (or, at 1/16, with a
mask rewriting at 47,942 Hz). Compressing the KV hand-off helps only where the link is the bottleneck
(GQA KV over 25 GbE at 14 req/s: hand-off p99 9,657.8 ms → 166.6 ms with fp8, SLO 41.0% → 100.0%), and
there the GPU does the same compression nearly free. Explained in
[FOptInf 03](https://brendanjameslynskey.github.io/FOptInf_03_Optical_Prefill_Pools/).

**Ports.** JavaScript: everything, bit-exact. Rust
([Rust_DES_Kernel](https://github.com/BrendanJamesLynskey/Rust_DES_Kernel)): heterogeneous pools only,
bit-exact; the FFT-mixing models, the optical transform engine and KV hand-off compression are
**Python and JS only, not in the Rust port**, and the Rust side rejects them by name.

`examples/results.py --keep-timings` regenerates sections 1–8 and 10–15 and keeps section 9 (wall
clock) from the previous run; `--optical-from-json` re-renders 10–15 from `examples/results_optical.json`.

---

## Modelling assumptions (read before trusting a number)

* A tensor-parallel group is treated as one larger device; **no all-reduce cost**.
* KV is transferred **after** the whole prefill (no layer-wise streaming), FCFS over
  `channels` independent link channels.
* Each sequence reserves its full KV (prompt + output) at admission; no pre-emption,
  swapping, prefix caching or chunked prefill.
* Efficiency factors (MFU/MBU derating) are constants, not shape-dependent.
* Disaggregated TPOT includes the KV transfer and decode queueing (DistServe's definition); TTFT
  does not (the first token leaves the prefill pool).
* The optical transform engine is first-order: noise as ENOB with ideal gain control, passes
  `4^(required − ENOB)`, integer mask rewrites, static lasers and tuning; no crosstalk or drift.
  Relaxed-tiling decode (FFTs at decode) and the accuracy of compressed KV are not modelled.
* DVFS is continuous (compute clock fraction `s ≥ s_min`, energy per op ∝ s²); real parts
  have discrete operating points, voltage floors and transition latency. Static power is
  charged for the whole run (no power gating).

Each of these is an exercise in deck 05, slide 10.

---

## How the measurements are made

The tools and methods this repository measures with are explained, with their overheads, accuracy and pitfalls, in [SimEng 12: Measurement Tools and Methods](https://brendanjameslynskey.github.io/SimEng_12_Measurement_Tools_and_Methods/) and the series glossaries:

* [roofline cost model](https://brendanjameslynskey.github.io/LLM_Hub_Inference_Simulators/#g-roofline)
* [queueing-theory checks (M/D/1)](https://brendanjameslynskey.github.io/LLM_Hub_Inference_Simulators/#g-queueing)
* [hot-spot attribution](https://brendanjameslynskey.github.io/LLM_Hub_Inference_Simulators/#g-hotspot)
* [Perfetto traces](https://brendanjameslynskey.github.io/LLM_Hub_Inference_Simulators/#g-perfetto)
* [calibrating power coefficients with DCGM](https://brendanjameslynskey.github.io/SimEng_12_Measurement_Tools_and_Methods/#card-dcgm)

## Part of

The [LLM Inference Simulators](https://github.com/BrendanJamesLynskey/LLM_Hub_Inference_Simulators)
series on the [LLMs hub](https://github.com/BrendanJamesLynskey/LLMs).
