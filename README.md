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
* **Causal Encoder-Decoder (CED)** (added 2026-10-05, for
  [Modern Architectures 06](https://brendanjameslynskey.github.io/Arch_06_Asymmetric_Causal_Encoder_Decoder/)):
  the encoder-only prefill of DeepSeek-V4.1-Flash ([arXiv:2609.19969](https://arxiv.org/abs/2609.19969), section 2.2).
  `--model llama3-70b-ced` (or `--ced-encoder-layers E` on any attention model): a prompt token runs only the causal
  encoder and the decoder's K/V projections; the last `--ced-replay` (default 128) prompt tokens are replayed through
  the decoder on the prefill instance (`--ced-replay-on prefill`, the paper) or as the decode instance's first step
  (`--ced-replay-on decode`, the asymmetric P/D deployment of
  [SGLang RFC #39963](https://github.com/sgl-project/sglang/issues/39963), where a prefill instance holds only the
  encoder's weights and the first token comes from the decode pool). Off by default: every other run is unchanged.
* **Scheduling and KV-memory levers** (added 2026-10-06, brief 20A1; colocated mode, all off by default):
  `--batch-policy prefill-priority|decode-priority|chunked` with `--max-num-batched-tokens` (Sarathi-Serve's
  chunked prefill, [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)); `--kv-policy oracle|pow2|max|paged`
  with `--kv-block-size` and `--preemption recompute|swap` over `--host-link` (vLLM's PagedAttention and the
  baselines it compares against, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)); `--prefix-caching`, an
  LRU cache of shared prompt segments (SGLang's RadixAttention, [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)),
  fed by `chat_sessions` workloads (shared system prompts, closed-loop multi-turn sessions: `--turns`, `--think`,
  `--system-prompts`, `--system-len`, `--prefix-share`). Validation presets `mistral-7b`, `yi-34b`, `opt-13b`,
  `a100-40g` and `pcie4`. With every lever off, every earlier number is unchanged.
* A JavaScript port (`web/sim_engine.js`) that runs live in
  [deck 05](https://brendanjameslynskey.github.io/InfSim_05_Disaggregated_Inference/) and
  [FOptInf deck 03](https://brendanjameslynskey.github.io/FOptInf_03_Optical_Prefill_Pools/),
  and is tested to match the Python exactly, every new feature included.
* 203 tests: unit, invariant, analytic (M/D/1, Little's law), behavioural,
  property-based (Hypothesis), power and energy, differential (fast path vs
  baseline, JS vs Python), the cost model pinned against an operator trace
  of the real Llama-3-8B, and (94, added 2026-10-04) the new features: identical pools
  reproduce the homogeneous run bit-exactly, the FFT-mixing op ledger equals the FOptInf
  analysis to the FLOP, a transformer on the transform device equals its digital part,
  the averaging-pass rule, monotone energy in static power, and JS parity on 21 new
  configurations; and (24, added 2026-10-05, `tests/test_ced.py`) the CED option: closed-form prefill FLOPs, the
  encoder-only residency, replay settings inert when CED is off (bit-identical runs), every request's stages and
  timestamps in both replay placements, the fast path exact, and JS parity on 10 configurations; and (49, added
  2026-10-06, `tests/test_levers.py`) the scheduling and KV-memory levers: inert settings and the new scheduler at
  default levers bit-identical to the old colocated instance, the mixed step's closed forms (chunks sum to the whole
  prompt plus KV re-reads), memory never over-committed, each paper's ordering, LRU eviction, closed-loop sessions,
  and JS parity on 15 configurations (preemption, swap, prefix hits, evictions and LRU ties all exercised).
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
pytest                                   # 203 tests, about a minute

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

# Causal Encoder-Decoder: prefill runs the encoder only; replay on prefill (the paper) or on decode (SGLang RFC)
disagg-sim --model llama3-70b-ced --prompt 8192 --output 128 --rate 3
disagg-sim --model llama3-70b-ced --ced-replay-on decode --prefill-devices-per-instance 2 --prompt 8192 --output 128 --rate 3

# scheduling and KV memory (colocated): chunked prefill, paged KV with preemption, prefix caching
disagg-sim --mode colocated --model mistral-7b --device a100 --devices-per-instance 1 --prefill 1 --decode 0 \
           --prompt 2666 --prompt-cv 1.17 --output 481 --output-cv 0.59 --rate 2 --batch-policy chunked --max-num-batched-tokens 512
disagg-sim --mode colocated --model opt-13b --device a100-40g --devices-per-instance 1 --prefill 1 --decode 0 \
           --prompt 161 --prompt-cv 1 --output 338 --output-cv 1 --rate 6 --kv-policy paged --preemption swap --host-link pcie4
disagg-sim --mode colocated --model mistral-7b --device a100 --devices-per-instance 1 --prefill 1 --decode 0 \
           --prompt 384 --prompt-cv 0.19 --output 6 --output-cv 0.19 --turns 4 --rate 4 --prefix-caching

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
| `src/disagg_sim/workload.py` | `Request` and all its timestamps; lognormal lengths; Poisson arrivals; sessions with shared prefixes (`chat_sessions`); JSON replay |
| `src/disagg_sim/sim.py` | SimPy processes for prefill, decode and colocated instances; the KV link as a `simpy.Resource`; routers; `FastDecodeInstance`; `ScheduledInstance` (batching policy, paged KV, preemption) and `PrefixCache` |
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

### Causal Encoder-Decoder (results.md sections 16–18)

Llama-3-70B shape split 40 + 40 (`llama3-70b-ced`), **illustrative** (DeepSeek-V4.1-Flash is a 552B MoE). A prompt
token touches 34.90B matmul parameters against 69.50B for a generated token (0.502; the paper:
8B vs 16B). One 8,192-token prefill: 564.3 ms decoder-only, 288.3 ms CED, 283.5 ms encoder only. Highest request rate with
90% of requests inside both SLOs, best split of 6 instances of 4×H100 (TTFT SLO = 5× the decoder-only unloaded
prefill; TPOT 25 ms; the prompt:output ratios are illustrative stand-ins for agent loops):

| Prompt : output | Decoder-only req/s | CED, replay on prefill | gain | CED, replay on decode | gain |
|---|---|---|---|---|---|
| 2,048 : 512 | 25.72 (4P2D) | 39.20 (3P3D) | 1.52× | 41.07 (3P3D) | 1.60× |
| 4,096 : 256 | 13.10 (4P2D) | 27.01 (4P2D) | 2.06× | 27.90 (4P2D) | 2.13× |
| 8,192 : 128 | 7.89 (5P1D) | 13.23 (4P2D) | 1.68× | 13.23 (4P2D) | 1.68× |
| 16,384 : 64 | 3.81 (5P1D) | 6.89 (5P1D) | 1.81× | 8.17 (5P1D) | 2.14× |

With the replay on decode a prefill instance holds 71.9 GB of weights instead of 141.1 GB: on two H100s that leaves
KV room for 220,048 tokens instead of 8,835. Gains above 2× are queueing (a fixed TTFT SLO), not FLOPs.

### Scheduling and KV-memory levers (results.md sections 19–22)

Validated against their papers, colocated instances; section 22 lists what reproduces and what does not.

* **Chunked prefill** (Mistral-7B on 1×A100, ShareGPT-like lengths, 2.02 req/s): ITL p99 246.1 ms with
  prefill-priority, 51.1 ms with a 512-token budget, 10.0 ms with decode-priority (whose TTFT p50 is 496,500 ms);
  TTFT p50 249.4 ms against 268.3 ms chunked. Capacity under the strict 100 ms TBT SLO: 1.31 → 4.25 req/s (3.24×;
  the paper reports 2.6–3.5×); under the relaxed 0.5 s SLO 1.68×. Chunking costs only 0.4% here against the paper's
  measured ~25% at 512 tokens, so the model never prefers the larger relaxed-SLO budget the paper chose.
* **Paged KV** (OPT-13B on one A100-40GB, ShareGPT means, 6 req/s): 27.04 requests batched against 16.98 with exact
  reservation and 4.99 with 2,048-token reservation; 97.3% of allocated KV holds tokens (paper 96.3%), 20.3% for
  max-length reservation (paper 20.4%). Sustainable rate 2.16× exact reservation (paper 1.7–2.7×). Swap costs
  0.037 s/GB with 1-token blocks against 0.031 with 64.
* **Prefix caching** (Mistral-7B on 1×A100): 4-turn chat throughput 2.48× with 4–8-token outputs and 1.27× with
  256–512 (paper: noticeable against almost none); a 1,024-token shared system prompt gives a 66.8% hit rate,
  3.19× lower TTFT p50 and 2.21× throughput; longer think times (reuse distance) cut an LRU cache's hit rate from
  68.8% to 23.1% when KV memory is tight.

All coefficients are the simulator's illustrative roofline ones; the papers measured real systems on other models
and traces, so only orderings and ranges are compared.

**Ports.** JavaScript: everything, bit-exact. Rust
([Rust_DES_Kernel](https://github.com/BrendanJamesLynskey/Rust_DES_Kernel)): heterogeneous pools only,
bit-exact; the FFT-mixing models, the optical transform engine, KV hand-off compression and the CED option
are **Python and JS only, not in the Rust port**; the Rust side rejects the first three by name and the
`*-ced` models as unknown models. The scheduling and KV-memory levers (sections 19–21) are also **Python and JS
only**: the JS port is bit-exact on every one of them, and Rust_DES_Kernel rejects a configuration that turns any of
them on ("... are Python and JS only, not in the Rust port"), closed-loop sessions and the validation presets
included.

`examples/results.py --keep-timings` regenerates sections 1–8 and 10–22 and keeps section 9 (wall
clock) from the previous run; `--optical-from-json` re-renders 10–15 from `examples/results_optical.json`,
`--ced-from-json` re-renders 16–18 from `examples/results_ced.json` and `--levers-from-json` re-renders 19–22
from `examples/results_levers.json`.

---

## Modelling assumptions (read before trusting a number)

* A tensor-parallel group is treated as one larger device; **no all-reduce cost**.
* KV is transferred **after** the whole prefill (no layer-wise streaming), FCFS over
  `channels` independent link channels.
* By default each sequence reserves its full KV (prompt + output) at admission, with no
  pre-emption, swapping, prefix caching or chunked prefill. The brief-20A1 levers add those to
  **colocated** instances only (disaggregated pools keep whole prompts and reserved KV). They are
  roofline-priced: chunking costs only its weight and KV re-reads plus the step overhead (no kernel
  or tile-quantisation penalty, so chunking overhead is far below Sarathi-Serve's measured ~25% at
  512 tokens); paged attention has no kernel slowdown; swapping costs blocks × link latency + bytes /
  bandwidth, serialised before the next step; a cached segment rounds up to whole KV blocks; prefix
  caching with swap preemption is not modelled.
* Efficiency factors (MFU/MBU derating) are constants, not shape-dependent.
* Disaggregated TPOT includes the KV transfer and decode queueing (DistServe's definition); TTFT
  does not (the first token leaves the prefill pool), except under CED with the replay on decode, where
  the first token comes from the decode pool and TTFT includes the hand-off.
* The CED models are dense Llama-3 shapes split half and half (DeepSeek-V4.1-Flash is a MoE with
  compressed sparse attention). A replayed token is charged a whole decoder layer (its local K/V
  included, standing in for the paper's sliding-window branch), and CED does not change the KV size:
  the paper's KV savings come from cross-layer sharing and FP4, which are not modelled.
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

## Licence

MIT — see [`LICENSE`](LICENSE).
