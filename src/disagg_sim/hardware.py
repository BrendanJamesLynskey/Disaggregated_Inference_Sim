"""Hardware and model descriptions, plus the roofline cost model.

Everything the simulator knows about *time* comes from this file. The
discrete-event engine (``sim.py``) only asks two questions:

* how long does this batch step take on this instance?   -> ``CostModel``
* how long does this KV-cache transfer take on the link? -> ``Link``

Keeping the cost model separate from the event engine is the single most
important structural decision in any architecture simulator: it lets you swap
a 10-line roofline for a calibrated table, a cycle-level model, or real
silicon measurements without touching the scheduling logic.

Added 2026-10-04: ``KVCompression``, compressing the KV hand-off on the prefill GPU before it is
sent (an elementwise pass over the KV: HBM read + write, plus the FLOPs). Illustrative.

Added 2026-10-05 (brief 11): the Causal Encoder-Decoder (CED) option of DeepSeek-V4.1-Flash
(arXiv:2609.19969, section 2.2). ``ModelSpec.ced_encoder_layers = E > 0`` makes the bottom E
layers a causal encoder; every decoder layer projects its KV from the last encoder hidden state
with its own weights, so a prompt token needs only the encoder plus the decoder's K/V
projections. The last ``ced_replay`` prompt tokens (the paper's n_win = 128) still go through
the decoder: on the prefill instance (``ced_replay_on="prefill"``, the paper's Decoder SWA
Bounded Replay) or as the decode instance's first step (``"decode"``, the asymmetric P/D
deployment of SGLang RFC #39963, where a prefill instance holds only the encoder's weights).
Off by default: every other model is unchanged. Decode always runs the whole model.

Added 2026-10-06 (brief 20A1): ``CostModel.step_mixed``, one forward pass that mixes decode rows
with prefill chunks that start after ``p0`` tokens already in the KV cache. It prices chunked
prefill (Sarathi-Serve, arXiv:2403.02310), prefill after a prefix-cache hit (SGLang, arXiv:2312.07104)
and the recompute of a preempted request (vLLM, arXiv:2309.06180); the schedulers are in ``sim.py``.
Validation presets for those papers: Mistral-7B, Yi-34B and OPT-13B shapes (from their Hugging Face
configs), an A100 40 GB and a PCIe Gen4 x16 host link. Nothing existing changes.

Added 2026-10-06 (brief 20A2), all off by default:

* Mixture-of-experts shapes (``ModelSpec.n_experts`` / ``top_k``; Mixtral-8x7B): a step's FLOPs use the active
  experts, its weight traffic the experts its tokens are expected to touch, E (1 - (1 - k/E)^tokens) per layer
  under uniform routing, so small decode batches read few experts and prefill reads them all.
* ``Parallel``: tensor, pipeline and expert parallelism inside an instance. Ring all-reduces after attention and
  the MLP (Megatron-LM, arXiv:1909.08053), GPipe micro-batches and bubbles (arXiv:1811.06965), all-to-all dispatch
  and combine with an expert load imbalance (GShard, arXiv:2006.16668), on the device's scale-up link; memory is
  checked per GPU, stage by stage.
* Quantisation as a performance lever: ``QUANT_FORMATS`` (bytes per value, block scales included) for weights and
  the KV cache, and the dense tensor-core rate of a native compute format (``Accelerator.native_formats``, from
  NVIDIA's datasheets). Accuracy is not simulated.
* ``CostModel.step_spec``: the target model verifying gamma drafted tokens per row (speculative decoding); the
  draft's weights and KV cache share the instance's memory (``draft``). The schedule is in ``sim.py``.
* Presets: H200 and B200 (datasheet figures; B200's power coefficients illustrative), NVLink 3 and 5 links,
  Mixtral-8x7B, and Llama-3.2-1B as a draft for the Llama-3 family (same 128,256-token vocabulary).

Added 2026-10-14 (brief 34): a GB200 preset (``gb200``), one Blackwell GPU of a GB200 superchip in an NVL72 rack
(NVIDIA's GB200 NVL72 page: 186 GB at 8 TB/s, 2.5 PFLOPS dense BF16, FP8 and FP4 at 2x and 4x; up to 1,200 W;
power coefficients illustrative, as for B200). The rack's 72-GPU NVLink domain is not modelled.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from functools import cached_property

GB = 1e9
TB = 1e12

def ipow(x: float, n: int) -> float:
    """x ** n for an integer n >= 0 by binary exponentiation: the same multiplications in every language
    (a library ``pow`` is not guaranteed to round identically in Python and JavaScript)."""
    r = 1.0
    while n:
        if n & 1:
            r *= x
        x *= x
        n >>= 1
    return r


# Bytes per value of each storage format, block scales included (brief 20A2). INT4 as AWQ/GPTQ store it: groups of
# 128 weights share an FP16 scale (arXiv:2306.00978), 4 + 16/128 bits. FP4 as the OCP microscaling MXFP4 format:
# blocks of 32 E2M1 values share an 8-bit exponent (arXiv:2310.10537), 4 + 8/32 bits. FP8 and INT8 per-tensor or
# per-channel scales are negligible. Accuracy is not simulated: see the Numerics site.
QUANT_FORMATS = {"bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5 + 2 / 128, "fp4": 0.5 + 1 / 32}


# ─────────────────────────────────────────────────────────────── model ──
@dataclass(frozen=True)
class ModelSpec:
    """A decoder-only transformer, described by its shape alone."""

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab: int
    weight_bytes: float = 2.0   # BF16 weights
    kv_bytes: float = 2.0       # BF16 KV cache
    prefill_lm_head: str = "all"  # "all": every prompt token (as now) | "last": last token only
    # Causal Encoder-Decoder (DeepSeek-V4.1-Flash, arXiv:2609.19969 section 2.2). 0 = off.
    ced_encoder_layers: int = 0   # E: the bottom E layers are the causal encoder (the paper: 20 of 40)
    ced_replay: int = 128         # W: last prompt tokens replayed through the decoder (paper: n_win = 128)
    ced_replay_on: str = "prefill"  # "prefill" (paper section 3.2.2) | "decode" (SGLang RFC #39963)
    # Mixture of experts (brief 20A2). 0 = dense. Every layer's MLP is n_experts SwiGLU experts of width d_ff, of
    # which top_k (plus n_shared_experts always-on ones) run per token; a router of d_model x n_experts picks them.
    n_experts: int = 0
    top_k: int = 0
    n_shared_experts: int = 0

    def __post_init__(self):
        if self.prefill_lm_head not in ("all", "last"):
            raise ValueError(f"unknown prefill_lm_head {self.prefill_lm_head!r}")
        if self.ced_encoder_layers:
            if not 0 < self.ced_encoder_layers < self.n_layers:
                raise ValueError("ced_encoder_layers must leave at least one decoder layer")
            if self.ced_replay < 1:
                raise ValueError("ced_replay must be at least 1 token")
            if self.ced_replay_on not in ("prefill", "decode"):
                raise ValueError(f"unknown ced_replay_on {self.ced_replay_on!r}")
        if self.n_experts:
            if self.ced_encoder_layers:
                raise ValueError("mixture-of-experts layers are modelled without CED only")
            if not 1 <= self.top_k <= self.n_experts:
                raise ValueError("top_k must be between 1 and n_experts")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    # ── Causal Encoder-Decoder (CED) ──
    @property
    def is_ced(self) -> bool:
        return self.ced_encoder_layers > 0

    @cached_property
    def ced_kv_proj_params(self) -> int:
        """The decoder layers' K and V projections (Wk, Wv), applied to the last encoder state."""
        return (self.n_layers - self.ced_encoder_layers) * 2 * self.d_model * self.n_kv_heads * self.head_dim

    @cached_property
    def ced_prompt_params(self) -> int:
        """Matmul parameters one prompt token touches under CED: encoder + decoder K/V
        projections (no LM head). The paper's "activated parameters during prefill"."""
        return self.ced_encoder_layers * self.params_per_layer + self.ced_kv_proj_params

    @cached_property
    def ced_prefill_resident_params(self) -> int:
        """What a prefill-only instance must hold when the replay runs on decode: the input
        embedding, the encoder and the decoder K/V projections (SGLang RFC #39963)."""
        return self.ced_prompt_params + self.vocab * self.d_model

    def ced_replay_lens(self, prompt_lens: list[int]) -> list[int]:
        return [min(s, self.ced_replay) for s in prompt_lens]

    def ced_replay_attention(self, layers: int, s: int, w: int) -> int:
        """QK^T and AV FLOPs of the last w positions of an s-token prompt, over ``layers`` causal
        layers: sum of 4 d c for c = s - w + 1 .. s (integers, so every language agrees)."""
        return 2 * layers * self.d_model * w * (2 * s - w + 1)

    @cached_property
    def params_per_layer(self) -> int:
        """Parameters one token uses in one layer (an MoE layer: the router and its active experts)."""
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        attn = 2 * d * d + 2 * d * kv          # Wq, Wo  +  Wk, Wv (GQA-narrow)
        mlp = 3 * d * self.d_ff                # SwiGLU: gate, up, down
        if self.n_experts:
            return attn + d * self.n_experts + (self.top_k + self.n_shared_experts) * mlp
        return attn + mlp

    # ── mixture of experts (brief 20A2) ──
    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @cached_property
    def expert_params(self) -> int:
        return 3 * self.d_model * self.d_ff

    @cached_property
    def moe_fixed_params_per_layer(self) -> int:
        """What every MoE step reads in each layer whatever the routing: attention, router, shared experts."""
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        return 2 * d * d + 2 * d * kv + d * self.n_experts + self.n_shared_experts * self.expert_params

    def experts_touched(self, tokens: int) -> float:
        """Expected routed experts that ``tokens`` tokens use in one layer, each picking top_k of n_experts uniformly:
        E (1 - (1 - k/E)^tokens)."""
        e = self.n_experts
        return e * (1.0 - ipow(1.0 - self.top_k / e, tokens))

    def expert_flops(self, tokens: int) -> float:
        """Routed-expert FLOPs of ``tokens`` tokens over all layers (what expert parallelism spreads)."""
        return 2.0 * self.n_layers * self.top_k * self.expert_params * tokens

    @cached_property
    def layer_params(self) -> int:
        """Parameters of all layers."""
        return self.n_layers * self.params_per_layer

    @cached_property
    def params(self) -> int:
        """Total parameters (untied input embedding and LM head; every expert of an MoE model)."""
        if self.n_experts:
            return (self.n_layers * (self.moe_fixed_params_per_layer + self.n_experts * self.expert_params)
                    + 2 * self.vocab * self.d_model)
        return self.layer_params + 2 * self.vocab * self.d_model

    @cached_property
    def matmul_params(self) -> int:
        """Parameters that take part in a matmul per token (embedding is a lookup)."""
        return self.layer_params + self.vocab * self.d_model

    @cached_property
    def weight_bytes_total(self) -> float:
        """Bytes the weights occupy in memory (residency). Not what one step reads: see below."""
        return self.params * self.weight_bytes

    @cached_property
    def weight_bytes_streamed(self) -> float:
        """Weights every forward pass reads in full: all layers plus the LM head."""
        return self.matmul_params * self.weight_bytes

    @cached_property
    def embedding_row_bytes(self) -> float:
        return self.d_model * self.weight_bytes

    def weight_bytes_read(self, tokens: int) -> float:
        """Weight traffic of one forward pass over ``tokens`` tokens.

        The layers and the LM head are read in full; the input embedding table is a
        lookup, so only the ``tokens`` rows actually indexed are read. (Before
        2026-10-03 every step was charged the whole table, ``weight_bytes_total``:
        1.05 GB too much per step for Llama-3-8B. An operator trace of the real
        model, in Torch_Sim_Frontend, found it.)

        An MoE model reads, in each layer, its fixed part and the routed experts the tokens are expected
        to touch (``experts_touched``), plus the LM head.
        """
        if self.n_experts:
            per_layer = self.moe_fixed_params_per_layer + self.experts_touched(tokens) * self.expert_params
            return ((self.n_layers * per_layer + self.vocab * self.d_model) * self.weight_bytes
                    + tokens * self.embedding_row_bytes)
        return self.weight_bytes_streamed + tokens * self.embedding_row_bytes

    @cached_property
    def kv_bytes_per_token(self) -> float:
        """K and V, for every layer, for one token."""
        return (2 * self.n_layers * self.n_kv_heads * self.head_dim) * self.kv_bytes

    def handoff_bytes(self, prompt_len: int) -> float:
        """What prefill hands to decode over the KV link: the prompt's KV."""
        return prompt_len * self.kv_bytes_per_token


LLAMA3_8B = ModelSpec("Llama-3-8B", n_layers=32, d_model=4096, n_heads=32,
                      n_kv_heads=8, d_ff=14336, vocab=128256)
LLAMA3_70B = ModelSpec("Llama-3-70B", n_layers=80, d_model=8192, n_heads=64,
                       n_kv_heads=8, d_ff=28672, vocab=128256)

# CED proxies (brief 11): the same dense shapes split half encoder, half decoder, as
# DeepSeek-V4.1-Flash splits its 40 layers 20/20 (arXiv:2609.19969, section 4.2.1). Illustrative:
# that model is a 552B MoE with compressed sparse attention, not a dense Llama.
LLAMA3_8B_CED = replace(LLAMA3_8B, name="Llama-3-8B-shape CED 16+16", ced_encoder_layers=16)
LLAMA3_70B_CED = replace(LLAMA3_70B, name="Llama-3-70B-shape CED 40+40", ced_encoder_layers=40)

# Validation shapes for brief 20A1 (from each model's Hugging Face config.json). Mistral-7B and Yi-34B are
# Sarathi-Serve's models; OPT-13B is vLLM's. Approximations, stated: Mistral-7B's 4,096-token sliding window
# is ignored (full attention); OPT's two-matrix GELU MLP (2 x 5120 x 20480 parameters) is written as this
# class's three-matrix MLP with d_ff = 13653 (209.7M parameters per layer either way, within 0.003%); OPT's
# tied embedding is counted twice like every model here (26.2 GB of weights; vLLM's Table 1: 26 GB).
MISTRAL_7B = ModelSpec("Mistral-7B", n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8, d_ff=14336, vocab=32000)
YI_34B = ModelSpec("Yi-34B", n_layers=60, d_model=7168, n_heads=56, n_kv_heads=8, d_ff=20480, vocab=64000)
OPT_13B = ModelSpec("OPT-13B", n_layers=40, d_model=5120, n_heads=40, n_kv_heads=40, d_ff=13653, vocab=50272)

# Brief 20A2 (Hugging Face config.json): Mixtral-8x7B, 8 experts of width 14,336, top-2 (46.7B parameters, 12.9B
# active; arXiv:2401.04088); Llama-3.2-1B, a draft for the Llama-3 models (same 128,256-token vocabulary; its tied
# embedding is counted twice like every model here, 0.5B parameters too many).
MIXTRAL_8X7B = ModelSpec("Mixtral-8x7B", n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8, d_ff=14336, vocab=32000,
                         n_experts=8, top_k=2)
LLAMA32_1B = ModelSpec("Llama-3.2-1B", n_layers=16, d_model=2048, n_heads=32, n_kv_heads=8, d_ff=8192, vocab=128256)

MODELS = {"llama3-8b": LLAMA3_8B, "llama3-70b": LLAMA3_70B,
          "llama3-8b-ced": LLAMA3_8B_CED, "llama3-70b-ced": LLAMA3_70B_CED,
          "mistral-7b": MISTRAL_7B, "yi-34b": YI_34B, "opt-13b": OPT_13B,
          "mixtral-8x7b": MIXTRAL_8X7B, "llama3.2-1b": LLAMA32_1B}


# ──────────────────────────────────────────────────────────── hardware ──
@dataclass(frozen=True)
class Accelerator:
    """One device. Peak numbers are datasheet values; efficiencies derate them."""

    name: str
    peak_flops: float          # dense BF16 FLOP/s
    mem_bw: float              # HBM bytes/s
    mem_capacity: float        # bytes
    flops_eff: float = 0.55    # achievable fraction of peak FLOP/s (MFU)
    bw_eff: float = 0.80       # achievable fraction of peak bandwidth (MBU)
    # Power model (illustrative coefficients -- calibrate against DCGM / nvidia-smi):
    # power = idle_w + (FLOPs x pj_per_flop + HBM bytes x pj_per_byte) / time
    tdp_w: float = 700.0       # board power limit
    idle_w: float = 100.0      # static power: leakage, clocks, fans' share, HBM refresh
    pj_per_flop: float = 1.0   # dynamic energy per FLOP, incl. on-chip SRAM/register traffic
    pj_per_byte: float = 60.0  # dynamic energy per HBM byte, incl. controller, PHY, on-chip moves
    # Brief 20A2. Dense tensor-core rate of each native low-precision compute format as a multiple of BF16 (from the
    # datasheet; a format not listed runs weight-only: dequantised to BF16 before the multiply). The scale-up link
    # (a LINKS key) carries tensor-, pipeline- and expert-parallel traffic inside an instance.
    native_formats: tuple = ()
    scale_up: str = "nvlink4"

    def format_speedup(self, fmt: str) -> float | None:
        """The compute-rate multiple of a compute format: 1.0 for BF16, None if the device has no such units."""
        if fmt == "bf16":
            return 1.0
        return dict(self.native_formats).get(fmt)

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) where compute and memory balance."""
        return (self.peak_flops * self.flops_eff) / (self.mem_bw * self.bw_eff)


# native_formats from NVIDIA's datasheets (dense; the sheets quote sparse figures at twice these): H100/H200 FP8 and
# INT8 1,979 TFLOPS/TOPS against BF16 989; A100 INT8 624 TOPS against BF16 312 (no FP8 units).
H100_SXM = Accelerator("H100-SXM", peak_flops=989 * TB, mem_bw=3.35 * TB, mem_capacity=80 * GB,
                       native_formats=(("fp8", 2.0), ("int8", 2.0)))
A100_SXM = Accelerator("A100-SXM", peak_flops=312 * TB, mem_bw=2.039 * TB, mem_capacity=80 * GB,
                       tdp_w=400.0, idle_w=60.0, pj_per_flop=1.6, pj_per_byte=70.0,
                       native_formats=(("int8", 2.0),), scale_up="nvlink3")
# The 40 GB A100 (HBM2, 1,555 GB/s; NVIDIA A100 datasheet): vLLM's OPT-13B server (arXiv:2309.06180, Table 1).
A100_40G = replace(A100_SXM, name="A100-SXM-40GB", mem_bw=1.555 * TB, mem_capacity=40 * GB)

# Brief 20A2 hardware presets. H200 SXM (NVIDIA H200 page): the H100's compute, 141 GB of HBM3e at 4.8 TB/s, 700 W.
# B200 (NVIDIA DGX B200 page, per GPU of eight): 180 GB, 8 TB/s, FP4 9 PFLOPS and FP8 4.5 PFLOPS dense; BF16 taken as
# half of FP8 (2.25 PFLOPS), NVLink 5 at 900 GB/s each way (14.4 TB/s over 8 GPUs, both directions). Its 1,000 W board
# limit and power coefficients are illustrative (the page quotes about 14.3 kW for the whole system).
H200_SXM = replace(H100_SXM, name="H200-SXM", mem_bw=4.8 * TB, mem_capacity=141 * GB)
B200 = Accelerator("B200", peak_flops=2250 * TB, mem_bw=8 * TB, mem_capacity=180 * GB, tdp_w=1000.0, idle_w=140.0,
                   pj_per_flop=0.6, pj_per_byte=45.0, native_formats=(("fp8", 2.0), ("fp4", 4.0)), scale_up="nvlink5")
# Brief 34. GB200 (NVIDIA GB200 NVL72 page, superchip column halved: one Blackwell GPU of a GB200 superchip in an NVL72
# rack): 186 GB HBM3e (372 GB per superchip) at 8 TB/s (16 TB/s); dense FP4 10 PFLOPS (20), FP8 5 PFLOPS and BF16
# 2.5 PFLOPS (the page states FP8 and BF16 sparse only, 20 and 10 PFLOPS per superchip: halved for dense, then halved
# again per GPU; BF16 is half of FP8, as for B200); NVLink 5 at 900 GB/s each way (3.6 TB/s per superchip, both
# directions). Board limit: Blackwell's "up to 1,200 W" TGP (NVIDIA's Blackwell Ultra blog, Table 2). Power coefficients
# are illustrative, B200's. The NVL72's 72-GPU NVLink domain is not modelled: no width in ``Parallel`` is limited by
# a scale-up domain, on any device (see results.md section 28).
GB200 = replace(B200, name="GB200", peak_flops=2500 * TB, mem_capacity=186 * GB, tdp_w=1200.0)

ACCELERATORS = {"h100": H100_SXM, "a100": A100_SXM, "a100-40g": A100_40G, "h200": H200_SXM, "b200": B200,
                "gb200": GB200}


@dataclass(frozen=True)
class Link:
    """The fabric that carries KV cache from prefill to decode instances."""

    name: str
    bandwidth: float           # bytes/s per channel
    latency: float = 10e-6     # seconds, per transfer
    channels: int = 1          # independent transfers that can proceed at once
    pj_per_bit: float = 10.0   # energy to move one bit end to end (SerDes, NIC, switch)

    def transfer_time(self, nbytes: float) -> float:
        return self.latency + nbytes / self.bandwidth


LINKS = {
    "nvlink4": Link("NVLink 4 (one direction)", bandwidth=450 * GB, latency=5e-6, pj_per_bit=5.0),
    "ib-ndr": Link("InfiniBand NDR 400G", bandwidth=50 * GB, latency=10e-6, pj_per_bit=15.0),
    "pcie5": Link("PCIe Gen5 x16", bandwidth=64 * GB, latency=5e-6, pj_per_bit=6.0),
    # An A100's host link (vLLM's swap-to-CPU path): about 32 GB/s each way; energy as Gen5 (illustrative).
    "pcie4": Link("PCIe Gen4 x16", bandwidth=32 * GB, latency=5e-6, pj_per_bit=6.0),
    "eth-100g": Link("100 GbE", bandwidth=12.5 * GB, latency=20e-6, pj_per_bit=15.0),
    "eth-25g": Link("25 GbE", bandwidth=3.125 * GB, latency=20e-6, pj_per_bit=15.0),
    # An illustrative co-packaged-optics switch link. Round numbers, not a product: multi-Tb/s optical I/O
    # chiplets exist (Wade et al., TeraPHY, IEEE Micro 2020); 1.6 Tb/s and 3 pJ/bit are this simulator's assumptions.
    "cpo-optical": Link("Co-packaged optics switch (illustrative)", bandwidth=200 * GB, latency=5e-6, pj_per_bit=3.0),
    # Scale-up links inside an instance (brief 20A2): an A100's NVLink 3 (600 GB/s both ways) and a B200's NVLink 5.
    "nvlink3": Link("NVLink 3 (one direction)", bandwidth=300 * GB, latency=5e-6, pj_per_bit=5.0),
    "nvlink5": Link("NVLink 5 (one direction)", bandwidth=900 * GB, latency=5e-6, pj_per_bit=5.0),
}


# ─────────────────────────────────────────── KV hand-off compression ──
@dataclass(frozen=True)
class KVCompression:
    """A compression of the KV hand-off (BF16 in), run on the prefill GPU before the hand-off is sent.
    ``ratio`` = bytes in / bytes out; ``ops_per_value``: elementwise work per BF16 value (amax, scale, round)."""

    name: str
    ratio: float
    ops_per_value: float

    def ops(self, nbytes: float, kv_bytes: float) -> float:
        """Operations to compress ``nbytes`` of hand-off."""
        values = nbytes / kv_bytes
        return self.ops_per_value * values


# Tolerable KV widths come from independent KV-quantisation work: KIVI (arXiv:2402.02750,
# 2-bit) and KVQuant (arXiv:2401.18079, 3-bit with < 0.1 perplexity loss), so 8- and 4-bit
# are within what they report. The simulator does not model accuracy.
KV_PRESETS = {
    "none": KVCompression("none", 1.0, 0.0),
    "fp8": KVCompression("fp8", 2.0, 1.0),                     # scale + round per value
    "fp4-block": KVCompression("fp4-block", 64 / 17, 2.0),     # E2M1, 32-value blocks + 8-bit scale
}


# ───────────────────────────────────────────── parallelism (brief 20A2) ──
@dataclass(frozen=True)
class Parallel:
    """How an instance's tp x pp GPUs share the model (brief 20A2; first-order, illustrative).

    * ``tp``: tensor parallelism inside each pipeline stage. Two ring all-reduces of the activations per layer, one
      after attention and one after the MLP (Megatron-LM, arXiv:1909.08053), each 2(n-1)/n S / B + 2(n-1) latency
      (the ring algorithm's bandwidth factor; NCCL's performance notes), not overlapped with compute.
    * ``pp``: pipeline stages, ``n_layers / pp`` layers each. A step's tokens are split into ``microbatches``
      (default pp, at most the tokens), and the stages run GPipe's schedule (arXiv:1811.06965): m + pp - 1 slots, so
      the bubble is (pp - 1) / (m + pp - 1); every micro-batch re-reads its stage's weights; one activation hop per
      boundary. Successive steps are not overlapped across stages (no cross-step pipelining).
    * ``ep``: expert parallelism for an MoE model, 1 (each expert split by TP, an all-reduce after it) or ``tp``
      (whole experts per GPU): the MLP all-reduce becomes an all-to-all dispatch and combine moving
      2 tokens k d (n-1)/n activations per layer (GShard, arXiv:2006.16668), and the busiest GPU's expert work is
      ``expert_imbalance`` times the mean.
    * ``link``: the scale-up link (a LINKS key); None means the device's own (``Accelerator.scale_up``).
    """

    tp: int = 1
    pp: int = 1
    ep: int = 1
    microbatches: int | None = None
    expert_imbalance: float = 1.0
    link: str | None = None

    @property
    def gpus(self) -> int:
        return self.tp * self.pp

    def check(self, model: ModelSpec, n_devices: int, where: str = "") -> None:
        """Clear errors for the combinations this model does not cover."""
        at = f"{where}: " if where else ""
        if self.tp < 1 or self.pp < 1 or (self.microbatches is not None and self.microbatches < 1):
            raise ValueError(f"{at}tp, pp and microbatches must be at least 1")
        if self.tp * self.pp != n_devices:
            raise ValueError(f"{at}parallel tp={self.tp} x pp={self.pp} needs {self.tp * self.pp} devices per instance "
                             f"(got {n_devices})")
        if self.ep not in (1, self.tp):
            raise ValueError(f"{at}ep must be 1 (experts split by TP) or equal to tp (whole experts per GPU)")
        if self.ep > 1 and not model.is_moe:
            raise ValueError(f"{at}expert parallelism needs a mixture-of-experts model")
        if self.expert_imbalance < 1.0:
            raise ValueError(f"{at}expert_imbalance is the busiest GPU's load over the mean: at least 1")
        if model.n_layers % self.pp:
            raise ValueError(f"{at}{model.name} has {model.n_layers} layers: not divisible into {self.pp} stages")
        if model.is_ced:
            raise ValueError(f"{at}parallelism is modelled for models without CED only")
        if self.link is not None and self.link not in LINKS:
            raise ValueError(f"{at}unknown scale-up link {self.link!r}")


# ────────────────────────────────────────────────────────── cost model ──
def _cbrt(x: float) -> float:
    return math.copysign(abs(x) ** (1 / 3), x)


@dataclass(frozen=True)
class StepCost:
    flops: float
    bytes: float
    time: float
    bound: str                 # "compute", "memory" or "power"
    energy: float = 0.0        # dynamic joules for this step (static power is added per instance)
    compute_j: float = 0.0
    memory_j: float = 0.0
    # Not fields: ordinary steps construct as fast as before (the simulator's hot path).
    lever = False              # brief 20A2: a LeverStepCost (scale-up traffic, draft passes)
    comm_time = 0.0
    comm_j = 0.0
    draft_time = 0.0


@dataclass(frozen=True)
class LeverStepCost(StepCost):
    """A step with brief-20A2 terms: scale-up link time and energy (parallelism) and draft-model time
    (speculative decoding). ``time`` already includes them; these say how much of it they were."""

    comm_time: float = 0.0
    comm_j: float = 0.0
    draft_time: float = 0.0
    lever = True


@dataclass(frozen=True)
class CostModel:
    """Roofline timing for one forward pass of a batch on one instance.

    An *instance* is ``n_devices`` accelerators running the model with tensor
    parallelism; we treat it as one bigger device and ignore TP all-reduce cost
    (a known optimism, flagged in the README).
    """

    model: ModelSpec
    device: Accelerator
    n_devices: int = 1
    step_overhead: float = 0.5e-3   # scheduler + kernel-launch time per step, seconds
    mem_util: float = 0.9           # fraction of HBM the engine may use
    power_cap_w: float | None = None  # per-device cap; the device's TDP always applies as well
    enforce_tdp: bool = True        # a real part throttles at its board power limit
    dvfs: bool = False              # lower the compute clock on memory-bound steps
    s_min: float = 0.4              # lowest compute clock, as a fraction of nominal
    # A prefill-pool instance. Changes nothing except for a CED model whose replay runs on
    # decode: then it holds only the encoder's weights and its prefill stops at the encoder.
    prefill_only: bool = False
    # Brief 20A2 (defaults change nothing): the compute format's rate multiple (W8A8 on FP8 units: 2.0), tensor /
    # pipeline / expert parallelism inside the instance, and a speculative-decoding draft that shares its memory.
    compute_speedup: float = 1.0
    parallel: Parallel | None = None
    draft: ModelSpec | None = None
    draft_resident_bytes: float = 0.0

    @cached_property
    def encoder_only(self) -> bool:
        m = self.model
        return self.prefill_only and m.is_ced and m.ced_replay_on == "decode"

    @cached_property
    def resident_weight_bytes(self) -> float:
        m = self.model
        if self.encoder_only:
            return m.ced_prefill_resident_params * m.weight_bytes
        return m.weight_bytes_total

    @cached_property
    def flops_rate(self) -> float:
        return self.device.peak_flops * self.device.flops_eff * self.n_devices * self.compute_speedup

    @cached_property
    def byte_rate(self) -> float:
        return self.device.mem_bw * self.device.bw_eff * self.n_devices

    @property
    def kv_capacity_tokens(self) -> int:
        """KV tokens that fit beside the weights."""
        if self.parallel is not None or self.draft is not None:
            return self._kv_capacity_levers()
        free = self.device.mem_capacity * self.n_devices * self.mem_util - self.resident_weight_bytes
        if free <= 0:
            raise ValueError(f"{self.model.name} does not fit on {self.n_devices}x {self.device.name}")
        return int(free // self.model.kv_bytes_per_token)

    def _kv_capacity_levers(self) -> int:
        """Brief 20A2: memory checked per GPU. Each pipeline stage holds its layers' weights (the first also the
        embedding, the last the LM head) split over its tp GPUs, and its layers' share of every token's KV; the
        stage with the least room sets the instance's token capacity. A draft model's weights and KV are added."""
        m, p = self.model, self.parallel
        tp, pp = (p.tp, p.pp) if p is not None else (self.n_devices, 1)
        room = self.device.mem_capacity * self.mem_util          # bytes the engine may use on one GPU
        unit = m.kv_bytes_per_token / pp
        if self.draft is not None:
            unit = unit + self.draft.kv_bytes_per_token
        per_layer = (m.moe_fixed_params_per_layer + m.n_experts * m.expert_params) if m.is_moe else m.params_per_layer
        layers = m.n_layers // pp
        cap = None
        for s in range(pp):
            params = layers * per_layer + (m.vocab * m.d_model if s == 0 else 0) + (m.vocab * m.d_model if s == pp - 1 else 0)
            w = params * m.weight_bytes + self.draft_resident_bytes
            if w / tp > room:
                raise ValueError(f"{m.name} does not fit: {w / tp / 1e9:.1f} GB of weights per GPU in stage {s} of "
                                 f"{tp} x {pp} {self.device.name} ({room / 1e9:.1f} GB usable)")
            c = int((room * tp - w) // unit)
            cap = c if cap is None or c < cap else cap
        return cap

    @cached_property
    def joules_per_flop(self) -> float:
        return self.device.pj_per_flop * 1e-12 / self.compute_speedup

    @cached_property
    def joules_per_byte(self) -> float:
        return self.device.pj_per_byte * 1e-12

    @cached_property
    def idle_w(self) -> float:
        return self.device.idle_w * self.n_devices

    @cached_property
    def dynamic_budget_w(self) -> float | None:
        """Power left for dynamic work under the cap and/or TDP (None: unlimited)."""
        caps = [c for c in (self.power_cap_w, self.device.tdp_w if self.enforce_tdp else None)
                if c is not None]
        if not caps:
            return None
        budget = (min(caps) - self.device.idle_w) * self.n_devices
        if budget <= 0:
            raise ValueError("power cap is below idle power")
        return budget

    def step_time(self, flops: float, nbytes: float) -> tuple[float, float, float, str]:
        """``step_time_raw`` plus the fixed per-step overhead."""
        t, ec, em, bound = self.step_time_raw(flops, nbytes)
        return t + self.step_overhead, ec, em, bound

    def step_time_raw(self, flops: float, nbytes: float) -> tuple[float, float, float, str]:
        """Time and energy of one step under a simple DVFS model (no per-step overhead).

        The compute clock runs at a fraction ``s`` of nominal (memory is unaffected).
        Compute time scales as 1/s and, because voltage tracks frequency, compute
        energy as s^2. So:

            t(s)  = max(tc / s, tm)             E(s) = Ec * s^2 + Em
            P(s)  = idle + E(s) / t(s)          -- increasing in s

        * ``dvfs=True``: a memory-bound step lowers s until compute just keeps up
          with memory (s = tc/tm, floored at ``s_min``): same time, less energy.
        * ``power_cap_w``: s is lowered until P(s) <= cap (solved in closed form);
          if even ``s_min`` is too hot, the step is stretched to fit the budget.

        Returns (seconds, compute joules, memory joules, bound) where bound is
        "compute", "memory" or "power".
        """
        tc, tm = flops / self.flops_rate, nbytes / self.byte_rate
        ec, em = flops * self.joules_per_flop, nbytes * self.joules_per_byte
        s = max(self.s_min, tc / tm) if (self.dvfs and tc < tm) else 1.0
        bound = "compute" if tc >= tm else "memory"
        budget = self.dynamic_budget_w
        if budget is not None and (ec * s * s + em) / max(tc / s, tm) > budget:
            bound = "power"
            # Region where memory still sets the time: (Ec s^2 + Em) / tm <= B
            x = (budget * tm - em) / ec
            if x > 0 and math.sqrt(x) * tm >= tc:
                s = min(s, math.sqrt(x))
            else:
                # Compute sets the time: (Ec s^3 + Em s) / tc = B, a monotone cubic.
                # Cardano for s^3 + p s + q = 0 with p > 0 (one real root).
                p, q = em / ec, -budget * tc / ec
                r = math.sqrt(q * q / 4 + p * p * p / 27)
                s = _cbrt(-q / 2 + r) + _cbrt(-q / 2 - r)
            s = max(s, self.s_min)
        t = max(tc / s, tm)
        ec = ec * s * s
        if budget is not None and (ec + em) / t > budget:      # still too hot at s_min
            t = (ec + em) / budget
        return t, ec, em, bound

    def _time(self, flops: float, nbytes: float) -> StepCost:
        t, ec, em, bound = self.step_time_raw(flops, nbytes)
        return StepCost(flops, nbytes, t + self.step_overhead, bound, ec + em, ec, em)

    def prefill(self, prompt_lens: list[int]) -> StepCost:
        """One prefill step over whole prompts (no chunking)."""
        m = self.model
        tokens = sum(prompt_lens)
        if m.is_ced:
            return self._prefill_ced(prompt_lens, tokens)
        # The LM head is charged for every prompt token, as a plain forward pass computes
        # (HF transformers by default, and Torch_Sim_Frontend's trace of it); serving
        # engines that keep only the last position's logits do less.
        if m.prefill_lm_head == "all":
            flops = 2 * m.matmul_params * tokens
        else:
            flops = 2 * m.layer_params * tokens + 2 * m.vocab * m.d_model * len(prompt_lens)
        # causal attention: QK^T and AV, each 2*d*c FLOPs at position c (diagonal included)
        flops += sum(2 * m.n_layers * m.d_model * s * (s + 1) for s in prompt_lens)
        if self.parallel is not None:
            return self._parallel(flops, m.weight_bytes_read(tokens), tokens * m.kv_bytes_per_token, tokens)
        nbytes = m.weight_bytes_read(tokens) + tokens * m.kv_bytes_per_token
        return self._time(flops, nbytes)

    def _prefill_ced(self, prompt_lens: list[int], tokens: int) -> StepCost:
        """CED prefill (arXiv:2609.19969 section 2.2). Every prompt token runs the E encoder
        layers and the decoder layers' K/V projections from the last encoder state; nothing
        else of the decoder. With the replay on prefill, the last W tokens of each prompt
        then run the decoder layers too (attending to the whole prompt), and only they reach
        the LM head. With the replay on decode, a prefill-only instance stops at the encoder
        and reads only the encoder's weights; the decode instance does the rest
        (``ced_step``). All FLOP counts are integers below 2^53: exact in every language."""
        m = self.model
        enc, dec = m.ced_encoder_layers, m.n_layers - m.ced_encoder_layers
        flops = 2 * m.ced_prompt_params * tokens
        flops += sum(2 * enc * m.d_model * s * (s + 1) for s in prompt_lens)
        if self.encoder_only:
            nbytes = m.ced_prompt_params * m.weight_bytes + tokens * m.embedding_row_bytes
            return self._time(flops, nbytes + tokens * m.kv_bytes_per_token)
        reps = m.ced_replay_lens(prompt_lens)
        rt = sum(reps)
        flops += 2 * dec * m.params_per_layer * rt
        flops += sum(m.ced_replay_attention(dec, s, w) for s, w in zip(prompt_lens, reps))
        flops += 2 * m.vocab * m.d_model * (rt if m.prefill_lm_head == "all" else len(prompt_lens))
        nbytes = m.weight_bytes_read(tokens) + tokens * m.kv_bytes_per_token
        return self._time(flops, nbytes)

    def ced_step(self, ctx: int, batch: int, replay: list[tuple[int, int]]) -> StepCost:
        """A decode-pool step under CED with the replay on decode (SGLang RFC #39963): ``batch``
        running sequences decode one token each over ``ctx`` cached positions (exactly as
        ``decode_sum``), while each newly admitted request runs its last w prompt tokens
        through the whole model (a bounded prefill) and emits its first token. ``replay`` is
        a list of (prompt length s, w)."""
        m = self.model
        rt = sum(w for _, w in replay)
        flops = 2 * m.matmul_params * batch + 4 * m.n_layers * m.d_model * (ctx + batch)
        flops += 2 * m.layer_params * rt
        flops += 2 * m.vocab * m.d_model * (rt if m.prefill_lm_head == "all" else len(replay))
        flops += sum(m.ced_replay_attention(m.n_layers, s, w) for s, w in replay)
        nbytes = m.weight_bytes_read(batch + rt) + (ctx + batch) * m.kv_bytes_per_token
        nbytes += sum(s for s, _ in replay) * m.kv_bytes_per_token
        return self._time(flops, nbytes)

    def decode(self, context_lens: list[int]) -> StepCost:
        """One decode step: every running sequence produces one token."""
        return self.decode_sum(sum(context_lens), len(context_lens))

    def decode_sum(self, ctx: int, batch: int) -> StepCost:
        """Decode cost depends only on the batch size and the *total* context, so a
        caller that keeps a running sum avoids an O(batch) pass per step.

        ``ctx`` is the cached context; each new token attends to it *and to itself*,
        hence ``ctx + batch`` positions (the self term was missing before 2026-10-03).
        """
        m = self.model
        flops = 2 * m.matmul_params * batch + 4 * m.n_layers * m.d_model * (ctx + batch)
        if self.parallel is not None:
            return self._parallel(flops, m.weight_bytes_read(batch), (ctx + batch) * m.kv_bytes_per_token, batch)
        nbytes = m.weight_bytes_read(batch) + (ctx + batch) * m.kv_bytes_per_token
        return self._time(flops, nbytes)

    def step_mixed(self, ctx: int, batch: int, chunks) -> StepCost:
        """One forward pass that mixes ``batch`` decode rows over ``ctx`` cached positions (as
        ``decode_sum``) with prefill chunks ``(p0, c, last)``: c new prompt tokens after p0 already
        in the KV cache (earlier chunks, a prefix-cache hit, or nothing). ``last`` marks a chunk that
        ends its prompt (it reaches the LM head under ``prefill_lm_head="last"``).

        A chunk's attention is the tail of the causal sum, 4 L d j for j = p0+1 .. p0+c, and its KV
        traffic reads the p0 cached positions and writes c (Sarathi-Serve, arXiv:2403.02310, section
        4.3: every chunk re-reads the KV of the chunks before it). Weights are read once per pass, which
        is why piggy-backing prefill tokens on a memory-bound decode batch is nearly free. With no decode
        rows and p0 = 0 this is exactly ``prefill``. Attention models only (``sim.py`` checks)."""
        m = self.model
        flops = 2 * m.matmul_params * batch + 4 * m.n_layers * m.d_model * (ctx + batch)
        tokens, kv = batch, ctx + batch
        for p0, c, last in chunks:
            if m.prefill_lm_head == "all":
                flops += 2 * m.matmul_params * c
            else:
                flops += 2 * m.layer_params * c + (2 * m.vocab * m.d_model if last else 0)
            flops += 2 * m.n_layers * m.d_model * c * (2 * p0 + c + 1)
            tokens += c
            kv += p0 + c
        if self.parallel is not None:
            return self._parallel(flops, m.weight_bytes_read(tokens), kv * m.kv_bytes_per_token, tokens)
        return self._time(flops, m.weight_bytes_read(tokens) + kv * m.kv_bytes_per_token)

    def step_spec(self, ctx: int, batch: int, c: int, chunks=()) -> StepCost:
        """The target's verify pass of speculative decoding (brief 20A2): ``batch`` rows each run ``c`` = gamma + 1
        positions (their last token and gamma drafted ones) after their cached context (``ctx`` in total, the
        ``decode_sum`` convention), plus any prefill ``chunks`` in the same pass (as ``step_mixed``). Row i's
        attention is the causal tail 4 L d j for j = ctx_i + 1 .. ctx_i + c, which sums over the rows to
        2 L d c (2 ctx + batch (c + 1)); every position reaches the LM head (each needs its logits)."""
        m = self.model
        flops = 2 * m.matmul_params * batch * c + 2 * m.n_layers * m.d_model * c * (2 * ctx + batch * (c + 1))
        tokens, kv = batch * c, ctx + batch * c
        for p0, cc, last in chunks:
            if m.prefill_lm_head == "all":
                flops += 2 * m.matmul_params * cc
            else:
                flops += 2 * m.layer_params * cc + (2 * m.vocab * m.d_model if last else 0)
            flops += 2 * m.n_layers * m.d_model * cc * (2 * p0 + cc + 1)
            tokens += cc
            kv += p0 + cc
        if self.parallel is not None:
            return self._parallel(flops, m.weight_bytes_read(tokens), kv * m.kv_bytes_per_token, tokens)
        return self._time(flops, m.weight_bytes_read(tokens) + kv * m.kv_bytes_per_token)

    @cached_property
    def stage(self) -> "CostModel":
        """One pipeline stage: its tp GPUs, no per-step overhead (``_parallel`` adds it once per step)."""
        return replace(self, n_devices=self.parallel.tp, parallel=None, step_overhead=0.0)

    @cached_property
    def scale_up_link(self) -> Link:
        return LINKS[self.parallel.link or self.device.scale_up]

    def _parallel(self, flops: float, wbytes: float, obytes: float, tokens: int) -> StepCost:
        """One forward pass on tp x pp GPUs (see ``Parallel``). ``wbytes`` are the weights the pass reads, ``obytes``
        everything else (KV, embedding rows already in ``wbytes``). Each of the m + pp - 1 pipeline slots is one
        micro-batch on one stage: its share of the FLOPs and bytes on tp GPUs (the roofline, DVFS and power cap of
        those GPUs), then its scale-up traffic. Energy counts the FLOPs and bytes actually done."""
        p, m = self.parallel, self.model
        tp, pp = p.tp, p.pp
        mb = min(p.microbatches or pp, max(1, tokens))
        f_eff = flops
        if m.is_moe and p.ep > 1 and p.expert_imbalance != 1.0:
            f_eff = flops + m.expert_flops(tokens) * (p.expert_imbalance - 1.0)
        t, ec, em, bound = self.stage.step_time_raw(f_eff / (mb * pp), wbytes / pp + obytes / (mb * pp))
        if f_eff != flops:
            ec = ec * (flops / f_eff)                 # the busiest GPU sets the time; energy is the work done
        link = self.scale_up_link
        act = tokens / mb * m.d_model * 2.0          # one micro-batch's activations, BF16
        layers = m.n_layers // pp
        ct, cbytes = 0.0, 0.0                         # per slot: seconds, bytes over all GPUs of the stage
        if tp > 1:
            n_ar = layers if (m.is_moe and p.ep > 1) else 2 * layers
            ct += n_ar * (2 * (tp - 1) / tp * act / link.bandwidth + 2 * (tp - 1) * link.latency)
            cbytes += n_ar * 2 * (tp - 1) * act
        if m.is_moe and p.ep > 1:
            a2a = 2 * (tokens / mb) * m.top_k * m.d_model * 2.0 * (p.ep - 1) / p.ep
            ct += layers * (a2a / link.bandwidth + 2 * (p.ep - 1) * link.latency)
            cbytes += layers * a2a
        hop = 0.0
        if pp > 1:
            hop = act / link.bandwidth + link.latency
        slots = mb + pp - 1
        comm_time = slots * (ct + hop)
        comm_bytes = mb * pp * cbytes + mb * (pp - 1) * act
        time = slots * t + comm_time + self.step_overhead
        n = mb * pp
        return LeverStepCost(flops, mb * wbytes + obytes, time, bound, n * (ec + em), n * ec, n * em,
                             comm_time, comm_bytes * 8 * link.pj_per_bit * 1e-12, 0.0)

    def compress_pass(self, cost: StepCost, ops: float, nbytes: float) -> StepCost:
        """Add a KV-compression kernel after a prefill step (endpoint compression): an
        elementwise pass with its own roofline time, not overlapped with the matmuls."""
        tc, tm = ops / self.flops_rate, nbytes / self.byte_rate
        ec, em = ops * self.joules_per_flop, nbytes * self.joules_per_byte
        return replace(cost, flops=cost.flops + ops, bytes=cost.bytes + nbytes, time=cost.time + max(tc, tm),
                       energy=cost.energy + ec + em, compute_j=cost.compute_j + ec,
                       memory_j=cost.memory_j + em)

    def with_devices(self, n: int) -> "CostModel":
        return replace(self, n_devices=n)
