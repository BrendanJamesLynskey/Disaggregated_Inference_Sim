"""Hardware and model descriptions, plus the roofline cost model.

Everything the simulator knows about *time* comes from this file. The
discrete-event engine (``sim.py``) only asks two questions:

* how long does this batch step take on this instance?   -> ``CostModel``
* how long does this KV-cache transfer take on the link? -> ``Link``

Keeping the cost model separate from the event engine is the single most
important structural decision in any architecture simulator: it lets you swap
a 10-line roofline for a calibrated table, a cycle-level model, or real
silicon measurements without touching the scheduling logic.

Added 2026-10-04 (Fourier-optics follow-up, all illustrative or speculative where marked):

* FFT-mixing model variants (``ModelSpec.mixer``): Hyena-style long convolutions, a 1:3
  attention:Hyena hybrid, and block-circulant weights. Their prefill FLOPs come from an op
  ledger (``Ops``) that reproduces the FOptInf phase-A analysis (``flop_share.py``) exactly.
* ``TransformEngine``: a Fourier-optical **transform** engine co-packaged with a digital part.
  It takes only FFT and Fourier-plane work (with conversion, precision-pass and mask-rewrite
  costs); everything else runs on the digital part. This is different from
  ``HYPOTHETICAL_OPTICAL``, an optical **MAC** that speeds up every matmul.
* ``KVCompression`` / ``KVTransit``: compressing the KV hand-off either in the link
  ("compute in transit") or on the prefill GPU, for comparison.

Added 2026-10-05 (brief 11): the Causal Encoder-Decoder (CED) option of DeepSeek-V4.1-Flash
(arXiv:2609.19969, section 2.2). ``ModelSpec.ced_encoder_layers = E > 0`` makes the bottom E
layers a causal encoder; every decoder layer projects its KV from the last encoder hidden state
with its own weights, so a prompt token needs only the encoder plus the decoder's K/V
projections. The last ``ced_replay`` prompt tokens (the paper's n_win = 128) still go through
the decoder: on the prefill instance (``ced_replay_on="prefill"``, the paper's Decoder SWA
Bounded Replay) or as the decode instance's first step (``"decode"``, the asymmetric P/D
deployment of SGLang RFC #39963, where a prefill instance holds only the encoder's weights).
Off by default: every other model is unchanged. Decode always runs the whole model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from functools import cached_property

GB = 1e9
TB = 1e12

# ENOB a transform engine's analogue output needs to match each format's own rounding
# (FOptInf A§6: 10.3, 8.0 and 6.3, rounded up so the pass count is an exact integer).
ENOB_REQUIRED = {"bf16": 11, "int8": 8, "fp8": 7}


# ─────────────────────────────────────────────────────────── op ledger ──
def rfft_flops(n: int) -> float:
    """Real FFT of length n (a power of two): 2.5 n log2 n, FFTW's convention. log2 is taken
    as an integer, so every language computes the same float."""
    return 2.5 * n * (n.bit_length() - 1)


def pow2_at_least(n: int) -> int:
    return 1 << (n - 1).bit_length()


@dataclass
class Ops:
    """FLOPs by class (the FOptInf analysis's ledger). ``transform`` = FFT/IFFT work and
    ``spectral`` = pointwise work in the Fourier domain: what a 4f optical system could take.
    ``dense`` = matmuls, ``attention`` = QK^T and AV, ``other`` = short convs, gating, sums."""

    dense: float = 0.0
    attention: float = 0.0
    transform: float = 0.0
    spectral: float = 0.0
    other: float = 0.0

    def __iadd__(self, o: "Ops") -> "Ops":
        self.dense += o.dense
        self.attention += o.attention
        self.transform += o.transform
        self.spectral += o.spectral
        self.other += o.other
        return self

    @property
    def total(self) -> float:
        # left to right, never sum(): Python's float sum() is compensated, which would break
        # bit-exact parity with the JavaScript port
        return self.dense + self.attention + self.transform + self.spectral + self.other

    @property
    def optical(self) -> float:
        return self.transform + self.spectral

    @property
    def digital(self) -> float:
        """Everything a transform engine cannot take."""
        return self.dense + self.attention + self.other


# ─────────────────────────────────────────────────────────────── model ──
MIXERS = ("attention", "hyena", "hybrid")


@dataclass(frozen=True)
class ModelSpec:
    """A decoder-only model, described by its shape alone.

    The default is a transformer (``mixer="attention"``). The other mixers keep the same
    shape and swap the token mixer (Hyena order-N long convolutions; ``hybrid`` keeps
    attention in every ``attn_every``-th layer) and optionally the weights
    (``circulant_block`` > 0: block-circulant projections and MLP, speculative at LLM
    scale). Hyena decode is ``"direct"`` (cache past projection inputs; O(context) dot
    product, no transform) or ``"distilled"`` (a recurrence with a constant state per
    sequence; Laughing Hyena). The FOptInf phase-A analysis derives all of this.
    """

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab: int
    weight_bytes: float = 2.0   # BF16 weights
    kv_bytes: float = 2.0       # BF16 KV cache
    mixer: str = "attention"    # "attention" | "hyena" | "hybrid"
    attn_every: int = 4         # hybrid: layer i is attention when i % attn_every == 0
    hyena_order: int = 2        # N long convolutions per Hyena layer (N + 1 projections)
    short_taps: int = 3         # Hyena short depthwise convolution
    circulant_block: int = 0    # 0: dense weights; k > 0: block-circulant (Hyena only)
    decode_style: str = "direct"  # Hyena decode: "direct" (cached inputs) | "distilled"
    distill_state: int = 16     # state size per channel for "distilled" (illustrative)
    act_format: str = "bf16"    # format a transform engine's output must match: bf16 | int8 | fp8
    prefill_lm_head: str = "all"  # "all": every prompt token (as now) | "last": last token only
    # Causal Encoder-Decoder (DeepSeek-V4.1-Flash, arXiv:2609.19969 section 2.2). 0 = off.
    ced_encoder_layers: int = 0   # E: the bottom E layers are the causal encoder (the paper: 20 of 40)
    ced_replay: int = 128         # W: last prompt tokens replayed through the decoder (paper: n_win = 128)
    ced_replay_on: str = "prefill"  # "prefill" (paper section 3.2.2) | "decode" (SGLang RFC #39963)

    def __post_init__(self):
        if self.mixer not in MIXERS:
            raise ValueError(f"unknown mixer {self.mixer!r}")
        if self.circulant_block and self.mixer != "hyena":
            raise ValueError("block-circulant weights are modelled for mixer='hyena' only")
        if self.decode_style not in ("direct", "distilled"):
            raise ValueError(f"unknown decode_style {self.decode_style!r}")
        if self.decode_style == "distilled" and self.mixer != "hyena":
            raise ValueError("decode_style='distilled' needs mixer='hyena' (no attention KV)")
        if self.prefill_lm_head not in ("all", "last"):
            raise ValueError(f"unknown prefill_lm_head {self.prefill_lm_head!r}")
        if self.act_format not in ENOB_REQUIRED:
            raise ValueError(f"unknown act_format {self.act_format!r}")
        if self.ced_encoder_layers:
            if self.mixer != "attention":
                raise ValueError("ced_encoder_layers is modelled for mixer='attention' only")
            if not 0 < self.ced_encoder_layers < self.n_layers:
                raise ValueError("ced_encoder_layers must leave at least one decoder layer")
            if self.ced_replay < 1:
                raise ValueError("ced_replay must be at least 1 token")
            if self.ced_replay_on not in ("prefill", "decode"):
                raise ValueError(f"unknown ced_replay_on {self.ced_replay_on!r}")

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

    # ── layer structure (all-attention models: every layer is attention) ──
    def layer_is_attention(self, i: int) -> bool:
        return self.mixer == "attention" or (self.mixer == "hybrid" and i % self.attn_every == 0)

    @cached_property
    def n_attention_layers(self) -> int:
        return sum(1 for i in range(self.n_layers) if self.layer_is_attention(i))

    @cached_property
    def n_hyena_layers(self) -> int:
        return self.n_layers - self.n_attention_layers

    @cached_property
    def is_transformer(self) -> bool:
        """True for the original models: every number they produce is unchanged."""
        return self.mixer == "attention"

    def _matrices(self) -> list[tuple[int, int]]:
        """(rows, cols) of each weight matrix in one Hyena layer: in-, out-projection, MLP."""
        d, n, ff = self.d_model, self.hyena_order, self.d_ff
        return [((n + 1) * d, d), (d, d), (ff, d), (ff, d), (d, ff)]

    @cached_property
    def hyena_params_per_layer(self) -> int:
        k = self.circulant_block
        return sum(m * n // k if k else m * n for m, n in self._matrices())

    @cached_property
    def params_per_layer(self) -> int:
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        attn = 2 * d * d + 2 * d * kv          # Wq, Wo  +  Wk, Wv (GQA-narrow)
        mlp = 3 * d * self.d_ff                # SwiGLU: gate, up, down
        return attn + mlp

    @cached_property
    def layer_params(self) -> int:
        """Parameters of all layers (attention layers: params_per_layer)."""
        return (self.n_attention_layers * self.params_per_layer
                + self.n_hyena_layers * self.hyena_params_per_layer)

    @cached_property
    def params(self) -> int:
        """Total parameters (untied input embedding and LM head)."""
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
        """
        return self.weight_bytes_streamed + tokens * self.embedding_row_bytes

    @cached_property
    def kv_bytes_per_token(self) -> float:
        """K and V, for every attention layer, for one token; plus, for Hyena layers decoded
        directly, the N cached projection inputs of width d (4x GQA's KV for Llama-3-8B)."""
        conv = self.n_hyena_layers * self.hyena_order * self.d_model if self.decode_style == "direct" else 0
        return (2 * self.n_attention_layers * self.n_kv_heads * self.head_dim + conv) * self.kv_bytes

    @cached_property
    def state_bytes_per_seq(self) -> float:
        """Distilled Hyena decode: a constant complex state (4 bytes per value) per sequence."""
        if self.decode_style != "distilled":
            return 0.0
        return self.n_hyena_layers * self.hyena_order * self.d_model * self.distill_state * 4.0

    def handoff_bytes(self, prompt_len: int) -> float:
        """What prefill hands to decode over the KV link: KV, a conv cache or a constant state."""
        if self.state_bytes_per_seq:
            return self.state_bytes_per_seq
        return prompt_len * self.kv_bytes_per_token

    @cached_property
    def cache_unit_bytes(self) -> float:
        """Admission control's unit: one token of KV (as before), or one sequence's state."""
        return self.kv_bytes_per_token if self.kv_bytes_per_token else self.state_bytes_per_seq

    def cache_units(self, prompt_len: int, output_len: int) -> int:
        """Units a request reserves: the whole sequence's tokens, or one state."""
        return prompt_len + output_len if self.kv_bytes_per_token else 1

    # ── op ledger (FFT-mixing variants; flop_share.py's conventions, same float order) ──
    def _dense_matrix(self, m: int, n: int, tokens: int) -> Ops:
        k = self.circulant_block
        if not k:
            return Ops(dense=2.0 * m * n * tokens)
        bins = k // 2 + 1
        # FFT each of the n/k input blocks, multiply by (m/k)(n/k) block spectra, accumulate,
        # IFFT each of the m/k output blocks (CirCNN's FFT -> multiply -> IFFT)
        return Ops(transform=tokens * ((n // k) * rfft_flops(k) + (m // k) * rfft_flops(k)),
                   spectral=tokens * (m // k) * (n // k) * bins * 6.0,
                   other=tokens * (m // k) * (n // k - 1) * bins * 2.0)

    def _mlp(self, tokens: int) -> Ops:
        o = Ops()
        d, ff = self.d_model, self.d_ff
        for m, n in ((ff, d), (ff, d), (d, ff)):          # gate, up, down
            o += self._dense_matrix(m, n, tokens)
        return o

    def _attention_layer(self, lens: list[int]) -> Ops:
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        o = Ops(dense=2.0 * sum(lens) * (2 * d * d + 2 * d * kv))
        for s in lens:
            o.attention += 2.0 * d * s * (s + 1)
        return o

    def _hyena_layer(self, lens: list[int]) -> Ops:
        d, n = self.d_model, self.hyena_order
        tokens = sum(lens)
        o = Ops()
        o += self._dense_matrix((n + 1) * d, d, tokens)                 # in-projection
        o += self._dense_matrix(d, d, tokens)                           # out-projection
        o.other += tokens * 2.0 * self.short_taps * (n + 1) * d         # short depthwise conv
        o.other += tokens * n * d                                       # gating multiplies
        for s in lens:
            f = pow2_at_least(2 * s)        # zero-padded to >= 2L: causal, no circular wrap
            o.transform += n * d * 2 * rfft_flops(f)
            o.spectral += n * d * (f // 2 + 1) * 6.0
        return o

    def prefill_ops(self, prompt_lens: list[int]) -> Ops:
        """Prefill FLOPs by class, layer by layer (equals CostModel.prefill's total for
        transformers, and flop_share.prefill for every variant)."""
        tokens = sum(prompt_lens)
        o = Ops()
        for i in range(self.n_layers):
            o += self._attention_layer(prompt_lens) if self.layer_is_attention(i) else self._hyena_layer(prompt_lens)
            o += self._mlp(tokens)
        head = tokens if self.prefill_lm_head == "all" else len(prompt_lens)
        o += Ops(dense=2.0 * self.vocab * self.d_model * head)
        return o

    def decode_ops(self, ctx: int, batch: int) -> Ops:
        """Decode FLOPs for ``batch`` new tokens over ``ctx`` cached positions (FFT-mixing
        variants; matches flop_share.decode_per_token at batch 1). No transforms at decode:
        a direct cached dot product or a distilled recurrence."""
        d, n = self.d_model, self.hyena_order
        kv = self.n_kv_heads * self.head_dim
        proj_in, proj_out, mlp, head = self._decode_parts(batch)
        o = Ops()
        for i in range(self.n_layers):
            if self.layer_is_attention(i):
                o += Ops(dense=2.0 * batch * (2 * d * d + 2 * d * kv), attention=4.0 * d * (ctx + batch))
            else:
                o += proj_in
                o += proj_out
                if self.decode_style == "direct":
                    o.other += n * d * 2.0 * (ctx + batch)
                else:
                    o.other += n * d * 8.0 * self.distill_state * batch
            o += mlp
        o += head
        return o

    def _decode_parts(self, batch: int) -> tuple:
        """The weight-matrix Ops of one decode step depend only on the batch: computed once per
        batch size (the same values, added in the same order, so results are unchanged)."""
        cache = self.__dict__.setdefault("_parts_cache", {})
        parts = cache.get(batch)
        if parts is None:
            d, n = self.d_model, self.hyena_order
            parts = (self._dense_matrix((n + 1) * d, d, batch), self._dense_matrix(d, d, batch),
                     self._mlp(batch), Ops(dense=2.0 * self.vocab * self.d_model * batch))
            cache[batch] = parts
        return parts

    # ── what a transform engine needs (FOptInf A§7, A§8) ──
    @cached_property
    def conversion_pairs_per_token(self) -> int:
        """DAC+ADC sample pairs per prompt token for the transform work (one pass): each long
        convolution converts one input and one output; a circulant matrix n in, m out."""
        if self.is_transformer:
            return 0
        per = self.hyena_order * self.d_model
        if self.circulant_block:
            per += sum(max(m, n) for m, n in self._matrices())
        return self.n_hyena_layers * per

    def mask_values(self, prompt_lens: list[int]) -> int:
        """Complex Fourier-plane mask values one forward pass needs: every long-convolution
        filter spectrum for each distinct padded length, plus every circulant block spectrum."""
        if self.is_transformer:
            return 0
        d, n = self.d_model, self.hyena_order
        vals = 0
        for f in sorted({pow2_at_least(2 * s) for s in prompt_lens}):
            vals += self.n_hyena_layers * n * d * (f // 2 + 1)
        k = self.circulant_block
        if k:
            vals += self.n_hyena_layers * sum((m // k) * (c // k) * (k // 2 + 1) for m, c in self._matrices())
        return vals

    def filter_spectrum_bytes(self, prompt_lens: list[int]) -> float:
        """On a digital device the cached filter spectra are read once per step (complex,
        4 bytes per value); on a transform engine they are mask loads instead."""
        if self.is_transformer:
            return 0.0
        d, n = self.d_model, self.hyena_order
        vals = 0
        for f in sorted({pow2_at_least(2 * s) for s in prompt_lens}):
            vals += self.n_hyena_layers * n * d * (f // 2 + 1)
        return vals * 4.0


LLAMA3_8B = ModelSpec("Llama-3-8B", n_layers=32, d_model=4096, n_heads=32,
                      n_kv_heads=8, d_ff=14336, vocab=128256)
LLAMA3_70B = ModelSpec("Llama-3-70B", n_layers=80, d_model=8192, n_heads=64,
                       n_kv_heads=8, d_ff=28672, vocab=128256)

# FFT-mixing variants of the Llama-3-8B shape (FOptInf phase A). Transform share of prefill
# FLOPs at 2,048 tokens: Hyena 0.20%, hybrid 0.15%, circulant 14.0% (84.5% with the LM head
# on the last token only). The circulant LLM is speculative: none of this size is published.
LLAMA3_8B_HYENA = replace(LLAMA3_8B, name="Llama-3-8B-shape Hyena-2", mixer="hyena")
LLAMA3_8B_HYENA_DIST = replace(LLAMA3_8B_HYENA, name="Llama-3-8B-shape Hyena-2 (distilled decode)",
                               decode_style="distilled")
LLAMA3_8B_HYBRID = replace(LLAMA3_8B, name="Llama-3-8B-shape hybrid 1:3", mixer="hybrid")
LLAMA3_8B_HYENA_CIRC = replace(LLAMA3_8B_HYENA, name="Llama-3-8B-shape Hyena-2 + block-circulant 256",
                               circulant_block=256)

# CED proxies (brief 11): the same dense shapes split half encoder, half decoder, as
# DeepSeek-V4.1-Flash splits its 40 layers 20/20 (arXiv:2609.19969, section 4.2.1). Illustrative:
# that model is a 552B MoE with compressed sparse attention, not a dense Llama.
LLAMA3_8B_CED = replace(LLAMA3_8B, name="Llama-3-8B-shape CED 16+16", ced_encoder_layers=16)
LLAMA3_70B_CED = replace(LLAMA3_70B, name="Llama-3-70B-shape CED 40+40", ced_encoder_layers=40)

MODELS = {"llama3-8b": LLAMA3_8B, "llama3-70b": LLAMA3_70B, "llama3-8b-hyena": LLAMA3_8B_HYENA,
          "llama3-8b-hyena-dist": LLAMA3_8B_HYENA_DIST, "llama3-8b-hybrid": LLAMA3_8B_HYBRID,
          "llama3-8b-hyena-circ": LLAMA3_8B_HYENA_CIRC,
          "llama3-8b-ced": LLAMA3_8B_CED, "llama3-70b-ced": LLAMA3_70B_CED}


# ──────────────────────────────────────────────────────────── hardware ──
@dataclass(frozen=True)
class TransformEngine:
    """A Fourier-optical transform engine (4f system), co-packaged with a digital part.

    Illustrative throughout. It takes FFT and Fourier-plane work only; each forward pass
    converts every transform input in (DAC) and output out (ADC), at Walden-rule energy
    (FoM x 2^ENOB per sample, FHESim 04's FoMs). Below the ENOB a format needs, passes are
    repeated and averaged: 4^(required - ENOB) passes (averaging k passes buys half a bit
    per doubling; Garg et al., arXiv:2102.06365). Intensity detection doubles the passes
    (sign recovery). The Fourier-plane mask holds ``mask_values`` complex values and is
    rewritten at ``mask_rate_hz`` (a 2 MP DMD at 8-bit depth, Miscuglio et al., Optica 2020).
    Lasers and thermal tuning burn ``laser_w + tuning_w`` whether or not work arrives.
    """

    samples_per_s: float = 1e12   # converter throughput per direction (as FHE OpticalEngine)
    enob: int = 8                 # effective bits of the DAC -> optics -> ADC chain
    fom_dac_fj: float = 10.0      # Walden FoM, fJ per conversion step
    fom_adc_fj: float = 20.0
    laser_w: float = 10.0         # static, per device
    tuning_w: float = 10.0        # static, per device
    mask_values: int = 2_000_000  # complex values the mask holds at once
    mask_rate_hz: float = 1031.0  # rewrites per second (about 20 kHz at 1-bit; LC SLMs tens of Hz)
    detection: str = "coherent"   # "coherent" | "intensity" (doubles the passes)
    overlap: bool = False         # True: optical and digital time overlap (max); False: they add

    def passes(self, act_format: str) -> int:
        k = 4 ** max(0, ENOB_REQUIRED[act_format] - self.enob)
        return 2 * k if self.detection == "intensity" else k

    @property
    def pj_per_pair(self) -> float:
        """One DAC sample plus one ADC sample, picojoules."""
        return self.fom_dac_fj * 2 ** self.enob * 1e-3 + self.fom_adc_fj * 2 ** self.enob * 1e-3

    @property
    def static_w(self) -> float:
        return self.laser_w + self.tuning_w


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
    transform: TransformEngine | None = None   # a co-packaged optical transform engine, if any
    # Fraction of the matmul rate this device's FFT and Fourier-domain work achieves (FFT-mixing
    # models only). 1.0 (the default) is optimistic for GPUs: FlashFFTConv (arXiv:2311.05908)
    # exists because FFTs use matmul units poorly. Time and dynamic energy both scale by 1/x.
    fft_efficiency: float = 1.0

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) where compute and memory balance."""
        return (self.peak_flops * self.flops_eff) / (self.mem_bw * self.bw_eff)


H100_SXM = Accelerator("H100-SXM", peak_flops=989 * TB, mem_bw=3.35 * TB, mem_capacity=80 * GB)
A100_SXM = Accelerator("A100-SXM", peak_flops=312 * TB, mem_bw=2.039 * TB, mem_capacity=80 * GB,
                       tdp_w=400.0, idle_w=60.0, pj_per_flop=1.6, pj_per_byte=70.0)
# A deliberately hypothetical part: abundant matmul throughput, ordinary memory.
# Use it to ask "what if compute were nearly free?" -- the answer is that decode
# barely moves, because decode is bandwidth-bound.
HYPOTHETICAL_OPTICAL = Accelerator("Hypothetical-optical-MAC", peak_flops=4000 * TB,
                                   mem_bw=3.35 * TB, mem_capacity=80 * GB, flops_eff=0.4,
                                   # cheap multiplies, but lasers, thermal tuning and
                                   # converters burn power whether or not work arrives
                                   idle_w=180.0, pj_per_flop=0.1, pj_per_byte=60.0)

# The transform engine, co-packaged with an H100-class digital part (``optical-fft``) or an
# A100-class one (``optical-fft-small``). Not HYPOTHETICAL_OPTICAL: that one speeds up every
# matmul; these speed up only FFT and Fourier-plane work, so they help only FFT-mixing models.
OPTICAL_FFT = replace(H100_SXM, name="Optical-FFT + H100-class", transform=TransformEngine())
OPTICAL_FFT_SMALL = replace(A100_SXM, name="Optical-FFT + A100-class", transform=TransformEngine())

ACCELERATORS = {"h100": H100_SXM, "a100": A100_SXM, "optical": HYPOTHETICAL_OPTICAL,
                "optical-fft": OPTICAL_FFT, "optical-fft-small": OPTICAL_FFT_SMALL}


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
    "eth-100g": Link("100 GbE", bandwidth=12.5 * GB, latency=20e-6, pj_per_bit=15.0),
    "eth-25g": Link("25 GbE", bandwidth=3.125 * GB, latency=20e-6, pj_per_bit=15.0),
    # Photonic interconnect (NOT Fourier optics): an illustrative co-packaged-optics link.
    # Round numbers, not a product: multi-Tb/s optical I/O chiplets exist (Wade et al., TeraPHY,
    # IEEE Micro 2020); 1.6 Tb/s and 3 pJ/bit are this simulator's assumptions.
    "cpo-optical": Link("Co-packaged optics (illustrative)", bandwidth=200 * GB, latency=5e-6, pj_per_bit=3.0),
}


# ─────────────────────────────────────────── KV hand-off compression ──
@dataclass(frozen=True)
class KVCompression:
    """A compression of the KV hand-off (BF16 in). ``ratio`` = bytes in / bytes out.
    ``ops_per_value``: elementwise work per BF16 value (amax, scale, round, select);
    ``fft``: also a transform along the token axis (forward and inverse, FreqKV-style)."""

    name: str
    ratio: float
    ops_per_value: float
    fft: bool = False

    def fft_ops_per_value(self, prompt_len: int) -> float:
        """Forward plus inverse real transform along the token axis, per value."""
        if not self.fft:
            return 0.0
        n = pow2_at_least(prompt_len)
        return 2 * rfft_flops(n) / n


# Tolerable KV widths come from independent KV-quantisation work: KIVI (arXiv:2402.02750,
# 2-bit) and KVQuant (arXiv:2401.18079, 3-bit with < 0.1 perplexity loss), so 8- and 4-bit
# are within what they report. FreqKV (arXiv:2505.00570) keeps half the DCT components by
# default (needs light fine-tuning). The simulator does not model accuracy.
KV_PRESETS = {
    "none": KVCompression("none", 1.0, 0.0),
    "fp8": KVCompression("fp8", 2.0, 1.0),                     # scale + round per value
    "fp4-block": KVCompression("fp4-block", 64 / 17, 2.0),     # E2M1, 32-value blocks + 8-bit scale
    "freq-keep-k": KVCompression("freq-keep-k", 2.0, 1.0, fft=True),   # keep half the bins
}


@dataclass(frozen=True)
class KVTransit:
    """Where the hand-off is compressed, and the in-transit stage's budget (illustrative,
    representative of published 2026 compute-in-transit prototypes; speculative mapping).

    ``where="transit"``: a stage in the link compresses the stream as it passes, at no GPU
    cost, with a compute budget of ``ops_per_byte`` per line byte, ``pj_per_bit`` per input
    bit and ``latency`` added. Its transforms are passive (``native_fft``). If the stage
    cannot keep up, the transfer takes its compute time instead (flagged as transit-bound).
    ``where="endpoint"``: the prefill GPU compresses (an elementwise pass over the KV: HBM
    read + write, plus the FLOPs) before sending."""

    compression: KVCompression
    where: str = "transit"
    ops_per_byte: float = 1.6
    pj_per_bit: float = 1.0
    latency: float = 1e-6
    native_fft: bool = True

    def __post_init__(self):
        if self.where not in ("transit", "endpoint"):
            raise ValueError(f"unknown kv compression site {self.where!r}")

    def ops(self, nbytes: float, prompt_len: int, kv_bytes: float, on_gpu: bool) -> float:
        """Operations to compress ``nbytes`` of hand-off."""
        values = nbytes / kv_bytes
        c = self.compression
        per = c.ops_per_value
        if c.fft and (on_gpu or not self.native_fft):
            per = per + c.fft_ops_per_value(prompt_len)
        return per * values


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
    # Not fields: ordinary steps construct as fast as before 2026-10-04 (the simulator's hot path).
    optical_j = 0.0            # transform engine: conversion energy (DAC + ADC)
    optical_flops = 0.0        # transform engine: FFT and Fourier-plane FLOPs it took


@dataclass(frozen=True)
class OpticalStepCost(StepCost):
    """A prefill step with transform-engine work."""

    optical_j: float = 0.0
    optical_flops: float = 0.0


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
        return self.device.peak_flops * self.device.flops_eff * self.n_devices

    @cached_property
    def byte_rate(self) -> float:
        return self.device.mem_bw * self.device.bw_eff * self.n_devices

    @property
    def kv_capacity_tokens(self) -> int:
        """Admission units that fit beside the weights: KV tokens (or distilled states)."""
        free = self.device.mem_capacity * self.n_devices * self.mem_util - self.resident_weight_bytes
        if free <= 0:
            raise ValueError(f"{self.model.name} does not fit on {self.n_devices}x {self.device.name}")
        return int(free // self.model.cache_unit_bytes)

    @cached_property
    def joules_per_flop(self) -> float:
        return self.device.pj_per_flop * 1e-12

    @cached_property
    def joules_per_byte(self) -> float:
        return self.device.pj_per_byte * 1e-12

    @cached_property
    def idle_w(self) -> float:
        return self.device.idle_w * self.n_devices

    @cached_property
    def optical_static_w(self) -> float:
        """Lasers and thermal tuning of a co-packaged transform engine (0 without one)."""
        eng = self.device.transform
        return eng.static_w * self.n_devices if eng is not None else 0.0

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
        if not m.is_transformer:
            return self._prefill_ops(prompt_lens, tokens)
        # The LM head is charged for every prompt token, as a plain forward pass computes
        # (HF transformers by default, and Torch_Sim_Frontend's trace of it); serving
        # engines that keep only the last position's logits do less.
        if m.prefill_lm_head == "all":
            flops = 2 * m.matmul_params * tokens
        else:
            flops = 2 * m.layer_params * tokens + 2 * m.vocab * m.d_model * len(prompt_lens)
        # causal attention: QK^T and AV, each 2*d*c FLOPs at position c (diagonal included)
        flops += sum(2 * m.n_layers * m.d_model * s * (s + 1) for s in prompt_lens)
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

    def _digital_flops(self, ops: Ops) -> float:
        """Effective FLOPs on the digital part: FFT and spectral work at ``fft_efficiency``."""
        eff = self.device.fft_efficiency
        return ops.total if eff == 1.0 else ops.digital + ops.optical / eff

    def optical_terms(self, prompt_lens: list[int], tokens: int) -> tuple[int, int, int, float]:
        """(passes, conversions, mask rewrites, optical seconds) of one prefill step on the
        transform engine. All counts are integers, so every language agrees exactly."""
        m, eng = self.model, self.device.transform
        k = eng.passes(m.act_format)
        conversions = m.conversion_pairs_per_token * tokens * k
        rewrites = -(-m.mask_values(prompt_lens) // eng.mask_values)   # integer ceil
        return k, conversions, rewrites, conversions / eng.samples_per_s + rewrites / eng.mask_rate_hz

    def _prefill_ops(self, prompt_lens: list[int], tokens: int) -> StepCost:
        """FFT-mixing variants: the op ledger, on the digital part or the transform engine."""
        m = self.model
        ops = m.prefill_ops(prompt_lens)
        nbytes = m.weight_bytes_read(tokens) + tokens * m.kv_bytes_per_token
        if m.state_bytes_per_seq:
            nbytes += len(prompt_lens) * m.state_bytes_per_seq
        eng = self.device.transform
        if eng is None:            # all digital: the filter spectra are read from memory
            return self._time(self._digital_flops(ops), nbytes + m.filter_spectrum_bytes(prompt_lens))
        # The engine takes the transforms and the Fourier-plane multiplies; the digital part
        # runs the rest (its DVFS and power cap apply to its share only).
        _, conversions, _, t_opt = self.optical_terms(prompt_lens, tokens)
        flops = ops.digital
        t_dig, ec, em, bound = self.step_time_raw(flops, nbytes)
        t = max(t_dig, t_opt) if eng.overlap else t_dig + t_opt
        if t_opt > t_dig:
            bound = "optical"
        oj = conversions * eng.pj_per_pair * 1e-12
        return OpticalStepCost(flops, nbytes, t + self.step_overhead, bound, ec + em + oj, ec, em, oj,
                               ops.optical)

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
        if not m.is_transformer:
            # FFT-mixing variants decode digitally (a transform engine's part is idle):
            # direct cached dot products, or a distilled recurrence with constant state.
            nbytes = m.weight_bytes_read(batch) + (ctx + batch) * m.kv_bytes_per_token
            if m.state_bytes_per_seq:
                nbytes += batch * m.state_bytes_per_seq
            return self._time(self._digital_flops(m.decode_ops(ctx, batch)), nbytes)
        flops = 2 * m.matmul_params * batch + 4 * m.n_layers * m.d_model * (ctx + batch)
        nbytes = m.weight_bytes_read(batch) + (ctx + batch) * m.kv_bytes_per_token
        return self._time(flops, nbytes)

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
