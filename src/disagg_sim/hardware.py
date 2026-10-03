"""Hardware and model descriptions, plus the roofline cost model.

Everything the simulator knows about *time* comes from this file. The
discrete-event engine (``sim.py``) only asks two questions:

* how long does this batch step take on this instance?   -> ``CostModel``
* how long does this KV-cache transfer take on the link? -> ``Link``

Keeping the cost model separate from the event engine is the single most
important structural decision in any architecture simulator: it lets you swap
a 10-line roofline for a calibrated table, a cycle-level model, or real
silicon measurements without touching the scheduling logic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from functools import cached_property

GB = 1e9
TB = 1e12


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

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @cached_property
    def params_per_layer(self) -> int:
        d, kv = self.d_model, self.n_kv_heads * self.head_dim
        attn = 2 * d * d + 2 * d * kv          # Wq, Wo  +  Wk, Wv (GQA-narrow)
        mlp = 3 * d * self.d_ff                # SwiGLU: gate, up, down
        return attn + mlp

    @cached_property
    def params(self) -> int:
        """Total parameters (untied input embedding and LM head)."""
        return self.n_layers * self.params_per_layer + 2 * self.vocab * self.d_model

    @cached_property
    def matmul_params(self) -> int:
        """Parameters that take part in a matmul per token (embedding is a lookup)."""
        return self.n_layers * self.params_per_layer + self.vocab * self.d_model

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
        """K and V, for every layer, for one token."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_bytes


LLAMA3_8B = ModelSpec("Llama-3-8B", n_layers=32, d_model=4096, n_heads=32,
                      n_kv_heads=8, d_ff=14336, vocab=128256)
LLAMA3_70B = ModelSpec("Llama-3-70B", n_layers=80, d_model=8192, n_heads=64,
                       n_kv_heads=8, d_ff=28672, vocab=128256)

MODELS = {"llama3-8b": LLAMA3_8B, "llama3-70b": LLAMA3_70B}


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

ACCELERATORS = {"h100": H100_SXM, "a100": A100_SXM, "optical": HYPOTHETICAL_OPTICAL}


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
}


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

    @cached_property
    def flops_rate(self) -> float:
        return self.device.peak_flops * self.device.flops_eff * self.n_devices

    @cached_property
    def byte_rate(self) -> float:
        return self.device.mem_bw * self.device.bw_eff * self.n_devices

    @property
    def kv_capacity_tokens(self) -> int:
        free = self.device.mem_capacity * self.n_devices * self.mem_util - self.model.weight_bytes_total
        if free <= 0:
            raise ValueError(f"{self.model.name} does not fit on {self.n_devices}x {self.device.name}")
        return int(free // self.model.kv_bytes_per_token)

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
        """Time and energy of one step under a simple DVFS model.

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
        return t + self.step_overhead, ec, em, bound

    def _time(self, flops: float, nbytes: float) -> StepCost:
        t, ec, em, bound = self.step_time(flops, nbytes)
        return StepCost(flops, nbytes, t, bound, ec + em, ec, em)

    def prefill(self, prompt_lens: list[int]) -> StepCost:
        """One prefill step over whole prompts (no chunking)."""
        m = self.model
        tokens = sum(prompt_lens)
        # The LM head is charged for every prompt token, as a plain forward pass computes
        # (HF transformers by default, and Torch_Sim_Frontend's trace of it); serving
        # engines that keep only the last position's logits do less.
        flops = 2 * m.matmul_params * tokens
        # causal attention: QK^T and AV, each 2*d*c FLOPs at position c (diagonal included)
        flops += sum(2 * m.n_layers * m.d_model * s * (s + 1) for s in prompt_lens)
        nbytes = m.weight_bytes_read(tokens) + tokens * m.kv_bytes_per_token
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
        nbytes = m.weight_bytes_read(batch) + (ctx + batch) * m.kv_bytes_per_token
        return self._time(flops, nbytes)

    def with_devices(self, n: int) -> "CostModel":
        return replace(self, n_devices=n)
