"""The SimPy discrete-event model of a prefill/decode-disaggregated server.

Topology (``mode="disagg"``)::

    arrivals ─► router ─► prefill[0..P) ─► KV link (C channels) ─► router ─► decode[0..D) ─► done
                              │                                                │
                         TTFT stamped here                          one token per step

``mode="colocated"`` replaces the two pools with N instances that do both
jobs, vLLM-v0 style: a waiting prefill pre-empts the next decode step, which
is exactly the interference that disaggregation exists to remove.

Each instance is one SimPy process that loops "form a batch → ask the cost
model how long the step takes → ``yield env.timeout(step)`` → retire tokens".
The KV link is a ``simpy.Resource`` whose capacity is the number of channels.

Added 2026-10-06 (brief 20A1), all off by default (``SimConfig.scheduled``): colocated batching
policies (prefill-priority as before, decode-priority, Sarathi-Serve chunked prefill with a token
budget), KV memory policies (whole-sequence reservation as before, or the over-reservations vLLM
compares against, or paged blocks with preemption by recompute or swap to host memory) and prefix
caching (an LRU cache of shared prompt segments). They run in ``ScheduledInstance``; with every lever
off, ``ColocatedInstance`` runs exactly as before.
"""

from __future__ import annotations

import bisect
import heapq
from collections import deque
from dataclasses import replace
from dataclasses import dataclass, field

import simpy

from .hardware import (H100_SXM, LINKS, LLAMA3_70B, Accelerator, CostModel, KVTransit, Link,
                       ModelSpec, StepCost)
from .trace import Tracer
from .workload import Request


@dataclass
class SimConfig:
    model: ModelSpec = LLAMA3_70B
    device: Accelerator = H100_SXM
    devices_per_instance: int = 4
    mode: str = "disagg"               # "disagg" | "colocated"
    n_prefill: int = 1
    n_decode: int = 1
    n_colocated: int = 2
    link: Link = LINKS["ib-ndr"]
    max_prefill_tokens: int = 8192     # token budget of one prefill step
    max_decode_batch: int = 256
    step_overhead: float = 0.5e-3
    ttft_slo: float = 1.0              # seconds
    tpot_slo: float = 0.025            # seconds per token
    warmup_frac: float = 0.1           # fraction of earliest requests excluded from latency stats
    sample_dt: float = 0.05            # time-series sampling period, seconds
    trace: bool = False                # record a Chrome/Perfetto trace
    fast_forward: bool = False         # exact accelerated decode path (FastDecodeInstance)
    power_cap_w: float | None = None   # per-device power cap (W); steps throttle to respect it
    prefill_power_cap_w: float | None = None   # overrides power_cap_w for the prefill pool
    decode_power_cap_w: float | None = None    # overrides power_cap_w for the decode pool
    dvfs: bool = False                 # lower compute clocks on memory-bound steps
    # Heterogeneous pools (Splitwise-style): None means "use device / devices_per_instance",
    # so every existing configuration is unchanged. Colocated mode uses ``device`` only.
    prefill_device: Accelerator | None = None
    decode_device: Accelerator | None = None
    prefill_devices_per_instance: int | None = None
    decode_devices_per_instance: int | None = None
    kv_transit: KVTransit | None = None   # compress the KV hand-off in the link or at the GPU
    # Brief 20A1 levers (colocated mode; every default reproduces the behaviour above exactly).
    batch_policy: str = "prefill-priority"   # | "decode-priority" | "chunked" (Sarathi-Serve)
    max_num_batched_tokens: int | None = None   # token budget of one step (None: max_prefill_tokens)
    kv_policy: str = "oracle"          # reserve prompt + output up front (as before) | "pow2" | "max" | "paged"
    kv_block_size: int = 16            # paged: tokens per KV block (vLLM's default)
    max_seq_len: int = 2048            # "max": every request reserves this many tokens
    kv_watermark: float = 0.01         # paged: fraction of blocks kept free when admitting (vLLM)
    preemption: str = "recompute"      # paged, out of blocks: "recompute" | "swap" (to host memory)
    host_link: Link = LINKS["pcie5"]   # swap path between device and host memory
    prefix_caching: bool = False       # LRU cache of shared prompt segments (Request.prefix)

    @property
    def scheduled(self) -> bool:
        """True when any brief-20A1 lever is on (inert settings such as the block size alone are not)."""
        return (self.batch_policy != "prefill-priority" or self.max_num_batched_tokens is not None
                or self.kv_policy != "oracle" or self.prefix_caching)

    @property
    def heterogeneous(self) -> bool:
        return any(x is not None for x in (self.prefill_device, self.decode_device,
                                           self.prefill_devices_per_instance,
                                           self.decode_devices_per_instance))

    def pool(self, role: str) -> tuple[Accelerator, int]:
        """The device and devices per instance that a pool's instances use."""
        dev = {"prefill": self.prefill_device, "decode": self.decode_device}.get(role)
        n = {"prefill": self.prefill_devices_per_instance,
             "decode": self.decode_devices_per_instance}.get(role)
        return (dev if dev is not None else self.device,
                n if n is not None else self.devices_per_instance)


# ──────────────────────────────────────────────────────────── instances ──
class Instance:
    """Common machinery: a FIFO queue, an idle/wake-up event, busy-time accounting."""

    role = "instance"

    def __init__(self, sim: "Simulation", idx: int):
        self.sim, self.env, self.idx = sim, sim.env, idx
        self.name = f"{self.role}-{idx}"
        self.cost = sim.cost_for(self.role)
        self.queue: deque[Request] = deque()
        self.running: list[Request] = []
        self.kv_used = 0
        try:
            self.kv_cap = self.cost.kv_capacity_tokens
        except ValueError as e:
            raise ValueError(f"{self.role} pool: {e}") from None
        if not sim.cfg.model.kv_bytes_per_token:     # a distilled-state model reserves one state
            self.kv_need = lambda r: 1
        self.busy = 0.0
        self.steps = 0
        self.compute_bound_time = 0.0
        self.flops = 0.0            # work actually done, for MFU / MBU
        self.bytes = 0.0
        self.batch_sum = 0          # sequences processed, summed over steps
        self.compute_j = 0.0        # dynamic joules (static power is added in metrics)
        self.memory_j = 0.0
        self.peak_power = 0.0       # highest average power over any single step, W
        self.power_bound_time = 0.0
        self.optical_j = 0.0        # transform engine: conversion energy
        self.optical_flops = 0.0    # transform engine: FLOPs it took
        self.optical_bound_time = 0.0
        self._wake: simpy.Event | None = None
        self.proc = self.env.process(self.run())

    def submit(self, r: Request) -> None:
        self.queue.append(r)
        if self._wake is not None and not self._wake.triggered:
            self._wake.succeed()

    def idle(self):
        self._wake = self.env.event()
        yield self._wake
        self._wake = None

    def load(self) -> float:
        return len(self.queue) + len(self.running)

    def step(self, kind: str, cost, label: str, batch: int):
        start = self.env.now
        yield self.env.timeout(cost.time)
        self.busy += cost.time
        self.steps += 1
        self.flops += cost.flops
        self.bytes += cost.bytes
        self.batch_sum += batch
        if cost.optical_j or cost.bound == "optical":
            self.account_optical(cost)
        else:
            self.account_power(cost.time, cost.compute_j, cost.memory_j, cost.bound)
        if cost.bound == "compute":
            self.compute_bound_time += cost.time
        self.sim.tracer.span(self.name, kind, label, start, cost.time)

    def account_power(self, dt: float, ec: float, em: float, bound: str) -> None:
        self.compute_j += ec
        self.memory_j += em
        p = self.cost.idle_w + (ec + em) / dt
        if p > self.peak_power:
            self.peak_power = p
        if bound == "power":
            self.power_bound_time += dt

    def account_optical(self, cost) -> None:
        """A step with transform-engine work: conversion energy and the optical bound."""
        self.compute_j += cost.compute_j
        self.memory_j += cost.memory_j
        self.optical_j += cost.optical_j
        self.optical_flops += cost.optical_flops
        dt = cost.time
        p = self.cost.idle_w + self.cost.optical_static_w + (cost.compute_j + cost.memory_j + cost.optical_j) / dt
        if p > self.peak_power:
            self.peak_power = p
        if cost.bound == "power":
            self.power_bound_time += dt
        elif cost.bound == "optical":
            self.optical_bound_time += dt

    # Token bookkeeping shared by decode and colocated instances.
    def kv_need(self, r: Request) -> int:
        return r.prompt_len + r.output_len   # reserve the whole sequence up front

    def emit_first_token(self, r: Request) -> None:
        r.first_token = r.last_token = self.env.now
        r.tokens_out = 1

    def decode_step_done(self) -> None:
        now = self.env.now
        still = []
        for r in self.running:
            r.itls.append(now - r.last_token)
            r.tokens_out += 1
            r.last_token = now
            if r.tokens_out >= r.output_len:
                self.kv_used -= self.kv_need(r)
                self.sim.finish(r)
            else:
                still.append(r)
        self.running = still

    def admit_decode(self) -> None:
        cfg = self.sim.cfg
        while (self.queue and len(self.running) < cfg.max_decode_batch
               and self.kv_used + self.kv_need(self.queue[0]) <= self.kv_cap):
            r = self.queue.popleft()
            self.kv_used += self.kv_need(r)
            r.decode_start = self.env.now
            self.running.append(r)

    def decode_once(self):
        ctx = [r.prompt_len + r.tokens_out for r in self.running]
        yield from self.step("decode", self.cost.decode(ctx), f"decode b={len(ctx)}", len(ctx))
        self.decode_step_done()


class PrefillInstance(Instance):
    role = "prefill"

    def load(self) -> float:
        return sum(r.prompt_len for r in self.queue)

    def run(self):
        budget = self.sim.cfg.max_prefill_tokens
        while True:
            if not self.queue:
                yield from self.idle()
                continue
            batch, tokens = [], 0
            while self.queue and (not batch or tokens + self.queue[0].prompt_len <= budget):
                r = self.queue.popleft()
                batch.append(r)
                tokens += r.prompt_len
            for r in batch:
                r.prefill_start = self.env.now
            cost = self.cost.prefill([r.prompt_len for r in batch])
            tr = self.sim.cfg.kv_transit
            if tr is not None and tr.where == "endpoint":
                cost = self.sim.endpoint_compress(self.cost, cost, batch)
            yield from self.step("prefill", cost, f"prefill n={len(batch)} tok={tokens}",
                                 len(batch))
            if self.cost.encoder_only:
                # CED, replay on decode: the decode instance emits the first token, so every
                # request is handed off, single-token ones included
                for r in batch:
                    r.prefill_done = self.env.now
                    self.env.process(self.sim.kv_transfer(r))
                continue
            for r in batch:
                self.emit_first_token(r)
                if r.output_len <= 1:
                    self.sim.finish(r)
                else:
                    self.env.process(self.sim.kv_transfer(r))


class DecodeInstance(Instance):
    role = "decode"

    def run(self):
        while True:
            self.admit_decode()
            if not self.running:
                yield from self.idle()
                continue
            yield from self.decode_once()


class CedDecodeInstance(DecodeInstance):
    """A decode instance for a CED model with the replay on decode (SGLang RFC #39963).

    A newly admitted request first runs its last ``ced_replay`` prompt tokens through the
    whole model (a bounded prefill) in the same step as the running batch's decode, and
    emits its first token at the end of that step; from then on it decodes as usual.
    """

    def run(self):
        while True:
            self.admit_decode()
            if not self.running:
                yield from self.idle()
                continue
            yield from self.ced_once()

    def ced_once(self):
        w = self.sim.cfg.model.ced_replay
        ctx, b, replay = 0, 0, []
        for r in self.running:
            if r.tokens_out:
                ctx += r.prompt_len + r.tokens_out
                b += 1
            else:
                replay.append((r.prompt_len, min(r.prompt_len, w)))
        cost = self.cost.ced_step(ctx, b, replay)
        yield from self.step("decode", cost, f"decode b={b} replay={len(replay)}", len(self.running))
        now, still = self.env.now, []
        for r in self.running:
            if r.tokens_out:
                r.itls.append(now - r.last_token)
                r.tokens_out += 1
                r.last_token = now
            else:
                self.emit_first_token(r)
            if r.tokens_out >= r.output_len:
                self.kv_used -= self.kv_need(r)
                self.sim.finish(r)
            else:
                still.append(r)
        self.running = still


class FastDecodeInstance(DecodeInstance):
    """An exact, accelerated decode instance (``SimConfig.fast_forward=True``).

    Three standard simulator-acceleration techniques, applied without changing
    the answer:

    1. **Incremental state.** The cost model needs only the batch size and the
       total context, so keep a running sum instead of re-summing every step.
    2. **Lazy bookkeeping.** Don't touch every sequence every step. Record the
       end time of each step once per instance; a sequence's token count, last
       token time and inter-token latencies are reconstructed from that array
       when it finishes. Finishes are found with a heap keyed by finish step.
    3. **Macro-stepping (time skipping).** Between state changes (an admission
       or a finish) the step sequence is deterministic, so compute all of it in
       a tight loop and hand SimPy *one* timeout instead of one per step. A new
       arrival interrupts the macro-step; the step in flight completes and the
       newcomer is admitted at that step boundary, exactly as in the baseline.

    Results match ``DecodeInstance`` to floating-point rounding (tested).
    """

    def __init__(self, sim: "Simulation", idx: int):
        super().__init__(sim, idx)
        self.running = {}                    # rid -> Request (len() works for probes)
        self.step_end: list[float] = []      # end time of every completed step
        self.ctx_sum = 0                     # sum of (prompt + tokens_out) over running
        self.finish_heap: list = []          # (finish step index, rid, request)
        self._macro = False

    def submit(self, r: Request) -> None:
        super().submit(r)
        if self._macro:                      # new work: cut the macro-step short
            self._macro = False
            self.proc.interrupt()

    def admit_decode(self) -> None:
        cfg = self.sim.cfg
        while (self.queue and len(self.running) < cfg.max_decode_batch
               and self.kv_used + self.kv_need(self.queue[0]) <= self.kv_cap):
            r = self.queue.popleft()
            self.kv_used += self.kv_need(r)
            r.decode_start = self.env.now
            r.adm_step = len(self.step_end)
            self.running[r.rid] = r
            self.ctx_sum += r.prompt_len + r.tokens_out
            heapq.heappush(self.finish_heap,
                           (r.adm_step + r.output_len - r.tokens_out - 1, r.rid, r))

    def run(self):
        # Hoisted constants; no StepCost object per step.
        m, cm = self.cost.model, self.cost
        fa, fb = 2 * m.matmul_params, 4 * m.n_layers * m.d_model
        w, er, kv = m.weight_bytes_streamed, m.embedding_row_bytes, m.kv_bytes_per_token
        step_time, overhead = cm.step_time_raw, cm.step_overhead
        generic = not m.is_transformer        # FFT-mixing variants: the cost model's own ledger
        horizon = 16                         # adaptive look-ahead, in steps
        while True:
            self.admit_decode()
            if not self.running:
                yield from self.idle()
                continue
            # Plan up to the next finish (nothing else can change meanwhile), but no
            # further than the look-ahead: work planned past an interrupt is wasted.
            b = len(self.running)
            k = min(self.finish_heap[0][0] - len(self.step_end) + 1, horizon)
            t, ctx, ends, steps = self.env.now, self.ctx_sum, [], []
            for _ in range(k):
                if generic:
                    c = cm.decode_sum(ctx, b)
                    flops, nbytes, dt, ec, em, bound = c.flops, c.bytes, c.time, c.compute_j, c.memory_j, c.bound
                else:
                    flops, nbytes = fa * b + fb * (ctx + b), w + b * er + (ctx + b) * kv
                    dt, ec, em, bound = step_time(flops, nbytes)
                    dt = dt + overhead
                t += dt
                ends.append(t)
                steps.append((dt, flops, nbytes, bound, ec, em))
                ctx += b
            self._macro = True
            try:
                yield self.env.timeout(ends[-1] - self.env.now)
                done = k
                horizon = min(4096, horizon * 2)
            except simpy.Interrupt:
                # Finish the step in flight, then stop at its boundary.
                j = min(bisect.bisect_right(ends, self.env.now), k - 1)
                yield self.env.timeout(ends[j] - self.env.now)
                done = j + 1
                horizon = max(8, 2 * done)
            self._macro = False
            self.commit(ends[:done], steps[:done], b)

    def commit(self, ends: list[float], steps: list, b: int) -> None:
        idle, tracing = self.cost.idle_w, self.sim.tracer.enabled
        for i, (dt, flops, nbytes, bound, ec, em) in enumerate(steps):
            self.busy += dt
            self.flops += flops
            self.bytes += nbytes
            self.compute_j += ec                    # account_power(), inlined for the hot loop
            self.memory_j += em
            p = idle + (ec + em) / dt
            if p > self.peak_power:
                self.peak_power = p
            if bound != "memory":
                if bound == "compute":
                    self.compute_bound_time += dt
                else:
                    self.power_bound_time += dt
            if tracing:
                self.sim.tracer.span(self.name, "decode", f"decode b={b}", ends[i] - dt, dt)
        self.steps += len(steps)
        self.batch_sum += b * len(steps)
        self.step_end.extend(ends)
        self.ctx_sum += b * len(steps)
        n, e = len(self.step_end), self.step_end
        while self.finish_heap and self.finish_heap[0][0] < n:
            fin, _, r = heapq.heappop(self.finish_heap)
            s = r.adm_step
            r.itls.append(e[s] - r.last_token)
            r.itls.extend(e[j] - e[j - 1] for j in range(s + 1, fin + 1))
            r.tokens_out = r.output_len
            r.last_token = e[fin]
            self.ctx_sum -= r.prompt_len + r.output_len
            self.kv_used -= self.kv_need(r)
            del self.running[r.rid]
            self.sim.finish(r, e[fin])


class ColocatedInstance(Instance):
    """Prefill and decode share one device; prefill has priority (vLLM v0)."""

    role = "colocated"

    def run(self):
        cfg = self.sim.cfg
        while True:
            batch, tokens = [], 0
            while (self.queue and len(self.running) + len(batch) < cfg.max_decode_batch
                   and (not batch or tokens + self.queue[0].prompt_len <= cfg.max_prefill_tokens)
                   and self.kv_used + self.kv_need(self.queue[0]) <= self.kv_cap):
                r = self.queue.popleft()
                self.kv_used += self.kv_need(r)
                batch.append(r)
                tokens += r.prompt_len
            if batch:
                for r in batch:
                    r.prefill_start = self.env.now
                cost = self.cost.prefill([r.prompt_len for r in batch])
                yield from self.step("prefill", cost, f"prefill n={len(batch)} tok={tokens}",
                                 len(batch))
                for r in batch:
                    self.emit_first_token(r)
                    if r.output_len <= 1:
                        self.kv_used -= self.kv_need(r)
                        self.sim.finish(r)
                    else:
                        r.decode_start = self.env.now
                        self.running.append(r)
                continue
            if self.running:
                yield from self.decode_once()
            else:
                yield from self.idle()


# ─────────────────────────────────────────────── brief 20A1: levers ──
BATCH_POLICIES = ("prefill-priority", "decode-priority", "chunked")
KV_POLICIES = ("oracle", "pow2", "max", "paged")


class _Node:
    __slots__ = ("key", "tokens", "units", "parent", "children", "refs", "last", "seq")

    def __init__(self, key, tokens, units, parent, now, seq):
        self.key, self.tokens, self.units, self.parent = key, tokens, units, parent
        self.children, self.refs, self.last, self.seq = 0, 0, now, seq


class PrefixCache:
    """Shared prompt segments kept in KV memory (a radix tree whose edges are whole segments).

    A request's ``prefix`` chain hits the longest run of leading segments that are cached. Segments a
    running request uses are pinned (``refs``); unpinned leaves are evicted least recently used first
    (ties: oldest insertion), as SGLang's RadixAttention does (arXiv:2312.07104, section 3). Memory is
    counted in the instance's units (KV blocks when paged, so a segment rounds up to whole blocks).
    """

    def __init__(self):
        self.nodes: dict[str, _Node] = {}
        self.heap: list = []           # (last, seq, key) of unpinned leaves; stale entries skipped
        self.seq = 0
        self.evicted_tokens = 0
        self.evicted_units = 0

    def match(self, chain) -> list[_Node]:
        out = []
        for key, _ in chain:
            n = self.nodes.get(key)
            if n is None:
                break
            out.append(n)
        return out

    def _maybe_leaf(self, n: _Node) -> None:
        if n.refs == 0 and n.children == 0:
            heapq.heappush(self.heap, (n.last, n.seq, n.key))

    def ref(self, n: _Node, now: float) -> None:
        n.refs += 1
        n.last = now

    def unref(self, n: _Node, now: float) -> None:
        n.refs -= 1
        n.last = now
        self._maybe_leaf(n)

    def insert(self, key: str, tokens: int, units: int, parent: _Node | None, now: float) -> _Node:
        """A new segment, pinned by the request that computed it."""
        n = _Node(key, tokens, units, parent, now, self.seq)
        self.seq += 1
        n.refs = 1
        if parent is not None:
            parent.children += 1
        self.nodes[key] = n
        return n

    def evict(self, units: int) -> int:
        """Evict unpinned leaves, LRU first, until ``units`` are freed or none is left; return the units freed."""
        freed = 0
        while freed < units and self.heap:
            last, seq, key = heapq.heappop(self.heap)
            n = self.nodes.get(key)
            if n is None or n.seq != seq or n.last != last or n.refs or n.children:
                continue                     # stale entry
            del self.nodes[key]
            freed += n.units
            self.evicted_tokens += n.tokens
            self.evicted_units += n.units
            if n.parent is not None:
                n.parent.children -= 1
                self._maybe_leaf(n.parent)
        return freed

    @property
    def cached_tokens(self) -> int:
        return sum(n.tokens for n in self.nodes.values())


class ScheduledInstance(Instance):
    """A colocated instance with the brief-20A1 levers (used only when ``SimConfig.scheduled``).

    Batching policy, one step at a time:

    * ``prefill-priority`` (vLLM v0, as ``ColocatedInstance``): admit waiting prompts whole, up to the
      token budget, and prefill them; decode only when nothing can be admitted.
    * ``decode-priority`` (request-level batching, FasterTransformer-style): decode while anything is
      running; admit new prompts only when the batch has drained (Sarathi-Serve section 3.2).
    * ``chunked`` (Sarathi-Serve stall-free batching, arXiv:2403.02310 Algorithm 3): every running decode
      row, then the unfinished prefill chunks, then new requests, each chunk cut to what is left of the
      token budget ``max_num_batched_tokens``; one forward pass (``CostModel.step_mixed``).

    KV memory, in units (tokens, or blocks of ``kv_block_size`` when paged): ``oracle`` reserves prompt +
    output at admission (as before; vLLM's "Orca (Oracle)"), ``pow2`` the prompt + the output rounded up
    to a power of two, ``max`` ``max_seq_len`` (vLLM's Orca baselines, arXiv:2309.06180 section 6.1).
    ``paged`` allocates the prompt's blocks at admission, keeps ``kv_watermark`` free, and adds a block
    when a sequence fills its last one; when none is free it preempts the latest-arrived running request
    (vLLM section 4.5): ``recompute`` frees its blocks and requeues it at the front, to prefill prompt +
    generated tokens again; ``swap`` copies its blocks to host memory over ``host_link`` (blocks x
    latency + bytes / bandwidth, added to the next step) and admits nothing new until it is back.
    A request whose prefill has not finished is always recomputed.

    Prefix caching: a waiting request's prefix chain is matched against ``PrefixCache`` (at most prompt
    - 1 tokens are skipped: the last token is always computed), its prefill starts after the hit, and the
    segments it computes are inserted when its prefill ends; its own segment (rest of prompt + output)
    when it finishes. Not with swap preemption (ValueError).
    """

    role = "colocated"

    def __init__(self, sim: "Simulation", idx: int):
        super().__init__(sim, idx)
        cfg = sim.cfg
        self.policy = cfg.batch_policy
        self.budget = cfg.max_num_batched_tokens if cfg.max_num_batched_tokens is not None else cfg.max_prefill_tokens
        self.paged = cfg.kv_policy == "paged"
        self.blk = cfg.kv_block_size if self.paged else 1
        self.kv_cap = self.kv_cap // self.blk                       # in units
        self.watermark = int(cfg.kv_watermark * self.kv_cap) if self.paged else 0
        self.swap = self.paged and cfg.preemption == "swap"
        self.cache = PrefixCache() if cfg.prefix_caching else None
        self.swapped: list[Request] = []
        self.swap_t = 0.0                       # host-link seconds owed by the next step
        # statistics
        self.preemptions = 0
        self.recompute_tokens = 0
        self.swap_out_bytes = self.swap_in_bytes = 0.0
        self.swap_time = 0.0
        self.swap_j = 0.0
        self.prompt_tokens = 0                  # first admissions: prompt tokens, and those found cached
        self.hit_tokens = 0
        self.computed_tokens = 0                # prefill tokens actually computed (recomputes included)
        self.run_area = 0.0                     # integral of running requests over busy time
        self.tok_area = 0.0                     # integral of KV token states held privately
        self.alloc_area = 0.0                   # integral of KV tokens allocated privately

    def load(self) -> float:
        """Queued plus decoding rows: a prompt in flight is not counted, exactly as ColocatedInstance."""
        return len(self.queue) + len(self.swapped) + sum(1 for r in self.running if r.done_prompt >= r.target)

    # ── memory ──
    def units(self, tokens: int) -> int:
        return -(-tokens // self.blk)

    def full_need(self, r: Request) -> int:
        """Units the whole sequence needs; more than the capacity (less the watermark) is rejected."""
        cfg, p, o = self.sim.cfg, r.prompt_len, r.output_len
        if self.paged:
            return self.units(p + o) + self.watermark
        if cfg.kv_policy == "pow2":
            return p + (1 << (o - 1).bit_length())
        if cfg.kv_policy == "max":
            return max(cfg.max_seq_len, p + o)
        return p + o

    def admit_need(self, r: Request, hit: int) -> int:
        """Private units to allocate when ``r`` is admitted with ``hit`` cached tokens."""
        cfg = self.sim.cfg
        if self.paged:
            return self.units(r.target - hit)
        if cfg.kv_policy == "pow2":
            return r.prompt_len - hit + (1 << (r.output_len - 1).bit_length())
        if cfg.kv_policy == "max":
            return max(cfg.max_seq_len, r.prompt_len + r.output_len) - hit
        return r.prompt_len - hit + r.output_len

    def room(self, need: int) -> bool:
        """Free ``need`` units, evicting cached segments if that is what it takes."""
        short = self.kv_used + need - self.kv_cap
        if short > 0 and self.cache is not None:
            self.kv_used -= self.cache.evict(short)
        return self.kv_used + need <= self.kv_cap

    def set_alloc(self, r: Request, units: int) -> None:
        self.kv_used += units - r.alloc
        r.alloc = units

    def release(self, r: Request) -> None:
        """Free a request's private units and unpin its cached segments."""
        self.kv_used -= r.alloc
        r.alloc = 0
        if r.nodes:
            now = self.env.now
            for n in r.nodes:
                self.cache.unref(n, now)
            r.nodes = []
        r.covered = 0

    # ── admission ──
    def lookup(self, r: Request) -> tuple[list, int]:
        if self.cache is None or not r.prefix:
            return [], 0
        nodes = self.cache.match(r.prefix)
        hit = sum(n.tokens for n in nodes)
        while nodes and hit > r.target - 1:       # always compute the last prompt token
            hit -= nodes.pop().tokens
        return nodes, hit

    def try_admit(self) -> Request | None:
        """Admit the head of the queue if its memory fits (plus the watermark); None otherwise."""
        r = self.queue[0]
        if r.target == 0:
            r.target = r.prompt_len
        nodes, hit = self.lookup(r)
        need = self.admit_need(r, hit)
        now = self.env.now
        for n in nodes:                       # pin the hits first, so making room cannot evict them
            self.cache.ref(n, now)
        if not self.room(need + self.watermark):
            for n in nodes:
                self.cache.unref(n, now)
            return None
        self.queue.popleft()
        r.nodes, r.covered = nodes, hit
        r.alloc = 0
        self.set_alloc(r, need)
        r.done_prompt = hit
        r.base_out = r.tokens_out
        if r.prefill_start is None:
            r.prefill_start = now
            self.prompt_tokens += r.prompt_len
            self.hit_tokens += hit
        self.running.append(r)
        self.running.sort(key=_by_arrival)
        return r

    # ── preemption (paged) ──
    def grow(self) -> None:
        """Before a step, every decoding row needs room for its context; preempt LIFO when out of blocks."""
        if not self.paged:
            return
        i = 0
        while i < len(self.running):
            r = self.running[i]
            if r.done_prompt < r.target:          # still prefilling: its prompt blocks are allocated
                i += 1
                continue
            need = self.units(r.prompt_len + r.tokens_out - r.covered)
            if need <= r.alloc:
                i += 1
                continue
            while not self.room(need - r.alloc):
                victim = self.running[-1]
                self.preempt(victim)
                if victim is r:
                    break
            else:
                self.set_alloc(r, need)
                i += 1

    def preempt(self, r: Request) -> None:
        _remove(self.running, r)
        self.preemptions += 1
        if self.swap and r.done_prompt >= r.target:
            host = self.sim.cfg.host_link
            nbytes = r.alloc * self.blk * self.sim.cfg.model.kv_bytes_per_token
            t = r.alloc * host.latency + nbytes / host.bandwidth
            self.swap_t += t
            self.swap_time += t
            self.swap_out_bytes += nbytes
            self.swap_j += nbytes * 8 * host.pj_per_bit * 1e-12
            r.swapped_units = r.alloc
            self.kv_used -= r.alloc
            r.alloc = 0
            self.swapped.append(r)
            self.swapped.sort(key=_by_arrival)
            return
        self.release(r)
        r.target = r.prompt_len + r.tokens_out
        r.done_prompt = 0
        self.queue.appendleft(r)

    def swap_in(self) -> None:
        host = self.sim.cfg.host_link
        while self.swapped and len(self.running) < self.sim.cfg.max_decode_batch:
            r = self.swapped[0]
            if not self.room(r.swapped_units):
                break
            self.swapped.pop(0)
            nbytes = r.swapped_units * self.blk * self.sim.cfg.model.kv_bytes_per_token
            t = r.swapped_units * host.latency + nbytes / host.bandwidth
            self.swap_t += t
            self.swap_time += t
            self.swap_in_bytes += nbytes
            self.swap_j += nbytes * 8 * host.pj_per_bit * 1e-12
            self.kv_used += r.swapped_units
            r.alloc, r.swapped_units = r.swapped_units, 0
            self.running.append(r)
            self.running.sort(key=_by_arrival)

    # ── prefix segments ──
    def insert_chain(self, r: Request) -> None:
        """After its prefill: cache the chain segments that were not hits."""
        if self.cache is None:
            return
        now, k = self.env.now, len(r.nodes)
        for key, tokens in r.prefix[k:]:
            parent = r.nodes[-1] if r.nodes else None
            n = self.cache.nodes.get(key)
            if n is not None and n.parent is parent:
                self.cache.ref(n, now)         # another request computed it meanwhile: share it
            else:
                if n is not None:
                    break
                units = self.units(tokens)
                if not self.room(units + self.private_units(r, r.covered + tokens) - r.alloc):
                    break
                n = self.cache.insert(key, tokens, units, parent, now)
                self.kv_used += units
            self.set_alloc(r, self.private_units(r, r.covered + tokens))
            r.nodes.append(n)
            r.covered += tokens

    def private_units(self, r: Request, covered: int) -> int:
        if self.paged:
            return self.units(max(0, r.done_prompt + r.tokens_out - r.base_out - covered))
        return r.alloc - (covered - r.covered)

    def emit(self, r: Request) -> None:
        """On finish: cache this request's own segment (rest of prompt + output) for the next turn."""
        if self.cache is None or r.emit_key is None or len(r.nodes) != len(r.prefix):
            return
        tokens = r.prompt_len - r.covered + r.output_len
        units = self.units(tokens)
        if r.emit_key in self.cache.nodes or not self.room(units - r.alloc):
            return
        n = self.cache.insert(r.emit_key, tokens, units, r.nodes[-1] if r.nodes else None, self.env.now)
        self.kv_used += units
        r.nodes.append(n)

    # ── steps ──
    def finish_row(self, r: Request) -> None:
        self.emit(r)
        self.release(r)
        self.sim.finish(r)

    def prefill_done(self, r: Request) -> None:
        """The last chunk of r's (re)prefill ran: cache its segments, emit a token, maybe finish."""
        self.insert_chain(r)
        if r.tokens_out:                       # a recompute: the pass also yields the next token
            now = self.env.now
            r.itls.append(now - r.last_token)
            r.tokens_out += 1
            r.last_token = now
        else:
            self.emit_first_token(r)
        if r.decode_start is None and r.output_len > 1:
            r.decode_start = self.env.now
        if r.tokens_out >= r.output_len:
            _remove(self.running, r)
            self.finish_row(r)

    def account(self, dt: float) -> None:
        tok = alloc = 0
        for r in self.running:
            tok += r.done_prompt - r.covered + r.tokens_out - r.base_out
            alloc += r.alloc
        self.run_area += dt * len(self.running)
        self.tok_area += dt * tok
        self.alloc_area += dt * (alloc * self.blk)

    def sched_step(self, cost, label: str, batch: int):
        if self.swap_t:
            cost = replace(cost, time=cost.time + self.swap_t)
            self.swap_t = 0.0
        self.account(cost.time)
        yield from self.step("step", cost, label, batch)

    def admit_prompts(self) -> list[Request]:
        """Prefill- or decode-priority: whole prompts up to the token budget."""
        batch, tokens = [], 0
        cap = self.sim.cfg.max_decode_batch
        while self.queue and len(self.running) < cap and not self.swapped:
            r = self.queue[0]
            t = (r.target or r.prompt_len) - self.lookup_hit(r)
            if batch and tokens + t > self.budget:
                break
            if self.try_admit() is None:
                break
            batch.append(r)
            tokens += r.target - r.done_prompt
        return batch

    def lookup_hit(self, r: Request) -> int:
        if r.target == 0:
            r.target = r.prompt_len
        return self.lookup(r)[1]

    def prefill_batch(self, batch: list[Request]):
        chunks = [(r.done_prompt, r.target - r.done_prompt, True) for r in batch]
        tokens = sum(c for _, c, _ in chunks)
        self.computed_tokens += tokens
        self.recompute_tokens += sum(c for r, (_, c, _) in zip(batch, chunks) if r.tokens_out)
        cost = self.cost.step_mixed(0, 0, chunks)
        yield from self.sched_step(cost, f"prefill n={len(batch)} tok={tokens}", len(batch))
        for r in batch:
            r.done_prompt = r.target
            self.prefill_done(r)

    def decode_all(self):
        self.grow()
        if not self.running:
            return
        rows = list(self.running)
        ctx = sum(r.prompt_len + r.tokens_out for r in rows)
        yield from self.sched_step(self.cost.decode_sum(ctx, len(rows)), f"decode b={len(rows)}", len(rows))
        self.after_decode(rows)

    def after_decode(self, rows: list[Request]) -> None:
        now = self.env.now
        for r in rows:
            r.itls.append(now - r.last_token)
            r.tokens_out += 1
            r.last_token = now
            if r.tokens_out >= r.output_len:
                _remove(self.running, r)
                self.finish_row(r)

    def chunked_step(self):
        self.grow()
        tau, cap = self.budget, self.sim.cfg.max_decode_batch
        dec = [r for r in self.running if r.done_prompt >= r.target]
        nt, ctx = len(dec), sum(r.prompt_len + r.tokens_out for r in dec)
        part = []
        for r in self.running:
            if r.done_prompt < r.target and nt < tau:
                c = min(r.target - r.done_prompt, tau - nt)
                part.append((r, c))
                nt += c
        while self.queue and nt < tau and len(self.running) < cap and not self.swapped:
            r = self.try_admit()
            if r is None:
                break
            c = min(r.target - r.done_prompt, tau - nt)
            part.append((r, c))
            nt += c
        if not dec and not part:
            return False
        chunks = [(r.done_prompt, c, r.done_prompt + c == r.target) for r, c in part]
        pre = sum(c for _, c in part)
        self.computed_tokens += pre
        self.recompute_tokens += sum(c for r, c in part if r.tokens_out)
        cost = self.cost.step_mixed(ctx, len(dec), chunks)
        yield from self.sched_step(cost, f"step b={len(dec)} chunks={len(part)} tok={nt}", len(dec) + len(part))
        self.after_decode(dec)
        for r, c in part:
            r.done_prompt += c
            if r.done_prompt == r.target:
                self.prefill_done(r)
        return True

    def run(self):
        while True:
            if self.swapped:
                self.swap_in()
            if self.policy == "chunked":
                stepped = yield from self.chunked_step()
                if not stepped:
                    yield from self.idle()
                continue
            if self.policy == "decode-priority" and self.running:
                yield from self.decode_all()
                continue
            batch = self.admit_prompts()
            if batch:
                yield from self.prefill_batch(batch)
            elif self.running:
                yield from self.decode_all()
            else:
                yield from self.idle()


def _remove(rows: list, r: Request) -> None:
    """Remove by identity (Request equality compares every field)."""
    for i, x in enumerate(rows):
        if x is r:
            del rows[i]
            return


def _by_arrival(r: Request):
    return (r.arrival, r.rid)


# ──────────────────────────────────────────────────────────── simulation ──
@dataclass
class LinkStats:
    busy: float = 0.0
    bytes: float = 0.0               # bytes on the link (after any compression)
    transfers: int = 0
    wait: float = 0.0
    energy: float = 0.0              # the link's own energy
    handoff_bytes: float = 0.0       # hand-off bytes before compression
    transit_j: float = 0.0           # in-transit stage energy
    transit_bound: int = 0           # transfers the in-transit stage's compute slowed


@dataclass
class SimResult:
    cfg: SimConfig
    requests: list[Request]
    instances: list[Instance]
    link: LinkStats
    horizon: float
    in_system_area: float                       # ∫ N(t) dt, for Little's law
    samples: list[dict] = field(repr=False, default_factory=list)
    trace: dict | None = field(repr=False, default=None)
    rejected: list[Request] = field(default_factory=list)


class Simulation:
    def __init__(self, cfg: SimConfig, workload: list[Request]):
        self.cfg = cfg
        self.env = simpy.Environment()
        self.cost = self.cost_for("colocated")
        self.tracer = Tracer(enabled=cfg.trace)
        self.requests = workload
        self.rejected: list[Request] = []
        self.done = self.env.event()
        self.n_done = 0
        self.n_in_system = 0
        self._area, self._area_t = 0.0, 0.0
        self.samples: list[dict] = []
        # closed-loop sessions (Request.after): released when their predecessor finishes
        self.successor = {r.after: r for r in workload if r.after is not None}
        self.roots = [r for r in workload if r.after is None] if self.successor else workload
        if self.successor and cfg.fast_forward:
            raise ValueError("closed-loop sessions (Request.after) need exact finish times: not with fast_forward")

        if cfg.kv_transit is not None and cfg.mode != "disagg":
            raise ValueError("kv_transit compresses the prefill-to-decode hand-off: mode='disagg' only")
        ced_on_decode = cfg.model.is_ced and cfg.model.ced_replay_on == "decode"
        if ced_on_decode and cfg.mode != "disagg":
            raise ValueError("ced_replay_on='decode' splits prefill across the two pools: mode='disagg' only")
        if ced_on_decode and cfg.fast_forward:
            raise ValueError("fast_forward does not model the CED replay step on decode")
        self.scheduled = cfg.scheduled
        if self.scheduled:
            self.check_levers(cfg)
        if cfg.mode == "disagg":
            self.prefill = [PrefillInstance(self, i) for i in range(cfg.n_prefill)]
            dec = FastDecodeInstance if cfg.fast_forward else DecodeInstance
            if ced_on_decode:
                dec = CedDecodeInstance
            self.decode = [dec(self, i) for i in range(cfg.n_decode)]
            self.instances: list[Instance] = self.prefill + self.decode
        elif cfg.mode == "colocated":
            cls = ScheduledInstance if self.scheduled else ColocatedInstance
            self.colocated = [cls(self, i) for i in range(cfg.n_colocated)]
            self.instances = list(self.colocated)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")
        self.link = simpy.Resource(self.env, capacity=cfg.link.channels)
        self.link_stats = LinkStats()
        self.env.process(self.arrivals())
        self.env.process(self.sampler())

    @staticmethod
    def check_levers(cfg: SimConfig) -> None:
        """The brief-20A1 levers: colocated attention models only, valid names, no cache with swap."""
        if cfg.batch_policy not in BATCH_POLICIES:
            raise ValueError(f"unknown batch_policy {cfg.batch_policy!r}")
        if cfg.kv_policy not in KV_POLICIES:
            raise ValueError(f"unknown kv_policy {cfg.kv_policy!r}")
        if cfg.preemption not in ("recompute", "swap"):
            raise ValueError(f"unknown preemption {cfg.preemption!r}")
        if cfg.mode != "colocated":
            raise ValueError("batch_policy, max_num_batched_tokens, kv_policy and prefix_caching are modelled for "
                             "mode='colocated' only (disaggregated pools keep whole prompts and reserved KV)")
        if not cfg.model.is_transformer or cfg.model.is_ced:
            raise ValueError("the brief-20A1 levers are modelled for attention models without CED only")
        if cfg.kv_block_size < 1 or (cfg.max_num_batched_tokens is not None and cfg.max_num_batched_tokens < 1):
            raise ValueError("kv_block_size and max_num_batched_tokens must be at least 1")
        if cfg.prefix_caching and cfg.kv_policy == "paged" and cfg.preemption == "swap":
            raise ValueError("prefix caching with swap preemption is not modelled: use preemption='recompute'")

    def cost_for(self, role: str) -> CostModel:
        """Each pool gets its own cost model, so pools can have different power caps and,
        with ``prefill_device`` / ``decode_device``, different hardware."""
        cfg = self.cfg
        cap = {"prefill": cfg.prefill_power_cap_w, "decode": cfg.decode_power_cap_w}.get(role)
        dev, n = cfg.pool(role)
        return CostModel(cfg.model, dev, n, cfg.step_overhead,
                         power_cap_w=cap if cap is not None else cfg.power_cap_w, dvfs=cfg.dvfs,
                         prefill_only=role == "prefill")

    def endpoint_compress(self, cm: CostModel, cost, batch: list[Request]):
        """Compress each request's hand-off on the prefill GPU after the step."""
        tr, m = self.cfg.kv_transit, self.cfg.model
        if tr.compression.ratio == 1.0 and not tr.compression.ops_per_value:
            return cost                       # "none": nothing to do
        ops = nbytes = 0.0
        for r in batch:
            if r.output_len <= 1 and not cm.encoder_only:
                continue                      # finished at prefill: nothing to hand off
            hb = m.handoff_bytes(r.prompt_len)
            ops += tr.ops(hb, r.prompt_len, m.kv_bytes, True)
            nbytes += hb + hb / tr.compression.ratio      # read the KV, write the compressed copy
        return cm.compress_pass(cost, ops, nbytes) if nbytes else cost

    # ── population accounting (exact time-average for Little's law) ──────
    def _population(self, delta: int) -> None:
        now = self.env.now
        self._area += self.n_in_system * (now - self._area_t)
        self._area_t = now
        self.n_in_system += delta

    def finish(self, r: Request, t: float | None = None) -> None:
        r.finish = self.env.now if t is None else t
        self._population(-1)
        self.n_done += 1
        if self.successor:
            nxt = self.successor.get(r.rid)
            if nxt is not None:
                ev = self.env.timeout(nxt.think)
                ev.callbacks.append(lambda _e, x=nxt: self._release(x))
        if self.n_done + len(self.rejected) == len(self.requests):
            self.done.succeed()

    # ── processes ────────────────────────────────────────────────────────
    def arrivals(self):
        for r in self.roots:
            yield self.env.timeout(max(0.0, r.arrival - self.env.now))
            self.arrive(r)

    def arrive(self, r: Request) -> None:
        self._population(+1)
        if self.scheduled:
            inst = self.colocated[0]
            if inst.full_need(r) > inst.kv_cap:
                self._reject(r)
                return
            min(self.colocated, key=lambda i: i.load()).submit(r)
            return
        units = self.cfg.model.cache_units(r.prompt_len, r.output_len)
        if self.cfg.mode == "disagg":
            if units > self.decode[0].kv_cap:
                self._reject(r)
                return
            min(self.prefill, key=lambda i: i.load()).submit(r)
        else:
            if units > self.colocated[0].kv_cap:
                self._reject(r)
                return
            min(self.colocated, key=lambda i: i.load()).submit(r)

    def _release(self, r: Request) -> None:
        """A closed-loop turn: it arrives now, ``think`` seconds after its predecessor finished."""
        r.arrival = self.env.now
        self.arrive(r)

    def _reject(self, r: Request) -> None:
        self._population(-1)
        self.rejected.append(r)
        nxt = self.successor.get(r.rid)
        while nxt is not None:                # its later turns can never be sent
            self.rejected.append(nxt)
            nxt = self.successor.get(nxt.rid)
        if self.n_done + len(self.rejected) == len(self.requests):
            self.done.succeed()

    def transfer(self, r: Request) -> tuple[float, float, float, float, bool]:
        """(seconds, bytes on the link, link joules, in-transit joules, transit-bound)."""
        link, tr, m = self.cfg.link, self.cfg.kv_transit, self.cfg.model
        nbytes = m.handoff_bytes(r.prompt_len)
        if tr is None:
            return link.transfer_time(nbytes), nbytes, nbytes * 8 * link.pj_per_bit * 1e-12, 0.0, False
        wire = nbytes / tr.compression.ratio
        if tr.where == "endpoint":            # compressed on the GPU already
            return link.transfer_time(wire), wire, wire * 8 * link.pj_per_bit * 1e-12, 0.0, False
        # In transit: the stage streams at line rate with a budget of ops_per_byte per line
        # byte; if its work does not fit, its compute time sets the transfer time.
        t_wire = wire / link.bandwidth
        t_ops = tr.ops(nbytes, r.prompt_len, m.kv_bytes, False) / (tr.ops_per_byte * link.bandwidth)
        t = link.latency + tr.latency + max(t_wire, t_ops)
        return (t, wire, wire * 8 * link.pj_per_bit * 1e-12, nbytes * 8 * tr.pj_per_bit * 1e-12,
                t_ops > t_wire)

    def kv_transfer(self, r: Request):
        stats = self.link_stats
        with self.link.request() as grant:
            yield grant
            r.kv_start = self.env.now
            stats.wait += r.kv_start - r.handoff_start
            t, nbytes, joules, transit_j, transit_bound = self.transfer(r)
            yield self.env.timeout(t)
        stats.busy += t
        stats.bytes += nbytes
        stats.transfers += 1
        stats.energy += joules
        if self.cfg.kv_transit is not None:
            stats.handoff_bytes += self.cfg.model.handoff_bytes(r.prompt_len)
            stats.transit_j += transit_j
            stats.transit_bound += transit_bound
        r.kv_ready = self.env.now
        self.tracer.span("kv-link", "kv", f"kv r{r.rid} {nbytes / 1e6:.0f} MB", r.kv_start, t)
        min(self.decode, key=lambda i: i.load()).submit(r)

    def sampler(self):
        while True:
            row = {"t": self.env.now, "in_system": self.n_in_system,
                   "link_queue": len(self.link.queue), "link_busy": self.link.count}
            for inst in self.instances:
                row[f"{inst.name}.queue"] = len(inst.queue)
                row[f"{inst.name}.running"] = len(inst.running)
                row[f"{inst.name}.kv"] = inst.kv_used / inst.kv_cap
            self.samples.append(row)
            self.tracer.counter("population", {"in_system": self.n_in_system,
                                               "link_queue": len(self.link.queue)}, self.env.now)
            yield self.env.timeout(self.cfg.sample_dt)

    def run(self) -> SimResult:
        self.env.run(until=self.done)
        self._population(0)
        return SimResult(self.cfg, self.requests, self.instances, self.link_stats,
                         self.env.now, self._area, self.samples,
                         self.tracer.export() if self.cfg.trace else None, self.rejected)


def simulate(cfg: SimConfig, workload: list[Request]) -> SimResult:
    """Convenience wrapper: build, run, return. ``workload`` is mutated in place."""
    return Simulation(cfg, workload).run()
