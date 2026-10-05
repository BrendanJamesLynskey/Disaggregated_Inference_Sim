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
"""

from __future__ import annotations

import bisect
import heapq
from collections import deque
from dataclasses import dataclass, field

import simpy

from .hardware import (H100_SXM, LINKS, LLAMA3_70B, Accelerator, CostModel, KVTransit, Link,
                       ModelSpec)
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

        if cfg.kv_transit is not None and cfg.mode != "disagg":
            raise ValueError("kv_transit compresses the prefill-to-decode hand-off: mode='disagg' only")
        ced_on_decode = cfg.model.is_ced and cfg.model.ced_replay_on == "decode"
        if ced_on_decode and cfg.mode != "disagg":
            raise ValueError("ced_replay_on='decode' splits prefill across the two pools: mode='disagg' only")
        if ced_on_decode and cfg.fast_forward:
            raise ValueError("fast_forward does not model the CED replay step on decode")
        if cfg.mode == "disagg":
            self.prefill = [PrefillInstance(self, i) for i in range(cfg.n_prefill)]
            dec = FastDecodeInstance if cfg.fast_forward else DecodeInstance
            if ced_on_decode:
                dec = CedDecodeInstance
            self.decode = [dec(self, i) for i in range(cfg.n_decode)]
            self.instances: list[Instance] = self.prefill + self.decode
        elif cfg.mode == "colocated":
            self.colocated = [ColocatedInstance(self, i) for i in range(cfg.n_colocated)]
            self.instances = list(self.colocated)
        else:
            raise ValueError(f"unknown mode {cfg.mode!r}")
        self.link = simpy.Resource(self.env, capacity=cfg.link.channels)
        self.link_stats = LinkStats()
        self.env.process(self.arrivals())
        self.env.process(self.sampler())

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
        if self.n_done + len(self.rejected) == len(self.requests):
            self.done.succeed()

    # ── processes ────────────────────────────────────────────────────────
    def arrivals(self):
        for r in self.requests:
            yield self.env.timeout(max(0.0, r.arrival - self.env.now))
            self._population(+1)
            units = self.cfg.model.cache_units(r.prompt_len, r.output_len)
            if self.cfg.mode == "disagg":
                if units > self.decode[0].kv_cap:
                    self._reject(r)
                    continue
                min(self.prefill, key=lambda i: i.load()).submit(r)
            else:
                if units > self.colocated[0].kv_cap:
                    self._reject(r)
                    continue
                min(self.colocated, key=lambda i: i.load()).submit(r)

    def _reject(self, r: Request) -> None:
        self._population(-1)
        self.rejected.append(r)
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
