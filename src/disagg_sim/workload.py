"""Requests and the synthetic workloads that generate them."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Request:
    """One inference request, plus every timestamp the simulator stamps on it.

    Stamps are ``None`` until the request reaches that point. In colocated mode
    the KV-transfer stamps stay ``None`` because there is no transfer.
    """

    rid: int
    arrival: float
    prompt_len: int
    output_len: int

    prefill_start: float | None = None
    first_token: float | None = None
    prefill_done: float | None = None  # CED with the replay on decode: prefill ends before the first token
    kv_start: float | None = None      # link granted
    kv_ready: float | None = None      # KV landed on the decode instance
    decode_start: float | None = None  # admitted into a decode batch
    finish: float | None = None

    tokens_out: int = 0
    last_token: float | None = None
    itls: list[float] = field(default_factory=list, repr=False)
    adm_step: int | None = field(default=None, repr=False)   # used by FastDecodeInstance

    # Shared prefixes (brief 20A1; read only when SimConfig.prefix_caching is on). ``prefix`` is the
    # chain of cacheable segments the prompt starts with, ((key, tokens), ...) from the root (a system
    # prompt, then earlier turns of the same session); ``emit_key`` names the segment this request adds
    # (the rest of its prompt plus its output), which the next turn's chain ends with.
    prefix: tuple = field(default=(), repr=False)
    emit_key: str | None = field(default=None, repr=False)
    # Closed-loop sessions: released ``think`` seconds after request ``after`` (a rid) finishes.
    after: int | None = field(default=None, repr=False)
    think: float = field(default=0.0, repr=False)
    # ScheduledInstance state: prompt tokens to (re)compute, done so far, tokens out at (re)admission,
    # private KV units held, pinned prefix segments and the tokens they cover, units parked in host memory
    target: int = field(default=0, repr=False)
    done_prompt: int = field(default=0, repr=False)
    base_out: int = field(default=0, repr=False)
    alloc: int = field(default=0, repr=False)
    nodes: list = field(default_factory=list, repr=False)
    covered: int = field(default=0, repr=False)
    swapped_units: int = field(default=0, repr=False)

    # ── derived latencies ────────────────────────────────────────────────
    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first (DistServe's definition)."""
        if self.output_len < 2:
            return None
        return (self.finish - self.first_token) / (self.output_len - 1)

    @property
    def e2e(self) -> float:
        return self.finish - self.arrival

    @property
    def handoff_start(self) -> float:
        """When prefill finished: the first token, except under CED with the replay on decode."""
        return self.prefill_done if self.prefill_done is not None else self.first_token

    def stages(self) -> dict[str, float]:
        """Where this request's time went, stage by stage. Sums to e2e."""
        handoff = self.kv_ready if self.kv_ready is not None else self.first_token
        s = {
            "prefill_queue": self.prefill_start - self.arrival,
            "prefill": self.handoff_start - self.prefill_start,
            "kv_wait": 0.0,
            "kv_transfer": 0.0,
            "decode_queue": 0.0,
            "decode": 0.0,
        }
        if self.kv_start is not None:
            s["kv_wait"] = self.kv_start - self.handoff_start
            s["kv_transfer"] = self.kv_ready - self.kv_start
        if self.decode_start is not None:
            s["decode_queue"] = self.decode_start - handoff
            s["decode"] = self.finish - self.decode_start
        return s


@dataclass(frozen=True)
class LengthDist:
    """Token-length distribution: lognormal with a given mean and coefficient of
    variation, clipped to [lo, hi]. ``cv=0`` gives a fixed length."""

    mean: float
    cv: float = 0.0
    lo: int = 1
    hi: int = 32768

    def sample(self, rng: random.Random) -> int:
        if self.cv <= 0:
            x = self.mean
        else:
            sigma2 = math.log(1 + self.cv ** 2)
            x = rng.lognormvariate(math.log(self.mean) - sigma2 / 2, math.sqrt(sigma2))
        return int(min(self.hi, max(self.lo, round(x))))


def poisson_workload(rate: float, n: int, prompt: LengthDist, output: LengthDist,
                     seed: int = 0) -> list[Request]:
    """``n`` requests with exponential inter-arrival times (Poisson process)."""
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)
        reqs.append(Request(i, t, prompt.sample(rng), output.sample(rng)))
    return reqs


def chat_sessions(rate: float, n: int, prompt: LengthDist, output: LengthDist, seed: int = 0, *,
                  turns: int = 1, think: float = 0.0, system_prompts: int = 0, system_len: int = 0,
                  share: float = 1.0) -> list[Request]:
    """``n`` requests in sessions with shared prefixes, for prefix caching (brief 20A1).

    Sessions start as a Poisson process at ``rate`` per second; each has ``turns`` turns (the last
    session is cut at ``n`` requests). ``prompt`` is the *new* input of a turn and ``output`` its reply.
    With probability ``share`` a session opens with one of ``system_prompts`` system prompts (chosen
    uniformly) of ``system_len`` tokens. Turn k's prompt is the system prompt, every earlier turn's input
    and output, then its own input, so its ``prefix`` chain is (system prompt, turn 0, ..., turn k-1).
    Turns after the first are closed-loop: released ``think`` x Exp(1) seconds after the previous turn
    finishes (``after`` / ``think``), so the reuse distance is set by the think time and the load.
    ``arrival`` of a released turn is a placeholder until the simulator releases it. With ``turns=1`` and
    no system prompt this draws the same requests as ``poisson_workload``.
    """
    rng = random.Random(seed)
    t, reqs = 0.0, []
    sid = 0
    while len(reqs) < n:
        t += rng.expovariate(rate)
        sys_seg = ()
        if system_prompts and system_len and (share >= 1.0 or rng.random() < share):
            sys_seg = ((f"sys{rng.randrange(system_prompts)}", system_len),)
        chain, hist = sys_seg, sum(x for _, x in sys_seg)
        prev = None
        for k in range(turns):
            if len(reqs) >= n:
                break
            new, out = prompt.sample(rng), output.sample(rng)
            gap = think * rng.expovariate(1.0) if (k and think > 0) else 0.0
            r = Request(len(reqs), t if k == 0 else 0.0, hist + new, out, prefix=chain,
                        emit_key=f"s{sid}t{k}" if turns > 1 else None, after=prev, think=gap)
            reqs.append(r)
            chain = chain + ((f"s{sid}t{k}", new + out),)
            hist += new + out
            prev = r.rid
        sid += 1
    return reqs


def _extras(r: Request) -> dict:
    return {"chain": [list(c) for c in r.prefix], "emit": r.emit_key, "after": r.after, "think": r.think}


def dump_workload(reqs: list[Request], path: str | Path) -> None:
    """Rows of [arrival, prompt, output]; a fourth column (prefix chain, emitted key, closed-loop
    predecessor and think time) only when some request has one (the JS port reads the same rows)."""
    Path(path).write_text(json.dumps(workload_rows(reqs)))


def workload_rows(reqs: list[Request]) -> list[list]:
    if not any(r.prefix or r.emit_key or r.after is not None for r in reqs):
        return [[r.arrival, r.prompt_len, r.output_len] for r in reqs]
    return [[r.arrival, r.prompt_len, r.output_len, _extras(r)] for r in reqs]


def load_workload(path: str | Path) -> list[Request]:
    rows = json.loads(Path(path).read_text())
    out = []
    for i, row in enumerate(rows):
        r = Request(i, float(row[0]), int(row[1]), int(row[2]))
        if len(row) > 3:
            x = row[3]
            r.prefix = tuple((k, int(v)) for k, v in x["chain"])
            r.emit_key, r.after, r.think = x["emit"], x["after"], float(x["think"])
        out.append(r)
    return out
