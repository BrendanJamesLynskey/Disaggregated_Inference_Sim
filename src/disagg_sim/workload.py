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


def dump_workload(reqs: list[Request], path: str | Path) -> None:
    rows = [[r.arrival, r.prompt_len, r.output_len] for r in reqs]
    Path(path).write_text(json.dumps(rows))


def load_workload(path: str | Path) -> list[Request]:
    rows = json.loads(Path(path).read_text())
    return [Request(i, float(a), int(p), int(o)) for i, (a, p, o) in enumerate(rows)]
