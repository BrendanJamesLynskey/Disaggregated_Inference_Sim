"""Accelerating the *experiment*, not just the run.

Architects rarely want one simulation; they want "what is the highest load this
configuration can sustain within SLO?" across many configurations. Three
techniques make that cheap:

* **Analytic bounds (multi-fidelity).** A closed-form capacity estimate costs
  microseconds and brackets the answer, so the expensive simulator only has to
  search inside the bracket.
* **Bisection instead of grids.** SLO attainment falls monotonically (to within
  noise) with load, so bisection finds the knee in ~log2(range/tol) runs.
* **Parallel replications.** Independent runs are embarrassingly parallel.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

from .hardware import CostModel
from .metrics import summarise
from .sim import SimConfig, simulate
from .workload import LengthDist, poisson_workload


@dataclass(frozen=True)
class Workload:
    prompt: LengthDist
    output: LengthDist
    n: int = 800
    seed: int = 1


def analytic_capacity(cfg: SimConfig, wl: Workload) -> dict:
    """Upper bounds on sustainable request rate, one per resource (req/s).

    Optimistic by construction: every pool runs flat out at its best batch and
    queueing is ignored. The minimum is the throughput ceiling; the simulator
    then finds how far below it the SLOs bite.
    """
    cm = CostModel(cfg.model, cfg.device, cfg.devices_per_instance, cfg.step_overhead,
                   power_cap_w=cfg.power_cap_w)
    p, o = wl.prompt.mean, wl.output.mean
    # prefill: batches of whole prompts up to the token budget
    per_batch = max(1, int(cfg.max_prefill_tokens // p))
    prefill_rate = per_batch / cm.prefill([int(p)] * per_batch).time
    # decode: the largest batch that fits in KV, at mean context halfway through generation
    b = max(1, min(cfg.max_decode_batch, int(cm.kv_capacity_tokens // (p + o))))
    step = cm.decode([int(p + o / 2)] * b).time
    decode_rate = b / step / max(1.0, o - 1)
    if cfg.mode == "disagg":
        nbytes = p * cfg.model.kv_bytes_per_token
        bounds = {"prefill": cfg.n_prefill * prefill_rate,
                  "decode": cfg.n_decode * decode_rate,
                  "kv-link": cfg.link.channels / cfg.link.transfer_time(nbytes)}
    else:
        # colocated: each instance splits its time between the phases
        per_inst = 1.0 / (1.0 / prefill_rate + 1.0 / decode_rate)
        bounds = {"colocated": cfg.n_colocated * per_inst}
    bounds["bottleneck"] = min(bounds, key=bounds.get)
    bounds["ceiling"] = bounds[bounds["bottleneck"]]
    return bounds


def _attainment(args) -> tuple[float, float]:
    cfg, wl, rate = args
    reqs = poisson_workload(rate, wl.n, wl.prompt, wl.output, seed=wl.seed)
    return rate, summarise(simulate(cfg, reqs))["throughput"]["slo_attainment"]


def sweep(cfg: SimConfig, wl: Workload, rates: list[float], workers: int = 1) -> list[tuple[float, float]]:
    """SLO attainment at each rate; ``workers > 1`` runs them in parallel processes."""
    jobs = [(cfg, wl, r) for r in rates]
    if workers <= 1:
        return [_attainment(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_attainment, jobs))


def max_sustainable_rate(cfg: SimConfig, wl: Workload, target: float = 0.9,
                         rel_tol: float = 0.02, lo: float = 0.0, hi: float | None = None) -> dict:
    """Highest Poisson rate whose SLO attainment is >= ``target``, by bisection.

    ``hi`` defaults to the analytic ceiling, so no simulation is spent above it.
    """
    if hi is None:
        hi = analytic_capacity(cfg, wl)["ceiling"]
    runs = 0
    # make sure the upper bracket really fails (the ceiling is optimistic, so it should)
    while _attainment((cfg, wl, hi))[1] >= target:
        runs += 1
        lo, hi = hi, hi * 1.5
    runs += 1
    while hi - lo > rel_tol * hi:
        mid = (lo + hi) / 2
        runs += 1
        if _attainment((cfg, wl, mid))[1] >= target:
            lo = mid
        else:
            hi = mid
    return {"rate": lo, "bracket": (lo, hi), "simulations": runs}


def with_fast(cfg: SimConfig) -> SimConfig:
    return replace(cfg, fast_forward=True)
