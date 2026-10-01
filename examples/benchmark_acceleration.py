"""Measure the simulator-acceleration techniques used in this repo.

    python examples/benchmark_acceleration.py

Each technique is checked for exactness (or, for search, for agreement with a
brute-force grid) as well as for speed: a faster simulator that gives a
different answer is a different simulator.
"""

import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from disagg_sim import LengthDist, SimConfig, poisson_workload, simulate  # noqa: E402
from disagg_sim.search import (Workload, analytic_capacity, max_sustainable_rate,  # noqa: E402
                               sweep)


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


def section(title):
    print(f"\n── {title} " + "─" * (66 - len(title)))


def main():
    prompt, output = LengthDist(2048, 0.5), LengthDist(256, 0.5)
    cfg = SimConfig()

    section("1 · event abstraction (one event per batch step)")
    reqs = poisson_workload(4.0, 2000, prompt, output, seed=1)
    res, t = timed(simulate, cfg, reqs)
    steps = sum(i.steps for i in res.instances)
    tokens = sum(r.output_len for r in reqs)
    print(f"simulated {res.horizon:.0f} s of serving in {t:.2f} s ({res.horizon / t:.0f}x real time)")
    print(f"batch-step events: {steps:,}   per-token events would be {tokens:,} "
          f"({tokens / steps:.0f}x)   per-layer events {steps * cfg.model.n_layers:,} "
          f"({cfg.model.n_layers}x)")

    section("2 · probe overhead (sampling period)")
    for dt in (0.005, 0.05, 1e9):
        _, t = timed(simulate, replace(cfg, sample_dt=dt), poisson_workload(4.0, 2000, prompt, output, seed=1))
        print(f"sample_dt={dt:<8g} {t:.2f} s")

    section("3 · exact fast path: incremental state + lazy bookkeeping + macro-steps")
    for name, c, o, rate in [("defaults", cfg, output, 4.0),
                             ("1k-token outputs, 1 req/s", cfg, LengthDist(1024, 0.5), 1.0),
                             ("2P2D", replace(cfg, n_prefill=2, n_decode=2), output, 6.0)]:
        a = poisson_workload(rate, 2000, prompt, o, seed=1)
        b = poisson_workload(rate, 2000, prompt, o, seed=1)
        _, t1 = timed(simulate, c, a)
        _, t2 = timed(simulate, replace(c, fast_forward=True), b)
        diff = max(max(abs(x.finish - y.finish), abs(x.first_token - y.first_token),
                       max((abs(p - q) for p, q in zip(x.itls, y.itls)), default=0.0))
                   for x, y in zip(a, b))
        print(f"{name:<28} baseline {t1:.2f} s   fast {t2:.2f} s   {t1 / t2:.1f}x   max |diff| {diff:.1e} s")

    section("4 · parallel sweep (independent runs)")
    wl = Workload(prompt, output, n=800)
    rates = [1, 2, 3, 4, 5, 6, 7, 8]
    fast = replace(cfg, fast_forward=True)
    serial, t1 = timed(sweep, fast, wl, rates, workers=1)
    workers = min(8, os.cpu_count() or 1)
    par, t2 = timed(sweep, fast, wl, rates, workers=workers)
    assert serial == par
    print(f"{len(rates)} runs: serial {t1:.2f} s   {workers} processes {t2:.2f} s   {t1 / t2:.1f}x (identical results)")

    section("5 · search: analytic bracket + bisection vs a fine grid")
    cap = analytic_capacity(fast, wl)
    print("analytic ceilings (req/s): " + ", ".join(
        f"{k} {v:.2f}" for k, v in cap.items() if k not in ("bottleneck", "ceiling"))
        + f"  -> bottleneck {cap['bottleneck']}")
    found, t1 = timed(max_sustainable_rate, fast, wl, target=0.9, rel_tol=0.02)
    grid = [round(0.25 * i, 2) for i in range(1, 41)]                      # 0.25 .. 10 req/s
    gres, t2 = timed(sweep, fast, wl, grid)
    best = max((r for r, a in gres if a >= 0.9), default=0.0)
    print(f"bisection: {found['rate']:.2f} req/s in {found['simulations']} simulations, {t1:.2f} s")
    print(f"grid:      {best:.2f} req/s in {len(grid)} simulations, {t2:.2f} s   "
          f"({t2 / t1:.1f}x more time)")


if __name__ == "__main__":
    main()
