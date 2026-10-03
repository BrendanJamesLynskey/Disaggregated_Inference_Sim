"""Property-based tests: invariants checked over randomly generated configurations."""

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from disagg_sim import LengthDist, SimConfig, poisson_workload, simulate  # noqa: E402


@settings(max_examples=40, deadline=None)
@given(rate=st.floats(0.5, 8), n_prefill=st.integers(1, 3), n_decode=st.integers(1, 3),
       seed=st.integers(0, 10_000), mode=st.sampled_from(["disagg", "colocated"]),
       fast=st.booleans())
def test_conservation_for_any_config(rate, n_prefill, n_decode, seed, mode, fast):
    cfg = SimConfig(mode=mode, n_prefill=n_prefill, n_decode=n_decode, fast_forward=fast)
    wl = poisson_workload(rate, 100, LengthDist(1024, 1.0), LengthDist(64, 1.0), seed)
    simulate(cfg, wl)
    for r in wl:
        assert r.tokens_out == r.output_len
        assert len(r.itls) == r.output_len - 1
        assert abs(sum(r.stages().values()) - r.e2e) < 1e-9


@settings(max_examples=200, deadline=None)
@given(xs=st.lists(st.floats(0, 10, allow_nan=False) | st.sampled_from([0.0, 1.5, 2.25]), max_size=60))
def test_sort_once_summary_equals_sorting_per_percentile(xs):
    """``_dist`` sorts once for p50/p90/p99; every value must equal the old sort-per-percentile one."""
    import math
    from statistics import fmean

    from disagg_sim.metrics import _dist, percentile

    old = {"mean": fmean(xs) if xs else math.nan, "p50": percentile(xs, 50), "p90": percentile(xs, 90),
           "p99": percentile(xs, 99), "max": max(xs) if xs else math.nan}
    new = _dist(xs)
    assert [repr(new[k]) for k in old] == [repr(old[k]) for k in old]
