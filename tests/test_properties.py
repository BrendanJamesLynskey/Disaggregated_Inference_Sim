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
