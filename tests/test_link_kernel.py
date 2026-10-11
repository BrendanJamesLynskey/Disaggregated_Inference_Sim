"""The link kernel (brief 06C): a declared compute-in-transit stage, off by default.

* hand formulas for every stage, the stream time and the energy;
* the budget and the kernel ledger reproduce the Compute in Transit hub's ``analysis/cit_budget.py`` results
  (vendored in tests/fixtures/cit_budget_hub.json from the hub at de8f33b);
* work that needs an operation the kernel lacks (top-k, requantisation) is refused, and so is combining streams
  in a pluggable;
* default off: KV hand-off runs without a kernel are identical to commit 69ffd2b
  (tests/fixtures/link_kernel_default_off.json, written by ``default_off_runs`` on that commit);
* the slowest stage binds, and the simulator counts which one.

Regenerate the default-off fixture on another commit with ``python tests/test_link_kernel.py <out.json>``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from disagg_sim.hardware import (CIT_LEDGER, GB, KV_PRESETS, LINK_KERNELS, LINKS, MODELS, KernelWork, KVTransit,
                                 LinkKernel, band_work, kv_work, pow2_at_least, transform_per_value)
from disagg_sim.metrics import summarise
from disagg_sim.sim import SimConfig, simulate
from disagg_sim.workload import LengthDist, poisson_workload

FIX = Path(__file__).parent / "fixtures"
HUB = json.loads((FIX / "cit_budget_hub.json").read_text())
LOW, HIGH = LINK_KERNELS["pluggable-10"], LINK_KERNELS["pluggable-99"]


def sig6(x: float) -> float:
    """The hub rounds results.json floats to 6 significant figures."""
    return x if x.is_integer() else float(f"{x:.6g}")


# ── calibration: the hub's budget ──────────────────────────────────────────────────────────────────────
def test_presets_are_off_the_hub_bracket():
    assert LOW.gflops == 10e9 and HIGH.gflops == 99e9
    assert LOW.port_rate == 25 * GB and LOW.optics_rate == 7 * GB
    assert LOW.static_w == 9.7 and LOW.pj_per_bit == 2.76
    assert "illustrative" in LOW.name


@pytest.mark.parametrize("row", HUB["budget"], ids=lambda r: f"{r['link']}|{r['claim']}")
def test_budget_reproduces_the_hub(row):
    k = LOW if row["claim"].startswith("low") else HIGH
    if row["link"].startswith("Prototype optics"):
        assert k.optics_rate == row["rate"]
        assert sig6(k.optics_budget) == row["ops_per_byte"]
    else:
        k = replace(k, port_rate=row["rate"])       # 400G and 800G: the same engine per link
        assert sig6(k.budget) == row["ops_per_byte"]


def test_headline_ranges():
    assert [LOW.budget, HIGH.budget] == HUB["qsfp56_range"] == [0.4, 3.96]
    assert [sig6(LOW.optics_budget), sig6(HIGH.optics_budget)] == HUB["prototype_optics"]["range"]
    assert f"{LOW.optics_budget:.2f}-{HIGH.optics_budget:.2f}" == "1.43-14.14"


def largest_charged_fft(budget: float) -> int | None:
    """Longest power-of-2 1-D band transform whose charged work per BF16 byte fits ``budget``."""
    best, n = None, 2
    while n <= 1 << 20:
        if LOW.charged_per_byte(band_work(n, 1 / 16)) <= budget:
            best = n
        n *= 2
    return best


def test_largest_fft_reproduces_the_hub():
    assert largest_charged_fft(LOW.budget) == HUB["largest_fft_bf16"]["low"]
    assert largest_charged_fft(HIGH.budget) == HUB["largest_fft_bf16"]["high"]
    po = HUB["prototype_optics"]["largest_fft_bf16"]
    assert largest_charged_fft(LOW.optics_budget) == po["low"]
    assert largest_charged_fft(HIGH.optics_budget) == po["high"]


@pytest.mark.parametrize("i", range(len(CIT_LEDGER)))
def test_kernel_ledger_reproduces_the_hub(i):
    w, row = CIT_LEDGER[i], HUB["kernels"][i]
    assert w.name == row["kernel"]
    assert sig6(w.transform_per_byte) == row["transform_per_byte"]
    assert sig6(w.other_per_byte) == row["other_per_byte"]
    sw = replace(LOW, placement="switch")         # in set: only placement-independent refusals count
    try:
        sw.check(w)
        in_set = True
    except ValueError:
        in_set = False
    assert in_set == row["in_set"]
    assert LOW.fits(w) == row["fits_low"]
    assert HIGH.fits(w) == row["fits_high"]
    assert replace(LOW, transform_charged=False).fits(w) == row["fits_free_low"]


# ── hand formulas ──────────────────────────────────────────────────────────────────────────────────────
def test_transform_flops_follow_fftw_convention():
    assert transform_per_value(1024) == 25.0
    assert transform_per_value(2) == 2.5


def test_stage_times_by_hand():
    k = LinkKernel(gflops=20e9, fpga_ops_per_s=40e9, planes=4)
    w = KernelWork("w", (("fft", 10.0), ("mac", 3.0), ("modular", 2.0)), ratio=2.0, fpga_ops_per_value=1.0)
    b = 1e9
    values = b / 2
    st = k.stage_times(b, w)
    assert st["module port"] == b / (25 * GB)
    assert st["module optics"] == values * 4 / 56e9
    assert st["module compute"] == b * (3.0 / 2 + 10.0 / 2) / 20e9          # transform charged by default
    assert st["FPGA"] == b * (2.0 / 2 + 1.0 / 2) / 40e9
    free = replace(k, transform_charged=False).stage_times(b, w)
    assert free["module compute"] == b * (3.0 / 2) / 20e9
    link = LINKS["ib-ndr"]
    t, stage, all_st = k.stream(b, w, link)
    assert all_st["link"] == b / 2 / link.bandwidth
    assert stage == max(all_st, key=all_st.get) == "module compute"
    assert t == link.latency + k.latency + all_st["module compute"]


def test_no_optics_time_without_a_transform():
    st = LOW.stage_times(1e9, KernelWork("dot", (("mul", 1.0), ("add", 1.0))))
    assert st["module optics"] == 0.0


def test_energy_static_laser_dynamic():
    e = LOW.energy(1e9, 0.5)
    assert e == {"static": 9.7 * 0.5, "laser": 2.0 * 0.5, "dynamic": 1e9 * 8 * 2.76e-12}


def test_kv_work_freq_band_is_the_training_band():
    c = KV_PRESETS["freq-keep-k"]
    w = kv_work(c, 3000, planes=8)
    n = pow2_at_least(3000)
    assert n == 4096
    assert w.transform_per_byte == 2.5 * 12 / 2
    assert w.ratio == c.ratio and w.fpga_ops_per_value == 2.0 + 8 * 0.5
    assert w.other_per_byte == 0.0
    assert kv_work(KV_PRESETS["none"], 100).needs() == frozenset()


# ── refusals ───────────────────────────────────────────────────────────────────────────────────────────
def test_topk_is_refused_for_lack_of_comparisons():
    topk = next(w for w in CIT_LEDGER if "top-k" in w.name)
    with pytest.raises(ValueError, match=r"compare.*no comparisons"):
        HIGH.check(topk)


@pytest.mark.parametrize("preset", ["fp8", "fp4-block"])
def test_requantising_kv_presets_are_refused(preset):
    with pytest.raises(ValueError, match="does not have"):
        KVTransit(KV_PRESETS[preset], kernel=LOW)


def test_kernel_accepts_the_fixed_band_and_none():
    KVTransit(KV_PRESETS["freq-keep-k"], kernel=LOW)
    KVTransit(KV_PRESETS["none"], kernel=LOW)


def test_summing_streams_needs_a_switch():
    s = CIT_LEDGER[0]
    with pytest.raises(ValueError, match="switch"):
        LOW.check(s)
    LINK_KERNELS["switch-10"].check(s)


def test_kernel_only_in_transit():
    with pytest.raises(ValueError, match="endpoint"):
        KVTransit(KV_PRESETS["freq-keep-k"], where="endpoint", kernel=LOW)


def test_unknown_operation_and_placement():
    with pytest.raises(ValueError, match="unknown operation"):
        KernelWork("x", (("sort", 1.0),))
    with pytest.raises(ValueError, match="placement"):
        LinkKernel(placement="nic")


def test_cli_refuses_with_a_message():
    r = subprocess.run([sys.executable, "-m", "disagg_sim", "--model", "llama3-8b", "--devices-per-instance", "1",
                        "--kv-compress", "fp8", "--link-kernel", "pluggable-10", "--requests", "5"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "does not have" in r.stderr


# ── default off: identical to 69ffd2b ───────────────────────────────────────────────────────────────────
DEFAULT_OFF_CASES = [(lk, pr, where) for lk in ("eth-25g", "ib-ndr")
                     for pr, where in (("none", None), ("fp8", "transit"), ("fp4-block", "transit"),
                                       ("freq-keep-k", "transit"), ("freq-keep-k", "endpoint"),
                                       ("fp8", "endpoint"))]


def default_off_run(lk, pr, where) -> dict:
    tr = None if where is None else KVTransit(KV_PRESETS[pr], where=where)
    cfg = SimConfig(model=MODELS["llama3-8b"], devices_per_instance=1, link=LINKS[lk], kv_transit=tr)
    wl = poisson_workload(8.0, 120, LengthDist(2048, 0.5), LengthDist(128, 0.5), seed=1)
    return json.loads(json.dumps(summarise(simulate(cfg, wl)), sort_keys=True))


def test_kernel_defaults_to_none():
    assert KVTransit(KV_PRESETS["fp8"]).kernel is None


@pytest.mark.parametrize("case", DEFAULT_OFF_CASES, ids=lambda c: "|".join(map(str, c)))
def test_default_off_identical_to_69ffd2b(case):
    gold = json.loads((FIX / "link_kernel_default_off.json").read_text())
    assert default_off_run(*case) == gold["|".join(map(str, case))]


# ── the slowest stage binds ─────────────────────────────────────────────────────────────────────────────
BAND = kv_work(KV_PRESETS["freq-keep-k"], 2048)


@pytest.mark.parametrize("kernel,link,stage", [
    (LINK_KERNELS["pluggable-99-free"], LINKS["eth-25g"], "link"),
    (LINK_KERNELS["pluggable-99-free"], LINKS["ib-ndr"], "module optics"),
    (LINK_KERNELS["pluggable-10"], LINKS["ib-ndr"], "module compute"),
    (replace(LINK_KERNELS["pluggable-99-free"], fpga_ops_per_s=5e9), LINKS["ib-ndr"], "FPGA"),
    (replace(LINK_KERNELS["pluggable-99-free"], port_rate=5 * GB), LINKS["ib-ndr"], "module port"),
])
def test_binding_stage(kernel, link, stage):
    _, got, st = kernel.stream(1e8, BAND, link)
    assert got == stage
    assert st[stage] == max(st.values())


def sim_with(kernel, link, n=40):
    tr = KVTransit(KV_PRESETS["freq-keep-k"], kernel=kernel)
    cfg = SimConfig(model=MODELS["llama3-8b"], devices_per_instance=1, link=link, kv_transit=tr)
    wl = poisson_workload(4.0, n, LengthDist(2048, 0.5), LengthDist(64, 0.5), seed=2)
    return simulate(cfg, wl), wl


def test_simulator_counts_the_binding_stage():
    res, _ = sim_with(LINK_KERNELS["pluggable-10"], LINKS["ib-ndr"])
    lk = res.link
    assert lk.kernel_stages == {"module compute": lk.transfers}
    assert lk.transit_bound == lk.transfers
    m = summarise(res)["kv_link"]["transit"]
    assert m["kernel"]["binding_stages"] == {"module compute": lk.transfers}
    assert m["transit_bound_frac"] == 1.0
    res, _ = sim_with(LINK_KERNELS["pluggable-99-free"], LINKS["eth-25g"])
    assert res.link.kernel_stages == {"link": res.link.transfers} and res.link.transit_bound == 0


def test_simulator_transfer_time_and_energy_follow_the_kernel():
    k, link = LINK_KERNELS["pluggable-10"], LINKS["ib-ndr"]
    res, wl = sim_with(k, link)
    m = MODELS["llama3-8b"]
    busy = joules = 0.0
    for r in wl:
        if r.output_len <= 1:
            continue
        nb = m.handoff_bytes(r.prompt_len)
        t, stage, st = k.stream(nb, kv_work(KV_PRESETS["freq-keep-k"], r.prompt_len), link)
        assert math.isclose(r.kv_ready - r.kv_start, t, rel_tol=1e-9)
        busy += t
        joules += sum(k.energy(nb, st[stage]).values())
    assert math.isclose(res.link.busy, busy, rel_tol=1e-12)
    assert math.isclose(res.link.transit_j, joules, rel_tol=1e-12)


def test_kernel_and_legacy_stage_differ_only_when_selected():
    a, _ = sim_with(None, LINKS["ib-ndr"])
    b, _ = sim_with(LINK_KERNELS["pluggable-99-free"], LINKS["ib-ndr"])
    assert a.link.kernel_stages == {} and b.link.kernel_stages
    assert "kernel" not in summarise(a)["kv_link"]["transit"]


if __name__ == "__main__":
    out = {"|".join(map(str, c)): default_off_run(*c) for c in DEFAULT_OFF_CASES}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True))
