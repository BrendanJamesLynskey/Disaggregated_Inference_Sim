"""Heterogeneous pools, FFT-mixing models, the optical transform engine and KV hand-off compression.

Added 2026-10-04 (FOptInf phase B). The rule these tests enforce first: none of it changes an
existing run. Then that each new model does what the FOptInf phase-A analysis says it should.
"""

import io
import json
import math
import re
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import pytest

from disagg_sim import (ACCELERATORS, LINKS, MODELS, CostModel, LengthDist, SimConfig,
                        poisson_workload, simulate, summarise)
from disagg_sim.hardware import (ENOB_REQUIRED, KV_PRESETS, OPTICAL_FFT, KVTransit, ModelSpec,
                                 TransformEngine)
from disagg_sim.ppa import device_area, dies_per_wafer, murphy_yield, ppa_report

H100, A100 = ACCELERATORS["h100"], ACCELERATORS["a100"]
L8B = MODELS["llama3-8b"]
HYENA, CIRC = MODELS["llama3-8b-hyena"], MODELS["llama3-8b-hyena-circ"]
FIXTURES = Path(__file__).parent / "fixtures"


def run(cfg, rate=6.0, n=200, seed=3, output=128):
    wl = poisson_workload(rate, n, LengthDist(2048, 0.5), LengthDist(output, 0.5), seed=seed)
    return wl, simulate(cfg, wl)


def stamps(wl):
    return [(r.prefill_start, r.first_token, r.kv_start, r.kv_ready, r.decode_start, r.finish, tuple(r.itls))
            for r in wl]


# ── 1. heterogeneous pools ───────────────────────────────────────────────
@pytest.mark.parametrize("fast", [False, True])
def test_identical_pools_reproduce_the_homogeneous_run_bit_exactly(fast):
    base = SimConfig(fast_forward=fast)
    same = replace(base, prefill_device=base.device, decode_device=base.device,
                   prefill_devices_per_instance=base.devices_per_instance,
                   decode_devices_per_instance=base.devices_per_instance)
    a, ra = run(base)
    b, rb = run(same)
    assert stamps(a) == stamps(b)
    ma, mb = summarise(ra), summarise(rb)
    assert mb.pop("pools") == {"prefill": "4x H100-SXM", "decode": "4x H100-SXM"}
    assert ma == mb


def test_h100_prefill_a100_decode_uses_each_device_per_pool():
    cfg = SimConfig(model=L8B, devices_per_instance=1, prefill_device=H100, decode_device=A100)
    _, res = run(cfg)
    pre, dec = res.instances
    assert (pre.cost.device, dec.cost.device) == (H100, A100)
    m = summarise(res)
    assert m["pools"] == {"prefill": "1x H100-SXM", "decode": "1x A100-SXM"}
    # decode is memory-bound, so A100's lower bandwidth sets TPOT: slower than all-H100
    _, h = run(replace(cfg, decode_device=None))
    assert m["latency_s"]["tpot"]["p50"] > summarise(h)["latency_s"]["tpot"]["p50"]
    # identical prefill pool: TTFT of the first request is the same
    assert res.requests[0].first_token == h.requests[0].first_token


def test_per_pool_devices_per_instance():
    cfg = SimConfig(model=L8B, devices_per_instance=1, decode_devices_per_instance=2)
    _, res = run(cfg, n=60)
    assert [i.cost.n_devices for i in res.instances] == [1, 2]


def test_a_pool_the_model_does_not_fit_is_named():
    cfg = SimConfig(devices_per_instance=4, decode_devices_per_instance=1)   # 70B on one H100
    with pytest.raises(ValueError, match="decode pool"):
        run(cfg, n=5)


# ── 2. FFT-mixing models: the op ledger ──────────────────────────────────
LEDGER = json.loads((FIXTURES / "flop_share_ledger.json").read_text())
VARIANT = {"transformer": "llama3-8b", "hyena": "llama3-8b-hyena", "hybrid": "llama3-8b-hybrid",
           "hyena_circ": "llama3-8b-hyena-circ"}


@pytest.mark.parametrize("row", LEDGER["prefill"], ids=lambda r: f"{r['variant']}-{r['prompt']}-{r['lm_head']}")
def test_prefill_ops_equal_the_phase_a_analysis(row):
    m = replace(MODELS[VARIANT[row["variant"]]], prefill_lm_head=row["lm_head"])
    o = m.prefill_ops([row["prompt"]])
    assert [o.dense, o.attention, o.transform, o.spectral, o.other, o.total] == \
        [row[k] for k in ("dense", "attention", "transform", "spectral", "other", "total")]


@pytest.mark.parametrize("row", LEDGER["decode"], ids=lambda r: f"{r['variant']}-{r['ctx']}")
def test_decode_ops_equal_the_phase_a_analysis(row):
    m = MODELS["llama3-8b-hyena" if row["variant"] == "hyena_direct" else "llama3-8b-hyena-dist"]
    assert m.decode_ops(row["ctx"], 1).total == row["total"]


@pytest.mark.parametrize("model", ["llama3-8b", "llama3-70b"])
def test_transformers_are_untouched(model):
    m = MODELS[model]
    cm = CostModel(m, H100, 4 if model == "llama3-70b" else 1)
    for lens in ([512], [2048], [300, 1700, 8192]):
        assert m.prefill_ops(lens).total == cm.prefill(lens).flops
    assert (m.is_transformer, m.conversion_pairs_per_token, m.mask_values([2048])) == (True, 0, 0)
    assert m.cache_units(100, 20) == 120 and m.handoff_bytes(100) == 100 * m.kv_bytes_per_token


def test_handoff_bytes_match_the_analysis():
    # FOptInf A§9: GQA KV 268.4 MB, Hyena direct cache 1,073.7 MB, distilled state 16.8 MB (2,048 tokens)
    assert L8B.handoff_bytes(2048) == 268_435_456
    assert HYENA.handoff_bytes(2048) == 4 * L8B.handoff_bytes(2048)
    dist = MODELS["llama3-8b-hyena-dist"]
    assert dist.handoff_bytes(2048) == dist.handoff_bytes(32768) == 16_777_216
    assert dist.cache_units(2048, 256) == 1 and dist.cache_unit_bytes == 16_777_216
    # A§7/A§8: 262,144 conversion pairs per token; 537,133,056 mask values at 2,048 tokens
    assert HYENA.conversion_pairs_per_token == 262_144
    assert HYENA.mask_values([2048]) == 537_133_056
    assert HYENA.mask_values([2048, 2000]) == HYENA.mask_values([2048])    # same padded length
    assert HYENA.mask_values([2048, 512]) == 537_133_056 + HYENA.mask_values([512])


def test_invalid_variants_are_rejected():
    with pytest.raises(ValueError):
        replace(L8B, mixer="fnet")                       # not causal: no decoder can use it
    with pytest.raises(ValueError):
        replace(L8B, circulant_block=256)               # circulant modelled for Hyena only
    with pytest.raises(ValueError):
        replace(MODELS["llama3-8b-hybrid"], decode_style="distilled")


@pytest.mark.parametrize("model", ["llama3-8b-hyena", "llama3-8b-hybrid", "llama3-8b-hyena-dist",
                                   "llama3-8b-hyena-circ"])
def test_fft_mixing_models_run_and_fast_forward_stays_exact(model):
    cfg = SimConfig(model=MODELS[model], devices_per_instance=1)
    a, ra = run(cfg, n=150)
    b, _ = run(replace(cfg, fast_forward=True), n=150)
    assert all(r.tokens_out == r.output_len for r in a)
    assert stamps(a) == stamps(b)
    assert all(i.kv_used == 0 for i in ra.instances)


# ── 3. the transform engine ──────────────────────────────────────────────
def test_transform_share_zero_equals_the_digital_fallback():
    """A transformer on optical-fft: same steps, same dynamic energy; only the engine's static
    power is added, (laser + tuning) x devices x horizon per instance."""
    base = SimConfig(model=L8B, devices_per_instance=1)
    a, ra = run(base)
    b, rb = run(replace(base, device=OPTICAL_FFT))
    assert stamps(a) == stamps(b)
    for x, y in zip(ra.instances, rb.instances):
        assert (x.compute_j, x.memory_j, x.busy) == (y.compute_j, y.memory_j, y.busy)
        assert y.optical_j == 0.0
    mb = summarise(rb)
    H = rb.horizon
    for name, v in mb["optical"].items():
        assert v["static_J"] == 20.0 * H and v["conversion_J"] == 0.0
    ea, eb = summarise(ra)["energy"]["total_J"], mb["energy"]["total_J"]
    assert eb - ea == pytest.approx(2 * 20.0 * H, rel=1e-12)


@pytest.mark.parametrize("fmt", sorted(ENOB_REQUIRED))
@pytest.mark.parametrize("enob", range(4, 17))
def test_passes_follow_the_averaging_rule(fmt, enob):
    want = 4 ** max(0, ENOB_REQUIRED[fmt] - enob)
    assert TransformEngine(enob=enob).passes(fmt) == want
    assert TransformEngine(enob=enob, detection="intensity").passes(fmt) == 2 * want


def test_mask_rewrites_round_up_exactly():
    vals = HYENA.mask_values([2048])
    for cap, want in [(vals, 1), (vals - 1, 2), (vals // 2, 2), (vals // 2 - 1, 3)]:
        dev = replace(OPTICAL_FFT, transform=replace(OPTICAL_FFT.transform, mask_values=cap))
        assert CostModel(HYENA, dev).optical_terms([2048], 2048)[2] == want
    # 05A's default: 269 rewrites of a 2 MP mask at 2,048 tokens
    assert CostModel(HYENA, OPTICAL_FFT).optical_terms([2048], 2048)[2] == 269


def test_engine_time_and_energy_follow_the_model():
    cm = CostModel(HYENA, OPTICAL_FFT)
    eng = OPTICAL_FFT.transform
    k, conv, rew, t_opt = cm.optical_terms([2048], 2048)
    assert k == 64 and conv == 262_144 * 2048 * 64                # BF16 at ENOB 8: 4^3 passes
    assert t_opt == conv / 1e12 + rew / 1031.0
    s = cm.prefill([2048])
    dig = CostModel(HYENA, H100)
    t_dig, ec, em, _ = dig.step_time_raw(HYENA.prefill_ops([2048]).digital, s.bytes)
    assert s.time == t_dig + t_opt + cm.step_overhead
    assert s.optical_j == conv * eng.pj_per_pair * 1e-12 and s.bound == "optical"
    over = replace(OPTICAL_FFT, transform=replace(eng, overlap=True))
    assert CostModel(HYENA, over).prefill([2048]).time == max(t_dig, t_opt) + cm.step_overhead
    # decode never touches the engine
    assert cm.decode([2048] * 4) == dig.decode([2048] * 4)


def test_static_power_is_charged_whether_or_not_work_arrives():
    _, res = run(SimConfig(model=HYENA, devices_per_instance=1, prefill_device=OPTICAL_FFT), rate=0.5, n=20)
    m = summarise(res)
    pre = m["optical"]["prefill-0"]
    assert pre["static_J"] == 20.0 * res.horizon and m["optical"]["decode-0"]["static_J"] == 0.0
    assert m["energy"]["breakdown"]["optical_static"] > m["energy"]["breakdown"]["optical_conversions"]


hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402


@settings(max_examples=25, deadline=None)
@given(w1=st.floats(0, 200), w2=st.floats(0, 200), seed=st.integers(0, 1000))
def test_energy_is_monotone_in_static_power(w1, w2, seed):
    lo, hi = sorted((w1, w2))
    out = []
    for w in (lo, hi):
        dev = replace(OPTICAL_FFT, transform=replace(OPTICAL_FFT.transform, laser_w=w / 2, tuning_w=w / 2))
        _, res = run(SimConfig(model=CIRC, devices_per_instance=1, prefill_device=dev), rate=1.0, n=20,
                     seed=seed, output=16)
        out.append(summarise(res)["energy"]["total_J"])
    assert out[0] <= out[1]


# ── 4. the KV hand-off: in transit vs at the endpoint ───────────────────
def test_passthrough_stage_only_adds_its_latency():
    base = SimConfig(model=L8B, devices_per_instance=1, link=LINKS["eth-25g"])
    _, a = run(base)
    tr = KVTransit(KV_PRESETS["none"], latency=1e-6)
    _, b = run(replace(base, kv_transit=tr))
    for x, y in zip(a.requests, b.requests):
        assert y.kv_ready - y.kv_start == pytest.approx(x.kv_ready - x.kv_start + 1e-6, abs=1e-12)
    assert a.link.bytes == b.link.bytes == b.link.handoff_bytes


def test_compression_shrinks_the_link_bytes_and_flags_the_budget():
    base = SimConfig(model=L8B, devices_per_instance=1, link=LINKS["eth-25g"])
    for preset, bound in [("fp8", False), ("fp4-block", True), ("freq-keep-k", False)]:
        _, res = run(replace(base, kv_transit=KVTransit(KV_PRESETS[preset])), n=80)
        lk = res.link
        assert lk.bytes == pytest.approx(lk.handoff_bytes / KV_PRESETS[preset].ratio, rel=1e-12)
        assert (lk.transit_bound == lk.transfers) == bound and (lk.transit_bound == 0) == (not bound)
    # without a passive transform the FreqKV-style preset no longer fits the budget
    _, res = run(replace(base, kv_transit=KVTransit(KV_PRESETS["freq-keep-k"], native_fft=False)), n=40)
    assert res.link.transit_bound == res.link.transfers


def test_endpoint_compression_costs_gpu_time_and_no_link_stage():
    base = SimConfig(model=L8B, devices_per_instance=1, link=LINKS["eth-25g"])
    _, t = run(replace(base, kv_transit=KVTransit(KV_PRESETS["fp8"], where="endpoint")), n=80)
    _, a = run(base, n=80)
    assert t.link.transit_j == 0.0 and t.link.bytes == pytest.approx(a.link.bytes / 2, rel=1e-12)
    assert t.instances[0].busy > a.instances[0].busy                   # the extra pass on the GPU
    assert t.instances[0].memory_j > a.instances[0].memory_j


def test_kv_transit_needs_disaggregation():
    with pytest.raises(ValueError, match="disagg"):
        run(SimConfig(mode="colocated", kv_transit=KVTransit(KV_PRESETS["fp8"])), n=5)


# ── 5. PPA ───────────────────────────────────────────────────────────────
def test_ppa_functions_match_fhe_sim_or_pinned_values():
    try:
        from fhe_sim import ppa as fhe
    except ImportError:
        fhe = None
    for a in (100.0, 814.0, 826.0, 917.0):
        if fhe is not None:
            assert murphy_yield(a, 0.1) == fhe.murphy_yield(a, 0.1)
            assert dies_per_wafer(a) == fhe.dies_per_wafer(a)
    assert repr(murphy_yield(814, 0.1)) == "0.4680943579969482"
    assert (dies_per_wafer(814), dies_per_wafer(100)) == (63, 640)
    assert device_area(H100)["usd"] == 339.09863688464617
    opt = device_area(OPTICAL_FFT)        # 20 converter channels x 0.15 mm² + a 100 mm² photonic die
    assert (opt["die_mm2"], opt["photonic_mm2"]) == (817.0, 100.0)
    assert device_area(ACCELERATORS["optical"]) is None


def test_ppa_report_totals():
    cfg = SimConfig(model=L8B, devices_per_instance=1, prefill_device=H100, decode_device=A100)
    _, res = run(cfg, n=60)
    m = summarise(res)
    p = ppa_report(cfg, m)
    assert p["total_mm2"] == 814.0 + 826.0
    assert p["perf_per_mm2"] == m["throughput"]["output_tok_per_s"] / 1640.0
    assert p["perf_per_W"] == m["energy"]["output_tokens_per_J"]


# ── 6. CLI ───────────────────────────────────────────────────────────────
def cli(*args):
    from disagg_sim.cli import main
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(list(args))
    return buf.getvalue()


def test_cli_default_output_is_unchanged():
    """results.md section 7 records this command's output before any of the new code existed."""
    md = (Path(__file__).parent.parent / "examples" / "results.md").read_text()
    want = re.search(r"## 7\..*?```\n(.*?)```", md, re.S).group(1)
    assert cli("--prefill", "2", "--rate", "6", "--link", "eth-25g") == want


def test_cli_heterogeneous_and_optical():
    out = cli("--model", "llama3-8b-hyena", "--devices-per-instance", "1", "--prefill-device", "optical-fft",
              "--decode-device", "h100", "--requests", "40", "--ft-enob", "11", "--ppa")
    assert "prefill 1x Optical-FFT + H100-class   decode 1x H100-SXM" in out
    assert "optical      prefill-0 optical-bound" in out and "ppa          prefill" in out
    out = cli("--model", "llama3-8b", "--devices-per-instance", "1", "--link", "eth-25g", "--requests", "40",
              "--kv-compress", "fp8")
    assert "kv hand-off  fp8 at transit (ratio 2.00)" in out


# ── 7. the JavaScript port (decks, the inference website) ────────────────
JS_CASES = {
    "H100 prefill + A100 decode": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="h100", decodeDevice="a100"),
    "A100 prefill + 2x H100 decode": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100",
                                          decodeDevicesPerInstance=2, nPrefill=2),
    "70B, explicit identical pools": dict(prefillDevice="h100", decodeDevice="h100", prefillDevicesPerInstance=4),
    "Hyena on GPUs": dict(model="llama3-8b-hyena", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100"),
    "Hyena, optical prefill": dict(model="llama3-8b-hyena", devicesPerInstance=1, prefillDevice="optical-fft"),
    "hybrid, optical prefill": dict(model="llama3-8b-hybrid", devicesPerInstance=1, prefillDevice="optical-fft"),
    "distilled, optical prefill": dict(model="llama3-8b-hyena-dist", devicesPerInstance=1, prefillDevice="optical-fft"),
    "circulant, optical prefill": dict(model="llama3-8b-hyena-circ", devicesPerInstance=1, prefillDevice="optical-fft",
                                       engine=dict(maskRate=20000.0)),
    "circulant last-token, ENOB 6": dict(model="llama3-8b-hyena-circ", lmHead="last", devicesPerInstance=1,
                                         prefillDevice="optical-fft", engine=dict(enob=6, maskRate=1e6)),
    "circulant last-token, ENOB 12, overlap": dict(model="llama3-8b-hyena-circ", lmHead="last", devicesPerInstance=1,
                                                   prefillDevice="optical-fft",
                                                   engine=dict(enob=12, maskRate=20000.0, overlap=True)),
    "circulant, intensity detection, small part": dict(model="llama3-8b-hyena-circ", devicesPerInstance=1,
                                                       prefillDevice="optical-fft-small",
                                                       engine=dict(detection="intensity", maskRate=1e5, staticW=50.0)),
    "circulant, GPU FFT at 1/16": dict(model="llama3-8b-hyena-circ", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100", fftEff=1 / 16),
    "colocated, optical part": dict(model="llama3-8b-hyena", devicesPerInstance=1, device="optical-fft", mode="colocated",
                                    nColocated=2, engine=dict(enob=11, maskRate=20000.0)),
    "optical prefill, power cap + DVFS": dict(model="llama3-8b-hyena", devicesPerInstance=1, prefillDevice="optical-fft",
                                              powerCap=300.0, dvfs=True),
    "fp8 in transit, 25 GbE": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100", link="eth-25g", kvCompress="fp8"),
    "fp4-block in transit (transit-bound)": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100", link="eth-25g",
                                                 kvCompress="fp4-block"),
    "freq-keep-k 1/4, no passive FFT": dict(model="llama3-8b-hyena", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100", link="eth-100g",
                                            kvCompress="freq-keep-k", kvKeep=0.25, transitNativeFft=False),
    "fp4-block at the GPU": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100", link="eth-25g", kvCompress="fp4-block",
                                 kvCompressAt="endpoint"),
    "freq-keep-k at the GPU": dict(model="llama3-8b-hyena", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100", link="eth-100g",
                                   kvCompress="freq-keep-k", kvCompressAt="endpoint"),
    "none in transit, co-packaged optics": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100", link="cpo-optical",
                                                kvCompress="none", transitOpsPerByte=1.0, transitPjPerBit=2.0),
    "distilled state, fp8 in transit": dict(model="llama3-8b-hyena-dist", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100", link="eth-25g",
                                            kvCompress="fp8"),
}


def py_config(c: dict) -> SimConfig:
    """The Python SimConfig a JS config means (the JS port's keys are camelCase)."""
    def dev(key):
        if key is None:
            return None
        d = ACCELERATORS[key]
        e = c.get("engine", {})
        if d.transform is not None and e:
            names = {"enob": "enob", "maskRate": "mask_rate_hz", "detection": "detection", "overlap": "overlap"}
            over = {names[k]: v for k, v in e.items() if k in names}
            if "staticW" in e:
                over["laser_w"] = over["tuning_w"] = e["staticW"] / 2
            d = replace(d, transform=replace(d.transform, **over))
        if "fftEff" in c:
            d = replace(d, fft_efficiency=c["fftEff"])
        return d
    model = replace(MODELS[c.get("model", "llama3-70b")], prefill_lm_head=c.get("lmHead", "all"))
    tr = None
    if "kvCompress" in c:
        comp = KV_PRESETS[c["kvCompress"]]
        if "kvKeep" in c:
            comp = replace(comp, ratio=1 / c["kvKeep"])
        kw = {k2: c[k1] for k1, k2 in (("transitOpsPerByte", "ops_per_byte"), ("transitPjPerBit", "pj_per_bit"),
                                        ("transitNativeFft", "native_fft")) if k1 in c}
        tr = KVTransit(comp, where=c.get("kvCompressAt", "transit"), **kw)
    return SimConfig(model=model, device=dev(c.get("device", "h100")), devices_per_instance=c.get("devicesPerInstance", 4),
                     mode=c.get("mode", "disagg"), n_prefill=c.get("nPrefill", 1), n_decode=c.get("nDecode", 1),
                     n_colocated=c.get("nColocated", 2), link=LINKS[c.get("link", "ib-ndr")],
                     power_cap_w=c.get("powerCap"), dvfs=c.get("dvfs", False),
                     prefill_device=dev(c.get("prefillDevice")), decode_device=dev(c.get("decodeDevice")),
                     prefill_devices_per_instance=c.get("prefillDevicesPerInstance"),
                     decode_devices_per_instance=c.get("decodeDevicesPerInstance"), kv_transit=tr)


def test_javascript_port_matches_python_for_the_new_features():
    """Every timestamp, every instance's energy accounts, the link's, the energy total and the PPA
    numbers, for each new feature. Bit-exact, except runs with power-bound steps (cube roots, 1e-9) and
    the PPA yield and dies-per-wafer maths (exp, pi, sqrt: compared to 1e-12)."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    engine = Path(__file__).parent.parent / "web" / "sim_engine.js"
    cases = []
    for name, c in JS_CASES.items():
        wl = poisson_workload(6.0, 150, LengthDist(2048, 0.6), LengthDist(96, 0.6), seed=11)
        rows = [[r.arrival, r.prompt_len, r.output_len] for r in wl]
        cfg = py_config(c)
        res = simulate(cfg, wl)
        m = summarise(res)
        cases.append({"name": name, "cfg": {"device": "h100", "devicesPerInstance": 4, "mode": "disagg", "nPrefill": 1,
                                            "nDecode": 1, "nColocated": 2, "link": "ib-ndr", "model": "llama3-70b", **c},
                      "rows": rows,
                      "stamps": [[r.prefill_start, r.first_token, r.kv_start, r.kv_ready, r.decode_start, r.finish] for r in wl],
                      "inst": [[i.compute_j, i.memory_j, i.optical_j, i.busy, i.peak_power, i.optical_bound_time]
                               for i in res.instances],
                      "link": [res.link.energy, res.link.transit_j, res.link.bytes, res.link.transit_bound],
                      "total": m["energy"]["total_J"], "ppa": ppa_report(cfg, m)["total_usd"],
                      "capped": any(i.power_bound_time > 0 for i in res.instances)})
    script = (f"require({json.dumps(str(engine))});"
              "const S=globalThis.DisaggSim, cases=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify(cases.map(c=>{const r=S.simulate(c.cfg,c.rows), m=S.summarise(r);"
              "return {stamps:r.reqs.map(q=>[q.prefillStart,q.firstToken,q.kvStart,q.kvReady,q.decodeStart,q.finish]),"
              "inst:r.insts.map(i=>[i.ec,i.em,i.oj,i.busy,i.peakW,i.opticalBound]),"
              "link:[r.link.energy,r.link.transitJ,r.link.bytes,r.link.transitBound],"
              "total:m.energy.totalJ, ppa:S.ppa(c.cfg,m).totalUsd};})));")
    out = subprocess.run([node, "-e", script], input=json.dumps(cases), capture_output=True, text=True, check=True)
    # the new code paths are checked bit-exactly: most cases have no power-bound step
    assert sum(not c["capped"] for c in cases) >= 18
    for c, js in zip(cases, json.loads(out.stdout)):
        tol = 1e-9 if c["capped"] else 0.0     # power-bound steps take cube roots: an ulp across libms
        for key in ("stamps", "inst", "link"):
            rows_py = c[key] if key != "link" else [c[key]]
            rows_js = js[key] if key != "link" else [js[key]]
            for a, b in zip(rows_py, rows_js):
                for x, y in zip(a, b):
                    assert (x is None and y is None) or abs(x - y) <= tol * max(1.0, abs(x)), (c["name"], key, a, b)
        assert abs(c["total"] - js["total"]) <= tol * c["total"], c["name"]
        if c["ppa"] is None:
            assert js["ppa"] is None
        else:
            assert js["ppa"] == pytest.approx(c["ppa"], rel=1e-12), c["name"]
