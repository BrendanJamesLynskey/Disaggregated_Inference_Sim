"""Heterogeneous pools, KV hand-off compression on the prefill GPU, and the area model.

Added 2026-10-04 (these features were split out of a larger test file on 2026-10-29). The rule these tests
enforce first: none of it changes an existing run.
"""

import io
import json
import re
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path

import pytest

from disagg_sim import ACCELERATORS, LINKS, MODELS, LengthDist, SimConfig, poisson_workload, simulate, summarise
from disagg_sim.hardware import KV_PRESETS
from disagg_sim.ppa import device_area, dies_per_wafer, murphy_yield, ppa_report

H100, A100 = ACCELERATORS["h100"], ACCELERATORS["a100"]
L8B = MODELS["llama3-8b"]


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


# ── 2. KV hand-off compression on the prefill GPU ───────────────────────
def test_none_preset_changes_nothing():
    base = SimConfig(model=L8B, devices_per_instance=1, link=LINKS["eth-25g"])
    a, ra = run(base)
    b, rb = run(replace(base, kv_compression=KV_PRESETS["none"]))
    assert stamps(a) == stamps(b)
    assert ra.link.bytes == rb.link.bytes == rb.link.handoff_bytes


def test_compression_costs_gpu_time_and_shrinks_the_link_bytes():
    base = SimConfig(model=L8B, devices_per_instance=1, link=LINKS["eth-25g"])
    _, a = run(base, n=80)
    for preset in ("fp8", "fp4-block"):
        _, t = run(replace(base, kv_compression=KV_PRESETS[preset]), n=80)
        assert t.link.bytes == pytest.approx(t.link.handoff_bytes / KV_PRESETS[preset].ratio, rel=1e-12)
        assert t.link.handoff_bytes == pytest.approx(a.link.bytes, rel=1e-12)
        assert t.instances[0].busy > a.instances[0].busy                   # the extra pass on the GPU
        assert t.instances[0].memory_j > a.instances[0].memory_j


def test_kv_compression_needs_disaggregation():
    with pytest.raises(ValueError, match="disagg"):
        run(SimConfig(mode="colocated", kv_compression=KV_PRESETS["fp8"]), n=5)


# ── 3. PPA ───────────────────────────────────────────────────────────────
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
    assert device_area(ACCELERATORS["b200"]) is None


def test_ppa_report_totals():
    cfg = SimConfig(model=L8B, devices_per_instance=1, prefill_device=H100, decode_device=A100)
    _, res = run(cfg, n=60)
    m = summarise(res)
    p = ppa_report(cfg, m)
    assert p["total_mm2"] == 814.0 + 826.0
    assert p["perf_per_mm2"] == m["throughput"]["output_tok_per_s"] / 1640.0
    assert p["perf_per_W"] == m["energy"]["output_tokens_per_J"]


# ── 4. CLI ───────────────────────────────────────────────────────────────
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


def test_cli_heterogeneous_and_compression():
    out = cli("--model", "llama3-8b", "--devices-per-instance", "1", "--prefill-device", "a100",
              "--decode-device", "h100", "--requests", "40", "--ppa")
    assert "prefill 1x A100-SXM   decode 1x H100-SXM" in out and "ppa          prefill" in out
    out = cli("--model", "llama3-8b", "--devices-per-instance", "1", "--link", "eth-25g", "--requests", "40",
              "--kv-compress", "fp8")
    assert "kv hand-off  fp8 at the prefill GPU (ratio 2.00)" in out


# ── 5. the JavaScript port (decks, the inference website) ────────────────
JS_CASES = {
    "H100 prefill + A100 decode": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="h100", decodeDevice="a100"),
    "A100 prefill + 2x H100 decode": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100",
                                          decodeDevicesPerInstance=2, nPrefill=2),
    "70B, explicit identical pools": dict(prefillDevice="h100", decodeDevice="h100", prefillDevicesPerInstance=4),
    "A100 pools, power cap + DVFS": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100",
                                         decodeDevice="a100", powerCap=300.0, dvfs=True),
    "fp8 at the GPU, 25 GbE": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100",
                                   link="eth-25g", kvCompress="fp8"),
    "fp4-block at the GPU": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100", decodeDevice="a100",
                                 link="eth-25g", kvCompress="fp4-block"),
    "none, co-packaged optics switch": dict(model="llama3-8b", devicesPerInstance=1, prefillDevice="a100",
                                            decodeDevice="a100", link="cpo-optical", kvCompress="none"),
    "colocated A100s": dict(model="llama3-8b", devicesPerInstance=1, device="a100", mode="colocated", nColocated=2),
}


def py_config(c: dict) -> SimConfig:
    """The Python SimConfig a JS config means (the JS port's keys are camelCase)."""
    def dev(key):
        return None if key is None else ACCELERATORS[key]
    model = replace(MODELS[c.get("model", "llama3-70b")], prefill_lm_head=c.get("lmHead", "all"))
    return SimConfig(model=model, device=dev(c.get("device", "h100")), devices_per_instance=c.get("devicesPerInstance", 4),
                     mode=c.get("mode", "disagg"), n_prefill=c.get("nPrefill", 1), n_decode=c.get("nDecode", 1),
                     n_colocated=c.get("nColocated", 2), link=LINKS[c.get("link", "ib-ndr")],
                     power_cap_w=c.get("powerCap"), dvfs=c.get("dvfs", False),
                     prefill_device=dev(c.get("prefillDevice")), decode_device=dev(c.get("decodeDevice")),
                     prefill_devices_per_instance=c.get("prefillDevicesPerInstance"),
                     decode_devices_per_instance=c.get("decodeDevicesPerInstance"),
                     kv_compression=KV_PRESETS[c["kvCompress"]] if "kvCompress" in c else None)


def test_javascript_port_matches_python_for_the_new_features():
    """Every timestamp, every instance's energy accounts, the link's, the energy total and the PPA
    numbers, for each feature. Bit-exact, except runs with power-bound steps (cube roots, 1e-9) and
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
                      "inst": [[i.compute_j, i.memory_j, i.busy, i.peak_power] for i in res.instances],
                      "link": [res.link.energy, res.link.bytes, res.link.handoff_bytes],
                      "total": m["energy"]["total_J"], "ppa": ppa_report(cfg, m)["total_usd"],
                      "capped": any(i.power_bound_time > 0 for i in res.instances)})
    script = (f"require({json.dumps(str(engine))});"
              "const S=globalThis.DisaggSim, cases=JSON.parse(require('fs').readFileSync(0,'utf8'));"
              "console.log(JSON.stringify(cases.map(c=>{const r=S.simulate(c.cfg,c.rows), m=S.summarise(r);"
              "return {stamps:r.reqs.map(q=>[q.prefillStart,q.firstToken,q.kvStart,q.kvReady,q.decodeStart,q.finish]),"
              "inst:r.insts.map(i=>[i.ec,i.em,i.busy,i.peakW]),"
              "link:[r.link.energy,r.link.bytes,r.link.handoff],"
              "total:m.energy.totalJ, ppa:S.ppa(c.cfg,m).totalUsd};})));")
    out = subprocess.run([node, "-e", script], input=json.dumps(cases), capture_output=True, text=True, check=True)
    assert sum(not c["capped"] for c in cases) >= 5      # A100 pools hit the 400 W board limit
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
