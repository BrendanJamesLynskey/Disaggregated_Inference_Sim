"""The GB200 preset (brief 34, 2026-10-14): one Blackwell GPU of a GB200 superchip in an NVL72 rack.

Figures from NVIDIA's GB200 NVL72 page (superchip column, halved per GPU; sparse-only figures halved for dense) and the
Blackwell "up to 1,200 W" TGP (NVIDIA's Blackwell Ultra blog, Table 2). These tests pin the figures, check that the
JavaScript port's device table equals the Python presets field for field (every preset, not only GB200), and that
adding the preset changed no other device. Run parity for GB200 configurations is in test_levers_2.JS_CASES.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from disagg_sim.hardware import ACCELERATORS, B200, GB200, LLAMA3_70B, CostModel


def test_gb200_figures():
    assert GB200.name == "GB200"
    assert GB200.mem_capacity == 186e9 and GB200.mem_bw == 8e12          # 372 GB and 16 TB/s per superchip
    assert GB200.peak_flops == 2.5e15                                     # dense BF16: 10 PF sparse per superchip / 2 / 2
    assert GB200.peak_flops * GB200.format_speedup("fp8") == 5e15         # dense FP8, 10 PF per superchip
    assert GB200.peak_flops * GB200.format_speedup("fp4") == 10e15        # dense FP4, 20 PF per superchip
    assert GB200.tdp_w == 1200.0 and GB200.scale_up == "nvlink5"
    # everything else (efficiencies, power coefficients) is B200's
    for f in ("flops_eff", "bw_eff", "idle_w", "pj_per_flop", "pj_per_byte", "native_formats", "transform", "fft_efficiency"):
        assert getattr(GB200, f) == getattr(B200, f), f
    assert ACCELERATORS["gb200"] is GB200


def test_gb200_holds_more_kv_than_b200_and_prefills_faster():
    g, b = CostModel(LLAMA3_70B, GB200, 4), CostModel(LLAMA3_70B, B200, 4)
    assert g.kv_capacity_tokens > b.kv_capacity_tokens
    assert g.prefill([2048]).time < b.prefill([2048]).time
    # decode at a small batch is bandwidth-bound: the same HBM bandwidth gives the same step time
    assert g.decode([2048] * 4).time == b.decode([2048] * 4).time


def test_javascript_device_table_matches_python():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    engine = Path(__file__).parent.parent / "web" / "sim_engine.js"
    script = (f"require({json.dumps(str(engine))});const D=globalThis.DisaggSim.DEVICES;"
              "console.log(JSON.stringify(Object.fromEntries(Object.entries(D).map(([k,d])=>[k,"
              "[d.name,d.F,d.B,d.M,d.fe,d.be,d.tdp,d.idle,d.pjF,d.pjB,d.native,d.scaleUp,!!d.transform]]))))")
    js = json.loads(subprocess.run([node, "-e", script], check=True, capture_output=True, text=True).stdout)
    assert set(js) == set(ACCELERATORS)
    for k, d in ACCELERATORS.items():
        name, F, B, M, fe, be, tdp, idle, pjF, pjB, native, scale_up, transform = js[k]
        if k not in ("optical",):        # the JS display name of the hypothetical part differs (no hyphen)
            assert name == d.name, k
        assert [F, B, M, fe, be, tdp, idle, pjF, pjB] == [d.peak_flops, d.mem_bw, d.mem_capacity, d.flops_eff, d.bw_eff,
                                                          d.tdp_w, d.idle_w, d.pj_per_flop, d.pj_per_byte], k
        assert native == dict(d.native_formats) and scale_up == d.scale_up and transform == (d.transform is not None), k
