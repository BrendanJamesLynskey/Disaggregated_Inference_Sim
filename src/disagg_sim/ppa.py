"""Area and silicon cost per pool, and perf/W, perf/mm² and perf/$ for a run.

The method and coefficients are FHE_Accelerator_Sim's (``fhe_sim/ppa.py``, brief 03): the four
functions below are copied from it unchanged so the two simulators cannot disagree, and
``tests/test_optical.py`` checks them against ``fhe_sim.ppa`` when that package is importable and
against pinned values otherwise. Everything here is **illustrative**:

* GPU dies: NVIDIA's published die sizes (GH100 814 mm² on TSMC 4N; GA100 826 mm² on N7; the
  Hopper and Ampere architecture posts on developer.nvidia.com). Silicon only: no HBM, interposer,
  packaging or test, so the cost is far below a product's price.
* The transform engine: one converter channel per 50 GS/s, each a DAC (0.05 mm²) and an ADC
  (0.10 mm²) on the digital die, plus a separate 100 mm² photonic die. **Speculative** round
  numbers, the same as FHE_Accelerator_Sim's optical engine.
* Yield and cost: Murphy's model, D0 = 0.1 defects/cm², $10,000 per 300 mm wafer (illustrative;
  FHE_Accelerator_Sim uses one wafer price for every node, and so does this file).

Area never feeds back into a simulated time or energy.
"""

from __future__ import annotations

import math

# Published die areas, mm² (Hopper / Ampere architecture in-depth posts, NVIDIA developer blog).
# The optical-FFT parts' digital dies are the same chips.
DIE_MM2 = {"H100-SXM": 814.0, "A100-SXM": 826.0,
           "Optical-FFT + H100-class": 814.0, "Optical-FFT + A100-class": 826.0}

# FHE_Accelerator_Sim AreaModel's optical-engine coefficients (speculative).
CONVERTER_GSPS = 50.0
MM2_PER_DAC = 0.05
MM2_PER_ADC = 0.10
PHOTONIC_DIE_MM2 = 100.0

# FHE_Accelerator_Sim DieCost (illustrative).
D0_PER_CM2 = 0.1
WAFER_MM = 300.0
WAFER_USD = 10000.0
RETICLE_MM2 = 858.0


# ── copied unchanged from fhe_sim/ppa.py ──────────────────────────────
def poisson_yield(area_mm2: float, d0_per_cm2: float) -> float:
    """Y = exp(-A D0): defects independent and uniform. Pessimistic for large dies."""
    return math.exp(-(area_mm2 / 100.0) * d0_per_cm2)


def murphy_yield(area_mm2: float, d0_per_cm2: float) -> float:
    """Y = ((1 - exp(-A D0)) / (A D0))^2: defect density varying across the wafer (triangular)."""
    x = (area_mm2 / 100.0) * d0_per_cm2
    if x == 0:
        return 1.0
    t = -math.expm1(-x) / x       # not 1 - exp(-x): that cancels to 0 for tiny dies (Hypothesis found it)
    return t * t


def dies_per_wafer(area_mm2: float, wafer_mm: float = 300.0) -> int:
    """Gross dies: wafer area / die area, minus the partial dies lost around the edge."""
    r = wafer_mm / 2.0
    n = math.pi * r * r / area_mm2 - math.pi * wafer_mm / math.sqrt(2.0 * area_mm2)
    return max(0, math.floor(n))


def die_cost(area: float) -> dict:
    dpw = dies_per_wafer(area, WAFER_MM)
    y = murphy_yield(area, D0_PER_CM2)
    good = dpw * y
    return {"area_mm2": area, "dies_per_wafer": dpw, "yield": y, "good_dies": good,
            "usd_per_good_die": WAFER_USD / good if good > 0 else math.inf,
            "fits_reticle": area <= RETICLE_MM2}


# ── this simulator's devices ──────────────────────────────────────────
def device_area(dev) -> dict | None:
    """mm² of one device: the digital die (+ converters) and any photonic die. None if unknown
    (the hypothetical optical MAC part has no published or modelled area)."""
    if dev.name not in DIE_MM2:
        return None
    die, photonic = DIE_MM2[dev.name], 0.0
    if dev.transform is not None:
        ch = math.ceil(dev.transform.samples_per_s / (CONVERTER_GSPS * 1e9))
        die = die + ch * (MM2_PER_DAC + MM2_PER_ADC)
        photonic = PHOTONIC_DIE_MM2
    usd = die_cost(die)["usd_per_good_die"]
    if photonic:
        usd = usd + die_cost(photonic)["usd_per_good_die"]
    return {"die_mm2": die, "photonic_mm2": photonic, "total_mm2": die + photonic, "usd": usd}


def ppa_report(cfg, m: dict) -> dict:
    """Per pool: area and silicon $ per instance; for the cluster: perf/W (output tokens per J),
    perf/mm² and perf/$ (output tokens/s per mm² and per $). ``m`` is ``summarise()`` output."""
    roles = {"prefill": cfg.n_prefill, "decode": cfg.n_decode} if cfg.mode == "disagg" else {"colocated": cfg.n_colocated}
    pools, mm2, usd, known = {}, 0.0, 0.0, True
    for role, count in roles.items():
        dev, n = cfg.pool(role) if role != "colocated" else (cfg.device, cfg.devices_per_instance)
        a = device_area(dev)
        if a is None:
            known = False
            pools[role] = {"device": dev.name, "instances": count, "devices_per_instance": n,
                           "mm2_per_instance": None, "usd_per_instance": None}
            continue
        pools[role] = {"device": dev.name, "instances": count, "devices_per_instance": n,
                       "mm2_per_instance": n * a["total_mm2"], "usd_per_instance": n * a["usd"]}
        mm2 = mm2 + count * n * a["total_mm2"]
        usd = usd + count * n * a["usd"]
    perf = m["throughput"]["output_tok_per_s"]
    return {"pools": pools, "total_mm2": mm2 if known else None, "total_usd": usd if known else None,
            "perf_per_W": m["energy"]["output_tokens_per_J"],
            "perf_per_mm2": perf / mm2 if known else None,
            "perf_per_kusd": 1e3 * perf / usd if known else None}
