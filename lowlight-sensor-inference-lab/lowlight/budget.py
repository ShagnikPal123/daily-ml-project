"""Layer 3: deployment budget - MACs, SRAM line buffers, weight bytes, energy, latency.

Energy constants are order-of-magnitude figures for 45 nm CMOS in the spirit of
Horowitz (ISSCC 2014, "Computing's energy problem"): they are *assumptions* exposed as
a dict you can override, not measurements made by this project. ADC energy is reported
only as a relative scaling (Walden-style, doubling per extra bit).
"""
from __future__ import annotations

import time

import numpy as np

from .classifiers import _Base
from .denoisers import Denoiser, make_denoiser
from .sensor import Sensor, SensorConfig

ENERGY_PJ = {"int8_mac": 0.23, "fp32_mac": 4.6}


def adc_relative_energy(bits: int, ref_bits: int = 8) -> float:
    """Relative ADC conversion energy, E ~ 2**bits (Walden figure-of-merit scaling)."""
    return float(2.0 ** (bits - ref_bits))


def denoiser_cost(d: Denoiser, shape, pixel_bytes: int = 2) -> dict:
    h, w = shape
    macs = d.macs_per_pixel() * h * w
    return {"macs": float(macs), "line_buffer_bytes": int(d.line_buffer_rows() * w * pixel_bytes)}


def classifier_cost(clf: _Base, weight_bits: int | None = None) -> dict:
    wb = clf.weight_bytes() if weight_bits is None else clf.weight_bytes(weight_bits)
    return {"macs": float(clf.macs()), "weight_bytes": int(wb)}


def frame_cost(denoiser: Denoiser, clf: _Base, shape, denoiser_dtype: str = "int8",
               clf_dtype: str = "int8", energy=None) -> dict:
    """Per-frame totals: compute, SRAM (line buffers + weights) and energy estimate (uJ)."""
    e = ENERGY_PJ if energy is None else energy
    d, c = denoiser_cost(denoiser, shape), classifier_cost(clf, 8 if clf_dtype == "int8" else 32)
    pj = d["macs"] * e[f"{denoiser_dtype}_mac"] + c["macs"] * e[f"{clf_dtype}_mac"]
    return {"denoiser_macs": d["macs"], "classifier_macs": c["macs"], "total_macs": d["macs"] + c["macs"],
            "sram_bytes": d["line_buffer_bytes"] + c["weight_bytes"], "weight_bytes": c["weight_bytes"],
            "line_buffer_bytes": d["line_buffer_bytes"], "energy_uj": pj * 1e-6}


def time_it(fn, repeats: int = 5, warmup: int = 1) -> float:
    """Median wall-clock seconds of ``fn()``."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def profile_denoisers(names, sensor_cfg: SensorConfig, scenes, repeats: int = 3, seed: int = 0) -> list[dict]:
    """Host-CPU latency per image for each denoiser (relative cost only)."""
    sensor = Sensor(sensor_cfg, scenes.shape[1:], seed=seed)
    noisy = sensor.acquire(scenes, np.random.default_rng(seed))
    rows = []
    for n in names:
        d = make_denoiser(n, sensor).fit(scenes, noisy)
        t = time_it(lambda: d(noisy), repeats)
        rows.append({"denoiser": n, "us_per_image": 1e6 * t / len(scenes),
                     **denoiser_cost(d, scenes.shape[1:])})
    return rows
