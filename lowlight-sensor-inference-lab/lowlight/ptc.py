"""Layer 2: photon-transfer-curve (PTC) sensor characterisation, EMVA-1288 style.

Given only flat-field captures (no knowledge of the simulator's internals) estimate the
system gain K [DN/e-] from the slope of temporal variance vs. mean, and the dark noise
floor [e- rms] from dark frames. Comparing the estimates to the configured truth is an
end-to-end check that the sensor model produces physically consistent statistics, and the
same procedure applies to a real camera's raw frames.
"""
from __future__ import annotations

import numpy as np

from . import imgops as io
from .sensor import Sensor, SensorConfig


def flat_stack(sensor: Sensor, radiance: float, n_frames: int, rng: np.random.Generator) -> np.ndarray:
    """ADC codes of ``n_frames`` captures of a uniform scene."""
    return sensor.capture(np.full((n_frames, *sensor.shape), radiance), rng)


def temporal_stats(stack: np.ndarray) -> tuple[float, float]:
    """Mean DN and mean per-pixel temporal variance (removes fixed-pattern noise)."""
    s = np.asarray(stack, dtype=float)
    return float(s.mean()), float(s.var(axis=0, ddof=1).mean())


def photon_transfer(cfg: SensorConfig, levels, n_frames: int = 64, shape=(16, 16), seed: int = 0) -> dict:
    """Mean/variance pairs at each radiance level plus a dark-frame measurement."""
    sensor = Sensor(cfg, shape, seed=seed)
    rng = np.random.default_rng(seed + 7)
    pts = [temporal_stats(flat_stack(sensor, float(r), n_frames, rng)) for r in levels]
    dark = temporal_stats(flat_stack(sensor, 0.0, n_frames, rng))
    full_code = 2**cfg.adc_bits - 1
    return {"levels": np.asarray(levels, float), "mean_dn": np.array([p[0] for p in pts]),
            "var_dn": np.array([p[1] for p in pts]), "dark_var_dn": dark[1], "dark_mean_dn": dark[0],
            "full_code": full_code}


def fit_ptc(ptc: dict, max_frac: float = 0.8) -> dict:
    """Linear fit var = K * mean + c on unsaturated points.

    Returns the gain K (DN per electron), its inverse (e- per DN), the dark noise floor in
    electrons (quantisation noise 1/12 DN^2 removed) and the fit's R^2.
    """
    m, v = ptc["mean_dn"], ptc["var_dn"]
    keep = m < max_frac * ptc["full_code"]
    if keep.sum() < 3:
        raise ValueError("need at least 3 unsaturated points")
    K, c = np.polyfit(m[keep], v[keep], 1)
    pred = K * m[keep] + c
    ss_res, ss_tot = np.sum((v[keep] - pred) ** 2), np.sum((v[keep] - v[keep].mean()) ** 2)
    floor = np.sqrt(max(ptc["dark_var_dn"] - 1.0 / 12.0, 0.0)) / K
    return {"gain_dn_per_e": float(K), "e_per_dn": float(1.0 / K), "dark_noise_e": float(floor),
            "intercept": float(c), "r2": float(1 - ss_res / ss_tot)}


def true_gain(cfg: SensorConfig, shape=(16, 16), seed: int = 0) -> float:
    """Gain implied by the configured ADC range (what PTC should recover)."""
    s = Sensor(cfg, shape, seed=seed)
    return (2**cfg.adc_bits - 1) / s.full_scale_e


def true_noise_floor_e(cfg: SensorConfig) -> float:
    """sqrt(dark shot + read noise variance) in electrons."""
    return float(np.sqrt(cfg.dark_electrons + cfg.read_noise**2))


def empirical_snr(cfg: SensorConfig, levels, n_frames: int = 64, shape=(16, 16), seed: int = 0) -> np.ndarray:
    """Measured SNR (mean / temporal std, both in electrons) at each radiance level."""
    sensor = Sensor(cfg, shape, seed=seed)
    rng = np.random.default_rng(seed + 11)
    out = []
    for r in levels:
        e = sensor.codes_to_electrons(flat_stack(sensor, float(r), n_frames, rng))
        out.append(e.mean() / e.std(axis=0, ddof=1).mean())
    return np.array(out)


def smoothness_check(cfg: SensorConfig, shape=(16, 16), seed: int = 0) -> float:
    """Fraction of noise variance removed by a 3x3 box filter on a flat frame (spatial-noise whiteness probe)."""
    s = Sensor(cfg, shape, seed=seed)
    f = s.acquire(np.full((64, *shape), 0.5), np.random.default_rng(seed))
    return float(1.0 - io.box_filter(f, 3).var() / f.var())
