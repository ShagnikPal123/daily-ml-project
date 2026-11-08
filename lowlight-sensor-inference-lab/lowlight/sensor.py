"""Layer 1: physical image-sensor model (photon shot noise -> read noise -> ADC).

Signal chain for one pixel with scene radiance r in [0, 1]::

    lambda = g_ij * N * r + D            (mean electrons: PRNU gain g, peak signal N, dark D)
    e      = Poisson(lambda) + N(0, s_r^2)   (shot noise, then read noise in electrons)
    code   = ADC_b(e + B)                (black-level pedestal B = 3 s_r, uniform b-bit quantizer)

This follows the Poisson-Gaussian raw-image noise model used in the EMVA 1288
camera standard and in Foi et al. (IEEE TIP 2008). The reconstruction subtracts the
calibrated pedestal and dark level and divides by the flat-field gain, returning a
radiance estimate whose expectation equals r (up to clipping at very low light).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .fixedpoint import quant_step, uniform_dequantize, uniform_quantize


@dataclass(frozen=True)
class SensorConfig:
    photons: float = 100.0        # peak signal electrons at radiance 1 (exposure x QE)
    read_noise: float = 2.0       # electrons rms
    dark_electrons: float = 1.0   # mean dark-current electrons per exposure
    adc_bits: int = 10
    prnu: float = 0.01            # photo-response non-uniformity (fractional sigma)
    headroom: float = 1.2         # ADC full scale / peak signal (auto-exposure margin)

    def __post_init__(self):
        if self.photons <= 0 or self.read_noise < 0 or self.dark_electrons < 0:
            raise ValueError("photons must be > 0; read_noise and dark must be >= 0")
        if not 1 <= self.adc_bits <= 16:
            raise ValueError("adc_bits must be in [1, 16]")
        if self.headroom < 1.0:
            raise ValueError("headroom must be >= 1")

    def replace(self, **kw) -> "SensorConfig":
        return dataclasses.replace(self, **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "SensorConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        bad = set(d) - names
        if bad:
            raise ValueError(f"unknown sensor keys: {sorted(bad)}")
        return cls(**d)


class Sensor:
    """A concrete sensor instance (fixed-pattern gain map is drawn once)."""

    def __init__(self, cfg: SensorConfig, shape: tuple[int, int], seed: int = 0):
        self.cfg = cfg
        self.shape = tuple(shape)
        rng = np.random.default_rng(seed)
        self.gain_map = 1.0 + cfg.prnu * rng.standard_normal(self.shape)
        self.black_e = 3.0 * cfg.read_noise
        self.full_scale_e = self.black_e + cfg.dark_electrons + cfg.headroom * cfg.photons * float(self.gain_map.max())

    # ---- forward model ----
    def expose(self, radiance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Analog electrons (float) after shot + read noise, before the ADC."""
        c = self.cfg
        lam = self.gain_map * c.photons * np.asarray(radiance, dtype=float) + c.dark_electrons
        return rng.poisson(lam).astype(float) + c.read_noise * rng.standard_normal(lam.shape)

    def digitize(self, electrons: np.ndarray) -> np.ndarray:
        return uniform_quantize(electrons + self.black_e, self.cfg.adc_bits, 0.0, self.full_scale_e)

    def capture(self, radiance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Integer ADC codes for a scene (H, W) or batch (N, H, W)."""
        return self.digitize(self.expose(radiance, rng))

    # ---- calibrated reconstruction ----
    def codes_to_electrons(self, codes: np.ndarray) -> np.ndarray:
        e = uniform_dequantize(codes, self.cfg.adc_bits, 0.0, self.full_scale_e)
        return e - self.black_e - self.cfg.dark_electrons

    def estimate(self, codes: np.ndarray, flat_field: bool = True) -> np.ndarray:
        """Radiance estimate from ADC codes (can fall slightly outside [0, 1])."""
        g = self.gain_map if flat_field else 1.0
        return self.codes_to_electrons(codes) / (self.cfg.photons * g)

    def acquire(self, radiance: np.ndarray, rng: np.random.Generator, flat_field: bool = True) -> np.ndarray:
        return self.estimate(self.capture(radiance, rng), flat_field)

    # ---- theory ----
    @property
    def lsb_electrons(self) -> float:
        return quant_step(self.cfg.adc_bits, 0.0, self.full_scale_e)

    def noise_var_electrons(self, signal_e) -> np.ndarray:
        """Variance (e-^2) of the reconstructed electron count at mean signal ``signal_e``."""
        c = self.cfg
        return np.asarray(signal_e, float) + c.dark_electrons + c.read_noise**2 + self.lsb_electrons**2 / 12.0

    def noise_var_radiance(self, radiance) -> np.ndarray:
        return self.noise_var_electrons(self.cfg.photons * np.asarray(radiance, float)) / self.cfg.photons**2

    def snr(self, radiance) -> np.ndarray:
        """Theoretical per-pixel SNR (linear) at a given radiance."""
        s = self.cfg.photons * np.asarray(radiance, float)
        return s / np.sqrt(self.noise_var_electrons(s))

    def dynamic_range_db(self) -> float:
        """20 log10(saturation / noise floor) with the sensor's own noise floor."""
        floor = np.sqrt(self.noise_var_electrons(0.0))
        return float(20 * np.log10(self.cfg.headroom * self.cfg.photons / floor))
