"""Layer 2: pluggable denoisers acting on calibrated radiance estimates.

Interface (see :class:`Denoiser`): ``fit(clean, noisy)`` (no-op for fixed filters),
``__call__(estimates) -> denoised``, plus hardware descriptors ``macs_per_pixel()``
and ``line_buffer_rows()`` used by the budget module.

Seven methods ship in the registry :data:`DENOISERS`:
identity, box, gaussian, median, wiener, gat (generalized Anscombe + Gaussian +
inverse) and ridge (learned shift-invariant linear patch filter).
"""
from __future__ import annotations

import numpy as np

from . import imgops as io
from .sensor import Sensor

SIGMA_GRID = (0.4, 0.6, 0.8, 1.0, 1.3, 1.7)
TRUNC = 3.0  # Gaussian support in sigmas (hardware-friendly)


class Denoiser:
    name = "base"

    def __init__(self, sensor: Sensor):
        self.sensor = sensor

    def fit(self, clean: np.ndarray, noisy: np.ndarray) -> "Denoiser":
        return self

    def __call__(self, est: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def macs_per_pixel(self) -> float:
        return 0.0

    def line_buffer_rows(self) -> int:
        return 0

    def describe(self) -> dict:
        return {"name": self.name}


def _best_sigma(fn, clean, noisy, grid, max_images: int = 600) -> float:
    c, n = clean[:max_images], noisy[:max_images]
    errs = [float(np.mean((np.clip(fn(n, s), 0, 1) - c) ** 2)) for s in grid]
    return float(grid[int(np.argmin(errs))])


class Identity(Denoiser):
    name = "identity"

    def __call__(self, est):
        return np.asarray(est, float)


class BoxFilter(Denoiser):
    name = "box"

    def __init__(self, sensor, size: int = 3):
        super().__init__(sensor)
        self.size = size

    def __call__(self, est):
        return io.box_filter(est, self.size)

    def macs_per_pixel(self):
        return 2.0 * self.size  # separable running sums

    def line_buffer_rows(self):
        return self.size - 1

    def describe(self):
        return {"name": self.name, "size": self.size}


class GaussianFilter(Denoiser):
    name = "gaussian"

    def __init__(self, sensor, sigma: float | None = None, grid=SIGMA_GRID):
        super().__init__(sensor)
        self.sigma, self.grid = sigma, tuple(grid)

    def fit(self, clean, noisy):
        if self.sigma is None:
            self.sigma = _best_sigma(lambda x, s: io.gaussian_filter(x, s, truncate=TRUNC), clean, noisy, self.grid)
        return self

    def _sigma(self):
        return 1.0 if self.sigma is None else self.sigma

    def __call__(self, est):
        return io.gaussian_filter(est, self._sigma(), truncate=TRUNC)

    def _taps(self):
        return 2 * int(TRUNC * self._sigma() + 0.5) + 1

    def macs_per_pixel(self):
        return 2.0 * self._taps()

    def line_buffer_rows(self):
        return self._taps() - 1

    def describe(self):
        return {"name": self.name, "sigma": self.sigma}


class MedianFilter(Denoiser):
    name = "median"

    def __init__(self, sensor, size: int = 3):
        super().__init__(sensor)
        self.size = size

    def __call__(self, est):
        return io.median_filter(est, self.size)

    def macs_per_pixel(self):  # comparator-equivalent ops of a sorting network
        k2 = self.size**2
        return float(k2 * int(np.ceil(np.log2(k2))))

    def line_buffer_rows(self):
        return self.size - 1

    def describe(self):
        return {"name": self.name, "size": self.size}


class WienerFilter(Denoiser):
    """Local Wiener filter with the sensor's *signal-dependent* noise variance."""

    name = "wiener"

    def __init__(self, sensor, size: int = 5, noise_scale: float = 1.0):
        super().__init__(sensor)
        self.size, self.noise_scale = size, noise_scale

    def __call__(self, est):
        b, single = io.as_batch(est)
        mu, _ = io.local_mean_var(b, self.size)
        nv = self.noise_scale * self.sensor.noise_var_radiance(np.clip(mu, 0.0, 1.0))
        out = io.wiener_filter(b, self.size, noise=nv)
        return out[0] if single else out

    def macs_per_pixel(self):
        return 2.0 * self.size**2 + 10

    def line_buffer_rows(self):
        return self.size - 1

    def describe(self):
        return {"name": self.name, "size": self.size}


class GATGaussian(Denoiser):
    """Generalized Anscombe transform -> Gaussian smoothing -> inverse transform.

    For z = Poisson(lambda) + N(0, s^2): A(z) = 2 sqrt(z + 3/8 + s^2) has ~unit variance
    (Anscombe 1948; Makitalo & Foi, IEEE TIP 2013), so a Gaussian filter is near-optimal
    in the transformed domain. ``inverse`` is 'algebraic' or 'asymptotic' (closed-form
    asymptotically unbiased inverse).
    """

    name = "gat"

    def __init__(self, sensor, sigma: float | None = None, inverse: str = "asymptotic", grid=SIGMA_GRID):
        super().__init__(sensor)
        if inverse not in ("algebraic", "asymptotic"):
            raise ValueError("inverse must be 'algebraic' or 'asymptotic'")
        self.sigma, self.inverse, self.grid = sigma, inverse, tuple(grid)

    @property
    def s2(self) -> float:
        return float(self.sensor.noise_var_electrons(0.0))

    def forward(self, e: np.ndarray) -> np.ndarray:
        return 2.0 * np.sqrt(np.maximum(e + 3.0 / 8.0 + self.s2, 0.0))

    def inverse_transform(self, d: np.ndarray) -> np.ndarray:
        alg = d**2 / 4.0 - 3.0 / 8.0 - self.s2
        if self.inverse == "algebraic":
            return alg
        dd = np.maximum(d, 1e-3)
        asy = (dd**2 / 4.0 + 0.25 * np.sqrt(1.5) / dd - 11.0 / 8.0 / dd**2
               + 5.0 / 8.0 * np.sqrt(1.5) / dd**3 - 1.0 / 8.0 - self.s2)
        return np.where(d >= 2.0 * np.sqrt(3.0 / 8.0 + self.s2), asy, alg)

    def _run(self, est, sigma):
        n = self.sensor.cfg.photons
        d = io.gaussian_filter(self.forward(np.asarray(est, float) * n), sigma, truncate=TRUNC)
        return self.inverse_transform(d) / n

    def fit(self, clean, noisy):
        if self.sigma is None:
            self.sigma = _best_sigma(self._run, clean, noisy, self.grid)
        return self

    def __call__(self, est):
        return self._run(est, 1.0 if self.sigma is None else self.sigma)

    def macs_per_pixel(self):
        taps = 2 * int(TRUNC * (self.sigma or 1.0) + 0.5) + 1
        return 2.0 * taps + 14

    def line_buffer_rows(self):
        return 2 * int(TRUNC * (self.sigma or 1.0) + 0.5)

    def describe(self):
        return {"name": self.name, "sigma": self.sigma, "inverse": self.inverse}


class RidgePatch(Denoiser):
    """Learned shift-invariant linear filter: centre pixel = w . patch + b (ridge regression).

    The normal equations are accumulated chunk-by-chunk, so memory is O(size^4).
    """

    name = "ridge"

    def __init__(self, sensor, size: int = 5, lam: float = 1e-3, chunk: int = 200):
        super().__init__(sensor)
        self.size, self.lam, self.chunk = size, lam, chunk
        self.w = None
        self.b = 0.0

    def fit(self, clean, noisy):
        k2 = self.size**2
        sxx, sx, sxy, sy, n = np.zeros((k2, k2)), np.zeros(k2), np.zeros(k2), 0.0, 0
        for s in range(0, len(noisy), self.chunk):
            P = io.extract_patches(noisy[s: s + self.chunk], self.size).reshape(-1, k2)
            y = np.asarray(clean[s: s + self.chunk], float).reshape(-1)
            sxx += P.T @ P
            sx += P.sum(0)
            sxy += P.T @ y
            sy += y.sum()
            n += len(y)
        xm, ym = sx / n, sy / n
        cov = sxx / n - np.outer(xm, xm)
        cxy = sxy / n - xm * ym
        self.w = np.linalg.solve(cov + self.lam * np.eye(k2), cxy)
        self.b = float(ym - xm @ self.w)
        return self

    def __call__(self, est):
        if self.w is None:
            raise RuntimeError("RidgePatch must be fit() before use")
        b, single = io.as_batch(est)
        out = io.extract_patches(b, self.size) @ self.w + self.b
        return out[0] if single else out

    def macs_per_pixel(self):
        return float(self.size**2)

    def line_buffer_rows(self):
        return self.size - 1

    def describe(self):
        return {"name": self.name, "size": self.size, "lam": self.lam}


DENOISERS = {c.name: c for c in (Identity, BoxFilter, GaussianFilter, MedianFilter, WienerFilter, GATGaussian, RidgePatch)}


def make_denoiser(name: str, sensor: Sensor, **kw) -> Denoiser:
    if name not in DENOISERS:
        raise ValueError(f"unknown denoiser {name!r}; choose from {sorted(DENOISERS)}")
    return DENOISERS[name](sensor, **kw)
