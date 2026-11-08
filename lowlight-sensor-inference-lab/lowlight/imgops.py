"""Layer 0: batched image-processing kernels and quality metrics (NumPy only).

All filters accept a single image (H, W) or a batch (N, H, W). Boundary modes use
SciPy naming: 'reflect' (edge pixel repeated), 'constant' (zeros), 'edge'.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_PAD = {"reflect": "symmetric", "constant": "constant", "edge": "edge", "mirror": "reflect"}


def as_batch(x) -> tuple[np.ndarray, bool]:
    """Return (N, H, W) float view and whether the input was a single image."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 2:
        return a[None], True
    if a.ndim != 3:
        raise ValueError("expected (H, W) or (N, H, W)")
    return a, False


def pad_images(x: np.ndarray, ph: int, pw: int, mode: str = "reflect") -> np.ndarray:
    if mode not in _PAD:
        raise ValueError(f"unknown boundary mode {mode!r}")
    return np.pad(x, ((0, 0), (ph, ph), (pw, pw)), mode=_PAD[mode])


def windows(x: np.ndarray, kh: int, kw: int, mode: str = "reflect") -> np.ndarray:
    """Sliding windows, shape (N, H, W, kh, kw), same spatial size as input."""
    xp = pad_images(x, kh // 2, kw // 2, mode)
    return sliding_window_view(xp, (kh, kw), axis=(1, 2))


def correlate2d(x, kernel, mode: str = "reflect") -> np.ndarray:
    """2-D correlation (cross-correlation, like scipy.ndimage.correlate)."""
    b, single = as_batch(x)
    k = np.asarray(kernel, dtype=float)
    out = np.einsum("nhwij,ij->nhw", windows(b, *k.shape, mode=mode), k)
    return out[0] if single else out


def box_kernel(size: int) -> np.ndarray:
    return np.full((size, size), 1.0 / size**2)


def gaussian_kernel1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0])
    r = int(truncate * sigma + 0.5)
    t = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def gaussian_kernel(sigma: float, truncate: float = 4.0) -> np.ndarray:
    k1 = gaussian_kernel1d(sigma, truncate)
    return np.outer(k1, k1)


def gaussian_filter(x, sigma: float, mode: str = "reflect", truncate: float = 4.0) -> np.ndarray:
    """Separable Gaussian smoothing implemented as two 1-D passes."""
    b, single = as_batch(x)
    k1 = gaussian_kernel1d(sigma, truncate)
    out = correlate2d(b, k1[:, None], mode)
    out = correlate2d(out, k1[None, :], mode)
    return out[0] if single else out


def box_filter(x, size: int = 3, mode: str = "reflect") -> np.ndarray:
    return correlate2d(x, box_kernel(size), mode)


def median_filter(x, size: int = 3, mode: str = "reflect") -> np.ndarray:
    b, single = as_batch(x)
    w = windows(b, size, size, mode)
    out = np.median(w.reshape(*w.shape[:3], -1), axis=-1)
    return out[0] if single else out


def local_mean_var(x, size: int = 3, mode: str = "reflect"):
    """Local mean and (population) variance over a size x size window."""
    b, single = as_batch(x)
    w = windows(b, size, size, mode).reshape(*b.shape, -1)
    mu = w.mean(-1)
    var = np.maximum((w**2).mean(-1) - mu**2, 0.0)
    return (mu[0], var[0]) if single else (mu, var)


def wiener_filter(x, size: int = 5, noise=None, mode: str = "reflect") -> np.ndarray:
    """Adaptive local Wiener filter (Lim, 1990). Same formula as scipy.signal.wiener.

    ``noise`` may be a scalar variance, an array broadcastable to ``x`` (signal-
    dependent noise), or None (use the mean local variance).
    """
    b, single = as_batch(x)
    mu, lvar = local_mean_var(b, size, mode)
    nz = np.mean(lvar, axis=(1, 2), keepdims=True) if noise is None else np.asarray(noise, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (b - mu) * (1.0 - nz / lvar) + mu
    out = np.where(lvar < nz, mu, res)
    return out[0] if single else out


def extract_patches(x, size: int, mode: str = "reflect") -> np.ndarray:
    """Return (N, H, W, size*size) patch matrix for learned linear filters."""
    b, _ = as_batch(x)
    w = windows(b, size, size, mode)
    return w.reshape(*b.shape, size * size)


def mse(ref, test) -> np.ndarray:
    r, _ = as_batch(ref)
    t, _ = as_batch(test)
    return ((r - t) ** 2).mean(axis=(1, 2))


def psnr(ref, test, data_range: float = 1.0) -> np.ndarray:
    """Per-image PSNR in dB (capped at 100 dB for identical images)."""
    m = np.maximum(mse(ref, test), 1e-10)
    return 10.0 * np.log10(data_range**2 / m)


def ssim(ref, test, data_range: float = 1.0, win_size: int = 7) -> np.ndarray:
    """Per-image mean SSIM, uniform window, valid region only (skimage defaults)."""
    r, _ = as_batch(ref)
    t, _ = as_batch(test)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    npx = win_size**2
    cov_norm = npx / (npx - 1.0)

    def wmean(a):
        return sliding_window_view(a, (win_size, win_size), axis=(1, 2)).mean(axis=(-2, -1))

    ux, uy = wmean(r), wmean(t)
    vx = cov_norm * (wmean(r * r) - ux * ux)
    vy = cov_norm * (wmean(t * t) - uy * uy)
    vxy = cov_norm * (wmean(r * t) - ux * uy)
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux**2 + uy**2 + c1) * (vx + vy + c2))
    return s.mean(axis=(1, 2))
