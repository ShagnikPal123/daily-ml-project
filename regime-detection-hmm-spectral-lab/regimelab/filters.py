"""Layer 1: causal filters and spectral estimators (all strictly causal: output[t] uses x[:t+1])."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, lfilter, lfilter_zi, sosfilt, sosfilt_zi


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """y[t] = alpha x[t] + (1-alpha) y[t-1], initialised at x[0] (single-pole IIR low-pass)."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha in (0,1]")
    x = np.asarray(x, float)
    zi = lfilter_zi([alpha], [1.0, -(1.0 - alpha)]) * x[0]
    return lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=zi)[0]


def sma(x: np.ndarray, w: int) -> np.ndarray:
    """Causal simple moving average; expanding mean while fewer than w samples exist."""
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    n = np.arange(1, len(x) + 1)
    lo = np.maximum(n - w, 0)
    return (c[n] - c[lo]) / (n - lo)


def butter_lowpass_causal(x: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    """Butterworth low-pass (cutoff as a fraction of Nyquist) run forward only, state primed to x[0]."""
    sos = butter(order, cutoff, btype="low", output="sos")
    x = np.asarray(x, float)
    zi = sosfilt_zi(sos) * x[0]
    return sosfilt(sos, x, zi=zi)[0]


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling standard deviation (ddof=1); expanding window (min 2 samples) at the start."""
    x = np.asarray(x, float)
    out = np.empty(len(x))
    head = min(w - 1, len(x))
    for t in range(head):
        out[t] = x[: t + 1].std(ddof=1) if t >= 1 else 0.0
    if len(x) >= w:
        out[w - 1:] = sliding_window_view(x, w).std(axis=1, ddof=1)
    if len(x) > 1:
        out[0] = out[1]
    return out


def ewma_vol(r: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics volatility: sigma2[t] = lam sigma2[t-1] + (1-lam) r[t]^2."""
    r2 = np.asarray(r, float) ** 2
    y = lfilter([1.0 - lam], [1.0, -lam], r2, zi=[lam * r2[0]])[0]
    return np.sqrt(np.maximum(y, 1e-18))


def rolling_band_fraction(x: np.ndarray, w: int, lo_frac: float = 0.125) -> np.ndarray:
    """Fraction of (Hann-windowed, de-meaned) window power below lo_frac * fs excluding DC.

    White noise gives approximately 2*lo_frac; positive autocorrelation (momentum) pushes it
    higher, negative autocorrelation (mean reversion) lower.
    """
    x = np.asarray(x, float)
    nbin = w // 2
    m = max(int(np.floor(lo_frac * w)), 1)
    white = m / nbin
    out = np.full(len(x), white)
    if len(x) < w:
        return out
    win = np.hanning(w)
    seg = sliding_window_view(x, w)
    seg = (seg - seg.mean(axis=1, keepdims=True)) * win
    p = np.abs(np.fft.rfft(seg, axis=1)) ** 2
    tot = p[:, 1:nbin + 1].sum(axis=1)
    lo = p[:, 1:m + 1].sum(axis=1)
    out[w - 1:] = np.where(tot > 0, lo / np.maximum(tot, 1e-300), white)
    return out


def welch_band_power(x: np.ndarray, fs: float = 1.0, nperseg: int = 128, band=(0.0, 0.1)) -> float:
    """Integrated Welch PSD over [band) - used to summarise a segment."""
    from scipy.signal import welch
    from scipy.integrate import trapezoid
    f, pxx = welch(np.asarray(x, float), fs=fs, nperseg=min(nperseg, len(x)))
    sel = (f >= band[0]) & (f < band[1])
    return float(trapezoid(pxx[sel], f[sel])) if sel.sum() > 1 else 0.0


def steady_state_gain(q: float, r: float) -> float:
    """Steady-state Kalman gain of the local-level model; P- solves P^2 - qP - qr = 0."""
    pp = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
    return float(pp / (pp + r))


@dataclass
class LocalLevelKalman:
    """x_t = x_{t-1} + w (var q), y_t = x_t + v (var r).  A one-state Kalman filter."""
    q: float
    r: float

    def filter(self, y: np.ndarray):
        y = np.asarray(y, float)
        n = len(y)
        xs, ps, ks = np.empty(n), np.empty(n), np.empty(n)
        x, p = y[0], self.r
        xs[0], ps[0], ks[0] = x, p, 1.0
        for t in range(1, n):
            pp = p + self.q
            k = pp / (pp + self.r)
            x = x + k * (y[t] - x)
            p = (1.0 - k) * pp
            xs[t], ps[t], ks[t] = x, p, k
        return xs, ps, ks
