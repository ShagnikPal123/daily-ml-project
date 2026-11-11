"""Layer 0: control-performance metrics computed from sampled time series."""
from __future__ import annotations

import numpy as np

NAN = float("nan")


def iae(t: np.ndarray, e: np.ndarray) -> float:
    """Integral of absolute error (rectangle rule on a uniform grid)."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return 0.0
    dt = float(t[1] - t[0])
    return float(np.sum(np.abs(e)) * dt)


def control_effort(u: np.ndarray, dt: float) -> float:
    """Integral of u^2 (proportional to resistive loss at constant R)."""
    return float(np.sum(np.asarray(u, float) ** 2) * dt)


def rms_delta(u: np.ndarray) -> float:
    """RMS of sample-to-sample change: a proxy for actuator chatter."""
    u = np.asarray(u, float)
    if len(u) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(u) ** 2)))


def saturation_fraction(u: np.ndarray, umax: float, tol: float = 1e-9) -> float:
    return float(np.mean(np.abs(u) >= umax - tol))


def _first_crossing(tt: np.ndarray, frac: np.ndarray, level: float) -> float:
    hit = frac >= level
    if not hit.any():
        return NAN
    return float(tt[int(np.argmax(hit))])


def step_metrics(t, y, ref, t0, t1, band=0.05) -> dict:
    """Rise time (10-90 %), overshoot [%], settling time and steady-state error.

    Evaluated on the window [t0, t1); the pre-step level is the sample just before t0.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = (t >= t0) & (t < t1)
    tt, yy = t[m], y[m]
    idx0 = int(np.searchsorted(t, t0))
    y0 = float(y[max(idx0 - 1, 0)])
    amp = ref - y0
    if abs(amp) < 1e-12 or len(tt) == 0:
        return {"rise": NAN, "overshoot_pct": NAN, "settle": NAN, "sse": NAN}
    frac = (yy - y0) / amp
    rise = _first_crossing(tt, frac, 0.9) - _first_crossing(tt, frac, 0.1)
    overshoot = max(0.0, float(frac.max()) - 1.0) * 100.0
    outside = np.abs(frac - 1.0) > band
    if outside[-1]:
        settle = NAN
    elif outside.any():
        settle = float(tt[int(np.where(outside)[0][-1]) + 1] - t0)
    else:
        settle = 0.0
    n = max(1, len(yy) // 10)
    sse = float(np.mean(yy[-n:]) - ref)
    return {"rise": rise, "overshoot_pct": overshoot, "settle": settle, "sse": sse}


def disturbance_metrics(t, y, ref, t0, band_abs) -> dict:
    """Peak deviation and recovery time after a disturbance applied at t0."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = t >= t0
    tt, dev = t[m], np.abs(ref - y[m])
    if len(tt) == 0:
        return {"peak": NAN, "recover": NAN}
    outside = dev > band_abs
    if outside[-1]:
        rec = NAN
    elif outside.any():
        rec = float(tt[int(np.where(outside)[0][-1]) + 1] - t0)
    else:
        rec = 0.0
    return {"peak": float(dev.max()), "recover": rec}


def window_std(t, y, t0, t1) -> float:
    t = np.asarray(t, float)
    m = (t >= t0) & (t < t1)
    return float(np.std(np.asarray(y, float)[m])) if m.any() else NAN
