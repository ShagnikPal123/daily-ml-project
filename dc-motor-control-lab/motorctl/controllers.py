"""Layer 1: discrete PID with anti-windup and LQI (LQR + integral action) gain design.

Controller interface used everywhere: `reset()` and `step(r, y, x_true=None) -> u` (volts).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motorctl.riccati import lqr_gain


@dataclass(frozen=True)
class PIDGains:
    kp: float
    ki: float
    kd: float = 0.0
    tau_d: float = 2e-3  # derivative low-pass time constant [s]
    umax: float = 12.0


class PID:
    """Positional PID, derivative-on-measurement, selectable anti-windup.

    antiwindup in {"none", "clamp", "backcalc"}.
    """

    name = "pi"

    def __init__(self, gains: PIDGains, Ts: float, antiwindup: str = "clamp"):
        if antiwindup not in ("none", "clamp", "backcalc"):
            raise ValueError(f"unknown anti-windup mode {antiwindup!r}")
        self.g, self.Ts, self.aw = gains, Ts, antiwindup
        self.kaw = gains.ki / max(gains.kp, 1e-12)  # back-calculation gain 1/Tt
        self.reset()

    def reset(self):
        self.i = 0.0
        self.d = 0.0
        self.y_prev = None

    def step(self, r, y, x_true=None):
        g = self.g
        e = r - y
        if self.y_prev is not None and g.kd != 0.0:
            raw = -(y - self.y_prev) / self.Ts
            a = g.tau_d / (g.tau_d + self.Ts)
            self.d = a * self.d + (1 - a) * raw
        self.y_prev = y
        v = g.kp * e + self.i + g.kd * self.d
        u = min(max(v, -g.umax), g.umax)
        if self.aw == "backcalc":
            self.i += self.Ts * (g.ki * e + self.kaw * (u - v))
        elif self.aw == "clamp":
            if u == v or e * v < 0:
                self.i += self.Ts * g.ki * e
        else:
            self.i += self.Ts * g.ki * e
        return u


@dataclass(frozen=True)
class LQIGains:
    Kx: np.ndarray  # (1, n) state-feedback gain
    Kz: float  # integral-state gain
    P: np.ndarray  # Riccati solution of the augmented problem
    Aaug: np.ndarray
    Baug: np.ndarray


def augment_integral(Ad, Bv, Cd, Ts):
    """Append z_{k+1} = z_k + Ts (r - C x_k) to the plant (reference handled separately)."""
    n = Ad.shape[0]
    Aa = np.zeros((n + 1, n + 1))
    Aa[:n, :n] = Ad
    Aa[n, :n] = -Ts * Cd[0]
    Aa[n, n] = 1.0
    Ba = np.vstack([Bv, np.zeros((1, 1))])
    return Aa, Ba


def design_lqi(Ad, Bv, Cd, Ts, q_state=(0.05, 1.0), q_int=3000.0, r=1.0) -> LQIGains:
    Aa, Ba = augment_integral(Ad, Bv, Cd, Ts)
    Q = np.diag(list(q_state) + [q_int])
    K, P = lqr_gain(Aa, Ba, Q, [[r]])
    n = Ad.shape[0]
    return LQIGains(Kx=K[:, :n], Kz=float(K[0, n]), P=P, Aaug=Aa, Baug=Ba)


def disturbance_feedforward_gain(Ad, Bv, Bdist, Cd, Kx) -> float:
    """N such that u_ff = -N * d_hat leaves the output unchanged in steady state.

    Steady state of x+ = Ad x + Bv(-Kx x + u_ff) + Bdist d:  x = M (Bv u_ff + Bdist d),
    M = (I - Ad + Bv Kx)^-1;  requiring C x = 0 gives N = (C M Bdist) / (C M Bv).
    """
    n = Ad.shape[0]
    M = np.linalg.inv(np.eye(n) - Ad + Bv @ Kx)
    return float((Cd @ M @ Bdist)[0, 0] / (Cd @ M @ Bv)[0, 0])
