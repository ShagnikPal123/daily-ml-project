"""Layer 0: armature-controlled DC motor plant and exact zero-order-hold discretisation.

Electrical side:   L di/dt = V - R i - Ke w
Mechanical side:   J dw/dt = Kt i - b w - tau_load

State x = [i, w], inputs u = [V, tau_load], measured output y = w.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotorParams:
    """Small brushed DC motor (values typical of a ~50 W hobby/robotics motor)."""

    R: float = 2.0  # armature resistance [ohm]
    L: float = 1.0e-3  # armature inductance [H]
    Kt: float = 0.05  # torque constant [N m / A]
    Ke: float = 0.05  # back-EMF constant [V s / rad]
    J: float = 1.0e-5  # rotor inertia [kg m^2]
    b: float = 2.0e-6  # viscous friction [N m s / rad]
    vmax: float = 12.0  # supply (H-bridge) limit [V]


def continuous_matrices(p: MotorParams):
    """Return (A, B, C) with B columns ordered [V, tau_load]."""
    A = np.array([[-p.R / p.L, -p.Ke / p.L], [p.Kt / p.J, -p.b / p.J]])
    B = np.array([[1.0 / p.L, 0.0], [0.0, -1.0 / p.J]])
    C = np.array([[0.0, 1.0]])
    return A, B, C


def expm(M: np.ndarray, taylor_terms: int = 20) -> np.ndarray:
    """Matrix exponential by scaling-and-squaring with a Taylor core."""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    norm = np.linalg.norm(M, 1)
    s = 0 if norm <= 0.5 else int(np.ceil(np.log2(norm / 0.5)))
    X = M / (2.0**s)
    term = np.eye(n)
    out = np.eye(n)
    for k in range(1, taylor_terms + 1):
        term = term @ X / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def zoh(A: np.ndarray, B: np.ndarray, Ts: float):
    """Exact ZOH discretisation via the block-matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A * Ts
    M[:n, n:] = B * Ts
    E = expm(M)
    return E[:n, :n], E[:n, n:]


def discretize(p: MotorParams, Ts: float):
    """Discrete plant (Ad, Bd, C); Bd columns are [V, tau_load]."""
    A, B, C = continuous_matrices(p)
    Ad, Bd = zoh(A, B, Ts)
    return Ad, Bd, C


def dc_gain_speed(p: MotorParams) -> float:
    """Steady-state rad/s per volt (no load)."""
    return p.Kt / (p.R * p.b + p.Kt * p.Ke)


def dc_gain_torque(p: MotorParams) -> float:
    """Steady-state rad/s per N m of load torque (negative)."""
    return -p.R / (p.R * p.b + p.Kt * p.Ke)


def mechanical_time_constant(p: MotorParams) -> float:
    """Dominant (slow) closed-armature time constant, J / (b + Kt Ke / R)."""
    return p.J / (p.b + p.Kt * p.Ke / p.R)


def encoder_velocity_step(cpr: float, Ts: float) -> float:
    """Speed resolution [rad/s] of a count-difference velocity estimate."""
    if not cpr:
        return 0.0
    return 2.0 * np.pi / (cpr * Ts)
