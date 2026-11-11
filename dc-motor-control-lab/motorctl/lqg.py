"""Layer 2: LQG speed controller = steady-state Kalman filter + LQI state feedback + torque feed-forward.

A computation delay of D samples is handled exactly by appending the D pending commands to the
state (a shift register), so the LQR gain is delay-aware and the Kalman predictor is consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motorctl.controllers import LQIGains, design_lqi, disturbance_feedforward_gain
from motorctl.estimators import SteadyStateKF, augment_disturbance, steady_state_kalman_gain
from motorctl.plant import MotorParams, discretize


@dataclass(frozen=True)
class LQGDesign:
    lqi: LQIGains
    Nd: float
    L: np.ndarray
    Aaug: np.ndarray
    Bv_aug: np.ndarray
    Caug: np.ndarray
    Ts: float
    delay: int


def delay_extend(Ad, Bv, Bdist, Cd, D):
    """Append D pending-command states q=[u_{k-1},...,u_{k-D}]; plant sees u_{k-D}."""
    n = Ad.shape[0]
    if D == 0:
        return Ad, Bv, Bdist, Cd
    A = np.zeros((n + D, n + D))
    A[:n, :n] = Ad
    A[:n, n + D - 1] = Bv[:, 0]
    A[n, :] = 0.0  # q1 <- u (input)
    for j in range(1, D):
        A[n + j, n + j - 1] = 1.0
    B = np.zeros((n + D, 1))
    B[n, 0] = 1.0
    Bd_ = np.vstack([Bdist, np.zeros((D, 1))])
    C = np.hstack([Cd, np.zeros((1, D))])
    return A, B, Bd_, C


def design_lqg(p: MotorParams, Ts: float, *, meas_std: float = 0.5, enc_step: float = 0.0,
               q_state=(0.05, 1.0), q_int=3.0e5, r=1.0, delay: int = 0,
               proc_std=(0.05, 0.5, 2e-4), use_dff: bool = True) -> LQGDesign:
    Ad, Bd, Cd = discretize(p, Ts)
    Bv, Bdist = Bd[:, :1], Bd[:, 1:2]
    Ae, Be, Bde, Ce = delay_extend(Ad, Bv, Bdist, Cd, delay)
    lqi = design_lqi(Ae, Be, Ce, Ts, q_state=tuple(q_state) + (0.0,) * delay, q_int=q_int, r=r)
    Nd = disturbance_feedforward_gain(Ae, Be, Bde, Ce, lqi.Kx) if use_dff else 0.0
    Aaug, Bv_aug, Caug = augment_disturbance(Ad, Bd)
    Q = np.diag(np.square(proc_std))
    R = np.array([[meas_std**2 + enc_step**2 / 12.0]])
    L, _ = steady_state_kalman_gain(Aaug, Caug, Q, R)
    return LQGDesign(lqi, Nd, L, Aaug, Bv_aug, Caug, Ts, delay)


class LQGController:
    """oracle=True feeds the true plant state (no estimator): the 'ideal LQI' upper bound."""

    def __init__(self, design: LQGDesign, umax: float, oracle: bool = False):
        self.d, self.umax, self.oracle = design, umax, oracle
        self.name = "lqi_oracle" if oracle else "lqg"
        self.reset()

    def reset(self):
        self.kf = SteadyStateKF(self.d.Aaug, self.d.Bv_aug, self.d.Caug, self.d.L)
        self.z = 0.0
        self.queue = [0.0] * self.d.delay  # oldest first

    def step(self, r, y, x_true=None):
        d = self.d
        if self.oracle:
            x, dhat = np.asarray(x_true, float), 0.0
        else:
            xa = self.kf.update(y)
            x, dhat = xa[:2], xa[2]
        xs = np.concatenate([x, np.array(self.queue[::-1], float)])  # newest pending first
        v = -float((d.lqi.Kx @ xs)[0]) - d.lqi.Kz * self.z - d.Nd * dhat
        u = min(max(v, -self.umax), self.umax)
        e = r - float(x[1])
        if not (u != v and e * v > 0):  # clamp anti-windup
            self.z += d.Ts * e
        self.queue.append(u)
        u_applied = self.queue.pop(0)
        if not self.oracle:
            self.kf.predict(u_applied)
        return u
