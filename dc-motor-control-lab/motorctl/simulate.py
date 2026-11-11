"""Layer 3: closed-loop simulation with encoder quantisation, noise, saturation and compute delay."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motorctl.plant import MotorParams, discretize, encoder_velocity_step


@dataclass(frozen=True)
class SimConfig:
    Ts: float = 1e-3
    T: float = 0.6
    t_step: float = 0.02
    ref: float = 100.0  # rad/s
    t_dist: float = 0.30
    tau_load: float = 0.02  # N m applied at t_dist
    enc_cpr: int = 4000  # encoder counts per revolution (0 = ideal)
    meas_sigma: float = 0.5  # rad/s of speed-measurement noise (pre-quantisation)
    delay: int = 1  # samples between computing u and it reaching the bridge
    seed: int = 0

    @property
    def enc_step(self) -> float:
        return encoder_velocity_step(self.enc_cpr, self.Ts)

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.Ts))


@dataclass
class SimResult:
    t: np.ndarray
    y: np.ndarray  # true speed
    y_meas: np.ndarray
    u: np.ndarray  # commanded voltage
    u_applied: np.ndarray
    r: np.ndarray
    i: np.ndarray


def simulate(p: MotorParams, controller, cfg: SimConfig) -> SimResult:
    Ad, Bd, _ = discretize(p, cfg.Ts)
    bv, bt = Bd[:, 0], Bd[:, 1]
    rng = np.random.default_rng(cfg.seed)
    controller.reset()
    n = cfg.n_steps
    x = np.zeros(2)
    qstep = cfg.enc_step
    queue = [0.0] * cfg.delay
    t = np.arange(n) * cfg.Ts
    out = {k: np.zeros(n) for k in ("y", "y_meas", "u", "u_applied", "r", "i")}
    for k in range(n):
        r = cfg.ref if t[k] >= cfg.t_step else 0.0
        tau = cfg.tau_load if t[k] >= cfg.t_dist else 0.0
        ym = x[1] + (rng.normal(0.0, cfg.meas_sigma) if cfg.meas_sigma > 0 else 0.0)
        if qstep > 0:
            ym = np.round(ym / qstep) * qstep
        u = float(np.clip(controller.step(r, ym, x_true=x.copy()), -p.vmax, p.vmax))
        queue.append(u)
        ua = queue.pop(0)
        out["y"][k], out["y_meas"][k], out["u"][k] = x[1], ym, u
        out["u_applied"][k], out["r"][k], out["i"][k] = ua, r, x[0]
        x = Ad @ x + bv * ua + bt * tau
    return SimResult(t=t, **out)
