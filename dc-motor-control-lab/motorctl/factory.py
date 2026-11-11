"""Layer 3: build any of the controllers by name with consistent design settings."""
from __future__ import annotations

from dataclasses import replace

from motorctl.controllers import PID, PIDGains
from motorctl.fixedpid import FixedPointPID
from motorctl.fixedpoint import QFormat
from motorctl.lqg import LQGController, design_lqg
from motorctl.plant import MotorParams

KINDS = ("pi", "pi_fixed", "lqi_oracle", "lqg")


def build_controller(kind: str, p: MotorParams, Ts: float, *, pi_gains: PIDGains,
                     meas_sigma: float = 0.5, enc_step: float = 0.0, delay: int = 0,
                     frac_bits: int = 14, guard_bits: int = 8, word_bits: int = 32,
                     antiwindup: str = "clamp", use_dff: bool = True):
    gains = replace(pi_gains, umax=p.vmax)
    if kind == "pi":
        return PID(gains, Ts, antiwindup)
    if kind == "pi_fixed":
        return FixedPointPID(gains, Ts, QFormat(word_bits, frac_bits), guard_bits, antiwindup)
    if kind in ("lqi_oracle", "lqg"):
        design = design_lqg(p, Ts, meas_std=meas_sigma, enc_step=enc_step, use_dff=use_dff, delay=delay)
        return LQGController(design, p.vmax, oracle=(kind == "lqi_oracle"))
    raise ValueError(f"unknown controller kind {kind!r}")
