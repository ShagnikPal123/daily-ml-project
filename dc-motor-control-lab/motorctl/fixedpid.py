"""Layer 2: integer-only PI controller (the algorithm as it would run on an MCU without an FPU).

Gains, error and output live in Q(total_bits, frac_bits); the integrator uses `guard_bits` extra
fractional bits so that the tiny per-sample increment ki*Ts*e does not underflow to zero.
"""
from __future__ import annotations

from motorctl.controllers import PIDGains
from motorctl.fixedpoint import QFormat, rescale_round


class FixedPointPID:
    name = "pi_fixed"

    def __init__(self, gains: PIDGains, Ts: float, fmt: QFormat, guard_bits: int = 0,
                 antiwindup: str = "clamp"):
        self.q = fmt
        self.g = guard_bits
        self.qa = QFormat(fmt.total_bits + guard_bits, fmt.frac_bits + guard_bits)
        self.kp_q = fmt.quantize(gains.kp)
        self.kits_q = self.qa.quantize(gains.ki * Ts)  # ki*Ts folded at design time
        self.umax_q = fmt.quantize(gains.umax)
        self.aw = antiwindup
        self.reset()

    def reset(self):
        self.acc = 0  # integrator in the guard-extended format

    def step(self, r, y, x_true=None):
        q = self.q
        e = q.saturate(q.quantize(r) - q.quantize(y))
        p = q.mul(self.kp_q, e)
        i_out = q.saturate(rescale_round(self.acc, self.g))
        v = q.add(p, i_out)
        u = min(max(v, -self.umax_q), self.umax_q)
        winding = (u != v) and (e * v > 0)
        if not (self.aw == "clamp" and winding):
            inc = rescale_round(e * self.kits_q, q.frac_bits)  # scale 2^(f+g)
            self.acc = self.qa.saturate(self.acc + inc)
        return q.to_float(u)

    @property
    def effective_ki_ts(self) -> float:
        """Integral gain per sample actually realised after quantisation."""
        return self.qa.to_float(self.kits_q)
