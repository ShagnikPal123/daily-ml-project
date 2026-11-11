"""Layer 0: signed two's-complement fixed-point helpers (Qm.n arithmetic with saturation)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def rescale_round(prod: int, shift: int) -> int:
    """Arithmetic right shift with round-half-up (what a DSP MAC + rounding does)."""
    if shift <= 0:
        return prod << (-shift)
    return (prod + (1 << (shift - 1))) >> shift


@dataclass(frozen=True)
class QFormat:
    """Signed fixed point: `total_bits` including sign, `frac_bits` fractional."""

    total_bits: int
    frac_bits: int

    def __post_init__(self):
        if not (2 <= self.total_bits <= 62):
            raise ValueError("total_bits must be in [2, 62]")
        if not (0 <= self.frac_bits < self.total_bits):
            raise ValueError("frac_bits must be in [0, total_bits)")

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def min_int(self) -> int:
        return -(1 << (self.total_bits - 1))

    @property
    def max_int(self) -> int:
        return (1 << (self.total_bits - 1)) - 1

    @property
    def lsb(self) -> float:
        return 1.0 / self.scale

    @property
    def max_value(self) -> float:
        return self.max_int / self.scale

    @property
    def min_value(self) -> float:
        return self.min_int / self.scale

    def saturate(self, i: int) -> int:
        return int(min(max(int(i), self.min_int), self.max_int))

    def quantize(self, x: float) -> int:
        return self.saturate(math.floor(x * self.scale + 0.5))

    def to_float(self, i: int) -> float:
        return i / self.scale

    def mul(self, a: int, b: int) -> int:
        """Q x Q -> Q multiply with rounding and saturation."""
        return self.saturate(rescale_round(a * b, self.frac_bits))

    def add(self, a: int, b: int) -> int:
        return self.saturate(a + b)

    def quantize_array(self, x) -> np.ndarray:
        q = np.floor(np.asarray(x, dtype=float) * self.scale + 0.5)
        return np.clip(q, self.min_int, self.max_int).astype(np.int64)

    def dequantize_array(self, q) -> np.ndarray:
        return np.asarray(q, dtype=float) / self.scale

    def roundtrip(self, x: float) -> float:
        return self.to_float(self.quantize(x))
