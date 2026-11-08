"""Layer 0: fixed-point and integer quantization primitives.

Everything here is dependency-free (NumPy only) and models what a camera ADC or
an integer inference engine does with bits: uniform quantization, symmetric
int8 quantization, Q-format fixed point with saturation, and integer matmul with
an explicit accumulator-width check.
"""
from __future__ import annotations

import numpy as np


def qrange(bits: int, signed: bool = True) -> tuple[int, int]:
    """Inclusive integer range representable with ``bits`` bits."""
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def quant_step(bits: int, lo: float = 0.0, hi: float = 1.0) -> float:
    """LSB size of a uniform quantizer spanning [lo, hi] with 2**bits codes."""
    return (hi - lo) / (2**bits - 1)


def uniform_quantize(x, bits: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Mid-tread uniform ADC: clip to [lo, hi], round to one of 2**bits codes."""
    if hi <= lo:
        raise ValueError("hi must exceed lo")
    t = (np.clip(np.asarray(x, dtype=float), lo, hi) - lo) / (hi - lo)
    return np.rint(t * (2**bits - 1)).astype(np.int64)


def uniform_dequantize(codes, bits: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Map ADC codes back to analog levels (code 0 -> lo, full code -> hi)."""
    return lo + np.asarray(codes, dtype=float) * quant_step(bits, lo, hi)


def symmetric_scale(x, bits: int = 8) -> float:
    """Per-tensor scale so that max|x| maps to the largest positive code."""
    qmax = qrange(bits)[1]
    m = float(np.max(np.abs(x))) if np.size(x) else 0.0
    return m / qmax if m > 0 else 1.0


def quantize_symmetric(x, scale: float, bits: int = 8) -> np.ndarray:
    """Round-to-nearest symmetric quantization with saturation (int32 storage)."""
    qmin, qmax = qrange(bits)
    return np.clip(np.rint(np.asarray(x, dtype=float) / scale), qmin, qmax).astype(np.int32)


def dequantize_symmetric(q, scale: float) -> np.ndarray:
    return np.asarray(q, dtype=float) * scale


def quantize_unsigned(x, scale: float, bits: int = 8) -> np.ndarray:
    """Unsigned quantization for non-negative tensors (pixels, post-ReLU)."""
    qmax = qrange(bits, signed=False)[1]
    return np.clip(np.rint(np.asarray(x, dtype=float) / scale), 0, qmax).astype(np.int32)


def to_fixed(x, int_bits: int, frac_bits: int, signed: bool = True) -> np.ndarray:
    """Float -> Q(int_bits).(frac_bits) integer with saturation."""
    total = int_bits + frac_bits + (1 if signed else 0)
    lo, hi = qrange(total, signed)
    return np.clip(np.rint(np.asarray(x, dtype=float) * 2**frac_bits), lo, hi).astype(np.int64)


def from_fixed(q, frac_bits: int) -> np.ndarray:
    return np.asarray(q, dtype=float) / 2**frac_bits


def saturating_add(a, b, bits: int, signed: bool = True) -> np.ndarray:
    lo, hi = qrange(bits, signed)
    return np.clip(np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64), lo, hi)


def fixed_mul(a, b, frac_bits: int) -> np.ndarray:
    """Q-format multiply with round-to-nearest rescale back to ``frac_bits``."""
    prod = np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)
    return (prod + (1 << (frac_bits - 1))) >> frac_bits if frac_bits > 0 else prod


def accumulator_fits(acc, bits: int = 32) -> bool:
    """True when every accumulator value fits a signed ``bits``-bit register."""
    lo, hi = qrange(bits)
    acc = np.asarray(acc)
    return bool(acc.size == 0 or (acc.min() >= lo and acc.max() <= hi))


def int_matmul(a_q, b_q, acc_bits: int = 32) -> np.ndarray:
    """Integer GEMM with exact int64 arithmetic and an accumulator-overflow check."""
    out = np.asarray(a_q, dtype=np.int64) @ np.asarray(b_q, dtype=np.int64)
    if not accumulator_fits(out, acc_bits):
        raise OverflowError(f"accumulator exceeds {acc_bits} bits")
    return out


def requantize(acc, multiplier: float, bits: int = 8, signed: bool = False) -> np.ndarray:
    """Scale an int accumulator by a real multiplier and saturate to ``bits`` bits."""
    lo, hi = qrange(bits, signed)
    return np.clip(np.rint(np.asarray(acc, dtype=float) * multiplier), lo, hi).astype(np.int32)
