"""Integer quantization helpers: symmetric int8 tensors and hardware-style requantization.

The accelerator only ever sees int8 operands and accumulates in wide integers, so the
float -> int8 mapping and the int32 -> int8 "requantize" step (a fixed-point multiply
followed by an arithmetic right shift, as in TFLite / NPU post-processing units) live here.
"""
from __future__ import annotations

import numpy as np

INT8_MIN, INT8_MAX = -128, 127


def quantize_sym(x: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor int8 quantization: q = clip(round(x / s)), s = max|x| / 127."""
    x = np.asarray(x, dtype=np.float64)
    if scale is None:
        m = float(np.max(np.abs(x))) if x.size else 0.0
        scale = m / INT8_MAX if m > 0 else 1.0
    q = np.clip(np.rint(x / scale), INT8_MIN, INT8_MAX).astype(np.int8)
    return q, float(scale)


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float64) * scale


def to_fixed_multiplier(m: float) -> tuple[int, int]:
    """Represent a positive real multiplier as (mult, shift): m ~= mult / 2**shift.

    mult is a 31-bit integer in [2**30, 2**31), the usual normalised mantissa used by
    integer-only inference engines.
    """
    if m <= 0:
        raise ValueError("multiplier must be positive")
    shift = 31
    while m < 0.5:
        m *= 2.0
        shift += 1
    while m >= 1.0:
        m /= 2.0
        shift -= 1
    mult = int(round(m * (1 << 31)))
    if mult == (1 << 31):  # rounding pushed us up to 1.0
        mult //= 2
        shift -= 1
    return mult, shift


def requantize_fixed(acc: np.ndarray, mult: int, shift: int, relu: bool = False) -> np.ndarray:
    """int32 accumulator -> int8 via (acc*mult + round) >> shift, optional fused ReLU."""
    acc = np.asarray(acc, dtype=np.int64)
    out = (acc * np.int64(mult) + (np.int64(1) << (shift - 1))) >> shift
    lo = 0 if relu else INT8_MIN
    return np.clip(out, lo, INT8_MAX).astype(np.int8)


def sqnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB."""
    ref = np.asarray(ref, dtype=np.float64)
    err = ref - np.asarray(approx, dtype=np.float64)
    p_err = float(np.mean(err**2))
    return float("inf") if p_err == 0 else 10 * np.log10(float(np.mean(ref**2)) / p_err)
