"""Quantization: affine integer inference (gemmlowp-style requantization) and Qm.n fixed point.

Two deployment-style inference engines are implemented with *integer-only* arithmetic
(int64 accumulators emulate what a MCU does with 32-bit MACs plus a 64-bit multiply):

* ``QMLP``       - per-tensor / per-channel symmetric integer weights, calibrated activations,
                   requantization by an int32 multiplier and a right shift.
* ``fixed_forward`` - plain Qm.n fixed point with saturation, no per-layer scales at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import MLP


def qmax(bits: int) -> int:
    """Largest magnitude of a symmetric signed integer with `bits` bits (e.g. 127 for 8)."""
    return 2 ** (bits - 1) - 1


def packed_bytes(n_values: int, bits: int) -> int:
    """Bytes needed to store n values bit-packed at `bits` bits each."""
    return int(math.ceil(n_values * bits / 8.0))


# --------------------------------------------------------------------------- weights
def quantize_weight(W: np.ndarray, bits: int, per_channel: bool = True):
    """Symmetric weight quantization. W is (in, out); a channel is an output column.

    Returns integer weights and a scale vector of shape (out,) so W ~= q * scale.
    """
    if per_channel:
        m = np.abs(W).max(axis=0)
    else:
        m = np.full(W.shape[1], np.abs(W).max())
    scale = np.maximum(m, 1e-12) / qmax(bits)
    q = np.clip(np.round(W / scale), -qmax(bits), qmax(bits)).astype(np.int64)
    return q, scale


def sqnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    """Signal-to-quantization-noise ratio in dB."""
    noise = np.sum((ref - approx) ** 2)
    sig = np.sum(ref ** 2)
    if noise == 0:
        return float("inf")
    return float(10 * np.log10(sig / noise))


# --------------------------------------------------------------------------- requantization
def quantize_multiplier(real: float):
    """Represent a positive real multiplier as (M0, total_shift) with M0 an int32 in [2^30, 2^31).

    real ~= M0 / 2**total_shift. This is the trick used by gemmlowp/TFLite-Micro so that the
    rescale after each layer needs only an integer multiply and an arithmetic shift.
    """
    if real <= 0:
        return 0, 31
    m, exp = math.frexp(real)  # real = m * 2**exp, m in [0.5, 1)
    M0 = int(round(m * (1 << 31)))
    if M0 == (1 << 31):
        M0 //= 2
        exp += 1
    total_shift = 31 - exp
    total_shift = max(1, min(62, total_shift))
    return M0, total_shift


def requantize(acc: np.ndarray, M0: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Round-to-nearest (acc * M0) >> shift, per-channel, entirely in int64."""
    prod = acc.astype(np.int64) * M0.astype(np.int64)
    rnd = np.left_shift(np.int64(1), (shift - 1).astype(np.int64))
    return np.right_shift(prod + rnd, shift.astype(np.int64))


# --------------------------------------------------------------------------- calibration
def calibrate(model: MLP, Xcal: np.ndarray, abits: int, pct: float = 100.0):
    """Activation scales for [input, layer1 out, ..., logits] from a calibration set."""
    _, acts = model.forward(Xcal, return_acts=True)
    scales = []
    for a in acts:
        mag = np.percentile(np.abs(a), pct) if pct < 100 else np.abs(a).max()
        scales.append(max(mag, 1e-9) / qmax(abits))
    return scales


@dataclass
class QLayer:
    Wq: np.ndarray       # (in, out) int64 holding w_bits-wide values
    bq: np.ndarray       # (out,) int64 bias at scale s_in * s_w
    M0: np.ndarray       # (out,) int64 fixed-point multiplier
    shift: np.ndarray    # (out,) int64 right shift
    relu: bool
    w_scale: np.ndarray  # (out,) float, kept for analysis only


@dataclass
class QMLP:
    layers: list
    s_input: float
    wbits: int
    abits: int
    s_out: float

    def int_forward(self, X: np.ndarray) -> np.ndarray:
        """Integer-only inference. Returns int logits at scale `s_out`."""
        hi = qmax(self.abits)
        x = np.clip(np.round(X / self.s_input), -hi, hi).astype(np.int64)
        for L in self.layers:
            acc = x @ L.Wq + L.bq
            y = requantize(acc, L.M0, L.shift)
            if L.relu:
                y = np.maximum(y, 0)
            x = np.clip(y, -hi, hi)
        return x

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.int_forward(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def nnz(self) -> int:
        return int(sum(np.count_nonzero(L.Wq) for L in self.layers))

    def max_accumulator_bits(self, X: np.ndarray) -> int:
        """Widest signed accumulator actually needed on X (to justify int32 MACs)."""
        hi = qmax(self.abits)
        x = np.clip(np.round(X / self.s_input), -hi, hi).astype(np.int64)
        worst = 1
        for L in self.layers:
            acc = x @ L.Wq + L.bq
            worst = max(worst, int(np.abs(acc).max()))
            y = requantize(acc, L.M0, L.shift)
            if L.relu:
                y = np.maximum(y, 0)
            x = np.clip(y, -hi, hi)
        return int(np.ceil(np.log2(worst + 1))) + 1


def build_qmlp(model: MLP, Xcal: np.ndarray, wbits: int = 8, abits: int = 8,
               per_channel: bool = True, pct: float = 100.0) -> QMLP:
    """Post-training quantization of a trained MLP using a calibration set."""
    scales = calibrate(model, Xcal, abits, pct)
    layers = []
    for i, (W, b) in enumerate(zip(model.W, model.b)):
        Wq, sw = quantize_weight(W, wbits, per_channel)
        s_in, s_out = scales[i], scales[i + 1]
        bq = np.round(b / (s_in * sw)).astype(np.int64)
        mults = [quantize_multiplier(s_in * s / s_out) for s in sw]
        M0 = np.array([m[0] for m in mults], dtype=np.int64)
        sh = np.array([m[1] for m in mults], dtype=np.int64)
        layers.append(QLayer(Wq, bq, M0, sh, i < model.n_layers - 1, sw))
    return QMLP(layers, scales[0], wbits, abits, scales[-1])


# --------------------------------------------------------------------------- Qm.n fixed point
def to_fixed(x: np.ndarray, word_bits: int, frac_bits: int, return_sat: bool = False):
    """Float -> signed Qm.n integer with round-to-nearest and saturation."""
    hi, lo = 2 ** (word_bits - 1) - 1, -(2 ** (word_bits - 1))
    raw = np.round(np.asarray(x) * (1 << frac_bits))
    q = np.clip(raw, lo, hi).astype(np.int64)
    if return_sat:
        return q, int(np.sum((raw > hi) | (raw < lo)))
    return q


def from_fixed(q: np.ndarray, frac_bits: int) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) / (1 << frac_bits)


def fixed_forward(model: MLP, X: np.ndarray, w_bits: int = 8, w_frac: int = 6,
                  a_bits: int = 16, a_frac: int = 8):
    """Fixed-point inference with a single Q format per tensor class.

    acc has (w_frac + a_frac) fractional bits; after each layer we shift right by w_frac
    (rounding) to get back to a_frac fractional bits, then saturate to the a_bits word.
    Returns (float logits, total saturation events, saturated fraction of activations).
    """
    a_hi, a_lo = 2 ** (a_bits - 1) - 1, -(2 ** (a_bits - 1))
    x, sat = to_fixed(X, a_bits, a_frac, return_sat=True)
    total = x.size
    for i, (W, b) in enumerate(zip(model.W, model.b)):
        Wq, s1 = to_fixed(W, w_bits, w_frac, return_sat=True)
        bq = np.round(b * (1 << (w_frac + a_frac))).astype(np.int64)
        acc = x @ Wq + bq
        y = np.right_shift(acc + (1 << (w_frac - 1)) if w_frac > 0 else acc, w_frac)
        s2 = int(np.sum((y > a_hi) | (y < a_lo)))
        y = np.clip(y, a_lo, a_hi)
        if i < model.n_layers - 1:
            y = np.maximum(y, 0)
        sat += s1 + s2
        total += Wq.size + y.size
        x = y
    return from_fixed(x, a_frac), sat, sat / total
