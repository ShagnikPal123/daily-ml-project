"""Fixed-point quantiser for hardware-style decoding, plus a memory/area cost model."""
import numpy as np


class FixedPoint:
    """Symmetric saturating uniform quantiser: q(x) = clip(round(x/step), +-(2^(b-1)-1)) * step."""

    def __init__(self, bits, max_llr=12.0):
        self.bits = bits
        self.levels = (1 << (bits - 1)) - 1
        self.step = max_llr / self.levels
        self.max_llr = max_llr

    def __call__(self, x):
        q = np.clip(np.rint(np.asarray(x) / self.step), -self.levels, self.levels)
        return q * self.step

    def to_int(self, x):
        return np.clip(np.rint(np.asarray(x) / self.step), -self.levels, self.levels).astype(np.int32)


def message_memory_bits(code, bits):
    """Edge memory for one check-to-variable message store plus channel LLR store."""
    return code.num_edges() * bits + code.n * bits


def compressed_check_memory_bits(code, bits):
    """Min-sum check-node compression: store (min1, min2, argmin idx, dc sign bits) per check."""
    idx_bits = int(np.ceil(np.log2(code.dc)))
    per_check = 2 * (bits - 1) + idx_bits + code.dc
    return code.m * per_check + code.n * bits
