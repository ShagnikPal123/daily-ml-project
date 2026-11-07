"""Cycle-accurate systolic arrays.

Two classic dataflows are modelled register-by-register. Every simulated cycle, each
PE latches the operands arriving from its left/top neighbour, so the skewed "wavefront"
of the textbook diagrams emerges from the update rule instead of being hard-coded.

Output stationary (OS):  PE(i,j) owns C[i,j]. A[i,k] enters row i at cycle i+k, B[k,j]
                         enters column j at cycle j+k, so both meet at PE(i,j) at
                         cycle i+j+k and the partial sum never leaves the PE.
Weight stationary (WS):  PE(k,j) holds W[k,j]. A[m,k] enters row k at cycle m+k and
                         moves right; partial sums move *down* one PE per cycle and
                         C[m,j] leaves the bottom of column j at cycle m+(K-1)+j.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def toggles(old: np.ndarray, new: np.ndarray) -> int:
    """Hamming distance between two int8 register images (dynamic-power proxy)."""
    x = old.astype(np.int8).view(np.uint8) ^ new.astype(np.int8).view(np.uint8)
    return int(_POP[x].sum())


@dataclass
class ArrayStats:
    cycles: int = 0
    macs: int = 0            # multiply-accumulates with both operands valid
    zero_macs: int = 0       # MACs where an operand is zero (clock/operand-gating candidates)
    pe_cycles: int = 0       # rows*cols*cycles -- the PE-time actually paid for
    weight_load_cycles: int = 0
    reg_toggles: int = 0     # bit flips in 8-bit operand registers

    def __add__(self, other: "ArrayStats") -> "ArrayStats":
        return ArrayStats(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    @property
    def utilization(self) -> float:
        return self.macs / self.pe_cycles if self.pe_cycles else 0.0

    @property
    def zero_fraction(self) -> float:
        return self.zero_macs / self.macs if self.macs else 0.0


def _check_int8(*arrs: np.ndarray) -> None:
    for a in arrs:
        if a.dtype != np.int8:
            raise TypeError(f"systolic array operands must be int8, got {a.dtype}")


class OutputStationaryArray:
    """rows x cols grid; computes C(M<=rows, N<=cols) = A(M,K) @ B(K,N), K unbounded."""

    name = "OS"

    def __init__(self, rows: int, cols: int, record_trace: bool = False):
        self.rows, self.cols, self.record_trace = rows, cols, record_trace
        self.trace: list[np.ndarray] = []

    def run(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, ArrayStats]:
        _check_int8(A, B)
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError("inner dimensions differ")
        if M > self.rows or N > self.cols:
            raise ValueError("tile does not fit the array")
        self.trace = []
        acc = np.zeros((M, N), dtype=np.int64)
        a_reg = np.zeros((M, N), dtype=np.int64)
        b_reg = np.zeros((M, N), dtype=np.int64)
        va_reg = np.zeros((M, N), dtype=bool)
        vb_reg = np.zeros((M, N), dtype=bool)
        st = ArrayStats()
        ii, jj = np.arange(M), np.arange(N)
        compute_cycles = K + M + N - 2
        for t in range(compute_cycles):
            ka = t - ii
            oka = (ka >= 0) & (ka < K)
            inj_a = np.zeros(M, dtype=np.int64)
            inj_a[oka] = A[ii[oka], ka[oka]]
            kb = t - jj
            okb = (kb >= 0) & (kb < K)
            inj_b = np.zeros(N, dtype=np.int64)
            inj_b[okb] = B[kb[okb], jj[okb]]

            a_in = np.concatenate([inj_a[:, None], a_reg[:, :-1]], axis=1)
            va_in = np.concatenate([oka[:, None], va_reg[:, :-1]], axis=1)
            b_in = np.concatenate([inj_b[None, :], b_reg[:-1, :]], axis=0)
            vb_in = np.concatenate([okb[None, :], vb_reg[:-1, :]], axis=0)

            valid = va_in & vb_in
            acc += a_in * b_in
            st.macs += int(valid.sum())
            st.zero_macs += int((valid & ((a_in == 0) | (b_in == 0))).sum())
            st.reg_toggles += toggles(a_reg, a_in) + toggles(b_reg, b_in)
            if self.record_trace:
                self.trace.append(valid.copy())
            a_reg, b_reg, va_reg, vb_reg = a_in, b_in, va_in, vb_in
        drain = M  # results shift out one row per cycle
        st.cycles = compute_cycles + drain
        st.pe_cycles = self.rows * self.cols * st.cycles
        return acc, st


class WeightStationaryArray:
    """rows x cols grid; preloads W(K<=rows, N<=cols), streams A(M,K); M unbounded."""

    name = "WS"

    def __init__(self, rows: int, cols: int, record_trace: bool = False):
        self.rows, self.cols, self.record_trace = rows, cols, record_trace
        self.trace: list[np.ndarray] = []

    def run(self, A: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, ArrayStats]:
        _check_int8(A, W)
        M, K = A.shape
        K2, N = W.shape
        if K != K2:
            raise ValueError("inner dimensions differ")
        if K > self.rows or N > self.cols:
            raise ValueError("weight tile does not fit the array")
        self.trace = []
        Wl = W.astype(np.int64)
        a_reg = np.zeros((K, N), dtype=np.int64)
        va_reg = np.zeros((K, N), dtype=bool)
        p_reg = np.zeros((K, N), dtype=np.int64)
        out = np.zeros((M, N), dtype=np.int64)
        st = ArrayStats()
        kk, jj = np.arange(K), np.arange(N)
        compute_cycles = M + K + N - 2
        for t in range(compute_cycles):
            m = t - kk
            ok = (m >= 0) & (m < M)
            inj = np.zeros(K, dtype=np.int64)
            inj[ok] = A[m[ok], kk[ok]]
            a_in = np.concatenate([inj[:, None], a_reg[:, :-1]], axis=1)
            va_in = np.concatenate([ok[:, None], va_reg[:, :-1]], axis=1)
            p_in = np.concatenate([np.zeros((1, N), dtype=np.int64), p_reg[:-1, :]], axis=0)
            p_new = p_in + a_in * Wl
            st.macs += int(va_in.sum())
            st.zero_macs += int((va_in & ((a_in == 0) | (Wl == 0))).sum())
            st.reg_toggles += toggles(a_reg, a_in)
            if self.record_trace:
                self.trace.append(va_in.copy())
            a_reg, va_reg, p_reg = a_in, va_in, p_new
            mo = t - (K - 1) - jj
            oko = (mo >= 0) & (mo < M)
            out[mo[oko], jj[oko]] = p_reg[K - 1, jj[oko]]
        st.weight_load_cycles = K  # one weight row enters per cycle
        st.cycles = K + compute_cycles
        st.pe_cycles = self.rows * self.cols * st.cycles
        return out, st


ARRAYS = {"OS": OutputStationaryArray, "WS": WeightStationaryArray}
