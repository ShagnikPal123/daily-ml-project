"""Tiling a large GEMM onto a fixed rows x cols array, plus closed-form models.

`tiled_matmul` actually drives the cycle-accurate arrays tile by tile; `analytic_*`
reproduce the same cycle / traffic counts in closed form (validated against the
simulator in the tests) so that big workloads can be swept without simulating.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pe_array import ARRAYS, ArrayStats


@dataclass
class Traffic:
    sram_read: int = 0    # bytes read from the on-chip scratchpad
    sram_write: int = 0   # bytes written to it (int32 partial sums count 4 B each)

    def __add__(self, o: "Traffic") -> "Traffic":
        return Traffic(self.sram_read + o.sram_read, self.sram_write + o.sram_write)


def _chunks(n: int, step: int):
    for s in range(0, n, step):
        yield s, min(step, n - s)


def tiled_matmul(A: np.ndarray, B: np.ndarray, rows: int, cols: int, dataflow: str,
                 ) -> tuple[np.ndarray, ArrayStats, Traffic]:
    """int8 GEMM through the simulated array. Returns exact int64 C, stats, scratchpad traffic."""
    M, K = A.shape
    N = B.shape[1]
    arr = ARRAYS[dataflow](rows, cols)
    C = np.zeros((M, N), dtype=np.int64)
    stats, traffic = ArrayStats(), Traffic()
    if dataflow == "OS":
        for n0, nt in _chunks(N, cols):
            for m0, mt in _chunks(M, rows):
                out, st = arr.run(np.ascontiguousarray(A[m0:m0 + mt]),
                                  np.ascontiguousarray(B[:, n0:n0 + nt]))
                C[m0:m0 + mt, n0:n0 + nt] = out
                stats += st
                traffic += Traffic(mt * K + K * nt, mt * nt * 4)
    elif dataflow == "WS":
        for n0, nt in _chunks(N, cols):
            for k0, kt in _chunks(K, rows):
                out, st = arr.run(np.ascontiguousarray(A[:, k0:k0 + kt]),
                                  np.ascontiguousarray(B[k0:k0 + kt, n0:n0 + nt]))
                C[:, n0:n0 + nt] += out
                stats += st
                traffic += Traffic(kt * nt + M * kt + (M * nt * 4 if k0 > 0 else 0), M * nt * 4)
    else:
        raise ValueError(f"unknown dataflow {dataflow!r}")
    return C, stats, traffic


def analytic_stats(M: int, K: int, N: int, rows: int, cols: int, dataflow: str,
                   toggle_rate: float = 0.5) -> ArrayStats:
    """Closed-form ArrayStats. Operand-register toggles are estimated (`toggle_rate`/bit)."""
    st = ArrayStats()
    if dataflow == "OS":
        for _, nt in _chunks(N, cols):
            for _, mt in _chunks(M, rows):
                st.cycles += (K + mt + nt - 2) + mt
        n_reg_bits = 2 * 8
    elif dataflow == "WS":
        for _, nt in _chunks(N, cols):
            for _, kt in _chunks(K, rows):
                st.cycles += kt + (M + kt + nt - 2)
                st.weight_load_cycles += kt
        n_reg_bits = 8
    else:
        raise ValueError(dataflow)
    st.macs = M * K * N
    st.pe_cycles = rows * cols * st.cycles
    st.reg_toggles = int(st.macs * n_reg_bits * toggle_rate)
    return st


def analytic_traffic(M: int, K: int, N: int, rows: int, cols: int, dataflow: str) -> Traffic:
    t = Traffic()
    if dataflow == "OS":
        for _, nt in _chunks(N, cols):
            for _, mt in _chunks(M, rows):
                t += Traffic(mt * K + K * nt, mt * nt * 4)
    else:
        for _, nt in _chunks(N, cols):
            for k0, kt in _chunks(K, rows):
                t += Traffic(kt * nt + M * kt + (M * nt * 4 if k0 > 0 else 0), M * nt * 4)
    return t


def overlapped_cycles(M: int, K: int, N: int, rows: int, cols: int, dataflow: str) -> int:
    """Idealised lower bound with perfect double buffering: successive tiles are issued
    back-to-back, so skew fill/drain is paid once instead of once per tile."""
    if dataflow == "OS":
        tiles = math.ceil(M / rows) * math.ceil(N / cols)
        return tiles * K + rows + cols - 2 + rows
    tiles = math.ceil(K / rows) * math.ceil(N / cols)
    return tiles * M + rows + cols - 2 + rows


def ideal_cycles(M: int, K: int, N: int, rows: int, cols: int) -> int:
    """Roofline: every PE does a MAC every cycle."""
    return math.ceil(M * K * N / (rows * cols))
