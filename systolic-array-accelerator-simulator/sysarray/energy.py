"""First-order energy / DRAM-traffic model for the accelerator.

Constants are order-of-magnitude figures for a ~45 nm process in the spirit of
Horowitz (ISSCC 2014): an 8-bit MAC costs a fraction of a pJ, an on-chip SRAM byte
a few times that, and an off-chip DRAM byte two orders of magnitude more. They are
*illustrative*; the point of the model is the relative ranking of dataflows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .pe_array import ArrayStats
from .tiling import Traffic


@dataclass(frozen=True)
class EnergyParams:
    mac_pj: float = 0.23            # int8 multiply + int32 add
    toggle_pj: float = 0.004        # per flipped operand-register bit
    sram_pj_per_byte: float = 1.25  # 32 KB-class scratchpad
    dram_pj_per_byte: float = 80.0  # LPDDR-class
    leak_pj_per_pe_cycle: float = 0.02
    gated_mac_fraction: float = 0.15  # residual energy of a MAC with operand gating on a zero input


def dram_bytes(dataflow: str, M: int, K: int, N: int, rows: int, cols: int,
               sram_bytes: int) -> int:
    """DRAM traffic with a scratchpad of `sram_bytes` (int8 operands in, int8 result out).

    Loop order is n-outer. An operand panel that is re-read once per n-tile must be
    re-fetched from DRAM each time unless the whole operand fits in half the scratchpad.
    For WS, int32 partial sums that exceed the other half spill to DRAM between k-tiles.
    """
    a_bytes, b_bytes, c_bytes = M * K, K * N, M * N
    n_tiles = math.ceil(N / cols)
    half = sram_bytes // 2
    a_fetch = a_bytes * (1 if a_bytes <= half else n_tiles)
    total = a_fetch + b_bytes + c_bytes
    if dataflow == "WS":
        k_tiles = math.ceil(K / rows)
        if k_tiles > 1 and M * min(N, cols) * 4 > half:
            total += 2 * M * N * 4 * (k_tiles - 1)  # write + read back of int32 psums
    return int(total)


def energy_breakdown(stats: ArrayStats, traffic: Traffic, dram: int,
                     p: EnergyParams = EnergyParams(), zero_gating: bool = False) -> dict[str, float]:
    """Energy in pJ split by component."""
    live = stats.macs - stats.zero_macs if zero_gating else stats.macs
    gated = stats.zero_macs if zero_gating else 0
    return {
        "mac": (live + gated * p.gated_mac_fraction) * p.mac_pj,
        "regs": stats.reg_toggles * p.toggle_pj,
        "leakage": stats.pe_cycles * p.leak_pj_per_pe_cycle,
        "sram": (traffic.sram_read + traffic.sram_write) * p.sram_pj_per_byte,
        "dram": dram * p.dram_pj_per_byte,
    }


def total_pj(breakdown: dict[str, float]) -> float:
    return float(sum(breakdown.values()))


def energy_delay_product(breakdown: dict[str, float], cycles: int, freq_hz: float = 500e6) -> float:
    """EDP in pJ*s."""
    return total_pj(breakdown) * cycles / freq_hz
