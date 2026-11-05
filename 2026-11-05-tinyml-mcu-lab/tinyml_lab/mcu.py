"""First-order MCU cost model: flash, SRAM, latency and energy of a deployed network.

Numbers are representative of each core class (datasheet-level order of magnitude), not
measurements of a specific part. The model exists to rank design points, not to predict them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .memplan import mlp_graph, liveness, plan_greedy_by_size
from .model import MLP
from .prune import layer_storage_bytes
from .quant import packed_bytes, QMLP

RUNTIME_SRAM_BYTES = 512   # stack + interpreter state, a crude constant


@dataclass(frozen=True)
class MCU:
    name: str
    flash_kb: int
    sram_kb: float
    mhz: float
    cyc_mac_int8: float      # cycles per int8 MAC (inner loop, dense)
    cyc_mac_fp32: float      # cycles per float32 MAC (soft-float if no FPU)
    cyc_sparse_mac: float    # cycles per nonzero MAC in CSR (index load + branch overhead)
    active_mw: float


MCUS = [
    MCU("AVR-class 8-bit", 32, 2, 16, 30.0, 400.0, 45.0, 15.0),
    MCU("Cortex-M0+", 128, 36, 64, 7.0, 90.0, 11.0, 8.0),
    MCU("Cortex-M4F", 512, 128, 80, 1.5, 3.0, 4.0, 25.0),
    MCU("Cortex-M7", 2048, 512, 400, 0.6, 1.2, 1.8, 180.0),
]


@dataclass
class Deployment:
    label: str
    macs: int            # MACs actually executed
    sparse_macs: int     # of which executed through the CSR path (cost more per MAC)
    flash_bytes: int
    sram_bytes: int
    wbits: int
    n_outputs: int = 0   # neurons requantized (per-output overhead)


def mlp_deployment(label: str, model: MLP, wbits: int, abits: int, per_channel: bool = True,
                   qmlp: QMLP | None = None) -> Deployment:
    """Describe an MLP as flash/SRAM/MAC numbers. Pass `qmlp` to use its true integer weights."""
    flash, macs, sparse = 0, 0, 0
    for i, W in enumerate(model.W):
        src = qmlp.layers[i].Wq if qmlp is not None else W
        b, fmt = layer_storage_bytes(src, wbits)
        flash += b
        nnz = int(np.count_nonzero(src))
        if fmt == "csr":
            macs += nnz
            sparse += nnz
        else:
            macs += W.size
        out = W.shape[1]
        if wbits >= 32:
            flash += 4 * out                       # float bias
        else:
            flash += 4 * out                       # int32 bias
            flash += (4 + 1) * (out if per_channel else 1)  # multiplier + shift
    arena = _arena(model.sizes, max(1, abits // 8))
    return Deployment(label, macs, sparse, flash, arena + RUNTIME_SRAM_BYTES, wbits,
                      sum(W.shape[1] for W in model.W))


def _arena(sizes, act_bytes) -> int:
    g = mlp_graph(sizes, act_bytes)
    _, arena = plan_greedy_by_size(liveness(g))
    return arena


def evaluate(dep: Deployment, mcu: MCU) -> dict:
    """Latency (ms), energy (uJ), and whether the model fits the part."""
    fp = dep.wbits >= 32
    dense_macs = dep.macs - dep.sparse_macs
    base = mcu.cyc_mac_fp32 if fp else mcu.cyc_mac_int8
    cyc = dense_macs * base + dep.sparse_macs * (mcu.cyc_sparse_mac if not fp else mcu.cyc_mac_fp32 * 1.5)
    if not fp and dep.wbits < 8:
        cyc += dep.macs * 0.5   # nibble unpack/shift per MAC
    if not fp:
        cyc += dep.n_outputs * 8        # requantize + clamp per output neuron
    ms = cyc / (mcu.mhz * 1e3)
    uj = ms * mcu.active_mw
    return {
        "latency_ms": float(ms),
        "energy_uJ": float(uj),
        "fits_flash": dep.flash_bytes <= mcu.flash_kb * 1024,
        "fits_sram": dep.sram_bytes <= mcu.sram_kb * 1024,
        "flash_util": dep.flash_bytes / (mcu.flash_kb * 1024),
        "sram_util": dep.sram_bytes / (mcu.sram_kb * 1024),
    }


def graph_sram(arena_bytes: int) -> int:
    return arena_bytes + RUNTIME_SRAM_BYTES


def fits_sram(arena_bytes: int, mcu: MCU) -> bool:
    return graph_sram(arena_bytes) <= mcu.sram_kb * 1024
