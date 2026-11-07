"""Matplotlib figures for the lab (Agg backend, written to PNG)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .pe_array import OutputStationaryArray, WeightStationaryArray  # noqa: E402
from .verify import random_int8  # noqa: E402

COLORS = {"OS": "#1f77b4", "WS": "#d62728"}
PARTS = ["mac", "regs", "leakage", "sram", "dram"]
PART_COLORS = ["#2ca02c", "#9467bd", "#7f7f7f", "#ff7f0e", "#d62728"]


def fig_wavefront(path: Path, n: int = 4, K: int = 6, seed: int = 0) -> None:
    """Active-PE maps at selected cycles for OS and WS -- the diagonal wavefront."""
    rng = np.random.default_rng(seed)
    A, B = random_int8(rng, (n, K)), random_int8(rng, (K, n))
    os_arr = OutputStationaryArray(n, n, record_trace=True)
    os_arr.run(A, B)
    ws_arr = WeightStationaryArray(n, n, record_trace=True)
    ws_arr.run(random_int8(rng, (K, n)), random_int8(rng, (n, n)))
    cyc = [1, 3, 5, 7]
    fig, axes = plt.subplots(2, len(cyc), figsize=(2.4 * len(cyc), 5.2))
    for r, (arr, nm) in enumerate(((os_arr, "Output stationary"), (ws_arr, "Weight stationary"))):
        for c, t in enumerate(cyc):
            tt = min(t, len(arr.trace) - 1)
            axes[r, c].imshow(arr.trace[tt], cmap="Greens", vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].set_title(f"{nm}\ncycle {tt}" if c == 0 else f"cycle {tt}", fontsize=8)
    fig.suptitle("Active PEs per cycle (green = MAC fires): diagonal wavefront", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_utilization(rows: list[dict], path: Path) -> None:
    wls = list(dict.fromkeys(r["workload"] for r in rows))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for ax, w in zip(axes.ravel(), wls):
        for df in ("OS", "WS"):
            sel = [r for r in rows if r["workload"] == w and r["dataflow"] == df]
            ax.plot([r["array"] for r in sel], [r["utilization"] for r in sel], "o-",
                    color=COLORS[df], label=df)
        ax.set_xscale("log", base=2); ax.set_title(w, fontsize=8); ax.grid(alpha=.3)
        ax.set_xlabel("array side"); 
    axes[0, 0].set_ylabel("PE utilization"); axes[1, 0].set_ylabel("PE utilization")
    axes[0, 0].legend()
    fig.suptitle("Utilization collapses when the array outgrows the GEMM")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_energy(rows: list[dict], path: Path, array: int = 16) -> None:
    wls = list(dict.fromkeys(r["workload"] for r in rows))
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(wls))
    for k, df in enumerate(("OS", "WS")):
        bottom = np.zeros(len(wls))
        for part, col in zip(PARTS, PART_COLORS):
            vals = np.array([next(r for r in rows if r["workload"] == w and r["array"] == array
                                  and r["dataflow"] == df)[f"e_{part}"] for w in wls])
            ax.bar(x + (k - .5) * .38, vals, .36, bottom=bottom, color=col,
                   label=part if k == 0 else None, edgecolor="white", linewidth=.4)
            bottom += vals
        for xi, b in zip(x, bottom):
            ax.text(xi + (k - .5) * .38, b, df, ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([w.split(" (")[0][:22] for w in wls],
                                                               rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("energy (pJ, log)"); ax.legend(ncol=5, fontsize=8)
    ax.set_title(f"Energy breakdown by component, {array}x{array} array, OS vs WS")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_sparsity(res: dict, path: Path) -> None:
    sp = res["sparsity"]
    x = [100 * s["sparsity"] for s in sp]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(x, [100 * s["acc"] for s in sp], "o-", color="#1f77b4", label="int8 accuracy")
    ax1.set_xlabel("weight sparsity (%)"); ax1.set_ylabel("test accuracy (%)", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(x, [s["saving_pct"] for s in sp], "s--", color="#2ca02c", label="energy saved by gating")
    ax2.plot(x, [s["saving_compressed_pct"] for s in sp], "D-.", color="#ff7f0e",
             label="gating + bitmap-compressed weights")
    ax2.plot(x, [100 * s["zero_mac_fraction"] for s in sp], "^:", color="#999", label="zero-operand MACs")
    ax2.set_ylabel("% (energy saved / zero MACs)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left"); ax1.grid(alpha=.3)
    ax1.set_title("Pruning, operand gating and weight compression, WS 8x8 (no fine-tuning)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_sram(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for df in ("OS", "WS"):
        sel = [r for r in rows if r["dataflow"] == df]
        ax.plot([r["sram_kb"] for r in sel], [r["dram_bytes"] / 1e6 for r in sel], "o-",
                color=COLORS[df], label=df)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.grid(alpha=.3, which="both")
    ax.set_xlabel("scratchpad (KB)"); ax.set_ylabel("DRAM traffic (MB)"); ax.legend()
    ax.set_title("DRAM traffic vs on-chip buffer, GEMM 256x512x256, 16x16 array")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
