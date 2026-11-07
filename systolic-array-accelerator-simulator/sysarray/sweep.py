"""Experiments: dataflow x array-size sweeps, scratchpad sweep, sparsity-vs-energy study."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .energy import EnergyParams, dram_bytes, energy_breakdown, energy_delay_product, total_pj
from .mlp import (accel_matmul_factory, accuracy, load_data, numpy_matmul, prune_magnitude,
                  quantize_model, train_float)
from .tiling import analytic_stats, analytic_traffic, ideal_cycles, overlapped_cycles

# (name, M, K, N): GEMM shapes of realistic edge workloads after lowering.
WORKLOADS = {
    "mlp_fc1 (64x64)": (128, 64, 64),
    "conv3x3 16x16x8->16 (im2col)": (196, 72, 16),
    "conv1x1 pointwise 32->64": (256, 32, 64),
    "fc 512->10 (batch 1)": (1, 512, 10),
    "transformer proj 128x128x128": (128, 128, 128),
    "ragged 100x50x30": (100, 50, 30),
}


def gemm_sweep(sizes=(2, 4, 8, 16, 32), sram_kb: int = 32, params: EnergyParams = EnergyParams()):
    rows = []
    for wname, (M, K, N) in WORKLOADS.items():
        for s in sizes:
            for df in ("OS", "WS"):
                st = analytic_stats(M, K, N, s, s, df)
                tr = analytic_traffic(M, K, N, s, s, df)
                dr = dram_bytes(df, M, K, N, s, s, sram_kb * 1024)
                e = energy_breakdown(st, tr, dr, params)
                rows.append({
                    "workload": wname, "M": M, "K": K, "N": N, "array": s, "dataflow": df,
                    "cycles": st.cycles, "overlapped_cycles": overlapped_cycles(M, K, N, s, s, df),
                    "ideal_cycles": ideal_cycles(M, K, N, s, s),
                    "utilization": round(st.utilization, 4),
                    "sram_bytes": tr.sram_read + tr.sram_write, "dram_bytes": dr,
                    "energy_pj": round(total_pj(e), 1),
                    **{f"e_{k}": round(v, 1) for k, v in e.items()},
                    "edp_pj_s": energy_delay_product(e, st.cycles),
                })
    return rows


def sram_sweep(workload=(256, 512, 256), size=16, sram_kbs=(4, 8, 16, 32, 64, 128, 256)):
    M, K, N = workload
    out = []
    for kb in sram_kbs:
        for df in ("OS", "WS"):
            out.append({"sram_kb": kb, "dataflow": df,
                        "dram_bytes": dram_bytes(df, M, K, N, size, size, kb * 1024)})
    return out


def mlp_experiment(sizes=(4, 8, 16), sparsities=(0.0, 0.5, 0.8, 0.9), n_eval: int = 96,
                   hidden: int = 64, seed: int = 0, params: EnergyParams = EnergyParams()):
    Xtr, Xte, ytr, yte = load_data(seed)
    base = train_float(Xtr, ytr, hidden, seed)
    result = {"float_acc": accuracy(base.predict(Xte), yte), "bit_exact": True,
              "dense": [], "sparsity": []}
    qbase = quantize_model(base, Xtr)
    result["int8_acc"] = accuracy(qbase.predict(Xte, numpy_matmul), yte)
    Xe, ye = Xte[:n_eval], yte[:n_eval]
    ref_pred = qbase.predict(Xe, numpy_matmul)
    for s in sizes:
        for df in ("OS", "WS"):
            mm, log = accel_matmul_factory(s, s, df)
            pred = qbase.predict(Xe, mm)
            result["bit_exact"] &= bool(np.array_equal(pred, ref_pred))
            st, tr = log["stats"], log["traffic"]
            dr = sum(dram_bytes(df, *g, s, s, 32 * 1024) for g in log["gemms"])
            e = energy_breakdown(st, tr, dr, params)
            result["dense"].append({"array": s, "dataflow": df, "cycles": st.cycles,
                                    "utilization": st.utilization, "energy_pj": total_pj(e),
                                    **{f"e_{k}": v for k, v in e.items()}})
    for sp in sparsities:
        q = quantize_model(prune_magnitude(base, sp), Xtr)
        mm, log = accel_matmul_factory(8, 8, "WS")
        pred = q.predict(Xe, mm)
        st, tr = log["stats"], log["traffic"]
        dr = sum(dram_bytes("WS", *g, 8, 8, 32 * 1024) for g in log["gemms"])
        e0 = total_pj(energy_breakdown(st, tr, dr, params, zero_gating=False))
        e1 = total_pj(energy_breakdown(st, tr, dr, params, zero_gating=True))
        # bitmap-compressed weights: 1 byte per non-zero + 1 bit per position (DRAM side)
        dense_w = sum(l.Wq.size for l in q.layers)
        comp_w = sum(int(np.count_nonzero(l.Wq)) + l.Wq.size / 8 for l in q.layers)
        e2 = e1 - max(dense_w - comp_w, 0) * params.dram_pj_per_byte
        result["sparsity"].append({
            "sparsity": sp, "acc": accuracy(q.predict(Xte, numpy_matmul), yte),
            "zero_mac_fraction": st.zero_fraction, "energy_pj": e0, "energy_gated_pj": e1,
            "energy_gated_compressed_pj": e2, "weight_compression": comp_w / dense_w,
            "saving_pct": 100 * (1 - e1 / e0), "saving_compressed_pct": 100 * (1 - e2 / e0)})
    return result


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))
