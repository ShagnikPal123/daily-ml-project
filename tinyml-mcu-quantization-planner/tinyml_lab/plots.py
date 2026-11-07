"""Matplotlib figures for the experiment results (headless Agg backend)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiments import pareto_front
from .memplan import PLANNERS, liveness, plan_random_search, tiny_resnet_graph, unet_graph


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_bits(res: dict, act: dict, path: str):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    b = res["bits"]
    for key, c in (("per_tensor", "tab:red"), ("per_channel", "tab:blue")):
        m = np.array([x["mean"] for x in res[key]]) * 100
        s = np.array([x["std"] for x in res[key]]) * 100
        ax[0].errorbar(b, m, yerr=s, marker="o", color=c, label=key.replace("_", "-"), capsize=3)
        ax[1].plot(b, res[key + "_sqnr_db"], marker="s", color=c, label=key.replace("_", "-"))
    ax[0].axhline(res["float"]["mean"] * 100, color="k", ls="--", label="float32")
    ax[0].set(xlabel="weight bits", ylabel="test accuracy (%)", title="Weight quantization")
    ax[1].set(xlabel="weight bits", ylabel="logit SQNR (dB)", title="Output SQNR (6 dB/bit slope)")
    for a in ax[:2]:
        a.legend(); a.grid(alpha=.3)
    for p, row in act["acc"].items():
        ax[2].plot(act["abits"], np.array(row) * 100, marker="o", label=f"clip @ p{p}")
    ax[2].set(xlabel="activation bits", ylabel="test accuracy (%)", title="Activation bits x calibration clip")
    ax[2].legend(); ax[2].grid(alpha=.3)
    return _save(fig, path)


def plot_fixed(res: dict, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, key, title in ((ax[0], "acc", "Accuracy"), (ax[1], "sat_frac", "Saturated fraction")):
        im = a.imshow(np.array(res[key]), aspect="auto", origin="lower", cmap="viridis" if key == "acc" else "magma")
        a.set_xticks(range(len(res["a_fracs"])), res["a_fracs"])
        a.set_yticks(range(len(res["w_fracs"])), res["w_fracs"])
        a.set(xlabel=f"activation frac bits (Q.{res['a_bits']}-bit word)",
              ylabel=f"weight frac bits ({res['w_bits']}-bit word)", title=title)
        fig.colorbar(im, ax=a)
        for i in range(len(res["w_fracs"])):
            for j in range(len(res["a_fracs"])):
                v = res[key][i][j]
                a.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=7)
    return _save(fig, path)


def plot_pareto(res: dict, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    styles = {("global", True): ("tab:blue", "o"), ("global", False): ("tab:cyan", "x"),
              ("layer", True): ("tab:green", "s"), ("layer", False): ("tab:olive", "+")}
    pts = []
    for r in res["unstructured"]:
        c, mk = styles[(r["scope"], r["finetune"])]
        pts.append((r["flash_bytes"], r["acc"]))
    for r in res["structured"]:
        pts.append((r["flash_bytes"], r["acc"]))
    front = set(pareto_front(pts))
    for k, (c, mk) in styles.items():
        rows = [r for r in res["unstructured"] if (r["scope"], r["finetune"]) == k]
        ax[0].plot([r["flash_bytes"] / 1024 for r in rows], [r["acc"] * 100 for r in rows], marker=mk, color=c,
                   label=f"{k[0]}{' + finetune' if k[1] else ''}")
        ax[1].plot([r["sparsity"] * 100 for r in rows], [r["acc"] * 100 for r in rows], marker=mk, color=c,
                   label=f"{k[0]}{' + finetune' if k[1] else ''}")
    s = res["structured"]
    ax[0].plot([r["flash_bytes"] / 1024 for r in s], [r["acc"] * 100 for r in s], "k^-", label="structured + finetune")
    fx = [pts[i][0] / 1024 for i in sorted(front)]
    fy = [pts[i][1] * 100 for i in sorted(front)]
    ax[0].scatter(fx, fy, s=140, facecolors="none", edgecolors="gold", linewidths=2, label="Pareto front")
    ax[0].set(xlabel="flash for weights (KiB)", ylabel="test accuracy (%)", title="Accuracy vs flash (int8)")
    ax[1].set(xlabel="target sparsity (%)", ylabel="test accuracy (%)", title="Pruning curves")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=.3)
    return _save(fig, path)


def plot_memplan(res: dict, path: str):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, :1])
    names = [g["name"] for g in res["graphs"]]
    x = np.arange(len(names))
    keys = list(PLANNERS) + ["lower_bound"]
    w = 0.16
    for i, k in enumerate(keys):
        vals = [(g["planners"][k] if k != "lower_bound" else g["lower_bound"]) / 1024 for g in res["graphs"]]
        ax0.bar(x + (i - 2) * w, vals, w, label=k)
    ax0.set_yscale("log"); ax0.set_xticks(x, names, fontsize=8)
    ax0.set(ylabel="arena (KiB, log)", title="Arena size by planner"); ax0.legend(fontsize=7)
    ax1 = fig.add_subplot(gs[0, 1])
    sav = [100 * (1 - g["planners"]["random_search"] / g["planners"]["naive"]) for g in res["graphs"]]
    ax1.bar(names, sav, color="tab:green"); ax1.set_ylabel("SRAM saved vs naive (%)")
    ax1.set_title("Planner savings"); ax1.tick_params(axis="x", labelsize=8)
    ax2 = fig.add_subplot(gs[1, :])
    g = unet_graph(32, 8, 3, 1)
    plan, arena = plan_random_search(liveness(g))
    cmap = plt.get_cmap("tab20")
    for i, b in enumerate(sorted(plan, key=lambda b: b.first)):
        ax2.barh(b.offset / 1024 + b.size / 2048, b.last - b.first + 1, left=b.first, height=b.size / 1024,
                 color=cmap(i % 20), edgecolor="k", linewidth=.5)
        ax2.text(b.first + .05, b.offset / 1024 + b.size / 2048, b.name, fontsize=6, va="center")
    ax2.set(xlabel="op index (time)", ylabel="arena address (KiB)",
            title=f"U-Net tensor arena layout ({arena / 1024:.1f} KiB; skip tensors live across the net)")
    return _save(fig, path)


def plot_mcu(res: dict, path: str):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    designs = res["designs"]
    mcus = res["mcus"]
    x = np.arange(len(mcus))
    w = 0.16
    for i, d in enumerate(designs):
        lat = [d["mcus"][m]["latency_ms"] for m in mcus]
        en = [d["mcus"][m]["energy_uJ"] for m in mcus]
        fit = [d["mcus"][m]["fits_flash"] and d["mcus"][m]["fits_sram"] for m in mcus]
        bars = ax[0].bar(x + (i - 2) * w, lat, w, label=f"{d['design']} ({d['acc'] * 100:.1f}%)")
        for bar, f in zip(bars, fit):
            if not f:
                bar.set_hatch("///"); bar.set_alpha(.4)
        ax[1].bar(x + (i - 2) * w, en, w)
    for a, t in zip(ax, ("Latency per inference (ms, log)", "Energy per inference (uJ, log)")):
        a.set_yscale("log"); a.set_xticks(x, mcus, fontsize=8); a.set_title(t); a.grid(alpha=.3, axis="y")
    ax[0].legend(fontsize=7)
    fig.text(0.5, 0.005, "hatched = does not fit flash/SRAM of that part", ha="center", fontsize=8)
    return _save(fig, path)
