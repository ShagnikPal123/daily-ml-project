"""Layer 5: experiment suite (headline comparison, sweeps, feature ablation, Kalman study, figures)."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import ExpConfig
from .detectors import DETECTOR_NAMES, build_detector
from .evaluation import evaluate_seed, make_data
from .filters import LocalLevelKalman, steady_state_gain
from .metrics import mean_ci

SWEEP_DETECTORS = ("threshold", "kmeans", "hmm-ret", "hmm-feat")
ABLATION_SETS = ("ret", "logvol", "ret+logvol", "logvol+spec", "ret+logvol+mom+spec")


def aggregate(runs: list[dict]) -> dict:
    """List of per-seed {method:{metric:val}} -> {method:{metric:[mean, ci_halfwidth]}}."""
    agg: dict = {}
    for method in runs[0]:
        agg[method] = {}
        for metric in runs[0][method]:
            m, h = mean_ci([r[method][metric] for r in runs])
            agg[method][metric] = [m, h]
    return agg


def run_headline(cfg: ExpConfig, names=DETECTOR_NAMES) -> dict:
    return aggregate([evaluate_seed(cfg, s, names) for s in range(cfg.seeds)])


def run_sweep(cfg: ExpConfig, param: str, values, names=SWEEP_DETECTORS, seeds: int = 4) -> dict:
    res = {}
    for v in values:
        c = cfg.replace(**{param: v, "seeds": seeds})
        res[str(v)] = aggregate([evaluate_seed(c, s, names, baselines=False) for s in range(seeds)])
    return res


def run_ablation(cfg: ExpConfig, sets=ABLATION_SETS, seeds: int = 4) -> dict:
    names = [f"hmm:{s}" for s in sets]
    return aggregate([evaluate_seed(cfg, s, names, baselines=False) for s in range(seeds)])


def run_kalman_study(ratios=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), n: int = 4000, seed: int = 0) -> list[dict]:
    """Empirical vs closed-form steady-state gain for the local-level Kalman filter."""
    rng = np.random.default_rng(seed)
    rows = []
    for ratio in ratios:
        r = 0.25
        q = ratio * r
        x = np.cumsum(rng.normal(scale=np.sqrt(q), size=n))
        y = x + rng.normal(scale=np.sqrt(r), size=n)
        xh, _, ks = LocalLevelKalman(q, r).filter(y)
        rows.append({"q_over_r": ratio, "gain_empirical": float(ks[-1]), "gain_theory": steady_state_gain(q, r),
                     "mse_kalman": float(np.mean((xh[200:] - x[200:]) ** 2)),
                     "mse_raw": float(np.mean((y[200:] - x[200:]) ** 2))})
    return rows


# ---------------- figures ----------------
def _errbar(ax, xs, agg_by_x, method, metric, label):
    m = [agg_by_x[str(x)][method][metric][0] for x in xs]
    h = [agg_by_x[str(x)][method][metric][1] for x in xs]
    ax.errorbar(xs, m, yerr=h, marker="o", capsize=3, label=label)


def fig_timeline(cfg: ExpConfig, path: str) -> None:
    sim = make_data(cfg, 0)
    n_tr = cfg.n_train
    det = build_detector("hmm-feat", cfg.k, cfg.feature_window, cfg.vol_lambda, cfg.n_iter, 0).fit(sim.returns[:n_tr])
    lab = det.predict(sim.returns[n_tr:])
    s = sim.states[n_tr:]
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    ax[0].plot(sim.price[n_tr:], lw=1)
    ax[0].set_ylabel("synthetic price")
    ax[1].step(range(len(s)), s, where="post", color="k")
    ax[1].set_ylabel("true regime")
    ax[2].step(range(len(lab)), lab, where="post", color="C3")
    ax[2].set_ylabel("HMM filter")
    ax[2].set_xlabel("test bar")
    fig.suptitle("Hidden regimes vs causal HMM-filter estimate (seed 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sweeps(sw_sep: dict, sw_stay: dict, cfg: ExpConfig, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, sw, xl in ((axes[0], sw_sep, "volatility separation"), (axes[1], sw_stay, "regime persistence p_stay")):
        xs = [float(k) for k in sw]
        for name in SWEEP_DETECTORS:
            m = [sw[k][name]["acc"][0] for k in sw]
            h = [sw[k][name]["acc"][1] for k in sw]
            ax.errorbar(xs, m, yerr=h, marker="o", capsize=3, label=name)
        ax.axhline(1.0 / cfg.k, color="gray", ls=":", label="chance")
        ax.set_xlabel(xl)
        ax.set_ylabel("test accuracy")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_sharpe(head: dict, path: str) -> None:
    names = list(head)
    m = [head[n]["sharpe"][0] for n in names]
    h = [head[n]["sharpe"][1] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, m, yerr=h, capsize=4, color=["C0"] * 5 + ["gray", "gray", "C2"][: len(names) - 5])
    ax.set_ylabel("net Sharpe (test, 95% CI)")
    ax.axhline(0, color="k", lw=0.5)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_ablation(abl: dict, path: str) -> None:
    names = list(abl)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([n.replace("hmm:", "") for n in names], [abl[n]["acc"][0] for n in names],
           yerr=[abl[n]["acc"][1] for n in names], capsize=4, color="C1")
    ax.set_ylabel("test accuracy")
    ax.set_xlabel("HMM feature set")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_kalman(rows: list[dict], path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    q = [r["q_over_r"] for r in rows]
    ax.semilogx(q, [r["gain_theory"] for r in rows], "k-", label="closed form")
    ax.semilogx(q, [r["gain_empirical"] for r in rows], "ro", label="filter (last step)")
    ax.set_xlabel("q / r")
    ax.set_ylabel("steady-state Kalman gain")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------- reporting ----------------
def _fmt(v, nd=3):
    return f"{v[0]:.{nd}f} ± {v[1]:.{nd}f}"


def table_headline(head: dict) -> str:
    lines = ["| Method | Accuracy | Delay (bars) | Net Sharpe | Max DD | Turnover | Fit (s) | Predict (us/bar) |",
             "|---|---|---|---|---|---|---|---|"]
    for n, m in head.items():
        acc = _fmt(m["acc"]) if "acc" in m else "-"
        dly = _fmt(m["delay"], 1) if "delay" in m else "-"
        fit = f"{m['fit_s'][0]:.2f}" if "fit_s" in m else "-"
        pr = f"{m['pred_us'][0]:.1f}" if "pred_us" in m else "-"
        lines.append(f"| {n} | {acc} | {dly} | {_fmt(m['sharpe'], 2)} | {m['max_dd'][0]:.3f} | {m['turnover'][0]:.3f} | {fit} | {pr} |")
    return "\n".join(lines)


def table_sweep(sw: dict, label: str) -> str:
    cols = list(next(iter(sw.values())))
    lines = [f"| {label} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, agg in sw.items():
        lines.append(f"| {k} | " + " | ".join(_fmt(agg[c]["acc"], 3) for c in cols) + " |")
    return "\n".join(lines)


def table_ablation(abl: dict) -> str:
    lines = ["| HMM features | Accuracy | Net Sharpe |", "|---|---|---|"]
    for n, m in abl.items():
        lines.append(f"| {n.replace('hmm:', '')} | {_fmt(m['acc'])} | {_fmt(m['sharpe'], 2)} |")
    return "\n".join(lines)


def run_all(cfg: ExpConfig, out_dir: str = "results", quick: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    seeds_sweep = 2 if quick else 4
    if quick:
        cfg = cfg.replace(seeds=min(cfg.seeds, 2))
    sep_vals = (0.5, 1.0) if quick else (0.25, 0.5, 1.0, 2.0)
    stay_vals = (0.9, 0.97) if quick else (0.85, 0.93, 0.97, 0.99)
    head = run_headline(cfg)
    sw_sep = run_sweep(cfg, "separation", sep_vals, seeds=seeds_sweep)
    sw_stay = run_sweep(cfg, "p_stay", stay_vals, seeds=seeds_sweep)
    abl = run_ablation(cfg, ABLATION_SETS[:3] if quick else ABLATION_SETS, seeds=seeds_sweep)
    kal = run_kalman_study()
    fig_timeline(cfg, os.path.join(out_dir, "timeline.png"))
    fig_sweeps(sw_sep, sw_stay, cfg, os.path.join(out_dir, "sweeps.png"))
    fig_sharpe(head, os.path.join(out_dir, "sharpe.png"))
    fig_ablation(abl, os.path.join(out_dir, "ablation.png"))
    fig_kalman(kal, os.path.join(out_dir, "kalman_gain.png"))
    summary = {"config": cfg.to_dict(), "headline": head, "sweep_separation": sw_sep,
               "sweep_p_stay": sw_stay, "ablation": abl, "kalman": kal}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    md = ["## Headline", table_headline(head), "", "## Sweep: separation", table_sweep(sw_sep, "separation"), "",
          "## Sweep: p_stay", table_sweep(sw_stay, "p_stay"), "", "## Ablation", table_ablation(abl), "",
          "## Kalman", "| q/r | gain empirical | gain theory | MSE kalman | MSE raw |", "|---|---|---|---|---|"]
    md += [f"| {r['q_over_r']:g} | {r['gain_empirical']:.4f} | {r['gain_theory']:.4f} | {r['mse_kalman']:.4f} | {r['mse_raw']:.4f} |" for r in kal]
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    return summary
