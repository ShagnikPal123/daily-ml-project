"""Layer 4: figures (matplotlib) and Markdown tables rendered from a results dict."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .denoisers import make_denoiser  # noqa: E402
from .sensor import Sensor, SensorConfig  # noqa: E402
from .stats import holm_reject, p_from_ci  # noqa: E402

PALETTE = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#9d755d", "#bab0ac"]


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pm(mu, ci, digits=3, pct=False) -> str:
    f = 100 if pct else 1
    return f"{mu * f:.{digits}f} ± {0.0 if ci != ci else ci * f:.{digits}f}"


def _series(rows, xkey, ykey, name_key="denoiser", where=None):
    out: dict = {}
    for r in rows:
        if where and not where(r):
            continue
        out.setdefault(r[name_key], []).append(r)
    return {k: sorted(v, key=lambda r: r[xkey]) for k, v in out.items()}


def _line(ax, rows, xkey, ykey, name_key="denoiser", where=None, logx=True, **kw):
    for i, (name, rs) in enumerate(_series(rows, xkey, ykey, name_key, where).items()):
        ax.errorbar([r[xkey] for r in rs], [r[ykey] for r in rs], yerr=[np.nan_to_num(r[ykey + "_ci"]) for r in rs],
                    label=str(name), color=PALETTE[i % len(PALETTE)], marker="o", ms=3.5, lw=1.4, capsize=2, **kw)
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.3)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_metric_vs_exposure(res, path, metric="acc"):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _line(ax, res["headline"], "photons", metric)
    ax.set_xlabel("peak signal (photo-electrons at full radiance)")
    ax.set_ylabel({"acc": "test accuracy", "psnr": "PSNR vs clean scene (dB)", "ssim": "SSIM"}[metric])
    ax.set_title(f"{metric.upper()} vs exposure (mean ± 95% CI over seeds)")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, path)


def fig_sweep(res, key, xkey, path, xlabel):
    rows = res[key]
    phs = sorted({r["photons"] for r in rows})
    fig, axes = plt.subplots(1, len(phs), figsize=(5.2 * len(phs), 3.8), squeeze=False)
    for ax, p in zip(axes[0], phs):
        _line(ax, rows, xkey, "acc", where=lambda r, p=p: r["photons"] == p, logx=False)
        ax.set_title(f"{int(p)} e- peak signal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test accuracy")
        ax.legend(fontsize=8)
    return _save(fig, path)


def fig_train_mode(res, path):
    rows = res["train_mode"]
    phs = sorted({r["photons"] for r in rows})
    dens = sorted({r["denoiser"] for r in rows})
    fig, axes = plt.subplots(1, len(phs), figsize=(4.2 * len(phs), 3.6), squeeze=False, sharey=True)
    for ax, p in zip(axes[0], phs):
        for j, mode in enumerate(("clean", "matched")):
            vals = [next((r for r in rows if r["photons"] == p and r["denoiser"] == d and r["train_mode"] == mode), None) for d in dens]
            ax.bar(np.arange(len(dens)) + 0.4 * j - 0.2, [v["acc"] if v else 0 for v in vals], 0.38,
                   yerr=[np.nan_to_num(v["acc_ci"]) if v else 0 for v in vals], label=f"trained on {mode}", color=PALETTE[j], capsize=2)
        ax.set_xticks(range(len(dens)), dens, rotation=30, fontsize=8)
        ax.set_title(f"{int(p)} e-")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("test accuracy")
    return _save(fig, path)


def _dep_level(res) -> float:
    """Headline exposure closest to the deployment exposure."""
    levels = sorted({r["photons"] for r in res["headline"]})
    return min(levels, key=lambda v: abs(np.log(v / res["config"]["deployment_photons"])))


def fig_pareto(res, path):
    p = _dep_level(res)
    acc = {r["denoiser"]: r for r in res["headline"] if r["photons"] == p}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, r in enumerate(res["deployment"]):
        a = acc[r["denoiser"]]
        ax.errorbar(r["energy_uj"] * 1e3, a["acc"], yerr=np.nan_to_num(a["acc_ci"]), fmt="o", color=PALETTE[i % 8], capsize=2)
        ax.annotate(r["denoiser"], (r["energy_uj"] * 1e3, a["acc"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("estimated ISP+classifier energy per frame (nJ, int8 assumption)")
    ax.set_ylabel(f"accuracy @ {int(p)} e-")
    ax.set_title("Accuracy vs compute energy")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def fig_failure(res, path):
    f = res["failure"]
    best = f["best_method"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.38
    axes[0].bar(np.arange(10) - w / 2, f["per_class_identity"], w, label="identity", color=PALETTE[0])
    axes[0].bar(np.arange(10) + w / 2, f[f"per_class_{best}"], w, label=best, color=PALETTE[1])
    axes[0].set_xticks(range(10))
    axes[0].set_xlabel("digit class")
    axes[0].set_ylabel("per-class accuracy")
    axes[0].set_title(f"Per-class accuracy @ {int(f['photons'])} e-")
    axes[0].legend()
    conf = np.array(f["confusion_best"], float)
    im = axes[1].imshow(conf / conf.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title(f"Confusion ({best}, rows = truth)")
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("true")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    return _save(fig, path)


def fig_samples(scenes, cfg, path, photons=(2, 5, 20, 100), methods=("identity", "gaussian", "gat", "ridge"), idx=(3, 17, 42)):
    """Montage: clean scene, then for each exposure the raw capture and denoised outputs."""
    x = scenes[list(idx)]
    cols = 1 + len(methods)
    fig, axes = plt.subplots(len(photons) * len(idx) // len(idx), cols * len(idx), figsize=(1.45 * cols * len(idx), 1.5 * len(photons)), squeeze=False)
    tr = scenes[200:700]
    for row, p in enumerate(photons):
        sensor = Sensor(SensorConfig.from_dict({**cfg["sensor"], "photons": float(p)}), scenes.shape[1:], seed=0)
        noisy_tr = sensor.acquire(tr, np.random.default_rng(1))
        est = sensor.acquire(x, np.random.default_rng(2))
        outs = {m: np.clip(make_denoiser(m, sensor).fit(tr, noisy_tr)(est), 0, 1) for m in methods}
        for k in range(len(idx)):
            panels = [("clean", x[k])] + [(m, outs[m][k]) for m in methods]
            for c, (t, im) in enumerate(panels):
                ax = axes[row][k * cols + c]
                ax.imshow(im, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(t, fontsize=7)
                if c == 0 and k == 0:
                    ax.set_ylabel(f"{p} e-", fontsize=8)
    return _save(fig, path)


def fig_ptc(curve, fit, path):
    """Photon-transfer curve with fitted line (slope = system gain)."""
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.plot(curve["mean_dn"], curve["var_dn"], "o", color=PALETTE[0], label="measured")
    xs = np.linspace(0, curve["mean_dn"].max(), 50)
    ax.plot(xs, fit["gain_dn_per_e"] * xs + fit["intercept"], color=PALETTE[3], label=f"fit: K={fit['gain_dn_per_e']:.3f} DN/e-, R²={fit['r2']:.4f}")
    ax.set_xlabel("mean signal (DN)")
    ax.set_ylabel("temporal variance (DN²)")
    ax.set_title("Photon transfer curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, path)


def render_figures(res, out_dir, scenes=None) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    j = lambda n: os.path.join(out_dir, n)
    paths = [fig_metric_vs_exposure(res, j("accuracy_vs_exposure.png"), "acc"),
             fig_metric_vs_exposure(res, j("psnr_vs_exposure.png"), "psnr"),
             fig_sweep(res, "bits", "adc_bits", j("adc_bits_sweep.png"), "ADC bits"),
             fig_sweep(res, "readnoise", "read_noise", j("read_noise_sweep.png"), "read noise (e- rms)"),
             fig_train_mode(res, j("train_mode_ablation.png")),
             fig_pareto(res, j("accuracy_vs_energy.png")),
             fig_failure(res, j("failure_analysis.png"))]
    if scenes is not None:
        paths.append(fig_samples(scenes, res["config"], j("sample_montage.png")))
    return paths


def render_tables(res) -> str:
    cfg = res["config"]
    s = []
    phs = sorted({r["photons"] for r in res["headline"]})
    dens = list(cfg["denoisers"])
    H = {(r["photons"], r["denoiser"]): r for r in res["headline"]}
    s.append("### TABLE headline_acc\n" + md_table(["e- (peak)"] + dens, [[f"{p:g}"] + [pm(H[(p, d)]["acc"], H[(p, d)]["acc_ci"], 1, True) for d in dens] for p in phs]))
    s.append("### TABLE headline_psnr\n" + md_table(["e- (peak)"] + dens, [[f"{p:g}"] + [pm(H[(p, d)]["psnr"], H[(p, d)]["psnr_ci"], 2) for d in dens] for p in phs]))
    pa = res["paired_acc"]
    holm = holm_reject([p_from_ci(r["diff"], r["ci"], r["n"]) for r in pa])
    P = {(r["photons"], r["denoiser"]): dict(r, holm=h) for r, h in zip(pa, holm)}
    nd = [d for d in dens if d != "identity"]
    s.append("### TABLE paired_acc\n" + md_table(["e- (peak)"] + nd, [[f"{p:g}"] + [
        (pm(P[(p, d)]["diff"], P[(p, d)]["ci"], 1, True) + ("†" if P[(p, d)]["holm"] else ("*" if P[(p, d)]["significant"] else ""))) for d in nd] for p in phs]))
    s.append("### TABLE train_mode\n" + md_table(["train mode", "e-", "denoiser", "accuracy %"], [[r["train_mode"], f"{r['photons']:g}", r["denoiser"], pm(r["acc"], r["acc_ci"], 1, True)] for r in res["train_mode"]]))
    s.append("### TABLE gat\n" + md_table(["variant", "e-", "acc %", "PSNR dB"], [[r["variant"], f"{r['photons']:g}", pm(r["acc"], r["acc_ci"], 1, True), pm(r["psnr"], r["psnr_ci"], 2)] for r in res["gat_ablation"]]))
    s.append("### TABLE patch\n" + md_table(["e-", "patch", "acc %", "PSNR dB"], [[f"{r['photons']:g}", f"{r['patch']}x{r['patch']}", pm(r["acc"], r["acc_ci"], 1, True), pm(r["psnr"], r["psnr_ci"], 2)] for r in res["patch"]]))
    s.append("### TABLE bits\n" + md_table(["e-", "denoiser", "ADC bits", "acc %"], [[f"{r['photons']:g}", r["denoiser"], r["adc_bits"], pm(r["acc"], r["acc_ci"], 1, True)] for r in res["bits"]]))
    s.append("### TABLE readnoise\n" + md_table(["e-", "denoiser", "read noise e-", "acc %"], [[f"{r['photons']:g}", r["denoiser"], f"{r['read_noise']:g}", pm(r["acc"], r["acc_ci"], 1, True)] for r in res["readnoise"]]))
    s.append("### TABLE classifier\n" + md_table(["e-", "classifier", "acc %"], [[f"{r['photons']:g}", r["classifier"], pm(r["acc"], r["acc_ci"], 1, True)] for r in res["classifier"]]))
    dep = _dep_level(res)
    acc = {r["denoiser"]: r for r in res["headline"] if r["photons"] == dep}
    prof = {r["denoiser"]: r for r in res["profile"]}
    s.append("### TABLE deployment\n" + md_table(["denoiser", "MACs/frame", "line buffer B", "weights B (int8)", "energy nJ (int8)", "energy nJ (fp32)", f"acc % @{dep:g} e-", "host µs/img"],
             [[r["denoiser"], f"{r['total_macs']:.0f}", r["line_buffer_bytes"], r["weight_bytes"], f"{r['energy_uj'] * 1e3:.1f}", f"{r['energy_uj_fp32'] * 1e3:.1f}",
               pm(acc[r["denoiser"]]["acc"], acc[r["denoiser"]]["acc_ci"], 1, True), f"{prof[r['denoiser']]['us_per_image']:.0f}"] for r in res["deployment"]]))
    f = res["failure"]
    s.append("### TABLE failure\n" + md_table(["quantity", "value"], [
        ["exposure analysed", f"{f['photons']:g} e-"], ["best method", f["best_method"]],
        ["top confusions (true→pred, count)", "; ".join(f"{a}→{b} ({c})" for a, b, c in f["top_confusions"])],
        ["test images wrong for every method", pm(*f["all_methods_wrong_frac"], 3, True) + " %"],
        ["significantly harmful (e-, method, Δacc)", "; ".join(f"{p:g} e-, {d}, {x * 100:.1f} pp" for p, d, x in f["harmful"]) or "none"],
        ["PSNR-winner ≠ accuracy-winner at", ", ".join(f"{d['photons']:g} e-" for d in f["metric_disagreement"] if not d["agree"]) or "no exposure"]]))
    return "\n\n".join(s) + "\n"
