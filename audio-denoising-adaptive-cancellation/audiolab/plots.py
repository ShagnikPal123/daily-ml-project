import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .stft import power_spectrogram_db
from .signals import FS


def plot_denoise(res, path):
    fig, axes = plt.subplots(1, len(res["noises"]), figsize=(5 * len(res["noises"]), 3.8), sharey=True)
    for ax, nk in zip(np.atleast_1d(axes), res["noises"]):
        for m in res["methods"]:
            y = [res["table"][f"{nk}/{s}/{m}"]["si_sdr"] for s in res["snrs"]]
            ax.plot(res["snrs"], y, "o-", label=m)
        ax.set_title(f"{nk} noise"); ax.set_xlabel("input SNR (dB)"); ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("output SI-SDR (dB)"); np.atleast_1d(axes)[0].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_adaptive(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, c in res["curves"].items():
        axes[0].plot(np.arange(len(c["mis"])) * 50, c["mis"], label=name)
        axes[1].plot(np.arange(len(c["power"])) * 40, c["power"], label=name)
    axes[0].set_title("Weight misalignment"); axes[0].set_ylabel("dB"); axes[1].set_title("Residual noise power (smoothed)")
    for a in axes: a.set_xlabel("sample"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_sweeps(sw, tracking, path):
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.5))
    titles = {"mu_lms": "LMS step size", "mu_nlms": "NLMS step size", "L_nlms": "NLMS filter length", "lam_rls": "RLS forgetting"}
    for ax, (k, t) in zip(axes, titles.items()):
        xs = [float(x) for x in sw[k]]; ys = list(sw[k].values())
        ax.plot(xs, ys, "o-"); ax.set_title(t); ax.set_ylabel("SI-SDR (dB)"); ax.grid(alpha=.3)
        if k != "L_nlms" and k != "lam_rls": ax.set_xscale("log")
    for name, r in tracking.items():
        axes[4].plot(np.arange(len(r["power"])) * 40, r["power"], label=name)
    axes[4].axvline(8000, color="k", ls="--"); axes[4].set_title("Tracking path change"); axes[4].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_classification(res, path):
    noises = sorted({k.split("/")[0] for k in res["cells"]})
    fig, axes = plt.subplots(1, len(noises), figsize=(5.5 * len(noises), 3.8), sharey=True)
    for ax, nk in zip(np.atleast_1d(axes), noises):
        snrs = sorted({int(k.split("/")[1]) for k in res["cells"] if k.startswith(nk)})
        for den in ["none", "specsub", "wiener"]:
            ax.plot(snrs, [res["cells"][f"{nk}/{s}/{den}"]["train_clean"] for s in snrs], "o-", label=f"clean-train + {den}")
        ax.plot(snrs, [res["cells"][f"{nk}/{s}/none"]["train_multicond"] for s in snrs], "k--s", label="multi-condition train")
        ax.set_title(f"{nk} noise"); ax.set_xlabel("SNR (dB)"); ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("accuracy"); np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_spectrograms(d, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, k in zip(axes, ["clean", "noisy", "specsub", "wiener"]):
        S = power_spectrogram_db(d[k])
        ax.imshow(S.T, origin="lower", aspect="auto", vmin=S.max() - 70, extent=[0, len(d[k]) / FS, 0, FS / 2])
        ax.set_title(k); ax.set_xlabel("s")
    axes[0].set_ylabel("Hz"); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
