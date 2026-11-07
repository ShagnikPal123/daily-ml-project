"""Command line entry point:  python -m codinglab.cli <experiment> [options]

Experiments
  compare   uncoded vs conv(hard/soft Viterbi) vs LDPC (SPA, MS, NMS) over AWGN
  iters     LDPC convergence: BER vs max iterations, and average iterations used
  quant     fixed-point min-sum: BER vs bit-width plus decoder memory cost
  tune      alpha grid search and per-iteration learned (unfolded) normalisation
  fading    same codes on a fast Rayleigh-fading channel
  all       run everything and write figures + results/metrics.json
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import channel, quantized, tuning
from .ldpc import LDPCCode
from .simulate import Uncoded, ConvScheme, LDPCScheme, ber_curve, points_to_dict


def _plot_curves(curves, title, path, ylim=(1e-6, 1), theory=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = "osd^v*xP"
    for i, (name, pts) in enumerate(curves.items()):
        x = [p.ebn0_db for p in pts]
        y = [max(p.ber, 1e-9) for p in pts]
        ax.semilogy(x, y, marker=markers[i % len(markers)], label=name, ms=5)
    if theory is not None:
        ax.semilogy(*theory[:2], "k--", lw=1, label=theory[2])
    ax.set_ylim(*ylim); ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel("Eb/N0 (dB)"); ax.set_ylabel("Bit error rate"); ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def exp_compare(out, args, metrics):
    code = LDPCCode(n=args.n, seed=args.seed)
    ebn0 = np.arange(0.0, 4.6, 0.5)
    schemes = [Uncoded(), ConvScheme(soft=False), ConvScheme(soft=True),
               LDPCScheme(code, "spa", 30, label="LDPC SPA (30 it)"),
               LDPCScheme(code, "ms", 30, label="LDPC min-sum (30 it)"),
               LDPCScheme(code, "nms", 30, alpha=0.8, label="LDPC NMS a=0.8 (30 it)")]
    curves = {s.name: ber_curve(s, ebn0, max_frames=args.max_frames, seed=args.seed) for s in schemes}
    xs = np.linspace(0, 4.5, 50)
    _plot_curves(curves, f"AWGN: n~{code.n} codes, rate ~1/2", os.path.join(out, "compare_ber.png"),
                 theory=(xs, channel.uncoded_ber_theory(xs), "BPSK theory"))
    metrics["compare"] = {k: points_to_dict(v) for k, v in curves.items()}
    metrics["ldpc"] = dict(n=code.n, k=code.k, rate=code.rate, four_cycle_free=code.girth_four_free())
    metrics["shannon_limit_ebn0_db_rate_half"] = channel.shannon_limit_ebn0_db(0.5)


def exp_iters(out, args, metrics):
    code = LDPCCode(n=args.n, seed=args.seed)
    its = [1, 2, 3, 5, 8, 12, 20, 30, 50]
    eb = args.iters_ebn0
    res = {}
    for method, alpha in [("spa", 1), ("ms", 1), ("nms", 0.8)]:
        bers, avg = [], []
        for it in its:
            p = ber_curve(LDPCScheme(code, method, it, alpha=alpha), [eb], min_frame_errors=40,
                          max_frames=args.max_frames, seed=args.seed)[0]
            bers.append(p.ber); avg.append(p.avg_iters)
        res[method] = dict(iters=its, ber=bers, avg_iters=avg)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, v in res.items():
        ax[0].semilogy(its, np.maximum(v["ber"], 1e-9), "o-", label=k.upper())
        ax[1].plot(its, v["avg_iters"], "o-", label=k.upper())
    ax[0].set(xlabel="max iterations", ylabel="BER", title=f"BER vs iteration cap (Eb/N0={eb} dB)")
    ax[1].set(xlabel="max iterations", ylabel="mean iterations used", title="Early-stopping saves work")
    for a in ax: a.grid(True, which="both", alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "iterations.png"), dpi=140); plt.close(fig)
    metrics["iterations"] = dict(ebn0_db=eb, **res)


def exp_quant(out, args, metrics):
    code = LDPCCode(n=args.n, seed=args.seed)
    eb = args.quant_ebn0
    bits_list = [3, 4, 5, 6, 8]
    ref = ber_curve(LDPCScheme(code, "nms", 20, alpha=0.8), [eb], min_frame_errors=60,
                    max_frames=args.max_frames, seed=args.seed)[0].ber
    rows = []
    for b in bits_list:
        q = quantized.FixedPoint(b, max_llr=args.max_llr)
        p = ber_curve(LDPCScheme(code, "nms", 20, alpha=0.8, qfn=q), [eb], min_frame_errors=60,
                      max_frames=args.max_frames, seed=args.seed)[0]
        rows.append(dict(bits=b, ber=p.ber, step=q.step,
                         mem_naive_bits=quantized.message_memory_bits(code, b),
                         mem_compressed_bits=quantized.compressed_check_memory_bits(code, b)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].semilogy([r["bits"] for r in rows], np.maximum([r["ber"] for r in rows], 1e-9), "o-", label="fixed-point NMS")
    ax[0].axhline(max(ref, 1e-9), color="k", ls="--", label="floating point")
    ax[0].set(xlabel="message bit-width", ylabel="BER", title=f"Quantisation loss (Eb/N0={eb} dB)")
    ax[1].plot([r["bits"] for r in rows], [r["mem_naive_bits"] / 8192 for r in rows], "o-", label="per-edge messages")
    ax[1].plot([r["bits"] for r in rows], [r["mem_compressed_bits"] / 8192 for r in rows], "s-", label="min1/min2 compressed")
    ax[1].set(xlabel="message bit-width", ylabel="decoder memory (KiB)", title="Memory cost")
    for a in ax: a.grid(True, which="both", alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "quantization.png"), dpi=140); plt.close(fig)
    metrics["quantization"] = dict(ebn0_db=eb, float_ber=ref, rows=rows)


def exp_tune(out, args, metrics):
    code = LDPCCode(n=args.n, seed=args.seed)
    eb = args.tune_ebn0
    best, grid, scores = tuning.tune_alpha(code, eb, iters=15)
    alphas, hist = tuning.learn_alpha_per_iter(code, eb, iters=10)
    pts = {}
    for name, sch in {"MS (alpha=1)": LDPCScheme(code, "ms", 10),
                      f"NMS alpha={best:.2f}": LDPCScheme(code, "nms", 10, alpha=best),
                      "NMS learned per-iter": LDPCScheme(code, "nms", 10, alpha=alphas)}.items():
        pts[name] = ber_curve(sch, np.arange(1.0, 3.6, 0.5), max_frames=args.max_frames, seed=args.seed + 7)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(grid, scores, "o-"); ax[0].axvline(best, color="r", ls="--")
    ax[0].set(xlabel="alpha", ylabel="training BER", title=f"Alpha grid search @ {eb} dB")
    ax[1].plot(range(1, len(alphas) + 1), alphas, "o-"); ax[1].set(xlabel="iteration", ylabel="alpha_t", title="Learned per-iteration alpha")
    for n_, p in pts.items():
        ax[2].semilogy([q.ebn0_db for q in p], np.maximum([q.ber for q in p], 1e-9), "o-", label=n_)
    ax[2].set(xlabel="Eb/N0 (dB)", ylabel="BER", title="10-iteration decoders"); ax[2].legend(fontsize=8)
    for a in ax: a.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "tuning.png"), dpi=140); plt.close(fig)
    metrics["tuning"] = dict(best_alpha=best, grid=grid, train_ber=scores, learned_alphas=list(map(float, alphas)),
                             surrogate_loss_history=hist, curves={k: points_to_dict(v) for k, v in pts.items()})


def exp_fading(out, args, metrics):
    code = LDPCCode(n=args.n, seed=args.seed)
    ebn0 = np.arange(0.0, 12.1, 2.0)
    schemes = [Uncoded(), ConvScheme(soft=True), LDPCScheme(code, "nms", 30, alpha=0.8, label="LDPC NMS (30 it)")]
    curves = {s.name: ber_curve(s, ebn0, max_frames=args.max_frames, seed=args.seed, fading=True) for s in schemes}
    xs = np.linspace(0, 12, 50)
    _plot_curves(curves, "Fast Rayleigh fading with perfect CSI", os.path.join(out, "fading_ber.png"),
                 ylim=(1e-5, 1), theory=(xs, channel.uncoded_ber_rayleigh_theory(xs), "BPSK Rayleigh theory"))
    metrics["fading"] = {k: points_to_dict(v) for k, v in curves.items()}


EXPS = dict(compare=exp_compare, iters=exp_iters, quant=exp_quant, tune=exp_tune, fading=exp_fading)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=list(EXPS) + ["all"])
    p.add_argument("--out", default="results")
    p.add_argument("--n", type=int, default=504, help="LDPC block length (multiple of 6)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--iters-ebn0", type=float, default=2.0)
    p.add_argument("--quant-ebn0", type=float, default=2.5)
    p.add_argument("--tune-ebn0", type=float, default=2.0)
    p.add_argument("--max-llr", type=float, default=12.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    metrics, names = {}, (list(EXPS) if args.experiment == "all" else [args.experiment])
    for nme in names:
        t = time.time()
        EXPS[nme](args.out, args, metrics)
        print(f"[{nme}] done in {time.time() - t:.1f}s")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    return metrics


if __name__ == "__main__":
    main()
