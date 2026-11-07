"""Command line: python -m audiolab.cli {denoise,adaptive,classify,all} [--quick] [--out results]"""
import argparse, json, os, time
from . import experiments as ex, plots


def main(argv=None):
    p = argparse.ArgumentParser(description="Audio denoising / adaptive cancellation / noisy classification lab")
    p.add_argument("command", choices=["denoise", "adaptive", "classify", "all"])
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="smaller sweeps (for CI)")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    metrics = {}
    t0 = time.time()
    if a.command in ("denoise", "all"):
        r = ex.exp_denoise(seed=a.seed, per_class=1 if a.quick else 3, snrs=(0, 10) if a.quick else (-5, 0, 5, 10, 15))
        metrics["denoise"] = r
        plots.plot_denoise(r, f"{a.out}/denoise_sweep.png")
        plots.plot_spectrograms(ex.demo_signals(a.seed), f"{a.out}/spectrograms.png")
    if a.command in ("denoise", "all"):
        metrics["vad"] = ex.exp_vad(seed=a.seed + 8, snrs=(5,) if a.quick else (0, 5, 10, 15), per_class=1 if a.quick else 3)
    if a.command in ("adaptive", "all"):
        r = ex.exp_adaptive(seed=a.seed + 1, n=6000 if a.quick else 16000)
        metrics["adaptive"] = {"final": r["final"]}
        plots.plot_adaptive(r, f"{a.out}/adaptive_convergence.png")
        sw = ex.exp_adaptive_sweeps(seed=a.seed + 2, n=4000 if a.quick else 12000)
        tr = ex.exp_tracking(seed=a.seed + 3, n=6000 if a.quick else 16000)
        metrics["adaptive_sweeps"] = sw
        metrics["tracking"] = {k: v["post_switch_si_sdr"] for k, v in tr.items()}
        plots.plot_sweeps(sw, tr, f"{a.out}/adaptive_sweeps.png")
        metrics["hum"] = ex.exp_hum(seed=a.seed + 4, n=6000 if a.quick else 16000)
    if a.command in ("classify", "all"):
        kw = dict(train_per_class=8, test_per_class=4, snrs=(0, 10), noises=("white",)) if a.quick else {}
        r = ex.exp_classify(seed=a.seed + 5, **kw)
        metrics["classify"] = r
        plots.plot_classification(r, f"{a.out}/classification.png")
        if not a.quick:
            metrics["model_comparison"] = ex.model_comparison(seed=a.seed + 6)
    metrics["runtime_s"] = round(time.time() - t0, 1)
    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k in ("hum", "tracking", "model_comparison", "runtime_s")}, indent=1))


if __name__ == "__main__":
    main()
