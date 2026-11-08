"""Layer 5 (apex): command-line interface producing the headline results.

    python -m lowlight run --config configs/default.json --out results --workers 2
    python -m lowlight run --quick
    python -m lowlight report --out results
    python -m lowlight budget
    python -m lowlight ptc --photons 2000
    python -m lowlight demo --photons 5 --digit 3
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import data, experiments, ptc, report
from .pipeline import Pipeline, PipelineConfig
from .sensor import SensorConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ascii(img: np.ndarray) -> list[str]:
    ramp = " .:-=+*#%@"
    return ["".join(ramp[int(v * 9.999)] * 2 for v in row) for row in np.clip(img, 0, 1)]


def cmd_run(a) -> int:
    cfg_path = os.path.join(HERE, "configs", "quick.json") if a.quick else a.config
    cfg = experiments.load_config(cfg_path)
    res = experiments.run_all(cfg, a.out, a.workers)
    scenes, _ = data.load_scenes(cfg["image_size"])
    figs = report.render_figures(res, os.path.join(a.out, "figures"), scenes)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write(report.render_tables(res))
    print(f"{res['n_trials']} trials in {res['runtime_sec']:.0f}s; {len(figs)} figures -> {a.out}")
    return 0


def cmd_report(a) -> int:
    with open(os.path.join(a.out, "results.json")) as f:
        res = json.load(f)
    scenes, _ = data.load_scenes(res["config"]["image_size"])
    figs = report.render_figures(res, os.path.join(a.out, "figures"), scenes)
    with open(os.path.join(a.out, "tables.md"), "w") as f:
        f.write(report.render_tables(res))
    print(f"re-rendered {len(figs)} figures and tables.md")
    return 0


def cmd_budget(a) -> int:
    cfg = experiments.load_config(a.config)
    scenes, labels = data.load_scenes(cfg["image_size"])
    print(f"{'denoiser':10s} {'MACs':>8s} {'linebuf B':>10s} {'weights B':>10s} {'nJ int8':>8s} {'nJ fp32':>8s}")
    for r in experiments.deployment_table(cfg, scenes, labels):
        print(f"{r['denoiser']:10s} {r['total_macs']:8.0f} {r['line_buffer_bytes']:10d} {r['weight_bytes']:10d} "
              f"{r['energy_uj'] * 1e3:8.1f} {r['energy_uj_fp32'] * 1e3:8.1f}")
    return 0


def cmd_ptc(a) -> int:
    """Characterise the simulated sensor from flat fields only and compare with ground truth."""
    cfg = SensorConfig(photons=a.photons, adc_bits=a.bits, read_noise=a.read_noise)
    curve = ptc.photon_transfer(cfg, np.linspace(0.05, 1.0, 12))
    fit = ptc.fit_ptc(curve)
    out = {"estimated": fit, "true_gain_dn_per_e": ptc.true_gain(cfg), "true_noise_floor_e": ptc.true_noise_floor_e(cfg)}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ptc.json"), "w") as f:
        json.dump(out, f, indent=1)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    report.fig_ptc(curve, fit, os.path.join(a.out, "figures", "photon_transfer.png"))
    print(f"gain {fit['gain_dn_per_e']:.4f} DN/e- (true {out['true_gain_dn_per_e']:.4f}); "
          f"noise floor {fit['dark_noise_e']:.2f} e- (true {out['true_noise_floor_e']:.2f}); R2={fit['r2']:.4f}")
    return 0


def cmd_demo(a) -> int:
    x, y = data.load_scenes(16)
    sp = data.make_splits(y, 0)
    cfg = PipelineConfig(sensor=SensorConfig(photons=a.photons), denoiser=a.denoiser)
    pipe = Pipeline(cfg, (16, 16), 0).fit(x[sp["train"]], y[sp["train"]], np.random.default_rng(1))
    i = int(sp["test"][np.flatnonzero(y[sp["test"]] == a.digit)[0]])
    est = pipe.acquire(x[i:i + 1], np.random.default_rng(2))
    out = pipe.isp(est)
    pred = int(pipe.clf.predict(out.reshape(1, -1))[0])
    for c, n, o in zip(_ascii(x[i]), _ascii(np.clip(est[0], 0, 1)), _ascii(out[0])):
        print(f"{c}  |  {n}  |  {o}")
    print(f"{'clean':32s}{'raw capture':32s}{a.denoiser}")
    print(f"photons={a.photons:g} true={y[i]} predicted={pred}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lowlight", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    default_cfg = os.path.join(HERE, "configs", "default.json")
    r = sub.add_parser("run", help="run all experiments, write results/figures/tables")
    r.add_argument("--config", default=default_cfg)
    r.add_argument("--out", default="results")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--quick", action="store_true", help="tiny config (smoke test)")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("report", help="re-render figures/tables from results.json")
    q.add_argument("--out", default="results")
    q.set_defaults(fn=cmd_report)
    b = sub.add_parser("budget", help="print per-frame MAC/SRAM/energy table")
    b.add_argument("--config", default=default_cfg)
    b.set_defaults(fn=cmd_budget)
    t = sub.add_parser("ptc", help="photon-transfer-curve characterisation of the simulated sensor")
    t.add_argument("--photons", type=float, default=2000.0)
    t.add_argument("--bits", type=int, default=12)
    t.add_argument("--read-noise", type=float, default=3.0)
    t.add_argument("--out", default="results")
    t.set_defaults(fn=cmd_ptc)
    d = sub.add_parser("demo", help="ASCII capture of one digit through the whole pipeline")
    d.add_argument("--photons", type=float, default=5.0)
    d.add_argument("--digit", type=int, default=3)
    d.add_argument("--denoiser", default="ridge")
    d.set_defaults(fn=cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
