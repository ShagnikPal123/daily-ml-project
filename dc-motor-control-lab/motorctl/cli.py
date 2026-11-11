"""Layer 5 (apex): command-line entry point.

python -m motorctl.cli run --out results --seeds 20
python -m motorctl.cli validate
python -m motorctl.cli tune
python -m motorctl.cli simulate --kind lqg --sigma 1.0 --cpr 1000
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from motorctl.experiments import (compare, evaluate, run_all, run_kind, tune_pi,
                                  validation_table)
from motorctl.factory import KINDS
from motorctl.plant import MotorParams
from motorctl.simulate import SimConfig


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="motorctl", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="full pipeline: tune, compare, sweeps, ablations, figures")
    r.add_argument("--out", default="results")
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--quick", action="store_true", help="smaller grids/seeds for a smoke run")
    sub.add_parser("validate", help="compare kernels against SciPy / closed forms")
    sub.add_parser("tune", help="grid-search PI gains")
    s = sub.add_parser("simulate", help="one closed-loop run, print metrics")
    s.add_argument("--kind", choices=KINDS, default="lqg")
    s.add_argument("--sigma", type=float, default=0.5)
    s.add_argument("--cpr", type=int, default=4000)
    s.add_argument("--ts", type=float, default=1e-3)
    s.add_argument("--delay", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    p, cfg = MotorParams(), SimConfig()
    if args.cmd == "run":
        summ = run_all(args.out, seeds=args.seeds, quick=args.quick)
        print("PI gains:", summ["pi_gains"])
        for k, v in summ["compare"].items():
            print(f"{k:11s} IAE={v['iae'][0]:.3f}±{v['iae'][1]:.3f}")
    elif args.cmd == "validate":
        print(json.dumps(validation_table(), indent=1))
    elif args.cmd == "tune":
        g, _ = tune_pi(p, cfg)
        print(f"best kp={g.kp} ki={g.ki}")
    else:
        cfg = replace(cfg, meas_sigma=args.sigma, enc_cpr=args.cpr, Ts=args.ts,
                      delay=args.delay, seed=args.seed)
        g, _ = tune_pi(p, replace(cfg, meas_sigma=0.5), kps=(0.05, 0.1, 0.2), kis=(3, 10, 30))
        m, res = run_kind(args.kind, p, cfg, g)
        print(json.dumps({k: round(v, 4) for k, v in m.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
