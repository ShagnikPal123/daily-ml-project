"""Layer 6 (apex): command-line entry point that produces the headline results and figures."""
from __future__ import annotations

import argparse
import json

from .config import ExpConfig
from .experiments import run_all, run_headline, table_headline, run_sweep, table_sweep


def _cfg(args) -> ExpConfig:
    cfg = ExpConfig.load(args.config) if args.config else ExpConfig()
    over = {k: getattr(args, k) for k in ("k", "seeds", "n_train", "n_test", "separation", "p_stay", "cost_bps")
            if getattr(args, k, None) is not None}
    return cfg.replace(**over) if over else cfg.validate()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="regimelab", description="Regime detection lab (synthetic data only)")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "demo", "sweep"):
        s = sub.add_parser(name)
        s.add_argument("--config", help="JSON config file")
        s.add_argument("--k", type=int)
        s.add_argument("--seeds", type=int)
        s.add_argument("--n-train", dest="n_train", type=int)
        s.add_argument("--n-test", dest="n_test", type=int)
        s.add_argument("--separation", type=float)
        s.add_argument("--p-stay", dest="p_stay", type=float)
        s.add_argument("--cost-bps", dest="cost_bps", type=float)
        if name == "run":
            s.add_argument("--out", default="results")
            s.add_argument("--quick", action="store_true", help="fewer seeds/sweep points")
        if name == "sweep":
            s.add_argument("--param", required=True, choices=["separation", "p_stay", "ar_strength", "cost_bps"])
            s.add_argument("--values", type=float, nargs="+", required=True)
    sub.add_parser("show-config")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "show-config":
        print(json.dumps(ExpConfig().to_dict(), indent=2))
        return 0
    cfg = _cfg(args)
    if args.cmd == "demo":
        print(table_headline(run_headline(cfg.replace(seeds=min(cfg.seeds, 2)), ("threshold", "hmm-ret"))))
    elif args.cmd == "sweep":
        print(table_sweep(run_sweep(cfg, args.param, args.values, seeds=min(cfg.seeds, 3)), args.param))
    else:
        run_all(cfg, args.out, args.quick)
        print(f"results written to {args.out}/")
    return 0
