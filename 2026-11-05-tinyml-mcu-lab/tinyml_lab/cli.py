"""Command line entry point: python -m tinyml_lab.cli {quant,prune,plan,mcu,all}."""
from __future__ import annotations

import argparse
import json
import os
import time

from . import experiments as E
from . import plots as P


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tinyml_lab", description=__doc__)
    ap.add_argument("command", choices=["quant", "prune", "plan", "mcu", "all"])
    ap.add_argument("--out", default="results", help="output directory for figures and metrics")
    ap.add_argument("--seeds", type=int, default=3, help="number of trained models for error bars")
    ap.add_argument("--epochs", type=int, default=60, help="float training epochs")
    ap.add_argument("--wbits", type=int, default=8, help="weight bits used in pruning/mcu studies")
    ap.add_argument("--ft-epochs", type=int, default=20, help="fine-tune epochs after pruning")
    return ap


def run(args) -> dict:
    t0 = time.time()
    out, metrics = args.out, {}
    cmd = args.command
    ctx = None
    if cmd != "plan":
        ctx = E.Context(seeds=tuple(range(args.seeds)), epochs=args.epochs)
        metrics["float_acc"] = E._ms(ctx.float_acc())
    if cmd in ("quant", "all"):
        bits = E.exp_weight_bits(ctx)
        act = E.exp_act_bits(ctx)
        fx = E.exp_fixed_point(ctx)
        metrics.update(weight_bits=bits, act_bits=act, fixed_point=fx)
        P.plot_bits(bits, act, f"{out}/fig_quantization.png")
        P.plot_fixed(fx, f"{out}/fig_fixed_point.png")
    if cmd in ("prune", "all"):
        pr = E.exp_pruning(ctx, wbits=args.wbits, ft_epochs=args.ft_epochs)
        metrics["pruning"] = pr
        P.plot_pareto(pr, f"{out}/fig_pruning_pareto.png")
    if cmd in ("plan", "all"):
        mp = E.exp_memplan()
        metrics["memplan"] = mp
        P.plot_memplan(mp, f"{out}/fig_memplan.png")
    if cmd in ("mcu", "all"):
        mc = E.exp_mcu(ctx)
        metrics["mcu"] = mc
        P.plot_mcu(mc, f"{out}/fig_mcu.png")
    metrics["runtime_s"] = round(time.time() - t0, 1)
    _dump(metrics, f"{out}/metrics_{cmd}.json" if cmd != "all" else f"{out}/metrics.json")
    return metrics


def main(argv=None):
    args = build_parser().parse_args(argv)
    m = run(args)
    print(f"done in {m['runtime_s']}s -> {args.out}/")


if __name__ == "__main__":
    main()
