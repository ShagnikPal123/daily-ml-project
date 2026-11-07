"""Command line: python -m sysarray {verify,sweep,mlp,all} [options]."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import plots
from .sweep import gemm_sweep, mlp_experiment, save_csv, save_json, sram_sweep
from .verify import verify


def _sizes(s: str) -> tuple[int, ...]:
    return tuple(int(v) for v in s.split(","))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sysarray", description=__doc__)
    p.add_argument("command", choices=["verify", "sweep", "mlp", "all"])
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    p.add_argument("--sizes", type=_sizes, default=(2, 4, 8, 16, 32), help="array sides, e.g. 4,8,16")
    p.add_argument("--sram-kb", type=int, default=32, help="scratchpad size for DRAM model")
    p.add_argument("--mlp-sizes", type=_sizes, default=(4, 8, 16), help="array sides simulated on the MLP")
    p.add_argument("--n-eval", type=int, default=96, help="test samples pushed through the simulator")
    p.add_argument("--hidden", type=int, default=64, help="MLP hidden width")
    p.add_argument("--cases", type=int, default=24, help="random cases for `verify`")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.command in ("verify", "all"):
        rec = verify(a.cases, a.seed)
        save_json(rec, a.out / "verify.json")
        print("verify:", rec)
        if not rec["all_passed"]:
            return 1
        plots.fig_wavefront(a.out / "wavefront.png", seed=a.seed)
    if a.command in ("sweep", "all"):
        rows = gemm_sweep(a.sizes, a.sram_kb)
        save_csv(rows, a.out / "gemm_sweep.csv")
        srows = sram_sweep()
        save_csv(srows, a.out / "sram_sweep.csv")
        plots.fig_utilization(rows, a.out / "utilization.png")
        plots.fig_energy(rows, a.out / "energy_breakdown.png", array=16 if 16 in a.sizes else a.sizes[-1])
        plots.fig_sram(srows, a.out / "sram_sweep.png")
        print(f"sweep: {len(rows)} rows")
    if a.command in ("mlp", "all"):
        res = mlp_experiment(a.mlp_sizes, n_eval=a.n_eval, hidden=a.hidden, seed=a.seed)
        save_json(res, a.out / "mlp_results.json")
        plots.fig_sparsity(res, a.out / "sparsity.png")
        print(f"mlp: float={res['float_acc']:.3f} int8={res['int8_acc']:.3f} bit_exact={res['bit_exact']}")
        if not res["bit_exact"]:
            return 1
    print(f"done in {time.time() - t0:.1f}s")
    return 0
