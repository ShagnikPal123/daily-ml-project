"""Experiments: each returns plain dicts (JSON-serialisable) consumed by plots and the README."""
from __future__ import annotations

import numpy as np

from .data import Dataset, load_dataset
from .mcu import MCUS, evaluate, fits_sram, mlp_deployment
from .memplan import (PLANNERS, lower_bound, liveness, mlp_graph, tiny_resnet_graph,
                      unet_graph, validate, random_graph)
from .model import MLP, fit
from .prune import prune_and_finetune, sparsity_of, structured_prune
from .quant import build_qmlp, fixed_forward, sqnr_db

SIZES = [64, 48, 32, 10]


class Context:
    """Trained float models (several seeds, for error bars) plus the dataset."""

    def __init__(self, seeds=(0, 1, 2), epochs: int = 60, sizes=SIZES):
        self.data: Dataset = load_dataset(seed=0)
        self.sizes = list(sizes)
        self.models = []
        for s in seeds:
            m = MLP(self.sizes, seed=s)
            fit(m, self.data.X_train, self.data.y_train, epochs=epochs, seed=s)
            self.models.append(m)
        self.Xcal = self.data.calibration()

    def float_acc(self) -> list:
        return [m.accuracy(self.data.X_test, self.data.y_test) for m in self.models]


def _ms(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def exp_weight_bits(ctx: Context, bits=(2, 3, 4, 5, 6, 8)) -> dict:
    """Test accuracy vs weight bit-width, per-tensor vs per-channel scales (int8 activations)."""
    d = ctx.data
    out = {"float": _ms(ctx.float_acc()), "bits": list(bits)}
    for pc in (False, True):
        key = "per_channel" if pc else "per_tensor"
        out[key] = []
        out[key + "_sqnr_db"] = []
        for b in bits:
            accs, sq = [], []
            for m in ctx.models:
                q = build_qmlp(m, ctx.Xcal, wbits=b, abits=8, per_channel=pc)
                accs.append(q.accuracy(d.X_test, d.y_test))
                lq = q.int_forward(d.X_test) * q.s_out
                sq.append(sqnr_db(m.forward(d.X_test), lq))
            out[key].append(_ms(accs))
            out[key + "_sqnr_db"].append(float(np.mean(sq)))
    return out


def exp_act_bits(ctx: Context, abits=(3, 4, 5, 6, 8, 10), pcts=(99.0, 99.9, 100.0)) -> dict:
    """Activation bit-width x calibration percentile (clip outliers vs resolution trade)."""
    d = ctx.data
    out = {"abits": list(abits), "pcts": list(pcts), "acc": {}}
    for p in pcts:
        row = []
        for a in abits:
            accs = [build_qmlp(m, ctx.Xcal, 8, a, True, p).accuracy(d.X_test, d.y_test) for m in ctx.models]
            row.append(float(np.mean(accs)))
        out["acc"][str(p)] = row
    m = ctx.models[0]
    q = build_qmlp(m, ctx.Xcal, 8, 8, True)
    out["accumulator_bits_int8"] = q.max_accumulator_bits(d.X_test)
    return out


def exp_fixed_point(ctx: Context, w_fracs=(2, 3, 4, 5, 6, 7), a_fracs=(2, 4, 6, 8, 10, 12, 14),
                    w_bits=8, a_bits=16) -> dict:
    """Plain Qm.n fixed point: grid over fractional bits shows the range/precision cliff."""
    d = ctx.data
    m = ctx.models[0]
    acc = np.zeros((len(w_fracs), len(a_fracs)))
    sat = np.zeros_like(acc)
    for i, wf in enumerate(w_fracs):
        for j, af in enumerate(a_fracs):
            logits, _, frac = fixed_forward(m, d.X_test, w_bits, wf, a_bits, af)
            acc[i, j] = np.mean(np.argmax(logits, 1) == d.y_test)
            sat[i, j] = frac
    wmax = float(max(np.abs(W).max() for W in m.W))
    _, acts = m.forward(d.X_test, return_acts=True)
    amax = float(max(np.abs(a).max() for a in acts))
    return {"w_fracs": list(w_fracs), "a_fracs": list(a_fracs), "w_bits": w_bits, "a_bits": a_bits,
            "acc": acc.tolist(), "sat_frac": sat.tolist(), "max_abs_weight": wmax,
            "max_abs_activation": amax, "float_acc": float(m.accuracy(d.X_test, d.y_test))}


def exp_pruning(ctx: Context, sparsities=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95),
                keep_fracs=(1.0, 0.75, 0.5, 0.35, 0.25), wbits=8, ft_epochs=20) -> dict:
    """Accuracy vs flash for unstructured (global/layer, with/without fine-tune) and structured."""
    d = ctx.data
    base = ctx.models[0]
    out = {"wbits": wbits, "unstructured": [], "structured": []}
    for scope in ("global", "layer"):
        for ft in (0, ft_epochs):
            for sp in sparsities:
                p, _ = prune_and_finetune(base, d.X_train, d.y_train, sp, scope, epochs=ft)
                q = build_qmlp(p, ctx.Xcal, wbits, 8, True)
                dep = mlp_deployment("p", p, wbits, 8, True, qmlp=q)
                out["unstructured"].append({
                    "scope": scope, "finetune": ft > 0, "sparsity": sp,
                    "actual_sparsity": sparsity_of(p),
                    "acc": q.accuracy(d.X_test, d.y_test), "flash_bytes": dep.flash_bytes,
                    "macs": dep.macs, "sparse_macs": dep.sparse_macs})
    for kf in keep_fracs:
        s = structured_prune(base, kf)
        if kf < 1.0:
            fit(s, d.X_train, d.y_train, epochs=ft_epochs, lr=1e-3, seed=1)
        q = build_qmlp(s, ctx.Xcal, wbits, 8, True)
        dep = mlp_deployment("s", s, wbits, 8, True, qmlp=q)
        out["structured"].append({"keep_frac": kf, "sizes": s.sizes, "acc": q.accuracy(d.X_test, d.y_test),
                                  "flash_bytes": dep.flash_bytes, "macs": dep.macs})
    return out


def exp_memplan(align: int = 4) -> dict:
    """Arena size per planner on three graphs of increasing topological difficulty."""
    graphs = [mlp_graph(SIZES, 1), tiny_resnet_graph(32, (8, 16, 32), 1), unet_graph(32, 8, 3, 1),
              tiny_resnet_graph(48, (16, 32, 64), 1)]
    graphs[3].name = "tiny_resnet_wide"
    graphs.append(random_graph(40, seed=3))
    out = {"graphs": []}
    for g in graphs:
        bufs = liveness(g)
        row = {"name": g.name, "n_tensors": len(bufs), "macs": g.total_macs(),
               "lower_bound": lower_bound(bufs, align), "planners": {}}
        for name, fn in PLANNERS.items():
            plan, arena = fn(bufs, align)
            assert validate(plan, arena, align)
            row["planners"][name] = arena
        row["mcu_fit"] = {m.name: fits_sram(row["planners"]["random_search"], m) for m in MCUS}
        row["mcu_fit_naive"] = {m.name: fits_sram(row["planners"]["naive"], m) for m in MCUS}
        out["graphs"].append(row)
    return out


def exp_mcu(ctx: Context) -> dict:
    """The same task deployed five ways on four MCU classes."""
    d = ctx.data
    base = ctx.models[0]
    designs = []
    designs.append(mlp_deployment("fp32 dense", base, 32, 32, False))
    q8 = build_qmlp(base, ctx.Xcal, 8, 8, True)
    designs.append(mlp_deployment("int8 dense", base, 8, 8, True, qmlp=q8))
    q4 = build_qmlp(base, ctx.Xcal, 4, 8, True)
    designs.append(mlp_deployment("int4 dense", base, 4, 8, True, qmlp=q4))
    p, _ = prune_and_finetune(base, d.X_train, d.y_train, 0.9, "global", epochs=20)
    qp = build_qmlp(p, ctx.Xcal, 8, 8, True)
    designs.append(mlp_deployment("int8 + 90% sparse", p, 8, 8, True, qmlp=qp))
    s = structured_prune(base, 0.5)
    fit(s, d.X_train, d.y_train, epochs=20, lr=1e-3, seed=1)
    qs = build_qmlp(s, ctx.Xcal, 8, 8, True)
    designs.append(mlp_deployment("int8 + 50% neurons", s, 8, 8, True, qmlp=qs))
    accs = [base.accuracy(d.X_test, d.y_test), q8.accuracy(d.X_test, d.y_test),
            q4.accuracy(d.X_test, d.y_test), qp.accuracy(d.X_test, d.y_test),
            qs.accuracy(d.X_test, d.y_test)]
    rows = []
    for dep, acc in zip(designs, accs):
        r = {"design": dep.label, "acc": float(acc), "flash_bytes": dep.flash_bytes,
             "sram_bytes": dep.sram_bytes, "macs": dep.macs, "mcus": {}}
        for m in MCUS:
            r["mcus"][m.name] = evaluate(dep, m)
        rows.append(r)
    return {"designs": rows, "mcus": [m.name for m in MCUS]}


def pareto_front(points):
    """Indices of points not dominated in (minimise x, maximise y)."""
    idx = []
    for i, (x, y) in enumerate(points):
        if not any((x2 <= x and y2 >= y) and (x2 < x or y2 > y) for j, (x2, y2) in enumerate(points) if j != i):
            idx.append(i)
    return idx
