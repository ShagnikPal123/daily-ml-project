"""Layer 4: config-driven experiment runner, aggregation and failure analysis.

One JSON config (see ``configs/``) fully determines a run: headline sweep over
exposure x denoiser x seed, ablations (train mode, VST inverse, patch size), sweeps over
ADC bits and read noise, a classifier-quantization study, a failure-case analysis, and a
deployment cost table. All trials share per-seed noise streams (paired design).
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import budget, data
from .classifiers import make_classifier
from .denoisers import DENOISERS, make_denoiser
from .pipeline import PipelineConfig, run_trial
from .sensor import Sensor, SensorConfig
from .stats import mean_ci, paired_diff_ci, significant

DEFAULTS = json.load(open(os.path.join(os.path.dirname(__file__), "default_config.json")))
METRICS = ("acc", "psnr", "ssim")


# ------------------------------------------------------------------ config
def load_config(spec=None) -> dict:
    """Merge a config (path, dict or None) over the defaults and validate it."""
    if isinstance(spec, (str, os.PathLike)):
        with open(spec) as f:
            spec = json.load(f)
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (spec or {}).items():
        if k not in cfg:
            raise ValueError(f"unknown config key {k!r}")
        if isinstance(cfg[k], dict) and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    if not cfg["seeds"]:
        raise ValueError("need at least one seed")
    if any(p <= 0 for p in cfg["photons"]):
        raise ValueError("photons must be positive")
    for d in cfg["denoisers"]:
        if d not in DENOISERS:
            raise ValueError(f"unknown denoiser {d!r}")
    if "identity" not in cfg["denoisers"]:
        raise ValueError("'identity' baseline must be in denoisers")
    if cfg["failure"]["photons"] not in cfg["photons"]:
        raise ValueError("failure.photons must be one of photons")
    SensorConfig.from_dict({**cfg["sensor"], "photons": 10.0})


# ------------------------------------------------------------------ tasks
def _pc(cfg, photons, den, *, clf=None, mode="matched", dkw=None, bits=None, read=None) -> PipelineConfig:
    s = dict(cfg["sensor"], photons=float(photons))
    if bits is not None:
        s["adc_bits"] = int(bits)
    if read is not None:
        s["read_noise"] = float(read)
    return PipelineConfig(sensor=SensorConfig.from_dict(s), denoiser=den, denoiser_kw=dkw or {},
                          classifier=clf or cfg["classifier"], train_mode=mode, n_noise_draws=cfg["n_noise_draws"])


def build_tasks(cfg: dict) -> list:
    """List of (experiment_tag, seed, PipelineConfig)."""
    T, S = [], cfg["seeds"]
    for s, p, d in itertools.product(S, cfg["photons"], cfg["denoisers"]):
        T.append(("headline", s, _pc(cfg, p, d)))
    a = cfg["train_mode_ablation"]
    for s, p, d in itertools.product(S, a["photons"], a["denoisers"]):
        T.append(("train_clean", s, _pc(cfg, p, d, mode="clean")))
    for s, p in itertools.product(S, cfg["gat_ablation"]["photons"]):
        T.append(("gat_algebraic", s, _pc(cfg, p, "gat", dkw={"inverse": "algebraic"})))
    pa = cfg["patch_ablation"]
    for s, p, k in itertools.product(S, pa["photons"], pa["sizes"]):
        T.append(("patch", s, _pc(cfg, p, "ridge", dkw={"size": k})))
    b = cfg["bits_sweep"]
    for s, p, bits, d in itertools.product(b["seeds"], b["photons"], b["bits"], b["denoisers"]):
        T.append(("bits", s, _pc(cfg, p, d, bits=bits)))
    r = cfg["read_noise_sweep"]
    for s, p, rn, d in itertools.product(r["seeds"], r["photons"], r["values"], r["denoisers"]):
        T.append(("readnoise", s, _pc(cfg, p, d, read=rn)))
    c = cfg["classifier_study"]
    for s, p, clf in itertools.product(c["seeds"], c["photons"], c["classifiers"]):
        T.append(("classifier", s, _pc(cfg, p, c["denoiser"], clf=clf)))
    return T


_G: dict = {}


def _init(size: int) -> None:
    _G["x"], _G["y"] = data.load_scenes(size)
    _G["splits"] = {}


def _run_task(task):
    tag, seed, pcfg = task
    if seed not in _G["splits"]:
        _G["splits"][seed] = data.make_splits(_G["y"], seed)
    r = run_trial(_G["x"], _G["y"], _G["splits"][seed], pcfg, seed)
    r["exp"] = tag
    return r


def execute(tasks, size: int = 16, workers: int = 1) -> list[dict]:
    if workers <= 1:
        _init(size)
        return [_run_task(t) for t in tasks]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(size,)) as ex:
        return list(ex.map(_run_task, tasks, chunksize=4))


# ------------------------------------------------------------------ aggregation
def group_stats(records, keys, metrics=METRICS) -> list[dict]:
    groups: dict = {}
    for r in records:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    rows = []
    for g in sorted(groups, key=lambda t: tuple(str(v) if isinstance(v, str) else v for v in t)):
        rs = groups[g]
        row = dict(zip(keys, g), n=len(rs))
        for m in metrics:
            mu, hw = mean_ci([r[m] for r in rs])
            row[m], row[m + "_ci"] = mu, hw
        rows.append(row)
    return rows


def paired_vs_baseline(records, base="identity", metric="acc") -> list[dict]:
    """Paired (by seed) difference of each denoiser against ``base`` at each exposure."""
    by = {(r["photons"], r["denoiser"], r["seed"]): r[metric] for r in records}
    out = []
    for (p, d) in sorted({(r["photons"], r["denoiser"]) for r in records}):
        if d == base:
            continue
        seeds = sorted({r["seed"] for r in records if r["photons"] == p and r["denoiser"] == d})
        a, b = [by[(p, d, s)] for s in seeds], [by[(p, base, s)] for s in seeds]
        mu, hw = paired_diff_ci(a, b)
        out.append({"photons": p, "denoiser": d, "metric": metric, "diff": mu, "ci": hw,
                    "significant": significant(mu, hw), "n": len(seeds)})
    return out


def failure_analysis(records, labels, cfg) -> dict:
    """Where and why the pipeline breaks: per-class accuracy, confusions, crossover, metric disagreement."""
    fp = float(cfg["failure"]["photons"])
    head = [r for r in records if r["exp"] == "headline"]
    stats = group_stats([r for r in head if r["photons"] == fp], ("denoiser",))
    best = max(stats, key=lambda r: r["acc"])["denoiser"]
    out = {"photons": fp, "best_method": best}
    for name in ("identity", best):
        rs = [r for r in head if r["photons"] == fp and r["denoiser"] == name]
        out[f"per_class_{name}"] = np.mean([r["per_class"] for r in rs], axis=0).tolist()
    conf = np.sum([np.array(r["confusion"]) for r in head if r["photons"] == fp and r["denoiser"] == best], axis=0)
    off = [(int(i), int(j), int(conf[i, j])) for i in range(10) for j in range(10) if i != j]
    out["top_confusions"] = sorted(off, key=lambda t: -t[2])[: cfg["failure"]["top_k"]]
    out["confusion_best"] = conf.tolist()
    # test images that every method gets wrong (irreducible at this exposure)
    fracs = []
    for s in cfg["seeds"]:
        te = data.make_splits(labels, s)["test"]
        wrong = np.ones(len(te), bool)
        for r in head:
            if r["photons"] == fp and r["seed"] == s:
                wrong &= np.array(r["pred"]) != labels[te]
        fracs.append(float(wrong.mean()))
    out["all_methods_wrong_frac"] = mean_ci(fracs)
    # exposures where a method is significantly *worse* than doing nothing
    pv = paired_vs_baseline(head)
    out["harmful"] = [(r["photons"], r["denoiser"], r["diff"]) for r in pv if r["significant"] and r["diff"] < 0]
    # does pixel fidelity (PSNR) pick the same winner as task accuracy?
    dis = []
    for p in cfg["photons"]:
        rows = [r for r in group_stats([r for r in head if r["photons"] == p], ("denoiser",))]
        a, q = max(rows, key=lambda r: r["acc"]), max(rows, key=lambda r: r["psnr"])
        dis.append({"photons": p, "best_acc": a["denoiser"], "best_psnr": q["denoiser"], "agree": a["denoiser"] == q["denoiser"]})
    out["metric_disagreement"] = dis
    return out


def deployment_table(cfg: dict, scenes, labels) -> list[dict]:
    """Per-frame MACs / SRAM / energy of each denoiser with an int8 softmax classifier."""
    p = cfg["deployment_photons"]
    sensor = Sensor(SensorConfig.from_dict({**cfg["sensor"], "photons": float(p)}), scenes.shape[1:], seed=0)
    noisy = sensor.acquire(scenes[:300], np.random.default_rng(0))
    clf = make_classifier("softmax").fit(scenes[:300].reshape(300, -1), labels[:300])
    rows = []
    for name in cfg["denoisers"]:
        d = make_denoiser(name, sensor).fit(scenes[:300], noisy)
        i8 = budget.frame_cost(d, clf, scenes.shape[1:], "int8", "int8")
        f32 = budget.frame_cost(d, clf, scenes.shape[1:], "fp32", "fp32")
        rows.append({"denoiser": name, **i8, "energy_uj_fp32": f32["energy_uj"]})
    return rows


# ------------------------------------------------------------------ driver
def slim(r: dict, keep_pred: bool) -> dict:
    r = dict(r)
    if not keep_pred:
        r.pop("pred", None)
        r.pop("confusion", None)
    return r


def run_all(cfg: dict, out_dir: str | None = None, workers: int = 1) -> dict:
    t0 = time.time()
    scenes, labels = data.load_scenes(cfg["image_size"])
    records = execute(build_tasks(cfg), cfg["image_size"], workers)
    by = lambda tag: [r for r in records if r["exp"] == tag]
    head = by("headline")
    res = {
        "config": cfg,
        "headline": group_stats(head, ("photons", "denoiser")),
        "paired_acc": paired_vs_baseline(head, metric="acc"),
        "paired_psnr": paired_vs_baseline(head, metric="psnr"),
        "train_mode": group_stats(by("train_clean") + [r for r in head if r["photons"] in cfg["train_mode_ablation"]["photons"]
                                  and r["denoiser"] in cfg["train_mode_ablation"]["denoisers"]], ("train_mode", "photons", "denoiser")),
        "gat_ablation": _gat_rows(cfg, records),
        "patch": _patch_rows(by("patch")),
        "bits": group_stats(by("bits"), ("photons", "denoiser", "adc_bits")),
        "readnoise": group_stats(by("readnoise"), ("photons", "denoiser", "read_noise")),
        "classifier": group_stats(by("classifier"), ("photons", "classifier")),
        "failure": failure_analysis(records, labels, cfg),
        "deployment": deployment_table(cfg, scenes, labels),
        "profile": budget.profile_denoisers(cfg["denoisers"], SensorConfig.from_dict({**cfg["sensor"], "photons": float(cfg["profile"]["photons"])}),
                                            scenes[: cfg["profile"]["n_images"]]),
        "n_trials": len(records),
    }
    res["runtime_sec"] = time.time() - t0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        keep = float(cfg["failure"]["photons"])
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(res, f, indent=1, default=_jsonable)
        with open(os.path.join(out_dir, "records.json"), "w") as f:
            json.dump([slim(r, False) for r in records], f, default=_jsonable)
        with open(os.path.join(out_dir, "config_resolved.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    return res


def _gat_rows(cfg, records) -> list[dict]:
    ps = cfg["gat_ablation"]["photons"]
    rows = []
    for variant, tag, den in (("gaussian (no VST)", "headline", "gaussian"), ("gat + algebraic inverse", "gat_algebraic", "gat"),
                              ("gat + asymptotic inverse", "headline", "gat")):
        for st in group_stats([r for r in records if r["exp"] == tag and r["denoiser"] == den and r["photons"] in ps], ("photons",)):
            rows.append({"variant": variant, **st})
    return rows


def _patch_rows(recs) -> list[dict]:
    for r in recs:
        r["patch"] = r["denoiser_params"]["size"]
    return group_stats(recs, ("photons", "patch"))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))
