"""Layer 4: one-seed evaluation of detectors and strategies on a train/test split."""
from __future__ import annotations

import time

import numpy as np

from .config import ExpConfig
from .detectors import build_detector
from .metrics import detection_delay, permutation_accuracy
from .strategy import (default_exposures, net_returns, regime_positions, summarize,
                       vol_target_positions)
from .synth import SimData, persistence_matrix, regime_params, simulate


def make_data(cfg: ExpConfig, seed: int) -> SimData:
    rng = np.random.default_rng(1000 + seed)
    params = regime_params(cfg.k, cfg.separation, cfg.ar_strength)
    return simulate(cfg.n_train + cfg.n_test, persistence_matrix(cfg.k, cfg.p_stay), params, rng,
                    cycle_amp=cfg.cycle_amp)


def _strategy_metrics(pos, r_test, cfg):
    pnl, turn = net_returns(pos, r_test, cfg.cost_bps)
    return summarize(pnl, turn)


def evaluate_seed(cfg: ExpConfig, seed: int, names, baselines: bool = True) -> dict:
    """Return {method: metrics} for the requested detectors plus buy-hold / vol-target / oracle baselines."""
    sim = make_data(cfg, seed)
    r_tr, r_te = sim.returns[:cfg.n_train], sim.returns[cfg.n_train:]
    s_te = sim.states[cfg.n_train:]
    expo = default_exposures(cfg.k)
    out: dict = {}
    for name in names:
        det = build_detector(name, cfg.k, cfg.feature_window, cfg.vol_lambda, cfg.n_iter, seed)
        t0 = time.perf_counter()
        det.fit(r_tr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        lab = det.predict(r_te)
        pred_us = (time.perf_counter() - t0) / len(r_te) * 1e6
        m = _strategy_metrics(regime_positions(lab, expo), r_te, cfg)
        m.update(acc=permutation_accuracy(s_te, lab, cfg.k), delay=detection_delay(s_te, lab, cfg.k),
                 fit_s=fit_s, pred_us=pred_us)
        out[name] = m
    if baselines:
        out["buy-hold"] = _strategy_metrics(np.ones(len(r_te)), r_te, cfg)
        out["vol-target"] = _strategy_metrics(vol_target_positions(r_te, float(r_tr.std()), cfg.vol_lambda), r_te, cfg)
        m = _strategy_metrics(regime_positions(s_te, expo), r_te, cfg)
        m.update(acc=1.0, delay=0.0)
        out["oracle"] = m
    return out
