"""Layer 2: regime-conditioned exposure rules and cost-aware P&L (simulation only - not advice)."""
from __future__ import annotations

import numpy as np

from .filters import ewma_vol
from .metrics import max_drawdown, sharpe


def default_exposures(k: int) -> np.ndarray:
    """Full exposure in the calmest regime, flat in the most stressed one."""
    return np.linspace(1.0, 0.0, k)


def regime_positions(labels: np.ndarray, exposures: np.ndarray, lag: int = 1) -> np.ndarray:
    """Position held over bar t is chosen from the label known at bar t-lag (no look-ahead)."""
    labels = np.asarray(labels)
    pos = np.ones(len(labels))
    pos[lag:] = np.asarray(exposures, float)[labels[:-lag]]
    return pos


def vol_target_positions(returns: np.ndarray, target_daily_vol: float, lam: float = 0.94, cap: float = 2.0) -> np.ndarray:
    vol = ewma_vol(returns, lam)
    pos = np.ones(len(vol))
    pos[1:] = np.minimum(cap, target_daily_vol / vol[:-1])
    return pos


def net_returns(pos: np.ndarray, returns: np.ndarray, cost_bps: float = 5.0) -> tuple[np.ndarray, float]:
    """Strategy returns after proportional costs; also returns mean absolute turnover per bar."""
    pos = np.asarray(pos, float)
    turn = np.abs(np.diff(pos, prepend=pos[0]))
    pnl = pos * np.asarray(returns, float) - turn * cost_bps * 1e-4
    return pnl, float(turn.mean())


def summarize(pnl: np.ndarray, turnover: float) -> dict:
    return {"sharpe": sharpe(pnl), "max_dd": max_drawdown(pnl),
            "ann_ret": float(np.mean(pnl) * 252), "turnover": turnover}
