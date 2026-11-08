"""Layer 0: scoring helpers (label matching, detection delay, risk metrics, confidence intervals)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


def confusion(true: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    C = np.zeros((k, k), dtype=int)
    np.add.at(C, (np.asarray(true), np.asarray(pred)), 1)
    return C


def match_labels(true: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    """Relabel pred to best agree with true (Hungarian assignment on the confusion matrix)."""
    C = confusion(true, pred, k)
    rows, cols = linear_sum_assignment(-C)
    mapping = np.arange(k)
    mapping[cols] = rows
    return mapping[np.asarray(pred)]


def permutation_accuracy(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    return float(np.mean(match_labels(true, pred, k) == np.asarray(true)))


def detection_delay(true: np.ndarray, pred: np.ndarray, k: int, max_delay: int = 50) -> float:
    """Mean steps until the (matched) prediction agrees with a new true regime (censored at max_delay)."""
    true = np.asarray(true)
    mp = match_labels(true, pred, k)
    cps = np.flatnonzero(np.diff(true) != 0) + 1
    if len(cps) == 0:
        return 0.0
    delays = []
    for i, c in enumerate(cps):
        end = min(cps[i + 1] if i + 1 < len(cps) else len(true), c + max_delay)
        hit = np.flatnonzero(mp[c:end] == true[c])
        delays.append(float(hit[0]) if len(hit) else float(end - c))
    return float(np.mean(delays))


def sharpe(returns: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(returns, float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return 0.0 if sd < 1e-15 else float(r.mean() / sd * np.sqrt(periods))


def max_drawdown(returns: np.ndarray) -> float:
    eq = np.cumprod(1.0 + np.asarray(returns, float))
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))


def mean_ci(x, conf: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the Student-t confidence interval."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else float("nan"), 0.0
    se = x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), float(stats.t.ppf(0.5 + conf / 2, n - 1) * se)
