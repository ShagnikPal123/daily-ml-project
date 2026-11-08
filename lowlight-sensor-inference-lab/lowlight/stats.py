"""Layer 0: small statistics helpers for multi-seed experiments."""
from __future__ import annotations

import numpy as np
from scipy import stats as _st


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of the Student-t confidence interval over seeds."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return float("nan"), float("nan")
    m = float(v.mean())
    if n == 1:
        return m, float("nan")
    se = float(v.std(ddof=1)) / np.sqrt(n)
    return m, float(_st.t.ppf(0.5 + confidence / 2, n - 1) * se)


def paired_diff_ci(a, b, confidence: float = 0.95) -> tuple[float, float]:
    """Mean difference a-b (paired by seed) with its t-interval half-width."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal shape")
    return mean_ci(a - b, confidence)


def bootstrap_ci(values, n_boot: int = 2000, confidence: float = 0.95, seed: int = 0):
    """Percentile bootstrap interval for the mean."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    a = (1 - confidence) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def significant(diff_mean: float, half_width: float) -> bool:
    """True when the paired CI excludes zero."""
    return bool(np.isfinite(half_width) and abs(diff_mean) > half_width)


def p_from_ci(diff: float, half_width: float, n: int, confidence: float = 0.95) -> float:
    """Two-sided p-value of a paired-t test recovered from a stored mean and CI half-width."""
    if n < 2 or not np.isfinite(half_width):
        return float("nan")
    if half_width == 0:
        return 0.0 if diff != 0 else 1.0
    se = half_width / _st.t.ppf(0.5 + confidence / 2, n - 1)
    return float(2 * _st.t.sf(abs(diff) / se, n - 1))


def holm_reject(pvals, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: which hypotheses survive family-wise error control."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    reject = np.zeros(len(p), dtype=bool)
    m = len(p)
    for rank, i in enumerate(order):
        if np.isnan(p[i]) or p[i] > alpha / (m - rank):
            break
        reject[i] = True
    return reject.tolist()
