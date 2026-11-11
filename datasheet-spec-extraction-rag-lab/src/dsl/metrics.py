"""Layer 0: ranking / extraction metrics and confidence intervals."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def recall_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    """ranks: 1-based rank of the relevant item per query (None = not retrieved)."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: Sequence[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def ndcg_at_k(ranks: Sequence[Optional[int]], k: int = 10) -> float:
    """Single-relevant-item nDCG: ideal DCG is 1, so nDCG = 1/log2(rank+1) when rank <= k."""
    if not ranks:
        return 0.0
    return sum(1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / len(ranks)


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def mean_ci(x: Iterable[float], conf: float = 0.95) -> Tuple[float, float]:
    """Mean and half-width of the Student-t confidence interval across independent runs."""
    a = np.asarray(list(x), dtype=float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    m = float(a.mean())
    if n < 2:
        return m, 0.0
    h = float(stats.t.ppf((1 + conf) / 2, n - 1) * a.std(ddof=1) / math.sqrt(n))
    return m, h


def bootstrap_ci(x: Sequence[float], n_boot: int = 1000, conf: float = 0.95, seed: int = 0) -> Tuple[float, float]:
    a = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    means = a[idx].mean(axis=1)
    return float(np.quantile(means, (1 - conf) / 2)), float(np.quantile(means, 1 - (1 - conf) / 2))


def fmt_ci(m: float, h: float, digits: int = 3) -> str:
    return f"{m:.{digits}f} ± {h:.{digits}f}"
