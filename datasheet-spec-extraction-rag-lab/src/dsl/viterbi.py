"""Layer 0: Viterbi decoding of the best label path given emission and transition scores."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def viterbi(emit: np.ndarray, trans: np.ndarray, start: Optional[np.ndarray] = None) -> Tuple[List[int], float]:
    """Maximise sum_t emit[t, y_t] + sum_t trans[y_{t-1}, y_t] (+ start[y_0]).

    emit: (T, K) scores, trans: (K, K) with trans[i, j] = score of moving i -> j.
    Returns (best path, best score).
    """
    T, K = emit.shape
    if T == 0:
        return [], 0.0
    dp = np.empty((T, K))
    bp = np.zeros((T, K), dtype=np.int64)
    dp[0] = emit[0] + (start if start is not None else 0.0)
    for t in range(1, T):
        cand = dp[t - 1][:, None] + trans
        bp[t] = cand.argmax(axis=0)
        dp[t] = cand.max(axis=0) + emit[t]
    path = [int(dp[-1].argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(bp[t][path[-1]]))
    return path[::-1], float(dp[-1].max())


def path_score(path: List[int], emit: np.ndarray, trans: np.ndarray, start: Optional[np.ndarray] = None) -> float:
    s = float(emit[0, path[0]]) + (float(start[path[0]]) if start is not None else 0.0)
    for t in range(1, len(path)):
        s += float(trans[path[t - 1], path[t]]) + float(emit[t, path[t]])
    return s
