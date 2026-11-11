"""Layer 3: token-level confusion analysis for the tagger (which labels get mistaken for which)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .metrics import prf
from .schema import L2I, LABELS


def confusion_matrix(tagger, lines: Sequence[Sequence[Tuple[str, str]]]) -> np.ndarray:
    """cm[g, p] = number of tokens with gold label g predicted as p (tagger needs ``predict_tokens``)."""
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for ln in lines:
        if not ln:
            continue
        pred = tagger.predict_tokens([t for t, _ in ln])
        for (_, g), p in zip(ln, pred):
            cm[L2I[g], L2I[p]] += 1
    return cm


def per_label_prf(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for i, lab in enumerate(LABELS):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        d = prf(tp, fp, fn)
        d["support"] = int(cm[i, :].sum())
        out[lab] = d
    return out


def top_confusions(cm: np.ndarray, n: int = 5) -> List[Tuple[str, str, int]]:
    """Largest off-diagonal cells as (gold, predicted, count)."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:n]
    return [(LABELS[i // len(LABELS)], LABELS[i % len(LABELS)], int(off.flat[i])) for i in flat if off.flat[i] > 0]


def accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / max(cm.sum(), 1))
