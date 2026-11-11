"""Layer 3: fixed-point (integer) inference for the trained tagger -- the deployment-minded angle.

Weights are quantised per label column to signed ``bits``-bit integers; emission scores are accumulated in
int32 and rescaled once per token, which is how a microcontroller without an FPU would run the model.
The tiny label-transition matrix (9 x 9) stays in float.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .tagger import SeqTagger
from .schema import LABELS
from .textnorm import tokenize_spans
from .viterbi import viterbi


class QuantTagger:
    """Drop-in replacement for :class:`SeqTagger` at inference time."""

    def __init__(self, tagger: SeqTagger, bits: int = 8):
        if not 2 <= bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        self.tagger, self.bits = tagger, bits
        qmax = 2 ** (bits - 1) - 1
        peak = np.abs(tagger.W).max(axis=0)
        self.scale = np.where(peak > 0, peak / qmax, 1.0)  # one scale per label column
        self.Wq = np.clip(np.round(tagger.W / self.scale), -qmax, qmax).astype(np.int8)
        self.T = tagger.T.copy()
        self.use_viterbi = tagger.use_viterbi

    def emissions(self, toks: Sequence[str]) -> np.ndarray:
        idx = self.tagger._idx(toks)
        acc = np.stack([self.Wq[f].astype(np.int32).sum(axis=0) for f in idx])  # integer accumulation
        return acc * self.scale

    def predict_tokens(self, toks: Sequence[str]) -> List[str]:
        if not toks:
            return []
        emit = self.emissions(toks)
        path = viterbi(emit, self.T)[0] if self.use_viterbi else emit.argmax(axis=1).tolist()
        return [LABELS[i] for i in path]

    def predict(self, text: str) -> Tuple[List[str], List[str]]:
        toks = [t for t, _, _ in tokenize_spans(text)]
        return toks, self.predict_tokens(toks)

    def token_accuracy(self, lines: Sequence[Sequence[Tuple[str, str]]]) -> float:
        ok = tot = 0
        for ln in lines:
            pred = self.predict_tokens([t for t, _ in ln])
            ok += sum(p == l for p, (_, l) in zip(pred, ln))
            tot += len(ln)
        return ok / max(tot, 1)

    def dense_bytes(self) -> float:
        """Packed dense weight table plus float32 scales and transitions."""
        return self.Wq.size * self.bits / 8 + 4 * (self.scale.size + self.T.size)

    def sparse_bytes(self) -> float:
        """Only non-zero weights stored as (int32 index, packed value) pairs plus scales and transitions."""
        nnz = int(np.count_nonzero(self.Wq))
        return nnz * (4 + self.bits / 8) + 4 * (self.scale.size + self.T.size)


def float_model_bytes(tagger: SeqTagger) -> float:
    """float32 dense table of the unquantised model, the reference footprint."""
    return tagger.W.size * 4 + 4 * tagger.T.size
