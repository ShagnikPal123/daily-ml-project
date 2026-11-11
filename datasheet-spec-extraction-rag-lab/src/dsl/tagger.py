"""Layer 2: averaged structured-perceptron token tagger with Viterbi decoding (Collins, 2002)."""
from __future__ import annotations

import re
import zlib
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

from .schema import L2I, LABELS
from .textnorm import SEP, clean_tok, shape, tokenize_spans
from .units import is_unit
from .viterbi import viterbi

GROUPS = ("lex", "shape", "ctx", "pos")
_NUMLIKE = re.compile(r"^-?[0-9OlISBb]+(\.[0-9OlISBb]+)?$")


def _flags(tok: str) -> Tuple[str, bool, bool]:
    c = clean_tok(tok)
    num = bool(_NUMLIKE.match(c)) and any(ch.isdigit() for ch in c)
    unit = is_unit(c) or is_unit(c.replace("rn", "m"))
    return c, num, unit


def token_features(toks: Sequence[str], groups: FrozenSet[str]) -> List[List[str]]:
    n = len(toks)
    info = [_flags(t) for t in toks]
    cells_before, k = [], 0
    for t in toks:
        cells_before.append(k)
        k += t == SEP
    total = k
    out: List[List[str]] = []
    for i, t in enumerate(toks):
        c, num, unit = info[i]
        f = ["b"]
        low = c.lower()
        if "lex" in groups:
            if low.isalpha() and len(low) <= 12:
                f.append("w=" + low)
            if low.isalpha():
                f.append("s3=" + low[-3:])
        if "shape" in groups:
            f += ["sh=" + shape(c), f"num={num}", f"unit={unit}", f"dash={c == '-'}", f"cap={c[:1].isupper()}",
                  f"allcap={c.isupper()}", f"paren={t.startswith('(')}", f"colon={t.endswith(':')}", f"len={min(len(c), 8)}"]
        if "ctx" in groups:
            for off in (-2, -1, 1, 2):
                j = i + off
                if 0 <= j < n:
                    cj, nj, uj = info[j]
                    f += [f"p{off}sh={shape(cj)}", f"p{off}n={nj}", f"p{off}u={uj}"]
                    if abs(off) == 1:
                        f.append(f"p{off}w={cj.lower()}")
                else:
                    f.append(f"p{off}=edge")
        if "pos" in groups:
            f += [f"cell={min(cells_before[i], 7)}", f"cellr={min(total - cells_before[i], 7)}", f"rel={int(5 * i / max(n, 1))}",
                  f"cell_num={min(cells_before[i], 7)}|{num}|{unit}"]
        out.append(f)
    return out


class SeqTagger:
    """Per-token linear scores + label-transition scores, trained with the averaged perceptron."""

    def __init__(self, groups: Sequence[str] = GROUPS, use_viterbi: bool = True, dim: int = 1 << 15, epochs: int = 6, seed: int = 0):
        self.groups = frozenset(groups)
        self.use_viterbi = use_viterbi
        self.dim, self.epochs, self.seed = dim, epochs, seed
        self.K = len(LABELS)
        self.W = np.zeros((dim, self.K))
        self.T = np.zeros((self.K, self.K))

    def _idx(self, toks: Sequence[str]) -> List[np.ndarray]:
        return [np.array([zlib.crc32(s.encode()) % self.dim for s in fs], dtype=np.int64) for fs in token_features(toks, self.groups)]

    @staticmethod
    def _emit(W: np.ndarray, feats: List[np.ndarray]) -> np.ndarray:
        return np.stack([W[f].sum(axis=0) for f in feats])

    def fit(self, lines: Sequence[Sequence[Tuple[str, str]]]) -> "SeqTagger":
        data = [(self._idx([t for t, _ in ln]), [L2I[l] for _, l in ln]) for ln in lines if ln]
        W, T = np.zeros((self.dim, self.K)), np.zeros((self.K, self.K))
        U, UT = np.zeros_like(W), np.zeros_like(T)
        rng = np.random.default_rng(self.seed)
        c = 1
        for _ in range(self.epochs):
            for n in rng.permutation(len(data)):
                feats, gold = data[n]
                emit = self._emit(W, feats)
                pred = viterbi(emit, T)[0] if self.use_viterbi else emit.argmax(axis=1).tolist()
                if pred != gold:
                    for t, (g, p) in enumerate(zip(gold, pred)):
                        if g != p:
                            np.add.at(W, (feats[t], g), 1.0)
                            np.add.at(W, (feats[t], p), -1.0)
                            np.add.at(U, (feats[t], g), c)
                            np.add.at(U, (feats[t], p), -c)
                    if self.use_viterbi:
                        for t in range(1, len(gold)):
                            if (gold[t - 1], gold[t]) != (pred[t - 1], pred[t]):
                                T[gold[t - 1], gold[t]] += 1
                                T[pred[t - 1], pred[t]] -= 1
                                UT[gold[t - 1], gold[t]] += c
                                UT[pred[t - 1], pred[t]] -= c
                c += 1
        self.W, self.T = W - U / c, T - UT / c
        return self

    def predict_tokens(self, toks: Sequence[str]) -> List[str]:
        if not toks:
            return []
        emit = self._emit(self.W, self._idx(toks))
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
