"""Layer 2: lexical and latent retrieval over chunks: BM25, TF-IDF, LSA, thesaurus expansion, RRF."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from .catalog import all_params
from .textnorm import edit_distance, words


def count_matrix(docs: Sequence[Sequence[str]]) -> Tuple[Dict[str, int], sp.csr_matrix]:
    vocab: Dict[str, int] = {}
    rows, cols, vals = [], [], []
    for i, toks in enumerate(docs):
        cnt: Dict[int, int] = {}
        for t in toks:
            j = vocab.setdefault(t, len(vocab))
            cnt[j] = cnt.get(j, 0) + 1
        for j, c in cnt.items():
            rows.append(i)
            cols.append(j)
            vals.append(float(c))
    return vocab, sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(vocab)))


def query_vector(vocab: Dict[str, int], weights: Dict[str, float]) -> np.ndarray:
    q = np.zeros(len(vocab))
    for t, w in weights.items():
        if t in vocab:
            q[vocab[t]] += w
    return q


class BM25:
    """Okapi BM25 with the Lucene idf  ln(1 + (N - n + 0.5) / (n + 0.5)) (Robertson & Zaragoza, 2009)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: Sequence[Sequence[str]]) -> "BM25":
        self.vocab, tf = count_matrix(docs)
        N = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avg = dl.mean() if N else 1.0
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log(1.0 + (N - df + 0.5) / (df + 0.5))
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avg)
        w = self.idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.M = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, weights: Dict[str, float]) -> np.ndarray:
        return self.M @ query_vector(self.vocab, weights)


class Tfidf:
    """TF-IDF with sublinear tf, idf = ln((1+N)/(1+df)) + 1 and L2-normalised rows (scikit-learn convention)."""

    def __init__(self, sublinear: bool = True):
        self.sublinear = sublinear

    def fit(self, docs: Sequence[Sequence[str]]) -> "Tfidf":
        self.vocab, tf = count_matrix(docs)
        N = tf.shape[0]
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        self.idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
        tf = tf.tocoo()
        data = (1.0 + np.log(tf.data)) if self.sublinear else tf.data
        X = sp.csr_matrix((data * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        self.X = sp.diags(1.0 / norms) @ X
        return self

    def vectorize(self, weights: Dict[str, float]) -> np.ndarray:
        q = query_vector(self.vocab, weights)
        nz = q > 0
        if self.sublinear:
            q[nz] = 1.0 + np.log(q[nz])
        q = q * self.idf
        n = np.linalg.norm(q)
        return q / n if n else q

    def scores(self, weights: Dict[str, float]) -> np.ndarray:
        return self.X @ self.vectorize(weights)


class LSA:
    """Latent semantic analysis: truncated SVD of the TF-IDF matrix (Deerwester et al., 1990)."""

    def __init__(self, k: int = 64):
        self.k = k

    def fit(self, docs: Sequence[Sequence[str]]) -> "LSA":
        self.tfidf = Tfidf().fit(docs)
        k = max(1, min(self.k, min(self.tfidf.X.shape) - 1))
        U, s, Vt = svds(self.tfidf.X.astype(float), k=k, v0=np.ones(min(self.tfidf.X.shape)))
        o = np.argsort(-s)
        self.U, self.s, self.Vt = U[:, o], s[o], Vt[o]
        self.D = self.U * self.s  # document embeddings = X V_k
        self.Dn = self.D / np.maximum(np.linalg.norm(self.D, axis=1, keepdims=True), 1e-12)
        return self

    def embed_query(self, weights: Dict[str, float]) -> np.ndarray:
        return self.Vt @ self.tfidf.vectorize(weights)

    def scores(self, weights: Dict[str, float]) -> np.ndarray:
        q = self.embed_query(weights)
        n = np.linalg.norm(q)
        return self.Dn @ (q / n) if n else np.zeros(self.Dn.shape[0])


class Thesaurus:
    """Domain synonym expansion built from the parameter catalogue (names, aliases, symbols)."""

    def __init__(self):
        self.groups: List[Tuple[List[Tuple[str, ...]], List[str]]] = []
        for pd in all_params():
            phrases = [tuple(words(p)) for p in list(pd.names) + list(pd.aliases)]
            self.groups.append((phrases, words(pd.symbol, drop_stop=False)))

    def expand(self, toks: Sequence[str], weight: float = 0.5) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for t in toks:
            out[t] = out.get(t, 0.0) + 1.0
        have = set(toks)
        for phrases, sym in self.groups:
            if any(ph and set(ph) <= have for ph in phrases):
                for ph in phrases:
                    for t in ph:
                        if t not in have:
                            out[t] = max(out.get(t, 0.0), weight)
                for t in sym:
                    out.setdefault(t, weight)
        return out


def rank_desc(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def rrf(score_vectors: Sequence[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion (Cormack et al., 2009): sum_i 1 / (k + rank_i)."""
    total = np.zeros(len(score_vectors[0]))
    for s in score_vectors:
        r = np.empty(len(s))
        r[rank_desc(s)] = np.arange(1, len(s) + 1)
        total += 1.0 / (k + r)
    return total


class Retriever:
    """kind in {bm25, tfidf, lsa, hybrid}; ``expand`` adds thesaurus terms; ``part_filter`` routes by part number."""

    def __init__(self, kind: str = "bm25", expand: bool = False, part_filter: bool = False, lsa_k: int = 64, max_edit: int = 2):
        self.kind, self.expand, self.part_filter, self.lsa_k, self.max_edit = kind, expand, part_filter, lsa_k, max_edit
        self.thes = Thesaurus()

    def fit(self, texts: Sequence[str], parts: Sequence[str]) -> "Retriever":
        docs = [words(t) for t in texts]
        self.parts = list(parts)
        self.part_names = sorted({p.lower() for p in parts})
        if self.kind in ("bm25", "hybrid"):
            self.bm25 = BM25().fit(docs)
        if self.kind == "tfidf":
            self.tfidf = Tfidf().fit(docs)
        if self.kind in ("lsa", "hybrid"):
            self.lsa = LSA(self.lsa_k).fit(docs)
        return self

    def detect_part(self, query: str) -> Optional[str]:
        best: Optional[Tuple[int, str]] = None
        for t in words(query, drop_stop=False, do_stem=False):
            if len(t) < 5 or not any(c.isdigit() for c in t):
                continue
            for p in self.part_names:
                d = edit_distance(t, p)
                if d <= self.max_edit and (best is None or d < best[0]):
                    best = (d, p)
        return best[1] if best else None

    def weights(self, query: str) -> Dict[str, float]:
        toks = words(query)
        if self.expand:
            return self.thes.expand(toks)
        w: Dict[str, float] = {}
        for t in toks:
            w[t] = w.get(t, 0.0) + 1.0
        return w

    def scores(self, query: str) -> np.ndarray:
        w = self.weights(query)
        if self.kind == "bm25":
            s = self.bm25.scores(w)
        elif self.kind == "tfidf":
            s = self.tfidf.scores(w)
        elif self.kind == "lsa":
            s = self.lsa.scores(w)
        else:
            s = rrf([self.bm25.scores(w), self.lsa.scores(w)])
        if self.part_filter:
            p = self.detect_part(query)
            if p is not None:
                mask = np.array([q.lower() == p for q in self.parts])
                s = np.where(mask, s + (abs(s).max() + 1.0), s)
        return s

    def rank(self, query: str) -> np.ndarray:
        return rank_desc(self.scores(query))


def rank_of(order: np.ndarray, target: int) -> Optional[int]:
    hit = np.nonzero(order == target)[0]
    return int(hit[0]) + 1 if len(hit) else None
