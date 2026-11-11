"""Ground-truth checks of base/middle layers against independent references.

Run ``python tests/test_reference.py`` to print the measured agreement table used in the README.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import constants, stats
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsl import metrics, retrieval as R, textnorm, units, viterbi  # noqa: E402
from dsl.indexcompress import dequantize, quantize_symmetric  # noqa: E402

CORPUS = ["low dropout regulator quiescent current", "op amp slew rate gain bandwidth", "buck converter switching frequency efficiency",
          "adc sampling rate signal noise ratio", "quiescent current standby mode battery", "slew rate and bandwidth of amplifier", "regulator output noise voltage"]
QUERIES = ["quiescent current regulator", "slew rate", "efficiency of buck", "unknown words only"]


def m_prefixes():
    ref = {"p": constants.pico, "n": constants.nano, "u": constants.micro, "m": constants.milli, "k": constants.kilo, "M": constants.mega, "G": constants.giga}
    return max(abs(units.parse_unit(p + "V")[0] / v - 1) for p, v in ref.items())


def m_viterbi_textbook():
    # Hidden-Markov example (healthy/fever; observations normal, cold, dizzy), worked in the Viterbi-algorithm literature.
    start = np.log([0.6, 0.4])
    trans = np.log([[0.7, 0.3], [0.4, 0.6]])
    em = np.log([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    obs = [0, 1, 2]
    path, score = viterbi.viterbi(np.stack([em[:, o] for o in obs]), trans, start)
    return path, math.exp(score)


def m_tfidf():
    vec = TfidfVectorizer(analyzer=lambda d: textnorm.words(d), sublinear_tf=True, smooth_idf=True, norm="l2")
    X = vec.fit_transform(CORPUS)
    ours = R.Tfidf().fit([textnorm.words(c) for c in CORPUS])
    err = 0.0
    for q in QUERIES:
        theirs = (X @ vec.transform([q]).T).toarray().ravel()
        w = {}
        for t in textnorm.words(q):
            w[t] = w.get(t, 0) + 1.0
        err = max(err, float(np.abs(ours.scores(w) - theirs).max()))
    return err


def m_svd():
    docs = [textnorm.words(c) for c in CORPUS]
    l = R.LSA(k=4).fit(docs)
    full = np.linalg.svd(l.tfidf.X.toarray(), compute_uv=False)[:4]
    return float(np.abs(l.s - full).max())


def bm25_manual(docs, query, k1=1.5, b=0.75):
    N, avg = len(docs), sum(map(len, docs)) / len(docs)
    out = []
    for d in docs:
        s = 0.0
        for t in set(query):
            n = sum(t in x for x in docs)
            if n == 0:
                continue
            tf = d.count(t)
            s += math.log(1 + (N - n + 0.5) / (n + 0.5)) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return np.array(out)


def m_bm25():
    docs = [textnorm.words(c) for c in CORPUS]
    bm = R.BM25().fit(docs)
    err = 0.0
    for q in QUERIES:
        toks = textnorm.words(q)
        err = max(err, float(np.abs(bm.scores({t: 1.0 for t in set(toks)}) - bm25_manual(docs, toks)).max()))
    return err


def m_ci():
    x = np.random.default_rng(0).normal(size=12)
    m, h = metrics.mean_ci(x)
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    return max(abs(m - h - lo), abs(m + h - hi))


def m_lev():
    pairs = [("kitten", "sitting", 3), ("flaw", "lawn", 2), ("", "abc", 3), ("same", "same", 0)]
    return sum(textnorm.edit_distance(a, b) != d for a, b, d in pairs)


def m_quant():
    X = np.random.default_rng(1).normal(size=(50, 32))
    Q, sc = quantize_symmetric(X, 8)
    return float((np.abs(dequantize(Q, sc) - X) - sc / 2).max())


def measure():
    path, p = m_viterbi_textbook()
    return [("SI prefix scales vs scipy.constants", "scipy.constants", f"max rel err {m_prefixes():.1e}"),
            ("Viterbi path, textbook HMM (obs normal,cold,dizzy)", "Healthy,Healthy,Fever; P=0.01512", f"path {path}, P={p:.5f}"),
            ("TF-IDF cosine scores", "scikit-learn TfidfVectorizer", f"max abs err {m_tfidf():.1e}"),
            ("LSA singular values", "numpy.linalg.svd (dense)", f"max abs err {m_svd():.1e}"),
            ("BM25 scores (Lucene idf)", "plain-python reference formula", f"max abs err {m_bm25():.1e}"),
            ("Student-t CI half-width", "scipy.stats.t.interval", f"max abs err {m_ci():.1e}"),
            ("Levenshtein distance", "4 textbook pairs", f"{m_lev()} mismatches"),
            ("int8 quantisation error", "bound scale/2", f"max excess {m_quant():.1e} (<=0 means bound holds)")]


def test_prefix_scales():
    assert m_prefixes() < 1e-12


def test_viterbi_textbook_example():
    path, p = m_viterbi_textbook()
    assert path == [0, 0, 1] and p == pytest.approx(0.01512, rel=1e-6)


def test_tfidf_matches_sklearn():
    assert m_tfidf() < 1e-12


def test_lsa_singular_values_match_numpy():
    assert m_svd() < 1e-9


def test_bm25_matches_reference_formula():
    assert m_bm25() < 1e-12


def test_ci_matches_scipy():
    assert m_ci() < 1e-12


def test_levenshtein_pairs():
    assert m_lev() == 0


def test_quantisation_bound():
    assert m_quant() <= 1e-12


if __name__ == "__main__":
    print("| check | reference | measured |\n|---|---|---|")
    for a, b, c in measure():
        print(f"| {a} | {b} | {c} |")
