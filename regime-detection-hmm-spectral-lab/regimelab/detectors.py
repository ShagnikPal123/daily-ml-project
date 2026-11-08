"""Layer 3: pluggable regime detectors. Every detector exposes fit(train_returns) and
predict(test_returns) -> integer labels ordered by volatility (0 = calmest). All except
HMM-Viterbi are causal: label[t] depends on returns[:t+1] only."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .features import FeatureBuilder
from .filters import ewma_vol
from .hmm import GaussianHMM


class Detector:
    name = "base"
    causal = True

    def __init__(self, k: int):
        self.k = k

    def fit(self, returns: np.ndarray) -> "Detector":
        raise NotImplementedError

    def predict(self, returns: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ThresholdVolDetector(Detector):
    """Baseline 1: bucket EWMA volatility by training quantiles (assumes equal regime occupancy)."""
    name = "threshold"

    def __init__(self, k: int, lam: float = 0.94):
        super().__init__(k)
        self.lam = lam

    def fit(self, returns):
        v = ewma_vol(returns, self.lam)
        self.edges_ = np.quantile(v, np.linspace(0, 1, self.k + 1)[1:-1])
        return self

    def predict(self, returns):
        return np.digitize(ewma_vol(returns, self.lam), self.edges_)


class KMeansDetector(Detector):
    """Baseline 2: k-means on standardised features; no temporal model, per-sample assignment."""
    name = "kmeans"

    def __init__(self, k: int, features=("logvol", "spec"), window: int = 64, lam: float = 0.94, seed: int = 0):
        super().__init__(k)
        self.fb = FeatureBuilder(features, window, lam)
        self.seed = seed

    def fit(self, returns):
        X = self.fb.fit_transform(returns)
        self.km_ = KMeans(self.k, n_init=5, random_state=self.seed).fit(X)
        col = self.fb.names.index("logvol")
        self.order_ = np.argsort(self.km_.cluster_centers_[:, col])
        self.rank_ = np.empty(self.k, dtype=int)
        self.rank_[self.order_] = np.arange(self.k)
        return self

    def predict(self, returns):
        return self.rank_[self.km_.predict(self.fb.transform(returns))]


class HMMDetector(Detector):
    """Gaussian HMM on a feature set. mode='filter' is causal; mode='viterbi' decodes offline."""

    def __init__(self, k: int, features=("ret",), mode: str = "filter", window: int = 64,
                 lam: float = 0.94, n_iter: int = 30, seed: int = 0, name: str | None = None):
        super().__init__(k)
        if mode not in ("filter", "viterbi"):
            raise ValueError("mode must be 'filter' or 'viterbi'")
        self.mode, self.n_iter, self.seed = mode, n_iter, seed
        self.fb = FeatureBuilder(features, window, lam)
        self.causal = mode == "filter"
        self.name = name or f"hmm-{mode}:{'+'.join(features)}"

    def fit(self, returns):
        X = self.fb.fit_transform(returns)
        self.hmm_ = GaussianHMM(self.k, n_iter=self.n_iter, seed=self.seed).fit(X)
        if "logvol" in self.fb.names:
            key = self.hmm_.means[:, self.fb.names.index("logvol")]
        else:
            key = self.hmm_.vars[:, 0]
        self.hmm_.permute(np.argsort(key))
        return self

    def predict(self, returns):
        X = self.fb.transform(returns)
        if self.mode == "viterbi":
            return self.hmm_.viterbi(X)
        post = self.hmm_.filter_posterior(X, startprob=self.hmm_.stationary())
        return post.argmax(axis=1)

    def posterior(self, returns):
        return self.hmm_.filter_posterior(self.fb.transform(returns), startprob=self.hmm_.stationary())


DETECTOR_NAMES = ("threshold", "kmeans", "hmm-ret", "hmm-feat", "hmm-viterbi")


def build_detector(name: str, k: int, window: int = 64, lam: float = 0.94, n_iter: int = 30, seed: int = 0) -> Detector:
    """Factory. 'hmm:<f1>+<f2>' builds a causal HMM on an arbitrary feature set (used by ablations)."""
    kw = dict(window=window, lam=lam)
    if name == "threshold":
        return ThresholdVolDetector(k, lam)
    if name == "kmeans":
        return KMeansDetector(k, seed=seed, **kw)
    if name == "hmm-ret":
        return HMMDetector(k, ("ret",), "filter", n_iter=n_iter, seed=seed, name=name, **kw)
    if name == "hmm-feat":
        return HMMDetector(k, ("ret", "logvol"), "filter", n_iter=n_iter, seed=seed, name=name, **kw)
    if name == "hmm-viterbi":
        return HMMDetector(k, ("ret", "logvol"), "viterbi", n_iter=n_iter, seed=seed, name=name, **kw)
    if name.startswith("hmm:"):
        feats = tuple(name[4:].split("+"))
        return HMMDetector(k, feats, "filter", n_iter=n_iter, seed=seed, name=name, **kw)
    raise ValueError(f"unknown detector {name!r}")
