"""Layer 2: causal feature extraction for regime detection (filters -> feature matrix)."""
from __future__ import annotations

import numpy as np

from .filters import ewma_vol, rolling_band_fraction, rolling_std, sma

FEATURE_NAMES = ("ret", "logvol", "mom", "spec")


class FeatureBuilder:
    """Builds standardised features from a return series.

    ret     r_t in percent
    logvol  log EWMA volatility (RiskMetrics)
    mom     rolling mean return divided by rolling std (a t-statistic-like trend score)
    spec    fraction of windowed spectral power in the lowest eighth of the band (momentum vs mean reversion)
    """

    def __init__(self, names, window: int = 64, lam: float = 0.94):
        names = tuple(names)
        bad = [n for n in names if n not in FEATURE_NAMES]
        if bad or not names:
            raise ValueError(f"unknown/empty feature names: {bad}")
        self.names, self.window, self.lam = names, window, lam
        self.mean_ = self.std_ = None

    def raw(self, returns: np.ndarray) -> np.ndarray:
        r = np.asarray(returns, float)
        cols = []
        for n in self.names:
            if n == "ret":
                cols.append(100.0 * r)
            elif n == "logvol":
                cols.append(np.log(ewma_vol(r, self.lam)))
            elif n == "mom":
                cols.append(sma(r, self.window) / np.maximum(rolling_std(r, self.window), 1e-8))
            else:
                cols.append(rolling_band_fraction(r, self.window))
        return np.column_stack(cols)

    def fit(self, returns: np.ndarray) -> "FeatureBuilder":
        X = self.raw(returns)
        self.mean_, self.std_ = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-12)
        return self

    def transform(self, returns: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("fit first")
        return (self.raw(returns) - self.mean_) / self.std_

    def fit_transform(self, returns: np.ndarray) -> np.ndarray:
        return self.fit(returns).transform(returns)
