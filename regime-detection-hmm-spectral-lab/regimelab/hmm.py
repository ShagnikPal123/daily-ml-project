"""Layer 1: Gaussian hidden Markov model (diagonal covariance) with scaled forward-backward,
Baum-Welch EM, Viterbi decoding and causal filtering. Pure numpy apart from k-means initialisation."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class GaussianHMM:
    def __init__(self, k: int, n_iter: int = 30, tol: float = 1e-5, seed: int = 0, min_var: float = 1e-8):
        self.k, self.n_iter, self.tol, self.seed, self.min_var = k, n_iter, tol, seed, min_var
        self.startprob = np.full(k, 1.0 / k)
        self.trans = np.full((k, k), 1.0 / k)
        self.means = self.vars = None
        self.loglik_history: list[float] = []

    # ---------- emissions ----------
    def log_emission(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.means[None]
        return -0.5 * (np.sum(np.log(2 * np.pi * self.vars), axis=1)[None]
                       + np.sum(diff ** 2 / self.vars[None], axis=2))

    @staticmethod
    def _scale(logB: np.ndarray):
        m = logB.max(axis=1)
        return np.exp(logB - m[:, None]), m

    # ---------- inference ----------
    def _forward(self, B: np.ndarray, startprob=None):
        T, k = B.shape
        pi = self.startprob if startprob is None else startprob
        alpha, c = np.empty((T, k)), np.empty(T)
        a = pi * B[0]
        c[0] = max(a.sum(), 1e-300)
        alpha[0] = a / c[0]
        A = self.trans
        for t in range(1, T):
            a = (alpha[t - 1] @ A) * B[t]
            c[t] = max(a.sum(), 1e-300)
            alpha[t] = a / c[t]
        return alpha, c

    def _backward(self, B: np.ndarray, c: np.ndarray):
        T, k = B.shape
        beta = np.ones((T, k))
        for t in range(T - 2, -1, -1):
            beta[t] = self.trans @ (B[t + 1] * beta[t + 1]) / c[t + 1]
        return beta

    def score(self, X: np.ndarray) -> float:
        B, m = self._scale(self.log_emission(X))
        _, c = self._forward(B)
        return float(np.log(c).sum() + m.sum())

    def filter_posterior(self, X: np.ndarray, startprob=None) -> np.ndarray:
        """P(s_t | x_1..t): strictly causal posterior."""
        B, _ = self._scale(self.log_emission(X))
        return self._forward(B, startprob)[0]

    def smooth_posterior(self, X: np.ndarray) -> np.ndarray:
        B, _ = self._scale(self.log_emission(X))
        alpha, c = self._forward(B)
        g = alpha * self._backward(B, c)
        return g / g.sum(axis=1, keepdims=True)

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        logB = self.log_emission(X)
        T, k = logB.shape
        logA = np.log(np.maximum(self.trans, 1e-300))
        delta = np.log(np.maximum(self.startprob, 1e-300)) + logB[0]
        psi = np.zeros((T, k), dtype=int)
        for t in range(1, T):
            cand = delta[:, None] + logA
            psi[t] = cand.argmax(axis=0)
            delta = cand.max(axis=0) + logB[t]
        path = np.empty(T, dtype=int)
        path[-1] = int(delta.argmax())
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def stationary(self) -> np.ndarray:
        w, v = np.linalg.eig(self.trans.T)
        pi = np.abs(np.real(v[:, int(np.argmin(np.abs(w - 1)))]))
        return pi / pi.sum()

    # ---------- learning ----------
    def _init_params(self, X: np.ndarray):
        km = KMeans(self.k, n_init=3, random_state=self.seed).fit(X)
        gv = X.var(axis=0) + self.min_var
        self.means = km.cluster_centers_.copy()
        self.vars = np.empty_like(self.means)
        for j in range(self.k):
            xj = X[km.labels_ == j]
            self.vars[j] = xj.var(axis=0) + self.min_var if len(xj) > 2 else gv
        self.trans = np.full((self.k, self.k), 0.1 / (self.k - 1))
        np.fill_diagonal(self.trans, 0.9)
        self.startprob = np.full(self.k, 1.0 / self.k)

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[:, None]
        self._init_params(X)
        self.loglik_history = []
        prev = -np.inf
        for _ in range(self.n_iter):
            B, m = self._scale(self.log_emission(X))
            alpha, c = self._forward(B)
            ll = float(np.log(c).sum() + m.sum())
            self.loglik_history.append(ll)
            if np.isfinite(prev) and (ll - prev) / len(X) < self.tol:
                break
            prev = ll
            beta = self._backward(B, c)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = self.trans * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
            nk = gamma.sum(axis=0)
            self.startprob = np.maximum(gamma[0], 1e-12)
            self.startprob /= self.startprob.sum()
            tr = np.maximum(xi, 1e-12)
            self.trans = tr / tr.sum(axis=1, keepdims=True)
            ok = nk > 1e-6
            means = (gamma.T @ X) / np.maximum(nk, 1e-12)[:, None]
            var = np.empty_like(self.vars)
            for j in range(self.k):
                var[j] = (gamma[:, j, None] * (X - means[j]) ** 2).sum(axis=0) / max(nk[j], 1e-12)
            self.means = np.where(ok[:, None], means, self.means)
            self.vars = np.where(ok[:, None], np.maximum(var, self.min_var), self.vars)
        return self

    def permute(self, order) -> "GaussianHMM":
        order = np.asarray(order)
        self.means, self.vars = self.means[order], self.vars[order]
        self.trans = self.trans[np.ix_(order, order)]
        self.startprob = self.startprob[order]
        return self


def brute_force_loglik(hmm: GaussianHMM, X: np.ndarray) -> float:
    """Exact likelihood by enumerating all k^T state paths (for validation, tiny T only)."""
    import itertools
    logB = hmm.log_emission(X)
    T, k = logB.shape
    terms = []
    for path in itertools.product(range(k), repeat=T):
        lp = np.log(hmm.startprob[path[0]]) + logB[0, path[0]]
        for t in range(1, T):
            lp += np.log(hmm.trans[path[t - 1], path[t]]) + logB[t, path[t]]
        terms.append(lp)
    return float(logsumexp(terms))
