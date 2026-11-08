"""Layer 1: from-scratch classifiers, including an integer-only int8 MLP.

* ``SoftmaxRegression`` - multinomial logistic regression, L-BFGS, analytic gradient.
* ``MLP``               - one-hidden-layer ReLU network trained with Adam.
* ``QuantizedMLP``      - post-training quantized MLP evaluated with integer GEMMs
                          (int8 weights/activations, int32 accumulators, requantization).
Inputs are flattened pixels in [0, 1].
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import fixedpoint as fp


def softmax_loss_grad(theta: np.ndarray, X: np.ndarray, Y1: np.ndarray, l2: float):
    """Mean cross-entropy + (l2/2)||W||^2 and its gradient wrt theta=[W.ravel(), b]."""
    n, d = X.shape
    k = Y1.shape[1]
    W, b = theta[: d * k].reshape(d, k), theta[d * k:]
    Z = X @ W + b
    logp = Z - logsumexp(Z, axis=1, keepdims=True)
    loss = -np.mean(np.sum(Y1 * logp, axis=1)) + 0.5 * l2 * np.sum(W * W)
    G = (np.exp(logp) - Y1) / n
    return loss, np.concatenate([(X.T @ G + l2 * W).ravel(), G.sum(0)])


class _Base:
    classes_: np.ndarray

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def predict_proba(self, X) -> np.ndarray:
        z = self.decision_function(X)
        return np.exp(z - logsumexp(z, axis=1, keepdims=True))


class SoftmaxRegression(_Base):
    def __init__(self, l2: float = 1e-3, max_iter: int = 300, seed: int = 0):
        self.l2, self.max_iter, self.seed = l2, max_iter, seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        Y1 = (np.asarray(y)[:, None] == self.classes_[None]).astype(float)
        d, k = X.shape[1], len(self.classes_)
        res = minimize(softmax_loss_grad, np.zeros(d * k + k), args=(X, Y1, self.l2), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.W = res.x[: d * k].reshape(d, k)
        self.b = res.x[d * k:]
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, float) @ self.W + self.b

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size

    def macs(self) -> int:
        return self.W.size

    def weight_bytes(self, bits: int = 32) -> int:
        return int(np.ceil(self.W.size * bits / 8)) + self.b.size * 4


class MLP(_Base):
    def __init__(self, hidden: int = 64, epochs: int = 60, lr: float = 3e-3, batch: int = 64,
                 l2: float = 1e-4, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.batch, self.l2, self.seed = hidden, epochs, lr, batch, l2, seed

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        Y1 = (np.asarray(y)[:, None] == self.classes_[None]).astype(float)
        n, d = X.shape
        k, H = len(self.classes_), self.hidden
        rng = np.random.default_rng(self.seed)
        P = [rng.standard_normal((d, H)) * np.sqrt(2 / d), np.zeros(H),
             rng.standard_normal((H, k)) * np.sqrt(2 / H), np.zeros(k)]
        m = [np.zeros_like(p) for p in P]
        v = [np.zeros_like(p) for p in P]
        t = 0
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch):
                ib = perm[s: s + self.batch]
                xb, yb = X[ib], Y1[ib]
                W1, b1, W2, b2 = P
                z1 = xb @ W1 + b1
                h = np.maximum(z1, 0)
                z2 = h @ W2 + b2
                p = np.exp(z2 - logsumexp(z2, axis=1, keepdims=True))
                dz2 = (p - yb) / len(ib)
                dh = dz2 @ W2.T
                dz1 = dh * (z1 > 0)
                grads = [xb.T @ dz1 + self.l2 * W1, dz1.sum(0), h.T @ dz2 + self.l2 * W2, dz2.sum(0)]
                t += 1
                for i, g in enumerate(grads):
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    P[i] = P[i] - self.lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
        self.W1, self.b1, self.W2, self.b2 = P
        return self

    def hidden_activations(self, X) -> np.ndarray:
        return np.maximum(np.asarray(X, float) @ self.W1 + self.b1, 0)

    def decision_function(self, X) -> np.ndarray:
        return self.hidden_activations(X) @ self.W2 + self.b2

    @property
    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def macs(self) -> int:
        return self.W1.size + self.W2.size

    def weight_bytes(self, bits: int = 32) -> int:
        return int(np.ceil((self.W1.size + self.W2.size) * bits / 8)) + (self.b1.size + self.b2.size) * 4


class QuantizedMLP(_Base):
    """Integer-only inference for a trained :class:`MLP` (post-training quantization)."""

    def __init__(self, mlp: MLP, X_calib, bits: int = 8):
        self.bits = bits
        self.classes_ = mlp.classes_
        qmax_u = fp.qrange(bits, signed=False)[1]
        self.sx = 1.0 / qmax_u                       # pixels live in [0, 1]
        self.sw1, self.sw2 = fp.symmetric_scale(mlp.W1, bits), fp.symmetric_scale(mlp.W2, bits)
        hmax = float(np.max(mlp.hidden_activations(X_calib)))
        self.sh = (hmax if hmax > 0 else 1.0) / qmax_u
        self.W1q = fp.quantize_symmetric(mlp.W1, self.sw1, bits)
        self.W2q = fp.quantize_symmetric(mlp.W2, self.sw2, bits)
        self.b1q = np.rint(mlp.b1 / (self.sx * self.sw1)).astype(np.int64)
        self.b2q = np.rint(mlp.b2 / (self.sh * self.sw2)).astype(np.int64)

    def decision_function(self, X) -> np.ndarray:
        xq = fp.quantize_unsigned(np.clip(X, 0.0, 1.0), self.sx, self.bits)
        acc1 = fp.int_matmul(xq, self.W1q) + self.b1q
        hq = fp.requantize(np.maximum(acc1, 0), self.sx * self.sw1 / self.sh, self.bits, signed=False)
        acc2 = fp.int_matmul(hq, self.W2q) + self.b2q
        return acc2 * (self.sh * self.sw2)

    def weight_bytes(self) -> int:
        return int(np.ceil((self.W1q.size + self.W2q.size) * self.bits / 8)) + (self.b1q.size + self.b2q.size) * 4

    def macs(self) -> int:
        return self.W1q.size + self.W2q.size

    @property
    def n_params(self) -> int:
        return self.W1q.size + self.W2q.size + self.b1q.size + self.b2q.size


class QuantizedMLPClassifier(_Base):
    """fit() trains a float MLP and then quantizes it with the training set as calibration."""

    def __init__(self, bits: int = 8, **mlp_kw):
        self.bits, self.mlp_kw = bits, mlp_kw

    def fit(self, X, y):
        self.float_model = MLP(**self.mlp_kw).fit(X, y)
        self.q = QuantizedMLP(self.float_model, X, self.bits)
        self.classes_ = self.q.classes_
        return self

    def decision_function(self, X) -> np.ndarray:
        return self.q.decision_function(X)

    def macs(self) -> int:
        return self.q.macs()

    def weight_bytes(self, bits: int | None = None) -> int:
        return self.q.weight_bytes()

    @property
    def n_params(self) -> int:
        return self.q.n_params


def make_classifier(name: str, seed: int = 0, **kw):
    if name == "softmax":
        return SoftmaxRegression(seed=seed, **kw)
    if name == "mlp":
        return MLP(seed=seed, **kw)
    if name.startswith("mlp_int"):
        return QuantizedMLPClassifier(bits=int(name[len("mlp_int"):]), seed=seed, **kw)
    raise ValueError(f"unknown classifier {name!r}")
