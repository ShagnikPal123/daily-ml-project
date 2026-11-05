"""A small numpy MLP with Adam training and optional pruning masks."""
from __future__ import annotations

import numpy as np


class MLP:
    """Fully connected ReLU network. Weights stored as (in, out) so y = x @ W + b."""

    def __init__(self, sizes, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    @property
    def n_layers(self) -> int:
        return len(self.W)

    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def copy(self) -> "MLP":
        m = MLP.__new__(MLP)
        m.sizes = list(self.sizes)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        return m

    def forward(self, X: np.ndarray, return_acts: bool = False):
        h = X
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < self.n_layers - 1 else z
            acts.append(h)
        return (h, acts) if return_acts else h

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def macs(self) -> int:
        return sum(w.size for w in self.W)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(model: MLP, X, y, epochs: int = 60, lr: float = 3e-3, batch: int = 32,
        l2: float = 1e-4, seed: int = 0, masks=None) -> list[float]:
    """Train with Adam on cross-entropy. If `masks` is given, pruned weights stay zero."""
    rng = np.random.default_rng(seed)
    n, k = len(X), model.sizes[-1]
    Y = np.eye(k)[y]
    params = model.W + model.b
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    losses = []
    L = model.n_layers
    if masks is not None:
        for W, M in zip(model.W, masks):
            W *= M
    for _ in range(epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb, yb = X[idx], Y[idx]
            logits, acts = model.forward(xb, return_acts=True)
            p = _softmax(logits)
            ep_loss += -np.sum(yb * np.log(p + 1e-12))
            d = (p - yb) / len(idx)
            gW, gb = [None] * L, [None] * L
            for i in range(L - 1, -1, -1):
                gW[i] = acts[i].T @ d + l2 * model.W[i]
                gb[i] = d.sum(axis=0)
                if i > 0:
                    d = (d @ model.W[i].T) * (acts[i] > 0)
            grads = gW + gb
            t += 1
            for j, (p_, g) in enumerate(zip(params, grads)):
                m[j] = 0.9 * m[j] + 0.1 * g
                v[j] = 0.999 * v[j] + 0.001 * g * g
                mh = m[j] / (1 - 0.9 ** t)
                vh = v[j] / (1 - 0.999 ** t)
                p_ -= lr * mh / (np.sqrt(vh) + 1e-8)
            if masks is not None:
                for W, M in zip(model.W, masks):
                    W *= M
        losses.append(ep_loss / n)
    return losses
