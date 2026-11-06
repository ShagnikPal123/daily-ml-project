"""Train a small MLP on sklearn digits, quantize it to int8, run it on the simulated array.

All layer arithmetic is integer: int8 x int8 -> int32 accumulate on the array, then a
fixed-point requantize (mult, shift) + ReLU between layers -- exactly what an NPU's
post-processing unit does. The matmul backend is injectable so the same quantized model
runs on numpy (reference) or on the cycle-accurate array, and results must match bit-exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from .quant import quantize_sym, requantize_fixed, to_fixed_multiplier

MatMul = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class FloatMLP:
    weights: list[np.ndarray]   # each (in, out)
    biases: list[np.ndarray]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1:
                h = np.maximum(h, 0)
        return h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=1)


@dataclass
class QLayer:
    Wq: np.ndarray          # int8 (in, out)
    bq: np.ndarray          # int32 (out,)
    mult: int | None        # requantize params (None for the logits layer)
    shift: int | None
    relu: bool


@dataclass
class QMLP:
    layers: list[QLayer]
    s_in: float

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        return quantize_sym(x, self.s_in)[0]

    def predict(self, x: np.ndarray, matmul: MatMul) -> np.ndarray:
        h = self.quantize_input(x)
        for layer in self.layers:
            acc = matmul(h, layer.Wq) + layer.bq.astype(np.int64)
            if layer.mult is None:
                return np.argmax(acc, axis=1)
            h = requantize_fixed(acc, layer.mult, layer.shift, relu=layer.relu)
        raise RuntimeError("model has no logits layer")


def load_data(seed: int = 0):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def train_float(Xtr, ytr, hidden: int = 64, seed: int = 0) -> FloatMLP:
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=400, random_state=seed)
    clf.fit(Xtr, ytr)
    return FloatMLP([w.copy() for w in clf.coefs_], [b.copy() for b in clf.intercepts_])


def prune_magnitude(model: FloatMLP, sparsity: float) -> FloatMLP:
    """Per-layer unstructured magnitude pruning (no fine-tuning)."""
    ws = []
    for w in model.weights:
        if sparsity <= 0:
            ws.append(w.copy())
            continue
        thr = np.quantile(np.abs(w), sparsity)
        ws.append(np.where(np.abs(w) <= thr, 0.0, w))
    return FloatMLP(ws, [b.copy() for b in model.biases])


def quantize_model(model: FloatMLP, Xcal: np.ndarray) -> QMLP:
    """Post-training int8 quantization, activation scales calibrated on `Xcal`."""
    s_in = float(np.max(np.abs(Xcal))) / 127.0
    layers, s_prev, h = [], s_in, Xcal
    n = len(model.weights)
    for i, (w, b) in enumerate(zip(model.weights, model.biases)):
        Wq, s_w = quantize_sym(w)
        bq = np.rint(b / (s_prev * s_w)).astype(np.int64).astype(np.int32)
        h = h @ w + b
        last = i == n - 1
        if last:
            layers.append(QLayer(Wq, bq, None, None, False))
        else:
            h = np.maximum(h, 0)
            s_out = float(np.max(np.abs(h))) / 127.0
            mult, shift = to_fixed_multiplier(s_prev * s_w / s_out)
            layers.append(QLayer(Wq, bq, mult, shift, True))
            s_prev = s_out
    return QMLP(layers, s_in)


def numpy_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.int64) @ b.astype(np.int64)


def accel_matmul_factory(rows: int, cols: int, dataflow: str):
    """Return (matmul, log) where log accumulates stats/traffic of every GEMM issued."""
    from .pe_array import ArrayStats
    from .tiling import Traffic, tiled_matmul

    log = {"stats": ArrayStats(), "traffic": Traffic(), "gemms": []}

    def mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        c, st, tr = tiled_matmul(a, b, rows, cols, dataflow)
        log["stats"] += st
        log["traffic"] += tr
        log["gemms"].append((a.shape[0], a.shape[1], b.shape[1]))
        return c

    return mm, log


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pred == y))
