"""Pruning: unstructured magnitude masks, structured neuron removal, sparse storage cost."""
from __future__ import annotations

import math

import numpy as np

from .model import MLP, fit
from .quant import packed_bytes


def magnitude_masks(model: MLP, sparsity: float, scope: str = "global"):
    """Binary masks that zero the smallest-|w| fraction of weights.

    scope="global": one threshold across layers (lets the net decide where to cut).
    scope="layer" : the same sparsity in every layer.
    """
    if sparsity <= 0:
        return [np.ones_like(W) for W in model.W]
    if scope == "global":
        allw = np.concatenate([np.abs(W).ravel() for W in model.W])
        k = int(sparsity * allw.size)
        thr = np.partition(allw, k)[k] if k < allw.size else np.inf
        return [(np.abs(W) >= thr).astype(np.float64) for W in model.W]
    if scope == "layer":
        out = []
        for W in model.W:
            a = np.abs(W).ravel()
            k = int(sparsity * a.size)
            thr = np.partition(a, k)[k] if k < a.size else np.inf
            out.append((np.abs(W) >= thr).astype(np.float64))
        return out
    raise ValueError(f"unknown scope {scope!r}")


def apply_masks(model: MLP, masks) -> MLP:
    m = model.copy()
    for W, M in zip(m.W, masks):
        W *= M
    return m


def prune_and_finetune(model: MLP, X, y, sparsity: float, scope: str = "global",
                       epochs: int = 15, lr: float = 1e-3, seed: int = 0):
    """Prune then (optionally, epochs>0) retrain with masks held fixed."""
    masks = magnitude_masks(model, sparsity, scope)
    pruned = apply_masks(model, masks)
    if epochs > 0:
        fit(pruned, X, y, epochs=epochs, lr=lr, seed=seed, masks=masks)
    return pruned, masks


def sparsity_of(model: MLP) -> float:
    z = sum(int(np.sum(W == 0)) for W in model.W)
    return z / sum(W.size for W in model.W)


def structured_prune(model: MLP, keep_frac: float) -> MLP:
    """Remove whole hidden neurons (columns of W_i, rows of W_{i+1}) with the least importance.

    Importance = ||incoming weights|| * ||outgoing weights||. The result is a smaller *dense*
    network, so latency drops on any MCU without sparse-index overhead.
    """
    m = model.copy()
    for i in range(m.n_layers - 1):
        imp = np.linalg.norm(m.W[i], axis=0) * np.linalg.norm(m.W[i + 1], axis=1)
        n_keep = max(2, int(round(keep_frac * len(imp))))
        keep = np.sort(np.argsort(-imp)[:n_keep])
        m.W[i] = m.W[i][:, keep]
        m.b[i] = m.b[i][keep]
        m.W[i + 1] = m.W[i + 1][keep, :]
    m.sizes = [m.W[0].shape[0]] + [W.shape[1] for W in m.W]
    return m


def layer_storage_bytes(Wq_or_mask: np.ndarray, wbits: int) -> tuple[int, str]:
    """Cheapest of dense bit-packed storage and CSR (value + column index + row pointers).

    Returns (bytes, format). Rows are the input dimension; column index needs ceil(log2(out)) bits.
    """
    rows, cols = Wq_or_mask.shape
    nnz = int(np.count_nonzero(Wq_or_mask))
    dense = packed_bytes(rows * cols, wbits)
    idx_bits = max(1, math.ceil(math.log2(max(cols, 2))))
    csr = packed_bytes(nnz, wbits + idx_bits) + 2 * (rows + 1)
    return (dense, "dense") if dense <= csr else (csr, "csr")
