"""Layer 3: memory-budget angle -- quantise LSA document embeddings to a few bits and measure retrieval."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .metrics import mrr, recall_at_k
from .retrieval import LSA, rank_desc, rank_of


def quantize_symmetric(X: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric uniform quantisation to signed ``bits``-bit integers (stored in int8)."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(X).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    Q = np.clip(np.round(X / scale), -qmax, qmax).astype(np.int8)
    return Q, scale


def dequantize(Q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return Q.astype(float) * scale


def dense_index_bytes(n: int, k: int, bits: int) -> float:
    """Packed storage: n*k*bits/8 plus one float32 scale per row (float32 baseline: bits=32, no scale)."""
    return n * k * bits / 8 + (0 if bits == 32 else 4 * n)


def sparse_index_bytes(M) -> int:
    """CSR with float32 values and int32 indices."""
    return int(M.nnz * 8 + (M.shape[0] + 1) * 4)


def eval_compressed_lsa(lsa: LSA, query_weights: Sequence[Dict[str, float]], targets: Sequence[int], bits_list: Sequence[int]) -> List[Dict]:
    out = []
    for bits in bits_list:
        if bits >= 32:
            D = lsa.D
        else:
            Q, sc = quantize_symmetric(lsa.D, bits)
            D = dequantize(Q, sc)
        Dn = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
        ranks = []
        for w, t in zip(query_weights, targets):
            q = lsa.embed_query(w)
            n = np.linalg.norm(q)
            s = Dn @ (q / n) if n else np.zeros(len(Dn))
            ranks.append(rank_of(rank_desc(s), t))
        out.append({"bits": bits, "bytes": dense_index_bytes(D.shape[0], D.shape[1], bits),
                    "recall@1": recall_at_k(ranks, 1), "recall@5": recall_at_k(ranks, 5), "mrr": mrr(ranks)})
    return out
