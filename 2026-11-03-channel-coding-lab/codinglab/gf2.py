"""GF(2) linear algebra for LDPC encoding (dense uint8 matrices)."""
import numpy as np


def rref(M):
    """Reduced row echelon form over GF(2). Returns (R, pivot_columns)."""
    A = (np.asarray(M, dtype=np.uint8) & 1).copy()
    rows, cols = A.shape
    pivots, r = [], 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        mask = A[:, c].astype(bool)
        mask[r] = False
        A[mask] ^= A[r]
        pivots.append(c)
        r += 1
    return A[:r], pivots


def rank(M):
    return len(rref(M)[1])


def systematic_generator(H):
    """Build a generator matrix G (k x n) with G H^T = 0 and an information-position list.

    Uses RREF of H: pivot columns are parity bits, free columns carry information.
    Returns (G, info_cols, parity_cols).
    """
    R, piv = rref(H)
    n = H.shape[1]
    free = [c for c in range(n) if c not in set(piv)]
    k = len(free)
    G = np.zeros((k, n), dtype=np.uint8)
    for i, f in enumerate(free):
        G[i, f] = 1
        # parity bit in pivot row r equals sum over free columns of R[r, f] * info bit
        for r, p in enumerate(piv):
            if R[r, f]:
                G[i, p] = 1
    return G, free, piv


def encode(G, info_bits):
    """info_bits: (B,k) -> codewords (B,n)."""
    return (np.asarray(info_bits, dtype=np.uint8) @ G) % 2
