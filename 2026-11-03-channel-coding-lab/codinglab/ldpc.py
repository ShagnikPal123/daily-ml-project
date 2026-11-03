"""Regular (dv, dc) LDPC code: Gallager-style construction with 4-cycle avoidance,
GF(2) systematic encoder and Tanner-graph index tables used by the decoders."""
import numpy as np
from . import gf2


def _overlap(H):
    A = H.astype(np.float32)
    o = A @ A.T
    np.fill_diagonal(o, 0)
    return o


def has_four_cycle(H):
    return bool((_overlap(H) > 1).any())


def _cycle_count(H):
    o = _overlap(H)
    return float((o * (o - 1) / 2)[o > 1].sum())


def gallager_H(n, dv=3, dc=6, seed=0, iters=4000):
    """Gallager construction: dv stacked bands, each band a column permutation of the first.

    Band 0 has row i covering columns [i*dc, (i+1)*dc).  Each later band is a random column
    permutation, then repaired by a randomised local search: pick a column shared by two rows
    that overlap in >1 columns (a 4-cycle) and swap it with a random column; keep the swap if
    the 4-cycle count does not increase.  Row/column weights stay regular by construction.
    """
    if n % dc:
        raise ValueError("n must be a multiple of dc")
    rng = np.random.default_rng(seed)
    m_band = n // dc
    band0 = np.zeros((m_band, n), dtype=np.uint8)
    for i in range(m_band):
        band0[i, i * dc:(i + 1) * dc] = 1
    H = band0
    for _ in range(1, dv):
        perm = rng.permutation(n)
        cur = _cycle_count(np.vstack([H, band0[:, perm]]))
        for _ in range(iters):
            if cur == 0:
                break
            full = np.vstack([H, band0[:, perm]])
            o = _overlap(full)
            r1, r2 = np.argwhere(np.triu(o > 1, 1))[rng.integers(0, int((np.triu(o > 1, 1)).sum()))]
            shared = np.nonzero(full[r1] & full[r2])[0]
            c = rng.choice(shared)
            j = rng.integers(0, n)
            pos_c, pos_j = np.nonzero(perm == c)[0][0], np.nonzero(perm == j)[0][0]
            perm[[pos_c, pos_j]] = perm[[pos_j, pos_c]]
            new = _cycle_count(np.vstack([H, band0[:, perm]]))
            if new <= cur:
                cur = new
            else:
                perm[[pos_c, pos_j]] = perm[[pos_j, pos_c]]
        H = np.vstack([H, band0[:, perm]])
    return H


class LDPCCode:
    def __init__(self, n=504, dv=3, dc=6, seed=0):
        self.H = gallager_H(n, dv, dc, seed)
        self.m, self.n = self.H.shape
        self.dv, self.dc = dv, dc
        self.G, self.info_cols, self.parity_cols = gf2.systematic_generator(self.H)
        self.k = self.G.shape[0]
        self.rate = self.k / self.n
        self.design_rate = 1 - dv / dc
        # Tanner graph tables. Edge e = c*dc + j  <->  (check c, slot j)
        self.Cn = np.zeros((self.m, dc), dtype=np.int64)
        for c in range(self.m):
            self.Cn[c] = np.nonzero(self.H[c])[0]
        self.Vn = np.zeros((self.n, dv), dtype=np.int64)        # edge ids per variable
        fill = np.zeros(self.n, dtype=int)
        for c in range(self.m):
            for j in range(dc):
                v = self.Cn[c, j]
                self.Vn[v, fill[v]] = c * dc + j
                fill[v] += 1

    def encode(self, info_bits):
        return gf2.encode(self.G, info_bits)

    def extract_info(self, codeword):
        return codeword[:, self.info_cols]

    def syndrome(self, bits):
        return (np.asarray(bits, dtype=np.uint8) @ self.H.T) % 2

    def girth_four_free(self):
        return not has_four_cycle(self.H)

    def num_edges(self):
        return self.m * self.dc
