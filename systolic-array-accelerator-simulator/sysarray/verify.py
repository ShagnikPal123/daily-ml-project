"""Self-checks: simulator vs numpy, closed-form models vs simulator."""
from __future__ import annotations

import numpy as np

from .tiling import analytic_stats, analytic_traffic, tiled_matmul


def random_int8(rng, shape, density: float = 1.0) -> np.ndarray:
    a = rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8)
    if density < 1.0:
        a = np.where(rng.random(shape) < density, a, 0).astype(np.int8)
    return a


def verify(n_cases: int = 24, seed: int = 0, max_dim: int = 40) -> dict:
    rng = np.random.default_rng(seed)
    rec = {"cases": 0, "exact": 0, "cycle_match": 0, "mac_match": 0, "traffic_match": 0}
    for _ in range(n_cases):
        M, K, N = (int(v) for v in rng.integers(1, max_dim, size=3))
        r, c = (int(v) for v in rng.choice([2, 3, 4, 8], size=2))
        A, B = random_int8(rng, (M, K)), random_int8(rng, (K, N))
        ref = A.astype(np.int64) @ B.astype(np.int64)
        for df in ("OS", "WS"):
            C, st, tr = tiled_matmul(A, B, r, c, df)
            an = analytic_stats(M, K, N, r, c, df)
            rec["cases"] += 1
            rec["exact"] += int(np.array_equal(C, ref))
            rec["cycle_match"] += int(st.cycles == an.cycles)
            rec["mac_match"] += int(st.macs == M * K * N)
            rec["traffic_match"] += int(tr == analytic_traffic(M, K, N, r, c, df))
    rec["all_passed"] = all(rec[k] == rec["cases"] for k in ("exact", "cycle_match", "mac_match", "traffic_match"))
    return rec
