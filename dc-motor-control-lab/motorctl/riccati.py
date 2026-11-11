"""Layer 0: discrete algebraic Riccati equation via the structured doubling algorithm (SDA).

P = A'PA - A'PB (R + B'PB)^-1 B'PA + Q.  SDA converges quadratically, which matters because the
integral-augmented plant has a unit-circle eigenvalue and plain value iteration crawls there.
"""
from __future__ import annotations

import numpy as np


def solve_dare(A, B, Q, R, tol: float = 1e-13, max_iter: int = 200):
    """Return the stabilising solution P of the DARE."""
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    Q = np.atleast_2d(np.asarray(Q, float))
    R = np.atleast_2d(np.asarray(R, float))
    n = A.shape[0]
    I = np.eye(n)
    Ak = A.copy()
    Gk = B @ np.linalg.solve(R, B.T)
    Hk = Q.copy()
    for _ in range(max_iter):
        W = np.linalg.solve(I + Gk @ Hk, I)  # (I + G H)^-1
        A_next = Ak @ W @ Ak
        G_next = Gk + Ak @ W @ Gk @ Ak.T
        H_next = Hk + Ak.T @ Hk @ W @ Ak
        delta = np.linalg.norm(H_next - Hk, 1)
        Ak, Gk, Hk = A_next, 0.5 * (G_next + G_next.T), 0.5 * (H_next + H_next.T)
        if delta <= tol * max(1.0, np.linalg.norm(Hk, 1)):
            return Hk
    raise RuntimeError("DARE did not converge (system not stabilisable/detectable?)")


def dare_residual(A, B, Q, R, P) -> float:
    """Max-abs residual of the DARE for a candidate P."""
    A, B, Q, R, P = (np.atleast_2d(np.asarray(m, float)) for m in (A, B, Q, R, P))
    gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    res = A.T @ P @ A - A.T @ P @ B @ gain + Q - P
    return float(np.max(np.abs(res)))


def lqr_gain(A, B, Q, R):
    """Return (K, P) for u = -K x minimising sum x'Qx + u'Ru."""
    A, B, R = (np.atleast_2d(np.asarray(m, float)) for m in (A, B, R))
    P = solve_dare(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P
