"""Layer 1: Kalman filters (time-varying and steady-state) for the disturbance-augmented plant."""
from __future__ import annotations

import numpy as np

from motorctl.riccati import solve_dare


def augment_disturbance(Ad, Bd):
    """x_aug = [i, w, d]; d (load torque) is a random-walk state. Returns (Aaug, Bv_aug, Baug_d, C)."""
    n = Ad.shape[0]
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = Ad
    A[:n, n] = Bd[:, 1]
    A[n, n] = 1.0
    Bv = np.vstack([Bd[:, :1], [[0.0]]])
    C = np.zeros((1, n + 1))
    C[0, n - 1] = 1.0
    return A, Bv, C


class KalmanFilter:
    """Textbook time-varying KF with Joseph-form covariance update. State = prior after predict()."""

    def __init__(self, A, B, C, Q, R, x0=None, P0=None):
        self.A, self.B, self.C = (np.atleast_2d(np.asarray(m, float)) for m in (A, B, C))
        self.Q, self.R = np.atleast_2d(Q), np.atleast_2d(R)
        n = self.A.shape[0]
        self.x = np.zeros(n) if x0 is None else np.asarray(x0, float).copy()
        self.P = np.eye(n) if P0 is None else np.asarray(P0, float).copy()

    def update(self, y):
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        self.x = self.x + (K @ (np.atleast_1d(y) - self.C @ self.x)).ravel()
        I_KC = np.eye(len(self.x)) - K @ self.C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R @ K.T
        return self.x

    def predict(self, u):
        self.x = self.A @ self.x + (self.B @ np.atleast_1d(u)).ravel()
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x


def steady_state_kalman_gain(A, C, Q, R):
    """Return (L, P_prior): update gain L = P C'(C P C' + R)^-1 and prior covariance P (dual DARE)."""
    A, C, Q, R = (np.atleast_2d(np.asarray(m, float)) for m in (A, C, Q, R))
    P = solve_dare(A.T, C.T, Q, R)
    L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    return L, P


class SteadyStateKF:
    """Fixed-gain predictor/corrector: the form you would actually run on a microcontroller."""

    def __init__(self, A, B, C, L):
        self.A, self.B, self.C, self.L = A, B, C, L
        self.x = np.zeros(A.shape[0])

    def update(self, y):
        self.x = self.x + (self.L @ (np.atleast_1d(y) - self.C @ self.x)).ravel()
        return self.x

    def predict(self, u):
        self.x = self.A @ self.x + (self.B @ np.atleast_1d(u)).ravel()
        return self.x
