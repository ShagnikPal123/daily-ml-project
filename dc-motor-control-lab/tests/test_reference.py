"""Ground-truth checks of our own kernels against independent SciPy implementations."""
import numpy as np
import pytest
import scipy.linalg as sla
import scipy.signal as sig

from motorctl.experiments import validation_table
from motorctl.plant import MotorParams, continuous_matrices, discretize, expm, zoh
from motorctl.riccati import solve_dare

P = MotorParams()


def test_expm_matches_scipy_on_motor_and_random():
    A, _, _ = continuous_matrices(P)
    assert np.max(np.abs(expm(A * 1e-3) - sla.expm(A * 1e-3))) < 1e-12
    rng = np.random.default_rng(0)
    for _ in range(5):
        M = rng.normal(size=(4, 4)) * 3
        assert np.max(np.abs(expm(M) - sla.expm(M))) / np.max(np.abs(sla.expm(M))) < 1e-10


@pytest.mark.parametrize("Ts", [2e-4, 1e-3, 5e-3])
def test_zoh_matches_scipy_cont2discrete(Ts):
    A, B, C = continuous_matrices(P)
    Ad, Bd = zoh(A, B, Ts)
    Ad2, Bd2, _, _, _ = sig.cont2discrete((A, B, C, np.zeros((1, 2))), Ts, method="zoh")
    assert np.max(np.abs(Ad - Ad2)) < 1e-10
    assert np.max(np.abs(Bd - Bd2)) < 1e-10


def test_dare_matches_scipy_random_systems():
    rng = np.random.default_rng(3)
    for _ in range(6):
        A = rng.normal(size=(3, 3)) * 0.6
        B = rng.normal(size=(3, 1))
        Q = np.eye(3)
        R = np.array([[0.5]])
        ours = solve_dare(A, B, Q, R)
        ref = sla.solve_discrete_are(A, B, Q, R)
        assert np.max(np.abs(ours - ref)) / np.max(np.abs(ref)) < 1e-8


def test_dare_matches_scipy_on_integral_augmented_motor():
    Ad, Bd, C = discretize(P, 1e-3)
    n = 2
    Aa = np.zeros((3, 3))
    Aa[:n, :n] = Ad
    Aa[2, :n] = -1e-3 * C[0]
    Aa[2, 2] = 1.0
    Ba = np.vstack([Bd[:, :1], [[0.0]]])
    Q = np.diag([0.1, 1.0, 3000.0])
    ours = solve_dare(Aa, Ba, Q, [[1.0]])
    ref = sla.solve_discrete_are(Aa, Ba, Q, np.array([[1.0]]))
    assert np.max(np.abs(ours - ref)) / np.max(np.abs(ref)) < 1e-7


def test_validation_table_reports_small_errors():
    tab = validation_table()
    assert tab["expm_vs_scipy"] < 1e-11
    assert tab["zoh_vs_scipy"] < 1e-9
    assert tab["dare_vs_scipy"] < 1e-7
    assert tab["golden_ratio_dare"] < 1e-12
    assert tab["kalman_gain_vs_scipy"] < 1e-7
