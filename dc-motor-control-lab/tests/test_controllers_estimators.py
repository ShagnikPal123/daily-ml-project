import numpy as np
import pytest

from motorctl.controllers import (PID, PIDGains, augment_integral, design_lqi,
                                  disturbance_feedforward_gain)
from motorctl.estimators import (KalmanFilter, SteadyStateKF, augment_disturbance,
                                 steady_state_kalman_gain)
from motorctl.fixedpid import FixedPointPID
from motorctl.fixedpoint import QFormat
from motorctl.lqg import delay_extend, design_lqg
from motorctl.plant import MotorParams, discretize

P = MotorParams()
G = PIDGains(0.1, 10.0, umax=12.0)


# ---------------------------------------------------------------- PID
def test_pid_output_is_clipped():
    c = PID(G, 1e-3)
    assert c.step(1e4, 0.0) == 12.0
    assert c.step(-1e4, 0.0) == -12.0


def test_p_term_only_on_first_step():
    c = PID(PIDGains(0.2, 0.0), 1e-3)
    assert c.step(10.0, 4.0) == pytest.approx(1.2)


def test_clamp_limits_integrator_vs_none():
    cn, cc = PID(G, 1e-3, "none"), PID(G, 1e-3, "clamp")
    for _ in range(500):
        cn.step(200.0, 0.0)
        cc.step(200.0, 0.0)
    assert cc.i < cn.i
    assert cc.i <= 12.0


def test_unknown_antiwindup_rejected():
    with pytest.raises(ValueError):
        PID(G, 1e-3, "magic")


def test_derivative_filter_responds_to_ramp():
    c = PID(PIDGains(0.0, 0.0, kd=0.01, tau_d=1e-3), 1e-3)
    out = [c.step(0.0, -k * 0.1) for k in range(50)]  # y falls -> positive derivative action
    assert out[-1] > 0


# ---------------------------------------------------------------- LQI
def test_augmented_dimensions_and_integrator_row():
    Ad, Bd, Cd = discretize(P, 1e-3)
    Aa, Ba = augment_integral(Ad, Bd[:, :1], Cd, 1e-3)
    assert Aa.shape == (3, 3) and Ba.shape == (3, 1)
    assert Aa[2, 1] == pytest.approx(-1e-3) and Aa[2, 2] == 1.0


def test_lqi_closed_loop_stable_and_feedforward_zeroes_steady_error():
    Ad, Bd, Cd = discretize(P, 1e-3)
    Bv, Bt = Bd[:, :1], Bd[:, 1:2]
    g = design_lqi(Ad, Bv, Cd, 1e-3)
    K = np.hstack([g.Kx, [[g.Kz]]])
    assert np.max(np.abs(np.linalg.eigvals(g.Aaug - g.Baug @ K))) < 1.0
    N = disturbance_feedforward_gain(Ad, Bv, Bt, Cd, g.Kx)
    d = 0.02
    M = np.linalg.inv(np.eye(2) - Ad + Bv @ g.Kx)
    x_ss = M @ (Bv * (-N * d) + Bt * d)
    assert abs((Cd @ x_ss)[0, 0]) < 1e-9


def test_delay_extension_shapes_and_shift():
    Ad, Bd, Cd = discretize(P, 1e-3)
    A, B, Bd_, C = delay_extend(Ad, Bd[:, :1], Bd[:, 1:2], Cd, 2)
    assert A.shape == (4, 4) and B[2, 0] == 1.0 and A[3, 2] == 1.0
    assert A[0, 3] == Bd[0, 0]  # plant sees the oldest command


def test_delay_aware_lqr_stable_with_delay():
    d = design_lqg(P, 1e-3, delay=1)
    K = np.hstack([d.lqi.Kx, [[d.lqi.Kz]]])
    assert np.max(np.abs(np.linalg.eigvals(d.lqi.Aaug - d.lqi.Baug @ K))) < 1.0


# ---------------------------------------------------------------- Kalman
def _aug():
    Ad, Bd, _ = discretize(P, 1e-3)
    return augment_disturbance(Ad, Bd)


def test_augmented_model_observable():
    A, Bv, C = _aug()
    O = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(3)])
    assert np.linalg.matrix_rank(O) == 3


def test_kf_covariance_converges_to_dare():
    A, Bv, C = _aug()
    Q, R = np.diag([0.05**2, 0.5**2, 2e-4**2]), np.array([[0.8]])
    kf = KalmanFilter(A, Bv, C, Q, R, P0=np.eye(3) * 5)
    for _ in range(3000):
        kf.update(0.0)
        kf.predict(0.0)
    _, Pss = steady_state_kalman_gain(A, C, Q, R)
    assert np.max(np.abs(kf.P - Pss)) / np.max(np.abs(Pss)) < 1e-6


def test_kf_beats_raw_measurement_rms():
    A, Bv, C = _aug()
    Ad, Bd, _ = discretize(P, 1e-3)
    Q, R = np.diag([0.05**2, 0.5**2, 2e-4**2]), np.array([[1.0]])
    L, _ = steady_state_kalman_gain(A, C, Q, R)
    kf = SteadyStateKF(A, Bv, C, L)
    rng = np.random.default_rng(0)
    x, err_kf, err_raw = np.zeros(2), [], []
    for k in range(2000):
        u = 5.0 + 3.0 * np.sin(k / 80)
        y = x[1] + rng.normal(0, 1.0)
        xh = kf.update(y)
        err_kf.append(xh[1] - x[1])
        err_raw.append(y - x[1])
        kf.predict(u)
        x = Ad @ x + Bd[:, 0] * u
    assert np.sqrt(np.mean(np.square(err_kf[200:]))) < 0.6 * np.sqrt(np.mean(np.square(err_raw[200:])))


# ---------------------------------------------------------------- fixed-point PI
def test_fixed_pi_tracks_float_pi_at_high_precision():
    fl, fx = PID(G, 1e-3), FixedPointPID(G, 1e-3, QFormat(40, 20), guard_bits=8)
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(300):
        r, y = 100.0, rng.uniform(0, 100)
        worst = max(worst, abs(fl.step(r, y) - fx.step(r, y)))
    assert worst < 0.01


def test_integral_gain_underflows_without_guard_bits():
    low = FixedPointPID(G, 1e-3, QFormat(32, 4), guard_bits=0)
    guarded = FixedPointPID(G, 1e-3, QFormat(32, 4), guard_bits=10)
    assert low.effective_ki_ts == 0.0  # ki*Ts = 0.01 < lsb/2 = 1/32
    assert guarded.effective_ki_ts == pytest.approx(0.01, rel=0.05)


def test_fixed_pi_output_bounded_and_resets():
    c = FixedPointPID(G, 1e-3, QFormat(32, 12), guard_bits=4)
    for _ in range(200):
        assert abs(c.step(500.0, 0.0)) <= 12.0
    c.reset()
    assert c.acc == 0
