import math

import numpy as np
import pytest

from motorctl.fixedpoint import QFormat, rescale_round
from motorctl.metrics import (control_effort, disturbance_metrics, iae, rms_delta,
                              saturation_fraction, step_metrics, window_std)
from motorctl.plant import (MotorParams, continuous_matrices, dc_gain_speed, dc_gain_torque,
                            discretize, encoder_velocity_step, mechanical_time_constant)
from motorctl.riccati import dare_residual, lqr_gain, solve_dare

P = MotorParams()


# ---------------------------------------------------------------- plant
def test_continuous_plant_is_stable():
    A, _, _ = continuous_matrices(P)
    assert np.all(np.linalg.eigvals(A).real < 0)


def test_discrete_plant_is_stable():
    Ad, _, _ = discretize(P, 1e-3)
    assert np.max(np.abs(np.linalg.eigvals(Ad))) < 1.0


def test_dc_gains_match_simulated_steady_state():
    Ad, Bd, _ = discretize(P, 1e-3)
    x = np.zeros(2)
    for _ in range(3000):
        x = Ad @ x + Bd[:, 0] * 6.0 + Bd[:, 1] * 0.01
    expected = dc_gain_speed(P) * 6.0 + dc_gain_torque(P) * 0.01
    assert x[1] == pytest.approx(expected, rel=1e-6)


def test_mechanical_time_constant_is_milliseconds():
    assert 5e-3 < mechanical_time_constant(P) < 12e-3


def test_load_torque_slows_motor():
    assert dc_gain_torque(P) < 0 < dc_gain_speed(P)


def test_encoder_velocity_step():
    assert encoder_velocity_step(4000, 1e-3) == pytest.approx(2 * math.pi / 4.0)
    assert encoder_velocity_step(0, 1e-3) == 0.0


# ---------------------------------------------------------------- fixed point
def test_roundtrip_error_within_half_lsb():
    q = QFormat(16, 8)
    for x in np.linspace(-100, 100, 401):
        assert abs(q.roundtrip(x) - x) <= q.lsb / 2 + 1e-12


def test_saturation_and_limits():
    q = QFormat(8, 4)
    assert q.quantize(1e6) == q.max_int
    assert q.quantize(-1e6) == q.min_int
    assert q.max_value == pytest.approx(127 / 16)


def test_mul_rounding_error_bounded():
    q = QFormat(32, 12)
    rng = np.random.default_rng(1)
    for a, b in rng.uniform(-20, 20, (200, 2)):
        got = q.to_float(q.mul(q.quantize(a), q.quantize(b)))
        ref = q.roundtrip(a) * q.roundtrip(b)
        assert abs(got - ref) <= q.lsb / 2 + 1e-12


def test_array_quantiser_matches_scalar():
    q = QFormat(16, 6)
    x = np.linspace(-300, 300, 77)
    assert [q.quantize(v) for v in x] == list(q.quantize_array(x))


def test_rescale_round_negative_and_shift():
    assert rescale_round(5, 1) == 3  # 2.5 -> 3 (half up)
    assert rescale_round(-5, 1) == -2  # -2.5 -> -2
    assert rescale_round(3, -2) == 12


def test_bad_format_raises():
    with pytest.raises(ValueError):
        QFormat(8, 8)
    with pytest.raises(ValueError):
        QFormat(1, 0)


# ---------------------------------------------------------------- metrics
def test_first_order_rise_time():
    tau = 0.1
    t = np.linspace(0, 2, 20001)
    y = 1 - np.exp(-t / tau)
    m = step_metrics(t, y, 1.0, 0.0, 2.0)
    assert m["rise"] == pytest.approx(2.197 * tau, rel=0.01)
    assert m["overshoot_pct"] == 0.0


def test_second_order_overshoot_matches_theory():
    zeta, wn = 0.5, 10.0
    wd = wn * math.sqrt(1 - zeta**2)
    t = np.linspace(0, 5, 50001)
    y = 1 - np.exp(-zeta * wn * t) / math.sqrt(1 - zeta**2) * np.sin(wd * t + math.acos(zeta))
    m = step_metrics(t, y, 1.0, 0.0, 5.0)
    theory = 100 * math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2))
    assert m["overshoot_pct"] == pytest.approx(theory, rel=0.01)


def test_iae_effort_and_helpers():
    t = np.arange(0, 1, 0.01)
    assert iae(t, np.ones_like(t) * 2) == pytest.approx(2.0, rel=0.02)
    assert control_effort(np.ones(100) * 3, 0.01) == pytest.approx(9.0)
    assert rms_delta(np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)
    assert saturation_fraction(np.array([12, 5, -12, 0.0]), 12) == 0.5
    assert window_std(t, np.ones_like(t), 0.1, 0.5) == 0.0


def test_disturbance_metrics_recovery():
    t = np.linspace(0, 1, 1001)
    y = 100 - 20 * np.exp(-(t - 0.5).clip(0) / 0.05) * (t >= 0.5)
    d = disturbance_metrics(t, y, 100.0, 0.5, 5.0)
    assert d["peak"] == pytest.approx(20.0, rel=0.01)
    assert d["recover"] == pytest.approx(0.05 * math.log(4), rel=0.05)


# ---------------------------------------------------------------- Riccati
def test_scalar_dare_is_golden_ratio():
    # A=B=Q=R=1  ->  P = (1 + sqrt 5)/2  (closed form from p = p - p^2/(1+p) + 1)
    Pm = solve_dare([[1.0]], [[1.0]], [[1.0]], [[1.0]])
    assert Pm[0, 0] == pytest.approx((1 + math.sqrt(5)) / 2, abs=1e-12)


def test_dare_residual_small_for_motor():
    Ad, Bd, _ = discretize(P, 1e-3)
    Q = np.diag([0.1, 1.0])
    Pm = solve_dare(Ad, Bd[:, :1], Q, [[1.0]])
    assert dare_residual(Ad, Bd[:, :1], Q, [[1.0]], Pm) < 1e-8


def test_lqr_closed_loop_is_stable():
    Ad, Bd, _ = discretize(P, 1e-3)
    K, _ = lqr_gain(Ad, Bd[:, :1], np.diag([0.1, 1.0]), [[1.0]])
    assert np.max(np.abs(np.linalg.eigvals(Ad - Bd[:, :1] @ K))) < 1.0
