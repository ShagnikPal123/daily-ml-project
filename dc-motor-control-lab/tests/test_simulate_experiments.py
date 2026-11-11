from dataclasses import replace

import numpy as np
import pytest

from motorctl.cli import build_parser, main
from motorctl.controllers import PIDGains
from motorctl.experiments import evaluate, mean_ci, run_kind
from motorctl.factory import KINDS, build_controller
from motorctl.plant import MotorParams
from motorctl.simulate import SimConfig, simulate

P = MotorParams()
CFG = SimConfig()
G = PIDGains(0.1, 10.0, umax=12.0)


@pytest.mark.parametrize("kind", KINDS)
def test_every_controller_tracks_step_and_rejects_load(kind):
    m, res = run_kind(kind, P, CFG, G)
    assert abs(m["sse"]) < 3.0
    assert m["dist_recover_ms"] < 150
    assert np.mean(res.y[-50:]) == pytest.approx(CFG.ref, abs=3.0)


def test_simulation_deterministic_per_seed_and_varies_across_seeds():
    a = simulate(P, build_controller("pi", P, 1e-3, pi_gains=G), CFG).y
    b = simulate(P, build_controller("pi", P, 1e-3, pi_gains=G), CFG).y
    c = simulate(P, build_controller("pi", P, 1e-3, pi_gains=G), replace(CFG, seed=5)).y
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_voltage_limit_respected():
    p2 = replace(P, vmax=4.0)
    _, res = run_kind("pi", p2, CFG, replace(G, umax=4.0))
    assert np.max(np.abs(res.u_applied)) <= 4.0 + 1e-12


def test_delay_shifts_applied_command():
    _, res = run_kind("pi", P, replace(CFG, delay=3), G)
    assert np.allclose(res.u_applied[3:], res.u[:-3])
    assert res.u_applied[0] == 0.0


def test_encoder_quantisation_visible_in_measurement():
    _, res = run_kind("pi", P, replace(CFG, enc_cpr=500, meas_sigma=0.0), G)
    step = CFG.__class__(enc_cpr=500).enc_step
    assert np.allclose(res.y_meas / step, np.round(res.y_meas / step))


def test_no_noise_no_quantisation_gives_clean_pi():
    m, _ = run_kind("pi", P, replace(CFG, meas_sigma=0.0, enc_cpr=0), G)
    assert m["ripple"] < 0.1


def test_actuator_chatter_grows_with_sensor_noise():
    lo, _ = run_kind("pi", P, replace(CFG, meas_sigma=0.25), G)
    hi, _ = run_kind("pi", P, replace(CFG, meas_sigma=4.0), G)
    assert hi["du_rms"] > lo["du_rms"]


def test_mean_ci_basics():
    m, h = mean_ci([1.0, 2.0, 3.0, float("nan")])
    assert m == 2.0 and h > 0
    assert mean_ci([float("nan")])[0] != mean_ci([float("nan")])[0]
    assert mean_ci([4.0])[1] != mean_ci([4.0])[1]


def test_evaluate_keys():
    _, res = run_kind("pi", P, CFG, G)
    assert {"iae", "overshoot_pct", "dist_peak", "du_rms", "sat_frac"} <= set(evaluate(res, CFG, P))


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_controller("bogus", P, 1e-3, pi_gains=G)


def test_cli_parser_and_validate(capsys):
    assert build_parser().parse_args(["run", "--seeds", "3"]).seeds == 3
    assert main(["validate"]) == 0
    assert "expm_vs_scipy" in capsys.readouterr().out
