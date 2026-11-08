import numpy as np
import pytest

from lowlight import ptc
from lowlight.sensor import SensorConfig

CFG = SensorConfig(photons=2000, read_noise=3.0, dark_electrons=2.0, adc_bits=12, prnu=0.02)
LEVELS = np.linspace(0.05, 1.0, 10)


@pytest.fixture(scope="module")
def curve():
    return ptc.photon_transfer(CFG, LEVELS, n_frames=48)


def test_ptc_mean_increases_with_light(curve):
    assert np.all(np.diff(curve["mean_dn"]) > 0)


def test_recovered_gain_matches_adc_design(curve):
    fit = ptc.fit_ptc(curve)
    assert fit["gain_dn_per_e"] == pytest.approx(ptc.true_gain(CFG), rel=0.05) and fit["r2"] > 0.99


def test_recovered_noise_floor_close_to_truth(curve):
    fit = ptc.fit_ptc(curve)
    assert fit["dark_noise_e"] == pytest.approx(ptc.true_noise_floor_e(CFG), rel=0.1)


def test_fit_needs_enough_points():
    c = ptc.photon_transfer(CFG, [0.1, 0.2], n_frames=8)
    with pytest.raises(ValueError):
        ptc.fit_ptc(c)


def test_empirical_snr_tracks_theory_and_grows():
    from lowlight.sensor import Sensor
    snr = ptc.empirical_snr(CFG.replace(prnu=0.0), [0.1, 0.5, 1.0])
    th = Sensor(CFG.replace(prnu=0.0), (16, 16)).snr(np.array([0.1, 0.5, 1.0]))
    assert np.all(np.diff(snr) > 0) and np.allclose(snr, th, rtol=0.08)


def test_spatial_noise_is_white():
    assert 0.6 < ptc.smoothness_check(CFG) < 0.95  # ideal white noise: 1 - 1/9 = 0.889


def test_temporal_stats_remove_fixed_pattern():
    stack = np.random.default_rng(0).normal(100, 2, (50, 8, 8)) + np.random.default_rng(1).normal(0, 20, (8, 8))
    m, v = ptc.temporal_stats(stack)
    assert v == pytest.approx(4.0, rel=0.1)
