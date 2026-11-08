import numpy as np
import pytest

from regimelab.config import ExpConfig
from regimelab.metrics import (confusion, detection_delay, match_labels, max_drawdown, mean_ci,
                               permutation_accuracy, sharpe)
from regimelab.synth import (persistence_matrix, regime_params, sample_markov_chain, simulate,
                             stationary_distribution)


def test_persistence_rows_sum_to_one():
    P = persistence_matrix(4, 0.9)
    assert np.allclose(P.sum(1), 1) and np.allclose(np.diag(P), 0.9)


def test_persistence_rejects_bad_input():
    with pytest.raises(ValueError):
        persistence_matrix(1, 0.5)
    with pytest.raises(ValueError):
        persistence_matrix(3, 1.0)


def test_stationary_is_fixed_point():
    P = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
    pi = stationary_distribution(P)
    assert np.allclose(pi @ P, pi, atol=1e-10) and abs(pi.sum() - 1) < 1e-12


def test_chain_empirical_transitions():
    P = persistence_matrix(3, 0.95)
    s = sample_markov_chain(P, 60000, np.random.default_rng(1))
    stay = np.mean(s[1:] == s[:-1])
    assert abs(stay - 0.95) < 0.01


def test_simulation_deterministic_per_seed():
    p = regime_params(3, 1.0)
    P = persistence_matrix(3, 0.97)
    a = simulate(500, P, p, np.random.default_rng(5))
    b = simulate(500, P, p, np.random.default_rng(5))
    assert np.array_equal(a.returns, b.returns) and np.all(a.price > 0)


def test_regime_volatility_ordering_in_data():
    p = regime_params(3, 1.5, 0.0)
    d = simulate(30000, persistence_matrix(3, 0.98), p, np.random.default_rng(2))
    stds = [d.returns[d.states == i].std() for i in range(3)]
    assert stds[0] < stds[1] < stds[2]
    assert np.allclose(stds, p.sigma, rtol=0.1)


def test_cycle_adds_spectral_line():
    p = regime_params(2, 1.0, 0.0)
    d = simulate(2100, persistence_matrix(2, 0.99), p, np.random.default_rng(3), cycle_amp=0.01, cycle_period=21)
    spec = np.abs(np.fft.rfft(d.returns - d.returns.mean()))
    assert np.argmax(spec) == 100


def test_match_labels_recovers_permutation():
    true = np.array([0, 0, 1, 1, 2, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1, 1])
    assert permutation_accuracy(true, pred, 3) == 1.0
    assert np.array_equal(match_labels(true, pred, 3), true)
    assert confusion(true, pred, 3).sum() == 7


def test_detection_delay_known_case():
    true = np.array([0] * 10 + [1] * 10)
    pred = np.array([0] * 13 + [1] * 7)
    assert detection_delay(true, pred, 2) == 3.0


def test_sharpe_and_drawdown():
    assert sharpe(np.zeros(10)) == 0.0
    r = np.array([0.1, -0.5, 0.1])
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert max_drawdown(np.full(5, 0.01)) == 0.0


def test_mean_ci_shrinks_with_n():
    rng = np.random.default_rng(0)
    m1, h1 = mean_ci(rng.normal(size=10))
    m2, h2 = mean_ci(rng.normal(size=1000))
    assert h2 < h1 and abs(m2) < 0.2


def test_config_roundtrip_and_validation(tmp_path):
    c = ExpConfig(k=2, seeds=3)
    c.save(tmp_path / "c.json")
    assert ExpConfig.load(tmp_path / "c.json") == c
    with pytest.raises(ValueError):
        ExpConfig.from_dict({"bogus": 1})
    with pytest.raises(ValueError):
        ExpConfig(p_stay=1.5).validate()
    assert c.replace(k=4).k == 4
