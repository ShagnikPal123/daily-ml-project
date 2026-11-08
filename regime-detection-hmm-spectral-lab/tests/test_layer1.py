import itertools

import numpy as np
import pytest

from regimelab.filters import (LocalLevelKalman, butter_lowpass_causal, ema, ewma_vol,
                               rolling_band_fraction, rolling_std, sma, steady_state_gain)
from regimelab.hmm import GaussianHMM, brute_force_loglik
from regimelab.synth import persistence_matrix, regime_params, simulate


def test_ema_matches_loop_and_is_causal():
    x = np.random.default_rng(0).normal(size=50)
    y = ema(x, 0.2)
    ref = np.empty(50)
    ref[0] = x[0]
    for t in range(1, 50):
        ref[t] = 0.2 * x[t] + 0.8 * ref[t - 1]
    assert np.allclose(y, ref)
    x2 = x.copy()
    x2[30:] += 5
    assert np.allclose(ema(x2, 0.2)[:30], y[:30])


def test_sma_and_rolling_std_match_naive():
    x = np.random.default_rng(1).normal(size=40)
    assert np.allclose(sma(x, 5)[10], x[6:11].mean())
    assert np.allclose(sma(x, 5)[2], x[:3].mean())
    assert np.allclose(rolling_std(x, 8)[20], x[13:21].std(ddof=1))
    assert np.all(np.isfinite(rolling_std(x, 8)))


def test_butter_passes_dc_attenuates_nyquist():
    n = 400
    dc = butter_lowpass_causal(np.ones(n) * 3.0, 0.2)
    assert np.allclose(dc, 3.0, atol=1e-9)
    alt = butter_lowpass_causal((-1.0) ** np.arange(n), 0.2)
    assert np.abs(alt[100:]).max() < 0.05


def test_ewma_vol_tracks_sigma():
    r = np.random.default_rng(2).normal(scale=0.01, size=5000)
    v = ewma_vol(r, 0.97)
    assert abs(v[1000:].mean() - 0.01) < 0.002


def test_band_fraction_separates_momentum_from_mean_reversion():
    rng = np.random.default_rng(3)
    e = rng.normal(size=6000)
    from scipy.signal import lfilter
    pos, neg = lfilter([1], [1, -0.6], e), lfilter([1], [1, 0.6], e)
    fp, fn, fw = (rolling_band_fraction(s, 64)[100:].mean() for s in (pos, neg, e))
    assert fn < fw < fp


def test_kalman_gain_converges_to_closed_form():
    kf = LocalLevelKalman(q=1e-3, r=1e-2)
    y = np.random.default_rng(4).normal(size=500)
    _, _, ks = kf.filter(y)
    assert ks[-1] == pytest.approx(steady_state_gain(1e-3, 1e-2), rel=1e-9)


def test_kalman_beats_raw_observations():
    rng = np.random.default_rng(5)
    x = np.cumsum(rng.normal(scale=0.05, size=2000))
    y = x + rng.normal(scale=0.5, size=2000)
    xh, _, _ = LocalLevelKalman(0.05 ** 2, 0.25).filter(y)
    assert np.mean((xh - x) ** 2) < 0.4 * np.mean((y - x) ** 2)


def _toy_hmm():
    h = GaussianHMM(2)
    h.means = np.array([[0.0], [2.0]])
    h.vars = np.array([[1.0], [0.5]])
    h.trans = np.array([[0.8, 0.2], [0.3, 0.7]])
    h.startprob = np.array([0.6, 0.4])
    return h


def test_forward_loglik_equals_brute_force():
    h = _toy_hmm()
    X = np.array([[0.1], [1.9], [2.2], [-0.3], [0.5], [1.7]])
    assert h.score(X) == pytest.approx(brute_force_loglik(h, X), abs=1e-10)


def test_viterbi_and_smoothing_equal_brute_force():
    h = _toy_hmm()
    X = np.array([[0.1], [1.9], [2.2], [-0.3], [0.5]])
    logB = h.log_emission(X)
    best, best_lp = None, -np.inf
    post = np.zeros((5, 2))
    allp = []
    for path in itertools.product(range(2), repeat=5):
        lp = np.log(h.startprob[path[0]]) + logB[0, path[0]]
        for t in range(1, 5):
            lp += np.log(h.trans[path[t - 1], path[t]]) + logB[t, path[t]]
        allp.append((path, lp))
        if lp > best_lp:
            best, best_lp = path, lp
    assert tuple(h.viterbi(X)) == best
    z = np.logaddexp.reduce([lp for _, lp in allp])
    for path, lp in allp:
        for t, s in enumerate(path):
            post[t, s] += np.exp(lp - z)
    assert np.allclose(h.smooth_posterior(X), post, atol=1e-10)


def test_filter_posterior_is_causal():
    h = _toy_hmm()
    X = np.random.default_rng(6).normal(size=(60, 1))
    a = h.filter_posterior(X)
    X2 = X.copy()
    X2[40:] += 10
    assert np.allclose(h.filter_posterior(X2)[:40], a[:40])


def test_baum_welch_monotone_and_recovers_parameters():
    p = regime_params(2, 2.0, 0.0)
    d = simulate(4000, persistence_matrix(2, 0.97), p, np.random.default_rng(7))
    h = GaussianHMM(2, n_iter=40, tol=0.0, seed=0).fit(d.returns)
    ll = np.array(h.loglik_history)
    assert np.all(np.diff(ll) > -1e-6)
    order = np.argsort(h.vars[:, 0])
    assert np.allclose(np.sqrt(h.vars[order, 0]), p.sigma, rtol=0.15)
    assert np.all(np.diag(h.trans) > 0.9)


def test_hmm_numerically_stable_for_extreme_inputs():
    h = _toy_hmm()
    X = np.vstack([np.full((200, 1), 50.0), np.full((200, 1), -50.0)])
    assert np.isfinite(h.score(X))
    assert np.all(np.isfinite(h.filter_posterior(X)))
    assert np.allclose(h.filter_posterior(X).sum(1), 1)
