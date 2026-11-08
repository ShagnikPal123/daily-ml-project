"""Ground-truth checks against independent references (scipy / numpy / brute force).
`measure()` returns the max error of each comparison; the README validation table is generated from it."""
import itertools

import numpy as np
from scipy import stats
from scipy.linalg import solve_discrete_are
from scipy.signal import butter, lfilter, lfilter_zi, welch
from scipy.stats import multivariate_normal

from regimelab.filters import (LocalLevelKalman, butter_lowpass_causal, ema, rolling_std,
                               steady_state_gain, welch_band_power)
from regimelab.hmm import GaussianHMM
from regimelab.metrics import mean_ci, permutation_accuracy, sharpe
from regimelab.synth import persistence_matrix, stationary_distribution

TOL = {"kalman_gain_vs_DARE": 1e-9, "butterworth_vs_lfilter": 1e-12, "hmm_emission_vs_scipy_mvn": 1e-10,
       "hmm_k1_loglik_vs_norm_logpdf": 1e-8, "stationary_vs_matrix_power": 1e-9, "label_match_vs_brute_force": 0.0,
       "mean_ci_vs_t_interval": 1e-12, "welch_band_vs_trapz": 1e-15, "ema_vs_pandas_style_recursion": 1e-12,
       "rolling_std_vs_numpy": 1e-12}


def measure() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    q, r = 2e-3, 1e-2
    P = solve_discrete_are(np.array([[1.0]]).T, np.array([[1.0]]), np.array([[q]]), np.array([[r]]))[0, 0]
    out["kalman_gain_vs_DARE"] = abs(steady_state_gain(q, r) - P / (P + r))
    x = rng.normal(size=300)
    b, a = butter(2, 0.15)
    ref = lfilter(b, a, x, zi=lfilter_zi(b, a) * x[0])[0]
    out["butterworth_vs_lfilter"] = np.abs(butter_lowpass_causal(x, 0.15) - ref).max()
    h = GaussianHMM(3)
    h.means, h.vars = rng.normal(size=(3, 2)), rng.uniform(0.5, 2, size=(3, 2))
    X = rng.normal(size=(20, 2))
    ref = np.column_stack([multivariate_normal(h.means[j], np.diag(h.vars[j])).logpdf(X) for j in range(3)])
    out["hmm_emission_vs_scipy_mvn"] = np.abs(h.log_emission(X) - ref).max()
    h1 = GaussianHMM(1)
    h1.means, h1.vars, h1.trans, h1.startprob = np.array([[0.3]]), np.array([[1.7]]), np.ones((1, 1)), np.ones(1)
    Y = rng.normal(size=(200, 1))
    out["hmm_k1_loglik_vs_norm_logpdf"] = abs(h1.score(Y) - stats.norm(0.3, np.sqrt(1.7)).logpdf(Y).sum())
    Pm = persistence_matrix(4, 0.8)
    out["stationary_vs_matrix_power"] = np.abs(stationary_distribution(Pm) - np.linalg.matrix_power(Pm, 300)[0]).max()
    true, pred = rng.integers(0, 3, 90), rng.integers(0, 3, 90)
    best = max(np.mean(np.array(p)[pred] == true) for p in itertools.permutations(range(3)))
    out["label_match_vs_brute_force"] = abs(permutation_accuracy(true, pred, 3) - best)
    s = rng.normal(size=12)
    lo, hi = stats.t.interval(0.95, 11, loc=s.mean(), scale=stats.sem(s))
    m, hw = mean_ci(s)
    out["mean_ci_vs_t_interval"] = max(abs(m - hw - lo), abs(m + hw - hi))
    sig = rng.normal(size=1024)
    f, pxx = welch(sig, fs=1.0, nperseg=128)
    sel = (f >= 0) & (f < 0.1)
    out["welch_band_vs_trapz"] = abs(welch_band_power(sig, 1.0, 128, (0, 0.1)) - np.trapezoid(pxx[sel], f[sel]))
    z = rng.normal(size=100)
    e = np.empty(100)
    e[0] = z[0]
    for t in range(1, 100):
        e[t] = 0.3 * z[t] + 0.7 * e[t - 1]
    out["ema_vs_pandas_style_recursion"] = np.abs(ema(z, 0.3) - e).max()
    out["rolling_std_vs_numpy"] = abs(rolling_std(z, 10)[50] - z[41:51].std(ddof=1))
    return {k: float(v) for k, v in out.items()}


def test_all_references_agree():
    errs = measure()
    for k, v in errs.items():
        assert v <= TOL[k], (k, v)


def test_sharpe_matches_definition():
    r = np.random.default_rng(1).normal(0.001, 0.01, 500)
    assert abs(sharpe(r) - r.mean() / r.std(ddof=1) * np.sqrt(252)) < 1e-12


def test_kalman_filter_vs_dare_covariance():
    q, r = 5e-4, 4e-3
    _, ps, _ = LocalLevelKalman(q, r).filter(np.random.default_rng(2).normal(size=800))
    Pm = solve_discrete_are(np.array([[1.0]]), np.array([[1.0]]), np.array([[q]]), np.array([[r]]))[0, 0]
    assert abs(ps[-1] - Pm * r / (Pm + r)) < 1e-12
