"""Layer 0: regime-switching return generator (hidden Markov chain + AR(1) + Gaussian noise).

Model: s_t ~ Markov(P);  r_t = mu[s_t] + phi[s_t] r_{t-1} + sigma[s_t] eps_t + A sin(2 pi t / T0).
Synthetic only: regimes are ordered by volatility (state 0 = calmest).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegimeParams:
    mu: np.ndarray
    sigma: np.ndarray
    phi: np.ndarray


@dataclass
class SimData:
    states: np.ndarray
    returns: np.ndarray
    price: np.ndarray
    params: RegimeParams


def persistence_matrix(k: int, p_stay: float) -> np.ndarray:
    """Symmetric transition matrix with diagonal p_stay."""
    if k < 2 or not 0.0 < p_stay < 1.0:
        raise ValueError("need k>=2 and 0<p_stay<1")
    P = np.full((k, k), (1.0 - p_stay) / (k - 1))
    np.fill_diagonal(P, p_stay)
    return P


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1, normalised to sum to one."""
    w, v = np.linalg.eig(np.asarray(P, float).T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, i])
    pi = pi / pi.sum()
    return np.clip(pi, 0.0, None) / np.clip(pi, 0.0, None).sum()


def sample_markov_chain(P: np.ndarray, n: int, rng: np.random.Generator, init=None) -> np.ndarray:
    P = np.asarray(P, float)
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("rows of P must sum to 1")
    k = P.shape[0]
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0
    pi0 = stationary_distribution(P) if init is None else np.asarray(init, float)
    u = rng.random(n)
    s = np.empty(n, dtype=int)
    s[0] = min(int(np.searchsorted(np.cumsum(pi0), u[0])), k - 1)
    for t in range(1, n):
        s[t] = min(int(np.searchsorted(cum[s[t - 1]], u[t])), k - 1)
    return s


def regime_params(k: int = 3, separation: float = 1.0, ar_strength: float = 0.15) -> RegimeParams:
    """Calm regimes drift up with mean reversion; stressed regimes drift down with momentum."""
    idx = np.arange(k)
    sigma = 0.006 * (1.0 + separation * idx)
    mu = np.linspace(0.0006, -0.0010, k)
    phi = ar_strength * np.linspace(-1.0, 1.0, k)
    return RegimeParams(mu=mu, sigma=sigma, phi=phi)


def simulate(n: int, P: np.ndarray, params: RegimeParams, rng: np.random.Generator,
             cycle_amp: float = 0.0, cycle_period: float = 21.0) -> SimData:
    states = sample_markov_chain(P, n, rng)
    eps = rng.standard_normal(n)
    r = np.empty(n)
    prev = 0.0
    for t in range(n):
        s = states[t]
        prev = params.mu[s] + params.phi[s] * prev + params.sigma[s] * eps[t]
        r[t] = prev
    if cycle_amp:
        r = r + cycle_amp * np.sin(2.0 * np.pi * np.arange(n) / cycle_period)
    price = 100.0 * np.exp(np.cumsum(r))
    return SimData(states=states, returns=r, price=price, params=params)
