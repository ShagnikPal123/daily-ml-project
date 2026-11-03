"""BPSK modulation and noisy channels, with LLR computation.

Mapping: bit 0 -> +1, bit 1 -> -1.  With code rate R and per-bit energy Eb the symbol
energy is Es = R*Eb, and the noise variance per real dimension is sigma^2 = N0/2, so

    sigma^2 = 1 / (2 * R * 10^(EbN0_dB/10))        (Es normalised to 1)

Channel LLR for AWGN:  L = log P(x=+1|y)/P(x=-1|y) = 2 y / sigma^2.
For Rayleigh fading with known gain h:  L = 2 h y / sigma^2.
"""
import numpy as np
from scipy.special import erfc


def sigma_from_ebn0(ebn0_db, rate):
    return np.sqrt(1.0 / (2.0 * rate * 10.0 ** (ebn0_db / 10.0)))


def bpsk(bits):
    return 1.0 - 2.0 * np.asarray(bits, dtype=np.float64)


def awgn(symbols, sigma, rng):
    return symbols + sigma * rng.standard_normal(symbols.shape)


def awgn_llr(y, sigma):
    return 2.0 * y / sigma ** 2


def rayleigh(symbols, sigma, rng):
    """Fast Rayleigh fading (independent gain per symbol, E[h^2]=1); returns (y, h)."""
    h = np.sqrt(rng.exponential(1.0, size=symbols.shape))
    return h * symbols + sigma * rng.standard_normal(symbols.shape), h


def rayleigh_llr(y, h, sigma):
    return 2.0 * h * y / sigma ** 2


def hard_decision(llr):
    return (llr < 0).astype(np.uint8)


def uncoded_ber_theory(ebn0_db):
    """Q(sqrt(2 Eb/N0)) for coherent BPSK."""
    ebn0 = 10.0 ** (np.asarray(ebn0_db) / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0))


def uncoded_ber_rayleigh_theory(ebn0_db):
    ebn0 = 10.0 ** (np.asarray(ebn0_db) / 10.0)
    return 0.5 * (1.0 - np.sqrt(ebn0 / (1.0 + ebn0)))


def biawgn_capacity(sigma, n=200001, rng=None):
    """Numerically integrate the BI-AWGN capacity (bits/channel use)."""
    y = np.linspace(-1 - 12 * sigma, 1 + 12 * sigma, n)
    p = lambda m: np.exp(-(y - m) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    p1, p0 = p(1.0), p(-1.0)
    mix = 0.5 * (p1 + p0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(mix > 0, mix * np.log2(mix), 0.0)
        c = 0.5 * (np.where(p1 > 0, p1 * np.log2(p1 / mix), 0.0) + np.where(p0 > 0, p0 * np.log2(p0 / mix), 0.0))
    return float(np.trapezoid(c, y))


def shannon_limit_ebn0_db(rate, lo=-2.0, hi=6.0):
    """Eb/N0 (dB) at which BI-AWGN capacity equals the code rate (bisection)."""
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if biawgn_capacity(sigma_from_ebn0(mid, rate), n=40001) < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
