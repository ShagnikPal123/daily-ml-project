"""Adaptive FIR filters for noise cancellation: LMS, NLMS, RLS."""
import numpy as np


def _taps(ref, n, L):
    """Tap-delay vector [r[n], r[n-1], ..., r[n-L+1]] with zero history."""
    lo = n - L + 1
    seg = ref[max(lo, 0):n + 1][::-1]
    if len(seg) < L:
        seg = np.concatenate([seg, np.zeros(L - len(seg))])
    return seg


def lms(primary, ref, L=32, mu=0.01):
    w = np.zeros(L)
    e = np.zeros(len(primary))
    W = []
    for n in range(len(primary)):
        u = _taps(ref, n, L)
        y = w @ u
        e[n] = primary[n] - y
        w = w + mu * e[n] * u
        if n % 50 == 0:
            W.append(w.copy())
    return e, w, np.array(W)


def nlms(primary, ref, L=32, mu=0.5, eps=1e-6):
    w = np.zeros(L)
    e = np.zeros(len(primary))
    W = []
    for n in range(len(primary)):
        u = _taps(ref, n, L)
        e[n] = primary[n] - w @ u
        w = w + mu * e[n] * u / (eps + u @ u)
        if n % 50 == 0:
            W.append(w.copy())
    return e, w, np.array(W)


def rls(primary, ref, L=32, lam=0.999, delta=100.0):
    w = np.zeros(L)
    P = np.eye(L) * delta
    e = np.zeros(len(primary))
    W = []
    for n in range(len(primary)):
        u = _taps(ref, n, L)
        Pu = P @ u
        k = Pu / (lam + u @ Pu)
        e[n] = primary[n] - w @ u
        w = w + k * e[n]
        P = (P - np.outer(k, Pu)) / lam
        P = 0.5 * (P + P.T)  # keep P symmetric: prevents numerical blow-up with small lambda
        if n % 50 == 0:
            W.append(w.copy())
    return e, w, np.array(W)


ALGOS = {"lms": lms, "nlms": nlms, "rls": rls}


def make_noise_path(rng, L=16, decay=4.0):
    """Unknown acoustic path from noise source to the primary microphone."""
    h = rng.standard_normal(L) * np.exp(-np.arange(L) / decay)
    return h / np.linalg.norm(h)


def misalignment_db(w, h):
    wp = np.zeros(max(len(w), len(h)))
    hp = np.zeros_like(wp)
    wp[:len(w)] = w
    hp[:len(h)] = h
    return 10 * np.log10(np.sum((wp - hp) ** 2) / np.sum(hp ** 2))


def smooth_power_db(e, win=400):
    k = np.ones(win) / win
    return 10 * np.log10(np.convolve(e ** 2, k, mode="same") + 1e-12)
