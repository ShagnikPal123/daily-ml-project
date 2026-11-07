"""Tuning the min-sum correction factors on a training set.

Two procedures:
  * tune_alpha:      grid search of a single normalisation factor to minimise BER.
  * learn_alpha_per_iter: coordinate descent over per-iteration factors that minimises the
    logistic surrogate loss  mean(log(1 + exp(-s * L_post)))  with s = +-1 the transmitted
    BPSK symbol (no early stopping, so every iteration contributes to the loss).
This is "deep unfolding" of the decoder with one learnable scalar per layer, optimised
without backprop.
"""
import numpy as np
from . import channel, decoders


def make_training_batch(code, ebn0_db, n_frames, rng):
    info = rng.integers(0, 2, size=(n_frames, code.k), dtype=np.uint8)
    cw = code.encode(info)
    sigma = channel.sigma_from_ebn0(ebn0_db, code.rate)
    y = channel.awgn(channel.bpsk(cw), sigma, rng)
    return cw, channel.awgn_llr(y, sigma)


def surrogate_loss(code, cw, llr, alphas):
    res = decoders.decode(code, llr, "nms", iters=len(alphas), alpha=np.asarray(alphas), early_stop=False)
    s = channel.bpsk(cw)
    z = -s * res.llr
    return float(np.mean(np.logaddexp(0.0, z)))


def ber_of(code, cw, llr, method, iters, **kw):
    res = decoders.decode(code, llr, method, iters=iters, **kw)
    return float(np.mean(res.bits != cw))


def tune_alpha(code, ebn0_db, iters=15, grid=None, n_frames=300, seed=1):
    grid = np.linspace(0.5, 1.0, 11) if grid is None else grid
    rng = np.random.default_rng(seed)
    cw, llr = make_training_batch(code, ebn0_db, n_frames, rng)
    scores = [ber_of(code, cw, llr, "nms", iters, alpha=a) for a in grid]
    return float(grid[int(np.argmin(scores))]), list(map(float, grid)), scores


def learn_alpha_per_iter(code, ebn0_db, iters=10, n_frames=200, sweeps=2, seed=2,
                         candidates=(-0.1, -0.05, 0.05, 0.1), init=0.8):
    rng = np.random.default_rng(seed)
    cw, llr = make_training_batch(code, ebn0_db, n_frames, rng)
    alphas = np.full(iters, init)
    best = surrogate_loss(code, cw, llr, alphas)
    history = [best]
    for _ in range(sweeps):
        for i in range(iters):
            for d in candidates:
                trial = alphas.copy()
                trial[i] = np.clip(trial[i] + d, 0.3, 1.2)
                loss = surrogate_loss(code, cw, llr, trial)
                if loss < best - 1e-9:
                    best, alphas = loss, trial
        history.append(best)
    return alphas, history
