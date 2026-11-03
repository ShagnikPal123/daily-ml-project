"""Iterative message-passing decoders for LDPC codes (flooding schedule, batch-vectorised).

Variable-to-check and check-to-variable messages are LLRs.

  SPA (sum-product):      c2v_j = 2 atanh( prod_{i!=j} tanh(v2c_i / 2) )
  Min-sum:                c2v_j = prod_{i!=j} sign(v2c_i) * min_{i!=j} |v2c_i|
  Normalised min-sum:     c2v_j <- alpha * c2v_j           (alpha < 1 corrects over-estimation)
  Offset min-sum:         c2v_j <- sign * max(|c2v_j| - beta, 0)

`alpha` may be a scalar or a per-iteration array.  `qfn` is an optional quantiser applied
to the channel LLRs and every message, which emulates fixed-point hardware.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class DecodeResult:
    bits: np.ndarray        # (B,n) hard decisions
    llr: np.ndarray         # (B,n) posterior LLRs
    iters: np.ndarray       # (B,) iterations until the syndrome became zero (max_iters if never)
    converged: np.ndarray   # (B,) bool


def _check_spa(v2c):
    t = np.tanh(np.clip(v2c, -30, 30) / 2.0)
    B, m, dc = t.shape
    pre = np.ones_like(t)
    suf = np.ones_like(t)
    for j in range(1, dc):
        pre[:, :, j] = pre[:, :, j - 1] * t[:, :, j - 1]
    for j in range(dc - 2, -1, -1):
        suf[:, :, j] = suf[:, :, j + 1] * t[:, :, j + 1]
    prod = np.clip(pre * suf, -1 + 1e-12, 1 - 1e-12)
    return 2.0 * np.arctanh(prod)


def _check_minsum(v2c, alpha=1.0, beta=0.0):
    mag = np.abs(v2c)
    sgn = np.where(v2c < 0, -1.0, 1.0)
    total_sign = sgn.prod(axis=2, keepdims=True)
    part = np.partition(mag, 1, axis=2)
    min1, min2 = part[:, :, 0:1], part[:, :, 1:2]
    is_min = mag <= min1
    # if two entries tie for the minimum, min2 == min1, so the result is still right
    excl = np.where(is_min, min2, min1)
    excl = np.maximum(alpha * excl - beta, 0.0)
    return total_sign * sgn * excl


def decode(code, llr, method="nms", iters=20, alpha=0.8, beta=0.0, qfn=None, early_stop=True):
    """Run the chosen decoder on a batch of channel LLRs (B,n)."""
    llr = np.asarray(llr, dtype=np.float64)
    if qfn is not None:
        llr = qfn(llr)
    B, n = llr.shape
    m, dc = code.m, code.dc
    alphas = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (iters,)) if np.ndim(alpha) else np.full(iters, alpha)
    c2v = np.zeros((B, m * dc))
    post = llr.copy()
    done = np.zeros(B, dtype=bool)
    iters_used = np.full(B, iters)
    final = (post < 0).astype(np.uint8)
    final_post = post.copy()
    synd0 = code.syndrome(final).any(axis=1)
    done = ~synd0
    iters_used[done] = 0
    for it in range(iters):
        if early_stop and done.all():
            break
        # variable -> check
        total = llr + c2v[:, code.Vn].sum(axis=2)                    # (B,n) posterior (pre-this-iter)
        edge_var = total[:, code.Cn.reshape(-1)]                      # (B, m*dc)
        v2c = edge_var - c2v
        if qfn is not None:
            v2c = qfn(v2c)
        v2c = v2c.reshape(B, m, dc)
        # check -> variable
        if method == "spa":
            new = _check_spa(v2c)
        elif method == "ms":
            new = _check_minsum(v2c)
        elif method == "nms":
            new = _check_minsum(v2c, alpha=alphas[it])
        elif method == "oms":
            new = _check_minsum(v2c, beta=beta)
        else:
            raise ValueError(f"unknown method {method}")
        c2v = new.reshape(B, m * dc)
        if qfn is not None:
            c2v = qfn(c2v)
        post = llr + c2v[:, code.Vn].sum(axis=2)
        bits = (post < 0).astype(np.uint8)
        ok = ~code.syndrome(bits).any(axis=1)
        newly = ok & ~done
        iters_used[newly] = it + 1
        # freeze frames that just converged; keep updating the rest
        upd = ~done
        final[upd] = bits[upd]
        final_post[upd] = post[upd]
        done |= ok
    return DecodeResult(final, final_post, iters_used, done)
