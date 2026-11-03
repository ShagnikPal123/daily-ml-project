"""Monte-Carlo BER/FER engine with pluggable coding schemes."""
from dataclasses import dataclass, field
import numpy as np
from . import channel, decoders
from .convolutional import ConvCode
from .ldpc import LDPCCode


@dataclass
class Point:
    ebn0_db: float
    ber: float
    fer: float
    bit_errors: int
    frame_errors: int
    n_bits: int
    n_frames: int
    avg_iters: float = float("nan")


class Uncoded:
    name = "Uncoded BPSK"
    rate = 1.0

    def __init__(self, frame_len=256):
        self.k = frame_len

    def run(self, ebn0_db, B, rng, fading=False):
        bits = rng.integers(0, 2, size=(B, self.k), dtype=np.uint8)
        sigma = channel.sigma_from_ebn0(ebn0_db, 1.0)
        if fading:
            y, h = channel.rayleigh(channel.bpsk(bits), sigma, rng)
            llr = channel.rayleigh_llr(y, h, sigma)
        else:
            llr = channel.awgn_llr(channel.awgn(channel.bpsk(bits), sigma, rng), sigma)
        dec = channel.hard_decision(llr)
        return (dec != bits).sum(), (dec != bits).any(1).sum(), self.k, None


class ConvScheme:
    def __init__(self, K=7, gens=(0o133, 0o171), k=252, soft=True):
        self.code = ConvCode(K, gens)
        self.k, self.soft = k, soft
        self.rate = self.code.rate(k)
        self.name = f"Conv K={K} {'soft' if soft else 'hard'} Viterbi"

    def run(self, ebn0_db, B, rng, fading=False):
        info = rng.integers(0, 2, size=(B, self.k), dtype=np.uint8)
        cw = self.code.encode(info)
        sigma = channel.sigma_from_ebn0(ebn0_db, self.rate)
        s = channel.bpsk(cw)
        if fading:
            y, h = channel.rayleigh(s, sigma, rng)
            llr = channel.rayleigh_llr(y, h, sigma)
        else:
            llr = channel.awgn_llr(channel.awgn(s, sigma, rng), sigma)
        dec = self.code.viterbi(llr if self.soft else channel.hard_decision(llr), soft=self.soft)
        err = dec != info
        return err.sum(), err.any(1).sum(), self.k, None


class LDPCScheme:
    def __init__(self, code=None, method="nms", iters=20, alpha=0.8, beta=0.0, qfn=None, label=None):
        self.code = code or LDPCCode()
        self.method, self.iters, self.alpha, self.beta, self.qfn = method, iters, alpha, beta, qfn
        self.k, self.rate = self.code.k, self.code.rate
        self.name = label or f"LDPC {method.upper()} it={iters}"

    def run(self, ebn0_db, B, rng, fading=False):
        c = self.code
        info = rng.integers(0, 2, size=(B, c.k), dtype=np.uint8)
        cw = c.encode(info)
        sigma = channel.sigma_from_ebn0(ebn0_db, c.rate)
        s = channel.bpsk(cw)
        if fading:
            y, h = channel.rayleigh(s, sigma, rng)
            llr = channel.rayleigh_llr(y, h, sigma)
        else:
            llr = channel.awgn_llr(channel.awgn(s, sigma, rng), sigma)
        res = decoders.decode(c, llr, self.method, iters=self.iters, alpha=self.alpha,
                              beta=self.beta, qfn=self.qfn)
        err = c.extract_info(res.bits) != info
        return err.sum(), err.any(1).sum(), c.k, float(res.iters.mean())


def ber_curve(scheme, ebn0_list, min_frame_errors=30, max_frames=4000, batch=100, seed=0, fading=False):
    rng = np.random.default_rng(seed)
    points = []
    for e in ebn0_list:
        be = fe = nb = nf = 0
        it_sum = 0.0
        while fe < min_frame_errors and nf < max_frames:
            b, f, k, it = scheme.run(e, batch, rng, fading=fading)
            be += int(b); fe += int(f); nb += batch * k; nf += batch
            if it is not None:
                it_sum += it * batch
        points.append(Point(float(e), be / nb, fe / nf, be, fe, nb, nf,
                            it_sum / nf if it_sum else float("nan")))
    return points


def points_to_dict(points):
    return [dict(ebn0_db=p.ebn0_db, ber=p.ber, fer=p.fer, bit_errors=p.bit_errors,
                 frame_errors=p.frame_errors, n_frames=p.n_frames, avg_iters=p.avg_iters) for p in points]
