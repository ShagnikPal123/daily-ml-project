"""Experiments: denoiser sweep, adaptive-canceller study, noisy-classification study. Each returns a dict of metrics."""
import numpy as np
from . import signals, spectral, adaptive, metrics, classifier
from .stft import power_spectrogram_db


def _rng(seed):
    return np.random.default_rng(seed)


# ---- 1. single-channel denoising sweep -------------------------------------
def exp_denoise(seed=0, per_class=3, snrs=(-5, 0, 5, 10, 15), noises=("white", "pink", "babble")):
    rng = _rng(seed)
    clips, _ = signals.make_dataset(per_class, rng)
    methods = {"specsub": classifier.DENOISERS["specsub"], "wiener": classifier.DENOISERS["wiener"]}
    res = {}
    for nk in noises:
        for snr in snrs:
            acc = {m: [] for m in ["noisy", *methods]}
            for c in clips:
                noisy, _ = signals.add_noise(c, nk, snr, rng)
                outs = {"noisy": noisy, **{m: f(noisy) for m, f in methods.items()}}
                for m, o in outs.items():
                    acc[m].append((metrics.si_sdr(c, o), metrics.segmental_snr(c, o), metrics.log_spectral_distance(c, o)))
            for m, v in acc.items():
                v = np.array(v).mean(0)
                res[f"{nk}/{snr}/{m}"] = dict(si_sdr=float(v[0]), seg_snr=float(v[1]), lsd=float(v[2]))
    return dict(snrs=list(snrs), noises=list(noises), methods=["noisy", *methods], table=res)


# ---- 2. adaptive noise cancellation ----------------------------------------
def anc_scenario(rng, n=16000, L_path=16, source="white", sensor_snr_db=40, switch=None):
    clean = signals.make_clip("vowel", rng, n)
    src = signals.NOISES[source](rng, n)
    src = src / np.std(src)
    h = adaptive.make_noise_path(rng, L_path)
    h2 = adaptive.make_noise_path(rng, L_path) if switch else h
    d = np.empty(n)
    full1 = np.convolve(src, h)[:n]
    full2 = np.convolve(src, h2)[:n]
    d[:] = full1
    if switch:
        d[switch:] = full2[switch:]
    noise_at_mic = d.copy() * 1.0
    primary = clean + 0.7 * noise_at_mic
    ref = src + 10 ** (-sensor_snr_db / 20) * rng.standard_normal(n)
    return clean, primary, ref, h, h2, 0.7


def erle_db(noise_in, resid, win=2000):
    """Echo/noise return loss enhancement over the last `win` samples."""
    return 10 * np.log10(np.sum(noise_in[-win:] ** 2) / (np.sum(resid[-win:] ** 2) + 1e-20))


def exp_adaptive(seed=1, L=16, n=16000):
    rng = _rng(seed)
    clean, primary, ref, h, _, g = anc_scenario(rng, n)
    out = {"curves": {}, "final": {}}
    cfgs = {"lms": dict(L=L, mu=0.002), "nlms": dict(L=L, mu=0.5), "rls": dict(L=L, lam=0.999)}
    for name, kw in cfgs.items():
        e, w, W = adaptive.ALGOS[name](primary, ref, **kw)
        mis = [adaptive.misalignment_db(wi, g * h) for wi in W]
        out["curves"][name] = dict(mis=mis, power=adaptive.smooth_power_db(e - clean).tolist()[::40])
        out["final"][name] = dict(misalign_db=float(mis[-1]), si_sdr=float(metrics.si_sdr(clean, e)),
                                  si_sdr_in=float(metrics.si_sdr(clean, primary)))
    return out


def exp_adaptive_sweeps(seed=2, n=12000):
    rng = _rng(seed)
    clean, primary, ref, h, _, g = anc_scenario(rng, n)
    sweeps = {"mu_lms": {}, "mu_nlms": {}, "L_nlms": {}, "lam_rls": {}}
    for mu in [0.0005, 0.001, 0.002, 0.004, 0.008]:
        e, w, W = adaptive.lms(primary, ref, L=16, mu=mu)
        sweeps["mu_lms"][mu] = metrics.si_sdr(clean, e)
    for mu in [0.05, 0.1, 0.25, 0.5, 1.0, 1.5]:
        e, w, W = adaptive.nlms(primary, ref, L=16, mu=mu)
        sweeps["mu_nlms"][mu] = metrics.si_sdr(clean, e)
    for L in [4, 8, 16, 32, 64]:
        e, w, W = adaptive.nlms(primary, ref, L=L, mu=0.5)
        sweeps["L_nlms"][L] = metrics.si_sdr(clean, e)
    for lam in [0.99, 0.995, 0.999, 0.9998]:
        e, w, W = adaptive.rls(primary, ref, L=16, lam=lam)
        sweeps["lam_rls"][lam] = metrics.si_sdr(clean, e)
    return {k: {str(a): float(b) for a, b in v.items()} for k, v in sweeps.items()}


def exp_tracking(seed=3, n=16000):
    """Path change mid-run: forgetting factor / step size vs tracking."""
    rng = _rng(seed)
    clean, primary, ref, h, h2, g = anc_scenario(rng, n, switch=n // 2)
    res = {}
    for name, kw in {"nlms": dict(L=16, mu=0.5), "rls_lam0.999": dict(L=16, lam=0.999), "rls_lam0.99": dict(L=16, lam=0.99)}.items():
        fn = adaptive.rls if name.startswith("rls") else adaptive.nlms
        e, w, W = fn(primary, ref, **kw)
        seg = (clean - e)
        res[name] = dict(power=adaptive.smooth_power_db(e - clean).tolist()[::40],
                         post_switch_si_sdr=float(metrics.si_sdr(clean[n // 2:], e[n // 2:])))
    return res


def exp_hum(seed=4, n=16000):
    """Mains hum cancellation: reference-based adaptive vs fixed notch comb vs spectral subtraction."""
    rng = _rng(seed)
    clean = signals.make_clip("vowel", rng, n)
    hum_ref = signals.hum_noise(rng, n)
    h = adaptive.make_noise_path(rng, 8)
    hum_mic = 1.5 * np.convolve(hum_ref, h)[:n]
    primary = clean + hum_mic
    ref = hum_ref + 0.01 * rng.standard_normal(n)
    out = {"input": metrics.si_sdr(clean, primary)}
    e, _, _ = adaptive.nlms(primary, ref, L=16, mu=0.5)
    out["nlms_ref"] = metrics.si_sdr(clean, e)
    e2, _, _ = adaptive.rls(primary, ref, L=16, lam=0.999)
    out["rls_ref"] = metrics.si_sdr(clean, e2)
    out["notch_comb"] = metrics.si_sdr(clean, spectral.notch_comb(primary, signals.FS, 60, 5))
    out["specsub"] = metrics.si_sdr(clean, spectral.spectral_subtraction(primary, tracking=True))
    return {k: float(v) for k, v in out.items()}


# ---- 3. classification under noise ------------------------------------------
def exp_classify(seed=5, train_per_class=30, test_per_class=12, snrs=(-5, 0, 5, 10, 20), noises=("white", "babble"),
                 model_kind="svm"):
    rng = _rng(seed)
    Xtr, ytr = signals.make_dataset(train_per_class, rng)
    Xte, yte = signals.make_dataset(test_per_class, rng)
    clean_model = classifier.make_model(model_kind).fit(classifier.featurize(Xtr), ytr)
    res = {"clean_test_acc": float(np.mean(clean_model.predict(classifier.featurize(Xte)) == yte)), "cells": {}}
    for nk in noises:
        # multi-condition training: noisy copies at random SNRs
        mt_clips, mt_y = [], []
        for c, y in zip(Xtr, ytr):
            s = rng.choice(snrs)
            mt_clips.append(signals.add_noise(c, nk, float(s), rng)[0])
            mt_y.append(y)
        mc_model = classifier.make_model(model_kind).fit(classifier.featurize(np.array(mt_clips)), np.array(mt_y))
        for snr in snrs:
            for den in classifier.DENOISERS:
                r1, r2 = _rng(seed * 100 + int(snr) + 50), _rng(seed * 100 + int(snr) + 50)  # same noise draws
                a_clean = classifier.evaluate(clean_model, Xte, yte, nk, snr, r1, den)
                a_mc = classifier.evaluate(mc_model, Xte, yte, nk, snr, r2, den)
                res["cells"][f"{nk}/{snr}/{den}"] = dict(train_clean=a_clean, train_multicond=a_mc)
    return res


def model_comparison(seed=6, snr=5, noise="white"):
    rng = _rng(seed)
    Xtr, ytr = signals.make_dataset(25, rng)
    Xte, yte = signals.make_dataset(10, rng)
    Ftr = classifier.featurize(Xtr)
    out = {}
    for kind in ["centroid", "logreg", "svm"]:
        m = classifier.make_model(kind).fit(Ftr, ytr)
        out[kind] = {den: classifier.evaluate(m, Xte, yte, noise, snr, _rng(seed + 9), den) for den in classifier.DENOISERS}
    return out


def demo_signals(seed=7, n=8000, snr=5):
    rng = _rng(seed)
    clean = signals.make_clip("siren", rng, n)
    noisy, _ = signals.add_noise(clean, "white", snr, rng)
    return dict(clean=clean, noisy=noisy, specsub=classifier.DENOISERS["specsub"](noisy),
                wiener=classifier.DENOISERS["wiener"](noisy))


# ---- 4. VAD-gated noise estimation vs min-statistics tracking --------------
def exp_vad(seed=8, snrs=(0, 5, 10, 15), per_class=3):
    """Bursty sounds with long silences: compare noise-PSD strategies (initial frames, min-stats, VAD-gated)."""
    from . import vad
    from .stft import stft
    rng = _rng(seed)
    clips, _ = signals.make_dataset(per_class, rng)
    out = {}
    for snr in snrs:
        acc = {k: [] for k in ["noisy", "minstats", "vad", "vad_precision", "vad_recall"]}
        for c in clips:
            x = np.concatenate([np.zeros(2000), c[:4000], np.zeros(2000)])  # sound burst inside silence
            truth_t = np.zeros(len(x), bool); truth_t[2000:6000] = True
            noisy, noise = signals.add_noise(np.where(truth_t, x, 0) + 1e-9, "white", snr, rng)
            noisy = noisy * 1.0
            S = stft(noisy)
            act = vad.detect(S)
            truth_f = np.array([truth_t[max(0, i * 64 - 256):max(0, i * 64)].any() for i in range(S.shape[0])])
            sc = vad.detection_scores(act, truth_f)
            acc["vad_precision"].append(sc["precision"]); acc["vad_recall"].append(sc["recall"])
            acc["noisy"].append(metrics.si_sdr(x, noisy))
            acc["minstats"].append(metrics.si_sdr(x, spectral.wiener_dd(noisy, tracking=True)))
            acc["vad"].append(metrics.si_sdr(x, vad.vad_wiener(noisy)))
        out[str(snr)] = {k: float(np.mean(v)) for k, v in acc.items()}
    return out
