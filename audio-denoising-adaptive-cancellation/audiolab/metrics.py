"""Objective speech/audio quality metrics."""
import numpy as np


def snr_db(clean, est):
    err = clean - est
    return 10 * np.log10((np.sum(clean ** 2) + 1e-20) / (np.sum(err ** 2) + 1e-20))


def si_sdr(clean, est):
    """Scale-invariant SDR (Le Roux et al., 2019)."""
    alpha = np.dot(est, clean) / (np.dot(clean, clean) + 1e-20)
    target = alpha * clean
    noise = est - target
    return 10 * np.log10((np.sum(target ** 2) + 1e-20) / (np.sum(noise ** 2) + 1e-20))


def segmental_snr(clean, est, frame=256, lo=-10, hi=35):
    vals = []
    for i in range(0, len(clean) - frame, frame):
        c, e = clean[i:i + frame], est[i:i + frame]
        if np.sum(c ** 2) < 1e-8:
            continue
        vals.append(np.clip(snr_db(c, e), lo, hi))
    return float(np.mean(vals)) if vals else float("nan")


def log_spectral_distance(clean, est, n_fft=256, hop=64):
    from .stft import power_spectrogram_db
    a = power_spectrogram_db(clean, n_fft, hop)
    b = power_spectrogram_db(est, n_fft, hop)
    return float(np.mean(np.sqrt(np.mean((a - b) ** 2, axis=1))))
