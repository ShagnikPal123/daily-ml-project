"""STFT / inverse STFT with weighted overlap-add (perfect reconstruction for Hann, 75% overlap)."""
import numpy as np


def hann(n):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)  # periodic


def stft(x, n_fft=256, hop=64):
    w = hann(n_fft)
    pad = n_fft
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])
    n_frames = 1 + (len(xp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(xp[idx] * w, axis=1)  # (frames, bins)


def istft(S, length, n_fft=256, hop=64):
    w = hann(n_fft)
    n_frames = S.shape[0]
    out = np.zeros(hop * (n_frames - 1) + n_fft)
    norm = np.zeros_like(out)
    frames = np.fft.irfft(S, n=n_fft, axis=1) * w
    for i in range(n_frames):
        out[i * hop:i * hop + n_fft] += frames[i]
        norm[i * hop:i * hop + n_fft] += w ** 2
    out /= np.maximum(norm, 1e-8)
    return out[n_fft:n_fft + length]


def power_spectrogram_db(x, n_fft=256, hop=64):
    S = stft(x, n_fft, hop)
    return 10 * np.log10(np.abs(S) ** 2 + 1e-10)
