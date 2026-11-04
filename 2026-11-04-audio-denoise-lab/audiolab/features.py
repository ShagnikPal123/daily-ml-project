"""MFCC front-end implemented from scratch (mel filterbank + DCT-II)."""
import numpy as np
from .stft import stft


def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def mel_filterbank(n_mels, n_fft, fs):
    pts = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(fs / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


def dct2(x, n_out):
    N = x.shape[-1]
    k = np.arange(n_out)[:, None]
    n = np.arange(N)[None, :]
    basis = np.cos(np.pi * k * (2 * n + 1) / (2 * N)) * np.sqrt(2.0 / N)
    basis[0] /= np.sqrt(2)
    return x @ basis.T


def mfcc(x, fs=8000, n_fft=256, hop=128, n_mels=24, n_mfcc=12):
    S = np.abs(stft(x, n_fft, hop)) ** 2
    mel = np.log(S @ mel_filterbank(n_mels, n_fft, fs).T + 1e-8)
    return dct2(mel, n_mfcc)


def clip_embedding(x, **kw):
    """Utterance-level feature: mean, std and mean |delta| of the MFCC trajectory + spectral flatness stats."""
    M = mfcc(x, **kw)
    d = np.diff(M, axis=0)
    S = np.abs(stft(x, 256, 128)) ** 2 + 1e-12
    flat = np.exp(np.mean(np.log(S), axis=1)) / np.mean(S, axis=1)
    return np.concatenate([M.mean(0), M.std(0), np.abs(d).mean(0), [flat.mean(), flat.std()]])
