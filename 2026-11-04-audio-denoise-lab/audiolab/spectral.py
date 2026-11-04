"""Single-channel spectral-domain denoisers: spectral subtraction, Wiener (decision-directed), min-statistics tracking."""
import numpy as np
from .stft import stft, istft


def noise_psd_initial(S, n_frames=8):
    """Assume the first frames are noise-only (classic voice-activity-free estimate)."""
    return np.mean(np.abs(S[:n_frames]) ** 2, axis=0)


def noise_psd_minstats(S, win=25, bias=1.5, alpha=0.85):
    """Running-minimum noise tracker (Martin-style, simplified): smooth |S|^2 then take windowed minimum."""
    P = np.abs(S) ** 2
    sm = np.empty_like(P)
    sm[0] = P[0]
    for t in range(1, len(P)):
        sm[t] = alpha * sm[t - 1] + (1 - alpha) * P[t]
    out = np.empty_like(P)
    for t in range(len(P)):
        out[t] = bias * sm[max(0, t - win):t + 1].min(axis=0)
    return out


def spectral_subtraction(x, n_fft=256, hop=64, over=2.0, floor=0.05, noise_psd=None, tracking=False):
    S = stft(x, n_fft, hop)
    if noise_psd is None:
        noise_psd = noise_psd_minstats(S) if tracking else noise_psd_initial(S)[None, :]
    P = np.abs(S) ** 2
    clean_p = np.maximum(P - over * noise_psd, floor * noise_psd)
    gain = np.sqrt(clean_p / (P + 1e-20))
    return istft(gain * S, len(x), n_fft, hop)


def wiener_dd(x, n_fft=256, hop=64, alpha=0.98, gmin=0.1, noise_psd=None, tracking=False):
    """Wiener filter with decision-directed a-priori SNR (Ephraim-Malah)."""
    S = stft(x, n_fft, hop)
    if noise_psd is None:
        noise_psd = noise_psd_minstats(S) if tracking else noise_psd_initial(S)[None, :]
    noise_psd = np.broadcast_to(noise_psd, S.shape) + 1e-12
    P = np.abs(S) ** 2
    post = P / noise_psd                       # a-posteriori SNR
    out = np.empty_like(S)
    prev_clean_p = np.zeros(S.shape[1])
    for t in range(S.shape[0]):
        xi = alpha * prev_clean_p / noise_psd[t] + (1 - alpha) * np.maximum(post[t] - 1, 0)
        g = np.maximum(xi / (1 + xi), gmin)
        out[t] = g * S[t]
        prev_clean_p = np.abs(out[t]) ** 2
    return istft(out, len(x), n_fft, hop)


def notch_comb(x, fs, f0=60.0, harmonics=5, q=30.0):
    """Cascade of IIR notch filters at mains harmonics (the classic non-adaptive baseline)."""
    from scipy import signal
    y = x
    for h in range(1, harmonics + 1):
        f = f0 * h
        if f >= fs / 2:
            break
        b, a = signal.iirnotch(f, q, fs)
        y = signal.lfilter(b, a, y)
    return y
