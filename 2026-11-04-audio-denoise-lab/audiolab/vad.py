"""Energy/flatness voice-activity detection used to gate noise-PSD updates (VAD-driven noise estimation)."""
import numpy as np
from .stft import stft, istft


def frame_energy_db(S):
    return 10 * np.log10(np.mean(np.abs(S) ** 2, axis=1) + 1e-12)


def detect(S, margin_db=6.0, hangover=4, init_frames=6, skip=4):
    """Frame is 'active' if energy exceeds a slowly adapted noise floor by margin_db; hangover bridges short gaps."""
    e = frame_energy_db(S)
    # skip the first frames: they overlap the STFT zero padding and under-estimate the floor
    floor = np.mean(e[skip:skip + init_frames])
    active = np.zeros(len(e), bool)
    hang = 0
    for t, et in enumerate(e):
        if t < skip:
            continue
        if et > floor + margin_db:
            active[t] = True
            hang = hangover
        elif hang > 0:
            active[t] = True
            hang -= 1
        else:
            floor = 0.95 * floor + 0.05 * et  # update only during noise
    return active


def gated_noise_psd(S, active, alpha=0.9):
    """Recursive average of |S|^2 over inactive frames; held constant while speech/sound is active."""
    P = np.abs(S) ** 2
    est = np.zeros_like(P)
    cur = P[:min(6, len(P))].mean(0)
    for t in range(len(P)):
        if not active[t]:
            cur = alpha * cur + (1 - alpha) * P[t]
        est[t] = cur
    return est


def vad_wiener(x, n_fft=256, hop=64, **kw):
    from .spectral import wiener_dd
    S = stft(x, n_fft, hop)
    return wiener_dd(x, n_fft, hop, noise_psd=gated_noise_psd(S, detect(S)), **kw)


def vad_specsub(x, n_fft=256, hop=64, **kw):
    from .spectral import spectral_subtraction
    S = stft(x, n_fft, hop)
    return spectral_subtraction(x, n_fft, hop, noise_psd=gated_noise_psd(S, detect(S)), **kw)


def detection_scores(active, truth):
    tp = np.sum(active & truth); fp = np.sum(active & ~truth); fn = np.sum(~active & truth)
    return dict(precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)))
