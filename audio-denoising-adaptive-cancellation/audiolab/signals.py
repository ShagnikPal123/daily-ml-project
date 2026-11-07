"""Synthetic sound classes and noise models (all generated, no datasets needed)."""
import numpy as np
from scipy import signal

FS = 8000
CLASSES = ["siren", "vowel", "click", "rain", "chirp"]


def _env(n, attack=0.02, release=0.05, fs=FS):
    e = np.ones(n)
    a, r = int(attack * fs), int(release * fs)
    e[:a] = np.linspace(0, 1, a)
    e[-r:] = np.linspace(1, 0, r)
    return e


def siren(rng, n=FS, fs=FS):
    t = np.arange(n) / fs
    f0 = rng.uniform(500, 800)
    rate = rng.uniform(1.5, 3.5)
    dev = rng.uniform(150, 300)
    phase = 2 * np.pi * (f0 * t - dev / (2 * np.pi * rate) * np.cos(2 * np.pi * rate * t))
    return np.sin(phase) * _env(n)


def vowel(rng, n=FS, fs=FS):
    """Glottal-like pulse train shaped by two formant resonators."""
    t = np.arange(n) / fs
    f0 = rng.uniform(100, 180)
    pulses = (np.diff(np.floor(t * f0), prepend=0) > 0).astype(float)
    x = pulses
    for fc, bw in [(rng.uniform(400, 800), 80), (rng.uniform(1100, 2200), 120)]:
        r = np.exp(-np.pi * bw / fs)
        th = 2 * np.pi * fc / fs
        x = signal.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
    x /= np.max(np.abs(x)) + 1e-12
    return x * _env(n)


def click(rng, n=FS, fs=FS):
    x = np.zeros(n)
    period = int(fs * rng.uniform(0.08, 0.2))
    ring = np.exp(-np.arange(80) / rng.uniform(8, 16)) * np.sin(2 * np.pi * rng.uniform(1500, 3000) * np.arange(80) / fs)
    for k in range(rng.integers(0, period), n - 80, period):
        x[k:k + 80] += ring
    return x / (np.max(np.abs(x)) + 1e-12)


def rain(rng, n=FS, fs=FS):
    """Band-limited noise with random amplitude modulation (broadband, textured)."""
    w = rng.standard_normal(n)
    b, a = signal.butter(4, [rng.uniform(1200, 1800) / (fs / 2), rng.uniform(2800, 3600) / (fs / 2)], "bandpass")
    x = signal.lfilter(b, a, w)
    am = 1 + 0.8 * np.sin(2 * np.pi * rng.uniform(8, 20) * np.arange(n) / fs + rng.uniform(0, 6))
    x = x * am
    return x / (np.max(np.abs(x)) + 1e-12)


def chirp(rng, n=FS, fs=FS):
    t = np.arange(n) / fs
    f_lo = rng.uniform(200, 500)
    f_hi = rng.uniform(2000, 3500)
    x = signal.chirp(t, f_lo, t[-1], f_hi, method="logarithmic" if rng.random() < 0.5 else "linear")
    return x * _env(n)


GENERATORS = {"siren": siren, "vowel": vowel, "click": click, "rain": rain, "chirp": chirp}


def make_clip(label, rng, n=FS):
    x = GENERATORS[label](rng, n)
    return x / (np.sqrt(np.mean(x ** 2)) + 1e-12)  # unit RMS


def make_dataset(per_class, rng, n=FS):
    X, y = [], []
    for ci, c in enumerate(CLASSES):
        for _ in range(per_class):
            X.append(make_clip(c, rng, n))
            y.append(ci)
    return np.array(X), np.array(y)


# ---- noise models ---------------------------------------------------------

def white_noise(rng, n):
    return rng.standard_normal(n)


def pink_noise(rng, n):
    """1/f noise via spectral shaping."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=float)
    f[0] = 1
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x


def hum_noise(rng, n, fs=FS, f0=60.0, harmonics=5):
    t = np.arange(n) / fs
    x = np.zeros(n)
    for h in range(1, harmonics + 1):
        x += rng.uniform(0.3, 1) / h * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28))
    return x + 0.05 * rng.standard_normal(n)


def babble_noise(rng, n, fs=FS):
    x = sum(vowel(rng, n, fs) * rng.uniform(0.5, 1) for _ in range(6))
    return x + 0.1 * rng.standard_normal(n)


NOISES = {"white": white_noise, "pink": pink_noise, "hum": hum_noise, "babble": babble_noise}


def scale_to_snr(clean, noise, snr_db):
    pc = np.mean(clean ** 2)
    pn = np.mean(noise ** 2) + 1e-20
    return noise * np.sqrt(pc / (pn * 10 ** (snr_db / 10)))


def add_noise(clean, kind, snr_db, rng):
    noise = NOISES[kind](rng, len(clean))
    noise = scale_to_snr(clean, noise, snr_db)
    return clean + noise, noise
