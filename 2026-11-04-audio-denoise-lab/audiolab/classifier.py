"""Tiny classifier zoo + evaluation under noise with optional front-end denoising."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid

from .features import clip_embedding
from .signals import add_noise
from . import spectral


def featurize(clips):
    return np.array([clip_embedding(c) for c in clips])


def make_model(kind="svm"):
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=5.0, gamma="scale"))
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if kind == "centroid":
        return make_pipeline(StandardScaler(), NearestCentroid())
    raise ValueError(kind)


DENOISERS = {
    "none": lambda x: x,
    "specsub": lambda x: spectral.spectral_subtraction(x, tracking=True),
    "wiener": lambda x: spectral.wiener_dd(x, tracking=True),
}


def noisy_set(clips, kind, snr_db, rng, denoiser="none"):
    f = DENOISERS[denoiser]
    out = []
    for c in clips:
        noisy, _ = add_noise(c, kind, snr_db, rng)
        out.append(f(noisy))
    return np.array(out)


def evaluate(model, clips, y, noise_kind, snr_db, rng, denoiser="none"):
    X = featurize(noisy_set(clips, noise_kind, snr_db, rng, denoiser))
    return float(np.mean(model.predict(X) == y))
