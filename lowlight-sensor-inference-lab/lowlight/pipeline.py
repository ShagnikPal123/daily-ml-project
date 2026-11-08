"""Layer 3: scene -> sensor -> ISP (denoiser) -> classifier, with train/eval protocol.

``train_mode='matched'`` trains the classifier on denoised noisy captures at the
deployment exposure (noise-aware); ``'clean'`` trains on ideal images (camera-unaware).
Noise realisations are seeded per (seed, split) so competing methods see *identical*
noisy test images, which makes paired comparisons across methods valid.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import imgops as io
from .classifiers import make_classifier
from .denoisers import make_denoiser
from .sensor import Sensor, SensorConfig


@dataclass
class PipelineConfig:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    denoiser: str = "identity"
    denoiser_kw: dict = field(default_factory=dict)
    classifier: str = "softmax"
    classifier_kw: dict = field(default_factory=dict)
    train_mode: str = "matched"
    n_noise_draws: int = 2

    def __post_init__(self):
        if self.train_mode not in ("matched", "clean"):
            raise ValueError("train_mode must be 'matched' or 'clean'")
        if self.n_noise_draws < 1:
            raise ValueError("n_noise_draws must be >= 1")

    def label(self) -> str:
        return f"{self.denoiser}/{self.classifier}/{self.train_mode}"


class Pipeline:
    def __init__(self, cfg: PipelineConfig, shape=(16, 16), seed: int = 0):
        self.cfg, self.shape, self.seed = cfg, tuple(shape), seed
        self.sensor = Sensor(cfg.sensor, self.shape, seed=1000 + seed)  # same fixed pattern for all methods
        self.denoiser = make_denoiser(cfg.denoiser, self.sensor, **cfg.denoiser_kw)
        self.clf = make_classifier(cfg.classifier, seed=seed, **cfg.classifier_kw)
        self.fitted = False

    # ---- stages ----
    def acquire(self, scenes, rng) -> np.ndarray:
        return self.sensor.acquire(scenes, rng)

    def isp(self, est) -> np.ndarray:
        """Denoise and clip to the displayable range [0, 1]."""
        return np.clip(self.denoiser(est), 0.0, 1.0)

    def features(self, est) -> np.ndarray:
        out = self.isp(est)
        return out.reshape(len(out), -1)

    # ---- protocol ----
    def fit(self, scenes, labels, rng) -> "Pipeline":
        n = self.cfg.n_noise_draws
        noisy = np.concatenate([self.acquire(scenes, rng) for _ in range(n)])
        clean = np.tile(scenes, (n, 1, 1))
        self.denoiser.fit(clean, noisy)
        if self.cfg.train_mode == "matched":
            X, y = self.features(noisy), np.tile(labels, n)
        else:
            X, y = scenes.reshape(len(scenes), -1), np.asarray(labels)
        self.clf.fit(X, y)
        self.fitted = True
        return self

    def evaluate(self, scenes, labels, rng) -> dict:
        if not self.fitted:
            raise RuntimeError("fit() before evaluate()")
        est = self.acquire(scenes, rng)
        out = self.isp(est)
        pred = self.clf.predict(out.reshape(len(out), -1))
        labels = np.asarray(labels)
        k = 10
        conf = np.zeros((k, k), dtype=int)
        np.add.at(conf, (labels, pred), 1)
        per_class = [float(conf[i, i] / max(conf[i].sum(), 1)) for i in range(k)]
        return {
            "acc": float(np.mean(pred == labels)),
            "psnr": float(io.psnr(scenes, out).mean()),
            "ssim": float(io.ssim(scenes, out).mean()),
            "psnr_raw": float(io.psnr(scenes, np.clip(est, 0, 1)).mean()),
            "per_class": per_class,
            "confusion": conf.tolist(),
            "pred": pred.tolist(),
        }


def run_trial(scenes, labels, splits, cfg: PipelineConfig, seed: int) -> dict:
    """One full train/evaluate cycle with seed-derived, method-independent noise streams."""
    pipe = Pipeline(cfg, scenes.shape[1:], seed)
    tr, te = splits["train"], splits["test"]
    pipe.fit(scenes[tr], labels[tr], np.random.default_rng([seed, 1]))
    res = pipe.evaluate(scenes[te], labels[te], np.random.default_rng([seed, 2]))
    res.update(seed=seed, denoiser=cfg.denoiser, classifier=cfg.classifier, train_mode=cfg.train_mode,
               photons=cfg.sensor.photons, adc_bits=cfg.sensor.adc_bits, read_noise=cfg.sensor.read_noise,
               denoiser_params=pipe.denoiser.describe())
    return res
