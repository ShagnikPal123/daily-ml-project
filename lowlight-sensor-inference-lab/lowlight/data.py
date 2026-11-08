"""Layer 0: scene source (real public dataset) and reproducible splits.

Scenes are the 1797 8x8 handwritten digits shipped with scikit-learn (the UCI
Optical Recognition of Handwritten Digits test set, Alpaydin & Kaynak). They are
bilinearly upsampled to ``size`` x ``size`` and treated as *scene radiance* in
[0, 1], i.e. what the lens would project onto the sensor before any noise.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_scenes(size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    d = load_digits()
    imgs = d.images.astype(float) / 16.0
    if size != 8:
        imgs = np.stack([zoom(im, size / 8.0, order=1) for im in imgs])
    return np.clip(imgs, 0.0, 1.0), d.target.astype(int)


def make_splits(labels, seed: int, test_frac: float = 0.25) -> dict[str, np.ndarray]:
    """Stratified train/test indices, deterministic in ``seed``."""
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=test_frac, stratify=labels, random_state=seed)
    return {"train": np.sort(tr), "test": np.sort(te)}
