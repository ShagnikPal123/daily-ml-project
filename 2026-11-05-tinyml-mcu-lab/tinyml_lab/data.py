"""Dataset loading: 8x8 handwritten digits (bundled with scikit-learn, no download)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return int(max(self.y_train.max(), self.y_test.max()) + 1)

    def calibration(self, n: int = 256) -> np.ndarray:
        """Small unlabeled subset used to choose activation ranges (as on a real MCU flow)."""
        return self.X_train[:n]


def load_dataset(seed: int = 0, test_size: float = 0.3) -> Dataset:
    """Pixels in [0,1] (raw 4-bit sensor counts / 16): mimics a 4-bit camera front end."""
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    return Dataset(Xtr, ytr, Xte, yte)
