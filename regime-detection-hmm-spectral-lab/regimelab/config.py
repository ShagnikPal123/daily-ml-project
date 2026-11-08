"""Layer 0: experiment configuration (JSON round-trippable, validated)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields


@dataclass
class ExpConfig:
    k: int = 3                 # number of hidden regimes
    p_stay: float = 0.97       # diagonal of the Markov transition matrix
    separation: float = 1.0    # volatility spread between regimes
    ar_strength: float = 0.15  # |AR(1)| coefficient in the extreme regimes
    cycle_amp: float = 0.0     # optional deterministic cycle added to returns
    n_train: int = 2000
    n_test: int = 1500
    seeds: int = 6
    cost_bps: float = 5.0      # one-way trading cost, basis points of notional
    feature_window: int = 64
    vol_lambda: float = 0.94   # EWMA decay (RiskMetrics style)
    n_iter: int = 30           # Baum-Welch iterations

    def validate(self) -> "ExpConfig":
        if self.k < 2:
            raise ValueError("k must be >= 2")
        if not 0.0 < self.p_stay < 1.0:
            raise ValueError("p_stay must be in (0, 1)")
        if self.separation <= 0:
            raise ValueError("separation must be > 0")
        if self.n_train < 10 * self.k or self.n_test < 10:
            raise ValueError("series too short")
        if self.feature_window < 8:
            raise ValueError("feature_window must be >= 8")
        if not 0.0 < self.vol_lambda < 1.0:
            raise ValueError("vol_lambda must be in (0, 1)")
        if self.seeds < 1:
            raise ValueError("seeds must be >= 1")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExpConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d).validate()

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path) -> "ExpConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def replace(self, **kw) -> "ExpConfig":
        d = self.to_dict()
        d.update(kw)
        return ExpConfig.from_dict(d)
