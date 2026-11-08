import numpy as np
import pytest

from regimelab.features import FeatureBuilder
from regimelab.strategy import (default_exposures, net_returns, regime_positions, summarize,
                                vol_target_positions)


def test_features_shape_and_standardisation():
    r = np.random.default_rng(0).normal(scale=0.01, size=800)
    fb = FeatureBuilder(["ret", "logvol", "mom", "spec"], window=32)
    X = fb.fit_transform(r)
    assert X.shape == (800, 4)
    assert np.allclose(X.mean(0), 0, atol=1e-9) and np.allclose(X.std(0), 1, atol=1e-9)


def test_features_are_causal():
    r = np.random.default_rng(1).normal(scale=0.01, size=600)
    fb = FeatureBuilder(["ret", "logvol", "mom", "spec"], window=32).fit(r)
    a = fb.transform(r)
    r2 = r.copy()
    r2[400:] *= 5
    b = fb.transform(r2)
    assert np.allclose(a[:400], b[:400])


def test_feature_builder_rejects_unknown_and_unfit():
    with pytest.raises(ValueError):
        FeatureBuilder(["nope"])
    with pytest.raises(RuntimeError):
        FeatureBuilder(["ret"]).transform(np.zeros(10))


def test_regime_positions_use_lagged_labels():
    labels = np.array([0, 1, 2, 2, 0])
    pos = regime_positions(labels, np.array([1.0, 0.5, 0.0]))
    assert np.allclose(pos, [1.0, 1.0, 0.5, 0.0, 0.0])


def test_costs_proportional_to_turnover():
    pos = np.array([1.0, 0.0, 1.0, 0.0])
    pnl, turn = net_returns(pos, np.zeros(4), cost_bps=10)
    assert np.allclose(pnl, [0, -1e-3, -1e-3, -1e-3]) and turn == pytest.approx(0.75)


def test_vol_target_scales_inverse_to_vol():
    r = np.concatenate([np.random.default_rng(2).normal(scale=0.005, size=500),
                        np.random.default_rng(3).normal(scale=0.02, size=500)])
    pos = vol_target_positions(r, 0.01, cap=5.0)
    assert pos[400:500].mean() > pos[900:].mean() * 2


def test_default_exposures_and_summary_keys():
    assert np.allclose(default_exposures(3), [1, 0.5, 0])
    s = summarize(np.full(10, 0.001), 0.0)
    assert set(s) == {"sharpe", "max_dd", "ann_ret", "turnover"}
