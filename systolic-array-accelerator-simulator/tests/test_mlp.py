import numpy as np
import pytest

from sysarray.mlp import (FloatMLP, accel_matmul_factory, accuracy, load_data, numpy_matmul,
                          prune_magnitude, quantize_model, train_float)


@pytest.fixture(scope="module")
def trained():
    Xtr, Xte, ytr, yte = load_data(0)
    return Xtr, Xte, ytr, yte, train_float(Xtr, ytr, hidden=32, seed=0)


def test_float_model_learns(trained):
    _, Xte, _, yte, m = trained
    assert accuracy(m.predict(Xte), yte) > 0.9


def test_int8_close_to_float(trained):
    Xtr, Xte, _, yte, m = trained
    q = quantize_model(m, Xtr)
    assert abs(accuracy(q.predict(Xte, numpy_matmul), yte) - accuracy(m.predict(Xte), yte)) < 0.03


@pytest.mark.parametrize("df", ["OS", "WS"])
def test_accelerator_bit_exact_vs_numpy(trained, df):
    Xtr, Xte, _, _, m = trained
    q = quantize_model(m, Xtr)
    mm, log = accel_matmul_factory(8, 8, df)
    assert np.array_equal(q.predict(Xte[:20], mm), q.predict(Xte[:20], numpy_matmul))
    assert log["stats"].macs == sum(a * b * c for a, b, c in log["gemms"])


def test_pruning_hits_target_sparsity(trained):
    m = trained[-1]
    pr = prune_magnitude(m, 0.7)
    for w in pr.weights:
        assert 0.68 <= np.mean(w == 0) <= 0.75


def test_pruning_increases_zero_macs(trained):
    Xtr, Xte, _, _, m = trained
    fracs = []
    for sp in (0.0, 0.8):
        q = quantize_model(prune_magnitude(m, sp), Xtr)
        mm, log = accel_matmul_factory(8, 8, "WS")
        q.predict(Xte[:16], mm)
        fracs.append(log["stats"].zero_fraction)
    assert fracs[1] > fracs[0]
