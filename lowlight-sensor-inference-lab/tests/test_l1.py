import numpy as np
import pytest
from scipy.optimize import check_grad

from lowlight import classifiers as cl, data
from lowlight.sensor import Sensor, SensorConfig


# ---------------- sensor ----------------
def sensor(**kw):
    return Sensor(SensorConfig(**kw), (16, 16), seed=0)


def test_config_validation():
    with pytest.raises(ValueError):
        SensorConfig(photons=0)
    with pytest.raises(ValueError):
        SensorConfig(adc_bits=20)
    with pytest.raises(ValueError):
        SensorConfig.from_dict({"bogus": 1})
    assert SensorConfig().replace(adc_bits=8).adc_bits == 8


def test_codes_in_range_and_integer():
    s = sensor(adc_bits=6, photons=50)
    c = s.capture(np.random.default_rng(0).random((5, 16, 16)), np.random.default_rng(1))
    assert c.dtype.kind == "i" and c.min() >= 0 and c.max() <= 63


def test_capture_deterministic_given_rng():
    s = sensor()
    img = np.random.default_rng(0).random((16, 16))
    a = s.capture(img, np.random.default_rng(5)); b = s.capture(img, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_estimate_unbiased_at_high_light():
    s = sensor(photons=5000, adc_bits=12)
    img = np.full((200, 16, 16), 0.5)
    est = s.acquire(img, np.random.default_rng(0))
    assert abs(est.mean() - 0.5) < 3e-3


def test_noise_variance_matches_theory():
    s = sensor(photons=200, read_noise=3.0, adc_bits=12, prnu=0.0)
    img = np.full((400, 16, 16), 0.4)
    est = s.acquire(img, np.random.default_rng(1))
    assert est.var() == pytest.approx(float(s.noise_var_radiance(0.4)), rel=0.05)


def test_more_bits_never_hurt_at_high_light():
    img = np.random.default_rng(0).random((50, 16, 16))
    err = []
    for b in (3, 6, 10):
        s = sensor(photons=3000, read_noise=0.5, adc_bits=b, prnu=0.0)
        err.append(np.mean((s.acquire(img, np.random.default_rng(2)) - img) ** 2))
    assert err[0] > err[1] > err[2] * 0.99


def test_flat_field_removes_fixed_pattern():
    s = Sensor(SensorConfig(photons=5000, prnu=0.05, read_noise=0.5, adc_bits=12), (16, 16), seed=3)
    img = np.full((100, 16, 16), 0.6)
    raw = s.estimate(s.capture(img, np.random.default_rng(0)), flat_field=False).mean(0)
    cor = s.estimate(s.capture(img, np.random.default_rng(0)), flat_field=True).mean(0)
    assert cor.std() < 0.3 * raw.std()


def test_dark_scene_estimate_near_zero_and_snr_monotone():
    s = sensor(photons=100)
    assert abs(s.acquire(np.zeros((100, 16, 16)), np.random.default_rng(0)).mean()) < 0.02
    assert s.snr(0.9) > s.snr(0.1) > 0
    assert s.dynamic_range_db() > 20


def test_saturation_clips_at_full_scale():
    s = sensor(photons=100, headroom=1.0)
    assert s.capture(np.full((4, 16, 16), 1.0), np.random.default_rng(0)).max() <= 2**10 - 1


# ---------------- classifiers ----------------
@pytest.fixture(scope="module")
def digits():
    x, y = data.load_scenes(8)
    sp = data.make_splits(y, 0)
    X = x.reshape(len(x), -1)
    return X[sp["train"]], y[sp["train"]], X[sp["test"]], y[sp["test"]]


def test_softmax_gradient_check():
    rng = np.random.default_rng(0)
    X = rng.random((20, 5)); Y1 = np.eye(3)[rng.integers(0, 3, 20)]
    f = lambda t: cl.softmax_loss_grad(t, X, Y1, 0.1)[0]
    g = lambda t: cl.softmax_loss_grad(t, X, Y1, 0.1)[1]
    assert check_grad(f, g, rng.normal(size=18)) < 1e-5


def test_softmax_learns_and_probs_normalised(digits):
    Xtr, ytr, Xte, yte = digits
    m = cl.SoftmaxRegression().fit(Xtr, ytr)
    assert m.score(Xte, yte) > 0.93
    assert np.allclose(m.predict_proba(Xte).sum(1), 1)
    assert m.n_params == 64 * 10 + 10


def test_mlp_learns_and_deterministic(digits):
    Xtr, ytr, Xte, yte = digits
    a = cl.MLP(epochs=30, seed=1).fit(Xtr, ytr); b = cl.MLP(epochs=30, seed=1).fit(Xtr, ytr)
    assert a.score(Xte, yte) > 0.93 and np.allclose(a.W1, b.W1)


def test_int8_mlp_agrees_with_float(digits):
    Xtr, ytr, Xte, yte = digits
    m = cl.MLP(epochs=30).fit(Xtr, ytr)
    q = cl.QuantizedMLP(m, Xtr, 8)
    assert np.mean(q.predict(Xte) == m.predict(Xte)) > 0.97
    assert q.weight_bytes() < m.weight_bytes() / 2.5


def test_lower_bits_smaller_and_still_functional(digits):
    Xtr, ytr, Xte, yte = digits
    m = cl.MLP(epochs=30).fit(Xtr, ytr)
    q4, q8 = cl.QuantizedMLP(m, Xtr, 4), cl.QuantizedMLP(m, Xtr, 8)
    assert q4.weight_bytes() < q8.weight_bytes() and q4.score(Xte, yte) > 0.5


def test_make_classifier_factory(digits):
    assert isinstance(cl.make_classifier("softmax"), cl.SoftmaxRegression)
    assert cl.make_classifier("mlp_int6").bits == 6
    with pytest.raises(ValueError):
        cl.make_classifier("svm")
