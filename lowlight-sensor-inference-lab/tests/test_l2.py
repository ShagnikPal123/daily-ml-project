import numpy as np
import pytest

from lowlight import data, imgops as io
from lowlight.denoisers import DENOISERS, GATGaussian, make_denoiser
from lowlight.sensor import Sensor, SensorConfig


@pytest.fixture(scope="module")
def world():
    x, y = data.load_scenes(16)
    sensor = Sensor(SensorConfig(photons=10, read_noise=2.0), (16, 16), seed=0)
    rng = np.random.default_rng(0)
    tr, te = x[:400], x[400:560]
    return sensor, tr, sensor.acquire(tr, rng), te, sensor.acquire(te, rng)


def fitted(name, world, **kw):
    sensor, tr, ntr, *_ = world
    return make_denoiser(name, sensor, **kw).fit(tr, ntr)


@pytest.mark.parametrize("name", sorted(DENOISERS))
def test_shape_preserved_and_finite(name, world):
    _, _, _, te, nte = world
    out = fitted(name, world)(nte)
    assert out.shape == nte.shape and np.isfinite(out).all()
    assert fitted(name, world)(nte[0]).shape == nte[0].shape


@pytest.mark.parametrize("name", ["box", "gaussian", "median", "wiener", "gat", "ridge"])
def test_denoisers_reduce_mse_at_low_light(name, world):
    _, _, _, te, nte = world
    base = np.mean((np.clip(nte, 0, 1) - te) ** 2)
    out = np.clip(fitted(name, world)(nte), 0, 1)
    assert np.mean((out - te) ** 2) < base


def test_identity_is_exact(world):
    _, _, _, _, nte = world
    assert np.array_equal(fitted("identity", world)(nte), nte)


@pytest.mark.parametrize("name", ["box", "gaussian", "median", "wiener", "ridge"])
def test_constant_image_preserved(name, world):
    c = np.full((2, 16, 16), 0.4)
    out = fitted(name, world)(c)
    assert np.abs(out - 0.4).max() < 0.02


def test_ridge_requires_fit(world):
    with pytest.raises(RuntimeError):
        make_denoiser("ridge", world[0])(world[4])


def test_ridge_beats_identity_on_training_mse(world):
    _, tr, ntr, *_ = world
    d = fitted("ridge", world)
    assert np.mean((d(ntr) - tr) ** 2) < np.mean((ntr - tr) ** 2)
    assert d.w.shape == (25,) and abs(d.w.sum()) < 1.5


def test_sigma_selected_from_grid(world):
    for n in ("gaussian", "gat"):
        d = fitted(n, world)
        assert d.sigma in d.grid


def test_gat_asymptotic_inverse_less_biased_than_algebraic():
    s = Sensor(SensorConfig(photons=1000, read_noise=1.0, dark_electrons=0.0, prnu=0.0), (4, 4))
    rng = np.random.default_rng(0)
    alg, asy = GATGaussian(s, inverse="algebraic"), GATGaussian(s, inverse="asymptotic")
    lam = 20.0
    z = rng.poisson(lam, 400000) + rng.standard_normal(400000)
    d = alg.forward(z)
    # smoothed domain emulation: average forward values over blocks of 9 to mimic filtering
    db = d[: (len(d) // 9) * 9].reshape(-1, 9).mean(1)
    bias_alg = abs(alg.inverse_transform(db).mean() - lam)
    bias_asy = abs(asy.inverse_transform(db).mean() - lam)
    assert bias_asy < bias_alg


def test_gat_inverse_validation_and_registry_errors(world):
    with pytest.raises(ValueError):
        GATGaussian(world[0], inverse="nope")
    with pytest.raises(ValueError):
        make_denoiser("bm3d", world[0])


def test_hardware_descriptors_sane(world):
    for n in DENOISERS:
        d = fitted(n, world)
        assert d.macs_per_pixel() >= 0 and d.line_buffer_rows() >= 0 and "name" in d.describe()
    assert fitted("identity", world).macs_per_pixel() == 0
    assert fitted("ridge", world).macs_per_pixel() == 25


def test_wiener_uses_signal_dependent_noise(world):
    sensor = world[0]
    bright = np.full((1, 16, 16), 0.9) + np.random.default_rng(0).normal(0, 0.05, (1, 16, 16))
    out = fitted("wiener", world)(bright)
    assert out.std() < bright.std()
