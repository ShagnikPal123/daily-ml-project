"""Ground-truth checks against independent SciPy / scikit-learn / scikit-image implementations."""
import numpy as np
import pytest
from scipy import ndimage, signal
from skimage import metrics as skm
from sklearn.linear_model import LogisticRegression, Ridge

from lowlight import classifiers as cl, data, imgops as io
from lowlight.denoisers import GATGaussian, RidgePatch
from lowlight.fixedpoint import quant_step, uniform_dequantize, uniform_quantize
from lowlight.sensor import Sensor, SensorConfig

RNG = np.random.default_rng(123)
IMG = RNG.random((3, 16, 16))


@pytest.mark.parametrize("sigma", [0.7, 1.5])
def test_gaussian_vs_scipy(sigma):
    ref = np.stack([ndimage.gaussian_filter(im, sigma, mode="reflect", truncate=3.0) for im in IMG])
    assert np.abs(io.gaussian_filter(IMG, sigma, truncate=3.0) - ref).max() < 1e-12


def test_median_vs_scipy():
    ref = np.stack([ndimage.median_filter(im, size=3, mode="reflect") for im in IMG])
    assert np.abs(io.median_filter(IMG, 3) - ref).max() < 1e-12


def test_box_vs_scipy_uniform_filter():
    ref = np.stack([ndimage.uniform_filter(im, size=5, mode="reflect") for im in IMG])
    assert np.abs(io.box_filter(IMG, 5) - ref).max() < 1e-12


@pytest.mark.parametrize("noise", [0.01, None])
def test_wiener_vs_scipy_signal(noise):
    ref = np.stack([signal.wiener(im, 5, noise=noise) for im in IMG])
    assert np.abs(io.wiener_filter(IMG, 5, noise=noise, mode="constant") - ref).max() < 1e-10


def test_ssim_psnr_vs_skimage():
    noisy = IMG + RNG.normal(0, 0.1, IMG.shape)
    s = [skm.structural_similarity(a, b, data_range=1.0, win_size=7) for a, b in zip(IMG, noisy)]
    p = [skm.peak_signal_noise_ratio(a, b, data_range=1.0) for a, b in zip(IMG, noisy)]
    assert np.abs(io.ssim(IMG, noisy) - s).max() < 1e-10
    assert np.abs(io.psnr(IMG, noisy) - p).max() < 1e-8


def test_softmax_vs_sklearn_logreg():
    x, y = data.load_scenes(8)
    X = x.reshape(len(x), -1)[:900]; y = y[:900]
    l2 = 1e-2
    ours = cl.SoftmaxRegression(l2=l2, max_iter=3000).fit(X, y)
    ref = LogisticRegression(C=1.0 / (l2 * len(y)), tol=1e-10, max_iter=5000).fit(X, y)
    assert np.abs(ours.predict_proba(X) - ref.predict_proba(X)).max() < 5e-3
    assert np.mean(ours.predict(X) == ref.predict(X)) > 0.995


def test_ridge_patch_vs_sklearn_ridge():
    from lowlight.sensor import Sensor
    clean = RNG.random((40, 16, 16)); noisy = clean + RNG.normal(0, 0.2, clean.shape)
    lam = 1e-2
    d = RidgePatch(Sensor(SensorConfig(), (16, 16)), size=3, lam=lam, chunk=7).fit(clean, noisy)
    P = io.extract_patches(noisy, 3).reshape(-1, 9)
    ref = Ridge(alpha=lam * P.shape[0]).fit(P, clean.reshape(-1))
    assert np.abs(d.w - ref.coef_).max() < 1e-9 and abs(d.b - ref.intercept_) < 1e-9


def test_shot_noise_is_poisson_fano_factor_one():
    s = Sensor(SensorConfig(photons=60, read_noise=0.0, dark_electrons=0.0, prnu=0.0, adc_bits=16), (16, 16))
    e = s.expose(np.full((300, 16, 16), 0.5), np.random.default_rng(0))
    assert e.var() / e.mean() == pytest.approx(1.0, rel=0.03)
    assert e.mean() == pytest.approx(30.0, rel=0.01)


@pytest.mark.parametrize("lam", [10.0, 50.0, 200.0])
def test_gat_stabilises_variance_to_unity(lam):
    s = Sensor(SensorConfig(photons=1000, read_noise=2.0, dark_electrons=0.0, prnu=0.0), (16, 16))
    rng = np.random.default_rng(1)
    z = rng.poisson(lam, 200000) + 2.0 * rng.standard_normal(200000)
    assert np.var(GATGaussian(s).forward(z)) == pytest.approx(1.0, abs=0.15)
    assert z.var() == pytest.approx(lam + 4.0, rel=0.03)  # raw variance is signal-dependent


def test_adc_quantisation_noise_is_lsb_squared_over_twelve():
    x = np.random.default_rng(0).uniform(0, 1, 400000)
    err = uniform_dequantize(uniform_quantize(x, 6), 6) - x
    assert err.var() == pytest.approx(quant_step(6) ** 2 / 12, rel=0.03)


def test_clean_digit_accuracy_sanity():
    x, y = data.load_scenes(16)
    sp = data.make_splits(y, 0)
    X = x.reshape(len(x), -1)
    acc = cl.SoftmaxRegression().fit(X[sp["train"]], y[sp["train"]]).score(X[sp["test"]], y[sp["test"]])
    assert acc > 0.95
