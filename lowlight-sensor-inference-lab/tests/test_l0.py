import numpy as np
import pytest
from scipy import stats as sst

from lowlight import data, fixedpoint as fp, imgops as io, stats


# ---------------- fixedpoint ----------------
def test_qrange():
    assert fp.qrange(8) == (-128, 127)
    assert fp.qrange(8, signed=False) == (0, 255)
    with pytest.raises(ValueError):
        fp.qrange(0)


def test_uniform_quantize_endpoints_and_monotone():
    x = np.linspace(-0.5, 1.5, 101)
    c = fp.uniform_quantize(x, 4)
    assert c.min() == 0 and c.max() == 15
    assert np.all(np.diff(c) >= 0)


def test_uniform_roundtrip_error_bounded_by_half_lsb():
    x = np.random.default_rng(0).uniform(0, 1, 5000)
    for b in (2, 5, 10):
        err = np.abs(fp.uniform_dequantize(fp.uniform_quantize(x, b), b) - x)
        assert err.max() <= fp.quant_step(b) / 2 + 1e-12


def test_symmetric_quantization_error_and_saturation():
    x = np.random.default_rng(1).normal(size=1000)
    s = fp.symmetric_scale(x)
    q = fp.quantize_symmetric(x, s)
    assert q.min() >= -128 and q.max() <= 127
    assert np.abs(fp.dequantize_symmetric(q, s) - x).max() <= s / 2 + 1e-12
    assert fp.quantize_symmetric(np.array([1e9]), 1.0)[0] == 127


def test_unsigned_quantization_clips_negative():
    assert fp.quantize_unsigned(np.array([-1.0, 0.5]), 1 / 255)[0] == 0


def test_q_format_roundtrip_and_saturation():
    q = fp.to_fixed(np.array([0.75, -1.5, 100.0]), 3, 8)
    assert q[2] == 2**11 - 1  # saturates at +7.996
    assert fp.from_fixed(q, 8)[0] == 0.75


def test_fixed_mul_matches_float():
    a, b = fp.to_fixed(0.3, 2, 12), fp.to_fixed(-0.7, 2, 12)
    assert abs(fp.from_fixed(fp.fixed_mul(a, b, 12), 12) - (0.3 * -0.7)) < 2e-3


def test_saturating_add():
    assert fp.saturating_add(120, 120, 8) == 127
    assert fp.saturating_add(-120, -120, 8) == -128


def test_int_matmul_exact_and_overflow():
    a = np.random.default_rng(2).integers(-5, 5, (4, 6))
    b = np.random.default_rng(3).integers(-5, 5, (6, 3))
    assert np.array_equal(fp.int_matmul(a, b), a @ b)
    with pytest.raises(OverflowError):
        fp.int_matmul(np.full((1, 4), 2**20), np.full((4, 1), 2**20))


def test_requantize_saturates():
    assert fp.requantize(np.array([1e6]), 1.0)[0] == 255


# ---------------- imgops ----------------
def test_identity_kernel_and_batch_single_agree():
    x = np.random.default_rng(0).random((3, 8, 8))
    k = np.zeros((3, 3)); k[1, 1] = 1
    assert np.allclose(io.correlate2d(x, k), x)
    assert io.correlate2d(x[0], k).shape == (8, 8)


def test_kernels_normalised_symmetric():
    g = io.gaussian_kernel(1.3)
    assert abs(g.sum() - 1) < 1e-12 and np.allclose(g, g.T) and np.allclose(g, g[::-1, ::-1])
    assert abs(io.box_kernel(5).sum() - 1) < 1e-12


def test_filters_preserve_constant_image():
    c = np.full((2, 9, 9), 0.37)
    for out in (io.gaussian_filter(c, 1.0), io.box_filter(c, 3), io.median_filter(c, 3), io.wiener_filter(c, 5, noise=0.01)):
        assert np.allclose(out, 0.37)


def test_median_removes_impulse():
    x = np.zeros((9, 9)); x[4, 4] = 1.0
    assert io.median_filter(x, 3).max() == 0.0


def test_gaussian_reduces_noise_variance():
    n = np.random.default_rng(0).normal(size=(10, 32, 32))
    assert io.gaussian_filter(n, 1.0).var() < 0.3 * n.var()


def test_bad_mode_and_ndim():
    with pytest.raises(ValueError):
        io.box_filter(np.zeros((4, 4)), 3, mode="nope")
    with pytest.raises(ValueError):
        io.as_batch(np.zeros(5))


def test_patches_center_pixel_is_image():
    x = np.random.default_rng(1).random((2, 6, 6))
    p = io.extract_patches(x, 3)
    assert p.shape == (2, 6, 6, 9) and np.allclose(p[..., 4], x)


def test_psnr_known_value_and_identical():
    a = np.zeros((1, 8, 8)); b = a + 0.1
    assert abs(io.psnr(a, b)[0] - 20.0) < 1e-9
    assert io.psnr(a, a)[0] == pytest.approx(100.0)


def test_ssim_identical_one_and_decreases_with_noise():
    x = np.random.default_rng(2).random((2, 16, 16))
    assert np.allclose(io.ssim(x, x), 1.0)
    noisy = x + np.random.default_rng(3).normal(0, 0.2, x.shape)
    assert np.all(io.ssim(x, noisy) < 0.9)


def test_local_mean_var_constant_has_zero_var():
    mu, v = io.local_mean_var(np.full((5, 5), 2.0), 3)
    assert np.allclose(mu, 2.0) and np.allclose(v, 0.0)


# ---------------- stats ----------------
def test_mean_ci_matches_scipy():
    v = [0.9, 0.85, 0.88, 0.91, 0.87]
    m, hw = stats.mean_ci(v)
    lo, hi = sst.t.interval(0.95, 4, loc=np.mean(v), scale=sst.sem(v))
    assert m == pytest.approx(np.mean(v)) and hw == pytest.approx(hi - m)


def test_mean_ci_edge_cases():
    assert np.isnan(stats.mean_ci([1.0])[1])
    assert np.isnan(stats.mean_ci([])[0])


def test_paired_diff_and_significance():
    a = np.array([0.9, 0.91, 0.92, 0.93]); b = a - 0.05 + np.array([0.001, -0.001, 0.0, 0.0005])
    d, hw = stats.paired_diff_ci(a, b)
    assert d == pytest.approx(0.05, abs=1e-3) and stats.significant(d, hw)
    with pytest.raises(ValueError):
        stats.paired_diff_ci([1, 2], [1])


def test_bootstrap_contains_mean():
    v = np.random.default_rng(0).normal(5, 1, 40)
    lo, hi = stats.bootstrap_ci(v)
    assert lo < v.mean() < hi


# ---------------- data ----------------
def test_scenes_shape_range():
    x, y = data.load_scenes(16)
    assert x.shape == (1797, 16, 16) and x.min() >= 0 and x.max() <= 1 and set(y) == set(range(10))


def test_splits_disjoint_stratified_deterministic():
    _, y = data.load_scenes(8)
    s1, s2 = data.make_splits(y, 3), data.make_splits(y, 3)
    assert np.array_equal(s1["test"], s2["test"])
    assert not set(s1["train"]) & set(s1["test"])
    assert len(s1["train"]) + len(s1["test"]) == len(y)
    frac = np.bincount(y[s1["test"]]) / len(s1["test"])
    assert np.abs(frac - np.bincount(y) / len(y)).max() < 0.01


def test_p_from_ci_roundtrip_and_holm():
    v = np.array([0.04, 0.05, 0.06, 0.05, 0.045]) 
    m, hw = stats.mean_ci(v)
    assert stats.p_from_ci(m, hw, len(v)) == pytest.approx(sst.ttest_1samp(v, 0).pvalue, rel=1e-6)
    assert np.isnan(stats.p_from_ci(0.1, float("nan"), 1))
    assert stats.holm_reject([0.001, 0.02, 0.04, 0.9]) == [True, False, False, False]
    assert stats.holm_reject([0.01, 0.011]) == [True, True]
