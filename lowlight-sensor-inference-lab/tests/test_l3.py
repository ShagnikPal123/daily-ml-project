import numpy as np
import pytest

from lowlight import budget, data
from lowlight.classifiers import make_classifier
from lowlight.denoisers import make_denoiser
from lowlight.pipeline import Pipeline, PipelineConfig, run_trial
from lowlight.sensor import Sensor, SensorConfig


@pytest.fixture(scope="module")
def world():
    x, y = data.load_scenes(16)
    return x, y, data.make_splits(y, 0)


def cfg(photons=50, den="identity", **kw):
    return PipelineConfig(sensor=SensorConfig(photons=photons), denoiser=den, **kw)


def test_config_validation():
    with pytest.raises(ValueError):
        PipelineConfig(train_mode="weird")
    with pytest.raises(ValueError):
        PipelineConfig(n_noise_draws=0)
    assert cfg(den="ridge").label() == "ridge/softmax/matched"


def test_end_to_end_metrics_keys_and_ranges(world):
    x, y, sp = world
    r = run_trial(x, y, sp, cfg(200, "gaussian"), 0)
    assert 0.5 < r["acc"] <= 1 and len(r["per_class"]) == 10 and np.array(r["confusion"]).sum() == len(sp["test"])
    assert r["psnr"] > 0 and -1 <= r["ssim"] <= 1 and len(r["pred"]) == len(sp["test"])


def test_deterministic_given_seed(world):
    x, y, sp = world
    a, b = run_trial(x, y, sp, cfg(20, "wiener"), 3), run_trial(x, y, sp, cfg(20, "wiener"), 3)
    assert a["acc"] == b["acc"] and a["pred"] == b["pred"]


def test_more_light_gives_higher_accuracy(world):
    x, y, sp = world
    lo = run_trial(x, y, sp, cfg(2), 0)["acc"]
    hi = run_trial(x, y, sp, cfg(500), 0)["acc"]
    assert hi > lo + 0.2


def test_denoising_helps_psnr_at_low_light(world):
    x, y, sp = world
    ident = run_trial(x, y, sp, cfg(5, "identity"), 0)
    ridge = run_trial(x, y, sp, cfg(5, "ridge"), 0)
    assert ridge["psnr"] > ident["psnr"] + 1.0 and ident["psnr_raw"] == pytest.approx(ident["psnr"])


def test_matched_training_beats_clean_training_at_low_light(world):
    x, y, sp = world
    m = run_trial(x, y, sp, cfg(5, train_mode="matched"), 0)["acc"]
    c = run_trial(x, y, sp, cfg(5, train_mode="clean"), 0)["acc"]
    assert m > c


def test_same_noise_across_methods(world):
    x, y, sp = world
    p1, p2 = Pipeline(cfg(10, "identity"), (16, 16), 4), Pipeline(cfg(10, "median"), (16, 16), 4)
    a = p1.acquire(x[:5], np.random.default_rng([4, 2])); b = p2.acquire(x[:5], np.random.default_rng([4, 2]))
    assert np.array_equal(a, b)


def test_evaluate_requires_fit(world):
    x, y, _ = world
    with pytest.raises(RuntimeError):
        Pipeline(cfg(), (16, 16)).evaluate(x[:5], y[:5], np.random.default_rng(0))


def test_isp_output_clipped(world):
    x, _, _ = world
    p = Pipeline(cfg(2), (16, 16))
    out = p.isp(p.acquire(x[:20], np.random.default_rng(0)))
    assert out.min() >= 0 and out.max() <= 1


def test_budget_denoiser_and_classifier_costs():
    s = Sensor(SensorConfig(), (16, 16))
    r = make_denoiser("ridge", s)
    assert budget.denoiser_cost(r, (16, 16)) == {"macs": 25.0 * 256, "line_buffer_bytes": 4 * 16 * 2}
    assert budget.denoiser_cost(make_denoiser("identity", s), (16, 16))["macs"] == 0
    x = np.random.default_rng(0).random((60, 256)); y = np.arange(60) % 10
    q = make_classifier("mlp_int8", epochs=2).fit(x, y); f = make_classifier("mlp", epochs=2).fit(x, y)
    assert budget.classifier_cost(q)["weight_bytes"] < budget.classifier_cost(f)["weight_bytes"]


def test_frame_cost_energy_scales_with_dtype():
    s = Sensor(SensorConfig(), (16, 16))
    d = make_denoiser("wiener", s)
    clf = make_classifier("softmax").fit(np.random.default_rng(0).random((50, 256)), np.arange(50) % 10)
    i8 = budget.frame_cost(d, clf, (16, 16), "int8", "int8")
    f32 = budget.frame_cost(d, clf, (16, 16), "fp32", "fp32")
    assert f32["energy_uj"] > 10 * i8["energy_uj"] and i8["total_macs"] == f32["total_macs"]


def test_adc_relative_energy_doubles_per_bit():
    assert budget.adc_relative_energy(9) == 2 * budget.adc_relative_energy(8) == 2.0


def test_profile_denoisers_rows(world):
    x, _, _ = world
    rows = budget.profile_denoisers(["identity", "gaussian"], SensorConfig(photons=20), x[:30], repeats=1)
    assert [r["denoiser"] for r in rows] == ["identity", "gaussian"] and rows[0]["us_per_image"] >= 0


def test_frame_cost_reports_int8_weight_bytes():
    s = Sensor(SensorConfig(), (16, 16))
    clf = make_classifier("softmax").fit(np.random.default_rng(0).random((50, 256)), np.arange(50) % 10)
    c8 = budget.frame_cost(make_denoiser("box", s), clf, (16, 16), "int8", "int8")
    c32 = budget.frame_cost(make_denoiser("box", s), clf, (16, 16), "fp32", "fp32")
    assert c8["weight_bytes"] == 2560 + 40 and c32["weight_bytes"] == 4 * 2560 + 40
