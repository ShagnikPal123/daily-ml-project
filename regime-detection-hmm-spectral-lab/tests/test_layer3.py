import numpy as np
import pytest

from regimelab.detectors import DETECTOR_NAMES, HMMDetector, build_detector
from regimelab.metrics import permutation_accuracy
from regimelab.synth import persistence_matrix, regime_params, simulate


@pytest.fixture(scope="module")
def data():
    p = regime_params(3, 1.5, 0.1)
    return simulate(3500, persistence_matrix(3, 0.97), p, np.random.default_rng(11))


@pytest.mark.parametrize("name", ["threshold", "kmeans", "hmm-ret", "hmm-feat", "hmm-viterbi"])
def test_detector_beats_chance_and_is_ordered(name, data):
    det = build_detector(name, 3, n_iter=20).fit(data.returns[:2000])
    lab = det.predict(data.returns[2000:])
    acc = permutation_accuracy(data.states[2000:], lab, 3)
    assert acc > 0.42
    # volatility ordering: label 0 should be calmer than label 2 in realised terms
    r = data.returns[2000:]
    assert r[lab == 0].std() < r[lab == 2].std()


@pytest.mark.parametrize("name", ["threshold", "kmeans", "hmm-ret", "hmm-feat"])
def test_causal_detectors_ignore_the_future(name, data):
    det = build_detector(name, 3, n_iter=15).fit(data.returns[:2000])
    r = data.returns[2000:3000].copy()
    a = det.predict(r)
    r2 = r.copy()
    r2[600:] *= 4.0
    assert np.array_equal(a[:600], det.predict(r2)[:600])
    assert det.causal


def test_viterbi_marked_non_causal_and_factory_errors():
    assert not build_detector("hmm-viterbi", 3).causal
    with pytest.raises(ValueError):
        build_detector("nope", 3)
    with pytest.raises(ValueError):
        HMMDetector(3, ("ret",), "bogus")
    assert set(DETECTOR_NAMES) >= {"threshold", "kmeans", "hmm-ret"}


def test_custom_feature_spec_and_posterior_normalised(data):
    det = build_detector("hmm:logvol+spec", 3, n_iter=10).fit(data.returns[:1500])
    post = det.posterior(data.returns[1500:2000])
    assert post.shape == (500, 3) and np.allclose(post.sum(1), 1)
