import numpy as np, pytest
from audiolab import signals, stft, metrics, spectral, adaptive, features, classifier, experiments


def rng(s=0):
    return np.random.default_rng(s)


def test_stft_perfect_reconstruction():
    x = rng().standard_normal(3000)
    y = stft.istft(stft.stft(x), len(x))
    assert np.max(np.abs(x - y)) < 1e-10


def test_stft_matches_sine_peak_bin():
    fs = 8000; t = np.arange(4000) / fs
    S = np.abs(stft.stft(np.sin(2 * np.pi * 1000 * t)))
    assert abs(np.argmax(S[20]) - 1000 / fs * 256) <= 1


def test_scale_to_snr_exact():
    c = rng().standard_normal(5000); n = rng(1).standard_normal(5000)
    n2 = signals.scale_to_snr(c, n, 7.0)
    assert 10 * np.log10(np.mean(c ** 2) / np.mean(n2 ** 2)) == pytest.approx(7.0, abs=1e-6)


@pytest.mark.parametrize("label", signals.CLASSES)
def test_clips_unit_rms_and_finite(label):
    x = signals.make_clip(label, rng())
    assert np.all(np.isfinite(x)) and np.sqrt(np.mean(x ** 2)) == pytest.approx(1.0, abs=1e-6)


def test_si_sdr_scale_invariant_and_snr_sensitive():
    c = rng().standard_normal(4000)
    assert metrics.si_sdr(c, 3 * c + 1e-9 * rng(1).standard_normal(4000)) > 60
    assert metrics.si_sdr(c, c + rng(2).standard_normal(4000)) == pytest.approx(0, abs=1.0)


def test_spectral_subtraction_improves_low_snr_tone():
    r = rng(3); t = np.arange(8000) / 8000
    clean = np.sin(2 * np.pi * 700 * t) * (t > 0.2)  # noise-only lead-in
    noisy, _ = signals.add_noise(clean, "white", 0, r)
    out = spectral.spectral_subtraction(noisy)  # initial-frames noise estimate
    assert metrics.si_sdr(clean, out) > metrics.si_sdr(clean, noisy) + 2


def test_wiener_improves_low_snr_tone_and_preserves_length():
    r = rng(4); t = np.arange(8000) / 8000
    clean = np.sin(2 * np.pi * 900 * t) * (t > 0.2)
    noisy, _ = signals.add_noise(clean, "white", 0, r)
    out = spectral.wiener_dd(noisy)
    assert len(out) == len(noisy) and metrics.si_sdr(clean, out) > metrics.si_sdr(clean, noisy)


def test_minstats_psd_tracks_white_noise_level():
    x = rng(5).standard_normal(16000)
    P = spectral.noise_psd_minstats(stft.stft(x))
    expected = np.mean(np.abs(stft.stft(x)) ** 2)
    assert 0.2 * expected < np.mean(P[-1]) < 3 * expected


def test_notch_comb_removes_hum():
    fs = 8000; t = np.arange(8000) / fs
    hum = np.sin(2 * np.pi * 60 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    out = spectral.notch_comb(hum, fs)
    assert np.mean(out[4000:] ** 2) < 0.02 * np.mean(hum ** 2)


@pytest.mark.parametrize("algo,kw", [("lms", dict(mu=0.01)), ("nlms", dict(mu=0.5)), ("rls", dict(lam=0.999))])
def test_adaptive_identifies_fir_path(algo, kw):
    r = rng(6); h = adaptive.make_noise_path(r, 8)
    ref = r.standard_normal(6000)
    d = np.convolve(ref, h)[:6000]
    e, w, _ = adaptive.ALGOS[algo](d, ref, L=8, **kw)
    assert adaptive.misalignment_db(w, h) < -20
    assert np.mean(e[-500:] ** 2) < 0.01 * np.mean(d ** 2)


def test_rls_faster_than_lms_on_colored_reference():
    r = rng(7); h = adaptive.make_noise_path(r, 8)
    ref = np.convolve(r.standard_normal(3000), [1, 0.9, 0.8])[:3000]
    d = np.convolve(ref, h)[:3000]
    el = adaptive.lms(d, ref, L=8, mu=0.005)[0]; er = adaptive.rls(d, ref, L=8)[0]
    assert np.mean(er[500:] ** 2) < np.mean(el[500:] ** 2)


def test_mel_filterbank_properties():
    fb = features.mel_filterbank(24, 256, 8000)
    assert fb.shape == (24, 129) and fb.min() >= 0 and fb.max() <= 1 + 1e-9


def test_dct2_matches_scipy():
    from scipy.fft import dct
    x = rng().standard_normal((5, 24))
    assert np.allclose(features.dct2(x, 12), dct(x, type=2, norm="ortho")[:, :12])


def test_embedding_dimension_and_determinism():
    x = signals.make_clip("siren", rng())
    a, b = features.clip_embedding(x), features.clip_embedding(x)
    assert a.shape == (38,) and np.allclose(a, b)


def test_classifier_separates_clean_classes():
    r = rng(8)
    Xtr, ytr = signals.make_dataset(10, r); Xte, yte = signals.make_dataset(5, r)
    m = classifier.make_model("svm").fit(classifier.featurize(Xtr), ytr)
    assert np.mean(m.predict(classifier.featurize(Xte)) == yte) > 0.9


def test_hum_experiment_rls_beats_input():
    res = experiments.exp_hum(n=6000)
    assert res["rls_ref"] > res["input"] + 5


def test_vad_detects_burst_in_silence():
    from audiolab import vad
    r = rng(9); x = np.zeros(8000); x[3000:5000] = np.sin(2 * np.pi * 500 * np.arange(2000) / 8000)
    S = stft.stft(x + 0.01 * r.standard_normal(8000))
    act = vad.detect(S)
    assert act.sum() > 5 and not act[:10].any()


def test_gated_noise_psd_holds_during_activity():
    from audiolab import vad
    S = stft.stft(rng(10).standard_normal(8000))
    act = np.zeros(S.shape[0], bool); act[40:60] = True
    est = vad.gated_noise_psd(S, act)
    assert np.allclose(est[41], est[59]) and not np.allclose(est[10], est[20])


def test_vad_wiener_returns_same_length():
    from audiolab import vad
    x = rng(11).standard_normal(5000)
    assert len(vad.vad_wiener(x)) == 5000
