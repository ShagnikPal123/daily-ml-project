import numpy as np, pytest
from codinglab import channel, gf2, convolutional, ldpc, decoders, quantized, tuning, simulate


@pytest.fixture(scope="module")
def code():
    return ldpc.LDPCCode(n=252, seed=3)


def test_sigma_formula():
    assert channel.sigma_from_ebn0(0, 0.5) == pytest.approx(1.0)


def test_bpsk_llr_sign():
    y = np.array([0.7, -0.2]); assert (channel.hard_decision(channel.awgn_llr(y, 1.0)) == [0, 1]).all()


def test_uncoded_theory_matches_sim():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2, 400000); y = channel.awgn(channel.bpsk(b), channel.sigma_from_ebn0(4, 1), rng)
    sim = np.mean(channel.hard_decision(y) != b)
    assert sim == pytest.approx(channel.uncoded_ber_theory(4), rel=0.15)


def test_shannon_limit_half_rate():
    assert channel.shannon_limit_ebn0_db(0.5) == pytest.approx(0.19, abs=0.1)


def test_gf2_rref_and_rank():
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    assert gf2.rank(M) == 2


def test_ldpc_codewords_satisfy_parity(code):
    rng = np.random.default_rng(1)
    cw = code.encode(rng.integers(0, 2, (20, code.k), dtype=np.uint8))
    assert not code.syndrome(cw).any()


def test_ldpc_structure(code):
    assert (code.H.sum(0) == code.dv).all() and (code.H.sum(1) == code.dc).all()
    assert code.rate >= code.design_rate - 1e-9 and code.girth_four_free()


def test_conv_roundtrip_noiseless():
    cc = convolutional.ConvCode(); rng = np.random.default_rng(2)
    info = rng.integers(0, 2, (5, 60), dtype=np.uint8)
    cw = cc.encode(info)
    assert (cc.viterbi(cw, soft=False) == info).all()
    assert (cc.viterbi(1.0 - 2.0 * cw, soft=True) == info).all()


def test_conv_known_impulse_response():
    cc = convolutional.ConvCode(3, (0o7, 0o5))
    out = cc.encode(np.array([[1, 0, 0]], dtype=np.uint8)).reshape(-1, 2)
    assert out[:3].tolist() == [[1, 1], [1, 0], [1, 1]]


def test_conv_corrects_errors():
    cc = convolutional.ConvCode(); rng = np.random.default_rng(3)
    info = rng.integers(0, 2, (4, 100), dtype=np.uint8); cw = cc.encode(info).copy()
    cw[:, [10, 50, 120, 170]] ^= 1
    assert (cc.viterbi(cw, soft=False) == info).all()


def test_soft_beats_hard_viterbi():
    rng = np.random.default_rng(4)
    h = simulate.ber_curve(simulate.ConvScheme(k=100, soft=False), [3.0], max_frames=400, seed=1)[0].ber
    s = simulate.ber_curve(simulate.ConvScheme(k=100, soft=True), [3.0], max_frames=400, seed=1)[0].ber
    assert s < h


@pytest.mark.parametrize("method", ["spa", "ms", "nms", "oms"])
def test_decoders_fix_noiseless_and_noisy(code, method):
    rng = np.random.default_rng(5)
    info = rng.integers(0, 2, (30, code.k), dtype=np.uint8); cw = code.encode(info)
    sig = channel.sigma_from_ebn0(5.0, code.rate)
    llr = channel.awgn_llr(channel.awgn(channel.bpsk(cw), sig, rng), sig)
    res = decoders.decode(code, llr, method, iters=40, alpha=0.8, beta=0.15)
    assert np.mean(res.bits != cw) < 0.01


def test_spa_not_worse_than_ms(code):
    rng = np.random.default_rng(6)
    info = rng.integers(0, 2, (200, code.k), dtype=np.uint8); cw = code.encode(info)
    sig = channel.sigma_from_ebn0(2.0, code.rate)
    llr = channel.awgn_llr(channel.awgn(channel.bpsk(cw), sig, rng), sig)
    e = {m: np.mean(decoders.decode(code, llr, m, iters=25).bits != cw) for m in ("spa", "ms")}
    assert e["spa"] <= e["ms"] * 1.05


def test_early_stop_iterations(code):
    rng = np.random.default_rng(7)
    cw = code.encode(rng.integers(0, 2, (10, code.k), dtype=np.uint8))
    llr = 6.0 * channel.bpsk(cw)
    res = decoders.decode(code, llr, "nms", iters=20)
    assert res.converged.all() and (res.iters == 0).all()


def test_check_minsum_leave_one_out():
    v = np.array([[[3.0, -1.0, 2.0, -4.0]]])
    out = decoders._check_minsum(v)[0, 0]
    # explicit: excluding idx0 -> sign (-)(+)(-)=+, min(1,2,4)=1 ; excluding idx1 -> sign (+)(+)(-)=-, min=2
    assert out[0] == 1.0 and out[1] == -2.0 and out[2] == 1.0 and out[3] == -1.0


def test_fixed_point_quantiser():
    q = quantized.FixedPoint(4, max_llr=7.0)
    assert q.levels == 7 and q.step == 1.0
    assert q(np.array([0.4, 9.0, -9.0, 2.6])).tolist() == [0.0, 7.0, -7.0, 3.0]


def test_memory_compression_smaller(code):
    assert quantized.compressed_check_memory_bits(code, 6) < quantized.message_memory_bits(code, 6)


def test_tuned_alpha_in_range(code):
    a, grid, sc = tuning.tune_alpha(code, 2.0, iters=8, n_frames=60)
    assert 0.5 <= a <= 1.0 and len(sc) == len(grid)


def test_rayleigh_theory_vs_sim():
    p = simulate.ber_curve(simulate.Uncoded(), [10.0], min_frame_errors=200, max_frames=4000, seed=2, fading=True)[0].ber
    assert p == pytest.approx(channel.uncoded_ber_rayleigh_theory(10.0), rel=0.2)
