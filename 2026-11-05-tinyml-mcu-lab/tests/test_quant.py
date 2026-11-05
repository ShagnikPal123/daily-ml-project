import math

import numpy as np

from tinyml_lab.quant import (build_qmlp, fixed_forward, qmax, packed_bytes, quantize_multiplier,
                              quantize_weight, requantize, sqnr_db, to_fixed, from_fixed)


def test_quantize_multiplier_reconstructs_real():
    for real in [1e-4, 0.0123, 0.5, 0.999, 1.7, 3.2]:
        M0, sh = quantize_multiplier(real)
        assert (1 << 30) <= M0 < (1 << 31)
        assert math.isclose(M0 / 2.0 ** sh, real, rel_tol=1e-8)


def test_requantize_matches_float_within_one_lsb():
    rng = np.random.default_rng(0)
    acc = rng.integers(-200000, 200000, size=(50, 4))
    reals = np.array([0.0007, 0.0031, 0.02, 0.0004])
    ms = [quantize_multiplier(r) for r in reals]
    M0 = np.array([m[0] for m in ms]); sh = np.array([m[1] for m in ms])
    got = requantize(acc, M0, sh)
    want = np.round(acc * reals)
    assert np.max(np.abs(got - want)) <= 1


def test_weight_quant_bounds_and_roundtrip():
    W = np.random.default_rng(1).normal(size=(16, 8))
    q, s = quantize_weight(W, 4, per_channel=True)
    assert q.max() <= qmax(4) and q.min() >= -qmax(4)
    assert np.max(np.abs(q * s - W)) <= s.max() / 2 + 1e-12


def test_per_channel_beats_per_tensor_on_skewed_channels():
    rng = np.random.default_rng(2)
    W = rng.normal(size=(32, 8)) * np.array([0.01, 0.02, 0.05, 0.1, 0.3, 1.0, 2.0, 4.0])
    e = {}
    for pc in (False, True):
        q, s = quantize_weight(W, 4, pc)
        e[pc] = sqnr_db(W, q * s)
    assert e[True] > e[False] + 3


def test_int8_matches_float_accuracy(ctx):
    d, m = ctx.data, ctx.models[0]
    q = build_qmlp(m, ctx.Xcal, 8, 8, True)
    assert abs(q.accuracy(d.X_test, d.y_test) - m.accuracy(d.X_test, d.y_test)) < 0.02


def test_two_bit_per_tensor_collapses_vs_int8(ctx):
    d, m = ctx.data, ctx.models[0]
    a8 = build_qmlp(m, ctx.Xcal, 8, 8, True).accuracy(d.X_test, d.y_test)
    a2 = build_qmlp(m, ctx.Xcal, 2, 8, False).accuracy(d.X_test, d.y_test)
    assert a2 < a8 - 0.2


def test_int32_accumulators_suffice_for_int8(ctx):
    q = build_qmlp(ctx.models[0], ctx.Xcal, 8, 8, True)
    assert q.max_accumulator_bits(ctx.data.X_test) <= 32


def test_to_fixed_saturates_and_roundtrips():
    q, n = to_fixed(np.array([0.5, 100.0, -100.0]), 8, 4, return_sat=True)
    assert n == 2 and q[1] == 127 and q[2] == -128
    assert from_fixed(q[:1], 4)[0] == 0.5


def test_fixed_forward_tracks_float_and_saturation_counted(ctx):
    d, m = ctx.data, ctx.models[0]
    logits, _, frac = fixed_forward(m, d.X_test, 8, 6, 16, 8)
    agree = np.mean(np.argmax(logits, 1) == m.predict(d.X_test))
    assert agree > 0.95 and frac < 0.01
    _, _, frac_bad = fixed_forward(m, d.X_test, 8, 6, 16, 14)
    assert frac_bad > frac


def test_packed_bytes():
    assert packed_bytes(10, 4) == 5 and packed_bytes(9, 3) == 4 and packed_bytes(8, 8) == 8
