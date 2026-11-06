import numpy as np

from sysarray.energy import EnergyParams, dram_bytes, energy_breakdown, total_pj
from sysarray.pe_array import ArrayStats
from sysarray.quant import (dequantize, quantize_sym, requantize_fixed, sqnr_db,
                            to_fixed_multiplier)
from sysarray.tiling import Traffic


def test_quantize_roundtrip_error_bounded():
    x = np.random.default_rng(0).normal(size=1000)
    q, s = quantize_sym(x)
    assert np.max(np.abs(dequantize(q, s) - x)) <= s / 2 + 1e-12
    assert sqnr_db(x, dequantize(q, s)) > 35


def test_fixed_multiplier_accuracy():
    for m in [0.0003, 0.07, 0.49, 0.5, 0.93, 1.7]:
        mult, shift = to_fixed_multiplier(m)
        assert (1 << 30) <= mult < (1 << 31)
        assert abs(mult / 2**shift - m) / m < 1e-8


def test_requantize_matches_float_rounding():
    rng = np.random.default_rng(1)
    acc = rng.integers(-200000, 200000, size=500)
    m = 0.00123
    mult, shift = to_fixed_multiplier(m)
    got = requantize_fixed(acc, mult, shift).astype(int)
    want = np.clip(np.rint(acc * m), -128, 127)
    assert np.max(np.abs(got - want)) <= 1


def test_requantize_relu_clamps_negative():
    mult, shift = to_fixed_multiplier(0.5)
    assert requantize_fixed(np.array([-100, 100, 1000]), mult, shift, relu=True).tolist() == [0, 50, 127]


def test_dram_traffic_drops_when_buffer_fits():
    small = dram_bytes("OS", 256, 512, 256, 16, 16, 4 * 1024)
    large = dram_bytes("OS", 256, 512, 256, 16, 16, 1024 * 1024)
    assert large < small
    assert large == 256 * 512 + 512 * 256 + 256 * 256


def test_ws_psum_spill_only_when_buffer_small():
    tight = dram_bytes("WS", 256, 512, 256, 16, 16, 4 * 1024)
    roomy = dram_bytes("WS", 256, 512, 256, 16, 16, 64 * 1024)
    assert tight > roomy


def test_gating_reduces_mac_energy_only():
    st = ArrayStats(macs=1000, zero_macs=600, pe_cycles=1000, reg_toggles=0)
    p = EnergyParams()
    plain = energy_breakdown(st, Traffic(), 0, p, zero_gating=False)
    gated = energy_breakdown(st, Traffic(), 0, p, zero_gating=True)
    assert gated["mac"] < plain["mac"]
    assert gated["leakage"] == plain["leakage"]
    assert total_pj(gated) < total_pj(plain)
