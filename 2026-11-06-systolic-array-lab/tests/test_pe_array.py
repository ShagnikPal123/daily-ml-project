import numpy as np
import pytest

from sysarray.pe_array import OutputStationaryArray, WeightStationaryArray, toggles
from sysarray.verify import random_int8


@pytest.mark.parametrize("cls", [OutputStationaryArray, WeightStationaryArray])
def test_single_tile_exact(cls):
    rng = np.random.default_rng(1)
    A, B = random_int8(rng, (5, 6)), random_int8(rng, (6, 4))
    C, st = cls(8, 8).run(A, B)
    assert np.array_equal(C, A.astype(np.int64) @ B.astype(np.int64))
    assert st.macs == 5 * 6 * 4


def test_os_cycle_formula():
    rng = np.random.default_rng(2)
    A, B = random_int8(rng, (4, 9)), random_int8(rng, (9, 3))
    _, st = OutputStationaryArray(4, 4).run(A, B)
    assert st.cycles == (9 + 4 + 3 - 2) + 4


def test_ws_cycle_formula_and_preload():
    rng = np.random.default_rng(3)
    A, W = random_int8(rng, (10, 4)), random_int8(rng, (4, 3))
    _, st = WeightStationaryArray(4, 4).run(A, W)
    assert st.weight_load_cycles == 4
    assert st.cycles == 4 + (10 + 4 + 3 - 2)


def test_extreme_values_no_overflow():
    A = np.full((4, 64), -128, dtype=np.int8)
    B = np.full((64, 4), -128, dtype=np.int8)
    C, _ = OutputStationaryArray(4, 4).run(A, B)
    assert np.all(C == 64 * 128 * 128)  # 1,048,576 fits int32 with room to spare


def test_oversized_tile_rejected():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        OutputStationaryArray(2, 2).run(random_int8(rng, (3, 3)), random_int8(rng, (3, 2)))
    with pytest.raises(ValueError):
        WeightStationaryArray(2, 2).run(random_int8(rng, (3, 3)), random_int8(rng, (3, 2)))


def test_non_int8_rejected():
    with pytest.raises(TypeError):
        OutputStationaryArray(2, 2).run(np.ones((2, 2), dtype=np.int32), np.ones((2, 2), dtype=np.int8))


def test_wavefront_is_diagonal():
    """In OS the first MAC fires at PE(0,0) in cycle 0; PE(i,j) first fires at cycle i+j."""
    rng = np.random.default_rng(5)
    arr = OutputStationaryArray(4, 4, record_trace=True)
    arr.run(random_int8(rng, (4, 5)), random_int8(rng, (5, 4)))
    first = np.full((4, 4), -1)
    for t, v in enumerate(arr.trace):
        first[(first < 0) & v] = t
    assert np.array_equal(first, np.add.outer(np.arange(4), np.arange(4)))


def test_zero_mac_accounting():
    A = np.zeros((4, 4), dtype=np.int8)
    B = np.ones((4, 4), dtype=np.int8)
    _, st = OutputStationaryArray(4, 4).run(A, B)
    assert st.zero_macs == st.macs == 64
    assert st.zero_fraction == 1.0


def test_utilization_bounded_and_underfill_penalised():
    rng = np.random.default_rng(6)
    A, B = random_int8(rng, (2, 16)), random_int8(rng, (16, 2))
    _, small = OutputStationaryArray(2, 2).run(A, B)
    _, big = OutputStationaryArray(16, 16).run(A, B)
    assert 0 < big.utilization < small.utilization <= 1.0


def test_toggle_counter():
    a = np.array([0, 1, -1], dtype=np.int8)
    b = np.array([0, 0, 0], dtype=np.int8)
    assert toggles(a, b) == 0 + 1 + 8
