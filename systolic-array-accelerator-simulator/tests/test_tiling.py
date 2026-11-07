import numpy as np
import pytest

from sysarray.tiling import (analytic_stats, analytic_traffic, ideal_cycles, overlapped_cycles,
                             tiled_matmul)
from sysarray.verify import random_int8, verify


@pytest.mark.parametrize("df", ["OS", "WS"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 13, 5), (33, 17, 9), (16, 16, 16)])
def test_tiled_matches_numpy(df, shape):
    M, K, N = shape
    rng = np.random.default_rng(M * 100 + K)
    A, B = random_int8(rng, (M, K)), random_int8(rng, (K, N))
    C, st, _ = tiled_matmul(A, B, 4, 4, df)
    assert np.array_equal(C, A.astype(np.int64) @ B.astype(np.int64))
    assert st.macs == M * K * N


@pytest.mark.parametrize("df", ["OS", "WS"])
def test_analytic_cycles_traffic_match_sim(df):
    rng = np.random.default_rng(11)
    M, K, N = 21, 30, 14
    A, B = random_int8(rng, (M, K)), random_int8(rng, (K, N))
    _, st, tr = tiled_matmul(A, B, 8, 4, df)
    an = analytic_stats(M, K, N, 8, 4, df)
    assert (st.cycles, st.macs, st.pe_cycles) == (an.cycles, an.macs, an.pe_cycles)
    assert tr == analytic_traffic(M, K, N, 8, 4, df)


def test_overlap_never_slower_than_serial_and_never_below_ideal():
    for df in ("OS", "WS"):
        for (M, K, N) in [(64, 64, 64), (100, 50, 30), (8, 300, 8)]:
            serial = analytic_stats(M, K, N, 8, 8, df).cycles
            ov = overlapped_cycles(M, K, N, 8, 8, df)
            assert ideal_cycles(M, K, N, 8, 8) <= ov <= serial


def test_batch1_fc_favours_os_over_ws():
    """With M=1 the WS array pays a full weight preload for one activation row."""
    os_c = analytic_stats(1, 512, 10, 16, 16, "OS").cycles
    ws_c = analytic_stats(1, 512, 10, 16, 16, "WS").cycles
    assert os_c < ws_c


def test_ws_has_more_scratchpad_traffic_than_os_when_k_tiled():
    t_os = analytic_traffic(64, 256, 64, 8, 8, "OS")
    t_ws = analytic_traffic(64, 256, 64, 8, 8, "WS")
    assert t_ws.sram_write > t_os.sram_write


def test_unknown_dataflow():
    with pytest.raises(ValueError):
        analytic_stats(4, 4, 4, 2, 2, "RS")


def test_verify_suite_passes():
    assert verify(n_cases=8, seed=3)["all_passed"]
