import json
import numpy as np

from tinyml_lab import memplan as mp
from tinyml_lab.cli import main
from tinyml_lab.mcu import MCUS, Deployment, evaluate, fits_sram, mlp_deployment
from tinyml_lab.prune import (layer_storage_bytes, magnitude_masks, apply_masks, sparsity_of,
                              structured_prune)


def test_global_pruning_hits_target(ctx):
    m = ctx.models[0]
    p = apply_masks(m, magnitude_masks(m, 0.8, "global"))
    assert abs(sparsity_of(p) - 0.8) < 0.01


def test_layerwise_pruning_is_uniform(ctx):
    m = ctx.models[0]
    for M in magnitude_masks(m, 0.6, "layer"):
        assert abs(1 - M.mean() - 0.6) < 0.02


def test_structured_prune_shapes_and_forward(ctx):
    m = ctx.models[0]
    s = structured_prune(m, 0.5)
    assert s.sizes[0] == 64 and s.sizes[-1] == 10 and s.sizes[1] < m.sizes[1]
    assert s.forward(ctx.data.X_test).shape == (len(ctx.data.X_test), 10)


def test_storage_picks_cheapest_format():
    dense = np.ones((20, 20), dtype=np.int64)
    sparse = np.zeros((20, 20), dtype=np.int64); sparse[0, 0] = 1
    assert layer_storage_bytes(dense, 8)[1] == "dense"
    assert layer_storage_bytes(sparse, 8)[1] == "csr"


def test_liveness_on_hand_graph():
    g = mp.Graph("t", {"a": 8, "b": 8, "c": 8, "d": 8},
                 [mp.Op("o1", ["a"], "b"), mp.Op("o2", ["b"], "c"), mp.Op("o3", ["c", "a"], "d")], ["a"])
    L = {b.name: (b.first, b.last) for b in mp.liveness(g)}
    assert L == {"a": (0, 2), "b": (0, 1), "c": (1, 2), "d": (2, 2)}
    assert L["b"] == (0, 1) and L["d"] == (2, 2)


def test_all_planners_valid_and_bounded():
    for g in (mp.mlp_graph([64, 48, 32, 10]), mp.tiny_resnet_graph(), mp.unet_graph(), mp.random_graph(30, 1)):
        bufs = mp.liveness(g)
        lb = mp.lower_bound(bufs)
        _, naive = mp.plan_naive(bufs)
        for name, fn in mp.PLANNERS.items():
            plan, arena = fn(bufs)
            assert mp.validate(plan, arena), (g.name, name)
            assert lb <= arena <= naive


def test_search_never_worse_than_greedy():
    bufs = mp.liveness(mp.random_graph(40, 3))
    assert mp.plan_random_search(bufs)[1] <= mp.plan_greedy_by_size(bufs)[1]


def test_validate_rejects_overlap():
    a = mp.Buf("a", 8, 0, 2, 0); b = mp.Buf("b", 8, 1, 3, 4)
    assert not mp.validate([a, b], 16)


def test_mcu_fit_and_scaling():
    big = Deployment("big", 1_000_000, 0, 600_000, 100_000, 8)
    r = {m.name: evaluate(big, m) for m in MCUS}
    assert not r["AVR-class 8-bit"]["fits_flash"] and r["Cortex-M7"]["fits_flash"]
    assert r["Cortex-M7"]["latency_ms"] < r["Cortex-M0+"]["latency_ms"]
    assert not fits_sram(100_000, MCUS[0]) and fits_sram(1000, MCUS[0])


def test_int8_deployment_smaller_and_faster_than_fp32(ctx):
    m = ctx.models[0]
    f = mlp_deployment("f", m, 32, 32, False)
    i = mlp_deployment("i", m, 8, 8, True)
    assert i.flash_bytes < f.flash_bytes / 3
    m4 = MCUS[1]
    assert evaluate(i, m4)["latency_ms"] < evaluate(f, m4)["latency_ms"]


def test_cli_plan_smoke(tmp_path):
    main(["plan", "--out", str(tmp_path)])
    data = json.load(open(tmp_path / "metrics_plan.json"))
    assert len(data["memplan"]["graphs"]) == 5 and (tmp_path / "fig_memplan.png").exists()
