import json
import os

import numpy as np

from regimelab.cli import build_parser, main
from regimelab.config import ExpConfig
from regimelab.evaluation import evaluate_seed, make_data
from regimelab.experiments import aggregate, run_kalman_study, table_headline


def small_cfg(**kw):
    return ExpConfig(n_train=800, n_test=600, seeds=2, n_iter=10, **kw)


def test_make_data_reproducible_and_split_sized():
    c = small_cfg()
    a, b = make_data(c, 3), make_data(c, 3)
    assert np.array_equal(a.returns, b.returns) and len(a.returns) == 1400
    assert not np.array_equal(make_data(c, 4).returns, a.returns)


def test_evaluate_seed_contains_baselines_and_valid_metrics():
    out = evaluate_seed(small_cfg(separation=1.5), 0, ["threshold", "hmm-ret"])
    assert {"threshold", "hmm-ret", "buy-hold", "vol-target", "oracle"} <= set(out)
    for n in ("threshold", "hmm-ret"):
        assert 0 <= out[n]["acc"] <= 1 and out[n]["fit_s"] >= 0
    assert out["oracle"]["acc"] == 1.0


def test_oracle_beats_buy_and_hold_on_average():
    c = small_cfg(separation=1.5)
    o, b = [], []
    for s in range(4):
        r = evaluate_seed(c, s, [])
        o.append(r["oracle"]["sharpe"])
        b.append(r["buy-hold"]["sharpe"])
    assert np.mean(o) > np.mean(b)


def test_aggregate_and_table():
    runs = [evaluate_seed(small_cfg(), s, ["threshold"]) for s in range(3)]
    agg = aggregate(runs)
    assert len(agg["threshold"]["acc"]) == 2 and "| threshold |" in table_headline(agg)


def test_kalman_study_matches_theory():
    for row in run_kalman_study(n=3000):
        assert row["gain_empirical"] == __import__("pytest").approx(row["gain_theory"], rel=1e-6)
        assert row["mse_kalman"] < row["mse_raw"]


def test_cli_parser_and_show_config(capsys):
    assert build_parser().parse_args(["run", "--seeds", "2", "--quick"]).quick
    assert main(["show-config"]) == 0
    assert json.loads(capsys.readouterr().out)["k"] == 3


def test_cli_run_quick_writes_artifacts(tmp_path):
    out = tmp_path / "res"
    rc = main(["run", "--seeds", "2", "--n-train", "700", "--n-test", "500", "--quick", "--out", str(out)])
    assert rc == 0
    for f in ("summary.json", "summary.md", "timeline.png", "sweeps.png", "sharpe.png", "ablation.png", "kalman_gain.png"):
        assert os.path.getsize(out / f) > 0
