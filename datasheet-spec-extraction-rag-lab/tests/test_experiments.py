import json

import pytest

from dsl import cli, experiments as X


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("res")
    cfg = X.load_config(None, quick=True)
    R = X.run_all(cfg, str(out), log=lambda *_: None)
    return cfg, R, out


def test_outputs_exist_and_nonempty(quick_run):
    cfg, R, out = quick_run
    for f in R["figures"] + ["results.json", "tables.md"]:
        assert (out / f).stat().st_size > 0
    assert len(R["figures"]) == 5


def test_result_structure(quick_run):
    cfg, R, _ = quick_run
    assert set(R["extraction"]) == {"naive-regex", "rules", "rules+ocr-repair", "tagger (ours)"}
    assert "full" in R["ablation"] and "no Viterbi (independent argmax)" in R["ablation"]
    assert "random" in R["retrieval"]["header=True|para=0.5"]
    assert set(R["qa"]["cats"]["0.0"]) <= {"correct", "retrieval_miss", "extract_error", "wrong_row"}
    assert all(0 <= v["mean"] <= 1 for d in R["extraction"].values() for v in d.values())


def test_tables_and_json_roundtrip(quick_run):
    _, R, out = quick_run
    t = (out / "tables.md").read_text()
    assert all(f"### T{i}." in t for i in range(1, 13))
    assert json.loads((out / "results.json").read_text())["config"]["seeds"] == [0, 1]


def test_experiments_are_deterministic():
    cfg = X.load_config(None, quick=True)
    a = X.exp_extraction(cfg, X.TaggerCache(cfg))
    b = X.exp_extraction(cfg, X.TaggerCache(cfg))
    assert a == b


def test_load_config_override(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"n_test": 5}')
    assert X.load_config(str(p))["n_test"] == 5 and X.load_config(str(p))["topk"] == 5


def test_cli_generate_and_extract(capsys):
    assert cli.main(["generate", "--style", "pipe", "--seed", "1"]) == 0
    assert "ELECTRICAL CHARACTERISTICS" in capsys.readouterr().out
    assert cli.main(["extract", "Slew Rate | SR | - | - | 12 | - | V/us"]) == 0
    out = capsys.readouterr().out
    assert "rules" in out and "V/us" in out


def test_cli_ask_reports_answer(capsys):
    from dsl.datasheetgen import GenConfig, make_corpus
    part = make_corpus(GenConfig(n_parts=24, noise=0.02, sep_damage=0.05, seed=0))[0].part
    assert cli.main(["ask", f"What is the typical Quiescent Current of the {part}?", "--seed", "0"]) == 0
    assert "answer (SI)" in capsys.readouterr().out
