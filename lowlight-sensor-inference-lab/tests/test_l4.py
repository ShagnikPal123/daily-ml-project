import json
import os

import numpy as np
import pytest

from lowlight import cli, data, experiments as E, report

TINY = {
    "seeds": [0, 1], "photons": [5, 50], "denoisers": ["identity", "gaussian", "gat"],
    "train_mode_ablation": {"photons": [5], "denoisers": ["identity"]},
    "gat_ablation": {"photons": [5]}, "patch_ablation": {"photons": [5], "sizes": [3]},
    "bits_sweep": {"photons": [20], "bits": [4], "denoisers": ["identity"], "seeds": [0, 1]},
    "read_noise_sweep": {"photons": [20], "values": [0.0], "denoisers": ["identity"], "seeds": [0, 1]},
    "classifier_study": {"photons": [50], "classifiers": ["softmax"], "denoiser": "identity", "seeds": [0]},
    "failure": {"photons": 5, "top_k": 3}, "profile": {"photons": 20, "n_images": 20},
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("res")
    cfg = E.load_config(TINY)
    return cfg, E.run_all(cfg, str(out)), out


def test_load_config_merges_and_rejects_unknown():
    cfg = E.load_config({"seeds": [7]})
    assert cfg["seeds"] == [7] and cfg["sensor"]["adc_bits"] == 10
    with pytest.raises(ValueError):
        E.load_config({"nonsense": 1})


@pytest.mark.parametrize("bad", [{"photons": [0, 5]}, {"denoisers": ["gaussian"]}, {"denoisers": ["identity", "bm3d"]},
                                 {"seeds": []}, {"failure": {"photons": 7, "top_k": 2}}])
def test_config_validation_errors(bad):
    with pytest.raises(ValueError):
        E.load_config(bad)


def test_default_and_quick_configs_load():
    for n in ("default", "quick"):
        assert E.load_config(os.path.join(os.path.dirname(cli.__file__), "..", "configs", f"{n}.json"))["denoisers"]


def test_build_tasks_count():
    cfg = E.load_config(TINY)
    tags = [t[0] for t in E.build_tasks(cfg)]
    assert tags.count("headline") == 2 * 2 * 3 and tags.count("train_clean") == 2 and tags.count("gat_algebraic") == 2
    assert tags.count("bits") == 2 and tags.count("classifier") == 1


def test_group_stats_and_paired_on_synthetic_records():
    recs = [{"photons": 5, "denoiser": d, "seed": s, "acc": a + 0.01 * s, "psnr": 10.0, "ssim": 0.5, "exp": "headline"}
            for d, a in (("identity", 0.5), ("ridge", 0.7)) for s in range(4)]
    g = E.group_stats(recs, ("photons", "denoiser"))
    assert len(g) == 2 and g[1]["acc"] == pytest.approx(0.715) and g[0]["n"] == 4
    pv = E.paired_vs_baseline(recs)
    assert len(pv) == 1 and pv[0]["diff"] == pytest.approx(0.2) and pv[0]["significant"]


def test_parallel_matches_serial():
    cfg = E.load_config(TINY)
    tasks = E.build_tasks(cfg)[:3]
    a = [r["acc"] for r in E.execute(tasks, 16, 1)]
    b = [r["acc"] for r in E.execute(tasks, 16, 2)]
    assert a == b


def test_run_all_outputs(run):
    cfg, res, out = run
    assert res["n_trials"] == len(E.build_tasks(cfg)) and res["runtime_sec"] > 0
    for k in ("headline", "paired_acc", "train_mode", "gat_ablation", "patch", "bits", "readnoise", "classifier", "failure", "deployment", "profile"):
        assert res[k], k
    for f in ("results.json", "records.json", "config_resolved.json"):
        assert (out / f).stat().st_size > 0
    assert json.load(open(out / "results.json"))["n_trials"] == res["n_trials"]


def test_failure_analysis_fields(run):
    _, res, _ = run
    f = res["failure"]
    assert len(f["per_class_identity"]) == 10 and len(f["top_confusions"]) == 3
    assert np.array(f["confusion_best"]).sum() > 0 and 0 <= f["all_methods_wrong_frac"][0] <= 1
    assert len(f["metric_disagreement"]) == 2


def test_deployment_table(run):
    _, res, _ = run
    rows = {r["denoiser"]: r for r in res["deployment"]}
    assert rows["identity"]["denoiser_macs"] == 0 and rows["gaussian"]["denoiser_macs"] > 0
    assert rows["gat"]["energy_uj_fp32"] > rows["gat"]["energy_uj"]


def test_report_figures_and_tables(run, tmp_path):
    _, res, _ = run
    scenes, _ = data.load_scenes(16)
    figs = report.render_figures(res, str(tmp_path), scenes)
    assert len(figs) == 8 and all(os.path.getsize(f) > 1000 for f in figs)
    t = report.render_tables(res)
    for name in ("headline_acc", "paired_acc", "deployment", "failure", "bits"):
        assert f"### TABLE {name}" in t


def test_md_table_and_pm():
    assert report.md_table(["a", "b"], [[1, 2]]).count("\n") == 2
    assert report.pm(0.5, 0.01, 1, True) == "50.0 ± 1.0" and "nan" not in report.pm(0.5, float("nan"))


def test_cli_parser_and_commands(tmp_path, capsys):
    p = cli.build_parser()
    assert p.parse_args(["run", "--quick"]).quick and p.parse_args(["demo", "--photons", "9"]).photons == 9
    cfgf = tmp_path / "c.json"; cfgf.write_text(json.dumps(TINY))
    assert cli.main(["run", "--config", str(cfgf), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "tables.md").stat().st_size > 0 and (tmp_path / "o" / "figures" / "failure_analysis.png").exists()
    assert cli.main(["report", "--out", str(tmp_path / "o")]) == 0
    assert cli.main(["budget", "--config", str(cfgf)]) == 0
    assert cli.main(["ptc", "--out", str(tmp_path / "p")]) == 0 and (tmp_path / "p" / "figures" / "photon_transfer.png").exists()
    assert cli.main(["demo", "--photons", "50", "--digit", "1", "--denoiser", "gaussian"]) == 0
    assert "predicted=" in capsys.readouterr().out
