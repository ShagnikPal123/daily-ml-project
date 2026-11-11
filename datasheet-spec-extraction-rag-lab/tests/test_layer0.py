import itertools
import math

import numpy as np
import pytest

from dsl import catalog, metrics, textnorm, units, viterbi


def test_parse_simple_and_prefixed_units():
    assert units.parse_unit("V") == (1.0, "V")
    assert units.parse_unit("mV") == (1e-3, "V")
    assert units.parse_unit("kSPS") == (1e3, "SPS")
    assert units.parse_unit("MHz")[1] == "Hz"


def test_unicode_unit_aliases():
    assert units.parse_unit("µA") == units.parse_unit("uA")
    assert units.parse_unit("mΩ") == (1e-3, "ohm")
    assert units.parse_unit("°C") == (1.0, "degC")


def test_compound_units():
    scale, dim = units.parse_unit("uA/MHz")
    assert dim == "A/Hz" and math.isclose(scale, 1e-12)
    assert units.parse_unit("V/us")[1] == "V/s"


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        units.parse_unit("furlong")
    assert not units.is_unit("xyz")


def test_convert_and_dimension_check():
    assert math.isclose(units.convert(1.5, "V", "mV"), 1500.0)
    assert math.isclose(units.convert(250, "uA", "mA"), 0.25)
    with pytest.raises(ValueError):
        units.convert(1, "V", "A")


def test_choose_unit_window():
    assert units.choose_unit(0.12, ["mV", "V"]) == "mV"
    assert units.choose_unit(2.4, ["mV", "V"]) == "V"
    assert units.choose_unit(3e-9, ["nA", "uA"]) == "nA"


def test_format_helpers():
    assert units.fmt_num(0.0125) == "0.0125"
    assert units.fmt_num(1500.0) == "1500"
    assert units.format_eng(0.0033, "V") == "3.3 mV"
    assert units.round_sig(123456, 3) == 123000.0


def test_normalize_and_words():
    assert textnorm.normalize("5 µA") == "5 uA"
    assert textnorm.words("What is the Dropout Voltage of TLV7421A?") == ["voltage"][:0] + ["dropout", "voltage", "tlv7421a"]
    assert textnorm.stem("currents") == "current" and textnorm.stem("mass") == "mass"


def test_tokenize_spans_columns():
    toks = textnorm.tokenize_spans("Dropout Voltage   VDO   IOUT = 100 mA | 120")
    texts = [t[0] for t in toks]
    assert texts.count(textnorm.SEP) == 3
    assert texts[:2] == ["Dropout", "Voltage"]
    line = "a | b"
    for tok, s, e in textnorm.tokenize_spans(line):
        assert line[s:e].strip() in (tok, "|", "")


def test_edit_distance_and_shape():
    assert textnorm.edit_distance("tlv7421", "tlv7412") == 2
    assert textnorm.edit_distance("", "abc") == 3
    assert textnorm.shape("IOUT") == "X" and textnorm.shape("1.5") == "d.d" and textnorm.shape("mA") == "xX"


def _rand_case(rng, T, K):
    return rng.normal(size=(T, K)), rng.normal(size=(K, K)), rng.normal(size=K)


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(30):
        T, K = int(rng.integers(1, 6)), int(rng.integers(2, 4))
        emit, trans, start = _rand_case(rng, T, K)
        best = max(itertools.product(range(K), repeat=T), key=lambda p: viterbi.path_score(list(p), emit, trans, start))
        path, score = viterbi.viterbi(emit, trans, start)
        assert math.isclose(score, viterbi.path_score(list(best), emit, trans, start))
        assert math.isclose(score, viterbi.path_score(path, emit, trans, start))


def test_viterbi_empty_and_single():
    assert viterbi.viterbi(np.zeros((0, 3)), np.zeros((3, 3))) == ([], 0.0)
    assert viterbi.viterbi(np.array([[0.1, 0.9]]), np.zeros((2, 2)))[0] == [1]


def test_rank_metrics():
    ranks = [1, 2, None, 5]
    assert metrics.recall_at_k(ranks, 1) == 0.25
    assert metrics.recall_at_k(ranks, 5) == 0.75
    assert math.isclose(metrics.mrr(ranks), (1 + 0.5 + 0.2) / 4)
    assert math.isclose(metrics.ndcg_at_k([1], 10), 1.0)
    assert metrics.recall_at_k([], 3) == 0.0


def test_prf_and_ci():
    d = metrics.prf(8, 2, 2)
    assert math.isclose(d["f1"], 0.8)
    assert metrics.prf(0, 0, 0)["f1"] == 0.0
    m, h = metrics.mean_ci([1.0, 1.0, 1.0])
    assert m == 1.0 and h == 0.0
    lo, hi = metrics.bootstrap_ci(np.arange(100.0), n_boot=200)
    assert 40 < lo < 49.5 < hi < 60


def test_catalog_is_consistent():
    n = 0
    for p in catalog.all_params():
        n += 1
        assert p.lo < p.hi or p.choices
        assert all(units.is_unit(u) for u in p.units)
        dims = {units.parse_unit(u)[1] for u in p.units}
        assert len(dims) == 1, p.pid
        for pat in p.pats:
            assert set(pat) <= {"min", "typ", "max"}
        if p.cond:
            assert units.is_unit(p.cond[1])
    assert n >= 28
