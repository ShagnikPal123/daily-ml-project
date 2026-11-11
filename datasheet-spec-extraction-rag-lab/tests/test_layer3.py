import numpy as np
import pytest

from dsl import datasheetgen as G, extract as E, indexcompress as IC, retrieval as R
from dsl.chunker import chunk_docs, line_to_chunk
from dsl.schema import Extracted
from dsl.tagger import SeqTagger
from dsl.textnorm import words


def docs(seed=0, n=20, **kw):
    return G.make_corpus(G.GenConfig(n_parts=n, seed=seed, **kw))


def test_assemble_pipe_row():
    toks = ["Dropout", "Voltage", "¦", "VDO", "¦", "IOUT", "=", "100", "mA", "¦", "-", "¦", "120", "¦", "250", "¦", "mV"]
    labs = ["NAME", "NAME", "SEP", "SYM", "SEP", "COND", "COND", "COND", "COND", "SEP", "MIN", "SEP", "TYP", "SEP", "MAX", "SEP", "UNIT"]
    e = E.assemble(toks, labs)
    assert (e.name, e.symbol, e.cond, e.vmin, e.vtyp, e.vmax, e.unit) == ("Dropout Voltage", "VDO", "IOUT = 100 mA", None, 120.0, 250.0, "mV")


def test_assemble_none_without_unit_or_values():
    assert E.assemble(["FEATURES"], ["O"]) is None
    assert E.assemble(["5"], ["TYP"]) is None


def test_assemble_repairs_ocr_digits_and_units():
    e = E.assemble(["x", "l2O", "rnV"], ["NAME", "TYP", "UNIT"], repair=True)
    assert e.vtyp == 120.0 and e.unit == "mV"
    assert E.assemble(["x", "l2O", "rnV"], ["NAME", "TYP", "UNIT"], repair=False) is None


def test_symbol_with_parentheses():
    assert E.sym_clean("(RDS(on)):") == "RDS(on)" and E.sym_clean("(VDO)") == "VDO"


def test_parse_cond():
    v, d = E.parse_cond("IOUT = 100 mA")
    assert d == "A" and v == pytest.approx(0.1)
    assert E.parse_cond("-") is None and E.parse_cond("f = l0O kHz")[0] == pytest.approx(1e5)


def test_values_match_unit_independent():
    g = G.Row("p", "x", "Iq", "IQ", "-", None, 5.0, 10.0, "uA")
    assert E.values_match(Extracted("Iq", "", "", None, 0.005, 0.01, "mA"), g)
    assert not E.values_match(Extracted("Iq", "", "", None, 5.0, 10.0, "mA"), g)
    assert not E.values_match(Extracted("Iq", "", "", 1.0, 5.0, 10.0, "uA"), g)


def test_rule_extractor_perfect_on_clean_tables():
    r = E.eval_extraction(E.RuleExtractor().extract, docs(styles=("pipe", "space", "inline")))
    assert r["row"]["f1"] == 1.0 and r["detect"]["precision"] == 1.0


def test_naive_extractor_is_much_worse():
    ds = docs(styles=("pipe", "space"))
    naive = E.eval_extraction(E.NaiveExtractor().extract, ds)
    assert naive["row"]["f1"] < 0.6 and naive["detect"]["precision"] < 1.0


def test_sep_damage_breaks_rules_more_than_tagger():
    train, test = docs(seed=1, sep_damage=0.15), docs(seed=2, n=15, sep_damage=0.15, styles=("pipe", "space"))
    tg = E.TaggerExtractor(SeqTagger(epochs=4).fit([ln.tags for d in train for ln in d.lines]))
    rule = E.eval_extraction(E.RuleExtractor().extract, test)["row"]["f1"]
    tag = E.eval_extraction(tg.extract, test)["row"]["f1"]
    assert tag > rule + 0.1


def test_ocr_repair_helps_rules_under_noise():
    test = docs(seed=3, n=20, noise=0.04, styles=("pipe", "space"))
    a = E.eval_extraction(E.RuleExtractor(repair=False).extract, test)["row"]["f1"]
    b = E.eval_extraction(E.RuleExtractor(repair=True).extract, test)["row"]["f1"]
    assert b > a


def test_eval_extraction_error_collection():
    r = E.eval_extraction(E.NaiveExtractor().extract, docs(n=6), keep_errors=True)
    kinds = {e["kind"] for e in r["errors"]}
    assert "false_positive" in kinds and "wrong_fields" in kinds
    assert r["tp"] + r["fn"] == sum(len(d.rows) for d in docs(n=6))


def test_quantize_error_bound_and_shapes():
    X = np.random.default_rng(0).normal(size=(30, 16))
    for bits in (8, 4, 2):
        Q, sc = IC.quantize_symmetric(X, bits)
        err = np.abs(IC.dequantize(Q, sc) - X)
        assert (err <= sc / 2 + 1e-12).all()
        assert np.abs(Q).max() <= 2 ** (bits - 1) - 1


def test_index_bytes_monotone():
    assert IC.dense_index_bytes(100, 32, 8) < IC.dense_index_bytes(100, 32, 32)
    assert IC.dense_index_bytes(100, 32, 4) < IC.dense_index_bytes(100, 32, 8)


def test_compressed_lsa_recall_degrades_gracefully():
    ds = docs(seed=6, n=20)
    ch = chunk_docs(ds)
    l2c = line_to_chunk(ch)
    lsa = R.LSA(32).fit([words(c.text) for c in ch])
    qs = G.make_questions(ds, np.random.default_rng(1), 50, paraphrase=0.0, typo=0.0)
    res = IC.eval_compressed_lsa(lsa, [{t: 1.0 for t in words(q.text)} for q in qs], [l2c[q.line_uid] for q in qs], [32, 8, 2])
    assert res[0]["recall@5"] >= res[2]["recall@5"] - 0.05 and res[1]["recall@5"] >= res[0]["recall@5"] - 0.1
    assert res[0]["bytes"] > res[1]["bytes"] > res[2]["bytes"]
