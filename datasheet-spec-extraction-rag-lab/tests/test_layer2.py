import numpy as np
import pytest

from dsl import datasheetgen as G, retrieval as R, rules, tagger as TG
from dsl.chunker import chunk_docs, line_to_chunk
from dsl.schema import Extracted
from dsl.textnorm import words


def docs(seed=0, n=24, **kw):
    return G.make_corpus(G.GenConfig(n_parts=n, seed=seed, **kw))


# ---------------------------------------------------------------- rules
def test_parse_number_and_repair():
    assert rules.parse_number("1.5") == 1.5 and rules.parse_number("-") is None
    assert rules.parse_number("l2O", repair=True) == 120.0
    assert rules.parse_number("l2O", repair=False) is None
    assert rules.parse_number("abc", repair=True) is None


def test_repair_unit():
    assert rules.repair_unit("rnV") == "mV" and rules.repair_unit("uA,") == "uA"


def test_table_parser_on_clean_rows():
    ok = tot = 0
    for d in docs(styles=("pipe", "space")):
        for ln in d.lines:
            if ln.row:
                tot += 1
                e = rules.parse_line(ln.text)
                ok += e is not None and e.vtyp == ln.row.vtyp and e.vmax == ln.row.vmax and e.unit == ln.row.unit
    assert ok == tot


def test_inline_parser_on_clean_rows():
    for d in docs(styles=("inline",), n=8):
        for ln in d.lines:
            if ln.row:
                e = rules.parse_line(ln.text)
                assert e is not None and (e.vmin, e.vtyp, e.vmax) == (ln.row.vmin, ln.row.vtyp, ln.row.vmax)


def test_rules_reject_distractors_and_damaged_gaps():
    assert rules.parse_line("1   VIN   Power pin, bypass with 1 uF capacitor") is None
    assert rules.parse_line("Parameter | Symbol | Conditions | Min | Typ | Max | Unit") is None
    assert rules.parse_line("Dropout Voltage VDO IOUT = 100 mA | - | 120 | 250 | mV") is None


def test_naive_baseline_makes_distractor_false_positive():
    e = rules.naive_line("1   VIN   Power pin, bypass with 1 uF capacitor")
    assert isinstance(e, Extracted) and e.unit == "uF"
    assert rules.naive_line("FEATURES") is None


# ---------------------------------------------------------------- tagger
def _lines(ds):
    return [ln.tags for d in ds for ln in d.lines]


def test_features_groups_toggle():
    toks = ["Dropout", "Voltage", "¦", "VDO", "¦", "120", "mV"]
    full = TG.token_features(toks, frozenset(TG.GROUPS))
    lex = TG.token_features(toks, frozenset(["lex"]))
    assert len(full[0]) > len(lex[0]) and "w=dropout" in full[0]
    assert any(f.startswith("cell=") for f in full[5])


def test_tagger_learns_clean_layouts():
    train, test = docs(seed=1, n=20), docs(seed=2, n=10)
    tg = TG.SeqTagger(epochs=4).fit(_lines(train))
    assert tg.token_accuracy(_lines(test)) > 0.97


def test_tagger_is_deterministic():
    train = docs(seed=1, n=6)
    a = TG.SeqTagger(epochs=2, seed=5).fit(_lines(train))
    b = TG.SeqTagger(epochs=2, seed=5).fit(_lines(train))
    assert np.allclose(a.W, b.W) and np.allclose(a.T, b.T)


def test_viterbi_transitions_help_or_equal_on_damaged_gaps():
    train = docs(seed=1, n=20, sep_damage=0.2)
    test = docs(seed=2, n=12, sep_damage=0.2)
    full = TG.SeqTagger(epochs=4).fit(_lines(train)).token_accuracy(_lines(test))
    indep = TG.SeqTagger(epochs=4, use_viterbi=False).fit(_lines(train)).token_accuracy(_lines(test))
    assert full >= indep - 0.01


def test_tagger_empty_input():
    tg = TG.SeqTagger(epochs=1).fit(_lines(docs(n=3)))
    assert tg.predict_tokens([]) == [] and tg.predict("")[1] == []


# ---------------------------------------------------------------- retrieval
def test_bm25_prefers_matching_doc():
    d = [words(t) for t in ["dropout voltage ldo", "slew rate opamp", "switching frequency buck"]]
    s = R.BM25().fit(d).scores({"slew": 1.0, "rate": 1.0})
    assert s.argmax() == 1 and s[0] == 0.0


def test_tfidf_rows_are_unit_norm():
    t = R.Tfidf().fit([words("a b c voltage"), words("voltage current"), words("zzz")])
    assert np.allclose(np.sqrt(t.X.multiply(t.X).sum(axis=1)).ravel(), 1.0)


def test_lsa_shapes_and_order():
    d = [words(t) for t in ["dropout voltage ldo regulator", "voltage regulator output", "slew rate opamp bandwidth", "opamp gain bandwidth"]]
    l = R.LSA(k=2).fit(d)
    assert l.D.shape == (4, 2) and l.s[0] >= l.s[1]
    assert l.scores({"opamp": 1.0}).argmax() in (2, 3)


def test_thesaurus_expands_alias_to_canonical_name():
    t = R.Thesaurus()
    w = t.expand(words("standby current"))
    assert "quiescent" in w and w["standby"] == 1.0 and w["quiescent"] == 0.5
    assert t.expand(words("hello world")) == {"hello": 1.0, "world": 1.0}


def test_rrf_combines_rankings():
    a, b = np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])
    f = R.rrf([a, b])
    assert f[1] < f[0] and np.isclose(f[0], f[2])


def test_retriever_end_to_end_finds_gold_chunk():
    ds = docs(seed=4, n=20)
    ch = chunk_docs(ds)
    l2c = line_to_chunk(ch)
    qs = G.make_questions(ds, np.random.default_rng(0), 60, paraphrase=0.0, typo=0.0)
    r = R.Retriever("bm25", part_filter=True).fit([c.text for c in ch], [c.part for c in ch])
    hits = sum(R.rank_of(r.rank(q.text), l2c[q.line_uid]) in (1, 2, 3) for q in qs)
    assert hits / len(qs) > 0.8


def test_detect_part_fuzzy():
    ds = docs(seed=4, n=10)
    ch = chunk_docs(ds)
    r = R.Retriever("bm25", part_filter=True).fit([c.text for c in ch], [c.part for c in ch])
    p = ds[3].part
    assert r.detect_part(f"what is the typical gain of {p}?") == p.lower()
    assert r.detect_part(f"what is the typical gain of {p[:-2]}?") is not None
    assert r.detect_part("nothing to see") is None


def test_part_filter_ranks_part_first():
    ds = docs(seed=5, n=12)
    ch = chunk_docs(ds)
    r = R.Retriever("tfidf", part_filter=True).fit([c.text for c in ch], [c.part for c in ch])
    order = r.rank(f"{ds[2].part} typical anything")
    assert ch[order[0]].part == ds[2].part
