import numpy as np

from dsl import datasheetgen as G, extract as E, qa
from dsl.chunker import chunk_docs, line_to_chunk
from dsl.retrieval import Retriever


def setup(n=20, seed=1, **kw):
    docs = G.make_corpus(G.GenConfig(n_parts=n, seed=seed, **kw))
    ch = chunk_docs(docs)
    r = Retriever("hybrid", expand=True, part_filter=True).fit([c.text for c in ch], [c.part for c in ch])
    return docs, ch, r


def test_parse_question_stat_and_cond():
    s, c = qa.parse_question("What is the maximum Dropout Voltage of TLV1234 at IOUT = 300 mA?")
    assert s == "max" and c[1] == "A" and abs(c[0] - 0.3) < 1e-12
    assert qa.parse_question("TLV1234 typical iq")[0] == "typ"
    assert qa.parse_question("TLV1234 iq")[0] == "typ" and qa.parse_question("x")[1] is None


def test_pipeline_clean_accuracy_is_high():
    docs, ch, r = setup(styles=("pipe", "space"))
    qs = G.make_questions(docs, np.random.default_rng(0), 80, paraphrase=0.0, typo=0.0)
    p = qa.QAPipeline(ch, r, E.RuleExtractor())
    res = qa.evaluate_qa(p, qs, docs, line_to_chunk(ch))
    assert res["acc"] > 0.85 and sum(res["cats"].values()) == 80


def test_category_counts_sum_and_keys():
    docs, ch, r = setup(n=10)
    qs = G.make_questions(docs, np.random.default_rng(2), 30)
    res = qa.evaluate_qa(qa.QAPipeline(ch, r, E.NaiveExtractor()), qs, docs, line_to_chunk(ch), keep=True)
    assert set(res["cats"]) == {"correct", "retrieval_miss", "extract_error", "wrong_row"}
    assert sum(res["cats"].values()) == 30 and len(res["ranks"]) == 30
    assert all(e["cat"] != "correct" for e in res["examples"])


def test_naive_extractor_underperforms_rules():
    docs, ch, r = setup(styles=("pipe", "space"))
    qs = G.make_questions(docs, np.random.default_rng(3), 60, paraphrase=0.0, typo=0.0)
    l2c = line_to_chunk(ch)
    a = qa.evaluate_qa(qa.QAPipeline(ch, r, E.RuleExtractor()), qs, docs, l2c)["acc"]
    b = qa.evaluate_qa(qa.QAPipeline(ch, r, E.NaiveExtractor()), qs, docs, l2c)["acc"]
    assert a > b


def test_is_correct_requires_dimension():
    q = G.Question("t", "p", "x", "typ", 0.1, "A", None, False, ("p", 0))
    assert qa.is_correct(qa.Answer(0.1, "A"), q) and not qa.is_correct(qa.Answer(0.1, "V"), q)
    assert not qa.is_correct(qa.Answer(None), q)


def test_empty_retrieval_returns_none():
    docs, ch, r = setup(n=6)
    p = qa.QAPipeline(ch, r, E.RuleExtractor(), topk=1)
    a = p.answer("zzzz qqqq", order=np.array([0]))  # chunk 0 is the overview chunk (no spec rows)
    assert a.value_si is None
