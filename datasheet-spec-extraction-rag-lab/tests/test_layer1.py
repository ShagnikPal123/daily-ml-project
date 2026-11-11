import numpy as np
import pytest

from dsl import chunker, datasheetgen as G
from dsl.schema import LABELS
from dsl.textnorm import tokenize_spans
from dsl.units import close, to_si


def corpus(**kw):
    return G.make_corpus(G.GenConfig(n_parts=15, seed=3, **kw))


def test_corpus_is_deterministic_and_unique():
    a, b = corpus(), corpus()
    assert [d.part for d in a] == [d.part for d in b]
    assert len({d.part for d in a}) == 15
    assert [ln.text for ln in a[0].lines] == [ln.text for ln in b[0].lines]


def test_tags_align_with_tokens_for_all_styles_and_noise():
    for noise in (0.0, 0.05):
        for d in corpus(noise=noise, sep_damage=0.1):
            for ln in d.lines:
                toks = [t for t, _, _ in tokenize_spans(ln.text)]
                assert [t for t, _ in ln.tags] == toks
                assert all(l in LABELS for _, l in ln.tags)


def test_clean_rows_render_their_values():
    for d in corpus():
        for ln in d.lines:
            if ln.row is None:
                continue
            r = ln.row
            for v in (r.vmin, r.vtyp, r.vmax):
                if v is not None:
                    assert G.fmt_num(v) in ln.text
            assert r.unit in ln.text


def test_gold_labels_recover_row_fields():
    d = corpus(styles=("pipe",))[0]
    ln = next(l for l in d.lines if l.row and l.row.section == "ec")
    labs = dict()
    for tok, lab in ln.tags:
        labs.setdefault(lab, []).append(tok)
    assert " ".join(labs["NAME"]) == ln.row.name
    assert labs["SYM"] == [ln.row.symbol]
    assert labs["UNIT"] == [ln.row.unit]


def test_inline_style_has_keywords():
    d = corpus(styles=("inline",))[0]
    ln = next(l for l in d.lines if l.row and l.row.section == "ec")
    assert "(" in ln.text and ("typ " in ln.text or "min " in ln.text or "max " in ln.text)


def test_noise_changes_text_but_not_gold_rows():
    clean = G.make_corpus(G.GenConfig(n_parts=10, seed=1, noise=0.0))
    noisy = G.make_corpus(G.GenConfig(n_parts=10, seed=1, noise=0.3))
    assert [d.part for d in clean] == [d.part for d in noisy]
    diff = sum(a.text != b.text for c, n in zip(clean, noisy) for a, b in zip(c.lines, n.lines) if a.row)
    assert diff > 0


def test_values_are_physically_ordered():
    for d in corpus():
        for r in d.rows:
            if r.section != "ec" or r.pid in ("res", "flash", "ram"):
                continue
            vals = [v for v in (r.vmin, r.vtyp, r.vmax) if v is not None]
            assert vals == sorted(vals), (r.pid, vals)


def test_every_row_unit_is_parsable():
    for d in corpus():
        for r in d.rows:
            to_si(1.0, r.unit)


def test_chunker_covers_every_line_once():
    docs = corpus()
    chunks = chunker.chunk_docs(docs, size=4)
    m = chunker.line_to_chunk(chunks)
    assert len(m) == sum(len(d.lines) for d in docs)
    assert all(len(c.line_idxs) <= 4 for c in chunks)


def test_chunk_header_toggle():
    docs = corpus()
    h = chunker.chunk_docs(docs, header=True)
    nh = chunker.chunk_docs(docs, header=False)
    assert len(h) == len(nh) and h[3].text.startswith("[") and not nh[3].text.startswith("[")


def test_questions_have_consistent_answers():
    docs = corpus()
    rng = np.random.default_rng(0)
    qs = G.make_questions(docs, rng, 60, paraphrase=0.5, typo=0.2)
    by = {d.part: d for d in docs}
    assert len(qs) == 60
    for q in qs:
        row = by[q.part].lines[q.line_uid[1]].row
        v = {"min": row.vmin, "typ": row.vtyp, "max": row.vmax}[q.stat]
        assert v is not None and close(q.answer_si, to_si(v, row.unit)[0], 1e-9)
        if row.cond_value is not None:
            assert "=" in q.text
    assert any(q.paraphrased for q in qs)


def test_typo_changes_name():
    rng = np.random.default_rng(0)
    assert G._typo("TLV7421A", rng) != "TLV7421A"
    assert G._typo("ABC", rng) == "ABC"
