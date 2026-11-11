import numpy as np
import pytest

from dsl import datasheetgen as G
from dsl.extract import TaggerExtractor, eval_extraction
from dsl.fixedpoint import QuantTagger, float_model_bytes
from dsl.tagger import SeqTagger


@pytest.fixture(scope="module")
def model():
    tr = G.make_corpus(G.GenConfig(n_parts=16, seed=1, noise=0.02, sep_damage=0.05))
    te = G.make_corpus(G.GenConfig(n_parts=10, seed=2, noise=0.02, sep_damage=0.05))
    return SeqTagger(epochs=3).fit([ln.tags for d in tr for ln in d.lines]), te


def test_weight_quantisation_error_bound(model):
    tg, _ = model
    for bits in (8, 4):
        q = QuantTagger(tg, bits)
        err = np.abs(q.Wq * q.scale - tg.W)
        assert (err <= q.scale / 2 + 1e-12).all()
        assert np.abs(q.Wq).max() <= 2 ** (bits - 1) - 1


def test_int8_matches_float_accuracy(model):
    tg, te = model
    lines = [ln.tags for d in te for ln in d.lines]
    assert abs(QuantTagger(tg, 8).token_accuracy(lines) - tg.token_accuracy(lines)) < 0.01


def test_int8_extraction_f1_close_to_float(model):
    tg, te = model
    f = eval_extraction(TaggerExtractor(tg).extract, te)["row"]["f1"]
    q = eval_extraction(TaggerExtractor(QuantTagger(tg, 8)).extract, te)["row"]["f1"]
    assert abs(f - q) < 0.05


def test_model_bytes_shrink_with_bits(model):
    tg, _ = model
    b = [QuantTagger(tg, k).dense_bytes() for k in (8, 4, 2)]
    assert b[0] > b[1] > b[2]
    assert b[0] < float_model_bytes(tg)
    assert QuantTagger(tg, 8).sparse_bytes() < QuantTagger(tg, 8).dense_bytes()


def test_invalid_bits_and_empty_input(model):
    tg, _ = model
    with pytest.raises(ValueError):
        QuantTagger(tg, 12)
    assert QuantTagger(tg, 8).predict_tokens([]) == []


def test_integer_accumulation_is_int32(model):
    tg, _ = model
    q = QuantTagger(tg, 8)
    emit = q.emissions(["Slew", "Rate", "¦", "SR"])
    assert emit.shape == (4, 9) and np.isfinite(emit).all()
