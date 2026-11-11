import numpy as np

from dsl import confusion as C
from dsl.schema import L2I, LABELS


class Fixed:
    """Stub tagger that predicts a fixed label sequence per token text."""

    def __init__(self, mapping):
        self.m = mapping

    def predict_tokens(self, toks):
        return [self.m.get(t, "O") for t in toks]


LINES = [[("Slew", "NAME"), ("SR", "SYM"), ("12", "TYP"), ("V/us", "UNIT")], [("pin", "O"), ("1", "O")]]


def test_confusion_counts():
    cm = C.confusion_matrix(Fixed({"Slew": "NAME", "SR": "NAME", "12": "TYP", "V/us": "UNIT"}), LINES)
    assert cm.sum() == 6 and cm[L2I["SYM"], L2I["NAME"]] == 1 and cm[L2I["O"], L2I["O"]] == 2
    assert C.accuracy(cm) == 5 / 6


def test_per_label_prf_values():
    cm = C.confusion_matrix(Fixed({"Slew": "NAME", "SR": "NAME", "12": "TYP", "V/us": "UNIT"}), LINES)
    d = C.per_label_prf(cm)
    assert d["NAME"]["precision"] == 0.5 and d["NAME"]["recall"] == 1.0
    assert d["SYM"]["recall"] == 0.0 and d["SYM"]["support"] == 1


def test_top_confusions_ignores_diagonal():
    cm = np.zeros((len(LABELS),) * 2, dtype=int)
    cm[1, 1] = 100
    cm[2, 1] = 7
    cm[4, 5] = 3
    assert C.top_confusions(cm, 5) == [("SYM", "NAME", 7), ("MIN", "TYP", 3)]


def test_empty_lines_are_skipped():
    assert C.confusion_matrix(Fixed({}), [[]]).sum() == 0
