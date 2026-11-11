"""Layer 0: plain data containers shared by every layer (no project imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Token labels used by the sequence tagger.  Order matters (index = label id).
LABELS = ("O", "NAME", "SYM", "COND", "MIN", "TYP", "MAX", "UNIT", "SEP")
L2I = {l: i for i, l in enumerate(LABELS)}


@dataclass
class Row:
    """One specification row of a datasheet table, values in the *displayed* unit."""

    part: str
    pid: str
    name: str
    symbol: str
    cond: str
    vmin: Optional[float]
    vtyp: Optional[float]
    vmax: Optional[float]
    unit: str
    section: str = "ec"  # "ec" = electrical characteristics, "abs" = absolute maximum ratings
    cond_value: Optional[float] = None  # numeric part of the condition, in display units
    cond_unit: str = ""


@dataclass
class Line:
    text: str
    section: str
    row: Optional[Row] = None
    tags: List[Tuple[str, str]] = field(default_factory=list)  # (token, label) aligned by char span


@dataclass
class Doc:
    part: str
    family: str
    style: str
    lines: List[Line]

    @property
    def rows(self) -> List[Row]:
        return [ln.row for ln in self.lines if ln.row is not None]


@dataclass
class Question:
    text: str
    part: str
    pid: str
    stat: str  # "min" | "typ" | "max"
    answer_si: float
    dim: str
    cond_value_si: Optional[float]
    paraphrased: bool
    line_uid: Tuple[str, int]  # (part, line index) of the gold row


@dataclass
class Extracted:
    """A specification row recovered from text by an extractor (values in the printed unit)."""

    name: str
    symbol: str
    cond: str
    vmin: Optional[float]
    vtyp: Optional[float]
    vmax: Optional[float]
    unit: str
