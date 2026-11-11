"""Layer 0: text normalisation, tokenisation, light stemming, edit distance, token shapes."""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

SEP = "¦"  # pseudo-token marking a column gap / pipe in a table line

_TRANS = str.maketrans({"µ": "u", "μ": "u", "−": "-", "–": "-", "—": "-", "‑": "-", "Ω": "ohm", "℃": "degC"})
STOP = frozenset("the a an of is what for at in to and or with on by from as are be its this that which give".split())


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_TRANS)
    return t.replace("°C", "degC")


_TOK = re.compile(r"\|| {2,}|[^\s|]+")


def tokenize_spans(line: str) -> List[Tuple[str, int, int]]:
    """Whitespace tokens of a table line, with column gaps (pipes or >=2 spaces) as SEP tokens."""
    out: List[Tuple[str, int, int]] = []
    for m in _TOK.finditer(line):
        t = m.group(0)
        if t == "|" or t.startswith("  "):
            t = SEP
        if t == SEP and out and out[-1][0] == SEP:
            out[-1] = (SEP, out[-1][1], m.end())
        else:
            out.append((t, m.start(), m.end()))
    return out


_WORD = re.compile(r"[a-z0-9]+(?:\.\d+)?")


def stem(w: str) -> str:
    if not w or w[0].isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def words(text: str, drop_stop: bool = True, do_stem: bool = True) -> List[str]:
    """Lower-cased alphanumeric tokens (part numbers and decimals stay whole)."""
    toks = _WORD.findall(normalize(text).lower())
    if drop_stop:
        toks = [t for t in toks if t not in STOP]
    return [stem(t) for t in toks] if do_stem else toks


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert/delete/substitute, unit cost)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def shape(tok: str) -> str:
    """Collapsed character-class shape: 'IOUT' -> 'X', '1.5' -> 'd.d', 'mA' -> 'xX'."""
    out = []
    for ch in tok:
        c = "d" if ch.isdigit() else "X" if ch.isupper() else "x" if ch.islower() else ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def clean_tok(tok: str) -> str:
    """Strip surrounding brackets and trailing punctuation from a token."""
    return tok.strip("(),;:[]")
