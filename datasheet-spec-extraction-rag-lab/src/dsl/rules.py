"""Layer 2: hand-written layout parsers and a naive regex baseline for specification lines."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .schema import Extracted
from .units import is_unit

_NUM = re.compile(r"^-?\d+(\.\d+)?$")
_REP = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "b": "6"})
_INLINE = re.compile(r"^(?P<name>.+?) \((?P<sym>.+?)\)(?: at (?P<cond>.+?))?: (?P<rest>(?:min|typ|max) .+)$")
_ITEM = re.compile(r"(min|typ|max) (\S+) (\S+)")


def parse_number(tok: str, repair: bool = False) -> Optional[float]:
    """Parse a decimal; with ``repair`` undo common OCR digit look-alikes (O->0, l->1, S->5...)."""
    t = tok.strip("(),;:")
    if _NUM.match(t):
        return float(t)
    if repair and any(c.isdigit() for c in t):
        r = t.translate(_REP)
        if _NUM.match(r):
            return float(r)
    return None


def repair_unit(u: str) -> str:
    u = u.strip(",;:")
    return u if is_unit(u) else u.replace("rn", "m")


def parse_table_line(line: str, repair: bool = False) -> Optional[Extracted]:
    """Seven-column table row (pipe- or multi-space-separated). Fails if the column count differs."""
    cells = [c.strip() for c in (line.split("|") if "|" in line else re.split(r"\s{2,}", line.strip()))]
    if len(cells) != 7:
        return None
    name, sym, cond, a, b, c, unit = cells
    vals: List[Optional[float]] = []
    for t in (a, b, c):
        if t == "-":
            vals.append(None)
            continue
        v = parse_number(t, repair)
        if v is None:
            return None
        vals.append(v)
    u = repair_unit(unit) if repair else unit
    if all(v is None for v in vals) or not is_unit(u):
        return None
    return Extracted(name, sym, "" if cond == "-" else cond, vals[0], vals[1], vals[2], u)


def parse_inline_line(line: str, repair: bool = False) -> Optional[Extracted]:
    m = _INLINE.match(line.strip())
    if not m:
        return None
    vals = {"min": None, "typ": None, "max": None}
    unit = ""
    for kw, num, u in _ITEM.findall(m.group("rest")):
        v = parse_number(num, repair)
        if v is None:
            return None
        vals[kw] = v
        unit = unit or (repair_unit(u) if repair else u.strip(",;:"))
    if not is_unit(unit):
        return None
    return Extracted(m.group("name"), m.group("sym"), m.group("cond") or "", vals["min"], vals["typ"], vals["max"], unit)


def parse_line(line: str, repair: bool = False) -> Optional[Extracted]:
    """Try the table parser first, then the inline parser."""
    return parse_table_line(line, repair) or parse_inline_line(line, repair)


def naive_line(line: str) -> Optional[Extracted]:
    """Baseline: every 'number unit' pair is a value; leading words are the name."""
    toks = line.replace("|", " ").split()
    pairs: List[Tuple[float, str]] = []
    for i in range(1, len(toks)):
        u = toks[i].strip(",;:")
        n = parse_number(toks[i - 1])
        if n is not None and is_unit(u):
            pairs.append((n, u))
    if not pairs:
        return None
    name_words: List[str] = []
    for t in toks:
        if re.fullmatch(r"[A-Za-z][A-Za-z\-]*", t) and len(name_words) < 4:
            name_words.append(t)
        else:
            break
    nums = [p[0] for p in pairs]
    if len(nums) >= 3:
        lo, ty, hi = nums[0], nums[1], nums[2]
    elif len(nums) == 2:
        lo, ty, hi = None, nums[0], nums[1]
    else:
        lo, ty, hi = None, nums[0], None
    return Extracted(" ".join(name_words), "", "", lo, ty, hi, pairs[-1][1])
