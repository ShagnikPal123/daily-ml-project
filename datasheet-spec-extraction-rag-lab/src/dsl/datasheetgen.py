"""Layer 1: synthetic datasheet generator with token-level gold labels and OCR-style noise.

Each document renders a part's pin list, absolute-maximum-ratings table and electrical-
characteristics table in one of three layouts (``pipe``, ``space`` and ``inline``).  Gold
token labels are derived from character spans recorded while rendering, so they stay
aligned with the text even after noise is applied to individual cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .catalog import ABS_ROWS, DESCRIPTIONS, FAMILIES, ParamDef
from .schema import Doc, Line, Question, Row
from .textnorm import SEP, tokenize_spans
from .units import choose_unit, fmt_num, parse_unit, round_sig, to_si

STYLES = ("pipe", "space", "inline")
_CONF = {"0": "O", "1": "l", "5": "S", "8": "B", "6": "b"}
SECTION_TITLES = {"title": "Overview", "feat": "Features", "pin": "Pin Configuration",
                  "abs": "Absolute Maximum Ratings", "ec": "Electrical Characteristics", "app": "Applications"}
STAT_WORD = {"min": "minimum", "typ": "typical", "max": "maximum"}


@dataclass
class GenConfig:
    n_parts: int = 40
    styles: Tuple[str, ...] = STYLES
    noise: float = 0.0  # per-character OCR confusion probability
    sep_damage: float = 0.0  # probability that a column gap collapses to a single space
    seed: int = 0


class _Builder:
    def __init__(self):
        self.text = ""
        self.segs: List[Tuple[int, int, str]] = []

    def add(self, s: str, label: str):
        a = len(self.text)
        self.text += s
        self.segs.append((a, len(self.text), label))

    def label_at(self, pos: int) -> str:
        for a, b, l in self.segs:
            if a <= pos < b:
                return l
        return "O"

    def tags(self) -> List[Tuple[str, str]]:
        out = []
        for tok, s, _ in tokenize_spans(self.text):
            out.append((tok, "SEP" if tok == SEP else self.label_at(s)))
        return out


def _ocr(text: str, label: str, rng, p: float) -> str:
    if p <= 0:
        return text
    out = []
    for ch in text:
        if rng.random() < p:
            if ch in _CONF:
                ch = _CONF[ch]
            elif ch == "m" and label in ("NAME", "UNIT", "COND"):
                ch = "rn"
        out.append(ch)
    return "".join(out)


def _num(v: Optional[float]) -> str:
    return "-" if v is None else fmt_num(v)


def render_row(row: Row, style: str, rng, cfg: GenConfig) -> Tuple[str, List[Tuple[str, str]]]:
    b = _Builder()
    p = cfg.noise
    cells = [(row.name, "NAME"), (row.symbol, "SYM"), (row.cond, "COND"), (_num(row.vmin), "MIN"),
             (_num(row.vtyp), "TYP"), (_num(row.vmax), "MAX"), (row.unit, "UNIT")]
    if style == "inline":
        b.add(_ocr(row.name, "NAME", rng, p), "NAME")
        b.add(" ", "O")
        b.add("(" + _ocr(row.symbol, "SYM", rng, p) + ")", "SYM")
        if row.cond != "-":
            b.add(" at ", "O")
            b.add(_ocr(row.cond, "COND", rng, p), "COND")
        b.add(": ", "O")
        first = True
        for val, lab, kw in ((row.vmin, "MIN", "min"), (row.vtyp, "TYP", "typ"), (row.vmax, "MAX", "max")):
            if val is None:
                continue
            if not first:
                b.add(", ", "O")
            first = False
            b.add(kw + " ", "O")
            b.add(_ocr(fmt_num(val), lab, rng, p), lab)
            b.add(" ", "O")
            b.add(_ocr(row.unit, "UNIT", rng, p), "UNIT")
        return b.text, b.tags()
    gap = " | " if style == "pipe" else "   "
    for i, (txt, lab) in enumerate(cells):
        if i:
            g = " " if rng.random() < cfg.sep_damage else gap
            b.add(g, "SEP")
        b.add(_ocr(txt, lab, rng, p), lab)
    return b.text, b.tags()


def _plain(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    return text, [(t, "SEP" if t == SEP else "O") for t, _, _ in tokenize_spans(text)]


def _values(rng, pd: ParamDef, typ_si: float, pat: Sequence[str]) -> Dict[str, Optional[float]]:
    if pd.choices:
        vals = {"min": typ_si, "typ": typ_si, "max": typ_si}
    else:
        vals = {"min": typ_si * rng.uniform(0.7, 0.95), "typ": typ_si, "max": typ_si * rng.uniform(1.08, 1.6)}
    return {k: (vals[k] if k in pat else None) for k in ("min", "typ", "max")}


def _make_row(rng, part: str, pd: ParamDef, name: str, typ_si: float, pat, cond: Optional[Tuple[str, str, float]]) -> Row:
    vals = _values(rng, pd, typ_si, pat)
    present = [abs(v) for v in vals.values() if v is not None]
    unit = choose_unit(max(present), pd.units)
    scale, _ = parse_unit(unit)
    disp = {k: (None if v is None else round_sig(v / scale, 3)) for k, v in vals.items()}
    if cond:
        csym, cunit, x = cond
        ctext, cval = f"{csym} = {fmt_num(x)} {cunit}", x
    else:
        ctext, cval, cunit = "-", None, ""
    return Row(part, pd.pid, name, pd.symbol, ctext, disp["min"], disp["typ"], disp["max"], unit, "ec", cval, cunit)


def _abs_rows(rng, part: str) -> List[Row]:
    rows = []
    for pid, name, sym, unit, lo, hi, which in ABS_ROWS:
        if which == "minmax":
            rows.append(Row(part, pid, name, sym, "-", float(rng.choice([-65, -55, -40])), None, float(rng.choice([125, 150, 165])), unit, "abs"))
        else:
            rows.append(Row(part, pid, name, sym, "-", None, None, round_sig(float(rng.uniform(lo, hi)), 3), unit, "abs"))
    return rows


def make_doc(rng, family: str, used: set, cfg: GenConfig, style: str, nrng=None) -> Doc:
    nrng = nrng if nrng is not None else np.random.default_rng(0)
    fam = FAMILIES[family]
    while True:
        part = f"{rng.choice(fam['prefixes'])}{int(rng.integers(100, 9999))}{rng.choice(['', 'A', 'B'])}"
        if part not in used:
            used.add(part)
            break
    params: Tuple[ParamDef, ...] = fam["params"]
    names = {pd.pid: str(rng.choice(pd.names)) for pd in params}
    ec: List[Row] = []
    for pd in params:
        typ0 = float(np.exp(rng.uniform(math.log(pd.lo), math.log(pd.hi)))) if not pd.choices else float(rng.choice(pd.choices))
        pat = pd.pats[int(rng.integers(len(pd.pats)))]
        if pd.cond:
            csym, cunit, cvals, alpha = pd.cond
            for ci in sorted(rng.choice(len(cvals), size=2, replace=False)):
                x = cvals[int(ci)]
                ec.append(_make_row(rng, part, pd, names[pd.pid], typ0 * (x / cvals[0]) ** alpha, pat, (csym, cunit, x)))
        else:
            ec.append(_make_row(rng, part, pd, names[pd.pid], typ0, pat, None))
    lines: List[Line] = []

    def plain(text, section):
        t, tg = _plain(text)
        lines.append(Line(t, section, None, tg))

    plain(f"{part} {fam['title']}", "title")
    plain(DESCRIPTIONS[family], "title")
    plain("FEATURES", "feat")
    for pd in [params[i] for i in rng.choice(len(params), size=3, replace=False)]:
        nm = names[pd.pid].lower()
        if pd.aliases:
            plain(f"- Low {nm} ({rng.choice(pd.aliases)}) for demanding designs", "feat")
        else:
            plain(f"- Specified {nm} ({pd.symbol}) over temperature", "feat")
    plain("PIN CONFIGURATION", "pin")
    for k, nm in enumerate(["VIN", "GND", "OUT"], 1):
        plain(f"{k}   {nm}   Power pin, bypass with {rng.choice(['1', '2.2', '4.7', '10'])} uF capacitor", "pin")
    header = {"pipe": "Parameter | Symbol | Conditions | Min | Typ | Max | Unit",
              "space": "Parameter   Symbol   Conditions   Min   Typ   Max   Unit"}.get(style)
    for section, title, rows in (("abs", "ABSOLUTE MAXIMUM RATINGS", _abs_rows(rng, part)), ("ec", "ELECTRICAL CHARACTERISTICS", ec)):
        plain(title, section)
        if header:
            plain(header, section)
        for r in rows:
            t, tg = render_row(r, style, nrng, cfg)
            lines.append(Line(t, section, r, tg))
    plain(f"Specifications at TA = 25 degC unless otherwise noted. VIN = {rng.choice(['3.3', '5', '12'])} V.", "ec")
    plain(f"Typical applications include sensor nodes operating from {rng.choice(['1.8', '2.5', '3.0'])} V to {rng.choice(['5.5', '12', '36'])} V.", "app")
    return Doc(part, family, style, lines)


def make_corpus(cfg: GenConfig) -> List[Doc]:
    rng = np.random.default_rng(cfg.seed)
    fams = list(FAMILIES)
    nrng = np.random.default_rng(cfg.seed + 7919)
    used: set = set()
    docs = []
    for i in range(cfg.n_parts):
        style = str(rng.choice(cfg.styles))
        docs.append(make_doc(rng, fams[i % len(fams)], used, cfg, style, nrng))
    return docs


def _typo(s: str, rng) -> str:
    if len(s) < 5:
        return s
    i = int(rng.integers(1, len(s) - 1))
    return s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i + 1] + s[i] + s[i + 2:]


def make_questions(docs: List[Doc], rng, n: int, paraphrase: float = 0.5, typo: float = 0.15) -> List[Question]:
    pdefs = {(fam, pd.pid): pd for fam, f in FAMILIES.items() for pd in f["params"]}
    cands = [(d, i) for d in docs for i, ln in enumerate(d.lines) if ln.row is not None and ln.row.section == "ec"]
    order = rng.permutation(len(cands))
    qs: List[Question] = []
    for j in range(n):
        d, i = cands[int(order[j % len(cands)])]
        r = d.lines[i].row
        stats = [s for s, v in (("min", r.vmin), ("typ", r.vtyp), ("max", r.vmax)) if v is not None]
        stat = str(rng.choice(stats))
        val = {"min": r.vmin, "typ": r.vtyp, "max": r.vmax}[stat]
        pd = pdefs[(d.family, r.pid)]
        para = bool(pd.aliases) and rng.random() < paraphrase
        pname = str(rng.choice(pd.aliases)) if para else r.name
        cp = f" at {r.cond}" if r.cond_value is not None else ""
        part = _typo(d.part, rng) if rng.random() < typo else d.part
        sw = STAT_WORD[stat]
        t = int(rng.integers(3))
        if t == 0:
            text = f"What is the {sw} {pname} of the {part}{cp}?"
        elif t == 1:
            text = f"{part} {sw} {pname}{cp}"
        else:
            text = f"Give the {sw} {r.symbol} for {part}{cp}"
            para = False
        ans, dim = to_si(val, r.unit)
        csi = to_si(r.cond_value, r.cond_unit)[0] if r.cond_value is not None else None
        qs.append(Question(text, d.part, r.pid, stat, ans, dim, csi, para, (d.part, i)))
    return qs
