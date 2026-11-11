"""Layer 3: extractors (rules, naive regex, learned tagger), row assembly, and extraction scoring."""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

from .metrics import prf
from .rules import naive_line, parse_line, parse_number, repair_unit
from .schema import Doc, Extracted, Row
from .tagger import SeqTagger
from .textnorm import clean_tok, edit_distance
from .units import close, is_unit, parse_unit, to_si


def sym_clean(t: str) -> str:
    t = t.rstrip(":,")
    return t[1:-1] if t.startswith("(") and t.endswith(")") else t


def assemble(toks: List[str], labels: List[str], repair: bool = True) -> Optional[Extracted]:
    """Turn a labelled token sequence into a specification row (None if no value+unit found)."""
    name, sym, cond = [], [], []
    vals: Dict[str, Optional[float]] = {"MIN": None, "TYP": None, "MAX": None}
    unit: Optional[str] = None
    for t, l in zip(toks, labels):
        if l == "NAME":
            name.append(t)
        elif l == "SYM":
            sym.append(sym_clean(t))
        elif l == "COND":
            cond.append(t.rstrip(":,"))
        elif l in vals and vals[l] is None:
            vals[l] = parse_number(t, repair)
        elif l == "UNIT" and unit is None:
            u = repair_unit(clean_tok(t)) if repair else clean_tok(t)
            if is_unit(u):
                unit = u
    if unit is None or all(v is None for v in vals.values()):
        return None
    c = " ".join(cond)
    return Extracted(" ".join(name), "".join(sym), "" if c == "-" else c, vals["MIN"], vals["TYP"], vals["MAX"], unit)


class RuleExtractor:
    def __init__(self, repair: bool = False):
        self.repair = repair

    def extract(self, text: str) -> Optional[Extracted]:
        return parse_line(text, self.repair)


class NaiveExtractor:
    def extract(self, text: str) -> Optional[Extracted]:
        return naive_line(text)


class TaggerExtractor:
    def __init__(self, tagger: SeqTagger, repair: bool = True):
        self.tagger, self.repair = tagger, repair

    def extract(self, text: str) -> Optional[Extracted]:
        toks, labs = self.tagger.predict(text)
        return assemble(toks, labs, self.repair)


def parse_cond(cond: str) -> Optional[Tuple[float, str]]:
    """'IOUT = 100 mA' -> (0.1, 'A'); None if absent or unparsable."""
    m = re.search(r"=\s*(\S+)\s+(\S+)", cond)
    if not m:
        return None
    v = parse_number(m.group(1), True)
    u = repair_unit(m.group(2).rstrip("?.,;:)"))
    if v is None or not is_unit(u):
        return None
    return to_si(v, u)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def values_match(e: Extracted, g: Row) -> bool:
    try:
        sc_e, d_e = parse_unit(e.unit)
        sc_g, d_g = parse_unit(g.unit)
    except ValueError:
        return False
    if d_e != d_g:
        return False
    for a, b in ((e.vmin, g.vmin), (e.vtyp, g.vtyp), (e.vmax, g.vmax)):
        if (a is None) != (b is None):
            return False
        if a is not None and not close(a * sc_e, b * sc_g, 1e-3):
            return False
    return True


def name_match(e: Extracted, g: Row, tol: int = 2) -> bool:
    return edit_distance(_norm(e.name), _norm(g.name)) <= tol


def eval_extraction(extract: Callable[[str], Optional[Extracted]], docs: List[Doc], keep_errors: bool = False) -> Dict:
    tp = fp = fn = 0
    det_tp = det_fp = det_fn = 0
    val_ok = name_ok = n_det_gold = 0
    errors: List[Dict] = []
    for d in docs:
        for i, ln in enumerate(d.lines):
            e = extract(ln.text)
            g = ln.row
            if g is not None and e is not None:
                det_tp += 1
                v, nm = values_match(e, g), name_match(e, g)
                n_det_gold += 1
                val_ok += v
                name_ok += nm
                if v and nm:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
                    if keep_errors:
                        errors.append({"kind": "wrong_fields", "part": d.part, "text": ln.text, "values_ok": v, "name_ok": nm})
            elif g is not None:
                det_fn += 1
                fn += 1
                if keep_errors:
                    errors.append({"kind": "missed_row", "part": d.part, "text": ln.text})
            elif e is not None:
                det_fp += 1
                fp += 1
                if keep_errors:
                    errors.append({"kind": "false_positive", "part": d.part, "text": ln.text})
    out = {"row": prf(tp, fp, fn), "detect": prf(det_tp, det_fp, det_fn),
           "value_acc": val_ok / max(n_det_gold, 1), "name_acc": name_ok / max(n_det_gold, 1),
           "tp": tp, "fp": fp, "fn": fn}
    if keep_errors:
        out["errors"] = errors
    return out
