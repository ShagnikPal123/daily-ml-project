"""Layer 4: retrieve -> extract -> select pipeline that answers spec questions with a value and unit."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .chunker import Chunk
from .datasheetgen import Question
from .extract import parse_cond, values_match
from .retrieval import Retriever, Thesaurus, rank_of
from .schema import Doc, Extracted
from .textnorm import words
from .units import close, to_si

_STAT = (("min", re.compile(r"\bmin(imum)?\b", re.I)), ("typ", re.compile(r"\btyp(ical)?\b", re.I)), ("max", re.compile(r"\bmax(imum)?\b", re.I)))


def parse_question(text: str) -> Tuple[str, Optional[Tuple[float, str]]]:
    """Return the requested statistic ('typ' by default) and the SI test-condition value if present."""
    stat = next((s for s, rx in _STAT if rx.search(text)), "typ")
    return stat, parse_cond(text)


@dataclass
class Answer:
    value_si: Optional[float]
    dim: str = ""
    row: Optional[Extracted] = None
    chunk_id: Optional[int] = None
    retrieved: Optional[List[int]] = None


class QAPipeline:
    def __init__(self, chunks: List[Chunk], retriever: Retriever, extractor, topk: int = 5):
        self.chunks, self.retriever, self.extractor, self.topk = chunks, retriever, extractor, topk
        self.thes = Thesaurus()
        self._cache: Dict[int, List[Extracted]] = {}

    def rows_in_chunk(self, cid: int) -> List[Extracted]:
        if cid not in self._cache:
            rows = []
            for ln in self.chunks[cid].text.split("\n"):
                if ln.startswith("["):
                    continue
                e = self.extractor.extract(ln)
                if e is not None:
                    rows.append(e)
            self._cache[cid] = rows
        return self._cache[cid]

    def _row_score(self, qw: Dict[str, float], stat: str, qcond, e: Extracted, rank: int) -> float:
        etoks = set(words(e.name)) | set(words(e.symbol, drop_stop=False))
        if not etoks:
            return -9.0
        s = sum(w for t, w in qw.items() if t in etoks) / len(etoks)
        val = {"min": e.vmin, "typ": e.vtyp, "max": e.vmax}[stat]
        if val is None:
            s -= 1.0
        ec = parse_cond(e.cond) if e.cond else None
        if qcond is not None:
            if ec is not None and ec[1] == qcond[1] and close(ec[0], qcond[0], 1e-3):
                s += 1.0
        elif ec is not None:
            s -= 0.01
        return s - 0.01 * rank

    def answer(self, text: str, order=None) -> Answer:
        order = self.retriever.rank(text) if order is None else order
        top = [int(c) for c in order[: self.topk]]
        stat, qcond = parse_question(text)
        qw = self.thes.expand(words(text))
        best, best_s = None, -1e9
        for rank, cid in enumerate(top):
            for e in self.rows_in_chunk(cid):
                s = self._row_score(qw, stat, qcond, e, rank)
                if s > best_s and {"min": e.vmin, "typ": e.vtyp, "max": e.vmax}[stat] is not None:
                    best, best_s = (e, cid), s
        if best is None:
            return Answer(None, retrieved=top)
        e, cid = best
        v = {"min": e.vmin, "typ": e.vtyp, "max": e.vmax}[stat]
        try:
            si, dim = to_si(v, e.unit)
        except ValueError:
            return Answer(None, row=e, chunk_id=cid, retrieved=top)
        return Answer(si, dim, e, cid, top)


def is_correct(a: Answer, q: Question) -> bool:
    return a.value_si is not None and a.dim == q.dim and close(a.value_si, q.answer_si, 1e-3)


def evaluate_qa(pipe: QAPipeline, questions: Sequence[Question], docs: List[Doc], l2c: Dict[Tuple[str, int], int], keep: bool = False) -> Dict:
    """End-to-end accuracy plus a stage breakdown: retrieval_miss / extract_error / wrong_row / correct."""
    by = {d.part: d for d in docs}
    cats = {"correct": 0, "retrieval_miss": 0, "extract_error": 0, "wrong_row": 0}
    ranks, examples = [], []
    for q in questions:
        order = pipe.retriever.rank(q.text)
        gold = l2c[q.line_uid]
        ranks.append(rank_of(order, gold))
        a = pipe.answer(q.text, order)
        if is_correct(a, q):
            cat = "correct"
        elif gold not in a.retrieved:
            cat = "retrieval_miss"
        else:
            gl = by[q.part].lines[q.line_uid[1]]
            e = pipe.extractor.extract(gl.text)
            cat = "extract_error" if e is None or not values_match(e, gl.row) else "wrong_row"
        cats[cat] += 1
        if keep and cat != "correct" and len(examples) < 60:
            examples.append({"cat": cat, "question": q.text, "expected_si": q.answer_si, "got_si": a.value_si})
    n = max(len(questions), 1)
    out = {"acc": cats["correct"] / n, "hit@1": sum(1 for r in ranks if r == 1) / n,
           "hit@k": sum(1 for r in ranks if r is not None and r <= pipe.topk) / n,
           "cats": cats, "ranks": ranks}
    if keep:
        out["examples"] = examples
    return out
