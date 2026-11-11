"""Layer 1: split documents into retrievable passages (chunks) with optional context header."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .datasheetgen import SECTION_TITLES
from .schema import Doc


@dataclass
class Chunk:
    cid: int
    part: str
    section: str
    text: str
    line_idxs: List[int]


def chunk_docs(docs: List[Doc], size: int = 4, header: bool = True) -> List[Chunk]:
    """Windows of ``size`` consecutive same-section lines; ``header`` prepends '[part | section]'."""
    chunks: List[Chunk] = []
    for d in docs:
        i = 0
        while i < len(d.lines):
            sec = d.lines[i].section
            j = i
            while j < len(d.lines) and d.lines[j].section == sec:
                j += 1
            for s in range(i, j, size):
                idxs = list(range(s, min(s + size, j)))
                body = "\n".join(d.lines[k].text for k in idxs)
                text = (f"[{d.part} | {SECTION_TITLES[sec]}]\n" if header else "") + body
                chunks.append(Chunk(len(chunks), d.part, sec, text, idxs))
            i = j
    return chunks


def line_to_chunk(chunks: List[Chunk]) -> Dict[Tuple[str, int], int]:
    return {(c.part, k): c.cid for c in chunks for k in c.line_idxs}
