"""Layer 5 (apex): command-line interface -- run experiments, generate sample datasheets, ask questions."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import experiments as X
from .chunker import chunk_docs
from .datasheetgen import STYLES, GenConfig, make_corpus
from .fixedpoint import QuantTagger, float_model_bytes
from .extract import RuleExtractor, TaggerExtractor
from .qa import QAPipeline
from .retrieval import Retriever
from .tagger import SeqTagger


def _tagger(seed: int, n_train: int = 24, epochs: int = 4) -> SeqTagger:
    docs = make_corpus(GenConfig(n_parts=n_train, noise=0.02, sep_damage=0.05, seed=1000 + seed))
    return SeqTagger(epochs=epochs, seed=seed).fit([ln.tags for d in docs for ln in d.lines])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dsl", description="Datasheet spec extraction + retrieval lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run all experiments and write results/")
    r.add_argument("--config", default=None)
    r.add_argument("--out", default="results")
    r.add_argument("--quick", action="store_true", help="tiny smoke configuration")
    g = sub.add_parser("generate", help="print a synthetic datasheet")
    g.add_argument("--style", choices=STYLES, default="pipe")
    g.add_argument("--noise", type=float, default=0.0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--index", type=int, default=0)
    e = sub.add_parser("extract", help="parse one table line with rules and the learned tagger")
    e.add_argument("text")
    e.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("quantize", help="token accuracy and model size of the fixed-point tagger")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3])
    q = sub.add_parser("ask", help="answer a question over a freshly generated corpus")
    q.add_argument("question")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--parts", type=int, default=24)
    q.add_argument("--noise", type=float, default=0.02)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "run":
        cfg = X.load_config(a.config, a.quick)
        R = X.run_all(cfg, a.out)
        print(f"done in {R['runtime_s']:.0f}s -> {a.out}/results.json, tables.md, {len(R['figures'])} figures")
    elif a.cmd == "generate":
        docs = make_corpus(GenConfig(n_parts=a.index + 1, styles=(a.style,), noise=a.noise, sep_damage=X.sd_for(a.noise), seed=a.seed))
        print("\n".join(ln.text for ln in docs[a.index].lines))
    elif a.cmd == "extract":
        print("rules :", RuleExtractor(True).extract(a.text))
        print("tagger:", TaggerExtractor(_tagger(a.seed)).extract(a.text))
    elif a.cmd == "quantize":
        tg = _tagger(a.seed)
        test = make_corpus(GenConfig(n_parts=12, noise=0.02, sep_damage=0.05, seed=77))
        lines = [ln.tags for d in test for ln in d.lines]
        print(f"float: token acc {tg.token_accuracy(lines):.4f}, {float_model_bytes(tg):.0f} bytes")
        for b in a.bits:
            qt = QuantTagger(tg, b)
            print(f"int{b}: token acc {qt.token_accuracy(lines):.4f}, dense {qt.dense_bytes():.0f} B, sparse {qt.sparse_bytes():.0f} B")
    elif a.cmd == "ask":
        docs = make_corpus(GenConfig(n_parts=a.parts, noise=a.noise, sep_damage=X.sd_for(a.noise), seed=a.seed))
        ch = chunk_docs(docs)
        r = Retriever("hybrid", expand=True, part_filter=True).fit([c.text for c in ch], [c.part for c in ch])
        ans = QAPipeline(ch, r, TaggerExtractor(_tagger(a.seed))).answer(a.question)
        print("answer (SI):", ans.value_si, ans.dim, "| row:", ans.row)
        if ans.chunk_id is not None:
            print("from passage:\n" + ch[ans.chunk_id].text)
        else:
            print("example part numbers:", ", ".join(d.part for d in docs[:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
