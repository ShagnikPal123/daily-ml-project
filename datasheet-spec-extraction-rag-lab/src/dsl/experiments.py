"""Layer 5 (apex): config-driven experiment runner -- benchmarks, ablations, sweeps, failures, figures."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .chunker import chunk_docs, line_to_chunk
from .datasheetgen import STYLES, GenConfig, make_corpus, make_questions
from .confusion import accuracy, confusion_matrix, per_label_prf, top_confusions
from .fixedpoint import QuantTagger, float_model_bytes
from .extract import NaiveExtractor, RuleExtractor, TaggerExtractor, eval_extraction
from .indexcompress import eval_compressed_lsa, sparse_index_bytes
from .metrics import fmt_ci, mean_ci, mrr, recall_at_k
from .qa import QAPipeline, evaluate_qa
from .retrieval import Retriever, rank_of
from .tagger import GROUPS, SeqTagger
from .textnorm import words

DEFAULT = {
    "seeds": [0, 1, 2, 3, 4], "n_train": 24, "n_test": 24, "n_questions": 150, "epochs": 4,
    "train_noise": 0.02, "noise_levels": [0.0, 0.01, 0.02, 0.04, 0.08], "ablation_noise": 0.02,
    "qa_noise_levels": [0.0, 0.02, 0.04, 0.08], "lsa_k": 64, "chunk_size": 4, "topk": 5,
    "lsa_k_sweep": [8, 16, 32, 64, 128], "chunk_sizes": [2, 4, 8], "bits": [32, 8, 4, 2], "quant_bits": [8, 6, 4, 3, 2], "profile_lines": 300, "profile_queries": 60,
}
QUICK = {"seeds": [0, 1], "n_train": 8, "n_test": 8, "n_questions": 30, "epochs": 2, "noise_levels": [0.0, 0.04],
         "qa_noise_levels": [0.0, 0.04], "lsa_k_sweep": [8, 32], "chunk_sizes": [2, 4], "bits": [32, 4], "quant_bits": [8, 3], "profile_lines": 40, "profile_queries": 10}

RETRIEVERS = {
    "tfidf": dict(kind="tfidf"), "bm25": dict(kind="bm25"), "lsa": dict(kind="lsa"),
    "bm25+thesaurus": dict(kind="bm25", expand=True), "bm25+part-filter": dict(kind="bm25", part_filter=True),
    "hybrid(bm25,lsa)+thesaurus": dict(kind="hybrid", expand=True),
    "hybrid+thesaurus+part-filter": dict(kind="hybrid", expand=True, part_filter=True),
}
ABLATIONS = {
    "full": dict(), "no Viterbi (independent argmax)": dict(use_viterbi=False),
    "no context features": dict(groups=("lex", "shape", "pos")), "no position features": dict(groups=("lex", "shape", "ctx")),
    "no shape features": dict(groups=("lex", "ctx", "pos")), "no lexical features": dict(groups=("shape", "ctx", "pos")),
}


def load_config(path=None, quick=False) -> Dict:
    cfg = dict(DEFAULT)
    if path:
        cfg.update(json.loads(Path(path).read_text()))
    if quick:
        cfg.update(QUICK)
    return cfg


def sd_for(noise: float) -> float:
    return min(0.3, 2.5 * noise)


def corpus(seed, n, noise, styles=STYLES):
    return make_corpus(GenConfig(n_parts=n, noise=noise, sep_damage=sd_for(noise), seed=seed, styles=tuple(styles)))


def summ(vals) -> Dict:
    m, h = mean_ci(vals)
    return {"mean": m, "ci": h, "n": len(list(vals))}


class TaggerCache:
    def __init__(self, cfg):
        self.cfg, self.store = cfg, {}

    def get(self, seed, groups=GROUPS, use_viterbi=True, styles=STYLES) -> SeqTagger:
        key = (seed, tuple(groups), use_viterbi, tuple(styles))
        if key not in self.store:
            c = self.cfg
            docs = corpus(1000 + seed, c["n_train"], c["train_noise"], styles)
            self.store[key] = SeqTagger(groups, use_viterbi, epochs=c["epochs"], seed=seed).fit([ln.tags for d in docs for ln in d.lines])
        return self.store[key]


def tag_lines(docs):
    return [ln.tags for d in docs for ln in d.lines]


# ------------------------------------------------------------------ experiments
def exp_extraction(cfg, cache) -> Dict:
    res: Dict = defaultdict(lambda: defaultdict(list))
    for s in cfg["seeds"]:
        tg = cache.get(s)
        ex = {"naive-regex": NaiveExtractor(), "rules": RuleExtractor(False), "rules+ocr-repair": RuleExtractor(True), "tagger (ours)": TaggerExtractor(tg)}
        for li, lv in enumerate(cfg["noise_levels"]):
            test = corpus(2000 + 10 * s + li, cfg["n_test"], lv)
            for name, e in ex.items():
                res[name][lv].append(eval_extraction(e.extract, test)["row"]["f1"])
    return {m: {str(lv): summ(v) for lv, v in d.items()} for m, d in res.items()}


def exp_ablation(cfg, cache) -> Dict:
    res: Dict = defaultdict(lambda: defaultdict(list))
    lv = cfg["ablation_noise"]
    for s in cfg["seeds"]:
        test = corpus(2500 + s, cfg["n_test"], lv)
        for name, kw in ABLATIONS.items():
            tg = cache.get(s, **kw)
            res[name]["f1"].append(eval_extraction(TaggerExtractor(tg).extract, test)["row"]["f1"])
            res[name]["tok"].append(tg.token_accuracy(tag_lines(test)))
        res["no OCR repair in assembly"]["f1"].append(eval_extraction(TaggerExtractor(cache.get(s), repair=False).extract, test)["row"]["f1"])
        res["no OCR repair in assembly"]["tok"].append(cache.get(s).token_accuracy(tag_lines(test)))
    return {m: {k: summ(v) for k, v in d.items()} for m, d in res.items()}


def exp_style_shift(cfg, cache) -> Dict:
    res: Dict = defaultdict(list)
    lv = cfg["ablation_noise"]
    for s in cfg["seeds"]:
        inline = corpus(2600 + s, cfg["n_test"], lv, ("inline",))
        tab = corpus(2700 + s, cfg["n_test"], lv, ("pipe", "space"))
        full, shifted = cache.get(s), cache.get(s, styles=("pipe", "space"))
        res["tagger trained on all layouts, tested on inline"].append(eval_extraction(TaggerExtractor(full).extract, inline)["row"]["f1"])
        res["tagger trained on pipe+space only, tested on inline"].append(eval_extraction(TaggerExtractor(shifted).extract, inline)["row"]["f1"])
        res["tagger trained on pipe+space only, tested on pipe+space"].append(eval_extraction(TaggerExtractor(shifted).extract, tab)["row"]["f1"])
        res["rules (parsers cover all layouts), tested on inline"].append(eval_extraction(RuleExtractor(True).extract, inline)["row"]["f1"])
    return {k: summ(v) for k, v in res.items()}


def _rank_stats(ranks) -> Dict:
    return {"recall@1": recall_at_k(ranks, 1), "recall@5": recall_at_k(ranks, 5), "mrr": mrr(ranks)}


def exp_retrieval(cfg) -> Dict:
    res: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # (header,para) -> method -> metric -> [per seed]
    for s in cfg["seeds"]:
        docs = corpus(3000 + s, cfg["n_test"], 0.02)
        qsets = {p: make_questions(docs, np.random.default_rng(4000 + s), cfg["n_questions"], paraphrase=p, typo=0.15) for p in (0.0, 0.5)}
        for header in (True, False):
            ch = chunk_docs(docs, cfg["chunk_size"], header)
            l2c = line_to_chunk(ch)
            texts, parts = [c.text for c in ch], [c.part for c in ch]
            fitted = {name: Retriever(lsa_k=cfg["lsa_k"], **kw).fit(texts, parts) for name, kw in RETRIEVERS.items()}
            rng = np.random.default_rng(s)
            for p, qs in qsets.items():
                key = f"header={header}|para={p}"
                for name, r in fitted.items():
                    ranks = [rank_of(r.rank(q.text), l2c[q.line_uid]) for q in qs]
                    for k, v in _rank_stats(ranks).items():
                        res[key][name][k].append(v)
                ranks = [int(rng.integers(1, len(ch) + 1)) for _ in qs]
                for k, v in _rank_stats(ranks).items():
                    res[key]["random"][k].append(v)
    return {key: {m: {k: summ(v) for k, v in d.items()} for m, d in methods.items()} for key, methods in res.items()}


def make_pipes(docs, tg, cfg):
    ch = chunk_docs(docs, cfg["chunk_size"])
    texts, parts = [c.text for c in ch], [c.part for c in ch]
    bm = Retriever("bm25").fit(texts, parts)
    hy = Retriever("hybrid", expand=True, part_filter=True, lsa_k=cfg["lsa_k"]).fit(texts, parts)
    k = cfg["topk"]
    return ch, {
        "bm25 + naive regex": QAPipeline(ch, bm, NaiveExtractor(), k),
        "bm25 + rules": QAPipeline(ch, bm, RuleExtractor(False), k),
        "hybrid-routed + rules": QAPipeline(ch, hy, RuleExtractor(False), k),
        "hybrid-routed + rules+repair": QAPipeline(ch, hy, RuleExtractor(True), k),
        "hybrid-routed + tagger (ours)": QAPipeline(ch, hy, TaggerExtractor(tg), k),
    }


def exp_qa(cfg, cache) -> Dict:
    acc: Dict = defaultdict(lambda: defaultdict(list))
    cats: Dict = defaultdict(lambda: defaultdict(int))
    examples: Dict = defaultdict(list)
    for s in cfg["seeds"]:
        tg = cache.get(s)
        for li, lv in enumerate(cfg["qa_noise_levels"]):
            docs = corpus(5000 + 10 * s + li, cfg["n_test"], lv)
            ch, pipes = make_pipes(docs, tg, cfg)
            l2c = line_to_chunk(ch)
            qs = make_questions(docs, np.random.default_rng(5500 + s), cfg["n_questions"], paraphrase=0.5, typo=0.15)
            for name, p in pipes.items():
                r = evaluate_qa(p, qs, docs, l2c, keep=name.endswith("(ours)"))
                acc[name][lv].append(r["acc"])
                if name.endswith("(ours)"):
                    for c, v in r["cats"].items():
                        cats[lv][c] += v
                    if s == cfg["seeds"][0]:
                        examples[str(lv)] = r["examples"][:8]
    return {"acc": {m: {str(lv): summ(v) for lv, v in d.items()} for m, d in acc.items()},
            "cats": {str(lv): dict(c) for lv, c in cats.items()}, "examples": dict(examples)}


def exp_extraction_failures(cfg, cache) -> Dict:
    lv = max(cfg["noise_levels"])
    kinds: Dict = defaultdict(int)
    samples: Dict = defaultdict(list)
    for s in cfg["seeds"]:
        test = corpus(2000 + 10 * s + len(cfg["noise_levels"]) - 1, cfg["n_test"], lv)
        r = eval_extraction(TaggerExtractor(cache.get(s)).extract, test, keep_errors=True)
        for e in r["errors"]:
            kinds[e["kind"]] += 1
            if len(samples[e["kind"]]) < 4:
                samples[e["kind"]].append(e["text"][:110])
    return {"noise": lv, "counts": dict(kinds), "samples": dict(samples)}


def exp_sweeps(cfg, cache) -> Dict:
    lsa_k: Dict = defaultdict(lambda: defaultdict(list))
    chunk: Dict = defaultdict(lambda: defaultdict(list))
    comp: Dict = defaultdict(lambda: defaultdict(list))
    sparse_b: List[float] = []
    for s in cfg["seeds"]:
        docs = corpus(6000 + s, cfg["n_test"], 0.02)
        qs = make_questions(docs, np.random.default_rng(6500 + s), cfg["n_questions"], paraphrase=0.5, typo=0.15)
        ch = chunk_docs(docs, cfg["chunk_size"])
        l2c = line_to_chunk(ch)
        texts, parts = [c.text for c in ch], [c.part for c in ch]
        for k in cfg["lsa_k_sweep"]:
            for name, kw in (("lsa", dict(kind="lsa")), ("hybrid+thesaurus+part-filter", dict(kind="hybrid", expand=True, part_filter=True))):
                r = Retriever(lsa_k=k, **kw).fit(texts, parts)
                ranks = [rank_of(r.rank(q.text), l2c[q.line_uid]) for q in qs]
                lsa_k[name][k].append(recall_at_k(ranks, 5))
        for size in cfg["chunk_sizes"]:
            c2 = chunk_docs(docs, size)
            l2 = line_to_chunk(c2)
            hy = Retriever("hybrid", expand=True, part_filter=True, lsa_k=cfg["lsa_k"]).fit([c.text for c in c2], [c.part for c in c2])
            p = QAPipeline(c2, hy, TaggerExtractor(cache.get(s)), cfg["topk"])
            r = evaluate_qa(p, qs, docs, l2)
            chunk[size]["recall@1"].append(r["hit@1"])
            chunk[size]["recall@5"].append(r["hit@k"])
            chunk[size]["e2e"].append(r["acc"])
            chunk[size]["n_chunks"].append(len(c2))
        lsa_r = Retriever("lsa", expand=True, lsa_k=cfg["lsa_k"]).fit(texts, parts)
        out = eval_compressed_lsa(lsa_r.lsa, [lsa_r.weights(q.text) for q in qs], [l2c[q.line_uid] for q in qs], cfg["bits"])
        for o in out:
            for k in ("bytes", "recall@1", "recall@5", "mrr"):
                comp[o["bits"]][k].append(o[k])
        sparse_b.append(sparse_index_bytes(Retriever("bm25").fit(texts, parts).bm25.M))
    f = lambda d: {str(k): {str(m): summ(v) for m, v in dd.items()} for k, dd in d.items()}
    return {"lsa_k": f(lsa_k), "chunk": f(chunk), "compress": f(comp), "bm25_sparse_bytes": summ(sparse_b)}


def exp_fixedpoint(cfg, cache) -> Dict:
    res: Dict = defaultdict(lambda: defaultdict(list))
    for s in cfg["seeds"]:
        test = corpus(7000 + s, cfg["n_test"], cfg["ablation_noise"])
        tg = cache.get(s)
        res["float64 reference"]["f1"].append(eval_extraction(TaggerExtractor(tg).extract, test)["row"]["f1"])
        res["float64 reference"]["dense_bytes"].append(float_model_bytes(tg))
        res["float64 reference"]["sparse_bytes"].append(float_model_bytes(tg))
        for b in cfg["quant_bits"]:
            q = QuantTagger(tg, b)
            res[f"int{b}"]["f1"].append(eval_extraction(TaggerExtractor(q).extract, test)["row"]["f1"])
            res[f"int{b}"]["dense_bytes"].append(q.dense_bytes())
            res[f"int{b}"]["sparse_bytes"].append(q.sparse_bytes())
    return {m: {k: summ(v) for k, v in d.items()} for m, d in res.items()}


def exp_profile(cfg, cache) -> Dict:
    """Wall-clock cost per extracted line and per retrieval query (single seed, machine dependent)."""
    s = cfg["seeds"][0]
    docs = corpus(7500, cfg["n_test"], cfg["ablation_noise"])
    lines = [ln.text for d in docs for ln in d.lines][: cfg["profile_lines"]]
    tg = cache.get(s)
    out: Dict = {}
    for name, ex in (("naive-regex", NaiveExtractor()), ("rules", RuleExtractor(True)), ("tagger float", TaggerExtractor(tg)), ("tagger int8", TaggerExtractor(QuantTagger(tg, 8)))):
        t0 = time.perf_counter()
        for ln in lines:
            ex.extract(ln)
        out[name] = {"us_per_line": 1e6 * (time.perf_counter() - t0) / len(lines)}
    ch = chunk_docs(docs, cfg["chunk_size"])
    texts, parts = [c.text for c in ch], [c.part for c in ch]
    qs = [q.text for q in make_questions(docs, np.random.default_rng(7600), cfg["profile_queries"])]
    for name, kw in (("bm25", RETRIEVERS["bm25"]), ("lsa", RETRIEVERS["lsa"]), ("hybrid+thesaurus+part-filter", RETRIEVERS["hybrid+thesaurus+part-filter"])):
        r = Retriever(lsa_k=cfg["lsa_k"], **kw).fit(texts, parts)
        t0 = time.perf_counter()
        for q in qs:
            r.rank(q)
        out[name] = {"us_per_query": 1e6 * (time.perf_counter() - t0) / len(qs)}
    out["n_chunks"] = len(ch)
    return out


def exp_confusion(cfg, cache) -> Dict:
    """Token-level confusion of the tagger at the highest noise level, pooled over seeds."""
    lv = max(cfg["noise_levels"])
    cm = 0
    for s in cfg["seeds"]:
        test = corpus(7800 + s, cfg["n_test"], lv)
        cm = cm + confusion_matrix(cache.get(s), tag_lines(test))
    return {"noise": lv, "accuracy": accuracy(cm), "per_label": per_label_prf(cm), "top": top_confusions(cm, 6), "matrix": cm.tolist()}


# ------------------------------------------------------------------ reporting
def md_table(head, rows) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cell(d, digits=3):
    return fmt_ci(d["mean"], d["ci"], digits)


def build_tables(R: Dict, cfg: Dict) -> str:
    T = []
    lv = [str(x) for x in cfg["noise_levels"]]
    T.append("### T1. Row-level extraction F1 vs OCR noise level (mean ± 95% CI over seeds)\n\n" + md_table(
        ["method"] + [f"noise {x}" for x in lv], [[m] + [cell(d[x]) for x in lv] for m, d in R["extraction"].items()]))
    T.append("### T2. Tagger ablations (noise %s)\n\n" % cfg["ablation_noise"] + md_table(
        ["variant", "row F1", "token accuracy"], [[m, cell(d["f1"]), cell(d["tok"])] for m, d in R["ablation"].items()]))
    T.append("### T3. Layout shift (noise %s)\n\n" % cfg["ablation_noise"] + md_table(["condition", "row F1"], [[k, cell(v)] for k, v in R["style_shift"].items()]))
    for key in ("header=True|para=0.0", "header=True|para=0.5", "header=False|para=0.5"):
        T.append(f"### T4. Retrieval, {key}\n\n" + md_table(["method", "recall@1", "recall@5", "MRR"], [
            [m, cell(d["recall@1"]), cell(d["recall@5"]), cell(d["mrr"])] for m, d in R["retrieval"][key].items()]))
    qa = R["qa"]["acc"]
    ql = [str(x) for x in cfg["qa_noise_levels"]]
    T.append("### T5. End-to-end QA exact-match accuracy vs noise\n\n" + md_table(["pipeline"] + [f"noise {x}" for x in ql], [[m] + [cell(d[x]) for x in ql] for m, d in qa.items()]))
    cats = R["qa"]["cats"]
    T.append("### T6. Outcome breakdown for the headline pipeline (counts pooled over seeds)\n\n" + md_table(
        ["noise", "correct", "retrieval_miss", "extract_error", "wrong_row"], [[x] + [cats[x].get(c, 0) for c in ("correct", "retrieval_miss", "extract_error", "wrong_row")] for x in ql]))
    sw = R["sweeps"]
    ks = [str(k) for k in cfg["lsa_k_sweep"]]
    T.append("### T7. LSA dimension sweep, recall@5 (paraphrase 0.5)\n\n" + md_table(["method"] + [f"k={k}" for k in ks], [[m] + [cell(d[k]) for k in ks] for m, d in sw["lsa_k"].items()]))
    T.append("### T8. Chunk-size sweep\n\n" + md_table(["lines/chunk", "#chunks", "recall@1", "recall@5", "end-to-end acc"], [
        [k, f"{d['n_chunks']['mean']:.0f}", cell(d["recall@1"]), cell(d["recall@5"]), cell(d["e2e"])] for k, d in sw["chunk"].items()]))
    T.append("### T9. Index compression of LSA embeddings (k=%d)\n\n" % cfg["lsa_k"] + md_table(["bits/dim", "index bytes", "recall@1", "recall@5", "MRR"], [
        [k, f"{d['bytes']['mean']:.0f}", cell(d["recall@1"]), cell(d["recall@5"]), cell(d["mrr"])] for k, d in sw["compress"].items()])
        + f"\n\nBM25 sparse index (float32 values + int32 indices): {sw['bm25_sparse_bytes']['mean']:.0f} bytes on average.")
    T.append("### T10. Fixed-point tagger (noise %s): row F1 and model size\n\n" % cfg["ablation_noise"] + md_table(
        ["weights", "row F1", "dense bytes", "sparse bytes"], [[m, cell(d["f1"]), f"{d['dense_bytes']['mean']:.0f}", f"{d['sparse_bytes']['mean']:.0f}"] for m, d in R["fixedpoint"].items()]))
    pr = R["profile"]
    T.append("### T11. Profiling (seed %d, %d chunks; machine dependent)\n\n" % (cfg["seeds"][0], pr["n_chunks"]) + md_table(
        ["component", "cost"], [[m, f"{d['us_per_line']:.0f} us/line" if "us_per_line" in d else f"{d['us_per_query']:.0f} us/query"] for m, d in pr.items() if m != "n_chunks"]))
    cf = R["confusion"]
    T.append("### T12. Tagger token confusion at noise %s (token accuracy %.4f)\n\n" % (cf["noise"], cf["accuracy"]) + md_table(
        ["label", "precision", "recall", "F1", "support"], [[l, f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['f1']:.3f}", d["support"]] for l, d in cf["per_label"].items()])
        + "\n\nLargest confusions (gold -> predicted: count): " + ", ".join(f"{g}->{p}: {c}" for g, p, c in cf["top"]))
    return "\n\n".join(T) + "\n"



def _line_plot(path, series, xs, title, ylabel, xlabel):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, d in series.items():
        ys = [d[str(x)]["mean"] for x in xs]
        ci = [d[str(x)]["ci"] for x in xs]
        ax.errorbar(xs, ys, yerr=ci, marker="o", capsize=3, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_figures(R: Dict, cfg: Dict, out: Path) -> List[str]:
    figs = []
    p = out / "fig1_extraction_vs_noise.png"
    _line_plot(p, R["extraction"], cfg["noise_levels"], "Row extraction F1 vs OCR noise", "row F1", "per-character noise probability")
    figs.append(p.name)
    p = out / "fig2_qa_vs_noise.png"
    _line_plot(p, R["qa"]["acc"], cfg["qa_noise_levels"], "End-to-end QA accuracy vs noise", "exact match", "per-character noise probability")
    figs.append(p.name)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    methods = list(R["retrieval"]["header=True|para=0.5"])
    w = 0.4
    for j, key in enumerate(("header=True|para=0.0", "header=True|para=0.5")):
        ys = [R["retrieval"][key][m]["recall@5"]["mean"] for m in methods]
        es = [R["retrieval"][key][m]["recall@5"]["ci"] for m in methods]
        ax.bar(np.arange(len(methods)) + (j - 0.5) * w, ys, w, yerr=es, capsize=2, label=key.split("|")[1])
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("recall@5")
    ax.set_title("Chunk retrieval by method (paraphrase rate 0 vs 0.5)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = out / "fig3_retrieval_methods.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p.name)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    comp = R["sweeps"]["compress"]
    bs = [comp[k]["bytes"]["mean"] for k in comp]
    rs = [comp[k]["recall@5"]["mean"] for k in comp]
    ax.plot(bs, rs, "o-")
    for k, x, y in zip(comp, bs, rs):
        ax.annotate(f"{k}b", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("index bytes (log)")
    ax.set_ylabel("recall@5")
    ax.set_title("LSA embedding quantisation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig4_index_compression.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p.name)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    cats = R["qa"]["cats"]
    ql = [str(x) for x in cfg["qa_noise_levels"]]
    bottom = np.zeros(len(ql))
    for c in ("correct", "retrieval_miss", "extract_error", "wrong_row"):
        v = np.array([cats[x].get(c, 0) for x in ql], dtype=float)
        v = v / np.maximum(np.array([sum(cats[x].values()) for x in ql]), 1)
        ax.bar(ql, v, bottom=bottom, label=c)
        bottom += v
    ax.set_xlabel("noise level")
    ax.set_ylabel("fraction of questions")
    ax.set_title("Outcome breakdown, headline pipeline")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = out / "fig5_failure_breakdown.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(p.name)
    return figs


def run_all(cfg: Dict, out_dir: str, log=print) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = TaggerCache(cfg)
    R: Dict = {"config": cfg}
    t0 = time.time()
    for name, fn in (("extraction", lambda: exp_extraction(cfg, cache)), ("ablation", lambda: exp_ablation(cfg, cache)),
                     ("style_shift", lambda: exp_style_shift(cfg, cache)), ("retrieval", lambda: exp_retrieval(cfg)),
                     ("qa", lambda: exp_qa(cfg, cache)), ("extraction_failures", lambda: exp_extraction_failures(cfg, cache)),
                     ("sweeps", lambda: exp_sweeps(cfg, cache)),
                     ("fixedpoint", lambda: exp_fixedpoint(cfg, cache)), ("profile", lambda: exp_profile(cfg, cache)), ("confusion", lambda: exp_confusion(cfg, cache))):
        R[name] = fn()
        log(f"[{time.time() - t0:6.1f}s] {name} done")
    R["figures"] = make_figures(R, cfg, out)
    R["runtime_s"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(R, indent=1, default=float))
    (out / "tables.md").write_text(build_tables(R, cfg))
    return R
