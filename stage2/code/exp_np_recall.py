# -*- coding: utf-8 -*-
"""诊断：词典匹配 vs +名词短语粗召回，在训练集上的召回与候选数（调参用）。

用法：
    python exp_np_recall.py --data <数据目录> --out <含 lexicon.json/hpo_subgraph.json 的输出目录>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matcher as matcher_mod          # noqa: E402
import np_recall as np_mod             # noqa: E402
from run_stage2 import rebuild_full_text, gold_from_doc  # noqa: E402


def recall(cands, golds):
    cov = ov = 0
    for g in golds:
        gs, ge = g["start"], g["end"]
        if any(c["start"] <= gs and ge <= c["end"] for c in cands):
            cov += 1
        if any(c["start"] < ge and gs < c["end"] for c in cands):
            ov += 1
    n = len(golds) or 1
    return cov / n, ov / n


def main():
    ap = argparse.ArgumentParser(description="名词短语粗召回 in-sample 诊断")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.out, "lexicon.json"), encoding="utf-8") as f:
        lexicon = json.load(f)["surfaces"]
    with open(os.path.join(args.out, "hpo_subgraph.json"), encoding="utf-8") as f:
        terms = json.load(f)["terms"]
    meta = {tid: {"depth": t["depth"], "branch": t["branch"]} for tid, t in terms.items()}
    with open(os.path.join(args.data, "PatientPheX-train.jsonl"), encoding="utf-8") as f:
        train = [json.loads(l) for l in f if l.strip()]

    heads = np_mod.derive_head_vocab(terms, min_freq=2)
    autom = matcher_mod.build_automaton(lexicon)

    docs = []
    for doc in train:
        text = rebuild_full_text(doc.get("full_text", []))
        docs.append((text, gold_from_doc(doc), matcher_mod.make_candidates(text, autom, lexicon, meta)))

    allgold = sum(len(g) for _, g, _ in docs) or 1
    cov = ov = 0
    nlex = 0
    for text, gold, lc in docs:
        c, o = recall(lc, gold)
        cov += c * len(gold); ov += o * len(gold); nlex += len(lc)
    result = {"baseline": {"cover": round(cov / allgold, 4), "overlap": round(ov / allgold, 4),
                           "cands_per_doc": round(nlex / len(docs), 1)}, "sweep": []}
    print(f"[baseline] cover={cov/allgold:.4f} overlap={ov/allgold:.4f} cand/doc={nlex/len(docs):.1f}")

    for ml, mr in [(3, 2), (4, 3), (5, 3), (5, 4), (6, 4)]:
        for keep in (False, True):
            cov2 = ov2 = 0
            ntot = 0
            for text, gold, lc in docs:
                chunks = np_mod.build_chunks(text, heads, max_left=ml, max_right=mr)
                comb = np_mod.combine(lc, chunks, meta, keep_unmapped=keep)
                c, o = recall(comb, gold)
                cov2 += c * len(gold); ov2 += o * len(gold); ntot += len(comb)
            tag = "all" if keep else "mapped"
            row = {"ml": ml, "mr": mr, "keep_unmapped": keep,
                   "cover": round(cov2 / allgold, 4), "overlap": round(ov2 / allgold, 4),
                   "cands_per_doc": round(ntot / len(docs), 1)}
            result["sweep"].append(row)
            print(f"[np ml={ml} mr={mr} {tag}] cover={row['cover']} overlap={row['overlap']} cand/doc={row['cands_per_doc']}")

    with open(os.path.join(args.out, "exp_np_recall.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
