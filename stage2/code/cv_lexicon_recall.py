# -*- coding: utf-8 -*-
"""词表补全的文档级 5 折交叉验证（诚实估计测试期 candidate recall）。

动机：`--augment-from-train` 在 train 上用自身金标 surface 补词表，train 召回≈1.0 属泄漏。
本脚本按文档 5 折：用「其余折」的 train 金标 surface 补词表，在「留出折」上量候选召回，
给出一个可外推到 A/B 的估计。不改变主产物，只做诊断。

用法：
    python stage2/cv_lexicon_recall.py --data "D:\\...\\PatienPheX" \
        --lexicon "D:\\...\\stage2\\output\\lexicon.json" --out "D:\\...\\stage2\\output" --folds 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpo as hpo_mod          # noqa: E402
import matcher as matcher_mod  # noqa: E402

TRAIN_FILE = "PatientPheX-train.jsonl"


def rebuild_full_text(passages):
    if not passages:
        return ""
    max_end = max(int(p["offset"]) + len(str(p.get("text", ""))) for p in passages)
    buf = [" "] * max_end
    for p in passages:
        s = int(p["offset"]); t = str(p.get("text", ""))
        for i, ch in enumerate(t):
            if s + i < max_end:
                buf[s + i] = ch
    return "".join(buf)


def parse_ids(identifier):
    return {x.strip() for x in str(identifier).split(";")
            if x.strip() and x.strip() != "-1"}


def gold_from_doc(doc):
    out = []
    for e in doc.get("entities", []):
        s = int(e.get("offset", 0)); ln = int(e.get("length", 0))
        out.append({"start": s, "end": s + ln,
                    "ids": parse_ids(e.get("identifier", "-1")),
                    "note": e.get("note")})
    return out


def add_surface(lex, surf, ids):
    if not surf or len(surf) < 3:
        return
    for v in hpo_mod.surface_variants(surf):
        key = v.lower()
        cur = set(lex.get(key, []))
        cur |= ids
        lex[key] = sorted(cur)


def run_cv(train, base_lexicon, meta, folds_n=5, verbose=True):
    """文档级 K 折：其余折 train surface 补词表，在留出折评估候选召回。"""
    folds = [[] for _ in range(folds_n)]
    for i, doc in enumerate(train):
        folds[i % folds_n].append(doc)

    tot = {"gold": 0, "cover": 0, "overlap": 0, "exact": 0, "strict": 0, "pred": 0}
    per_fold = []
    for fi in range(folds_n):
        held = folds[fi]
        others = [d for j, fo in enumerate(folds) if j != fi for d in fo]
        lex = dict(base_lexicon)
        for doc in others:
            text = rebuild_full_text(doc.get("full_text", []))
            for g in gold_from_doc(doc):
                if g["ids"]:
                    add_surface(lex, text[g["start"]:g["end"]].strip(), g["ids"])
        autom = matcher_mod.build_automaton(lex)

        f = {"gold": 0, "cover": 0, "overlap": 0, "exact": 0, "strict": 0, "pred": 0}
        for doc in held:
            text = rebuild_full_text(doc.get("full_text", []))
            cands = matcher_mod.make_candidates(text, autom, lex, meta)
            f["pred"] += len(cands)
            cand_sets = [(c["start"], c["end"], set(c["ids"])) for c in cands]
            for g in gold_from_doc(doc):
                gs, ge, gids = g["start"], g["end"], g["ids"]
                f["gold"] += 1
                if any(cs <= gs and ge <= ce for cs, ce, _ in cand_sets):
                    f["cover"] += 1
                if any(cs < ge and gs < ce for cs, ce, _ in cand_sets):
                    f["overlap"] += 1
                if any(cs == gs and ce == ge for cs, ce, _ in cand_sets):
                    f["exact"] += 1
                    if any(cs == gs and ce == ge and (gids & ids) for cs, ce, ids in cand_sets):
                        f["strict"] += 1
        per_fold.append(f)
        for k in tot:
            tot[k] += f[k]
        if verbose:
            print(f"[fold {fi}] gold={f['gold']} cover={f['cover']/max(f['gold'],1):.3f} "
                  f"overlap={f['overlap']/max(f['gold'],1):.3f} pred={f['pred']}")

    return {
        "folds": folds_n,
        "note": "文档级交叉验证：用其余折 train surface 补词表，在留出折评估。"
                "这是可外推到 A/B 的召回估计，非 in-sample 泄漏值。",
        "candidate_recall_cover": round(tot["cover"] / tot["gold"], 4),
        "candidate_recall_overlap": round(tot["overlap"] / tot["gold"], 4),
        "recall_exact_span": round(tot["exact"] / tot["gold"], 4),
        "recall_strict": round(tot["strict"] / tot["gold"], 4),
        "num_gold": tot["gold"],
        "num_pred": tot["pred"],
        "per_fold": per_fold,
    }


def main():
    ap = argparse.ArgumentParser(description="词表补全文档级交叉验证")
    ap.add_argument("--data", required=True)
    ap.add_argument("--lexicon", required=True, help="ontology-only lexicon.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    sub_path = os.path.join(args.out, "hpo_subgraph.json")
    with open(args.lexicon, encoding="utf-8") as f:
        base_lexicon = json.load(f)["surfaces"]
    with open(sub_path, encoding="utf-8") as f:
        sub = json.load(f)["terms"]
    meta = {tid: {"depth": t["depth"], "branch": t["branch"]} for tid, t in sub.items()}

    with open(os.path.join(args.data, TRAIN_FILE), encoding="utf-8") as f:
        train = [json.loads(l) for l in f if l.strip()]

    res = run_cv(train, base_lexicon, meta, args.folds)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cv_recall.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[cv] cover =", res["candidate_recall_cover"],
          " overlap =", res["candidate_recall_overlap"],
          " strict =", res["recall_strict"])


if __name__ == "__main__":
    main()
