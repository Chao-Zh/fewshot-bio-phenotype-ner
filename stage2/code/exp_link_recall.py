# -*- coding: utf-8 -*-
"""诊断：补词表 + 粗召回 + 轻量链接，在留出集（外推）上的 cover/overlap/strict 与链接准确率。

用法：
    python exp_link_recall.py --data <数据目录> --out <含 lexicon.json/hpo_subgraph.json 的输出目录>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpo as hpo_mod          # noqa: E402
import matcher as matcher_mod  # noqa: E402
import np_recall as np_mod      # noqa: E402
from linker import HpoLinker    # noqa: E402


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


def ids_of(identifier):
    return {x.strip() for x in str(identifier).split(";")
            if x.strip() and x.strip() != "-1"}


def gold_of(doc, text):
    out = []
    for e in doc.get("entities", []):
        s = int(e["offset"]); en = s + int(e["length"])
        out.append((s, en, ids_of(e.get("identifier", "-1")), text[s:en]))
    return out


def build_aug(ontology, docs, docs_text):
    lex = {k: set(v) for k, v in ontology.items()}
    for doc in docs:
        text = docs_text[str(doc["pmc_id"])]
        for e in doc.get("entities", []):
            s = int(e["offset"]); en = s + int(e["length"])
            ids = ids_of(e.get("identifier", "-1"))
            if not ids:
                continue
            for v in hpo_mod.surface_variants(text[s:en].strip()):
                lex.setdefault(v.lower(), set()).update(ids)
    return {k: sorted(v) for k, v in lex.items() if v}


def build_linked(lex_cands, chunks, text, meta, linker, thr):
    out = []
    seen = set()
    for c in lex_cands:
        key = (c["start"], c["end"])
        if key not in seen:
            seen.add(key)
            out.append({**c, "source": "lexicon"})
    for cs, ce in sorted(chunks):
        if (cs, ce) in seen:
            continue
        ids = set()
        for c in lex_cands:
            if c["start"] >= cs and c["end"] <= ce:
                ids |= set(c["ids"])
        src = "np"
        if not ids:
            lids, _ = linker.link(text[cs:ce], thr=thr)
            if lids:
                ids = set(lids); src = "np_linked"
            else:
                out.append({"start": cs, "end": ce, "ids": [], "primary": None,
                            "ambiguous": False, "depth": 0, "source": "np_unmapped"})
                seen.add((cs, ce)); continue
        ids_sorted = sorted(ids)
        primary = max(ids_sorted, key=lambda i: (meta.get(i, {}).get("depth", 0), i))
        out.append({"start": cs, "end": ce, "ids": ids_sorted, "primary": primary,
                    "ambiguous": len(ids_sorted) > 1,
                    "depth": meta.get(primary, {}).get("depth", 0), "source": src})
        seen.add((cs, ce))
    return out


def metrics(cands, gold):
    cs = [(c["start"], c["end"], set(c["ids"])) for c in cands]
    cov = ov = strict = 0
    for s, e, ids, _ in gold:
        if any(a <= s and e <= b for a, b, _ in cs):
            cov += 1
        if any(a < e and s < b for a, b, _ in cs):
            ov += 1
        if any(a == s and b == e and (ids & gi) for a, b, gi in cs):
            strict += 1
    return cov, ov, strict


def main():
    ap = argparse.ArgumentParser(description="补词+粗召回+链接 外推诊断")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out, data = args.out, args.data

    with open(os.path.join(out, "lexicon.json"), encoding="utf-8") as f:
        ontology = json.load(f)["surfaces"]
    with open(os.path.join(out, "hpo_subgraph.json"), encoding="utf-8") as f:
        sub = json.load(f)["terms"]
    meta = {tid: {"depth": t["depth"], "branch": t["branch"]} for tid, t in sub.items()}
    with open(os.path.join(data, "PatientPheX-train.jsonl"), encoding="utf-8") as f:
        train = [json.loads(l) for l in f if l.strip()]
    docs_text = {str(d["pmc_id"]): rebuild_full_text(d.get("full_text", [])) for d in train}

    held = [d for i, d in enumerate(train) if i % 4 == 0]
    pool = [d for i, d in enumerate(train) if i % 4 != 0]
    lex_aug = build_aug(ontology, pool, docs_text)
    autom = matcher_mod.build_automaton(lex_aug)
    heads = np_mod.derive_head_vocab(sub, min_freq=2)
    linker = HpoLinker(sub)

    cache = []
    for doc in held:
        text = docs_text[str(doc["pmc_id"])]
        gold = gold_of(doc, text)
        lex_cands = matcher_mod.make_candidates(text, autom, lex_aug, meta)
        chunks = np_mod.build_chunks(text, heads, max_left=5, max_right=4)
        chunks = chunks | np_mod.build_segments(text, heads, max_chars=90)
        cache.append((doc, text, gold, lex_cands, chunks))
    tot_gold = sum(len(c[2]) for c in cache)

    def pct(v):
        return round(v / tot_gold, 4) if tot_gold else 0.0

    base = [0, 0, 0]; npl = [0, 0, 0]
    for doc, text, gold, lex_cands, chunks in cache:
        for i, v in enumerate(metrics(lex_cands, gold)):
            base[i] += v
        for i, v in enumerate(metrics(np_mod.combine(lex_cands, chunks, meta, keep_unmapped=True), gold)):
            npl[i] += v

    result = {"held_out_docs": len(held), "gold": tot_gold,
              "augment_only": {"cover": pct(base[0]), "overlap": pct(base[1]), "strict": pct(base[2])},
              "np_inherit": {"cover": pct(npl[0]), "overlap": pct(npl[1]), "strict": pct(npl[2])},
              "link": []}
    print(f"held-out docs={len(held)} gold={tot_gold}")
    print(f"[补词表]      cover={pct(base[0])} overlap={pct(base[1])} strict={pct(base[2])}")
    print(f"[+NP继承ID]   cover={pct(npl[0])} overlap={pct(npl[1])} strict={pct(npl[2])}")

    for thr in (0.4, 0.5, 0.6):
        lm = [0, 0, 0]; hard = 0; ok1 = 0; ok3 = 0
        for doc, text, gold, lex_cands, chunks in cache:
            for i, v in enumerate(metrics(build_linked(lex_cands, chunks, text, meta, linker, thr), gold)):
                lm[i] += v
            for s, e, ids, surf in gold:
                if not ids:
                    continue
                if any(c["start"] == s and c["end"] == e and (ids & set(c["ids"])) for c in lex_cands):
                    continue
                hard += 1
                l1, _ = linker.link(surf, thr=thr, topk=1)
                if l1 and (ids & set(l1)):
                    ok1 += 1
                l3, _ = linker.link(surf, thr=thr, topk=3)
                if l3 and (ids & set(l3)):
                    ok3 += 1
        row = {"thr": thr, "cover": pct(lm[0]), "overlap": pct(lm[1]), "strict": pct(lm[2]),
               "hard": hard, "link_top1": round(ok1 / hard, 4) if hard else None,
               "link_top3": round(ok3 / hard, 4) if hard else None}
        result["link"].append(row)
        print(f"[+NP链接 thr={thr}] cover={row['cover']} overlap={row['overlap']} strict={row['strict']} "
              f"| 难例链接 top1={ok1}/{hard} top3={ok3/hard:.3f}")

    with open(os.path.join(out, "exp_link_recall.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(out, "exp_link_recall.json"))


if __name__ == "__main__":
    main()
