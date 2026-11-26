# -*- coding: utf-8 -*-
"""阶段一词表补全：HPO 本体 + 80 篇金标未覆盖 surface，产出可溯源词表。

产出（写到 --out）：
- lexicon_aug.json  : 供 matcher 使用（surface -> [hpo_ids] + 歧义表）
- lexicon.tsv       : PDF 阶段一要求的 lexicon.tsv（surface / ids / source / 数量 / 歧义）
- lexicon_workorder.tsv : 补词工单（新增的 corpus surface、命中次数、映射、可疑标记）
- lexicon_unmapped.tsv  : id=-1（无 HPO 映射）的金标 surface，需人工确认
- lexicon_aug_report.md : 补词前后可映射召回对比 + 统计

口径：corpus surface 来自 80 篇人工金标，属阶段一“补同义词/通俗说法”的资源建设；
对 train 是 in-sample（偏乐观），对 A/B 的可外推估计见 cv_recall.json（5 折）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpo as hpo_mod          # noqa: E402
import matcher as matcher_mod  # noqa: E402


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


def split_ids(identifier):
    raw = [x.strip() for x in str(identifier).split(";")]
    return [x for x in raw if x and x != "-1"], ("-1" in raw)


def recall_of(lexicon, meta, train, docs_text):
    autom = matcher_mod.build_automaton(lexicon)
    cov = ov = strict = 0
    gold_n = 0
    for doc in train:
        text = docs_text[str(doc["pmc_id"])]
        cands = matcher_mod.make_candidates(text, autom, lexicon, meta)
        cs = [(c["start"], c["end"], set(c["ids"])) for c in cands]
        for e in doc.get("entities", []):
            s = int(e["offset"]); en = s + int(e["length"])
            ids, _ = split_ids(e.get("identifier", "-1"))
            gids = set(ids)
            gold_n += 1
            if any(a <= s and en <= b for a, b, _ in cs):
                cov += 1
            if any(a < en and s < b for a, b, _ in cs):
                ov += 1
            if any(a == s and b == en and (gids & gi) for a, b, gi in cs):
                strict += 1
    n = gold_n or 1
    return {"gold": gold_n, "cover": cov / n, "overlap": ov / n, "strict": strict / n}


def main():
    ap = argparse.ArgumentParser(description="补词表：HPO 本体 + 80 篇金标 surface")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ontology-lexicon", default=None,
                    help="ontology-only lexicon.json（默认 <out>/lexicon.json）")
    ap.add_argument("--subgraph", default=None,
                    help="hpo_subgraph.json（默认 <out>/hpo_subgraph.json）")
    args = ap.parse_args()

    out = args.out
    lex_onto_path = args.ontology_lexicon or os.path.join(out, "lexicon.json")
    sub_path = args.subgraph or os.path.join(out, "hpo_subgraph.json")

    with open(lex_onto_path, encoding="utf-8") as f:
        onto = json.load(f)
    onto_surfaces = onto["surfaces"]
    with open(sub_path, encoding="utf-8") as f:
        sub = json.load(f)["terms"]
    meta = {tid: {"depth": t["depth"], "branch": t["branch"]} for tid, t in sub.items()}

    with open(os.path.join(args.data, "PatientPheX-train.jsonl"), encoding="utf-8") as f:
        train = [json.loads(l) for l in f if l.strip()]
    docs_text = {str(d["pmc_id"]): rebuild_full_text(d.get("full_text", [])) for d in train}

    # 1) 以 ontology 词表为底
    entries = {k: set(v) for k, v in onto_surfaces.items()}
    src = {k: {"ontology"} for k in entries}
    onto_keys = set(entries)

    # 2) 灌入 corpus surface（原串 + 变体）
    corpus_orig = {}  # lower -> {ids:set, count:int, surface:str}
    unmapped = Counter()
    unmapped_ids_present = set()
    for doc in train:
        text = docs_text[str(doc["pmc_id"])]
        for e in doc.get("entities", []):
            s = int(e["offset"]); en = s + int(e["length"])
            surf = text[s:en].strip()
            if len(surf) < 3:
                continue
            ids, has_neg = split_ids(e.get("identifier", "-1"))
            if not ids:
                unmapped[surf] += 1
                continue
            ids_set = set(ids)
            low = surf.lower()
            rec = corpus_orig.setdefault(low, {"ids": set(), "count": 0, "surface": surf})
            rec["ids"] |= ids_set
            rec["count"] += 1
            for v in hpo_mod.surface_variants(surf):
                key = v.lower()
                if key not in entries:
                    entries[key] = set()
                    src[key] = set()
                entries[key] |= ids_set
                src[key].add("corpus")

    # 3) 歧义表 + 统计
    surfaces_out = {k: sorted(v) for k, v in entries.items() if v}
    ambiguity = {k: v for k, v in surfaces_out.items() if len(v) > 1}
    new_keys = [k for k in entries if src[k] == {"corpus"} and entries[k]]
    both_keys = [k for k in entries if src[k] == {"ontology", "corpus"} and entries[k]]

    lex_aug = {
        "version": hpo_mod.VERSION,
        "root": hpo_mod.ROOT,
        "num_surfaces": len(surfaces_out),
        "num_ambiguous": len(ambiguity),
        "source_counts": {
            "ontology_only": len(onto_keys),
            "corpus_new": len(new_keys),
            "overlap_both": len(both_keys),
        },
        "surfaces": surfaces_out,
        "ambiguity": ambiguity,
    }
    with open(os.path.join(out, "lexicon_aug.json"), "w", encoding="utf-8") as f:
        json.dump(lex_aug, f, ensure_ascii=False)

    # 4) lexicon.tsv
    with open(os.path.join(out, "lexicon.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("surface\thpo_ids\tsource\tnum_ids\tambiguous\n")
        for k in sorted(surfaces_out):
            ids = surfaces_out[k]
            srcs = "+".join(sorted(src.get(k, {"ontology"})))
            f.write(f"{k}\t{';'.join(ids)}\t{srcs}\t{len(ids)}\t"
                    f"{'1' if len(ids) > 1 else '0'}\n")

    # 5) 补词工单（仅 corpus 新增 / 与本体冲突）
    work_rows = []
    for low, rec in corpus_orig.items():
        if low in onto_keys:
            continue  # 本体已覆盖，不算补词
        ids = sorted(rec["ids"])
        flags = []
        if len(ids) > 1:
            flags.append("ambiguous")
        if re.search(r"\d", rec["surface"]):
            flags.append("has_digit")
        if len(rec["surface"]) > 50:
            flags.append("long")
        work_rows.append((low, ";".join(ids), rec["count"], rec["surface"], ",".join(flags)))
    work_rows.sort(key=lambda r: (-r[2], r[0]))
    with open(os.path.join(out, "lexicon_workorder.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("surface\thpo_ids\tcorpus_count\texample\tflags\n")
        for low, ids, cnt, ex, fl in work_rows:
            f.write(f"{low}\t{ids}\t{cnt}\t{ex}\t{fl}\n")

    # 6) 未映射（id=-1）金标
    with open(os.path.join(out, "lexicon_unmapped.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("surface\tcount\n")
        for surf, cnt in unmapped.most_common():
            f.write(f"{surf}\t{cnt}\n")

    # 7) 补词前后可映射召回（in-sample）
    before = recall_of(onto_surfaces, meta, train, docs_text)
    after = recall_of(surfaces_out, meta, train, docs_text)

    # 8) 报告
    lines = ["# 补词表报告（HPO 本体 + 80 篇金标 surface）", ""]
    lines += ["## 词表规模", ""]
    lines.append(f"- 本体 surface：{len(onto_keys)}")
    lines.append(f"- corpus 新增 surface：{len(new_keys)}；与本体重叠：{len(both_keys)}")
    lines.append(f"- 合并后 surface：{len(surfaces_out)}；歧义 surface：{len(ambiguity)}")
    lines.append(f"- 未映射（id=-1）金标 surface 种类：{len(unmapped)}（见 lexicon_unmapped.tsv）")
    lines += ["", "## 补词前后可映射召回（in-sample，80 篇）", "",
              "| 词表 | candidate recall(覆盖) | recall(overlap) | recall_strict |",
              "| --- | --- | --- | --- |",
              f"| ontology-only | {before['cover']:.4f} | {before['overlap']:.4f} | {before['strict']:.4f} |",
              f"| 补词后 lexicon_aug | {after['cover']:.4f} | {after['overlap']:.4f} | {after['strict']:.4f} |",
              ""]
    lines += ["## 高频新增补词（前 30）", "", "| surface | hpo_ids | 次数 |", "| --- | --- | --- |"]
    for low, ids, cnt, ex, fl in work_rows[:30]:
        lines.append(f"| {ex} | {ids} | {cnt} |")
    lines += ["", "## 歧义 surface（前 20，本体+corpus 合并后一词多义）", "",
              "| surface | hpo_ids |", "| --- | --- |"]
    for k in list(ambiguity)[:20]:
        lines.append(f"| {k} | {';'.join(ambiguity[k])} |")
    lines += ["", "## 口径与说明", "",
              "- 补词来源是 80 篇人工金标，属阶段一“补同义词/通俗说法”的资源建设，对 train 为 in-sample（偏乐观）。",
              "- 对 A/B 的可外推召回见 `cv_recall.json`（文档级 5 折，用其余折 surface 补词表）。",
              "- `lexicon.tsv` 为 PDF 阶段一要求的交付；`lexicon_aug.json` 供匹配器直接使用。"]
    with open(os.path.join(out, "lexicon_aug_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[lexicon] ontology surfaces =", len(onto_keys))
    print("[lexicon] corpus new       =", len(new_keys), " overlap =", len(both_keys))
    print("[lexicon] merged surfaces  =", len(surfaces_out), " ambiguous =", len(ambiguity))
    print(f"[lexicon] recall BEFORE cover/overlap/strict = "
          f"{before['cover']:.4f}/{before['overlap']:.4f}/{before['strict']:.4f}")
    print(f"[lexicon] recall AFTER  cover/overlap/strict = "
          f"{after['cover']:.4f}/{after['overlap']:.4f}/{after['strict']:.4f}")


if __name__ == "__main__":
    main()
