# -*- coding: utf-8 -*-
"""导出 identifier=-1（无 HPO 映射）金标实体的人工补标工单。

这些金标在原数据里没有 HPO ID，词典无法映射；本工单给出位置与上下文，便于人工补 ID 或
据此补词表。输出 unmapped_gold_worklist.tsv。

用法：
    python stage2/export_unmapped_worklist.py --data "D:\\...\\PatienPheX" --out "D:\\...\\stage2\\output_lex"
"""
from __future__ import annotations

import argparse
import json
import os


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


def section_at(passages, offset):
    for p in passages:
        s = int(p.get("offset", 0))
        if s <= offset < s + len(str(p.get("text", ""))):
            return str(p.get("section_type", "")), str(p.get("type", ""))
    return "", ""


def has_hpo(identifier):
    return any(x.strip() and x.strip() != "-1" for x in str(identifier).split(";"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=60, help="上下文窗口字符数")
    args = ap.parse_args()

    with open(os.path.join(args.data, "PatientPheX-train.jsonl"), encoding="utf-8") as f:
        docs = [json.loads(l) for l in f if l.strip()]

    rows = []
    for doc in docs:
        pmc = str(doc.get("pmc_id"))
        passages = doc.get("full_text", [])
        text = rebuild_full_text(passages)
        for e in doc.get("entities", []):
            if has_hpo(e.get("identifier", "-1")):
                continue
            s = int(e.get("offset", 0)); ln = int(e.get("length", 0)); en = s + ln
            sec, ptype = section_at(passages, s)
            ctx = text[max(0, s - args.ctx):min(len(text), en + args.ctx)].replace("\t", " ").replace("\n", " ")
            rows.append((pmc, doc.get("pmid"), s, ln, text[s:en], str(e.get("note")), sec, ptype, ctx))

    path = os.path.join(args.out, "unmapped_gold_worklist.tsv")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("pmc_id\tpmid\toffset\tlength\ttext\tnote\tsection_type\ttype\tcontext\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    print(f"[unmapped] {len(rows)} 个 id=-1 金标实体 -> {path}")
    # 按 surface 聚合统计
    from collections import Counter
    c = Counter(r[4].lower() for r in rows)
    print("[unmapped] 去重 surface 数 =", len(c))
    print("[unmapped] 高频：", c.most_common(10))


if __name__ == "__main__":
    main()
