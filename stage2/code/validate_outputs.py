# -*- coding: utf-8 -*-
"""校验阶段二产出：pmc_id 覆盖/唯一、offset 零漂移、文本一致、字段合法。

用法：
    python stage2/validate_outputs.py --data "D:\\...\\PatienPheX" --out "D:\\...\\stage2\\output"
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


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_submission(path, raw_docs, full_texts):
    errs = []
    rows = load_jsonl(path)
    ids = [str(r.get("pmc_id")) for r in rows]
    if len(ids) != len(set(ids)):
        errs.append(f"{os.path.basename(path)}: duplicate pmc_id")
    if set(ids) != set(raw_docs):
        errs.append(f"{os.path.basename(path)}: pmc_id set mismatch "
                    f"(missing={len(set(raw_docs)-set(ids))}, extra={len(set(ids)-set(raw_docs))})")
    n_ent = 0
    for r in rows:
        pmc = str(r.get("pmc_id"))
        text = full_texts.get(pmc, "")
        for e in r.get("entities", []):
            n_ent += 1
            s, l = int(e["offset"]), int(e["length"])
            end = s + l
            if s < 0 or end > len(text) or s >= end:
                errs.append(f"{pmc}: bad span {s},{end}")
                continue
            if e.get("text") != text[s:end]:
                errs.append(f"{pmc}: text mismatch at {s}:{end}")
            if int(e.get("length", -1)) != len(str(e.get("text", ""))):
                errs.append(f"{pmc}: length != len(text)")
            if not str(e.get("identifier", "")).strip():
                errs.append(f"{pmc}: empty identifier")
            if e.get("type") != "Phenotype":
                errs.append(f"{pmc}: bad type")
        if not isinstance(r.get("association", []), list):
            errs.append(f"{pmc}: association not list")
    return errs, n_ent


def check_candidates(path, raw_docs, full_texts):
    errs = []
    rows = load_jsonl(path)
    ids = [str(r.get("pmc_id")) for r in rows]
    if set(ids) != set(raw_docs):
        errs.append(f"{os.path.basename(path)}: pmc_id set mismatch")
    for r in rows:
        pmc = str(r.get("pmc_id"))
        text = full_texts.get(pmc, "")
        cands = r.get("candidates", [])
        if int(r.get("num_candidates", -1)) != len(cands):
            errs.append(f"{pmc}: num_candidates != len")
        for c in cands:
            s, e = int(c["start"]), int(c["end"])
            if s < 0 or e > len(text) or s >= e:
                errs.append(f"{pmc}: bad candidate {s},{e}")
                continue
            if c.get("text") != text[s:e]:
                errs.append(f"{pmc}: cand text mismatch {s}:{e}")
            if not c.get("ids") and c.get("source") != "np_unmapped":
                errs.append(f"{pmc}: cand empty ids (source={c.get('source')})")
            if "flags" not in c or "primary_reason" not in c:
                errs.append(f"{pmc}: cand missing flags/reason")
    return errs, sum(int(r.get("num_candidates", 0)) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = {}
    for name in ("PatientPheX-A.jsonl", "PatientPheX-V1-B.jsonl"):
        for d in load_jsonl(os.path.join(args.data, name)):
            raw[str(d["pmc_id"])] = d
    full_texts = {k: rebuild_full_text(v.get("full_text", [])) for k, v in raw.items()}
    a_set = {str(d["pmc_id"]) for d in load_jsonl(os.path.join(args.data, "PatientPheX-A.jsonl"))}
    b_set = {str(d["pmc_id"]) for d in load_jsonl(os.path.join(args.data, "PatientPheX-V1-B.jsonl"))}

    all_errs = []
    for fn in ("submit_pred_A.jsonl", "submit_pred_A_filtered.jsonl"):
        errs, n = check_submission(os.path.join(args.out, fn),
                                   {k: v for k, v in raw.items() if k in a_set}, full_texts)
        print(f"{fn}: entities={n}, errors={len(errs)}")
        all_errs += errs[:5]
    for fn in ("submit_pred_B.jsonl", "submit_pred_B_filtered.jsonl"):
        errs, n = check_submission(os.path.join(args.out, fn),
                                   {k: v for k, v in raw.items() if k in b_set}, full_texts)
        print(f"{fn}: entities={n}, errors={len(errs)}")
        all_errs += errs[:5]
    for fn in ("candidates_A.jsonl", "candidates_B.jsonl", "candidates_train.jsonl"):
        sub = a_set if fn == "candidates_A.jsonl" else (b_set if fn == "candidates_B.jsonl" else None)
        if sub is None:
            rows = load_jsonl(os.path.join(args.out, fn))
            print(f"{fn}: docs={len(rows)}, candidates={sum(r['num_candidates'] for r in rows)}")
            continue
        errs, n = check_candidates(os.path.join(args.out, fn),
                                   {k: v for k, v in raw.items() if k in sub}, full_texts)
        print(f"{fn}: candidates={n}, errors={len(errs)}")
        all_errs += errs[:5]

    if all_errs:
        print("VALIDATION FAILED:")
        for e in all_errs:
            print("  -", e)
    else:
        print("VALIDATION OK")


if __name__ == "__main__":
    main()
