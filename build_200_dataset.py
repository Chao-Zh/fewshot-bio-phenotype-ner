"""构建 200 篇完整带标注数据集 (4:1:5 划分)。

数据来源:
- train (80 篇): PatientPheX-train.jsonl, 人工金标注
- val   (20 篇): PatientPheX-A.jsonl 原文 + submit_pred_A.jsonl 的 AI 标注
- test  (100 篇): PatientPheX-V1-B.jsonl 原文 + submit_pred_B.jsonl 的 AI 标注

流程: 按 pmc_id 把 AI 标注回填到原文 -> 重建全文 -> 转标准 Sample -> 4:1:5 划分 -> 统计报告

注意: 输出含病例原文与标注, 属天池保密数据, 严禁上传公共仓库 (已加入 .gitignore)。

用法:
    # 方式一: 指定数据目录
    python build_200_dataset.py --data-root "D:/path/to/数据目录"
    # 方式二: 逐个指定文件
    python build_200_dataset.py --train a.jsonl --a-raw b.jsonl --a-pred c.jsonl \\
        --b-raw d.jsonl --b-pred e.jsonl
    # 方式三: 环境变量 PATIENTPHEX_ROOT
    set PATIENTPHEX_ROOT=D:/path/to/数据目录 && python build_200_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Tuple

from patientpheX_adapter import load_patientphe_x, save_samples

BASE = os.path.dirname(os.path.abspath(__file__))

# 原始数据目录: 通过环境变量 PATIENTPHEX_ROOT 或命令行参数指定,
# 避免在代码中硬编码本地绝对路径。
DATA_ROOT = os.environ.get("PATIENTPHEX_ROOT", "")

TRAIN_PATH = ""
A_RAW_PATH = ""
A_PRED_PATH = ""
B_RAW_PATH = ""
B_PRED_PATH = ""

OUT_DIR = os.path.join(BASE, "data", "full")
REPORT_DIR = os.path.join(BASE, "output_full")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_ai_annotations(
    raw_path: str, pred_path: str, subset: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """把 AI 标注 (entities/association) 按 pmc_id 回填到原文文档。"""
    raws = load_jsonl(raw_path)
    preds = load_jsonl(pred_path)
    pred_map = {str(p["pmc_id"]): p for p in preds}

    merged: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "subset": subset,
        "raw_docs": len(raws),
        "pred_docs": len(preds),
        "matched": 0,
        "unmatched_raw": [],
        "entities_added": 0,
        "associations_added": 0,
        "ai_note_counter": Counter(),
        "ai_identifier_missing": 0,
        "ai_identifier_multi": 0,
        "span_mismatch": 0,
    }

    for r in raws:
        pmc = str(r.get("pmc_id", ""))
        p = pred_map.get(pmc)
        if p is None:
            stats["unmatched_raw"].append(pmc)
            continue
        doc = dict(r)
        ents = list(p.get("entities", []))
        assoc = list(p.get("association", []))
        doc["entities"] = ents
        doc["association"] = assoc
        merged.append(doc)
        stats["matched"] += 1
        stats["entities_added"] += len(ents)
        stats["associations_added"] += len(assoc)
        for e in ents:
            stats["ai_note_counter"][str(e.get("note"))] += 1
            ident = str(e.get("identifier", ""))
            if not ident or ident == "-1":
                stats["ai_identifier_missing"] += 1
            elif ";" in ident:
                stats["ai_identifier_multi"] += 1
    stats["ai_note_counter"] = dict(stats["ai_note_counter"])
    return merged, stats


def write_tmp(docs: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def convert(docs: List[Dict[str, Any]], subset: str, tmp_path: str):
    """复用适配器把原始文档转成 Sample。"""
    write_tmp(docs, tmp_path)
    return load_patientphe_x(tmp_path, subset=subset)


def build_markdown(stats: Dict[str, Any]) -> str:
    lines = [
        "# 200 篇带标注数据集构建报告",
        "",
        "## 数据来源与划分 (4:1:5)",
        "",
        "| 集合 | 来源 | 篇数 | 标注来源 | 实体数 | 关联数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("train", "val", "test"):
        s = stats["splits"][name]
        lines.append(
            f"| {name} | {s['source']} | {s['docs']} | {s['annotation']} | "
            f"{s['entities']} | {s['associations']} |"
        )
    total = stats["total"]
    lines += [
        "",
        f"**合计**: {total['docs']} 篇, {total['entities']} 个实体, {total['associations']} 条关联。",
        "",
        "## AI 标注质量提示",
        "",
        "- A/B 组的 `entities` 与 `association` 来自 AI 预测 (`submit_pred_A/B.jsonl`)，"
        "属于**弱监督伪标签**，不是人工金标准。",
        "- 若把这 200 篇用于模型评估，val/test 上的指标会受伪标签噪声影响；"
        "正式评估应优先使用人工标注数据。",
        "- HPO identifier 的有效性需结合 `hp.obo` 本体校验（本脚本未做本体合法性过滤）。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    global TRAIN_PATH, A_RAW_PATH, A_PRED_PATH, B_RAW_PATH, B_PRED_PATH

    ap = argparse.ArgumentParser(description="构建 200 篇 4:1:5 带标注数据集")
    ap.add_argument("--data-root", default=DATA_ROOT,
                    help="原始数据所在目录(优先), 未提供时回退到环境变量 PATIENTPHEX_ROOT")
    ap.add_argument("--train", default=None)
    ap.add_argument("--a-raw", default=None)
    ap.add_argument("--a-pred", default=None)
    ap.add_argument("--b-raw", default=None)
    ap.add_argument("--b-pred", default=None)
    args = ap.parse_args()

    root = args.data_root or ""
    TRAIN_PATH = args.train or os.path.join(root, "PatientPheX-train.jsonl")
    A_RAW_PATH = args.a_raw or os.path.join(root, "PatientPheX-A.jsonl")
    A_PRED_PATH = args.a_pred or os.path.join(root, "submit_pred_A.jsonl")
    B_RAW_PATH = args.b_raw or os.path.join(root, "PatientPheX-V1-B.jsonl")
    B_PRED_PATH = args.b_pred or os.path.join(root, "submit_pred_B.jsonl")

    missing = [p for p in (TRAIN_PATH, A_RAW_PATH, A_PRED_PATH, B_RAW_PATH, B_PRED_PATH)
               if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "以下数据文件不存在, 请用 --train/--a-raw/--a-pred/--b-raw/--b-pred 指定:\n  "
            + "\n  ".join(missing)
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("[1/5] 读取并合并 A 组 (原文 + AI 标注) ...")
    a_docs, a_stats = merge_ai_annotations(A_RAW_PATH, A_PRED_PATH, "val")
    print("      ", {k: v for k, v in a_stats.items() if k != "unmatched_raw"})

    print("[2/5] 读取并合并 B 组 (原文 + AI 标注) ...")
    b_docs, b_stats = merge_ai_annotations(B_RAW_PATH, B_PRED_PATH, "test")
    print("      ", {k: v for k, v in b_stats.items() if k != "unmatched_raw"})

    print("[3/5] 转换为标准 Sample ...")
    train, train_stats = load_patientphe_x(TRAIN_PATH, subset="train")
    val, val_stats = convert(a_docs, "val", os.path.join(OUT_DIR, "_tmp_val.jsonl"))
    test, test_stats = convert(b_docs, "test", os.path.join(OUT_DIR, "_tmp_test.jsonl"))
    for tmp in ("_tmp_val.jsonl", "_tmp_test.jsonl"):
        p = os.path.join(OUT_DIR, tmp)
        if os.path.exists(p):
            os.remove(p)

    print("[4/5] 按 4:1:5 写出三集合 ...")
    for name, samples in (("train", train), ("val", val), ("test", test)):
        save_samples(samples, os.path.join(OUT_DIR, f"{name}.jsonl"))
        print(f"       {name}: {len(samples)} 篇")

    print("[5/5] 生成统计报告 ...")
    stats = {
        "total": {
            "docs": len(train) + len(val) + len(test),
            "entities": train_stats["num_entities"] + val_stats["num_entities"] + test_stats["num_entities"],
            "associations": train_stats["num_associations"] + val_stats["num_associations"] + test_stats["num_associations"],
        },
        "splits": {
            "train": {
                "source": "PatientPheX-train.jsonl",
                "annotation": "人工金标注",
                "docs": len(train),
                "entities": train_stats["num_entities"],
                "associations": train_stats["num_associations"],
            },
            "val": {
                "source": "PatientPheX-A.jsonl",
                "annotation": "AI 预测 (submit_pred_A.jsonl)",
                "docs": len(val),
                "entities": val_stats["num_entities"],
                "associations": val_stats["num_associations"],
            },
            "test": {
                "source": "PatientPheX-V1-B.jsonl",
                "annotation": "AI 预测 (submit_pred_B.jsonl)",
                "docs": len(test),
                "entities": test_stats["num_entities"],
                "associations": test_stats["num_associations"],
            },
        },
        "merge_stats": {"A": a_stats, "B": b_stats},
        "adapter_stats": {"train": train_stats, "val": val_stats, "test": test_stats},
    }
    with open(os.path.join(REPORT_DIR, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(REPORT_DIR, "dataset_report.md"), "w", encoding="utf-8") as f:
        f.write(build_markdown(stats))

    print(f"\n完成! 数据集: {OUT_DIR}   报告: {REPORT_DIR}")


if __name__ == "__main__":
    main()
