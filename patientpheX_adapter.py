"""PatientPheX (CHIP2026 Task 1) 真实数据 -> 统一 Sample 数据模型适配器。

关键设计:
1. 原始 full_text 是按章节切开的段落列表, entities/patient.mention 的 offset
   是"完整全文"上的全局字符位置。本适配器先按 offset 重建完整文本
   (段落间用单个空格填充), 再直接按全局 offset 映射实体, 保证零漂移。
2. 保留任务特有元数据: HPO identifier / note(NO否定, D重复) / patient 关联 /
   association, 供后续 NEN / RE 任务扩展使用。
3. label 统一为 "Phenotype"(真实数据只有这一类); note 信息编码进 label 后缀
   吗? —— 不。BIO 标签体系用干净的 "Phenotype", note 放 metadata,
   避免把 6027 个实体拆成 3 类导致少样本采样失衡。

用法:
    python patientpheX_adapter.py \
        --train "D:/.../PatientPheX-train.jsonl" \
        --out-dir data/real
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from medical_ner_preprocess.schema import Entity, Sample


def rebuild_full_text(passages: List[Dict[str, Any]]) -> str:
    """按 offset 重建完整全文。

    PatientPheX 的 full_text 每段带全局 offset, 段落之间的间隙(通常是换行)
    用空格填充。实体 offset 直接作用于重建后的字符串。
    """
    if not passages:
        return ""
    max_end = max(int(p["offset"]) + len(str(p.get("text", ""))) for p in passages)
    buffer = [" "] * max_end
    for p in passages:
        start = int(p["offset"])
        text = str(p.get("text", ""))
        for i, ch in enumerate(text):
            if start + i < max_end:
                buffer[start + i] = ch
    return "".join(buffer)


def load_patientphe_x(
    path: str,
    subset: str = "train",
    label: str = "Phenotype",
    keep_negative: bool = True,
) -> Tuple[List[Sample], Dict[str, Any]]:
    """读取 PatientPheX JSONL 并转换为 Sample 列表。

    Args:
        path: JSONL 文件路径。
        subset: 子集名称, 写入 Sample.source (train/dev/test)。
        label: 统一实体类别名。
        keep_negative: 是否保留 note=NO 的否定实体(保留在 metadata 中,
            但其 BIO 标签仍是 Phenotype; 不应删除, 因为评测要求识别否定)。
    Returns:
        (samples, stats)
    """
    samples: List[Sample] = []
    stats: Dict[str, Any] = {
        "subset": subset,
        "num_samples": 0,
        "num_entities": 0,
        "num_patients": 0,
        "num_associations": 0,
        "note_counter": Counter(),
        "identifier_missing": 0,
        "identifier_multi": 0,
        "span_mismatch": 0,
        "out_of_bounds": 0,
        "empty_passages": 0,
    }

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            full_text = rebuild_full_text(obj.get("full_text", []))
            if not full_text:
                stats["empty_passages"] += 1
                continue

            entities: List[Entity] = []
            raw_entities: List[Dict[str, Any]] = []
            for e in obj.get("entities", []):
                start = int(e.get("offset", 0))
                length = int(e.get("length", 0))
                end = start + length
                text = str(e.get("text", ""))
                identifier = str(e.get("identifier", "-1"))
                note = e.get("note")

                stats["note_counter"][str(note)] += 1
                if identifier == "-1":
                    stats["identifier_missing"] += 1
                elif ";" in identifier:
                    stats["identifier_multi"] += 1

                if start < 0 or end > len(full_text):
                    stats["out_of_bounds"] += 1
                    continue
                if full_text[start:end] != text:
                    stats["span_mismatch"] += 1
                    # 不丢弃: 以真实文本为准重写 surface text
                    text = full_text[start:end]

                if note == "NO" and not keep_negative:
                    continue

                entities.append(
                    Entity(
                        start=start,
                        end=end,
                        label=label,
                        text=text,
                        raw_label=identifier,
                    )
                )
                raw_entities.append(
                    {
                        "identifier": identifier,
                        "note": note,
                        "type": str(e.get("type", "Phenotype")),
                    }
                )

            patients = [
                {
                    "patient_id": str(p.get("patient_id", "")),
                    "mention": [
                        {
                            "text": str(m.get("text", "")),
                            "offset": int(m.get("offset", 0)),
                            "length": int(m.get("length", 0)),
                        }
                        for m in p.get("mention", [])
                    ],
                }
                for p in obj.get("patient", [])
            ]
            associations = [
                {
                    "patient_id": str(a.get("patient_id", "")),
                    "phenotype": [str(x) for x in a.get("phenotype", [])],
                }
                for a in obj.get("association", [])
            ]

            metadata = {
                "pmc_id": str(obj.get("pmc_id", "")),
                "pmid": str(obj.get("pmid", "")),
                "raw_entities": raw_entities,
                "patients": patients,
                "associations": associations,
                "passages": [
                    {
                        "section_type": str(p.get("section_type", "")),
                        "type": str(p.get("type", "")),
                        "offset": int(p.get("offset", 0)),
                    }
                    for p in obj.get("full_text", [])
                ],
            }

            sample = Sample(
                id=f"PheX-{subset}-{obj.get('pmc_id', lineno)}",
                text=full_text,
                entities=entities,
                source=f"PatientPheX/{subset}",
                metadata=metadata,
            )
            samples.append(sample)
            stats["num_samples"] += 1
            stats["num_entities"] += len(entities)
            stats["num_patients"] += len(patients)
            stats["num_associations"] += len(associations)

    stats["note_counter"] = dict(stats["note_counter"])
    return samples, stats


def save_samples(samples: List[Sample], path: str) -> None:
    """保存为管线标准 JSONL (Sample.to_dict 格式)。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="PatientPheX -> 统一 Sample 数据集")
    ap.add_argument("--train", required=True, help="PatientPheX-train.jsonl 路径")
    ap.add_argument("--dev", default=None, help="PatientPheX-A.jsonl (无标注,可选)")
    ap.add_argument("--out-dir", default="data/real", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train, train_stats = load_patientphe_x(args.train, subset="train")
    save_samples(train, os.path.join(args.out_dir, "train_standardized.jsonl"))
    print("[train]", json.dumps(train_stats, ensure_ascii=False, indent=2))

    if args.dev:
        dev, dev_stats = load_patientphe_x(args.dev, subset="dev")
        save_samples(dev, os.path.join(args.out_dir, "dev_standardized.jsonl"))
        print("[dev]", json.dumps(dev_stats, ensure_ascii=False, indent=2))

    with open(os.path.join(args.out_dir, "adapter_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"train": train_stats, "dev": dev_stats if args.dev else None},
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
