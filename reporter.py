"""样本统计报告生成（Markdown + JSON）。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from .analyzer import analyze


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _len_table(stats: Dict) -> str:
    return _md_table(
        ["min", "max", "mean", "median", "p90", "p95", "p99"],
        [[stats.get(k, 0) for k in ["min", "max", "mean", "median", "p90", "p95", "p99"]]],
    )


def build_report(context: Dict[str, Any]) -> str:
    """根据管线上下文构建 Markdown 报告。"""
    parts: List[str] = []
    parts.append("# 实验一：数据集分析与完整数据预处理工程 —— 样本统计报告")
    parts.append(f"\n> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append(f"\n> 原始输入：`{context.get('input_path', 'N/A')}`  ")
    parts.append(f"> 检测格式：`{context.get('input_format', 'N/A')}`  ")
    parts.append(f"> 随机种子：`{context.get('seed', 42)}`")

    # ---- 1. 原始数据分析 ----
    raw = context.get("raw_analysis")
    if raw:
        parts.append("\n## 1. 原始数据集结构分析")
        parts.append(
            _md_table(
                ["指标", "数值"],
                [
                    ["样本数", raw["num_samples"]],
                    ["实体总数", raw["num_entities"]],
                    ["实体类别数", raw["num_labels"]],
                    ["空文本样本数", raw["empty_texts"]],
                    ["非法 span 数", raw["invalid_spans"]],
                    ["重叠实体数", raw["overlapping_entities"]],
                    ["超过 510 字符的样本数", raw["samples_over_510_chars"]],
                ],
            )
        )
        parts.append("\n**文本长度分布（字符）：**\n")
        parts.append(_len_table(raw["text_len_stats"]))
        parts.append("\n**每句实体数分布：**\n")
        parts.append(_len_table(raw["entities_per_sample_stats"]))

    # ---- 2. 清洗统计 ----
    clean = context.get("clean_stats")
    if clean:
        parts.append("\n## 2. 清洗统计")
        rows = [[k, v] for k, v in sorted(clean.items(), key=lambda x: -x[1])]
        parts.append(_md_table(["清洗动作", "次数"], rows))

    # ---- 3. 标签体系与实体分布 ----
    clean_an = context.get("clean_analysis")
    if clean_an:
        parts.append("\n## 3. 标签体系与实体类别分布（清洗后）")
        dist_rows = []
        for d in clean_an["label_distribution"]:
            ls = d["entity_len_stats"]
            dist_rows.append([
                d["label"], d["entity_count"], d["sample_count"],
                ls["min"], ls["mean"], ls["max"],
                "、".join(d["examples"][:3]),
            ])
        parts.append(
            _md_table(
                ["类别", "实体数", "样本数", "最短", "平均长度", "最长", "示例"],
                dist_rows,
            )
        )
        if clean_an["top_cooccurrences"]:
            parts.append("\n**高频共现类别对：**\n")
            parts.append(
                _md_table(
                    ["类别A", "类别B", "共现次数"],
                    [[c["a"], c["b"], c["count"]] for c in clean_an["top_cooccurrences"]],
                )
            )

    # ---- 4. 滑动窗口切分 ----
    slicer = context.get("slicer_stats")
    coverage = context.get("coverage")
    if slicer:
        parts.append("\n## 4. 长文本滑动窗口切分")
        parts.append(
            _md_table(
                ["指标", "数值"],
                [[k, v] for k, v in slicer.items()],
            )
        )
        parts.append(
            f"\n- 窗口大小：`{context.get('window_size')}` tokens（max_seq_len={context.get('max_seq_len')} - special_tokens=2）"
        )
        parts.append(f"- 步长 stride：`{context.get('stride')}`，重叠 overlap：`{context.get('overlap')}`")
        if coverage:
            parts.append(
                f"\n**实体覆盖率：** {coverage['covered_entities']}/{coverage['total_entities']} "
                f"= **{coverage['coverage_rate'] * 100:.2f}%**"
            )
            if coverage["uncovered_examples"]:
                parts.append("\n未覆盖示例：\n")
                for ex in coverage["uncovered_examples"]:
                    parts.append(f"- `{ex}`")

    # ---- 5. 数据集划分 ----
    split = context.get("split_stats")
    if split:
        parts.append("\n## 5. 训练集/验证集/测试集划分")
        parts.append(
            _md_table(
                ["集合", "样本数"],
                [[k, v] for k, v in split["splits"].items()],
            )
        )
        lds = split["label_distribution_by_split"]
        labels = sorted({l for d in lds.values() for l in d})
        parts.append("\n**各类别在三集合中的实体数：**\n")
        parts.append(
            _md_table(
                ["类别"] + list(lds.keys()),
                [[l] + [lds[s].get(l, 0) for s in lds] for l in labels],
            )
        )

    # ---- 6. Episode 采样 ----
    ep = context.get("episode_stats")
    if ep:
        parts.append("\n## 6. 少样本 Episode 采样（Few-NERD, ACL2021）")
        parts.append(
            _md_table(
                ["配置/指标", "数值"],
                [
                    ["N-way", ep.get("n_way")],
                    ["K-shot", ep.get("k_shot")],
                    ["Q-query", ep.get("q_query")],
                    ["Episode 数", ep.get("num_episodes")],
                ]
                + [[f"可用类别数({k})", v["num_labels_available"]] for k, v in ep.get("per_split", {}).items()],
            )
        )
        if ep.get("sample_episode"):
            se = ep["sample_episode"]
            parts.append(f"\n**单个 episode 示例：** ways = `{se['ways']}`  ")
            parts.append(f"support = {se['support_size']} 条, query = {se['query_size']} 条")

    # ---- 7. Dataloader ----
    dl = context.get("dataloader_stats")
    if dl:
        parts.append("\n## 7. Dataloader 验证")
        parts.append(
            _md_table(
                ["集合", "样本数(feature)", "batch 数", "数值格式"],
                [[k, v["num_features"], v["num_batches"], v["backend"]] for k, v in dl.items()],
            )
        )

    # ---- 8. 输出清单 ----
    parts.append("\n## 8. 产出文件清单")
    for f in context.get("output_files", []):
        parts.append(f"- `{f}`")

    return "\n".join(parts)


def save_report(context: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    md = build_report(context)
    name = context.get("report_name", "sample_statistics_report.md")
    md_path = os.path.join(output_dir, name)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    # 同时输出 JSON 版本（便于程序读取）
    json_path = os.path.join(output_dir, "statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2, default=str)

    return md_path
