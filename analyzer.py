"""数据集结构 / 标签体系 / 实体类别分布分析。"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from .schema import Sample


def _percentile(sorted_vals: List[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def length_stats(values: List[int]) -> Dict[str, float]:
    s = sorted(values)
    if not s:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "p99": 0}
    return {
        "min": s[0],
        "max": s[-1],
        "mean": round(sum(s) / len(s), 2),
        "median": round(_percentile(s, 0.5), 1),
        "p90": round(_percentile(s, 0.9), 1),
        "p95": round(_percentile(s, 0.95), 1),
        "p99": round(_percentile(s, 0.99), 1),
    }


def analyze(samples: List[Sample], name: str = "dataset") -> Dict[str, Any]:
    """对样本集合做全面统计分析。"""
    n = len(samples)
    text_lens = [s.char_len for s in samples]
    ent_counts = [len(s.entities) for s in samples]

    label_counter: Counter = Counter()
    ent_len_by_label: Dict[str, List[int]] = defaultdict(list)
    samples_per_label: Dict[str, set] = defaultdict(set)
    cooccurrence: Counter = Counter()

    invalid_spans = 0
    empty_texts = 0
    overlapping = 0
    label_text_examples: Dict[str, List[str]] = defaultdict(list)

    for s in samples:
        if not s.text.strip():
            empty_texts += 1
        seen = set()
        for e in s.entities:
            label_counter[e.label] += 1
            seen.add(e.label)
            ent_len_by_label[e.label].append(e.length)
            samples_per_label[e.label].add(s.id)
            if e.text and len(label_text_examples[e.label]) < 5:
                label_text_examples[e.label].append(e.text)
            if e.start < 0 or e.end > len(s.text) or e.end <= e.start:
                invalid_spans += 1
        # 实体重叠统计
        ents = sorted([e for e in s.entities if e.end > e.start], key=lambda x: x.start)
        for i in range(1, len(ents)):
            if ents[i].start < ents[i - 1].end:
                overlapping += 1
        for a in seen:
            for b in seen:
                if a < b:
                    cooccurrence[(a, b)] += 1

    label_dist = []
    for label, count in label_counter.most_common():
        lens = ent_len_by_label[label]
        label_dist.append({
            "label": label,
            "entity_count": count,
            "sample_count": len(samples_per_label[label]),
            "entity_len_stats": length_stats(lens),
            "examples": label_text_examples[label],
        })

    # 超过 BioBERT 512 长度的样本数
    over_long = sum(1 for l in text_lens if l > 510)

    return {
        "name": name,
        "num_samples": n,
        "num_entities": sum(ent_counts),
        "num_labels": len(label_counter),
        "empty_texts": empty_texts,
        "invalid_spans": invalid_spans,
        "overlapping_entities": overlapping,
        "text_len_stats": length_stats(text_lens),
        "entities_per_sample_stats": length_stats(ent_counts),
        "samples_over_510_chars": over_long,
        "label_distribution": label_dist,
        "label_counter": dict(label_counter),
        "top_cooccurrences": [
            {"a": a, "b": b, "count": c}
            for (a, b), c in cooccurrence.most_common(15)
        ],
    }


def infer_label_set(samples: List[Sample]) -> List[str]:
    """从数据推断标签集合(按首次出现顺序)。"""
    labels: List[str] = []
    seen = set()
    for s in samples:
        for e in s.entities:
            if e.label and e.label not in seen:
                seen.add(e.label)
                labels.append(e.label)
    return sorted(labels)
