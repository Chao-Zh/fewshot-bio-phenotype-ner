"""训练集 / 验证集 / 测试集划分。

支持按主标签分层（stratify_by_label），保证各类别在三集合中分布相近。
不依赖 sklearn。
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

from .analyzer import analyze
from .config import Config
from .schema import DatasetBundle, Sample


def _primary_label(sample: Sample) -> str:
    """样本的主标签：优先取实体数最多的类别，平局取字面上最小的。"""
    if not sample.entities:
        return "<NO_ENTITY>"
    c = Counter(e.label for e in sample.entities if e.label)
    if not c:
        return "<NO_ENTITY>"
    return sorted(c.items(), key=lambda x: (-x[1], x[0]))[0][0]


def split_dataset(
    samples: List[Sample], cfg: Config, bundle_name: str = "sliced"
) -> DatasetBundle:
    """按比例划分数据集（可选分层）。"""
    rng = random.Random(cfg.seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    train_ratio, val_ratio = cfg.train_ratio, cfg.val_ratio
    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio 不能 > 1")

    if not cfg.stratify_by_label:
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    else:
        # 按主标签分组，组内按比例划分（大余数法处理取整）
        groups: Dict[str, List[int]] = defaultdict(list)
        for i in indices:
            groups[_primary_label(samples[i])].append(i)

        train_idx, val_idx, test_idx = [], [], []
        for label, group in sorted(groups.items()):
            n = len(group)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            # 小样本组保证每集合至少 1 条（若总量允许）
            if n >= 3 and n_val == 0:
                n_val = 1
            if n >= 3 and n - n_train - n_val == 0:
                n_train = max(1, n - n_val - 1)
            train_idx.extend(group[:n_train])
            val_idx.extend(group[n_train:n_train + n_val])
            test_idx.extend(group[n_train + n_val:])

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]

    label_set = sorted({e.label for s in samples for e in s.entities if e.label})

    # 各集合独立重新编号 id 前缀，避免窗口 #wN 冲突
    for prefix, subset in (("train", train), ("val", val), ("test", test)):
        for s in subset:
            s.metadata["split"] = prefix

    stats = {
        "stratified": cfg.stratify_by_label,
        "seed": cfg.seed,
        "splits": {
            "train": analyze(train, "train")["num_samples"],
            "val": analyze(val, "val")["num_samples"],
            "test": analyze(test, "test")["num_samples"],
        },
        "label_distribution_by_split": _label_dist_by_split(train, val, test, label_set),
    }

    if cfg.verbose:
        print(f"[splitter] 划分结果: train={len(train)}, val={len(val)}, test={len(test)}")

    return DatasetBundle(
        train=train, val=val, test=test, label_set=label_set, split_stats=stats
    )


def _label_dist_by_split(train, val, test, label_set) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for name, subset in (("train", train), ("val", val), ("test", test)):
        c = Counter(e.label for s in subset for e in s.entities)
        result[name] = {label: c.get(label, 0) for label in label_set}
    return result
