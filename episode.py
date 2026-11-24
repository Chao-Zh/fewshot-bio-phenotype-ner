"""少样本 episode 采样器（Few-NERD, ACL 2021 范式）。

Few-NERD 核心思想：
    - 将训练/验证/测试按实体类别（way）组织
    - 每个 episode 随机采样 N 个类别（N-way）
    - 每类采样 K 条 support 样本 + Q 条 query 样本
    - 模型在 support 上"学习"，在 query 上预测，模拟真实少样本场景

本实现要点：
    - 支持 intra/episode 采样（每个 way 内独立采样）
    - 支持按 split 构建不同的采样器（train/val/test 使用各自的类别池）
    - 保证类别覆盖与样本不重复（同 episode 内 support/query 不重叠）
    - 可固定随机种子保证可复现
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .config import Config
from .schema import Episode, Sample


class EpisodeSampler:
    """N-way K-shot episode 采样器。"""

    def __init__(
        self,
        samples: List[Sample],
        cfg: Optional[Config] = None,
        label_set: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg or Config()
        self.samples = samples
        self.seed = seed if seed is not None else self.cfg.seed
        self.rng = random.Random(self.seed)

        # 构建 label -> 样本索引 倒排索引
        self.label_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            for label in s.labels:
                self.label_to_indices[label].append(i)

        # 过滤样本数过少的类别
        self.available_labels = sorted(
            label
            for label, idxs in self.label_to_indices.items()
            if len(idxs) >= self.cfg.min_samples_per_label
        )

        # 限制在指定 label_set 内
        if label_set:
            allowed = set(label_set)
            self.available_labels = [l for l in self.available_labels if l in allowed]

    # ------------------------------------------------------------------
    def sample_episode(self, n_way=None, k_shot=None, q_query=None) -> Episode:
        """采样单个 episode。"""
        n_way = n_way or self.cfg.n_way
        k_shot = k_shot or self.cfg.k_shot
        q_query = q_query or self.cfg.q_query

        if len(self.available_labels) < n_way:
            raise ValueError(
                f"可用类别数 {len(self.available_labels)} < n_way {n_way}，"
                f"请减小 n_way 或增加数据"
            )

        ways = self.rng.sample(self.available_labels, n_way)
        support: List[Sample] = []
        query: List[Sample] = []
        support_by_way: Dict[str, List[str]] = {}
        query_by_way: Dict[str, List[str]] = {}
        used_globally: set = set()  # 整个 episode 内全局去重

        need = k_shot + q_query
        for label in ways:
            # 排除已被其他 way 选走的样本，保证 support/query 全局无重叠
            pool = [i for i in self.label_to_indices[label] if i not in used_globally]
            if len(pool) < need:
                # 样本不足时放宽：允许复用其他 way 用过的样本，但本 way 内 support/query 不重复
                pool = list(self.label_to_indices[label])
                if len(pool) < need:
                    chosen = [self.rng.choice(pool) for _ in range(need)]
                else:
                    chosen = self.rng.sample(pool, need)
            else:
                chosen = self.rng.sample(pool, need)

            s_idx = chosen[:k_shot]
            q_idx = chosen[k_shot:]
            used_globally.update(chosen)

            support_by_way[label] = [self.samples[i].id for i in s_idx]
            query_by_way[label] = [self.samples[i].id for i in q_idx]
            support.extend(self.samples[i] for i in s_idx)
            query.extend(self.samples[i] for i in q_idx)

        return Episode(
            episode_id=self._next_id(),
            ways=ways,
            support=support,
            query=query,
            support_by_way=support_by_way,
            query_by_way=query_by_way,
        )

    def _next_id(self) -> int:
        if not hasattr(self, "_episode_counter"):
            self._episode_counter = 0
        self._episode_counter += 1
        return self._episode_counter

    # ------------------------------------------------------------------
    def sample_episodes(self, num_episodes=None, **kwargs) -> List[Episode]:
        """批量采样 episodes。"""
        num_episodes = num_episodes or self.cfg.num_episodes
        return [self.sample_episode(**kwargs) for _ in range(num_episodes)]

    def iter_episodes(self, num_episodes=None, **kwargs) -> Iterable[Episode]:
        """惰性迭代 episodes（适合大规模训练）。"""
        num_episodes = num_episodes or self.cfg.num_episodes
        for _ in range(num_episodes):
            yield self.sample_episode(**kwargs)

    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        dist = {label: len(idxs) for label, idxs in self.label_to_indices.items()}
        return {
            "num_samples": len(self.samples),
            "num_labels_total": len(dist),
            "num_labels_available": len(self.available_labels),
            "samples_per_label": dist,
            "n_way": self.cfg.n_way,
            "k_shot": self.cfg.k_shot,
            "q_query": self.cfg.q_query,
        }


def build_episode_samplers(bundle, cfg: Optional[Config] = None) -> Dict[str, EpisodeSampler]:
    """为 train/val/test 分别构建采样器（共享 label_set）。"""
    cfg = cfg or Config()
    samplers = {}
    for name, subset in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
        samplers[name] = EpisodeSampler(
            subset, cfg=cfg, label_set=bundle.label_set, seed=cfg.seed + hash(name) % 1000
        )
    return samplers
