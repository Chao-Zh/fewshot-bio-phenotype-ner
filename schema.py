"""统一数据模型。

所有格式最终都被解析为 ``Sample``；清洗、切分、采样、加载各阶段
仅依赖这些数据结构,与原始格式解耦。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Entity:
    """一个实体标注(字符级 span,左闭右开)。"""

    start: int
    end: int
    label: str
    text: str = ""
    # 原始标签(映射前)
    raw_label: str = ""
    # 是否跨滑动窗口边界被截断
    truncated: bool = False
    # 是否由清洗阶段修复过
    fixed: bool = False

    def __post_init__(self):
        if not self.text and self.start >= 0 and self.end > self.start:
            # text 由外部填充;此处只做占位
            pass

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Entity") -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text": self.text,
            "raw_label": self.raw_label,
            "truncated": self.truncated,
            "fixed": self.fixed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            start=int(d.get("start", d.get("start_offset", d.get("begin", 0)))),
            end=int(d.get("end", d.get("end_offset", d.get("stop", 0)))),
            label=str(d.get("label", d.get("type", d.get("label_type", d.get("category", ""))))),
            text=str(d.get("text", d.get("entity_text", ""))),
            raw_label=str(d.get("raw_label", "")),
        )


@dataclass
class Sample:
    """一条样本(一个句子/段落)。"""

    id: str
    text: str
    entities: List[Entity] = field(default_factory=list)
    source: str = ""                      # 来源文件名/子集
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 清洗阶段记录
    cleaning_notes: List[str] = field(default_factory=list)

    @property
    def labels(self) -> List[str]:
        return sorted({e.label for e in self.entities if e.label})

    @property
    def char_len(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "source": self.source,
            "metadata": self.metadata,
            "cleaning_notes": self.cleaning_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(
            id=str(d.get("id", "")),
            text=str(d.get("text", "")),
            entities=[Entity.from_dict(e) for e in d.get("entities", [])],
            source=str(d.get("source", "")),
            metadata=dict(d.get("metadata", {})),
            cleaning_notes=list(d.get("cleaning_notes", [])),
        )


@dataclass
class Feature:
    """模型可直接消费的一条特征。"""

    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]                     # BIO id,-100 表示忽略(特殊 token / padding)
    token_type_ids: Optional[List[int]] = None
    offset_mapping: Optional[List[Tuple[int, int]]] = None  # token -> (char_start, char_end)
    sample_id: str = ""
    window_id: int = 0
    is_truncated: bool = False

    def to_dict(self) -> dict:
        d = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "sample_id": self.sample_id,
            "window_id": self.window_id,
            "is_truncated": self.is_truncated,
        }
        if self.offset_mapping is not None:
            d["offset_mapping"] = [list(x) for x in self.offset_mapping]
        if self.token_type_ids is not None:
            d["token_type_ids"] = self.token_type_ids
        return d


@dataclass
class Episode:
    """Few-NERD 风格 N-way K-shot episode。"""

    episode_id: int
    ways: List[str]                       # 本 episode 采样出的 N 个类别
    support: List[Sample]                 # N*K 条
    query: List[Sample]                   # N*Q 条
    # 每个类别对应的 support/query 样本 id,便于评测时按类统计
    support_by_way: Dict[str, List[str]] = field(default_factory=dict)
    query_by_way: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "ways": self.ways,
            "n_way": len(self.ways),
            "support_size": len(self.support),
            "query_size": len(self.query),
        }


@dataclass
class DatasetBundle:
    """划分后的数据集集合。"""

    train: List[Sample]
    val: List[Sample]
    test: List[Sample]
    label_set: List[str]
    split_stats: Dict[str, Any] = field(default_factory=dict)

    def all_samples(self) -> List[Sample]:
        return self.train + self.val + self.test
