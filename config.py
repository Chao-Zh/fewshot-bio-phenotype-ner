"""配置与超参数。

所有参数均可通过构造 ``Config`` 或直接修改默认值来覆盖。
字段附带说明,便于复现实验。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class Config:
    # ---------------- 基础 ----------------
    seed: int = 42
    verbose: bool = True

    # ---------------- 输入格式 ----------------
    # "auto" | "json" | "bio"；auto 会抽样探测
    input_format: str = "auto"
    # JSON 中文本字段名候选(按优先级)
    json_text_keys: List[str] = field(
        default_factory=lambda: ["text", "content", "sentence", "原文", "title+text"]
    )
    # JSON 中实体列表字段名候选
    json_entities_keys: List[str] = field(
        default_factory=lambda: ["entities", "entity", "labels", "spans", "标注结果"]
    )

    # ---------------- 标签体系 ----------------
    # 合法实体类别;为空表示从数据自动推断
    label_set: List[str] = field(default_factory=list)
    # 外层标签映射(如天池原始大类 -> 统一类别),为空则不映射
    label_alias: Dict[str, str] = field(default_factory=dict)

    # ---------------- 清洗 ----------------
    # 是否将 B/M/E/S 中标签归一为 BIO
    normalize_bmes_to_bio: bool = True
    # 是否过滤控制字符 / 不可见字符
    strip_control_chars: bool = True
    # 是否压缩连续空白为单空格。对于实体 offset 已按原始文本精确对齐的
    # 数据集(如 PatientPheX 的全局 offset), 开启会改变文本长度导致 offset
    # 漂移, 应设为 False。
    normalize_whitespace: bool = True
    # 是否移除实体重叠中置信度低的(保留更长/更早的)
    resolve_overlap: bool = True
    # 是否丢弃完全无实体的句子
    drop_empty_entities: bool = False
    # 是否丢弃空文本
    drop_empty_text: bool = True
    # 自定义需要剥离的特殊医学/噪声符号(逐字符)
    chars_to_remove: str = "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f"
    # 保留的医学符号白名单(这些符号不被当作噪声过滤)
    medical_symbol_whitelist: str = "±×÷≥≤≈→←↑↓℃℉%‰°μ～-—/·:：,，.。()（）[]【】+-"

    # ---------------- 滑动窗口 ----------------
    # BioBERT max_position_embeddings = 512;预留 [CLS]/[SEP] 2 个位置
    max_seq_len: int = 512
    special_tokens: int = 2
    # 窗口步长;None 表示 = window_size (无重叠)
    stride: Optional[int] = None
    # 重叠 token 数(与 stride 二选一,设置后 stride = window_size - overlap)
    overlap: int = 128
    # 切分粒度: "char" (中文按字) 或 "word" (按空白切词)
    tokenize_level: str = "char"
    # 窗口内是否保留无实体的片段
    keep_empty_windows: bool = False
    # 跨窗口被截断的实体是否保留其片段 (otherwise 丢弃不完整片段)
    keep_truncated_entities: bool = True

    # ---------------- 划分 ----------------
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test = 1 - train - val
    stratify_by_label: bool = True

    # ---------------- 少样本 episode ----------------
    n_way: int = 5
    k_shot: int = 5
    q_query: int = 5
    num_episodes: int = 100
    # 每类中至少要求的样本数
    min_samples_per_label: int = 1

    # ---------------- Dataloader ----------------
    batch_size: int = 16
    shuffle: bool = True
    pad_to_multiple_of: int = 8
    # 使用 torch;若环境无 torch 自动回退到 numpy DataLoader
    use_torch: bool = True

    # ---------------- 报告 ----------------
    report_name: str = "sample_statistics_report.md"

    def __post_init__(self):
        if self.stride is None:
            self.stride = self.window_size - self.overlap
        if self.stride <= 0:
            raise ValueError("stride 必须 > 0,请减小 overlap")
        if self.window_size <= self.special_tokens:
            raise ValueError("window_size 必须 > special_tokens")

    @property
    def window_size(self) -> int:
        """每个窗口可容纳的真实 token 数。"""
        return self.max_seq_len - self.special_tokens

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
