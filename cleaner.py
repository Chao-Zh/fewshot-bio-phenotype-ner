"""清洗模块：BIO 标签校验、脏数据清洗、特殊医学符号过滤。

清洗动作（按顺序）：
    1. 文本控制字符 / 不可见字符过滤（保留医学符号白名单）
    2. 空白规范化
    3. 标签别名映射 + BMES → BIO 归一化
    4. 实体 span 校验与修复（越界裁剪、负长度丢弃、surface text 校正）
    5. 嵌套/重叠实体消解
    6. 空文本 / 空实体样本处理

每条样本记录 cleaning_notes，便于追踪与统计。
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .config import Config
from .schema import Entity, Sample


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class Cleaner:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------
    def clean(self, samples: List[Sample]) -> Tuple[List[Sample], Dict]:
        """执行清洗,返回 (清洗后样本, 清洗统计)。"""
        self.stats = Counter()
        output: List[Sample] = []

        for s in samples:
            notes: List[str] = []
            text = s.text

            # 1) 控制字符 / 医学符号过滤
            text = self._normalize_text(text, notes)

            # 2) 空文本
            if self.cfg.drop_empty_text and not text.strip():
                self.stats["dropped_empty_text"] += 1
                continue

            # 3) 实体清洗
            new_entities: List[Entity] = []
            for e in s.entities:
                cleaned = self._clean_entity(e, text, notes)
                if cleaned is not None:
                    new_entities.append(cleaned)

            # 4) 重叠消解
            if self.cfg.resolve_overlap and len(new_entities) > 1:
                new_entities = self._resolve_overlaps(new_entities, notes)

            # 5) 空实体样本
            if self.cfg.drop_empty_entities and not new_entities:
                self.stats["dropped_empty_entities"] += 1
                continue

            # 重建 Sample（保留 id/source/metadata）
            s.text = text
            s.entities = new_entities
            s.cleaning_notes = notes
            output.append(s)

        return output, dict(self.stats)

    # ------------------------------------------------------------------
    def _normalize_text(self, text: str, notes: List[str]) -> str:
        original_len = len(text)

        # Unicode NFKC 规范化（全角→半角等），但保留中文语义
        # 注意：NFKC 会改变字符长度，需同步修复 span；这里仅在长度不变时应用
        nfkc = unicodedata.normalize("NFKC", text)
        if len(nfkc) == len(text):
            text = nfkc

        # 控制字符过滤（保留白名单符号）
        if self.cfg.strip_control_chars:
            removed = []
            chars = []
            for ch in text:
                if _CTRL_RE.match(ch) or unicodedata.category(ch) in ("Cc", "Cf"):
                    if ch not in self.cfg.medical_symbol_whitelist:
                        removed.append(repr(ch))
                        continue
                if ch in self.cfg.chars_to_remove and ch not in self.cfg.medical_symbol_whitelist:
                    removed.append(repr(ch))
                    continue
                chars.append(ch)
            text = "".join(chars)
            if removed:
                self.stats["removed_control_chars"] += len(removed)
                notes.append(f"过滤控制字符 {len(removed)} 个")

        # 空白规范化：连续空白压成单空格（中文间空格可保留语义,谨慎处理）
        # 注意：若 offset 已按原文精确对齐(如 PatientPheX), 压缩空白会改变
        # 文本长度并导致所有下游实体 span 漂移, 此时应关闭。
        if self.cfg.normalize_whitespace:
            new_text = re.sub(r"[ \t\r\f\v]+", " ", text)
            new_text = re.sub(r"\n+", " ", new_text).strip()
            if new_text != text:
                self.stats["normalized_whitespace"] += 1
                notes.append("空白规范化")
                text = new_text

        if len(text) != original_len:
            self.stats["text_length_changed"] += 1
        return text

    # ------------------------------------------------------------------
    def _clean_entity(self, e: Entity, text: str, notes: List[str]) -> Optional[Entity]:
        # 标签别名 + BMES → BIO
        label = e.label.strip()
        if not label:
            self.stats["dropped_no_label"] += 1
            return None

        if label in self.cfg.label_alias:
            label = self.cfg.label_alias[label]
            e.fixed = True

        # BMES 标签体系里的前缀归一（如 "S-疾病" → "疾病"，实体级只需类别）
        m = re.match(r"^[BIESM][-_](.+)$", label)
        if m and self.cfg.normalize_bmes_to_bio:
            label = m.group(1)
            e.fixed = True
            self.stats["normalized_bmes_label"] += 1

        e.label = label
        if e.raw_label and e.raw_label != label:
            pass  # raw_label 保留原始信息
        else:
            e.raw_label = e.raw_label or label

        # span 校验
        start, end = e.start, e.end
        if end <= start:
            self.stats["dropped_invalid_span"] += 1
            return None
        if start < 0 or end > len(text):
            old = (start, end)
            start = max(0, start)
            end = min(len(text), end)
            if end <= start:
                self.stats["dropped_out_of_range"] += 1
                return None
            e.fixed = True
            self.stats["clamped_span"] += 1
            notes.append(f"span 越界裁剪 {old}→({start},{end})")

        e.start, e.end = start, end

        # surface text 与 span 对齐（span 为准）
        span_text = text[start:end]
        if e.text and e.text != span_text:
            self.stats["corrected_entity_text"] += 1
            e.fixed = True
        e.text = span_text

        # 过滤纯空白实体
        if not e.text.strip():
            self.stats["dropped_whitespace_entity"] += 1
            return None

        return e

    # ------------------------------------------------------------------
    def _resolve_overlaps(self, entities: List[Entity], notes: List[str]) -> List[Entity]:
        """嵌套/交叉实体消解：按 (长度降序, start 升序) 贪心保留。"""
        entities = sorted(entities, key=lambda x: (-x.length, x.start))
        kept: List[Entity] = []
        for e in entities:
            conflict = False
            for k in kept:
                if e.overlaps(k):
                    conflict = True
                    self.stats["resolved_overlap"] += 1
                    e.fixed = True
                    break
            if not conflict:
                kept.append(e)
        if self.stats["resolved_overlap"]:
            notes.append("消解重叠实体")
        return sorted(kept, key=lambda x: x.start)


# ---------------------------------------------------------------------------
# BIO 序列校验（token 级）
# ---------------------------------------------------------------------------
def validate_bio(tags: List[str]) -> Tuple[bool, List[str]]:
    """校验 BIO 序列合法性。

    规则：
      - I-X 必须紧跟在 B-X 或 I-X 之后
      - B-X 后若接 I-Y (Y≠X) 视为非法
      - 序列首不能是 I-
    返回 (is_valid, issues)
    """
    issues: List[str] = []
    prev_label: Optional[str] = None
    for i, tag in enumerate(tags):
        if tag == "O":
            prev_label = None
            continue
        if "-" not in tag:
            issues.append(f"pos {i}: 非法标签格式 {tag}")
            prev_label = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B":
            prev_label = label
        elif prefix == "I":
            if prev_label != label:
                issues.append(f"pos {i}: I-{label} 前不是 B-{label}/I-{label}")
            prev_label = label
        elif prefix in ("S", "E", "M"):
            # BMES 体系允许
            prev_label = None
        else:
            issues.append(f"pos {i}: 未知前缀 {prefix}")
            prev_label = None
    return (len(issues) == 0), issues


def entities_to_bio(text_len: int, entities: List[Entity]) -> List[str]:
    """从实体列表生成字符级 BIO 标签（供切分/建模使用）。"""
    tags = ["O"] * text_len
    # 非法 span 直接跳过
    valid = [e for e in entities if 0 <= e.start < e.end <= text_len]
    valid.sort(key=lambda x: (x.start, -x.length))
    occupied = set()
    for e in valid:
        idxs = set(range(e.start, e.end))
        if idxs & occupied:
            continue  # 已消解重叠时不应发生
        occupied |= idxs
        tags[e.start] = f"B-{e.label}"
        for k in range(e.start + 1, e.end):
            tags[k] = f"I-{e.label}"
    return tags


def bio_to_entities(tags: List[str], text: str) -> List[Entity]:
    """从 BIO 标签还原实体（用于校验切分正确性）。"""
    entities: List[Entity] = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        prefix, label = tag.split("-", 1)
        if prefix != "B":
            i += 1
            continue
        j = i + 1
        while j < len(tags) and tags[j] == f"I-{label}":
            j += 1
        entities.append(Entity(start=i, end=j, label=label, text=text[i:j]))
        i = j
    return entities
