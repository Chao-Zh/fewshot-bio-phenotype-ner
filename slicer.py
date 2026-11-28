"""长文本滑动窗口切分。

解决 BioBERT max_position_embeddings=512 的上下文限制。
核心难点：切分后 BIO 标签必须保持合法，且实体不丢失/不错标。

策略：
    1. 先在全文上生成字符级 BIO 标签。
    2. 以 window_size 为窗口、stride 为步长滑动。
    3. 对每个窗口修正边界处的非法 BIO：
       - 窗口首的 I-X（实体在窗口外开始）→ 转为 O
       - 窗口尾被截断的实体（B 在窗内、I 超出）→
         保留可见片段并标记 truncated，或按配置丢弃转 O
    4. 每个原始实体的 B 位置必然落在某个窗口内（stride<=window_size），
       因此保证实体至少被一个窗口覆盖（覆盖率验证见 verify_coverage）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .cleaner import bio_to_entities, entities_to_bio, validate_bio
from .config import Config
from .schema import Entity, Sample


@dataclass
class Window:
    """一个切分后的窗口。"""
    sample_id: str
    window_id: int
    text: str
    tags: List[str]                       # 与 text 等长的字符级 BIO
    char_start: int                       # 在原文本中的起始字符位置
    char_end: int
    entities: List[Entity] = field(default_factory=list)
    is_truncated: bool = False

    def to_sample(self) -> Sample:
        return Sample(
            id=f"{self.sample_id}#w{self.window_id}",
            text=self.text,
            entities=self.entities,
            source=self.sample_id,
            metadata={
                "char_start": self.char_start,
                "char_end": self.char_end,
                "window_id": self.window_id,
                "is_truncated": self.is_truncated,
            },
        )


class Slicer:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.stats: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def slice_all(self, samples: List[Sample]) -> Tuple[List[Window], Dict]:
        self.stats = {
            "input_samples": 0,
            "output_windows": 0,
            "windows_dropped_empty": 0,
            "truncated_windows": 0,
            "fixed_leading_I": 0,
            "fixed_trailing_truncated": 0,
            "repaired_bio_windows": 0,
            "samples_sliced": 0,
        }
        windows: List[Window] = []
        for s in samples:
            self.stats["input_samples"] += 1
            if s.char_len <= self.cfg.window_size:
                # 无需切分，直接作为一个窗口
                tags = entities_to_bio(len(s.text), s.entities)
                ok, issues = validate_bio(tags)
                if not ok:
                    self.stats["repaired_bio_windows"] += 1
                w = Window(
                    sample_id=s.id,
                    window_id=0,
                    text=s.text,
                    tags=tags,
                    char_start=0,
                    char_end=len(s.text),
                    entities=bio_to_entities(tags, s.text),
                )
                if self.cfg.keep_empty_windows or w.entities:
                    windows.append(w)
                    self.stats["samples_sliced"] += 1
                else:
                    self.stats["windows_dropped_empty"] += 1
                continue

            # 需要切分
            self.stats["samples_sliced"] += 1
            ws = 0
            wid = 0
            n = len(s.text)
            tags_full = entities_to_bio(n, s.entities)
            while ws < n:
                we = min(ws + self.cfg.window_size, n)
                win_tags = list(tags_full[ws:we])
                win_text = s.text[ws:we]

                # --- 边界 BIO 修复 ---
                # (a) 窗口首连续的 I-X → O（实体在窗口外开始，可能跨多个 token）
                k = 0
                while k < len(win_tags) and win_tags[k].startswith("I-"):
                    win_tags[k] = "O"
                    self.stats["fixed_leading_I"] += 1
                    k += 1

                # (b) 尾部截断实体
                truncated = False
                for k in range(len(win_tags) - 1, -1, -1):
                    t = win_tags[k]
                    if t.startswith("I-"):
                        # 找到其 B
                        label = t[2:]
                        j = k - 1
                        while j >= 0 and win_tags[j] == f"I-{label}":
                            j -= 1
                        if j >= 0 and win_tags[j] == f"B-{label}":
                            truncated = True
                            if not self.cfg.keep_truncated_entities:
                                # 丢弃整个截断实体
                                for m in range(j, len(win_tags)):
                                    if win_tags[m] in (f"B-{label}", f"I-{label}"):
                                        win_tags[m] = "O"
                                self.stats["fixed_trailing_truncated"] += 1
                            break
                        else:
                            # 没有 B（理论上已被 (a) 处理），转 O
                            win_tags[k] = "O"
                    elif t.startswith("B-"):
                        # B 后面没有 I，且它是最后一个 token → 单字实体，合法
                        break

                # 校验修复后 BIO
                ok, _ = validate_bio(win_tags)
                if not ok:
                    self.stats["repaired_bio_windows"] += 1
                    win_tags = self._force_valid(win_tags)

                win_entities = bio_to_entities(win_tags, win_text)
                # 将原始实体的附加信息(raw_label=HPO ID 等)按字符偏移映射回窗口实体。
                # bio_to_entities 只能恢复 BIO 类别, 丢失了这些任务特有字段。
                self._restore_entity_metadata(win_entities, ws, s.entities)
                # 标记截断
                if truncated and self.cfg.keep_truncated_entities:
                    for e in win_entities:
                        if e.end == len(win_tags):
                            e.truncated = True

                w = Window(
                    sample_id=s.id,
                    window_id=wid,
                    text=win_text,
                    tags=win_tags,
                    char_start=ws,
                    char_end=we,
                    entities=win_entities,
                    is_truncated=truncated,
                )
                if self.cfg.keep_empty_windows or w.entities:
                    windows.append(w)
                    self.stats["output_windows"] += 1
                    if truncated:
                        self.stats["truncated_windows"] += 1
                else:
                    self.stats["windows_dropped_empty"] += 1

                if we >= n:
                    break
                ws += self.cfg.stride
                wid += 1

        self.stats["output_windows"] = len(windows)
        return windows, self.stats

    # ------------------------------------------------------------------
    @staticmethod
    def _restore_entity_metadata(
        win_entities: List[Entity], ws: int, orig_entities: List[Entity]
    ) -> None:
        """把原文实体的 raw_label 等附加信息按绝对偏移写回窗口实体。

        窗口实体的 start/end 是窗口内相对坐标, 加上 ws 即原文绝对坐标。
        优先精确匹配 (start,end); 截断实体退化为按 start 匹配。
        """
        by_span = {(e.start, e.end): e for e in orig_entities}
        by_start: Dict[int, List[Entity]] = {}
        for e in orig_entities:
            by_start.setdefault(e.start, []).append(e)

        for we in win_entities:
            abs_span = (ws + we.start, ws + we.end)
            orig = by_span.get(abs_span)
            if orig is None:
                candidates = [
                    e for e in by_start.get(abs_span[0], [])
                    if e.start <= abs_span[0] and e.end >= abs_span[0]
                ]
                orig = candidates[0] if candidates else None
            if orig is not None and orig.raw_label:
                we.raw_label = orig.raw_label

    # ------------------------------------------------------------------
    @staticmethod
    def _force_valid(tags: List[str]) -> List[str]:
        """兜底：强制将 BIO 序列修复为合法（孤立的 I- 转 O）。"""
        out = list(tags)
        prev_label = None
        for i, t in enumerate(out):
            if t == "O":
                prev_label = None
                continue
            if "-" not in t:
                out[i] = "O"
                prev_label = None
                continue
            prefix, label = t.split("-", 1)
            if prefix == "B":
                prev_label = label
            elif prefix == "I":
                if prev_label != label:
                    out[i] = "O"
                else:
                    prev_label = label
            else:
                out[i] = "O"
                prev_label = None
        return out

    # ------------------------------------------------------------------
    def verify_coverage(self, samples: List[Sample], windows: List[Window]) -> Dict:
        """验证每个原始实体的 B 位置是否被某窗口覆盖。"""
        by_sample: Dict[str, List[Window]] = {}
        for w in windows:
            by_sample.setdefault(w.sample_id, []).append(w)

        total_entities = 0
        covered = 0
        examples: List[str] = []
        for s in samples:
            wins = by_sample.get(s.id, [])
            for e in s.entities:
                if not (0 <= e.start < len(s.text)):
                    continue
                total_entities += 1
                hit = any(w.char_start <= e.start < w.char_end for w in wins)
                if hit:
                    covered += 1
                elif len(examples) < 5:
                    examples.append(f"{s.id}: ({e.start},{e.end}) {e.text}")
        return {
            "total_entities": total_entities,
            "covered_entities": covered,
            "coverage_rate": round(covered / total_entities, 4) if total_entities else 1.0,
            "uncovered_examples": examples,
        }
