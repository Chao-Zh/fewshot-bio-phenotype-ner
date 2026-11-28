"""数据格式自动检测与统一解析。

支持：
    1) JSON Lines: 每行 {"text": ..., "entities": [{"start":, "end":, "label":}, ...]}
       - 兼容 start_offset/end_offset/label_type 等天池常见字段别名
       - entities 也支持 dict 形式 {label: [[start,end], ...]}
    2) BIO / CONLL: 每行 "字\\t标签",句子间空行分隔;也支持 "字 标签" 空格分隔
       - 兼容 B-X/I-X/O,以及 BMES、[] 包裹的实体标注
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .config import Config
from .schema import Entity, Sample


# ---------------------------------------------------------------------------
# 自动检测
# ---------------------------------------------------------------------------
def detect_format(path: str, cfg: Optional[Config] = None) -> str:
    """抽样探测文件格式。返回 'json' 或 'bio'。"""
    cfg = cfg or Config()
    if cfg.input_format in ("json", "bio"):
        return cfg.input_format

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(line)
            if len(samples) >= 20:
                break
    if not samples:
        return "json"

    json_votes = sum(1 for s in samples if s.startswith("{") or s.startswith("["))
    bio_votes = 0
    for s in samples:
        parts = re.split(r"\s+", s)
        if len(parts) >= 2:
            tag = parts[-1]
            if tag == "O" or re.match(r"^[BIESM][-_/.]?\S*", tag):
                bio_votes += 1

    if json_votes >= bio_votes and json_votes > 0:
        return "json"
    if bio_votes > 0:
        return "bio"
    # 兜底:看扩展名
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl", ".ndjson"):
        return "json"
    return "bio"


# ---------------------------------------------------------------------------
# JSON Lines
# ---------------------------------------------------------------------------
def _first_present(d: dict, keys: Iterable[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def parse_entities_json(raw, text: str) -> List[Entity]:
    """将 JSON 中的实体字段统一解析为 Entity 列表。"""
    entities: List[Entity] = []

    if raw is None:
        return entities

    # 情形 1: dict {label: [[s,e],...]}
    if isinstance(raw, dict):
        for label, spans in raw.items():
            if isinstance(spans, (int, float)):
                spans = [spans]
            if not isinstance(spans, (list, tuple)):
                continue
            for sp in spans:
                if isinstance(sp, dict):
                    entities.append(_entity_from_dict(sp, text, override_label=label))
                elif isinstance(sp, (list, tuple)) and len(sp) >= 2:
                    entities.append(_make_entity(int(sp[0]), int(sp[1]), label, text))
        return entities

    if not isinstance(raw, (list, tuple)):
        return entities

    for item in raw:
        if isinstance(item, dict):
            entities.append(_entity_from_dict(item, text))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            # [start, end, label]
            entities.append(_make_entity(int(item[0]), int(item[1]), str(item[2]), text))
    return entities


def _entity_from_dict(d: dict, text: str, override_label: Optional[str] = None) -> Entity:
    start = int(_first_present(d, ["start", "start_offset", "begin", "begin_offset", "s"], 0))
    end = int(_first_present(d, ["end", "end_offset", "stop", "end_pos", "e"], 0))
    label = override_label or str(
        _first_present(d, ["label", "type", "label_type", "category", "entity_type", "name"], "")
    )
    # 兼容 {"span": [s,e]}
    if end <= start and "span" in d and isinstance(d["span"], (list, tuple)):
        start, end = int(d["span"][0]), int(d["span"][1])
    ent_text = str(_first_present(d, ["text", "entity_text", "word", "mention"], ""))
    return _make_entity(start, end, label, text, ent_text)


def _make_entity(start: int, end: int, label: str, text: str, ent_text: str = "") -> Entity:
    if not ent_text and 0 <= start < end <= len(text):
        ent_text = text[start:end]
    return Entity(start=start, end=end, label=label, text=ent_text)


def load_json(path: str, cfg: Optional[Config] = None) -> List[Sample]:
    """读取 JSON 文件,兼容三种布局:
    1) JSON Lines (逐行 JSON 对象)
    2) 整个文件是一个 JSON 数组
    3) 一个 JSON 对象,其中某个字段是样本数组(如 {"data": [...]})
    """
    cfg = cfg or Config()
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    stripped = content.strip()
    # 情形 2 / 3: 整体是一个合法 JSON
    try:
        whole = json.loads(stripped)
        objs = _unwrap_sample_array(whole, cfg)
        if objs is not None:
            for i, o in enumerate(objs):
                samples.append(_json_obj_to_sample(o, cfg, str(i)))
            return samples
    except json.JSONDecodeError:
        pass

    # 情形 1: JSON Lines
    for lineno, line in enumerate(stripped.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list):
            for i, o in enumerate(obj):
                samples.append(_json_obj_to_sample(o, cfg, f"{lineno}.{i}"))
        elif isinstance(obj, dict):
            samples.append(_json_obj_to_sample(obj, cfg, str(lineno)))
    return samples


def _unwrap_sample_array(obj, cfg: Config):
    """若 obj 是样本数组/含样本数组的对象,返回该数组,否则 None。"""
    if isinstance(obj, list):
        if not obj:
            return []
        # 判断元素是否像样本(含文本字段)
        first = obj[0]
        if isinstance(first, dict) and any(k in first for k in cfg.json_text_keys):
            return obj
        return None
    if isinstance(obj, dict):
        for key in ("data", "samples", "train", "annotations", "items", "examples", "records"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    return None


def _is_standard_sample(obj: dict) -> bool:
    """判断是否为管线自身 ``Sample.to_dict()`` 输出的标准化格式。

    特征: 含 id/text/entities, 且 entities 元素带 start/end(而非原始 offset)。
    这类对象必须用 Sample.from_dict 精确还原, 否则 raw_label(HPO ID) 和
    metadata 会在通用解析中丢失(通用 parse_entities_json 不读这些字段)。
    """
    if not (isinstance(obj, dict) and "id" in obj and "text" in obj and "entities" in obj):
        return False
    ents = obj.get("entities")
    if isinstance(ents, list):
        for e in ents[:3]:
            if isinstance(e, dict) and "start" in e and "end" in e:
                return True
        if not ents:
            # 空实体列表: 再看是否有 source/metadata 字段佐证
            return "source" in obj or "metadata" in obj
    return False


def _json_obj_to_sample(obj: dict, cfg: Config, idx: str) -> Sample:
    # 优先走标准格式精确还原(保留 raw_label / metadata / cleaning_notes)
    if _is_standard_sample(obj):
        return Sample.from_dict(obj)

    text = _first_present(obj, cfg.json_text_keys, "")
    if text is None:
        text = ""
    if isinstance(text, (list, tuple)):
        text = "".join(str(t) for t in text)
    text = str(text)

    raw_entities = _first_present(obj, cfg.json_entities_keys, [])
    entities = parse_entities_json(raw_entities, text)

    sid = str(_first_present(obj, ["id", "sample_id", "uid", "guid"], f"json-{idx}"))
    source = str(_first_present(obj, ["source", "file", "dataset"], ""))
    metadata = {k: v for k, v in obj.items() if k not in cfg.json_text_keys + cfg.json_entities_keys + ["id"]}
    return Sample(id=sid, text=text, entities=entities, source=source, metadata=metadata)


# ---------------------------------------------------------------------------
# BIO / CONLL
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"^([BIESM])(?:[-_/.](.+))?$")


def load_bio(path: str, cfg: Optional[Config] = None) -> List[Sample]:
    """读取 BIO/CONLL 序列标注文件。"""
    samples: List[Sample] = []
    cur_tokens: List[str] = []
    cur_tags: List[str] = []
    sid = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                if cur_tokens:
                    samples.append(_bio_to_sample(cur_tokens, cur_tags, f"bio-{sid}"))
                    sid += 1
                    cur_tokens, cur_tags = [], []
                continue
            # -DOCSTART- 忽略
            if raw.startswith("-DOCSTART-"):
                continue
            parts = re.split(r"\s+", raw.strip())
            if len(parts) < 2:
                continue
            token, tag = parts[0], parts[-1]
            cur_tokens.append(token)
            cur_tags.append(tag)
        if cur_tokens:
            samples.append(_bio_to_sample(cur_tokens, cur_tags, f"bio-{sid}"))

    return samples


def _bio_to_sample(tokens: List[str], tags: List[str], sid: str) -> Sample:
    # 逐 token 拼接文本；中文按字拼接,英文词之间加空格
    text = ""
    char_spans: List[Tuple[int, int]] = []
    for t in tokens:
        start = len(text)
        text += t
        end = len(text)
        char_spans.append((start, end))

    entities: List[Entity] = []
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        m = _TAG_RE.match(tag)
        if not m or tag == "O":
            i += 1
            continue
        prefix, label = m.group(1), m.group(2) or "ENTITY"

        if prefix in ("S",):
            entities.append(_make_entity(char_spans[i][0], char_spans[i][1], label, text))
            i += 1
            continue

        # B / M / E / I 开头：向后收集同标签连续片段
        j = i + 1
        while j < n:
            m2 = _TAG_RE.match(tags[j])
            if not m2:
                break
            p2, l2 = m2.group(1), m2.group(2) or "ENTITY"
            if l2 != label:
                break
            if p2 in ("E", "S"):
                j += 1
                break
            if p2 in ("B", "O"):
                break
            j += 1
        start = char_spans[i][0]
        end = char_spans[j - 1][1]
        entities.append(_make_entity(start, end, label, text))
        i = j

    return Sample(id=sid, text=text, entities=entities)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def load_dataset(path: str, cfg: Optional[Config] = None, fmt: Optional[str] = None) -> List[Sample]:
    """自动检测并加载数据集。"""
    cfg = cfg or Config()
    if fmt is None:
        fmt = detect_format(path, cfg)
    if cfg.verbose:
        print(f"[formats] 检测到格式: {fmt}  <- {path}")
    if fmt == "json":
        samples = load_json(path, cfg)
    elif fmt == "bio":
        samples = load_bio(path, cfg)
    else:
        raise ValueError(f"未知格式: {fmt}")
    return samples


def save_jsonl(samples: List[Sample], path: str) -> None:
    """保存为标准 JSON Lines。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def save_bio(samples: List[Sample], path: str) -> None:
    """保存为 BIO 格式(逐字)。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            tags = ["O"] * len(s.text)
            for ent in s.entities:
                if not (0 <= ent.start < ent.end <= len(s.text)):
                    continue
                tags[ent.start] = f"B-{ent.label}"
                for k in range(ent.start + 1, ent.end):
                    tags[k] = f"I-{ent.label}"
            for ch, tag in zip(s.text, tags):
                f.write(f"{ch}\t{tag}\n")
            f.write("\n")
