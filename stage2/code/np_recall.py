# -*- coding: utf-8 -*-
"""名词短语粗召回层（无外部依赖，对应 PDF 风险预案「名词短语粗召回」兜底候选源）。

动机：纯 HPO 词典只覆盖标准术语，金标里大量专家自由书写的临床短语（长度、措辞常超出 HPO name/synonym）
召回不到。本层在词典命中之外补一类「临床名词短语」候选：

- head 词表：由 HPO term 名称的末词派生（频次过滤）+ 少量人工临床头名词；
- 围绕每个 head 出现处，向左右扩展「内容词」（允许 of 等介词在内部），以标点/停用词/动词为界，
  限制最大词数与字符数，得到短语 span；
- 短语的 HPO 映射：继承落在短语内、词典匹配到的内层 HPO id（取并集，primary 取最深）。

这样既抬高召回（覆盖长短语），又尽量保留 ID 映射（供阶段三边界收紧 + 重排序），且不依赖金标 surface。
"""
from __future__ import annotations

import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")
# 短语内不得跨过的硬边界（逗号/分号/句号/括号/斜杠/换行/等号）
HARD_GAP = re.compile(r"[,;:.()\[\]{}/\n\r\t|=]")

STOP = {
    "the", "a", "an", "and", "or", "but", "nor", "so", "that", "which", "who",
    "whom", "whose", "this", "these", "those", "it", "its", "they", "them",
    "their", "he", "she", "his", "her", "him", "we", "our", "us", "was", "were",
    "is", "are", "be", "been", "being", "has", "have", "had", "do", "does",
    "did", "can", "could", "may", "might", "must", "should", "will", "would",
    "with", "without", "in", "on", "at", "to", "for", "by", "from", "as",
    "because", "however", "also", "respectively", "showed", "revealed",
    "presented", "diagnosed", "treated", "associated", "characterized",
    "caused", "including", "include", "includes", "due", "resulting", "after",
    "before", "during", "since", "there", "here", "when", "where", "while",
    "than", "then", "thus", "hence", "both", "each", "all", "any", "some",
    "no", "not", "very", "more", "most", "less", "least", "much", "many",
}
ALLOW_INSIDE = {"of", "with", "to"}  # 允许出现在短语内部，但不作为首尾

MANUAL_HEADS = {
    "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
    "abnormality", "abnormalities", "anomaly", "anomalies", "malformation",
    "malformations", "defect", "defects", "delay", "delays", "impairment",
    "impairments", "loss", "losses", "pain", "pains", "seizure", "seizures",
    "infection", "infections", "inflammation", "hemorrhage", "haemorrhage",
    "hyperplasia", "hypoplasia", "dysplasia", "atrophy", "hypertrophy",
    "aplasia", "stenosis", "insufficiency", "failure", "dysfunction", "palsy",
    "paresis", "paralysis", "edema", "oedema", "erythema", "rash", "lesion",
    "lesions", "mass", "masses", "nodule", "nodules", "tumor", "tumors",
    "neoplasm", "pallor", "jaundice", "cyanosis", "rigidity", "tremor",
    "ataxia", "dystonia", "myopathy", "neuropathy", "retinopathy",
    "nephropathy", "cardiomyopathy", "hepatomegaly", "splenomegaly",
    "lymphadenopathy", "hemoglobin", "glucose", "protein", "enzymes", "levels",
    "counts", "score", "auscultation", "congestion", "discoloration",
    "asphyxia", "incisor", "arch", "hemangioma", "sclerosis", "fibrosis",
}


def derive_head_vocab(subgraph_terms, min_freq=2):
    """由 HPO term 名称末词派生 head 词表（频次过滤）+ 人工头名词。"""
    cnt = Counter()
    for t in subgraph_terms.values():
        name = t.get("name")
        if not name:
            continue
        last = re.split(r"[\s\-]+", name.strip())[-1].lower()
        if last.isalpha() and len(last) >= 3:
            cnt[last] += 1
    heads = {w for w, c in cnt.items() if c >= min_freq}
    heads |= MANUAL_HEADS
    return heads


def _tokens(text):
    toks = []
    for m in TOKEN_RE.finditer(text):
        w = m.group()
        toks.append((m.start(), m.end(), w.lower(), any(ch.isalpha() for ch in w)))
    return toks


def build_chunks(text, head_vocab, max_left=4, max_right=3, max_tokens=8, max_chars=80):
    """围绕 head 词扩展临床短语，返回 set[(start, end)]。"""
    toks = _tokens(text)
    n = len(toks)
    chunks = set()
    for i, (s, e, w, isw) in enumerate(toks):
        if not isw or w not in head_vocab:
            continue
        left = i
        cnt = 0
        while left - 1 >= 0 and cnt < max_left:
            ps, pe, pw, pisw = toks[left - 1]
            gap = text[toks[left - 1][1]:toks[left][0]]
            if HARD_GAP.search(gap):
                break
            if (not pisw) or (pw in STOP and pw not in ALLOW_INSIDE):
                break
            left -= 1
            cnt += 1
        right = i
        cnt = 0
        while right + 1 < n and cnt < max_right:
            ns, ne, nw, nisw = toks[right + 1]
            gap = text[toks[right][1]:toks[right + 1][0]]
            if HARD_GAP.search(gap):
                break
            if (not nisw) or (nw in STOP and nw not in ALLOW_INSIDE):
                break
            right += 1
            cnt += 1
        # 收边：去掉首尾的停用词/介词
        while left < right and toks[left][2] in (STOP | ALLOW_INSIDE):
            left += 1
        while right > left and toks[right][2] in (STOP | ALLOW_INSIDE):
            right -= 1
        cs, ce = toks[left][0], toks[right][1]
        if ce - cs < 3 or ce - cs > max_chars:
            continue
        if right - left + 1 > max_tokens:
            continue
        chunks.add((cs, ce))
    return chunks


SEG_SPLIT = re.compile(r"[.;:!?()\[\]\n\r]")


def _trim(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def build_segments(text, head_vocab, max_chars=120):
    """把文本切成子句段，保留「含 head 词」的段作为外层候选（覆盖子句式长金标）。"""
    segs = set()
    last = 0
    cuts = [m.start() for m in SEG_SPLIT.finditer(text)] + [len(text)]
    for cut in cuts:
        a, b = _trim(text, last, cut)
        if 3 <= b - a <= max_chars:
            piece = text[a:b]
            if any(t[2] in head_vocab for t in _tokens(piece)):
                segs.add((a, b))
        last = cut + 1  # 跳过分隔符
    return segs


def combine(lex_cands, chunks, meta, keep_unmapped=True):
    """把词典候选与短语候选合并。

    - 短语若包含词典候选，继承其 id 并集（primary 取最深），source="np"；
    - 短语若无任何内层词典命中：keep_unmapped=True 时保留 ids=[]（仅用于召回上限，
      不参与 strict F1 / 提交），否则丢弃；
    - 与词典候选完全同 span 时不重复添加。
    """
    out = []
    seen = set()
    for c in lex_cands:
        key = (c["start"], c["end"])
        if key in seen:
            continue
        seen.add(key)
        out.append({**c, "source": "lexicon"})
    for (cs, ce) in sorted(chunks):
        if (cs, ce) in seen:
            continue
        inner = [c for c in lex_cands if c["start"] >= cs and c["end"] <= ce]
        ids = set()
        for c in inner:
            ids |= set(c["ids"])
        if not ids:
            if not keep_unmapped:
                continue
            out.append({
                "start": cs, "end": ce, "ids": [], "primary": None,
                "ambiguous": False, "depth": 0, "source": "np_unmapped",
            })
            seen.add((cs, ce))
        else:
            ids_sorted = sorted(ids)
            primary = max(ids_sorted, key=lambda i: (meta.get(i, {}).get("depth", 0), i))
            out.append({
                "start": cs, "end": ce, "ids": ids_sorted, "primary": primary,
                "ambiguous": len(ids_sorted) > 1,
                "depth": meta.get(primary, {}).get("depth", 0), "source": "np",
            })
            seen.add((cs, ce))
    return out
