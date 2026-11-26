# -*- coding: utf-8 -*-
"""轻量实体链接器：把无精确词典命中的名词短语 span 链到 HPO term。

无外部依赖、不使用金标。做法：把 HPO term 名称 + 同义词切成内容词，建「词 → term」倒排；
对一个短语，取其内容词，在候选 term 里按 token-F1 打分，取得分最高且 ≥ 阈值的 term。
这是粗链接（ID 精度有限），用于量化「粗召回 span 能否转成带 ID 候选」的上限。
"""
from __future__ import annotations

import re
from collections import defaultdict

_WORD = re.compile(r"[A-Za-z]+")
# 弱词：不参与匹配打分
STOP = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "without", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "has", "have", "had", "this", "that", "these", "those",
    "his", "her", "its", "their", "our", "he", "she", "they", "we", "it",
    "which", "who", "whom", "whose", "was", "not", "no", "or", "and",
    "abnormality", "abnormalities", "abnormal",  # 过于泛化的头词不单独作信号
}


def content_tokens(s):
    out = set()
    for w in _WORD.findall(str(s).lower()):
        if len(w) >= 3 and w not in STOP:
            out.add(w)
    return out


class HpoLinker:
    def __init__(self, sub_terms):
        # sub_terms: tid -> {"name","synonyms":[{"text"}...]}
        self.term_tokens = {}
        self.index = defaultdict(set)
        for tid, t in sub_terms.items():
            toks = content_tokens(t.get("name") or "")
            for s in t.get("synonyms", []) or []:
                toks |= content_tokens(s.get("text", ""))
            if not toks:
                continue
            self.term_tokens[tid] = toks
            for w in toks:
                self.index[w].add(tid)
        self._depth = None

    def link(self, text, thr=0.5, topk=2):
        """返回 (ids, score)。ids 为空表示未链接到可信 term。"""
        ct = content_tokens(text)
        if not ct:
            return [], 0.0
        cand = set()
        for w in ct:
            cand |= self.index.get(w, set())
        if not cand:
            return [], 0.0
        scored = []
        for tid in cand:
            tt = self.term_tokens.get(tid)
            if not tt:
                continue
            inter = len(ct & tt)
            if inter == 0:
                continue
            p = inter / len(tt)
            r = inter / len(ct)
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            scored.append((f, tid))
        if not scored:
            return [], 0.0
        scored.sort(reverse=True)
        best = scored[0][0]
        if best < thr:
            return [], best
        ids = [tid for f, tid in scored[:topk] if f >= thr]
        return ids, best
