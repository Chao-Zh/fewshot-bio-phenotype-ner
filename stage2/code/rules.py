# -*- coding: utf-8 -*-
"""四类硬负例上下文标签器（方案 A：只打标签、不删除候选）。

四类（与阶段二说明一致）：
1. negation      否定表述（no evidence of / negative for / absence of ...）  -> note="NO"
2. family_history 家族史（family history / maternal / mother had ...）
3. animal_model   动物/细胞模型（mouse / knockout / cell line / in vitro ...）
4. method_section 方法学与统计段落（METHODS/TABLE 段 或 cohort/statistical analysis ...）

设计：规则「只做减法、不做加法」——本标签器从不新增候选，只对匹配器产出的候选做上下文判定；
默认保留候选（方案 A），另以 filtered 模式在评测中做消融，对应阶段五 E7。
"""
from __future__ import annotations

import bisect
import re

# 转折/例外标记：截断上下文，避免把 "except for / but" 之前的否定误算到当前候选
_CONTRAST = re.compile(
    r"\b(except for|except|but|however|other than|apart from|aside from|though|although)\b",
    re.I,
)
_SENT_BOUND_BEFORE = re.compile(r"[.;!?]\s")
_SENT_BOUND_AFTER = re.compile(r"[.;!?]\s")

_NEG_BEFORE = re.compile(
    r"(no evidence of|no sign of|no signs of|no history of|without|"
    r"negative for|absence of|\babsent\b|\bdenies\b|\bdenied\b|ruled out|free of|"
    r"\bneither\b|\bnor\b|not present|unremarkable for|\bnever\b|\bnot\b|\bno\b)",
    re.I,
)
_NEG_AFTER = re.compile(
    r"(was negative|were negative|was absent|were absent|not present|"
    r"was not|were not|ruled out|was excluded|were excluded)",
    re.I,
)
_FAM = re.compile(
    r"\b(family history|familial|family members?|maternal|paternal|sibling|"
    r"brother|sister|mother|father|parents?|pedigree|inherited from|"
    r"affected relatives?|relatives?)\b",
    re.I,
)
_ANIMAL = re.compile(
    r"\b(mouse|mice|murine|rat|rats|zebrafish|drosophila|knockout|knock-out|"
    r"transgenic|cell lines?|in vitro|in vivo|hek-?293|hela|fibroblasts?|"
    r"cultured cells|organoids?|xenograft|primary cells)\b",
    re.I,
)
_METHOD_CUE = re.compile(
    r"\b(cohort|participants|inclusion criteria|exclusion criteria|recruited|"
    r"study population|statistical analysis|chi-?square|kaplan-?meier|"
    r"retrospective|prospective|questionnaire|sample size|we enrolled|"
    r"were enrolled|data collection|odds ratio|confidence interval|"
    r"p\s*[<=>]\s*0?\.\d+)\b",
    re.I,
)

_METHOD_SECTIONS = {"METHODS", "METHOD", "TABLE", "REF", "SUPPLEMENTARY"}


def _before(text: str, start: int, window: int) -> str:
    lo = max(0, start - window)
    seg = text[lo:start]
    m = None
    for m in _SENT_BOUND_BEFORE.finditer(seg):
        pass
    if m:
        seg = seg[m.end():]
    c = None
    for c in _CONTRAST.finditer(seg):
        pass
    if c:
        seg = seg[c.end():]
    return seg


def _after(text: str, end: int, window: int) -> str:
    seg = text[end:end + window]
    m = _SENT_BOUND_AFTER.search(seg)
    if m:
        seg = seg[:m.start()]
    return seg


class RuleTagger:
    def __init__(self, passages, before_window: int = 100, after_window: int = 40):
        self.before_window = before_window
        self.after_window = after_window
        # 段落 -> 按 offset 排序，便于二分定位
        self.starts = [int(p.get("offset", 0)) for p in passages]
        self.passages = passages
        self._order = sorted(range(len(passages)), key=lambda i: self.starts[i])
        self._sorted_starts = [self.starts[i] for i in self._order]

    def section_at(self, offset: int):
        if not self._sorted_starts:
            return "", ""
        idx = bisect.bisect_right(self._sorted_starts, offset) - 1
        if idx < 0:
            return "", ""
        p = self.passages[self._order[idx]]
        return str(p.get("section_type", "")), str(p.get("type", ""))

    def tag(self, text: str, cand: dict) -> dict:
        s, e = cand["start"], cand["end"]
        before = _before(text, s, self.before_window)
        after = _after(text, e, self.after_window)
        sec, ptype = self.section_at(s)

        neg = bool(_NEG_BEFORE.search(before) or _NEG_AFTER.search(after))
        fam = bool(_FAM.search(before) or _FAM.search(after))
        animal = bool(_ANIMAL.search(before) or _ANIMAL.search(after))

        sec_norm = sec.upper()
        method = (
            sec_norm in _METHOD_SECTIONS
            or ptype == "table_caption"
            or bool(_METHOD_CUE.search(before) or _METHOD_CUE.search(after))
        )

        if neg:
            reason = "negation"
        elif fam:
            reason = "family_history"
        elif animal:
            reason = "animal_model"
        elif method:
            reason = "method_section"
        else:
            reason = None

        cand["flags"] = {
            "negation": neg,
            "family_history": fam,
            "animal_model": animal,
            "method_section": method,
        }
        cand["primary_reason"] = reason
        cand["section_type"] = sec
        cand["passage_type"] = ptype
        cand["note"] = "NO" if neg else None
        return cand


def tag_candidates(candidates, text: str, passages, **kw):
    tagger = RuleTagger(passages, **kw)
    return [tagger.tag(text, c) for c in candidates]
