# -*- coding: utf-8 -*-
"""Aho-Corasick 多模式匹配 + 候选重叠合并。

匹配约束（对齐阶段一说明）：
- 大小写不敏感：文本与模式统一取小写；用 lower_preserve_len 保证索引与原文 1:1。
- 词边界约束：命中区间前后若为字母/数字，则判为子串噪声丢弃。
- 最长匹配优先 / 嵌套取外层 / 同起点取最长 / 完全重叠保留全部 HPO 并打歧义标记。

候选 span 始终来自真实 pattern 命中（不做跨模式拼接），避免生成超长包络。
"""
from __future__ import annotations

import ahocorasick


def lower_preserve_len(s: str) -> str:
    """逐字符小写，保持长度不变（个别 Unicode 小写会变长时保留原字符）。"""
    out = []
    for ch in s:
        lo = ch.lower()
        out.append(lo if len(lo) == 1 else ch)
    return "".join(out)


def build_automaton(lexicon):
    """lexicon: dict surface(lower) -> list/set ids。返回已 make 的自动机。"""
    autom = ahocorasick.Automaton()
    for surface in lexicon:
        autom.add_word(surface, surface)
    autom.make_automaton()
    return autom


def find_spans(text: str, autom, lexicon):
    """扫描全文，返回 {(start, end): set(ids)}。end 为开区间。"""
    low = lower_preserve_len(text)
    raw = {}
    n = len(low)
    for end, surface in autom.iter(low):
        length = len(surface)
        start = end - length + 1
        if start < 0:
            continue
        # 词边界：仅当模式边缘是字母/数字时才检查，避免误伤标点起止的模式
        if surface[0].isalnum() and start > 0 and low[start - 1].isalnum():
            continue
        if surface[-1].isalnum() and end + 1 < n and low[end + 1].isalnum():
            continue
        ids = lexicon.get(surface)
        if not ids:
            continue
        raw.setdefault((start, end + 1), set()).update(ids)
    return raw


def merge_spans(raw):
    """合并重叠候选 span。

    规则：
    - 同一起始位置取最长；
    - 嵌套取外层（内层丢弃）；
    - 完全重叠（同 span）已在 raw 中按 key 合并，保留全部 HPO id；
    - 交叉部分重叠（少见）：保留更长者，长度相同保留先出现者。
    返回 list[(start, end, set(ids))]，按 start 升序且互不重叠。
    """
    items = sorted(raw.items(), key=lambda kv: (kv[0][0], -(kv[0][1] - kv[0][0])))
    chosen = []  # list[[start, end, set(ids)]]
    for (s, e), ids in items:
        if not chosen:
            chosen.append([s, e, set(ids)])
            continue
        ls, le, lids = chosen[-1]
        if s >= le:
            chosen.append([s, e, set(ids)])
        elif s == ls:
            # 同起点：先出现的是最长者，其余丢弃
            continue
        elif e <= le:
            # 嵌套于外层内，丢弃
            continue
        else:
            # 交叉重叠：保留更长
            if (e - s) > (le - ls):
                chosen[-1] = [s, e, set(ids)]
    return [(s, e, ids) for s, e, ids in chosen]


def make_candidates(text: str, autom, lexicon, meta):
    """匹配 + 合并 + 选主 id，返回候选列表。

    每个候选: {start, end, ids(sorted list), primary, ambiguous, depth}
    primary = 该 span 上最深（最具体）的 HPO term。
    """
    raw = find_spans(text, autom, lexicon)
    merged = merge_spans(raw)
    out = []
    for s, e, ids in merged:
        ids_sorted = sorted(ids)
        primary = max(ids_sorted, key=lambda i: (meta.get(i, {}).get("depth", 0), i))
        out.append({
            "start": s,
            "end": e,
            "ids": ids_sorted,
            "primary": primary,
            "ambiguous": len(ids_sorted) > 1,
            "depth": meta.get(primary, {}).get("depth", 0),
        })
    return out
