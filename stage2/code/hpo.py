# -*- coding: utf-8 -*-
"""HPO 本体子图与词典构建（阶段二对阶段一资源底座的最小依赖版）。

职责：
1. 解析 hp.obo（OBO 1.2），逐 [Term] 提取 id/name/def/synonym/is_a/alt_id/replaced_by/xref。
2. 从 HP:0000118 取全部后代（含自身），并计算 depth（距根最短层数）与 branch（根的直接子分支）。
3. 折叠 alt_id（把 alt_id 映射回主 id），避免同一概念两个出口。
4. 由 name + 全部 synonym 构建 surface -> {primary_hpo_id} 词典；补充连字符/空格、希腊字母两类
   安全变体（不做缩写/单复数自动扩展，避免噪声）。同时导出歧义表（一个 surface 对应多个 term）。

用法（独立）：
    python stage2/hpo.py --obo path/to/hp.obo --out stage2/output
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict, deque

ROOT = "HP:0000118"  # Phenotypic abnormality
VERSION = "2026-06-23"

# synonym: "text" SCOPE [synonymtype] []
_SYN_RE = re.compile(
    r'^synonym:\s+"((?:[^"\\]|\\.)*)"\s+(EXACT|NARROW|BROAD|RELATED)\b(.*)$'
)
_DEF_RE = re.compile(r'^def:\s+"((?:[^"\\]|\\.)*)"')

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
}


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').strip()


def parse_obo(path: str):
    """解析 OBO 文件，返回 (terms, is_a_children_map, is_a_parents_map)。"""
    terms = {}
    parents = defaultdict(list)
    cur = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "[Term]":
                cur = {"synonyms": [], "alt_ids": [], "definition": None,
                       "xrefs": [], "parents": []}
                continue
            if cur is None:
                continue
            if line == "":
                cur = None
                continue
            if line.startswith("id: "):
                cur["id"] = line[4:].strip()
                terms[cur["id"]] = cur
            elif line.startswith("name: "):
                cur["name"] = line[6:].strip()
            elif line.startswith("def: "):
                m = _DEF_RE.match(line)
                if m:
                    cur["definition"] = _unescape(m.group(1))
            elif line.startswith("synonym: "):
                m = _SYN_RE.match(line)
                if m:
                    tail = m.group(3)
                    # tail 形如 `layperson []` 或 ` []`；取方括号前的 synonym type
                    stype = tail.split("[")[0].strip() or "default"
                    cur["synonyms"].append(
                        {"text": _unescape(m.group(1)), "scope": m.group(2),
                         "type": stype}
                    )
            elif line.startswith("is_a: "):
                p = line[6:].split("!")[0].strip()
                if p:
                    cur["parents"].append(p)
                    parents[cur["id"]].append(p)
            elif line.startswith("alt_id: "):
                cur["alt_ids"].append(line[8:].strip())
            elif line.startswith("xref: "):
                cur["xrefs"].append(line[6:].strip())
            elif line.startswith("replaced_by: "):
                cur["replaced_by"] = line[13:].strip()
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
    # 丢弃未成功记录 id 的残片
    terms = {k: v for k, v in terms.items() if "id" in v}
    return terms, parents


def descendants_with_meta(terms, parents_map, root=ROOT):
    """BFS 取 root 后代，返回 (subgraph_ids, meta)。

    meta[id] = {depth, branch}；branch 为根的直接子分支（root 自身 branch=root）。
    """
    children = defaultdict(list)
    for c, ps in parents_map.items():
        for p in ps:
            children[p].append(c)

    depth = {root: 0}
    branch = {root: root}
    order = []
    dq = deque([root])
    seen = set()
    while dq:
        x = dq.popleft()
        if x in seen:
            continue
        seen.add(x)
        order.append(x)
        for ch in children.get(x, []):
            if ch not in seen:
                if x == root:
                    branch[ch] = ch
                else:
                    branch[ch] = branch.get(x, ch)
                depth[ch] = depth.get(x, 0) + 1
                dq.append(ch)
    meta = {tid: {"depth": depth.get(tid, 0), "branch": branch.get(tid, tid)}
            for tid in seen}
    return seen, meta


def _valid_surface(s: str) -> bool:
    s = s.strip()
    if len(s) < 3:
        return False
    if not any(ch.isalpha() for ch in s):
        return False
    return True


def _plural_variants(text: str):
    """对末词做安全的单复数变换（避免 ss/us/is 结尾的误伤）。"""
    out = set()
    toks = text.split()
    if not toks:
        return out
    last = toks[-1]
    low = last.lower()
    if len(last) < 4 or not last.isalpha():
        return out
    cand = None
    if low.endswith("ies") and len(last) > 4:
        cand = last[:-3] + "y"
    elif low.endswith("y") and last[-2].lower() not in "aeiou":
        cand = last[:-1] + "ies"
    elif low.endswith("s"):
        if not (low.endswith("ss") or low.endswith("us") or low.endswith("is")):
            cand = last[:-1]
    else:
        cand = last + "s"
    if cand:
        out.add(" ".join(toks[:-1] + [cand]))
    return out


def surface_variants(text: str):
    """安全变体：原串 + 连字符↔空格 + 希腊字母↔单词 + 末词单复数。"""
    text = text.strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    out = {text}
    if "-" in text:
        out.add(text.replace("-", " "))
    if text.count(" ") == 1:
        out.add(text.replace(" ", "-"))
    low = text.lower()
    for word, ch in GREEK.items():
        if re.search(r"\b" + re.escape(word) + r"\b", low):
            out.add(re.sub(r"\b" + re.escape(word) + r"\b", ch, text, flags=re.I))
        if ch in text:
            out.add(text.replace(ch, word))
    out |= _plural_variants(text)
    return {v for v in out if _valid_surface(v)}


def build_lexicon(terms, subgraph, meta):
    """构建 surface(lower) -> set(primary id) 词典及歧义表。"""
    lex = defaultdict(set)
    src_count = defaultdict(int)
    for tid in subgraph:
        t = terms.get(tid)
        if not t:
            continue
        surfaces = []
        if t.get("name"):
            surfaces.append((t["name"], "name"))
        for s in t.get("synonyms", []):
            surfaces.append((s["text"], "syn:" + s["scope"]))
        for surface, src in surfaces:
            for v in surface_variants(surface):
                key = v.lower()
                lex[key].add(tid)
                src_count[src] += 1
    ambiguity = {k: sorted(v) for k, v in lex.items() if len(v) > 1}
    return (
        {k: sorted(v) for k, v in lex.items()},
        ambiguity,
        dict(src_count),
    )


def build_subgraph_json(terms, subgraph, meta):
    terms_out = {}
    for tid in subgraph:
        t = terms.get(tid)
        if not t:
            continue
        terms_out[tid] = {
            "name": t.get("name"),
            "definition": t.get("definition"),
            "synonyms": [
                {"text": s["text"], "scope": s["scope"], "type": s["type"]}
                for s in t.get("synonyms", [])
            ],
            "alt_ids": t.get("alt_ids", []),
            "parents": t.get("parents", []),
            "xrefs": t.get("xrefs", []),
            "depth": meta[tid]["depth"],
            "branch": meta[tid]["branch"],
        }
    return {
        "version": VERSION,
        "root": ROOT,
        "num_terms": len(terms_out),
        "terms": terms_out,
    }


def build(obo_path: str, out_dir: str, verbose: bool = True):
    """构建子图与词典并写出 JSON。返回 (terms, subgraph, meta, lexicon, ambiguity)。"""
    os.makedirs(out_dir, exist_ok=True)
    terms, parents_map = parse_obo(obo_path)
    subgraph, meta = descendants_with_meta(terms, parents_map)
    lexicon, ambiguity, src_count = build_lexicon(terms, subgraph, meta)

    sub_json = build_subgraph_json(terms, subgraph, meta)
    with open(os.path.join(out_dir, "hpo_subgraph.json"), "w", encoding="utf-8") as f:
        json.dump(sub_json, f, ensure_ascii=False)

    lex_json = {
        "version": VERSION,
        "root": ROOT,
        "num_terms": len(subgraph),
        "num_surfaces": len(lexicon),
        "num_ambiguous": len(ambiguity),
        "source_counts": src_count,
        "surfaces": lexicon,
        "ambiguity": ambiguity,
    }
    with open(os.path.join(out_dir, "lexicon.json"), "w", encoding="utf-8") as f:
        json.dump(lex_json, f, ensure_ascii=False)

    if verbose:
        print("[hpo] terms in subgraph =", len(subgraph))
        print("[hpo] lexicon surfaces   =", len(lexicon))
        print("[hpo] ambiguous surfaces =", len(ambiguity))
    return terms, subgraph, meta, lexicon, ambiguity


def main():
    ap = argparse.ArgumentParser(description="HPO 子图与词典构建")
    ap.add_argument("--obo", required=True, help="hp.obo 路径")
    ap.add_argument("--out", required=True, help="输出目录")
    args = ap.parse_args()
    build(args.obo, args.out)


if __name__ == "__main__":
    main()
