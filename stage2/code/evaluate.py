# -*- coding: utf-8 -*-
"""阶段二评测：三个核心数 + E0 严格 F1 + 五类错误分析 + 分支召回。

记录结构（records，每个元素为一篇文献）：
    {
        "pmc_id": str,
        "text": str,
        "gold": [ {"start","end","ids": set,"note"} ... ],
        "candidates": [ {"start","end","ids": set,"flags":{...},
                         "primary_reason","note"} ... ]   # 已按 mode 过滤
    }

指标定义：
- candidate_recall     金标 span 被某候选包络覆盖（c.start<=g.start<=g.end<=c.end）的比例（召回上限）
- recall_exact_span    金标 span 与某候选边界完全相等（忽略 ID）
- recall_strict        边界相等且 HPO ID 有交集
- precision_span_exact 候选中边界与某金标完全相等的比例（"候选精确率"主口径）
- precision_strict     边界相等且 ID 有交的比例
- precision_overlap    候选与某金标区间有交的比例
- E0 strict F1         把候选当预测，按 (边界相等 + ID 有交) 一对一匹配算 P/R/F1
"""
from __future__ import annotations

from collections import Counter, defaultdict


def _overlap(a_s, a_e, b_s, b_e):
    return a_s < b_e and b_s < a_e


def _summ(counts):
    n = len(counts)
    if n == 0:
        return {"n": 0}
    s = sorted(counts)
    q = lambda p: s[min(n - 1, int(round(p * (n - 1))))]
    return {
        "n": n, "min": s[0], "max": s[-1],
        "mean": round(sum(s) / n, 2),
        "median": q(0.5), "p90": q(0.9),
    }


def _hist(counts, edges):
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        out[f"[{lo},{hi})"] = sum(1 for c in counts if lo <= c < hi)
    out[f"[{edges[-1]},+)"] = sum(1 for c in counts if c >= edges[-1])
    return out


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def evaluate(records, meta=None, max_examples=8):
    """对已（可选）过滤后的 records 计算全部指标。"""
    num_gold = num_pred = 0

    # 召回累计
    cov = ov = exact = strict = 0
    # 正类金标（note != NO）召回
    pos_gold = pos_cov = pos_ov = pos_exact = pos_strict = 0
    # E0 一对一匹配
    tp_strict = tp_span = 0
    tp_strict_pos = tp_span_pos = 0
    gold_pos_total = 0
    # 候选状态
    status = Counter()
    fp_reason = Counter()
    fp_examples = defaultdict(list)
    # 映射准确率（span 正确前提下）
    span_matched_preds = span_matched_id_ok = 0
    # 分支召回
    branch_gold = Counter()
    branch_cov = Counter()
    # 否定标签质量（仅统计与金标 span 完全相等且金标有标记者）
    neg_tp = neg_fp = neg_fn = 0
    # 每篇候选数
    per_doc = []
    # 未被候选覆盖的金标 surface（指导阶段一词表补全）
    uncovered_count = 0
    uncovered_examples = []

    for rec in records:
        text = rec["text"]
        gold = rec["gold"]
        cands = rec["candidates"]
        G, C = len(gold), len(cands)
        num_gold += G
        num_pred += C
        per_doc.append(C)

        # 候选 ids 预转 set
        for c in cands:
            c["_ids"] = set(c.get("ids", []))

        # ---- 召回（以金标为主体）----
        for g in gold:
            gs, ge = g["start"], g["end"]
            gids = set(g["ids"])
            is_pos = g.get("note") != "NO"
            if is_pos:
                pos_gold += 1
            c_cov = c_ov = c_ex = c_st = False
            for c in cands:
                cs, ce = c["start"], c["end"]
                if _overlap(gs, ge, cs, ce):
                    c_ov = True
                if cs <= gs and ge <= ce:
                    c_cov = True
                if cs == gs and ce == ge:
                    c_ex = True
                    if gids & c["_ids"]:
                        c_st = True
            if not c_cov:
                uncovered_count += 1
                if len(uncovered_examples) < 40:
                    uncovered_examples.append(text[gs:ge])
            if c_ov:
                ov += 1
                if is_pos:
                    pos_ov += 1
            if c_cov:
                cov += 1
                if is_pos:
                    pos_cov += 1
            if c_ex:
                exact += 1
                if is_pos:
                    pos_exact += 1
            if c_st:
                strict += 1
                if is_pos:
                    pos_strict += 1
            # 分支
            if meta and gids:
                fid = sorted(gids)[0]
                b = meta.get(fid, {}).get("branch", "UNMAPPED")
            else:
                b = "UNMAPPED"
            branch_gold[b] += 1
            if c_cov:
                branch_cov[b] += 1

        # ---- 候选状态 / 精确率 / 映射准确率 ----
        for c in cands:
            cs, ce = c["start"], c["end"]
            ex_all = [g for g in gold if g["start"] == cs and g["end"] == ce]
            ov_all = [g for g in gold if _overlap(cs, ce, g["start"], g["end"])]
            if ex_all:
                if any(set(g["ids"]) & c["_ids"] for g in ex_all):
                    status["exact_id"] += 1
                    span_matched_preds += 1
                    span_matched_id_ok += 1
                else:
                    status["id_error"] += 1
                    span_matched_preds += 1
            elif ov_all:
                status["boundary_error"] += 1
            else:
                status["fp"] += 1
                reason = c.get("primary_reason") or "other"
                fp_reason[reason] += 1
                surf = text[cs:ce]
                if len(fp_examples[reason]) < max_examples and surf not in fp_examples[reason]:
                    fp_examples[reason].append(surf)

        # ---- 否定标签质量（金标 span 对齐后）----
        gold_by_span = {}
        for g in gold:
            gold_by_span.setdefault((g["start"], g["end"]), []).append(g)
        for c in cands:
            gs_list = gold_by_span.get((c["start"], c["end"]))
            if not gs_list:
                continue
            g_neg = any(g.get("note") == "NO" for g in gs_list)
            c_neg = bool(c.get("flags", {}).get("negation"))
            if g_neg and c_neg:
                neg_tp += 1
            elif g_neg and not c_neg:
                neg_fn += 1
            elif (not g_neg) and c_neg:
                neg_fp += 1

        # ---- E0 一对一匹配（全部金标）----
        used = set()
        for c in cands:
            for j, g in enumerate(gold):
                if j in used:
                    continue
                if c["start"] == g["start"] and c["end"] == g["end"]:
                    if set(g["ids"]) & c["_ids"]:
                        used.add(j)
                        tp_strict += 1
                        break
        # span-only：按边界一对一
        tp_span += _span_only_tp(cands, gold)

        # 正类金标 E0
        gold_pos = [g for g in gold if g.get("note") != "NO"]
        gold_pos_total += len(gold_pos)
        usedp = set()
        pspan = 0
        for c in cands:
            for j, g in enumerate(gold_pos):
                if j in usedp:
                    continue
                if c["start"] == g["start"] and c["end"] == g["end"]:
                    pspan += 1
                    usedp.add(j)
                    if set(g["ids"]) & c["_ids"]:
                        tp_strict_pos += 1
                    break
        tp_span_pos += pspan

    total_pred = num_pred
    recall_cover = cov / num_gold if num_gold else 0.0
    precision_span_exact = (total_pred - status["fp"] - status["boundary_error"]) / total_pred if total_pred else 0.0
    precision_strict = status["exact_id"] / total_pred if total_pred else 0.0
    precision_overlap = (total_pred - status["fp"]) / total_pred if total_pred else 0.0

    result = {
        "num_docs": len(records),
        "num_gold": num_gold,
        "num_pred": total_pred,
        "gold_positive": pos_gold,
        # 召回
        "candidate_recall": round(recall_cover, 4),
        "candidate_recall_overlap": round(ov / num_gold, 4) if num_gold else 0.0,
        "recall_exact_span": round(exact / num_gold, 4) if num_gold else 0.0,
        "recall_strict": round(strict / num_gold, 4) if num_gold else 0.0,
        "candidate_recall_positive": round(pos_cov / pos_gold, 4) if pos_gold else 0.0,
        "candidate_recall_overlap_positive": round(pos_ov / pos_gold, 4) if pos_gold else 0.0,
        "recall_strict_positive": round(pos_strict / pos_gold, 4) if pos_gold else 0.0,
        "uncovered_gold_count": uncovered_count,
        "uncovered_gold_rate": round(uncovered_count / num_gold, 4) if num_gold else 0.0,
        "uncovered_gold_examples": uncovered_examples,
        # 精确率
        "precision_span_exact": round(precision_span_exact, 4),
        "precision_strict": round(precision_strict, 4),
        "precision_overlap": round(precision_overlap, 4),
        # 每篇候选数
        "candidates_per_doc": _summ(per_doc),
        "candidates_hist": _hist(per_doc, [0, 50, 100, 200, 400, 800]),
        # E0 F1
        "e0_strict_all": _prf(tp_strict, total_pred - tp_strict, num_gold - tp_strict),
        "e0_span_all": _prf(tp_span, total_pred - tp_span, num_gold - tp_span),
        "e0_strict_positive": _prf(tp_strict_pos, total_pred - tp_strict_pos,
                                   gold_pos_total - tp_strict_pos),
        # 映射准确率
        "hpo_mapping_accuracy": round(span_matched_id_ok / span_matched_preds, 4)
        if span_matched_preds else 0.0,
        "span_matched_preds": span_matched_preds,
        # 状态
        "status_counts": dict(status),
        "fp_reason_counts": dict(fp_reason),
        "fp_by_primary_reason": dict(fp_reason),
        "fp_examples": {k: v for k, v in fp_examples.items()},
        # 否定标签质量
        "negation_tag": {
            "tp": neg_tp, "fp": neg_fp, "fn": neg_fn,
            "precision": round(neg_tp / (neg_tp + neg_fp), 4) if (neg_tp + neg_fp) else None,
            "recall": round(neg_tp / (neg_tp + neg_fn), 4) if (neg_tp + neg_fn) else None,
        },
        # 分支召回
        "branch_recall": {
            b: {"gold": branch_gold[b], "covered": branch_cov[b],
                "recall": round(branch_cov[b] / branch_gold[b], 4)}
            for b in branch_gold
        },
    }
    # FP 四类合计占比（按 primary_reason）
    fp_total = sum(fp_reason.values()) or 1
    four = sum(fp_reason.get(k, 0) for k in
               ("negation", "family_history", "animal_model", "method_section"))
    result["fp_four_class_share"] = round(four / fp_total, 4)
    result["fp_other_share"] = round(fp_reason.get("other", 0) / fp_total, 4)
    return result


def _span_only_tp(cands, gold):
    used = set()
    tp = 0
    for c in cands:
        for j, g in enumerate(gold):
            if j in used:
                continue
            if c["start"] == g["start"] and c["end"] == g["end"]:
                used.add(j)
                tp += 1
                break
    return tp


def filter_records(records, drop_flags):
    """返回去掉命中 drop_flags 任一标记的候选后的新 records（浅拷贝候选）。"""
    if not drop_flags:
        return records
    out = []
    for rec in records:
        kept = [c for c in rec["candidates"]
                if not any(c.get("flags", {}).get(f) for f in drop_flags)]
        out.append({**rec, "candidates": kept})
    return out
