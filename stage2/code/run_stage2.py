# -*- coding: utf-8 -*-
"""阶段二：候选生成与无学习基线 —— 编排入口。

一条命令跑完：HPO 子图/词典 -> 全文匹配+合并 -> 四类硬负例打标签 ->
train 金标评测（三指标 / E0 F1 / 五类错误 / 分支召回）-> A/B 候选与基线提交。

用法：
    python stage2/run_stage2.py --data "D:\\...\\PatienPheX" --out "D:\\...\\stage2\\output"

方案 A：否定/家族史等四类硬负例只打标签、不删除候选；另以 filtered 模式做消融对照。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 允许以脚本方式运行时导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hpo as hpo_mod              # noqa: E402
import matcher as matcher_mod      # noqa: E402
import rules as rules_mod          # noqa: E402
import evaluate as eval_mod        # noqa: E402
import cv_lexicon_recall as cv_mod  # noqa: E402
import np_recall as np_mod          # noqa: E402

TRAIN_FILE = "PatientPheX-train.jsonl"
A_FILE = "PatientPheX-A.jsonl"
B_FILE = "PatientPheX-V1-B.jsonl"
OBO_FILE = "hp.obo"


# ---------------------------------------------------------------- 基础工具
def rebuild_full_text(passages):
    """按全局 offset 重建完整全文（段落间用空格填充），零漂移。

    与参考仓库 patientpheX_adapter.rebuild_full_text 同构：实体 offset 直接作用于该字符串。
    """
    if not passages:
        return ""
    max_end = max(int(p["offset"]) + len(str(p.get("text", ""))) for p in passages)
    buf = [" "] * max_end
    for p in passages:
        s = int(p["offset"])
        t = str(p.get("text", ""))
        for i, ch in enumerate(t):
            if s + i < max_end:
                buf[s + i] = ch
    return "".join(buf)


def load_jsonl(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def parse_ids(identifier):
    ids = set()
    for x in str(identifier).split(";"):
        x = x.strip()
        if x and x != "-1":
            ids.add(x)
    return ids


def gold_from_doc(doc):
    gold = []
    for e in doc.get("entities", []):
        s = int(e.get("offset", 0))
        ln = int(e.get("length", 0))
        gold.append({
            "start": s, "end": s + ln,
            "ids": parse_ids(e.get("identifier", "-1")),
            "note": e.get("note"),
        })
    return gold


def build_doc_candidates(doc, autom, lexicon, meta, np_cfg=None):
    """对单篇：重建全文 -> 精确匹配合并 ->（可选）名词短语粗召回 -> 打标签。

    返回 (text, precise, coarse)：
    - precise：词典精确匹配候选（可映射，供阶段三闭集打分 / 精确基线）；
    - coarse ：precise + 名词短语/子句段粗召回（含未映射 span，标 source），用于召回上限。
    """
    passages = doc.get("full_text", [])
    text = rebuild_full_text(passages)
    precise = matcher_mod.make_candidates(text, autom, lexicon, meta)
    if np_cfg:
        chunks = np_mod.build_chunks(text, np_cfg["heads"],
                                     max_left=np_cfg["max_left"],
                                     max_right=np_cfg["max_right"])
        if np_cfg["seg_chars"]:
            chunks = chunks | np_mod.build_segments(text, np_cfg["heads"],
                                                    max_chars=np_cfg["seg_chars"])
        coarse = np_mod.combine(precise, chunks, meta, keep_unmapped=True)
        rules_mod.tag_candidates(precise, text, passages)
        rules_mod.tag_candidates(coarse, text, passages)
    else:
        coarse = precise
        rules_mod.tag_candidates(precise, text, passages)
    return text, precise, coarse


# ---------------------------------------------------------------- 写出
def _serialize_candidate(text, c):
    return {
        "start": c["start"], "end": c["end"],
        "text": text[c["start"]:c["end"]],
        "ids": c["ids"], "primary": c["primary"],
        "ambiguous": c["ambiguous"], "depth": c["depth"],
        "source": c.get("source", "lexicon"),
        "flags": c["flags"], "primary_reason": c["primary_reason"],
        "section_type": c["section_type"], "note": c["note"],
    }


def write_candidates(path, docs, texts_cands):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            pmc = str(doc.get("pmc_id"))
            text, _precise, cands = texts_cands[pmc]
            rec = {
                "pmc_id": pmc, "pmid": doc.get("pmid"),
                "num_candidates": len(cands),
                "candidates": [_serialize_candidate(text, c) for c in cands],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_submission(path, docs, texts_cands, drop_flags=None):
    """写 E0 基线提交（association 置空：本任务只做子任务1）。"""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            pmc = str(doc.get("pmc_id"))
            text, _precise, cands = texts_cands[pmc]
            ents = []
            for c in cands:
                if c.get("source", "lexicon") != "lexicon":
                    continue  # E0 提交只用精确词典候选；粗召回 span 不计入提交
                if not c.get("ids"):
                    continue
                if drop_flags and any(c.get("flags", {}).get(fl) for fl in drop_flags):
                    continue
                ents.append({
                    "identifier": ";".join(c["ids"]),
                    "type": "Phenotype",
                    "offset": c["start"],
                    "length": c["end"] - c["start"],
                    "text": text[c["start"]:c["end"]],
                    "note": c["note"],
                })
            rec = {"pmc_id": pmc, "pmid": doc.get("pmid"),
                   "entities": ents, "association": []}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += len(ents)
    return n


# ---------------------------------------------------------------- 报告
def _row(mode, m):
    return (f"| {mode} | {m['candidates_per_doc']['mean']} | {m['candidate_recall']:.3f} | "
            f"{m['candidate_recall_overlap']:.3f} | {m['recall_exact_span']:.3f} | "
            f"{m['recall_strict']:.3f} | {m['precision_span_exact']:.3f} | "
            f"{m['precision_strict']:.3f} | {m['e0_strict_all']['f1']:.3f} | "
            f"{m['e0_span_all']['f1']:.3f} |")


def render_report(cfg, lex_stats, metrics, sub_counts):
    raw = metrics["raw"]
    coarse = metrics.get("raw_coarse")
    lines = []
    A = lines.append
    A("# 阶段二报告：候选生成与无学习基线")
    A("")
    A("## 0. 口径与数据")
    A("")
    A(f"- 数据目录：`{cfg['data']}`")
    A(f"- 训练集：{raw['num_docs']} 篇，金标实体 {raw['num_gold']} 个"
      f"（其中 note=NO 否定 {raw['num_gold'] - raw['gold_positive']} 个）")
    A(f"- HPO：v{hpo_mod.VERSION}，HP:0000118 分支 {lex_stats['num_terms']} 个 term；"
      f"词典 surface {lex_stats['num_surfaces']} 条，歧义 surface {lex_stats['num_ambiguous']} 条")
    A(f"- 方案：**A（否定/家族史等硬负例只打标签、不删除）**；"
      f"另报 filtered 消融。association 置空（仅子任务1）。")
    if cfg.get("augment"):
        A("- ⚠ 词典已用 **train 金标 surface 扩充**（模拟阶段一补词表）。"
          "此时 train 指标偏乐观（存在泄漏），仅用于展示“补词表后能否过 gate”与生成 A/B 候选；"
          "诚实基线请看未扩充的 ontology-only 运行。")
    if cfg.get("lexicon_file") and not cfg.get("augment"):
        A(f"- ⚠ 使用**自定义词典** `{os.path.basename(cfg['lexicon_file'])}`（可能含 corpus 补词）。"
          "对 train 指标偏乐观（in-sample）；A/B 的可外推估计见 `cv_recall.json`（5 折）。")
    if cfg.get("np_recall"):
        A(f"- 已启用**名词短语粗召回**（子句段 ≤ {cfg.get('np_seg_chars')} 字符）。"
          "候选文件带 `source=lexicon/np/np_unmapped`：`np_unmapped` 无 HPO id，仅抬召回上限，"
          "不参与 strict F1 / 提交；阶段三闭集打分只用带 id 的候选。")
    A("")
    A("## 1. 三个核心数")
    A("")
    A(f"**精确集（词典匹配，可映射，E0 基线用）：**")
    A(f"- candidate recall（覆盖）/（overlap） = {raw['candidate_recall']:.4f} / "
      f"{raw['candidate_recall_overlap']:.4f}")
    A(f"- 候选精确率（span 边界相等）= {raw['precision_span_exact']:.4f}"
      f"（严格 span+ID = {raw['precision_strict']:.4f}）；"
      f"E0 strict F1 = {raw['e0_strict_all']['f1']:.4f}")
    A(f"- 每篇平均候选数 = {raw['candidates_per_doc']['mean']}"
      f"（中位 {raw['candidates_per_doc']['median']}，最大 {raw['candidates_per_doc']['max']}）；"
      f"HPO 映射准确率 = {raw['hpo_mapping_accuracy']:.4f}")
    if coarse:
        A("")
        A("**粗召回集（精确集 + 名词短语/子句段兜底，衡量候选召回上限）：**")
        A(f"- candidate recall（覆盖）/（overlap） = **{coarse['candidate_recall']:.4f} / "
          f"{coarse['candidate_recall_overlap']:.4f}**")
        A(f"- 每篇平均候选数 = {coarse['candidates_per_doc']['mean']}"
          f"（中位 {coarse['candidates_per_doc']['median']}，"
          f"最大 {coarse['candidates_per_doc']['max']}）")
        A("")
        A("> 粗召回层含「未映射」span（`source=np_unmapped`，无 HPO id），只抬高**召回上限**；"
          "能进阶段三闭集打分的是 `source=lexicon/np`（带 id）的候选。阶段三/B 提交只用带 id 的候选。")
    A("")
    A("## 2. 规则基线结果表（三种模式）")
    A("")
    A("| 模式 | 候选/篇 | recall(覆盖) | recall(overlap) | recall_exact_span | "
      "recall_strict | precision_span_exact | precision_strict | E0 strict F1 | E0 span F1 |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    A(_row("raw 精确集（方案A：不删）", metrics["raw"]))
    if coarse:
        A(_row("raw + NP 粗召回（召回上限）", coarse))
    A(_row("filtered -否定-家族史", metrics["filtered_negfam"]))
    A(_row("filtered -四类全删", metrics["filtered_all4"]))
    A("")
    A("> filtered 行展示「若按 PDF 做减法硬删」的对照：precision 上升但 recall 下降，"
      "印证方案A保留候选不伤召回上限。")
    A("")
    A("## 3. 每篇候选数分布（raw）")
    A("")
    d = raw["candidates_per_doc"]
    A(f"min={d['min']} / mean={d['mean']} / median={d['median']} / p90={d['p90']} / max={d['max']}")
    A("")
    A("| 候选数区间 | 篇数 |")
    A("| --- | --- |")
    for k, v in raw["candidates_hist"].items():
        A(f"| {k} | {v} |")
    A("")
    A("## 4. 候选状态五分类（raw）")
    A("")
    sc = raw["status_counts"]
    tot = raw["num_pred"] or 1
    names = {
        "exact_id": "完全正确（边界+ID）",
        "id_error": "HPO 映射错误（边界对 ID 错）",
        "boundary_error": "边界错（与金标重叠但边界不等）",
        "fp": "多抽/未命中金标（FP）",
    }
    A("| 状态 | 数量 | 占候选比 |")
    A("| --- | --- | --- |")
    for k in ("exact_id", "id_error", "boundary_error", "fp"):
        A(f"| {names[k]} | {sc.get(k, 0)} | {sc.get(k, 0) / tot:.3f} |")
    A("")
    A("## 5. 误报类型清单（FP 四类 + other）")
    A("")
    fr = raw["fp_by_primary_reason"]
    fpt = sum(fr.values()) or 1
    A(f"FP 合计 {sum(fr.values())} 个；能归入四类的占比 = **{raw['fp_four_class_share']:.3f}**，"
      f"other 占比 = {raw['fp_other_share']:.3f}。")
    A("")
    A("| 误报类别 | 数量 | 占 FP 比 | 示例 surface |")
    A("| --- | --- | --- | --- |")
    label = {"negation": "否定", "family_history": "家族史",
             "animal_model": "动物/细胞模型", "method_section": "方法/统计段",
             "other": "其他（未被四类归因）"}
    for k in ("negation", "family_history", "animal_model", "method_section", "other"):
        v = fr.get(k, 0)
        ex = ", ".join(raw["fp_examples"].get(k, [])[:5]).replace("|", "/")
        A(f"| {label[k]} | {v} | {v / fpt:.3f} | {ex} |")
    A("")
    A("> other 多为讨论段的泛化 HPO 词（正文提及但未被标注），并非四类硬负例；"
      "这部分正是阶段三打分头需要靠上下文判别的工作量。")
    A("")
    cv = metrics.get("cv_augmented")
    if cv:
        A(f"## 5bis. 词表补全文档级 {cv['folds']} 折交叉验证（可外推估计）")
        A("")
        A("用其余折 train surface 补词表，在留出折评估——用于估计 A/B 上词表补全能达到的召回，"
          "避免 in-sample 泄漏值。")
        A("")
        A("| 指标 | 值 |")
        A("| --- | --- |")
        A(f"| candidate_recall（覆盖） | {cv['candidate_recall_cover']:.4f} |")
        A(f"| candidate_recall（overlap） | {cv['candidate_recall_overlap']:.4f} |")
        A(f"| recall_exact_span | {cv['recall_exact_span']:.4f} |")
        A(f"| recall_strict | {cv['recall_strict']:.4f} |")
        A("")
    A("## 6. 分支召回（系统性漏检检查）")
    A("")
    br_src = coarse if coarse else raw
    br = sorted(br_src["branch_recall"].items(), key=lambda kv: kv[1]["recall"])
    # UNMAPPED 是「金标本身 identifier=-1、无 HPO 分支」的条目，不算表型分支漏检，单列说明
    weak = [(b, v) for b, v in br
            if b != "UNMAPPED" and v["gold"] >= 10 and v["recall"] < 0.5]
    unm = br_src["branch_recall"].get("UNMAPPED")
    A(f"表型分支共 {len(br) - (1 if unm else 0)} 个。gold>=10 且召回<0.5 的弱分支：**{len(weak)}** 个"
      f"（>0 表示需回阶段一补词表）。")
    if unm:
        A(f"另有 `UNMAPPED`（金标 identifier=-1、本体无对应 term）：{unm['gold']} 个，"
          f"span 覆盖 {unm['recall']:.3f}，不计入表型分支漏检（见 lexicon_unmapped.tsv）。")
    A("")
    A("| 分支 | 金标数 | 覆盖数 | 召回 |")
    A("| --- | --- | --- | --- |")
    for b, v in br[:20]:
        flag = " ⚠" if (b != "UNMAPPED" and v["gold"] >= 10 and v["recall"] < 0.5) else ""
        A(f"| {b}{flag} | {v['gold']} | {v['covered']} | {v['recall']:.3f} |")
    A("")
    A("## 7. 否定标签质量（与金标 note=NO 对齐）")
    A("")
    nt = raw["negation_tag"]
    A(f"precision={nt['precision']} / recall={nt['recall']} "
      f"(tp={nt['tp']}, fp={nt['fp']}, fn={nt['fn']})")
    A("")
    A("## 8. 词表缺口（精确集未覆盖的金标 surface，前 40）")
    A("")
    A(f"精确集未覆盖 = {raw['uncovered_gold_count']}（{raw['uncovered_gold_rate']:.3f}）")
    if coarse:
        A(f"；启用 NP 粗召回后未覆盖 = {coarse['uncovered_gold_count']}"
          f"（{coarse['uncovered_gold_rate']:.3f}）")
    A("")
    A("```")
    A("\n".join(raw["uncovered_gold_examples"]))
    A("```")
    A("")
    A("## 9. Gate 判定")
    A("")
    gate_src = coarse if coarse else raw
    gate_recall = gate_src["candidate_recall"] >= 0.85
    gate_overlap = gate_src["candidate_recall_overlap"] >= 0.85
    gate_branch = len(weak) == 0
    tag = "粗召回集" if coarse else "精确集"
    A(f"- 候选召回（{tag}，覆盖口径）≥ 85%：{'通过 ✅' if gate_recall else '未通过 ❌'}"
      f"（{gate_src['candidate_recall']:.3f}；精确集 {raw['candidate_recall']:.3f}）")
    A(f"- 候选召回（{tag}，overlap 口径）≥ 85%：{'通过 ✅' if gate_overlap else '未通过 ❌'}"
      f"（{gate_src['candidate_recall_overlap']:.3f}；精确集 {raw['candidate_recall_overlap']:.3f}）")
    A(f"- 无整类表型系统性漏检（弱分支=0）：{'通过' if gate_branch else '未通过'}"
      f"（弱分支 {len(weak)} 个）")
    A(f"- 误报可归入四类：四类占比 {raw['fp_four_class_share']:.3f}，"
      f"other {raw['fp_other_share']:.3f}（other 高说明主要是泛化 HPO 词而非硬负例，见 §5）")
    A("")
    A("## 10. 产出文件")
    A("")
    A("- `hpo_subgraph.json` / `lexicon.json`：本体子图与词典（含歧义表）")
    A("- `candidates_train.jsonl` / `candidates_A.jsonl` / `candidates_B.jsonl`：候选+标签（阶段三输入）")
    A(f"- `submit_pred_A.jsonl`（{sub_counts.get('A_raw')} 实体）/ `submit_pred_A_filtered.jsonl`"
      f"（{sub_counts.get('A_filtered')} 实体）")
    A(f"- `submit_pred_B.jsonl`（{sub_counts.get('B_raw')} 实体）/ `submit_pred_B_filtered.jsonl`"
      f"（{sub_counts.get('B_filtered')} 实体）")
    A("- `metrics.json`：全部指标")
    A("")
    A("## 11. 复现命令")
    A("")
    A("```bash")
    A(f'python stage2/run_stage2.py --data "{cfg["data"]}" --out "{cfg["out"]}"')
    A("```")
    A("")
    return "\n".join(lines)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="阶段二：候选生成与无学习基线")
    ap.add_argument("--data", required=True, help="含 jsonl 与 hp.obo 的数据目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 篇 train（调试用）")
    ap.add_argument("--rebuild-lexicon", action="store_true", help="强制重建子图/词典")
    ap.add_argument("--augment-from-train", action="store_true",
                    help="用 train 金标 surface 扩充词典（模拟阶段一词表补全；"
                         "对 train 指标为泄漏，仅作诊断/A-B 提交用）")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="词表补全文档级 K 折交叉验证（仅 ontology 模式；给可外推的召回估计）")
    ap.add_argument("--np-recall", action="store_true",
                    help="启用名词短语粗召回兜底层（抬高候选召回上限，含未映射 span）")
    ap.add_argument("--np-seg-chars", type=int, default=90,
                    help="子句段粗召回最大字符数（0=关闭段层；默认 90）")
    ap.add_argument("--np-max-left", type=int, default=5)
    ap.add_argument("--np-max-right", type=int, default=4)
    ap.add_argument("--lexicon-file", default=None,
                    help="自定义词典 json（如 build_corpus_lexicon 产出的 lexicon_aug.json）；"
                         "默认用 <out>/lexicon.json")
    args = ap.parse_args()

    data, out = args.data, args.out
    os.makedirs(out, exist_ok=True)

    # 1) HPO 子图 + 词典
    obo = os.path.join(data, OBO_FILE)
    lex_path = os.path.join(out, "lexicon.json")
    sub_path = os.path.join(out, "hpo_subgraph.json")
    lex_source = args.lexicon_file or lex_path
    need_build = args.rebuild_lexicon or not os.path.exists(sub_path)
    if need_build:
        print("[1/6] 构建 HPO 子图与词典 ...")
        terms, subgraph, meta, onto_lex, onto_amb = hpo_mod.build(obo, out)
        sub_terms = {tid: {"name": terms[tid].get("name")}
                     for tid in subgraph if tid in terms}
    else:
        print("[1/6] 复用已有子图 ...")
        with open(sub_path, encoding="utf-8") as f:
            sub_json = json.load(f)
        meta = {tid: {"depth": t["depth"], "branch": t["branch"]}
                for tid, t in sub_json["terms"].items()}
        subgraph = set(meta)
        sub_terms = sub_json["terms"]

    if os.path.abspath(lex_source) == os.path.abspath(lex_path) and need_build:
        lexicon, ambiguity = onto_lex, onto_amb
    else:
        with open(lex_source, encoding="utf-8") as f:
            lex_json = json.load(f)
        lexicon = lex_json["surfaces"]
        ambiguity = lex_json.get("ambiguity", {})
        print("       lexicon file:", lex_source)
    lex_stats = {"num_terms": len(subgraph), "num_surfaces": len(lexicon),
                 "num_ambiguous": len(ambiguity)}
    print("       lexicon:", lex_stats)

    np_cfg = None
    if args.np_recall:
        heads = np_mod.derive_head_vocab(sub_terms, min_freq=2)
        np_cfg = {"heads": heads, "seg_chars": args.np_seg_chars,
                  "max_left": args.np_max_left, "max_right": args.np_max_right}
        print(f"       NP heads = {len(heads)}, seg_chars = {args.np_seg_chars}")

    # 2) 读数据
    print("[2/6] 读取数据 ...")
    train = load_jsonl(os.path.join(data, TRAIN_FILE))
    a_docs = load_jsonl(os.path.join(data, A_FILE))
    b_docs = load_jsonl(os.path.join(data, B_FILE))

    # 2.1) 可选：用 train 金标 surface 扩充词典（模拟阶段一词表补全）
    if args.augment_from_train:
        added = 0
        for doc in train:
            text = rebuild_full_text(doc.get("full_text", []))
            for g in gold_from_doc(doc):
                if not g["ids"]:
                    continue
                surf = text[g["start"]:g["end"]].strip()
                if not surf or len(surf) < 3:
                    continue
                for v in hpo_mod.surface_variants(surf):
                    key = v.lower()
                    cur = set(lexicon.get(key, []))
                    if not g["ids"].issubset(cur):
                        cur |= g["ids"]
                        lexicon[key] = sorted(cur)
                        added += 1
        lex_stats["num_surfaces"] = len(lexicon)
        lex_stats["num_ambiguous"] = sum(1 for v in lexicon.values() if len(v) > 1)
        lex_stats["augmented_added"] = added
        print("       augment from train surfaces: added keys =", added)

    if args.limit:
        train = train[: args.limit]

    # 3) 自动机
    print("[3/6] 构建 Aho-Corasick 自动机 ...")
    autom = matcher_mod.build_automaton(lexicon)

    # 3.1) 生成候选
    print("[4/6] 生成候选 ...")

    def process(docs, name):
        tc = {}
        for i, doc in enumerate(docs, 1):
            text, precise, coarse = build_doc_candidates(doc, autom, lexicon, meta, np_cfg)
            tc[str(doc.get("pmc_id"))] = (text, precise, coarse)
            if i % 20 == 0 or i == len(docs):
                print(f"       {name}: {i}/{len(docs)}")
        return tc

    train_tc = process(train, "train")
    a_tc = process(a_docs, "A")
    b_tc = process(b_docs, "B")

    # 5) 评测（精确集=E0 基线；粗召回集=候选召回上限）
    print("[5/6] 评测 ...")
    precise_records = []
    coarse_records = []
    for doc in train:
        pmc = str(doc.get("pmc_id"))
        text, precise, coarse = train_tc[pmc]
        gold = gold_from_doc(doc)
        precise_records.append({"pmc_id": pmc, "text": text, "gold": gold,
                                "candidates": precise})
        if np_cfg:
            coarse_records.append({"pmc_id": pmc, "text": text, "gold": gold,
                                   "candidates": coarse})
    metrics = {
        "raw": eval_mod.evaluate(precise_records, meta),
        "filtered_negfam": eval_mod.evaluate(
            eval_mod.filter_records(precise_records, ["negation", "family_history"]), meta),
        "filtered_all4": eval_mod.evaluate(
            eval_mod.filter_records(
                precise_records, ["negation", "family_history", "animal_model", "method_section"]),
            meta),
    }
    if np_cfg:
        metrics["raw_coarse"] = eval_mod.evaluate(coarse_records, meta)
    if args.cv_folds and not args.augment_from_train:
        print(f"      [cv] 词表补全 {args.cv_folds} 折交叉验证 ...")
        metrics["cv_augmented"] = cv_mod.run_cv(train, lexicon, meta, args.cv_folds)

    # 6) 写产出
    print("[6/6] 写出候选与提交文件 ...")
    write_candidates(os.path.join(out, "candidates_train.jsonl"), train, train_tc)
    write_candidates(os.path.join(out, "candidates_A.jsonl"), a_docs, a_tc)
    write_candidates(os.path.join(out, "candidates_B.jsonl"), b_docs, b_tc)

    four = ["negation", "family_history", "animal_model", "method_section"]
    sub_counts = {}
    sub_counts["A_raw"] = write_submission(
        os.path.join(out, "submit_pred_A.jsonl"), a_docs, a_tc)
    sub_counts["A_filtered"] = write_submission(
        os.path.join(out, "submit_pred_A_filtered.jsonl"), a_docs, a_tc, drop_flags=four)
    sub_counts["B_raw"] = write_submission(
        os.path.join(out, "submit_pred_B.jsonl"), b_docs, b_tc)
    sub_counts["B_filtered"] = write_submission(
        os.path.join(out, "submit_pred_B_filtered.jsonl"), b_docs, b_tc, drop_flags=four)

    cfg = {"data": data, "out": out, "augment": bool(args.augment_from_train),
           "np_recall": bool(args.np_recall), "np_seg_chars": args.np_seg_chars,
           "lexicon_file": args.lexicon_file}
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "lexicon": lex_stats, "submission_counts": sub_counts,
                   **metrics}, f, ensure_ascii=False, indent=2)

    report = render_report(cfg, lex_stats, metrics, sub_counts)
    with open(os.path.join(out, "baseline_report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # 控制台摘要
    raw = metrics["raw"]
    print("[done] 核心指标(raw):")
    print("       candidate_recall =", raw["candidate_recall"])
    print("       precision_span   =", raw["precision_span_exact"],
          " precision_strict =", raw["precision_strict"])
    print("       cand/doc mean    =", raw["candidates_per_doc"]["mean"])
    print("       E0 strict F1     =", raw["e0_strict_all"]["f1"],
          " span F1 =", raw["e0_span_all"]["f1"])
    print("       HPO map accuracy =", raw["hpo_mapping_accuracy"])
    print("       FP four-class    =", raw["fp_four_class_share"],
          " other =", raw["fp_other_share"])
    if np_cfg:
        c = metrics["raw_coarse"]
        print("       [coarse] candidate_recall =", c["candidate_recall"],
              " overlap =", c["candidate_recall_overlap"],
              " cand/doc =", c["candidates_per_doc"]["mean"])
    print("       输出目录:", out)


if __name__ == "__main__":
    main()
