"""用真实 PatientPheX 数据跑完整预处理管线。

与 run_pipeline.py 的区别:
- 关闭空白规范化(真实数据 offset 已精确对齐, 压缩空白会致 span 漂移)
- 标签体系固定为 Phenotype(官方单类), episode 采样降级为 1-way 并如实记录
- 保留 HPO identifier / note / patient association 元数据

用法:
    python run_real_pipeline.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

from medical_ner_preprocess.config import Config
from medical_ner_preprocess.formats import load_dataset, save_jsonl
from medical_ner_preprocess.analyzer import analyze
from medical_ner_preprocess.cleaner import Cleaner, validate_bio
from medical_ner_preprocess.slicer import Slicer
from medical_ner_preprocess.splitter import split_dataset
from medical_ner_preprocess.episode import EpisodeSampler
from medical_ner_preprocess.dataset import (
    build_vocabs,
    samples_to_features,
    NerDataset,
    NerDataLoader,
    episode_to_batch,
    TORCH_AVAILABLE,
)
from medical_ner_preprocess.reporter import build_report

BASE = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(BASE, "data", "real")
OUT_DIR = os.path.join(BASE, "output_real")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = Config(
        seed=42,
        label_set=["Phenotype"],
        normalize_whitespace=False,   # 关键: 保护 offset
        n_way=1,                       # 官方单类, N-way 采样退化为 1-way
        k_shot=5,
        q_query=5,
        num_episodes=20,
        batch_size=16,
        overlap=128,
    )
    cfg.save_json(os.path.join(OUT_DIR, "config.json"))

    print("[1/7] 载入真实数据 ...")
    samples = load_dataset(
        os.path.join(REAL_DIR, "train_standardized.jsonl"), cfg, fmt="json"
    )
    raw_stats = analyze(samples, "json")

    print("[2/7] 清洗 ...")
    clean, clean_stats = Cleaner(cfg).clean(samples)
    save_jsonl(clean, os.path.join(OUT_DIR, "01_cleaned.jsonl"))
    print("      ", clean_stats)

    print("[3/7] 滑动窗口切分 ...")
    slicer = Slicer(cfg)
    windows, slice_stats = slicer.slice_all(clean)
    coverage = slicer.verify_coverage(clean, windows)
    print(f"       windows={len(windows)}, coverage={coverage['coverage_rate']}")
    sliced = [w.to_sample() for w in windows]
    save_jsonl(sliced, os.path.join(OUT_DIR, "02_sliced_windows.jsonl"))

    print("[4/7] 划分 train/val/test ...")
    bundle = split_dataset(sliced, cfg)
    for name in ("train", "val", "test"):
        save_jsonl(getattr(bundle, name), os.path.join(OUT_DIR, f"{name}.jsonl"))
    print("      ", bundle.split_stats)

    print("[5/7] 词表与特征化 ...")
    token_vocab, label_vocab = build_vocabs(sliced, bundle.label_set)
    with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "token2id": token_vocab.token2id,
                "label2id": label_vocab.label2id,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    loaders = {}
    dataloader_stats = {}
    for name in ("train", "val", "test"):
        feats = samples_to_features(getattr(bundle, name), token_vocab, label_vocab, cfg)
        loader = NerDataLoader(
            NerDataset(feats),
            batch_size=cfg.batch_size,
            pad_id=token_vocab.pad_id,
            shuffle=(name == "train"),
            use_torch=True,
        )
        loaders[name] = loader
        batch = next(iter(loader))
        shape = (
            tuple(batch["input_ids"].shape)
            if TORCH_AVAILABLE and hasattr(batch["input_ids"], "shape")
            else [len(batch["input_ids"]), len(batch["input_ids"][0])]
        )
        dataloader_stats[name] = {
            "num_features": len(feats),
            "num_batches": len(loader),
            "first_batch_shape": shape,
            "backend": "torch" if loader.use_torch else "numpy",
        }
        print(f"       {name}: batches={len(loader)}, shape={shape}")

    print("[6/7] Episode 采样 (1-way, 官方单类限制) ...")
    sampler = EpisodeSampler(
        bundle.train, cfg, label_set=bundle.label_set
    )
    episodes = [sampler.sample_episode(n_way=1, k_shot=5, q_query=5) for _ in range(20)]
    sample_ep = episodes[0]
    ep_batch = episode_to_batch(sample_ep, token_vocab, label_vocab, cfg)
    episode_stats = {
        "note": "PatientPheX 官方仅 Phenotype 一类, N-way 少样本退化为 1-way; "
                "若需标准 N-way 实验应使用多类 NER 数据集(如样例数据的 6 类)",
        "num_episodes": len(episodes),
        "sample_episode": sample_ep.summary(),
        "support_batch_shape": tuple(ep_batch["support"]["input_ids"].shape)
        if TORCH_AVAILABLE and hasattr(ep_batch["support"]["input_ids"], "shape")
        else None,
        "query_batch_shape": tuple(ep_batch["query"]["input_ids"].shape)
        if TORCH_AVAILABLE and hasattr(ep_batch["query"]["input_ids"], "shape")
        else None,
    }
    print("      ", episode_stats["sample_episode"])

    print("[7/7] 生成报告 ...")
    stats = {
        "dataset": "PatientPheX (CHIP2026 Task 1) - 真实数据",
        "input_format": "json",
        "raw_analysis": raw_stats,
        "cleaner_stats": clean_stats,
        "slicer_stats": slice_stats,
        "coverage": coverage,
        "split_stats": bundle.split_stats,
        "label_set": bundle.label_set,
        "dataloader_stats": dataloader_stats,
        "episode_stats": episode_stats,
    }
    with open(os.path.join(OUT_DIR, "statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    # 人读报告(用内置 reporter, 若字段不匹配则降级为简化版)
    try:
        md = build_report(stats)
    except Exception as e:
        print(f"      [warn] 内置报告生成失败({e}), 使用简化版")
        md = _simple_markdown(stats)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n完成! 所有输出在: {OUT_DIR}")


def _simple_markdown(stats: dict) -> str:
    import json as _json
    lines = ["# PatientPheX 真实数据预处理报告", ""]
    for k, v in stats.items():
        lines.append(f"## {k}")
        lines.append("```json")
        lines.append(_json.dumps(v, ensure_ascii=False, indent=2, default=str)[:3000])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
