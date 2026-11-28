"""端到端预处理管线编排。"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from .analyzer import analyze, infer_label_set
from .cleaner import Cleaner
from .config import Config
from .dataset import (
    NerDataset, build_dataloaders, build_vocabs,
    episode_to_batch, samples_to_features, TORCH_AVAILABLE,
)
from .episode import EpisodeSampler, build_episode_samplers
from .formats import detect_format, load_dataset, save_bio, save_jsonl
from .reporter import save_report
from .slicer import Slicer
from .splitter import split_dataset


class PreprocessingPipeline:
    """完整预处理管线。

    流程：
        加载 → 原始分析 → 清洗 → 清洗后分析 → 滑动窗口切分
        → 划分 → 特征化 → DataLoader → Episode 采样 → 报告 & 落盘
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.context: Dict = {}

    # ------------------------------------------------------------------
    def run(self, input_path: str, output_dir: str) -> Dict:
        cfg = self.cfg
        os.makedirs(output_dir, exist_ok=True)
        if cfg.verbose:
            print(f"\n{'='*60}\n[pipeline] 开始预处理: {input_path}\n{'='*60}")

        # ---- 1. 加载 ----
        fmt = detect_format(input_path, cfg)
        samples = load_dataset(input_path, cfg, fmt=fmt)
        if not samples:
            raise ValueError("未从输入文件解析到任何样本")
        self.context.update({
            "input_path": input_path,
            "input_format": fmt,
            "seed": cfg.seed,
            "window_size": cfg.window_size,
            "max_seq_len": cfg.max_seq_len,
            "stride": cfg.stride,
            "overlap": cfg.overlap,
            "report_name": cfg.report_name,
        })
        if cfg.verbose:
            print(f"[1/8] 加载完成: {len(samples)} 条样本, 格式={fmt}")

        # ---- 2. 原始分析 ----
        raw_analysis = analyze(samples, "raw")
        self.context["raw_analysis"] = raw_analysis

        # 标签体系
        if not cfg.label_set:
            cfg.label_set = infer_label_set(samples)
            if cfg.verbose:
                print(f"[2/8] 推断标签体系: {cfg.label_set}")
        else:
            if cfg.verbose:
                print(f"[2/8] 使用配置标签体系: {cfg.label_set}")

        # ---- 3. 清洗 ----
        cleaner = Cleaner(cfg)
        clean_samples, clean_stats = cleaner.clean(samples)
        if not clean_samples:
            raise ValueError("清洗后无有效样本")
        self.context["clean_stats"] = clean_stats
        clean_analysis = analyze(clean_samples, "cleaned")
        self.context["clean_analysis"] = clean_analysis
        if cfg.verbose:
            print(f"[3/8] 清洗完成: {len(clean_samples)} 条, 统计={clean_stats}")

        # ---- 4. 滑动窗口切分 ----
        slicer = Slicer(cfg)
        windows, slicer_stats = slicer.slice_all(clean_samples)
        sliced_samples = [w.to_sample() for w in windows]
        coverage = slicer.verify_coverage(clean_samples, windows)
        self.context["slicer_stats"] = slicer_stats
        self.context["coverage"] = coverage
        if cfg.verbose:
            print(f"[4/8] 切分完成: {len(windows)} 个窗口, "
                  f"实体覆盖率={coverage['coverage_rate']*100:.2f}%")

        # ---- 5. 划分 ----
        bundle = split_dataset(sliced_samples, cfg)
        self.context["split_stats"] = bundle.split_stats
        self.context["label_set"] = bundle.label_set

        # ---- 6. 特征化 + DataLoader ----
        token_vocab, label_vocab = build_vocabs(sliced_samples, bundle.label_set)
        bundle_features = {}
        for split, subset in (("train", bundle.train), ("val", bundle.val), ("test", bundle.test)):
            bundle_features[split] = samples_to_features(subset, token_vocab, label_vocab, cfg)
        loaders = build_dataloaders(bundle_features, token_vocab, label_vocab, cfg)

        dl_stats = {}
        for split, loader in loaders.items():
            # 迭代一个 batch 验证可用性
            first_batch = next(iter(loader))
            if TORCH_AVAILABLE and hasattr(first_batch["input_ids"], "shape"):
                shape = tuple(first_batch["input_ids"].shape)
            else:
                shape = (len(first_batch["input_ids"]), len(first_batch["input_ids"][0]))
            dl_stats[split] = {
                "num_features": len(bundle_features[split]),
                "num_batches": len(loader),
                "first_batch_shape": shape,
                "backend": "torch" if loader.use_torch else "numpy",
            }
        self.context["dataloader_stats"] = dl_stats
        self.context["vocab_sizes"] = {
            "token_vocab": len(token_vocab),
            "label_vocab": len(label_vocab),
        }
        if cfg.verbose:
            print(f"[5-6/8] 特征化 & DataLoader 完成 "
                  f"(backend={'torch' if TORCH_AVAILABLE else 'numpy'})")

        # ---- 7. Episode 采样 ----
        samplers = build_episode_samplers(bundle, cfg)
        episode_stats = {"per_split": {}}
        sample_episode_repr = None
        for name, sampler in samplers.items():
            st = sampler.stats()
            episode_stats["per_split"][name] = st
            if name == "train":
                try:
                    ep = sampler.sample_episode()
                    sample_episode_repr = ep.summary()
                    # 验证 episode 可转为 batch
                    batch = episode_to_batch(ep, token_vocab, label_vocab, cfg)
                    sample_episode_repr["batch_support_shape"] = (
                        tuple(batch["support"]["input_ids"].shape)
                        if TORCH_AVAILABLE and hasattr(batch["support"]["input_ids"], "shape")
                        else "numpy"
                    )
                except ValueError as e:
                    if cfg.verbose:
                        print(f"[episode] 跳过示例采样: {e}")
        episode_stats.update({
            "n_way": cfg.n_way,
            "k_shot": cfg.k_shot,
            "q_query": cfg.q_query,
            "num_episodes": cfg.num_episodes,
            "sample_episode": sample_episode_repr,
        })
        self.context["episode_stats"] = episode_stats
        if cfg.verbose:
            print(f"[7/8] Episode 采样器构建完成 "
                  f"({cfg.n_way}-way {cfg.k_shot}-shot)")

        # ---- 8. 落盘 & 报告 ----
        out = {
            "raw": os.path.join(output_dir, "00_raw_standardized.jsonl"),
            "cleaned": os.path.join(output_dir, "01_cleaned_standardized.jsonl"),
            "sliced": os.path.join(output_dir, "02_sliced_windows.jsonl"),
            "train": os.path.join(output_dir, "train.jsonl"),
            "val": os.path.join(output_dir, "val.jsonl"),
            "test": os.path.join(output_dir, "test.jsonl"),
            "train_bio": os.path.join(output_dir, "train.bio"),
            "features": os.path.join(output_dir, "features.jsonl"),
            "vocab": os.path.join(output_dir, "vocab.json"),
        }
        save_jsonl(samples, out["raw"])
        save_jsonl(clean_samples, out["cleaned"])
        save_jsonl(sliced_samples, out["sliced"])
        save_jsonl(bundle.train, out["train"])
        save_jsonl(bundle.val, out["val"])
        save_jsonl(bundle.test, out["test"])
        save_bio(bundle.train, out["train_bio"])

        # 特征（仅前若干条写出，避免文件过大；完整特征由 Dataloader 动态生成）
        import json
        with open(out["features"], "w", encoding="utf-8") as f:
            for split, feats in bundle_features.items():
                for feat in feats[:50]:
                    d = feat.to_dict()
                    d["split"] = split
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

        with open(out["vocab"], "w", encoding="utf-8") as f:
            json.dump({
                "token_vocab": token_vocab.token2id,
                "label_vocab": label_vocab.label2id,
                "id2label": label_vocab.id2label,
            }, f, ensure_ascii=False, indent=2)

        cfg.save_json(os.path.join(output_dir, "config.json"))

        self.context["output_files"] = list(out.values()) + [
            os.path.join(output_dir, "config.json"),
            os.path.join(output_dir, cfg.report_name),
            os.path.join(output_dir, "statistics.json"),
        ]
        report_path = save_report(self.context, output_dir)
        self.context["report_path"] = report_path

        if cfg.verbose:
            print(f"[8/8] 落盘 & 报告完成\n{'='*60}\n[pipeline] 全部完成 ✓")
            print(f"  统计报告: {report_path}")
            print(f"  标准化数据目录: {output_dir}")

        return {
            "context": self.context,
            "bundle": bundle,
            "loaders": loaders,
            "samplers": samplers,
            "vocabs": (token_vocab, label_vocab),
            "output_files": self.context["output_files"],
        }
