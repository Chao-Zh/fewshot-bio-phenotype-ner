"""实验一：数据集分析与完整数据预处理工程 —— 主入口。

用法:
    # 运行完整预处理管线（自动检测格式）
    python run_pipeline.py --input <你的数据文件>.json --output-dir output

    # 指定格式 / 少样本参数
    python run_pipeline.py --input <你的数据文件>.json --output-dir output \
        --format json --n-way 5 --k-shot 5 --q-query 5 --num-episodes 100

管线兼容：
    - JSON Lines / JSON 数组 / {"data": [...]} 形式
    - BIO / CONLL 序列标注
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medical_ner_preprocess.config import Config
from medical_ner_preprocess.pipeline import PreprocessingPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="实验一：数据集分析与完整数据预处理工程",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="输入数据文件路径 (JSON 或 BIO)")
    p.add_argument("--output-dir", default="output", help="输出目录")
    p.add_argument("--format", default="auto", choices=["auto", "json", "bio"], help="输入格式")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    # 切分
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)

    # 滑动窗口
    p.add_argument("--max-seq-len", type=int, default=512, help="BioBERT 最大序列长度")
    p.add_argument("--overlap", type=int, default=128, help="滑动窗口重叠 token 数")

    # 少样本
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-query", type=int, default=5)
    p.add_argument("--num-episodes", type=int, default=100)

    # Dataloader
    p.add_argument("--batch-size", type=int, default=16)

    # 其他
    p.add_argument("--no-stratify", action="store_true", help="关闭分层划分")
    p.add_argument("--label-set", default="", help="逗号分隔的标签集合（为空则自动推断）")
    p.add_argument("--quiet", action="store_true", help="减少输出")
    return p


def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.input):
        print(f"错误：输入文件不存在: {args.input}")
        sys.exit(1)

    cfg = Config(
        seed=args.seed,
        input_format=args.format,
        verbose=not args.quiet,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_seq_len=args.max_seq_len,
        overlap=args.overlap,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        stratify_by_label=not args.no_stratify,
        label_set=[x.strip() for x in args.label_set.split(",") if x.strip()],
    )

    pipeline = PreprocessingPipeline(cfg)
    result = pipeline.run(args.input, args.output_dir)

    print("\n=== 产出文件 ===")
    for f in result["output_files"]:
        print(f"  {f}")
    print(f"\n=== 统计报告 ===\n  {result['context']['report_path']}")


if __name__ == "__main__":
    main()
