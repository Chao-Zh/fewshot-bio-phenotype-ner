"""medical_ner_preprocess
=======================

天池医学表型识别数据集 —— 完整数据预处理工程。

模块概览:
    config      配置与超参
    schema      统一数据模型 (Sample / Entity / Feature)
    formats     JSON / BIO-CONLL 自动检测与解析
    analyzer    数据集结构 / 标签体系 / 实体分布分析
    cleaner     BIO 校验 / 脏数据清洗 / 特殊医学符号过滤
    slicer      长文本滑动窗口切分 (BioBERT 512 限制)
    splitter    训练集/验证集/测试集划分
    episode     Few-NERD (ACL2021) N-way K-shot episode 采样
    dataset     PyTorch 兼容 Dataset / DataLoader (含 numpy 回退)
    reporter    样本统计报告
    pipeline    端到端编排

用法:
    python run_pipeline.py --input <你的数据文件>.json --output-dir output
"""

__version__ = "1.0.0"

from .schema import Entity, Sample, Feature, Episode, DatasetBundle

__all__ = [
    "Entity",
    "Sample",
    "Feature",
    "Episode",
    "DatasetBundle",
    "__version__",
]
