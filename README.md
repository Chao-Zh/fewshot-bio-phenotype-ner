# 实验一：数据集分析与完整数据预处理工程

面向「天池医学表型识别」NER 任务的一站式数据预处理工程，对应整体流程中的**数据预处理**阶段。

## 核心能力

| 工作项                           | 实现模块                         | 说明                                                                             |
| ----------------------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| 1. 数据集结构解析、标签体系、实体分布分析        | `formats.py` / `analyzer.py` | 自动检测 JSON / BIO 格式；统计样本数、实体类别分布、长度分布、共现关系                                      |
| 2. 医学长文本滑动窗口切分                | `slicer.py`                  | 解决 BioBERT 512 上下文限制；窗口 510 tokens、stride 382、overlap 128；保证 BIO 合法与实体 100% 覆盖 |
| 3. BIO 校验、脏数据清洗、医学符号过滤        | `cleaner.py`                 | 控制字符过滤、BMES→BIO 归一、span 越界裁剪、重叠消解、空标签丢弃                                        |
| 4. 少样本 episode 采样数据集          | `episode.py`                 | Few-NERD (ACL2021) 范式 N-way K-shot；support/query 划分；train/val/test 独立采样器       |
| 5. 训练集/验证集/测试集划分              | `splitter.py`                | 按主标签分层划分，各类别在三集合中分布均匀                                                          |
| 产出：标准化数据集 / Dataloader / 统计报告 | `dataset.py` / `reporter.py` | PyTorch Dataset + DataLoader（无 torch 时自动 numpy 回退）；Markdown + JSON 双份报告        |

## 目录结构

```
.
├── medical_ner_preprocess/     # 核心包
│   ├── config.py               # 配置与超参
│   ├── schema.py               # 统一数据模型
│   ├── formats.py              # 格式自动检测与解析
│   ├── analyzer.py             # 数据分析
│   ├── cleaner.py              # 清洗
│   ├── slicer.py               # 滑动窗口
│   ├── splitter.py             # 集合划分
│   ├── episode.py              # Few-NERD episode 采样
│   ├── dataset.py              # Dataset / DataLoader
│   ├── reporter.py             # 统计报告
│   └── pipeline.py             # 端到端编排
├── patientpheX_adapter.py      # 真实 PatientPheX JSONL → Sample 转换器
├── run_real_pipeline.py        # 真实数据全流程入口（80篇）
├── build_200_dataset.py        # 构建 200 篇带标注数据集（4:1:5 划分）
├── run_pipeline.py             # 通用管线 CLI（接受任意 JSON/BIO 输入）
├── requirements.txt            # 依赖说明（核心包仅需标准库）
└── README.md
```

## 快速开始（真实 PatientPheX 数据）

```bash
# 1. 把天池下载的 JSONL 转成标准格式（路径换成你本机实际路径）
python patientpheX_adapter.py \
    --train "D:/.../PatientPheX-train.jsonl" \
    --dev   "D:/.../PatientPheX-A.jsonl" \
    --out-dir data/real

# 2. 跑完整预处理管线（清洗 → 切窗 → 划分 → episode → Dataloader → 报告）
python run_real_pipeline.py

# 3. （可选）构建 200 篇带标注数据集（4:1:5 划分）
#    需要 train / A榜 / B榜 原文 + A/B 组预测文件放在同一目录
python build_200_dataset.py --data-root "D:/.../PatientPheX"
#    也可逐个指定：--train --a-raw --a-pred --b-raw --b-pred
#    或设环境变量 PATIENTPHEX_ROOT
```

产出在 `output_real/`（80 篇管线）与 `output_full/`（200 篇数据集）；其中 `report.md`、`statistics.json`、`dataset_report.md` 为统计结果，其余数据文件不纳入版本控制。

## 交接说明（给接手者）

本仓库**只含代码与聚合统计报告**，不含任何原始数据。接手后需自行准备以下文件并放到同一目录：

| 文件 | 说明 | 来源 |
| --- | --- | --- |
| `PatientPheX-train.jsonl` | 80 篇训练集原文 + 人工标注 | 天池数据集包 |
| `PatientPheX-A.jsonl` | A 榜 20 篇测试集原文（无标注） | 天池数据集包 |
| `PatientPheX-V1-B.jsonl` | B 榜 100 篇测试集原文（无标注） | 天池数据集包 |
| `submit_pred_A.jsonl` | A 组 AI 标注预测 | 本项目产出，需向数据持有者索取 |
| `submit_pred_B.jsonl` | B 组 AI 标注预测 | 本项目产出，需向数据持有者索取 |
| `hp.obo` | HPO 本体（评测指定 2026-06-23 版） | 天池数据集包 / HPO 官网 |

准备就绪后按上面"快速开始"三步即可复现全部产出。

> ⚠️ 数据受天池保密条款约束，**不得上传公共仓库或对外分发**。仓库的 `.gitignore` 已全局屏蔽 `*.jsonl`、`*.obo`、`*.csv` 等数据文件，请勿强行提交。

> ⚠️ 已知局限：`run_real_pipeline.py` 采用"先切窗、后按窗口随机划分"流程，同一篇文献的窗口可能跨 train/val/test（存在数据泄漏）；`build_200_dataset.py` 的 4:1:5 划分按文献来源划分，无此问题。后续实验建议优先使用后者，或改造 splitter 为按 `source` 分组划分。

## 接入其他格式数据

`run_pipeline.py` 是通用 CLI，可对任意符合以下格式的 JSON/BIO 数据跑同一套管线：

### 1. JSON（自动兼容三种布局）

```jsonc
// 布局 A：JSON Lines，每行一条
{"text": "患者多饮多尿，诊断为2型糖尿病", "entities": [
    {"start": 2, "end": 4, "label": "症状"},
    {"start": 9, "end": 14, "label": "疾病"}
]}

// 布局 B：整个文件是数组 [...]
// 布局 C：{"data": [...]}
```

字段名自动兼容：`text/content/sentence/原文`；`entities/entity/labels/spans/标注结果`；  
实体字段兼容 `start/start_offset/begin`、`end/end_offset/stop`、`label/type/label_type/category`。  
也支持 `{"疾病": [[9,14]], "症状": [[2,4]]}` 这种按类别分组的写法。

### 2. BIO / CONLL

```
患  O
者  O
多  B-症状
饮  I-症状
，  O
诊  O
断  O
为  O
2  B-疾病
型  I-疾病
糖  I-疾病
尿  I-疾病
病  I-疾病
（空行分隔句子）
```

若真实数据有特殊字段名或标签别名，可在 `Config` 中配置 `json_text_keys`、`json_entities_keys`、`label_alias` 等。

## 产出文件说明

运行管线后会在 `--output-dir` 指定目录生成以下文件（该目录已加入 `.gitignore`，不纳入版本控制，可随时重新生成）：

| 文件                                         | 说明                                            |
| ------------------------------------------ | --------------------------------------------- |
| `00_raw_standardized.jsonl`                | 原始格式统一后的标准化数据                                 |
| `01_cleaned_standardized.jsonl`            | 清洗后的标准数据（含 `cleaning_notes`）                  |
| `02_sliced_windows.jsonl`                  | 滑动窗口切分后的数据（含 `char_start/char_end/window_id`） |
| `train.jsonl` / `val.jsonl` / `test.jsonl` | 划分后的三集合                                       |
| `train.bio`                                | BIO 格式训练集（便于导入其他工具）                           |
| `features.jsonl`                           | 特征样例（前 50 条/集合），完整特征由 Dataloader 动态生成         |
| `vocab.json`                               | 字符词表 + BIO 标签词表                               |
| `config.json`                              | 本次运行的完整配置（便于复现）                               |
| `sample_statistics_report.md`              | **样本统计报告**（人类可读）                              |
| `statistics.json`                          | 统计数据（程序可读）                                    |

> 真实数据管线（`run_real_pipeline.py`）的实际输出文件名略有不同：`01_cleaned.jsonl`、`02_sliced_windows.jsonl`、`train/val/test.jsonl`、`vocab.json`、`config.json`、`report.md`、`statistics.json`。200 篇数据集（`build_200_dataset.py`）输出到 `data/full/` 与 `output_full/`。

## 在训练代码中使用 Dataloader

```python
from medical_ner_preprocess.pipeline import PreprocessingPipeline
from medical_ner_preprocess.config import Config

result = PreprocessingPipeline(Config(seed=42)).run("<你的数据路径>.json", "output")
loaders = result["loaders"]          # {'train': NerDataLoader, 'val': ..., 'test': ...}
samplers = result["samplers"]        # Few-NERD episode 采样器

for batch in loaders["train"]:
    input_ids = batch["input_ids"]       # torch.LongTensor 或 list
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]             # BIO id，-100 为忽略位
    # ... 送入 BioBERT 等模型

# 少样本 episode 训练
for episode in samplers["train"].iter_episodes(num_episodes=100):
    ways = episode.ways                  # 本 episode 的 N 个类别
    support = episode.support            # N*K 条支持集
    query = episode.query                # N*Q 条查询集
```

> 环境已安装 torch 时使用原生 PyTorch Dataset/DataLoader；否则自动回退到 numpy 实现，API 一致。

## 依赖

- Python 3.8+
- 必需：标准库（numpy 可选，用于部分统计加速）
- 可选：`torch`（安装后自动启用 PyTorch DataLoader）

```bash
pip install torch  # 可选，CPU 版：pip install torch --index-url https://download.pytorch.org/whl/cpu
```
