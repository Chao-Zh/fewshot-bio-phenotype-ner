# 阶段二交付说明：候选生成与无学习基线（PatientPheX / CHIP2026 子任务1）

> 本包为阶段二的完整交付：代码 + 两次运行的产物 + 说明文档。**原始数据不在包内**（保密），按 §6 放好即可复现。
> 复现依赖：**HPO 2026-06-23 版** + `pyahocorasick==2.3.1`（见 `code/requirements.txt`）。

## 1. 包内容

```
stage2_delivery/
├── README.md            # 方法、设计决策、指标定义、复现命令（开发/方法说明）
├── DELIVERY.md          # 本交付说明
├── MANIFEST.md          # 文件清单 + 大小 + SHA256
├── code/                # 全部脚本 + requirements.txt（见 §7）
├── output/              # 诚实基线：ontology-only + 名词短语粗召回上限 + 5折CV + 规模曲线
└── output_lex/          # 正式产物：补词表后（lexicon.tsv）+ 候选 + 提交 + 报告
```

> 范围说明：本包是**阶段二**交付。阶段一其余 required 产物（`annotations.jsonl` 四元组、`data_report.md`、
> 偏移映射表）由阶段一交付，本包不重复；阶段一要求的 `hpo_subgraph.json` 与 `lexicon.tsv` 已在包内。

### output/（ontology-only 基线 + 诊断）
| 文件 | 说明 |
| --- | --- |
| `baseline_report.md` | 三指标、候选数分布、五类错误、FP 清单、分支召回、CV、Gate |
| `metrics.json` | 全部指标（raw / filtered / raw_coarse / cv_augmented） |
| `hpo_subgraph.json` | HP:0000118 子图（19087 term，name/def/synonym/alt_id/depth/branch） |
| `lexicon.json` | ontology-only 词表（93279 surface）+ 歧义表 |
| `candidates_{train,A,B}.jsonl` | 候选（精确集 + 粗召回集，带 `source` 字段） |
| `submit_pred_{A,B}[_filtered].jsonl` | E0 提交格式基线 |
| `cv_recall.json` | 文档级 5 折交叉验证（可外推召回估计） |
| `lexicon_scale_recall.{json,md}` | 补词规模-外推召回曲线 |
| `exp_link_recall.json` / `exp_np_recall.json` | 链接/粗召回 诊断结果（聚合数字） |

### output_lex/（正式交付：补词表后）
| 文件 | 说明 |
| --- | --- |
| `baseline_report.md` / `metrics.json` | 补词表后的阶段二报告与指标 |
| **`lexicon.tsv`** | **PDF 阶段一要求的词表交付**：surface / hpo_ids / source / 歧义（96,919 行） |
| `lexicon_aug.json` | 合并词表（匹配器用；num_surfaces=96,919，歧义 211） |
| `lexicon_workorder.tsv` | 补词工单（1,530 条去重原始 surface） |
| `lexicon_unmapped.tsv` | id=-1 无映射金标 surface 清单（97 种） |
| `unmapped_gold_worklist.tsv` | 109 个 id=-1 金标实体的人工补标工单（带 pmc/offset/上下文） |
| `lexicon_aug_report.md` | 补词前后召回对比与统计 |
| `candidates_{train,A,B}.jsonl` | 阶段三训练/推理候选 |
| `submit_pred_{A,B}[_filtered].jsonl` | E0 提交（仅精确词典候选） |
| `cv_recall.json` | 5 折外推召回估计（与 output/ 同一实验的复制件，见 §3） |

## 2. 关键指标（80 篇 / 6027 金标，in-sample）

| 运行 | cover | overlap | 候选精确率(span) | 候选/篇 | E0 strict F1 | HPO 映射准确率 |
| --- | --- | --- | --- | --- | --- | --- |
| ontology-only（output/） | 0.6232 | 0.6942 | 0.8499 | 54.6 | 0.7074 | 0.9900 |
| + 名词短语粗召回（span 上限，含未映射 span） | 0.8752 | 0.9325 | — | 286.2 | — | — |
| **补词表 lexicon_aug（output_lex/，in-sample）** | **0.9743** | **0.9754** | 0.8667 | 82.7 | **0.9056** | 0.9984 |

**口径结论（同时看利好与不足）**：
- 训练集（in-sample）补词后 **带 ID 召回 0.974 过 85% Gate ✅**；弱分支 0（`id=-1` 的 143 条单列）。
- **误报归入四类仅部分满足 ⚠**：FP 中四类硬负例仅占 **22.8%**，主体是讨论段泛化 HPO 词（详见各 `baseline_report.md §5`）。
- 外推见 §3：**span 召回可过 85%，但 ID 正确（strict）约 0.72，未过 85%**。

> 指标定义（cover/overlap/strict/precision/E0）：见 `README.md §指标定义`。

## 3. 外推评测协议与结果（**两套协议，不可混用**）

### 3.1 文档级 5 折 CV（主口径）
- 协议：按文档序号 `i % 5` 分 5 折；每折留出，用**其余折的金标 surface** 扩充 **ontology 基线词表**，在留出折测召回；分折明细见 `cv_recall.json` 的 `per_fold`。
- 结果（汇总）：**cover 0.6745 / overlap 0.7546 / strict 0.6585**。
- 注意：`output_lex/cv_recall.json` 与 `output/cv_recall.json` 内容相同（同一实验复制件），其 base 是 ontology 词表、**不是** output_lex 的全量词表；故 **output_lex 的 in-sample 0.974 不能当作外推**。

### 3.2 补词规模曲线（仅示趋势，`output/lexicon_scale_recall.md`）
固定 20 篇留出，k=用于补词表的文档数，5 次重复均值：

| k | 0 | 5 | 10 | 20 | 40 | 60 |
| --- | --- | --- | --- | --- | --- | --- |
| cover | 0.673 | 0.685 | 0.689 | 0.696 | 0.703 | 0.708 |
| overlap | 0.742 | 0.755 | 0.761 | 0.770 | 0.782 | 0.790 |
| strict | 0.655 | 0.667 | 0.670 | 0.677 | 0.684 | 0.688 |

→ 补词表在未见文档上饱和在 cover≈0.71 / overlap≈0.79，**单靠补词表到不了 85%**。

### 3.3 名词短语粗召回（无金标）留出集 span 召回
`output/lexicon_scale_recall.json` 的 `np_coarse_heldout`：**cover 0.9095 / overlap 0.9573**（含未映射 span，需链接才可供阶段三用）。

### 3.4 轻量实体链接（`output/exp_link_recall.json`）
留出集（1381 金标）：

| 配置 | cover | overlap | strict |
| --- | --- | --- | --- |
| 补词表 | 0.7082 | 0.7900 | 0.6879 |
| +NP 继承 ID | 0.9131 | 0.9616 | 0.6995 |
| +NP 轻量链接（thr=0.4） | 0.9131 | 0.9616 | **0.7190** |

链接在难例上 top-1 仅 **0.234**（top-3 0.331）→ 轻量词面链接不足以把外推 **ID 正确** 召回抬到 85%，需语义/embedding 链接（属阶段三）或更多标注。

## 4. 交付给阶段三用什么
- `output_lex/candidates_{train,A,B}.jsonl`：候选（start/end/text/ids/primary/ambiguous/source/flags/primary_reason/section_type/note）。
- `output_lex/lexicon_aug.json` + `lexicon.tsv`：拼 D0（名称）/D1（+同义词）/D2（+definition）输入。
- `output_lex/baseline_report.md` + `metrics.json`：E0 对照与指标基线。
- 诚实评测建议用 `output/`（ontology-only）或 §3 的外推口径。

## 5. 复现命令

```bash
pip install -r code/requirements.txt   # pyahocorasick==2.3.1；Python>=3.8
# 数据目录需含：PatientPheX-train.jsonl / PatientPheX-A.jsonl / PatientPheX-V1-B.jsonl / hp.obo(2026-06-23)

# 1) 诚实基线 + 粗召回 + 5折CV -> output/
python code/run_stage2.py --data "<数据目录>" --out "<本包>/output" --np-recall --cv-folds 5

# 2) 补词表 -> output_lex/lexicon.tsv / lexicon_aug.json
python code/build_corpus_lexicon.py --data "<数据目录>" --out "<本包>/output_lex" \
    --ontology-lexicon "<本包>/output/lexicon.json" --subgraph "<本包>/output/hpo_subgraph.json"

# 3) 用补后词表跑阶段二 -> output_lex/ 候选+提交+报告
python code/run_stage2.py --data "<数据目录>" --out "<本包>/output_lex" \
    --lexicon-file "<本包>/output_lex/lexicon_aug.json"
#    注：output_lex/ 无 hpo_subgraph.json 时，run_stage2 会自动重建子图（较慢）；可先把 output/hpo_subgraph.json 复制过来加速。

# 4) 校验
python code/validate_outputs.py --data "<数据目录>" --out "<本包>/output_lex"
```

## 6. 数据与保密
- **包内不含原始数据**（PatientPheX 受保密条款约束）。复现时把 4 个数据文件放到同一目录，替换 `<数据目录>`。
- **本体版本**：`hp.obo` 必须为 **HPO 2026-06-23（HP:0000118 分支）**；`hpo_subgraph.json` / `lexicon*.json` / `lexicon.tsv` 均基于此版本，混用其它版本会破坏 ID 一致性。
- **公开仓库/外发限制**：`candidates_*.jsonl`、`submit_pred_*.jsonl`、`lexicon_aug.json`、`lexicon.tsv`、`*_workorder/unmapped`、含金标 surface 的报告均**由保密数据派生**，**不得上传公共仓库或对外分发**；仅可外发代码、说明文档、`requirements.txt` 与不含文本的聚合数字（见包外的 GitHub 上传目录）。

## 7. 代码清单（code/）
| 文件 | 作用 |
| --- | --- |
| `requirements.txt` | 依赖固定（pyahocorasick==2.3.1） |
| `run_stage2.py` | 编排入口（匹配→合并→打标签→评测→写产出） |
| `hpo.py` | 解析 hp.obo → HP:0000118 子图 + 词表 + 歧义表 |
| `matcher.py` | Aho-Corasick 多模式匹配 + 词边界 + 重叠合并 |
| `rules.py` | 四类硬负例上下文打标签（否定/家族史/动物细胞/方法统计段） |
| `evaluate.py` | 三指标 + E0 F1 + 五类错误 + 分支召回 + 诊断 |
| `np_recall.py` | 名词短语/子句段粗召回（抬召回上限） |
| `linker.py` | 轻量实体链接（token 重叠，诊断） |
| `build_corpus_lexicon.py` | 补词表 → lexicon.tsv / lexicon_aug.json / 工单 / 报告 |
| `cv_lexicon_recall.py` | 词表补全文档级 5 折 CV |
| `lexicon_scale_recall.py` | 补词规模-外推召回曲线 |
| `export_unmapped_worklist.py` | 导出 id=-1 金标补标工单 |
| `exp_np_recall.py` / `exp_link_recall.py` | 粗召回 / 链接 的调参与效果度量（参数化，`--data/--out`） |
| `validate_outputs.py` | 产出校验（pmc 覆盖/唯一、offset 零漂移、字段合法） |

## 8. 校验状态
- `validate_outputs.py` 对 `output/` 与 `output_lex/`：**VALIDATION OK**（pmc 全覆盖无重复、offset 零漂移、字段合法）。
- `MANIFEST.md`：覆盖除自身外全部文件；SHA256 校验 0 处不符（以本文件整改后重生成的清单为准）。

## 9. 已知限制与口径
1. **补词表为 in-sample（偏乐观）**；外推主报 5 折 CV（0.675/0.755/0.659），规模曲线仅示趋势（k=60 为 0.708/0.790/0.688），两协议不可直接比较。
2. **「误报归入四类」仅部分满足**：FP 中四类硬负例占 22.8%，主体是讨论段泛化 HPO 词。
3. **否定标签** precision≈0.40、recall≈0.885（仅辅助，不参与 strict F1）。
4. **外推 ID 正确召回 ~0.72**，轻量链接不足，需语义链接。
5. 补词规模两处数字口径：`lexicon_aug_report.md` 的「corpus 新增 3640」= 含变体的词表键；`lexicon_workorder.tsv` 的 1,530 行 = 去重后的原始 surface。
