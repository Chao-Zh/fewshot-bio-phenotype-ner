# 阶段二：候选生成与无学习基线（PatientPheX / CHIP2026 子任务1）

> 一条「零参数」流水线：HPO 子图 → 词典 → Aho-Corasick 全文匹配 → 重叠合并 → 四类硬负例打标签 →
> 在 80 篇训练金标上量化 **candidate recall / 候选精确率 / 每篇候选数**，产出阶段三可直接消费的候选集。
>
> 依赖：**HPO 2026-06-23（HP:0000118 分支）**、`pyahocorasick==2.3.1`、Python ≥ 3.8（见 `requirements.txt`）。

## 结论速览（80 篇训练集，6027 个金标实体）

| 运行 | candidate recall (覆盖) | recall (overlap) | 候选精确率 (span) | 候选/篇 | E0 strict F1 | HPO 映射准确率 |
| --- | --- | --- | --- | --- | --- | --- |
| 精确集（词典匹配，可映射） | 0.6232 | 0.6942 | 0.8499 | 54.64 | 0.7074 | 0.9900 |
| **+ 名词短语粗召回（召回上限，无金标）** | **0.8752** | **0.9325** | — | 286.23 | — | — |
| **补词表 lexicon_aug（in-sample）** | **0.9743** | **0.9754** | 0.8667 | 82.69 | 0.9056 | 0.9984 |
| 词表补全 5 折 CV（可外推估计） | 0.6745 | 0.7546 | — | — | — | — |

- **召回已过 85% Gate**：精确集只有 0.623，加上**无金标泄漏的名词短语/子句段粗召回层**后
  覆盖召回 **0.875**、overlap **0.933**，且粗召回集**弱分支为 0**、未覆盖金标从 37.7% 降到 12.5%。
- **两层候选**：`source=lexicon`（精确、带 ID，供阶段三闭集打分 / E0，精确率 0.85）；
  `source=np / np_unmapped`（粗召回外层，`np_unmapped` 无 ID，只抬召回上限，不进提交）。
  粗召回让候选数升到 ~286/篇（对 6027 金标约 1:3 正负比，适合阶段三负采样）。
- **噪声结构**：精确集里 84.1% 边界+ID 完全正确，10.3% 边界错、0.8% ID 错、4.7% FP —— 瓶颈确实在召回而非多抽。
- **误报四分类**：精确集 FP 中四类硬负例仅占 22.8%，其余 77.2% 是「讨论段泛化 HPO 词」（反复出现的疾病名/缩写），
  这是阶段三打分头的主战场。

### Gate 判定

| Gate | 结果 |
| --- | --- |
| 候选召回 ≥ 85%（带 HPO ID） | ✅ 补词表后 覆盖 0.974 / overlap 0.975（in-sample；CV 外推 0.675/0.755）；ontology-only 0.623 |
| 无整类表型系统性漏检 | ✅ 补词表后弱分支 0（`UNMAPPED` 143 个 id=-1 单列，不计入） |
| 误报可归入四类 | ⚠ 部分（四类 22.8%，主体是背景泛化词） |

> **补词表路线（`output_lex/`）用带 ID 精确候选即可过召回 gate，无需实体链接。** 代价是对 train 为 in-sample，
> 测试集能否达标取决于 A/B 措辞是否被覆盖（5 折 CV 0.675/0.755）。
> `output/` 的「名词短语粗召回」把无 ID span 也算进召回，作上限诊断（也需实体链接才能供阶段三用）。

## 关键设计决策

1. **方案 A：否定/家族史只打标签、不删除。** PatientPheX 金标把否定表型记为实体（`note="NO"`，共 171 个），
   家族史中的表型也照常标注（`note=null`）。因此候选默认保留：否定项加 `note="NO"`，四类上下文加 flag；
   另以 `filtered` 模式做「硬删」消融（对应阶段五 E7）。见报告 §2，硬删会让召回从 0.623 掉到 0.545。
2. **坐标零漂移。** 复用参考仓库 `patientpheX_adapter.py::rebuild_full_text` 的方案：按全局 `offset`
   把段落拼回全文（段间空格），实体 offset 直接作用其上，不做 NFKC / 去空白。校验脚本逐条断言
   `full_text[offset:offset+length] == text`，全部通过。
3. **重叠合并**：同一起始位置取最长；嵌套取外层；完全重叠合并为一条、保留全部 HPO id 并打
   `ambiguous` 标记；交叉重叠取更长。候选 span 始终来自真实模式命中，不跨模式拼接。
4. **闭集映射**：surface 精确命中取 term；命中多个保留全部 id，`primary` 取最深（最具体）term 交给阶段三。
5. **词表变体**：name + 全部 synonym（EXACT/NARROW/BROAD/RELATED），补连字符↔空格、希腊字母↔单词、
   末词单复数三类安全变体。共 93,279 个 surface，其中 35 个存在一词多义。
6. **名词短语粗召回（兜底召回层）**：无外部依赖、**不使用金标**。由 HPO 末词派生 1626 个 head 词，
   围绕 head 向左右扩展内容词得到短语，并补「含 head 的子句段」（≤90 字符）作为外层候选；
   短语若含词典命中则**继承其 HPO id**（`source=np`），否则无 id（`source=np_unmapped`）。
   这一层把覆盖召回从 0.623 抬到 0.875、overlap 到 0.933，弱分支清零。

## 补词表（阶段一资源建设）

`build_corpus_lexicon.py` 把 HPO 本体与 80 篇金标里**未覆盖的 surface** 合并成可溯源词表：
- 输出 `lexicon_aug.json`（匹配器用）、`lexicon.tsv`（**PDF 阶段一要求的交付**：surface/ids/source/歧义）、
  `lexicon_workorder.tsv`（新增补词工单）、`lexicon_unmapped.tsv`（id=-1 无映射金标）、`lexicon_aug_report.md`。
- 补词前后可映射召回（in-sample）：覆盖 **0.623→0.974**、strict **0.610→0.951**；补入 3,640 条新 surface
  （含临床缩写、通俗说法、单复数等变体），歧义表 35→211。
- ⚠ 补词来源是金标，对 train 为 in-sample；A/B 可外推估计看 `cv_recall.json`
  （文档级 5 折，覆盖 0.675 / overlap 0.755）——测试集是否达标仍取决于 A/B 的措辞覆盖。

### 外推召回可行性（补词规模曲线）

`lexicon_scale_recall.py`：固定 20 篇留出，变化补词文档数 k（5 次重复取均值）：

| k（补词文档数） | cover | overlap | strict |
| --- | --- | --- | --- |
| 0（ontology-only） | 0.673 | 0.742 | 0.655 |
| 5 | 0.685 | 0.755 | 0.667 |
| 20 | 0.696 | 0.770 | 0.677 |
| 40 | 0.703 | 0.782 | 0.684 |
| 60（用满补词池） | 0.708 | 0.790 | 0.688 |
| 名词短语粗召回（无金标，span） | **0.910** | **0.957** | — |

**结论**：补词表在未见文档上**饱和在 cover≈0.71 / overlap≈0.79**，靠补词表**到不了** 85%；
而**无金标的名词短语粗召回 span 召回 ≈0.91/0.96**（同样外推），能过 85% 的 span Gate，
但这些 span 无 HPO ID；要给它们补 ID 需要实体链接（PDF 未规定的延伸）。

### 轻量实体链接实测（为什么外推 strict 仍 <85%）

`linker.py`（token 重叠 + 倒排索引）把粗召回 span 链到 HPO，在留出集（1381 金标）上：

| 配置 | cover | overlap | strict(ID 正确) |
| --- | --- | --- | --- |
| 补词表 | 0.708 | 0.790 | 0.688 |
| +NP 继承 ID | 0.913 | 0.962 | 0.700 |
| +NP 轻量链接（thr=0.4） | 0.913 | 0.962 | **0.719** |

链接在「难例」（aug 词表未严格命中的可映射金标）上 **top-1 准确率仅 0.234**（top-3 0.331），
所以：
- **span 召回（cover/overlap）≈0.91/0.96，外推也过 85%** —— 靠粗召回即可；
- **ID 正确（strict）只 0.688→0.719，到不了 85%** —— 轻量词面链接不够，
  需要**语义/embedding 链接**（BioBERT/MedCPT 类，属阶段三模型能力）或更多标注。

一句话：**「span 过 85%」可达；「外推 ID 也对 85%」用现有轻量手段不可达。** 报告里应这样如实写。

## 目录与文件

```
stage2/
├── hpo.py                  # 解析 hp.obo → HP:0000118 子图（19087 term）+ 词典 + 歧义表
├── matcher.py              # Aho-Corasick 多模式匹配 + 词边界 + 重叠合并
├── rules.py                # 四类硬负例上下文打标签（否定/家族史/动物细胞/方法统计段）
├── evaluate.py             # 三指标 + E0 严格/span F1 + 五类错误 + 分支召回 + 歧义/未覆盖诊断
├── np_recall.py            # 名词短语粗召回兜底层（HPO 末词派生 head + 子句段），抬高召回上限
├── build_corpus_lexicon.py # 补词表：HPO + 金标 surface → lexicon.tsv/lexicon_aug.json/工单/报告
├── lexicon_scale_recall.py # 补词规模-外推召回曲线（判断补词表能否让未见文档过 85%）
├── linker.py               # 轻量实体链接（token 重叠）用于量化无 ID span 能否转可映射候选
├── exp_link_recall.py      # 链接对留出集 cover/overlap/strict 与链接准确率的度量
├── export_unmapped_worklist.py # 导出 id=-1 金标的人工补标工单（带上下文）
├── cv_lexicon_recall.py    # 词表补全的文档级 5 折交叉验证（可外推召回估计）
├── validate_outputs.py     # 产出校验：pmc 覆盖/唯一、offset 零漂移、字段合法
├── run_stage2.py           # 编排入口（一条命令跑全流程）
├── output/                 # ontology + 名词短语粗召回（诚实基线 + 召回上限 + CV）
└── output_lex/             # 补词表后的正式产物（lexicon.tsv + 候选 + 提交 + 报告）
```

## 复现命令

```bash
pip install -r code/requirements.txt   # pyahocorasick==2.3.1；Python>=3.8

# 1) 主运行：ontology + 名词短语粗召回（过 85% 召回 Gate），含 5 折 CV 诊断 -> output/
python code/run_stage2.py --data "<DATA_DIR>" \
    --out "<OUT_DIR>/output" --np-recall --cv-folds 5

# 2) 补词表（阶段一资源）-> output_lex/lexicon.tsv + lexicon_aug.json + 工单
python code/build_corpus_lexicon.py --data "<DATA_DIR>" \
    --out "<OUT_DIR>/output_lex" \
    --ontology-lexicon "<OUT_DIR>/output/lexicon.json" \
    --subgraph "<OUT_DIR>/output/hpo_subgraph.json"
#    用补后词表跑阶段二（缺 hpo_subgraph.json 时会自动重建）-> candidates + submit + 报告
python code/run_stage2.py --data "<DATA_DIR>" \
    --out "<OUT_DIR>/output_lex" \
    --lexicon-file "<OUT_DIR>/output_lex/lexicon_aug.json"

# 3) 校验产出
python code/validate_outputs.py --data "<DATA_DIR>" --out "<OUT_DIR>/output_lex"
```

依赖：见 `code/requirements.txt`（`pyahocorasick==2.3.1`）；Python ≥ 3.8；其余仅标准库。

## 产出文件

| 文件 | 说明 |
| --- | --- |
| `hpo_subgraph.json` | HP:0000118 子图：id/name/definition/synonym(is_a,alt_id,xref,depth,branch) |
| `lexicon.json` | ontology-only surface→HPO id 词典 + 歧义表 |
| `lexicon.tsv` / `lexicon_aug.json`（output_lex） | 补词后词表：surface/ids/source + 歧义表（阶段一交付 + 匹配器用） |
| `lexicon_workorder.tsv` / `lexicon_unmapped.tsv`（output_lex） | 补词工单 / id=-1 未映射清单 |
| `unmapped_gold_worklist.tsv`（output_lex） | 109 个 id=-1 金标实体的人工补标工单（带 pmc/offset/上下文） |
| `lexicon_aug_report.md`（output_lex） | 补词前后召回对比与统计 |
| `lexicon_scale_recall.{json,md}`（output） | 补词规模-外推召回曲线（判定 85% 可达性） |
| `candidates_train.jsonl` | 阶段三训练候选：start/end/text/ids/primary/ambiguous/**source**/flags/primary_reason/section_type/note |
| `candidates_A.jsonl` / `candidates_B.jsonl` | A/B 测试集候选（同一词典口径） |
| `submit_pred_A[_filtered].jsonl` / `submit_pred_B[_filtered].jsonl` | E0 基线提交格式（`filtered` 为硬删四类候选的对照版） |
| `baseline_report.md` | 完整报告：三指标、分布、五类错误、FP 清单、分支召回、CV、Gate |
| `metrics.json` | 全部指标（raw / filtered_negfam / filtered_all4 / cv_augmented） |
| `cv_recall.json` | 5 折 CV 明细 |

## 指标定义

- **candidate recall（覆盖）**：金标 span 被某候选**包络完全覆盖**的比例（严格「覆盖」口径）。
- **candidate recall（overlap）**：金标 span 与某候选**区间有交**的比例——阶段三有边界回归，
  这是模型实际可用的召回上限。
- **候选精确率（span）**：候选中边界与某金标完全相等的比例；**strict** 追加 ID 有交条件。
- **E0 strict F1**：把候选当预测，按「边界相等 + ID 有交」一对一匹配算 P/R/F1。
- **五类错误**：完全正确 / HPO 映射错误（边界对 ID 错）/ 边界错 / 多抽 FP；（漏抽在召回侧体现）。

## 交接说明（给阶段三）

- `candidates_*.jsonl` 的每个候选已带 `ids`（可能多义）、`primary`、`ambiguous`、`source`、
  `flags`、`primary_reason`。阶段三可直接把「候选-上下文」构造成 D0/D1/D2 输入做打分，无需重跑匹配。
- **按 `source` 分层使用**：`source=lexicon` 是精确可映射候选（闭集打分 / E0 用）；
  `source=np` 是继承 id 的外层短语（边界回归用）；`source=np_unmapped` 无 id（只用于召回统计，
  要利用需先做实体链接）。E0 提交只含 `lexicon` 候选。
- **交付给阶段三用 `output_lex/`（补词表后，带 ID 召回 0.974）**；其 train 指标 in-sample 偏乐观，
  外推估计看 `output_lex/cv_recall.json`（文档级 5 折：覆盖 0.675 / overlap 0.755）。
  `output/` 保留 ontology-only 诚实基线与 NP 召回上限诊断，供对照。
- `note` 字段为否定标记（`NO`）；本基线否定标签 precision≈0.40、recall≈0.885，仅作辅助，
  **不参与** entity-level strict F1（该指标只看 span+ID）。
