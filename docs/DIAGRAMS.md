# Diagrams / 架构与流程图

> **2026-10-09:** the status diagram has been updated to completed training and pending product delivery. Other diagrams describe contracts; live R5 fusion is pending. [Current tasks](TASKS.md). / 状态图已更新为训练完成、产品交付待验；其他图为契约，R5 融合服务尚待接入。

Copies of every diagram in `docs/*.mmd`, with their standing. The `.mmd` files are the sources. The current status diagram was updated on 2026-10-09; the test-bank review diagram records the completed original review. Other diagrams describe contracts or historical plans. Copied diagrams use HTML line breaks for GitHub rendering.

本页汇总 `docs/*.mmd` 全部图表并标注时效：当前状态图更新于 2026-10-09；测试库审查图记录首轮已完成审查，其他图表示契约或历史计划。

## 1. Current status / 当前状态 — current (`status-current.mmd`)

```mermaid
flowchart TD
    A["cc0e7f0 / Verified implementation"] --> B["Six CUDA runs complete"]
    C["Sealed validation v4"] --> D["R5 offline candidate"]
    B --> E["Stop pairwise direction"]
    F["Evaluated BM25 release"] --> G["CPU API verified"]
    G --> H["Final integration and large asset handoff pending"]
    D --> I["Live rank fusion integration pending"]
    E --> H
    I --> H
    H --> J["Reviewed version-1 delivery"]
```

## 2. Formal test-bank review as executed / 正式测试库审查流程 — historical execution (`review-flow.mmd`)

Counts are from the original formal test-bank audit and manifest (agent judgments only). They do not describe the later validation-v4 review; see [DATA_CARD](DATA_CARD.md) for that version.

数量来自首轮正式测试库的审计与清单，均为 Agent 判定；不代表后续验证 v4 的复审，后者见数据说明。

```mermaid
flowchart TB
  A["Primary pass: 9,852 items<br/>9,823 resolved, 29 unsure 首轮"] --> S["Seeded stratified sample: 924<br/>(label x rank band, 197/197 questions) 分层抽样"]
  S --> X{"Unrounded disagreement above 10 percent?<br/>分歧率是否超过百分之十"}
  X -->|"yes: 4 positive strata (18.6, 32.3, 59.4, 51.5 percent)"| E["Full second review: +553<br/>(1,477 second reviews) 扩审"]
  X -->|"no: negative strata (at most 2.0 percent)"| K["Sample only 仅抽样"]
  E --> J["Adjudication: 284 items<br/>(255 disputes + 29 unsure primaries) 裁决"]
  K --> J
  A -.->|"29 unsure"| J
  J --> J2["13 unsure -> second adjudicator<br/>1 unsure -> third; none forced 再裁决"]
  J2 --> F["Audit exit 0: 9,852/9,852 final<br/>542 positives, 9,310 negatives 审计通过"]
  F --> P["Bank sealed in package v2<br/>agent 11,627 / human 0 封存"]
```

## 3. Accelerator run procedure / 加速器运行流程 — current design (`accelerator-run.mmd`)

The procedure `gpu.py` implements. The first formal campaign on `e3168ca` ran every step of it on an RTX 4090 (BM25 stays the default;
[evidence index §9](../reports/EVIDENCE_INDEX.md)).

```mermaid
flowchart TB
  C["Implemented package + backend locks<br/>完成代码及分后端依赖锁"] --> D["Offline diagnostic rehearsal<br/>离线诊断全流程演练"]
  P["Pinned corpus + train/validation labels<br/>固定语料与训练验证标注"] --> I["Index + immutable candidate lists<br/>索引与固定候选"]
  I --> J["Blindly review ALL test candidates + known positives<br/>预审全部测试候选及已知正例"]
  J --> S["Seal test inputs + complete judgment bank<br/>封存测试输入及完整判断库"]
  D --> H["Host doctor + calibration<br/>目标主机自检与校准"]
  S --> L["Locked execution plan<br/>锁定执行计划"]
  H --> L
  L --> T["Frozen/full training + validation<br/>冻结及全量训练与验证"]
  T --> V["Lock checkpoints + default on validation<br/>仅用验证锁定模型及默认模式"]
  V --> R["Rank fixed test candidates<br/>对固定测试候选排序"]
  R --> Q["Coverage check + second seal<br/>覆盖核验及第二次封存"]
  Q --> E["Shared-qrels scores + measured report<br/>统一标注计分及实测报告"]
  E --> B["Offline export + reload/API checks + backup<br/>离线导出及重载接口备份检查"]
  B --> F["Completed or explicit failed gate<br/>完成或明确门禁失败"]
```

## 4. System architecture / 系统架构 — design (`architecture.mmd`)

Design revision 3. "NQ optional transfer data" was never used: the current data are the `pytorch/tutorials` corpus with synthetic,
agent-reviewed questions. The online path (UI, API, SearchPipeline, BM25, reranker or explicit baseline) is implemented and was exercised on
CPU with a diagnostic model only.

```mermaid
flowchart TB
  subgraph OFF["Offline preparation 离线准备"]
    S["Pinned documentation 固定版本文档"] --> C["Parse and chunk 解析与切块"]
    C --> I["Corpus and BM25 index 语料与索引"]
    C --> L["Evidence labels and splits 证据标注与划分"]
    N["NQ optional transfer data 可选通用训练数据"] --> L
    L -- "training only 仅训练" --> T["Native PyTorch training 原生训练流程"]
    T --> M["Candidate model 候选模型"]
  end
  subgraph GATE["Validation and release 评测与发布"]
    I --> V["Validation and compatibility checks 验证与兼容检查"]
    M --> V
    L -- "validation only 仅验证" --> V
    V --> B["Candidate release bundle 候选发布包"]
    B --> J["Blind pooled judging then freeze qrels 盲审候选池后锁定标签"]
    L -. "sealed test only 封存测试" .-> J
    J --> E["Same final qrels for all models 全模型统一最终评测"]
    E --> R["Versioned local release 本地版本化发布"]
  end
  subgraph ON["Online search 在线检索"]
    U["Question and filters 问题与筛选"] --> A["UI and API 界面与接口"]
    A --> P["SearchPipeline 检索流程"]
    R --> P
    P --> K["BM25 candidates 候选召回"]
    K --> X["Reranker or explicit baseline 重排序或明确基线模式"]
    X --> O["Top passages and sources 相关段落与来源"]
  end
```

## 5. Roadmap / 路线图 — history (`roadmap.mmd`)

Drawn when the target was a Mac mini. Since then the target became a rented NVIDIA host (MPS stays optional). M0–M2 and the first
comparison are done: the first formal campaign kept BM25 as the default. Domain adaptation (M3), a model release (M4) and further
evaluation (M5) have not been pursued; a second round is an exploratory experiment whose complete evidence is pending delivery.

```mermaid
flowchart LR
    M0["M0 Package and storage contracts<br/>包结构与存储契约"] --> M1["M1 Usable BM25 search<br/>可用检索基线"]
    M0 --> H["Target Mac and MiniLM probe<br/>目标机器与模型探测"]
    M1 --> D["Labels, group split and initial test seal<br/>标注分组与首次测试封存"]
    D --> M2["M2 Train and compare<br/>训练闭环与对照"]
    H --> M2
    M2 --> M3["M3 Domain adaptation<br/>领域适配与按需扩量"]
    M3 --> M4["M4 Release candidate<br/>集成候选版本"]
    M4 --> M5["M5 Final evaluation<br/>最终评测与交付"]
    M2 -.-> L["Independent learning evidence<br/>用户独立学习证据"]
```
