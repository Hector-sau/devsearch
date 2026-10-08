# Diagrams / 架构与流程图

Rendered copies of every diagram in `docs/*.mmd`, with their standing. The `.mmd` files are the sources; the two marked *current* were
added on 2026-10-08 and reflect [the evidence index](../reports/EVIDENCE_INDEX.md). The older diagrams are unchanged and describe the
design or the plan at the time they were drawn. In the copies below `\n` line breaks are written as `<br/>` so that GitHub renders them;
the wording is unchanged.

本页汇总 `docs/*.mmd` 全部图表并标注时效：两张“当前”图为 2026-10-08 新增；旧图保持原样，表示当时的设计或计划。

## 1. Current status / 当前状态 — current (`status-current.mmd`)

```mermaid
flowchart LR
  subgraph DONE["Done, with evidence (CPU / no device) 已完成"]
    C["Code candidate 577b0ce<br/>CI + evidence green 代码候选"]
    L["Exact CUDA lock<br/>offline install, 847 tests 0 skipped 准确依赖锁"]
    M["MiniLM 44acabbe<br/>CPU check 10/10, crash 137 + resume 真实模型"]
    D["Real corpus + tokenizer<br/>230 docs, 4,259 passages 真实语料"]
    B["Review bank rev. 5<br/>9,852/9,852 final, agent only 审查库"]
    P["Package v2 06e7ae62<br/>seal 9a59905c, private release 数据包"]
  end
  subgraph WAIT["Pending 待完成"]
    O["Owner: approve commit,<br/>package access 所有者决定"]
    G["GPU host: check, prepare,<br/>diagnose 主机检查与诊断"]
    R["Campaign: BM25 vs frozen vs full,<br/>sealed test, latency 正式实验"]
    F["Report and release<br/>报告与发布"]
  end
  C --> O
  L --> G
  M --> G
  P --> G
  D --> P
  B --> P
  O --> G --> R --> F
```

## 2. Review protocol as executed / 实际执行的审查流程 — current (`review-flow.mmd`)

Counts are from the final audit and the bank manifest (agent judgments only).

```mermaid
flowchart TB
  A["Primary pass: 9,852 items<br/>9,823 resolved, 29 unsure 首轮"] --> S["Seeded stratified sample: 924<br/>(label x rank band, 197/197 questions) 分层抽样"]
  S --> X{"Unrounded disagreement > 10%?<br/>分歧率是否超过 10%"}
  X -- "4 positive strata: 18.6% / 32.3% / 59.4% / 51.5%" --> E["Full second review: +553<br/>(1,477 second reviews) 扩审"]
  X -- "negative strata <= 2.0%" --> K["Sample only 仅抽样"]
  E --> J["Adjudication: 284 items<br/>(255 disputes + 29 unsure primaries) 裁决"]
  K --> J
  A -. "29 unsure" .-> J
  J --> J2["13 unsure -> second adjudicator<br/>1 unsure -> third; none forced 再裁决"]
  J2 --> F["Audit exit 0: 9,852/9,852 final<br/>542 positives, 9,310 negatives 审计通过"]
  F --> P["Bank sealed in package v2<br/>agent 11,627 / human 0 封存"]
```

## 3. Accelerator run procedure / 加速器运行流程 — current design (`accelerator-run.mmd`)

The procedure `gpu.py` implements. Everything up to "Seal test inputs + complete judgment bank" has evidence; the host doctor and every
step after the locked plan are pending on the GPU host.

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

Drawn when the target was a Mac mini. Since then the target became a rented NVIDIA host (MPS stays optional); M0, M1 and the "labels, group split and initial test seal" step are done; the
on-device MiniLM probe, training and comparison (M2–M3), release (M4) and final evaluation (M5) are pending on the GPU host.

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
