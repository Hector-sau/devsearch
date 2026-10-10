# Project guide / 项目导读

DevSearch is an English technical-document retrieval application and a reproducible PyTorch experiment project. The evaluated serving default is BM25; R5 plus reciprocal rank fusion is an experimental candidate. [Current results](../reports/RESULTS.md) and [release tasks](TASKS.md) define the present state.

DevSearch 是英文技术文档检索应用与可复现的 PyTorch 实验项目。已评测服务默认使用 BM25，R5 加倒数排名融合为实验候选。当前状态以[结果报告](../reports/RESULTS.md)和[发布任务](TASKS.md)为准。

## Reading routes / 阅读路线

- **Project review:** README → RESULTS → MODEL_CARD / DATA_CARD → EVIDENCE_INDEX.
  **项目审阅：**项目首页 → 实验结果 → 模型与数据说明 → 证据索引。
- **Development:** REPRODUCIBILITY → CODE_WALKTHROUGH → owning numbered specification → TASKS.
  **继续开发：**复现指南 → 代码导读 → 对应编号规格 → 当前任务。
- **Demonstration/interview:** DEMO_SCRIPT → INTERVIEW_GUIDE; practice explaining one training modification and one failed experiment.
  **演示与面试：**演示脚本 → 面试指南；独立解释一次训练修改和一次失败实验。

## File map / 目录职责

```text
devsearch/
├── README.md                 # project entry and verified result summary
├── pyproject.toml            # Python package and optional dependencies
├── src/devsearch/
│   ├── ingestion/            # pinned source parsing and chunking
│   ├── data/                 # split ownership, negative mining and review
│   ├── retrieval/            # BM25 candidates and deterministic ordering
│   ├── modeling/             # tokenizer, MiniLM and checkpoint/model I/O
│   ├── training/             # native PyTorch loops and recovery
│   ├── evaluation/           # metrics, selection, bootstrap and latency
│   ├── serving/              # shared pipeline, FastAPI and Streamlit
│   └── execution/            # bounded campaigns and artifact gates
├── configs/                  # reviewed experiment and serving configurations
├── locks/                    # hashed, backend-specific dependencies
├── scripts/                  # operator entry points and evidence checks
├── tests/                    # unit/integration checks and small fixtures
├── docs/                     # operation, model/data cards and contracts
└── reports/
    ├── RESULTS.md            # current outcome and limits
    ├── EVIDENCE_INDEX.md     # provenance and historical records
    └── evidence/             # compact, sanitized evidence
```

Keep implementation paths stable: scripts, package imports and frozen evidence already depend on them. Organize the reader-facing entry points instead of moving source files into a cosmetic new layout.

保留实现路径稳定：脚本、包导入和冻结证据已经依赖这些路径。通过统一阅读入口完成整理，不为外观移动源码目录。

| File / 文件 | Responsibility / 职责 |
| --- | --- |
| [HANDOFF](HANDOFF.md) | Current technical state and next action / 当前技术状态与下一步 |
| [TASKS](TASKS.md) | Concrete completion evidence and remaining release gates / 已完成证据与剩余发布验收 |
| [REPRODUCIBILITY](REPRODUCIBILITY.md) | Small runnable example, CI checks and full artifact requirements / 小型可运行示例、CI 与完整产物要求 |
| [MODEL_CARD](MODEL_CARD.md) | Candidate identities, uses and limits / 模型标识、用途和限制 |
| [DATA_CARD](DATA_CARD.md) | Dataset origin, splits and label provenance / 数据来源、划分和标签来源 |
| [RESULTS](../reports/RESULTS.md) | One authoritative result summary / 统一结果摘要 |
| [DEMO_SCRIPT](DEMO_SCRIPT.md) | Verified BM25 release and separate real R5 candidate API/UI demonstrations / 已核验 BM25 发布包与独立真实 R5 候选接口／界面演示 |

## Runtime storage / 运行数据存储

A private `DEVSEARCH_RUNTIME_ROOT` holds corpus snapshots, indexes, review state, model assets, environments and checkpoints. The Git repository contains manifests/configuration, small fixtures and sanitized result summaries. Selected R5 inference assets and current CPU evidence are restored and hash-verified on owner-approved external storage. Native training checkpoints remain on the server; all-checkpoint backup is not complete. The small committed evidence does not reproduce private judgments or restore model weights on its own.

私有 `DEVSEARCH_RUNTIME_ROOT` 保存语料快照、索引、审查状态、模型资产、环境和断点。Git 保存清单／配置、小样例及脱敏结果摘要。选定 R5 推理资产及当前 CPU 证据已在获认可的外置盘恢复并核验；原训练断点保留服务器，全部断点备份未完成。已提交的小型证据本身不能复现私有判定或恢复权重。

The separate R5 extension has working fixed-fusion CPU API/UI and a completed 59-question independent evaluation. Its two quality gates failed, so it remains an experimental candidate requiring `--allow-candidate`; published v1.0.0 keeps BM25 as default. [Independent report](../reports/r5-independent-holdout-2026-10-10.md) · [inference reproduction](R5_INTEGRATION.md).

独立 R5 扩展已完成固定融合 CPU 接口／界面与 59 题独立评测。两个质量门槛未通过，继续作为须显式允许的实验候选；已发布 v1.0.0 保持 BM25 默认。范围见独立报告与推理复现说明。

## Contracts and history / 契约与历史

Numbered documents remain authoritative for their technical topic: [02 data](02-data-and-evaluation.md), [03 training](03-implementation.md), [04 serving](04-integration-and-decisions.md), [10 execution](10-accelerator-execution-contract.md), and [16 grouped comparison](16-grouped-loss-experiment.md). Their dated readiness paragraphs describe the date written; [HANDOFF](HANDOFF.md) supersedes those status paragraphs only.

编号文档继续管理对应技术契约：数据、训练、服务、执行及分组对照。内部带日期的就绪段落描述当时状态；[当前交接](HANDOFF.md)仅替代这些状态段落，不改写契约。

Claude's organization work is preserved in Git history at [`afcba198`](https://github.com/Hector-sau/devsearch/tree/afcba198433991d504a9c8ebc3d886d6b0585420). The [first-campaign draft](../reports/experiment-report-draft.md), original receipts and older code permalinks remain historical evidence. This update incorporates code base `cc0e7f0`, later experiments and the completed validation-label release; it does not rerun or relabel historical experiments.

Claude 的整理稿保留在上述 Git 提交中。首轮报告草稿、原回执及旧代码永久链接继续作为历史证据。本次整理纳入 `cc0e7f0` 代码基线、后续实验与已完成的验证标签发布，不重新运行或改标历史实验。
