# Current handoff / 当前交接

**Updated 2026-10-09: training complete; version 1.0.0 candidate preparation in progress.** This file supersedes older dated status paragraphs. Technical contracts and historical receipts keep their original meaning.

**更新于 2026-10-09：训练完成，正在准备 1.0.0 发布候选。**本文替代较早的状态段落，技术契约与历史回执保留原义。

## Identities / 标识

| Role / 角色 | Identity / 标识 |
| --- | --- |
| First formal GPU campaign / 首轮正式 GPU 实验 | `e3168ca6fef9b64ea0688ea059332c729125e30f` |
| Latest executed six-run code / 最新六次训练代码 | `cc0e7f0c6898af8fc7635c5ce9ffd4a4df8c4f89` |
| Documentation branch / 文档分支 | `codex/claude-docs`, [PR #4](https://github.com/Hector-sau/devsearch/pull/4), incorporates the above code base / 已纳入上述代码基线 |
| Evaluated release / 已评测发布包 | `release-ce3160fc579a52ae2ec30d15412853eb`, BM25 default / BM25 默认 |
| Validation labels / 验证标签 | `validation-v2-sampled202-human6-v4`, sealed, 4,571 pairs / 已封存 4,571 对 |
| Better validation candidate / 较好验证候选 | R5 + fixed RRF; three-seed results in [RESULTS](../reports/RESULTS.md) / R5 加固定融合 |

**Source identity matters:** do not point a historical CI receipt at this updated documentation commit. CUDA evidence belongs to the executed commits. Deploying a new full checkout requires matching receipts and runtime checks even if only docs changed.

**源码标识必须匹配：**历史 CI 回执不能改指向新文档提交；CUDA 证据属于实际执行提交。部署新的完整检出目录，即使只改文档，也需要对应回执和运行检查。

## Done / 已完成

- Original formal campaign: train, validation selection, sealed test evaluation, export, reload and backup. BM25 retained as default.
  首轮正式流程已完成训练、验证选择、封存测试评测、导出、重载和备份，默认保留 BM25。
- R2–R7 development and expanded v4 reevaluation of all 25 historical systems; six owner-confirmed human-assisted resolutions recorded explicitly.
  R2～R7 开发及历史 25 系统的扩展 v4 重评完成，六条所有者确认明确记为人工辅助。
- Six matched-loss CUDA runs complete; stop pairwise direction under the declared rule. Small results downloaded and hashed; checkpoint tensors retained on the server.
  六次匹配损失 CUDA 运行完成，按规则停止排序损失方向；小型结果已下载核验，断点张量保留服务器。
- Exact-lock CI for `cc0e7f0`: 906 tests, zero failures/errors/skips; ordinary CPU CI and separate real-model evidence jobs passed.
  `cc0e7f0` 准确依赖 CI 为 906 项、零失败／错误／跳过，普通 CPU CI 与真实模型证据作业通过。

## Next actions / 下一步

Follow [TASKS](TASKS.md): review/merge the dependency chain when authorized, verify the final demonstration scope, freeze delivery metadata and archive the selected artifacts when external storage is available. If demonstrating R5+fusion, implement that exact offline ordering in serving and verify parity before claiming the demonstrated model achieved its reported metric.

按发布清单完成：获授权后审阅合并依赖链，验证最终演示范围，冻结交付信息，并在外置存储就绪后归档选定产物。如果展示 R5 加融合，须将同一离线排序接入服务并验证一致性，再把演示模型与报告指标对应。

No additional GPU training is required for the current engineering scope. Keep the formal BM25/+0.02 contract, original test seals and old checkpoints. Do not start new paid services or download large model/checkpoint files while the owner is waiting for external storage. The 316-pair extension is optional for a complete new-loss quality report, not required for the already-proven stopping decision.

当前工程范围无需新增 GPU 训练。保留正式 BM25／+0.02 契约、原测试封存与旧断点；所有者等待外置存储期间，不新增付费服务或回传大型模型／断点。316 对扩展仅用于完整的新损失质量报告，已证明的停止决定无需补标。

## Reading order / 阅读顺序

[PROJECT_GUIDE](PROJECT_GUIDE.md) → [RESULTS](../reports/RESULTS.md) → [TASKS](TASKS.md) → [REPRODUCIBILITY](REPRODUCIBILITY.md). Code owners: [02 data](02-data-and-evaluation.md), [03 training](03-implementation.md), [04 serving](04-integration-and-decisions.md), [16 grouped loss](16-grouped-loss-experiment.md).

Prior handoff: [Claude's fixed snapshot](https://github.com/Hector-sau/devsearch/blob/afcba198433991d504a9c8ebc3d886d6b0585420/docs/HANDOFF.md). Private runtime paths and access details are maintained locally, not in this document.

旧交接保留在 Claude 的固定提交中；私有运行路径与接入信息仅在本地维护。

## Prepared release candidate / 已准备发布候选

`codex/release-v1` starts at `b23b0f48a035fce7f68c923affd15b5cc94964c9`; versions are `1.0.0`. The real-corpus BM25 UI, API and isolated package checks are verified. Follow [RELEASE](RELEASE.md) for final integration, matching-main checks and publication after the owner's decision. Candidate assets stay in the separate runtime/draft release; regenerate them against final main before tagging. Repository visibility remains private; project license selection remains the owner's decision.

发布准备分支从上述提交开始，版本为 `1.0.0`；真实 BM25 界面、接口和隔离安装已核验。按发布流程在所有者决策后完成整合、对应主分支检查及发布。候选附件在独立运行目录／草稿发布保存，打标签前须对最终主分支重新生成。仓库维持私有，项目许可证由所有者决定。
