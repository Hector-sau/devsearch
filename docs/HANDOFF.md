# Current handoff / 当前交接

**Current update, 2026-10-13: final 966 configuration selected; `1.2.0` publication is being prepared.** Three fresh same-host seeds 42/43/44 each finish two epochs/966 updates and pass the frozen47-question development gates with fixed fusion0.5/k60. A separate fresh seed 42 four-epoch/1, 932-update run does not meet its additional-duration improvement rule. The owner retains 966; no further GPU training is needed. The existing verified966 seed 42 CPU demo remains preferred, BM25 remains the public model-free entry, comparison and failure fallback, and explicit `--allow-candidate` controls remain required.

**当前更新，2026-10-13：已选定最终966配置，正在准备`1.2.0`发布。**同机42／43／44三个全新种子均完成两轮／966次更新，固定融合0.5／60并通过冻结47题开发门槛；另行从头训练的seed42四轮／1, 932次更新未通过延长训练规则。所有者保留966，无需继续GPU训练；已有已验收966 seed 42 CPU演示保持首选，BM25仍为公开无模型入口、对照及降级，显式`--allow-candidate`控制继续保留。

[Final results and scope](../reports/final966-replication-2026-10-13.md) · [Aggregate evidence](../reports/evidence/final966-20261013/summary.json). The updated result is synthetic, agent-labeled development replication, distinct from original R5 and the historical adaptive202 benchmark. Private labels, new frozen execution wrappers and complete checkpoints remain outside the public repository. Exact-source release checks and private archival completion are separate requirements; do not infer complete checkpoint backup from publication.

最终报告和汇总证据记录范围；新结果为合成题、Agent标签的开发复现，与原R5及历史反复使用202题基准区分。私有标签、新冻结执行包装器和完整断点保存在公开仓库之外；准确源码发布检查与私有归档完成属于不同要求，不能因发布而推断断点全部备份。

**Historical local acceptance snapshot, 2026-10-12:** 966-update seed 42 + fixed fusion is the owner-authorized local demo default. The second actual CPU attempt passes 80/80 checks, with warm real-TCP p95 635.736 ms across 150 sequential requests; the first attempt's valid 150-request p95 1577.089 ms sample and failed full-run status remain preserved. Independent actual-receipt review passes 36 checks, 32 owner-control files are externally hash-verified, and external-manager/API/browser activation passes for default, compare and explicit BM25 modes. [DEMO966_CPU](DEMO966_CPU.md) gives exact identities, reusable controls, explicit relocated-harness inputs and both measured scopes; [activation evidence](../reports/evidence/demo966-cpu-20261012/demo-activation-summary.json) binds the live checks. The provisional local service is `1.1.1.dev0`, installed separately. Published `v1.1.0`, its BM25 default and original R5 receipts remain unchanged. The new 966 comparison is single-seed development evidence, distinct from original R5's three-seed 202 results; no extra GPU training or holdout tuning is part of this integration.

**历史本地验收快照，2026-10-12：**966 次更新的 seed 42 加固定融合已成为所有者授权的本地演示默认。第二次真实 CPU 运行通过 80 项检查，150 次串行真实 TCP 请求预热后 p95 为 635.736 毫秒；第一次有效 150 请求的 p95=1577.089 毫秒样本和完整运行失败状态继续保留。独立实际回执审核通过 36 项检查，32 个所有者工具文件已外置核验，外置管理器／接口／浏览器的默认、对照及显式 BM25 模式均通过激活检查。上述接入文档说明准确身份、可复用工具、迁移后显式输入及两次实测范围；激活证据绑定真实检查。本地服务为单独安装的暂定 `1.1.1.dev0`；已发布 `v1.1.0`、其 BM25 默认及原 R5 回执保持不变。新 966 对照为单一种子的开发证据，与原 R5 三种子 202 题结果区分；本次接入不增加 GPU 训练，也不使用留出集调参。

**Updated 2026-10-10: public version 1.1.0.** The owner authorized Apache-2.0 publication. R5+RRF has repeated improvement on the same 202-question v4 validation benchmark; the separate 59-question holdout fails the unchanged quality gates, so BM25 remains the default. This file supersedes older status paragraphs. Historical training and CI receipts keep their executed source identity.

**更新于 2026-10-10：公开版 1.1.0。**所有者授权 Apache-2.0 发布。R5+RRF 在同一 202 题 v4 验证基准上复现提升；独立 59 题留出集未通过原门槛，默认仍为 BM25。本文替代旧状态段落，历史训练与 CI 回执保留实际执行源码归属。

## Identities / 标识

| Role / 角色 | Identity / 标识 |
| --- | --- |
| First formal GPU campaign / 首轮正式 GPU 实验 | `e3168ca6fef9b64ea0688ea059332c729125e30f` |
| Latest executed six-run code / 最新六次训练代码 | `cc0e7f0c6898af8fc7635c5ce9ffd4a4df8c4f89` |
| Code/documentation integration / 代码文档整合 | PR #2 → #3 → #4 → #5; exact final SHA in the [versioned release](https://github.com/Hector-sau/devsearch/releases/tag/v1.0.0) manifest / 准确最终源码见版本发布清单 |
| Evaluated release / 已评测发布包 | `release-ce3160fc579a52ae2ec30d15412853eb`, BM25 default / BM25 默认 |
| Validation labels / 验证标签 | `validation-v2-sampled202-human6-v4`, sealed, 4, 571 pairs / 已封存 4, 571 对 |
| Better validation candidate / 较好验证候选 | R5 + fixed RRF; three-seed results in [RESULTS](../reports/RESULTS.md) / R5 加固定融合 |
| Frozen R5 implementation / 冻结 R5 实现 | `b76e349adcbb46a73c430889632810cbaebbd66a`; [CI 37958581548](https://github.com/Hector-sau/devsearch/actions/runs/37958581548), four jobs green / 四项作业通过 |
| R5 independent evaluation / R5 独立评测 | Seed 42/epoch 3, 59 synthetic questions in 59 unused evidence groups; [report](../reports/r5-independent-holdout-2026-10-10.md) / 59 题、59 个未用证据组 |

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
- Frozen R5 weights exported and restored on owner-approved external storage; real CPU reload, API/UI and offline/online ordering verified. All 59 server/Mac ranking orders agree. Native training checkpoints remain on the server.
  冻结 R5 推理权重已导出并恢复到获认可的外置存储，真实 CPU 重载、接口／界面及离在线排序通过；服务器与 Mac 的 59 题排序一致，原训练断点保留服务器。
- Independent labels sealed before scoring: 789 A judgments, 165 B judgments and 12 independent C resolutions. R5 fused MRR change is −0.008024 and nDCG change is −0.065773; both quality gates fail. Warm Mac CPU p95 is 622.065 ms under the fixed sequential workload.
  独立标签在计分前封存：789 条 A 审、165 条 B 审、12 条独立 C 裁决。R5 融合 MRR 差值为 −0.008024，nDCG 为 −0.065773，两个质量门槛均未通过；固定串行负载下 Mac CPU 预热后 p95 为 622.065 毫秒。

## Historical v1.1.0 next actions / 历史v1.1.0后续

Follow [TASKS](TASKS.md) and the [v1.1.0 release manifest](https://github.com/Hector-sau/devsearch/releases/tag/v1.1.0) for final source, CI, package hashes and publication verification. [Portable CPU replay](VALIDATION_REPLAY.md) recomputes validation metrics from hash-verified owner-held inputs. Selected R5 inference assets and sealed datasets are preserved externally; this is not a complete backup of every native training checkpoint. Serve the verified candidate only with `--allow-candidate`.

后续以任务清单和 v1.1.0 发布清单为准，准确关联源码、CI 与附件哈希。便携 CPU 重放可用哈希核验后的所有者输入重算验证指标。选定推理模型和封存数据已外置保存，但不等于全部原生训练断点完整备份；候选服务须显式使用 `--allow-candidate`。

No additional GPU training is required for this completed integration/evaluation. Keep the formal BM25/+0.02 contract, original test seals and old checkpoints. Do not tune or retrain on the opened holdout, promote the failed candidate, or start new paid services. The 316-pair extension is optional for a complete new-loss quality report, not required for the already-proven stopping decision.

本次接入／评测已完成，无需新增 GPU 训练。保留正式 BM25／+0.02 契约、原测试封存与旧断点；不使用已打开的留出集调参或重训，不晋升未达门槛的候选，不启动新增付费服务。316 对扩展仅用于完整的新损失质量报告，已证明的停止决定无需补标。

## Reading order / 阅读顺序

[PROJECT_GUIDE](PROJECT_GUIDE.md) → [RESULTS](../reports/RESULTS.md) → [TASKS](TASKS.md) → [REPRODUCIBILITY](REPRODUCIBILITY.md). Code owners: [02 data](02-data-and-evaluation.md), [03 training](03-implementation.md), [04 serving](04-integration-and-decisions.md), [16 grouped loss](16-grouped-loss-experiment.md).

Prior handoff: [Claude's fixed snapshot](https://github.com/Hector-sau/devsearch/blob/afcba198433991d504a9c8ebc3d886d6b0585420/docs/HANDOFF.md). Private runtime paths and access details are maintained locally, not in this document.

旧交接保留在 Claude 的固定提交中；私有运行路径与接入信息仅在本地维护。

## Historical version-1 release / 第一版历史发布

The release preparation started at `b23b0f48a035fce7f68c923affd15b5cc94964c9`; package versions are `1.0.0`. The real-corpus BM25 UI/API and isolated package checks are verified. The owner authorized private integration/publication on 2026-10-09. Follow [RELEASE](RELEASE.md) for exact-source checks and post-publication verification; the versioned release carries the manifest and durable small receipts. Candidate `386aa0b` CI remains preparation evidence, distinct from final-main checks. That release was initially private. The owner authorized public visibility and Apache-2.0 for the subsequent v1.1.0 release on 2026-10-10.

发布准备从上述提交开始，包版本为 `1.0.0`；真实 BM25 界面／接口及隔离安装已核验。所有者于 2026-10-09 授权私有整合与发布。准确源码检查和发布后复验遵循发布流程，版本附件持久保存清单及小型回执。候选 `386aa0b` 的 CI 保留为准备证据，与最终主分支检查区分。该版本最初以私有形式发布；所有者于 2026-10-10 授权后续 v1.1.0 公开并采用 Apache-2.0。
