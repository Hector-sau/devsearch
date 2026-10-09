# Current handoff / 当前交接

**Updated 2026-10-09: training complete; private version 1.0.0 integration/publication authorized.** This file supersedes older dated status paragraphs. Technical contracts and historical receipts keep their original meaning.

**更新于 2026-10-09：训练完成，私有 1.0.0 整合与发布已获授权。**本文替代较早的状态段落，技术契约与历史回执保留原义。

## Identities / 标识

| Role / 角色 | Identity / 标识 |
| --- | --- |
| First formal GPU campaign / 首轮正式 GPU 实验 | `e3168ca6fef9b64ea0688ea059332c729125e30f` |
| Latest executed six-run code / 最新六次训练代码 | `cc0e7f0c6898af8fc7635c5ce9ffd4a4df8c4f89` |
| Code/documentation integration / 代码文档整合 | PR #2 → #3 → #4 → #5; exact final SHA in the [versioned release](https://github.com/Hector-sau/devsearch/releases/tag/v1.0.0) manifest / 准确最终源码见版本发布清单 |
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

Follow [TASKS](TASKS.md) for deferred large-artifact transfer when external storage is available and for the owner's independent project explanation. The versioned release manifest supplies final source, matching CI and small asset identities. For any later deployment, verify that source and installed artifacts match that identity. If extending live R5+fusion, implement the exact offline ordering and verify parity before connecting its reported metric to the demonstration.

外置存储就绪后按发布清单处理暂缓的大文件交接，并完成所有者独立项目讲解。版本发布清单提供最终源码、对应 CI 和小型附件标识；后续部署须核验源码与安装产物一致。如果扩展 R5 融合服务，先实现同一离线排序并验证一致性，再将报告指标与演示对应。

No additional GPU training is required for the current engineering scope. Keep the formal BM25/+0.02 contract, original test seals and old checkpoints. Do not start new paid services or download large model/checkpoint files while the owner is waiting for external storage. The 316-pair extension is optional for a complete new-loss quality report, not required for the already-proven stopping decision.

当前工程范围无需新增 GPU 训练。保留正式 BM25／+0.02 契约、原测试封存与旧断点；所有者等待外置存储期间，不新增付费服务或回传大型模型／断点。316 对扩展仅用于完整的新损失质量报告，已证明的停止决定无需补标。

## Reading order / 阅读顺序

[PROJECT_GUIDE](PROJECT_GUIDE.md) → [RESULTS](../reports/RESULTS.md) → [TASKS](TASKS.md) → [REPRODUCIBILITY](REPRODUCIBILITY.md). Code owners: [02 data](02-data-and-evaluation.md), [03 training](03-implementation.md), [04 serving](04-integration-and-decisions.md), [16 grouped loss](16-grouped-loss-experiment.md).

Prior handoff: [Claude's fixed snapshot](https://github.com/Hector-sau/devsearch/blob/afcba198433991d504a9c8ebc3d886d6b0585420/docs/HANDOFF.md). Private runtime paths and access details are maintained locally, not in this document.

旧交接保留在 Claude 的固定提交中；私有运行路径与接入信息仅在本地维护。

## Version-1 release / 第一版发布

The release preparation started at `b23b0f48a035fce7f68c923affd15b5cc94964c9`; package versions are `1.0.0`. The real-corpus BM25 UI/API and isolated package checks are verified. The owner authorized private integration/publication on 2026-10-09. Follow [RELEASE](RELEASE.md) for exact-source checks and post-publication verification; the versioned release carries the manifest and durable small receipts. Candidate `386aa0b` CI remains preparation evidence, distinct from final-main checks. Repository visibility stays private; project license selection remains the owner's decision.

发布准备从上述提交开始，包版本为 `1.0.0`；真实 BM25 界面／接口及隔离安装已核验。所有者于 2026-10-09 授权私有整合与发布。准确源码检查和发布后复验遵循发布流程，版本附件持久保存清单及小型回执。候选 `386aa0b` 的 CI 保留为准备证据，与最终主分支检查区分。仓库维持私有，项目许可证由所有者决定。
