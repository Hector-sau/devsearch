# Version-1 release procedure / 第一版发布流程

## Scope / 范围

The `1.0.0` candidate contains a usable BM25 default, native PyTorch training/evaluation tools and verified, scoped experiment reports. No additional GPU training is required. R5 live rank fusion remains optional; published R5 metrics remain adaptive-validation family results.

`1.0.0` 候选包含可用的 BM25 默认检索、原生 PyTorch 训练／评测工具及注明范围的实测报告，无需新增 GPU 训练。R5 融合服务为可选扩展；其指标仍是反复使用验证集上的候选组结果。

Preparation starts at `b23b0f48a035fce7f68c923affd15b5cc94964c9` on `codex/release-v1`. Required work is package identity, clean-reader verification, real-corpus demonstration, scoped assets and exact-source CPU evidence. [TASKS](TASKS.md) tracks completion.

准备从上述提交的 `codex/release-v1` 分支开始，必需工作是包标识、独立复现、真实语料演示、明确资产范围和对应源码的 CPU 证据；状态见任务清单。

## Integration and final acceptance / 整合与最终验收

1. Review the tested release candidate and obtain the owner's final merge/publication decision. Keep repository visibility private unless explicitly changed by the owner; license selection is separate.
   审阅已测试候选，取得所有者最终合并／发布决定。未经明确改变，仓库维持私有，许可证由所有者另定。
2. Integrate PR #2 → #3 → #4 → release-preparation PR. Prefer merge commits to preserve ancestry. After each merge, retarget the next PR to `main`, inspect its new diff and require the applicable checks. Do not delete source branches until the entire stack is safely integrated.
   顺序整合依赖链，优先保留提交祖先；每次重新设置下一 PR 目标、审阅新差异并检查测试。整条依赖链安全整合前不删除来源分支。
3. Freeze final `main` SHA. Dispatch `scaffold-check.yml` and `non-gpu-evidence.yml` at the frozen branch/ref and verify every run's `head_sha` matches. Ordinary jobs must all succeed; ML and exact-lock implementation checks require zero skips. Historical CUDA evidence stays on its executed source commits.
   冻结最终主分支 SHA，主动触发两工作流并核对运行源码。普通作业全部通过，ML 和准确依赖检查零跳过；历史 CUDA 证据保留实际执行提交。
4. Build the wheel from that exact source, verify installed version and isolated use, and regenerate a release manifest containing the final source SHA, workflow runs, locks, artifact identities, sizes and hashes.
   从准确源码构建 wheel，核验安装版本与隔离运行，重新生成包含最终源码、CI、依赖锁、产物标识、大小与哈希的发布清单。
5. Preserve compact receipts beyond Actions retention. Publish `v1.0.0` only after those checks and asset validation, then verify the tag and the actual downloaded small release assets. A preparation draft tied to the release branch must be updated to the final `main` SHA before publication.
   小型回执持久保存，通过后发布版本并验证标签与实际下载的小附件。准备分支上的草稿必须在发布前改绑最终主分支 SHA。

Do not commit a manifest that contains its own source commit identity; attach it to the release instead. Any later committed change creates a different release identity and requires new applicable verification. A candidate CI result is not a merged-main CI result.

包含自身源码提交号的清单作为发布附件保存，不提交回源码。之后任何提交会改变发布标识，须重新核验；候选 CI 不等于合并后主分支 CI。

## Assets / 资产

| Asset / 资产 | Storage and scope / 保存与范围 |
| --- | --- |
| Source archive and wheel / 源码与安装包 | Small release assets; no corpus, model weights or labels / 小型发布附件，不含语料、权重或标签 |
| Manifest and SHA256SUMS / 清单与哈希 | Bind exact source/version, file sizes, hashes and workflow run IDs / 关联准确源码、版本、大小、哈希及工作流 |
| Reports, demo and compact receipts / 报告、演示与小型回执 | Durable release assets; preserve original experiment commit identities / 持久发布附件，保留实验原提交标识 |
| Private runtime inputs / 私有运行输入 | Owner-held dataset/review/index artifacts; required for full historical reproduction / 所有者持有，完整历史复现所需 |
| GPU checkpoints / GPU 断点 | Remain on server; transfer deferred to external storage / 保留服务器，回传等待外置存储 |

CPU toy reproduction, real-corpus BM25 demonstration, historical GPU results and full private-data reconstruction have different requirements; the documentation must state which is verified. Read [REPRODUCIBILITY](REPRODUCIBILITY.md), [RESULTS](../reports/RESULTS.md) and [MODEL_CARD](MODEL_CARD.md).

CPU 小样例、真实语料 BM25 演示、历史 GPU 实测及完整私有数据重建要求不同，文档须说明哪项经过验证；详见复现、结果及模型说明。
