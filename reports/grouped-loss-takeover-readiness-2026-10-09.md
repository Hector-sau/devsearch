# Grouped-loss preparation after local takeover / 本地接管后的分组损失准备

The optional next experiment is implemented and locally checked against the actual frozen R6 training material. This is **CPU preparation for a bounded hardware smoke**, not a completed CUDA experiment, a complete validation-v2 label release or an improved model. BM25 remains the formal default and the +0.02 promotion gate is unchanged.

可选的下一轮实验已实现，并使用真实冻结 R6 训练资料完成本地检查。本报告证明的是**有限硬件短测前的 CPU 准备**，不代表 CUDA 实验完成、完整验证 v2 封存或模型改善。正式默认仍为 BM25，+0.02 晋升门槛不变。

## Exact inputs and experiment / 精确输入与实验

- Reconstructed the reviewed R6 manifest byte-identically: `f9a091365f8cba75cfc8ee5e2973049692941a3114c19a0ed55f73106d773bcb`. Verified original prepared manifest `793840c54cdb166921dab1bda9bed4c05d776e485b406c02a8622e9ecb47e854`, child files, train-query/qrel identities and the eight-replacement review provenance. Strict grouped loading accepts **515 queries, 2,575 pairs: one positive, three hard and one random negative per query**.
  逐字节重建并核验 R6、原准备清单、子文件、训练问题／标签与八处替换审查来源；严格分组加载接受 **515 题、2,575 对，每题一正例、三困难负例、一随机负例**。
- Two fresh arms: grouped weighted BCE versus pairwise logistic; seeds 42/43/44, three epochs, 387 updates/run, fixed epoch-three selection. Same data/model/order/forward exposures; historical flat BCE is not the causal control. Six immutable configurations were generated locally. See [specification 16](../docs/16-grouped-loss-experiment.md).
  两组均重新训练，只比较分组加权 BCE 与排序损失；固定三个种子、三轮及第三轮候选，数据／模型／顺序／曝光相同。已生成六份不可变配置；旧平铺 BCE 不充当因果对照。
- Local plan SHA256: `77545737f6d17cf1455a2be05bca7f1e2ab769cb26d9e7efa5e78f752a6feae2`. The local plan binds absolute Mac paths and must not be relocated; the existing target creates its own plan from verified paths and matching content. Model identity is `6860def41c570f1ccdbe5681a8258fea455e9cff1e98590a2746c035a7beb21c`, MiniLM revision `44acabbec0ef496f6dbc93adadea57f376b7c0ec`.
  本地计划绑定 Mac 绝对路径，不能直接搬迁；目标机须按已验证路径及相同内容另建计划。模型绑定上述固定 MiniLM 身份。

## Review and verification / 审核与验证

Independent review found five concrete input/operator defects, now fixed: merged-positive-group negatives were not excluded; pair child symlinks could escape the runtime; smoke lacked production checkpoint restoration; timeout could leave descendants alive after their leader exited; and CUDA profile/actual TF32 settings were not fully enforced. The independent reviewer accepted the fixes with no remaining P1/P2 in the reviewed scope.

独立审核发现并修复五项具体问题：未排除整个合并正例组、子文件链接越出运行目录、短测缺少生产断点恢复、超时可能遗留后代进程，以及 CUDA 配置／实际 TF32 设置约束不完整。独立复核接受修复，所审范围内无剩余 P1/P2。

| Check / 检查 | Actual result / 实际结果 |
|---|---|
| Local targeted training, CLI and bounded-runner tests | **110 passed, 0 failures, 0 errors, 0 skips**, 12.920 s; Python 3.11.16, existing macOS CPU Torch 2.13.0. / 本地 CPU 专项 110 项通过，零失败、错误或跳过。 |
| Installed wheel outside checkout | **20 grouped ML checks passed, zero skips**, plus baseline index/search/CLI checks. No dependency or model downloads. / 安装包脱离源码目录后通过 20 项分组 ML 检查及基础检索／命令检查，无依赖或模型下载。 |
| Local actual-input plan/check/status | Exit 0; six runs `not_started`; `data_verified=true`. Model/history byte verification, `cuda_verified` and `actual_host_ready` are explicitly false. / 真实输入规划检查通过；六次均未开始，模型／历史文件实体及 GPU 仍未核验。 |
| Review lifecycle tooling | 42 synthetic checks pass: immutable provenance, missing/disputed-label sealing refusal, blind C then final adjudication, human-concern handling and grouped uncertainty. This is tooling evidence, not semantic-label accuracy. / 审核工具 42 项合成检查通过，不代表实际标签准确率。 |

The package source snapshot is `a8b75251b78467a83a4cd5cf4d406c399cb937fb3238333114a0742dc88a4ff8` (111 files); tested wheel SHA256 is `be0b9f28d6cdc3a4b206e591eeea543c151be33a25a67c0438d977179b4b36d9` (302,353 bytes). Final runner and test hashes are recorded in the linked [machine receipt](evidence/grouped-takeover-cpu-20261009.json). The earlier [CPU report](grouped-loss-cpu-verification-2026-10-09.md) remains historical: its 866-test local full suite had 19 baseline-reproduced failures/errors and 67 skips, so it was not a clean full pass. Exact-lock Linux CI is separately required; no result is invented here.

包源码、安装包及运行器哈希均按实际检查记录。旧 CPU 报告保留为历史证据，其中本地全量 866 项存在 19 项可在原提交复现的失败／错误和 67 项跳过，不能称为全量通过。Linux 精确锁 CI 仍单独验证，不在此预填结果。

## Remaining execution gates / 剩余执行条件

1. Verify the supported CI result for the final code commit. Preserve all existing private runtime data and 60 server checkpoints.
   核对最终代码提交的支持环境 CI；保留私有运行资料及服务器 60 份既有断点。
2. On the existing host: verify exact locked dependencies, model/tokenizer bytes, all historical checkpoint hashes, disk space and idle GPU; create a new host-specific plan. The read-only SSH attempt was refused in this preparation session; current power state and files were not reverified remotely.
   在现有主机核验依赖、模型／分词器、旧断点、空间及 GPU 空闲状态，并创建目标机计划。本轮只读 SSH 被拒，未远端复核开关机状态及文件。
3. Run both bounded real-model recovery probes, then admit the six serial full runs only if they pass and the measured estimate fits the persistent **45-minute window plus at most two-minute termination allowance**. No sweep, automatic new rental or paid model API.
   两目标真实恢复短测通过且估算满足固定时限后，才串行执行六次全量训练。不扫参、不自动新租机或调用付费模型 API。
4. Finish independent A/B/C validation labels and any new-model shared-pool extension before full-v2/v2.1 quality claims. Original v1 remains immutable; frozen-v1 exploratory execution is technically separate from full-v2 completion. No test data was used for tuning; no new test evaluation, promotion, model export or large checkpoint transfer was performed.
   完成独立双审／裁决及新模型候选池扩展后，才能声明完整 v2／v2.1 的效果。原 v1 不变；冻结 v1 的探索运行与全量 v2 完成是不同条件。本轮未根据测试集调参、重新测试、晋升模型、导出模型或回传大断点。
