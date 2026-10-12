# Release checklist / 发布清单

## Final966 release preparation — 2026-10-13 / 最终966发布准备

The owner authorizes updating GitHub and preserving native checkpoints in no-card mode. No additional GPU training is needed. The selected configuration is 966 updates/two epochs/fixed fusion, supported by three fresh seeds; keep the existing accepted seed 42 demo endpoint and BM25 control/fallback. [Final result](../reports/final966-replication-2026-10-13.md).

所有者授权更新GitHub并以无卡模式保存原生断点；无需追加GPU训练。最终配置为966次更新／两轮／固定融合，已有三个全新种子支持；保持已有已验收seed42演示模型及BM25对照／降级。

- [x] Complete and independently audit seeds 42/43/44 at fixed966 updates and CPU scoring on the same frozen81 graded development questions.
  完成并独立审核三个固定966种子，在同一冻结81题分级开发标签上完成CPU计分。
- [x] Preserve original 202 labels and independently audit partial binary regression scoring with5, 530 natural `UNKNOWN` records retained.
  保留原202标签，独立审核部分二元回顾计分，保留5, 530项自然池未知标签。
- [x] Complete the single freshseed42/1, 932-update comparison; report the failed declared duration rule and owner decision to retain966.
  完成一个全新seed42／1, 932次更新对照，记录时长规则未通过及所有者保留966的决定。
- [x] Prepare aggregate-only public results, evidence commitments and current model/release documentation without private queries, labels, paths or credentials.
  准备仅含汇总的公开结果、证据哈希及当前模型／发布文档，不包含私有问题、标签、路径或凭据。
- [ ] Verify the final integrated source: applicable zero-skip CI, versioned isolated wheel, model-free public CPU example and candidate service checks with historical identities preserved.
  核验最终整合源码的所需零跳过CI、带版本隔离安装包、公开无模型CPU样例及候选服务检查；保持历史身份。
- [ ] Publish immutable `v1.2.0`, verify the exact tag/source and downloaded small attachments, and preserve the matching release copy externally.
  发布不可变v1.2.0，核验准确标签／源码及实际下载的小附件，在外置盘保存同版副本。
- [ ] Complete private no-card checkpoint backup and independent full-file hashes before permitting instance release; record storage completion separately.
  完成无卡模式私有断点备份及独立完整文件哈希后，才能允许释放实例；另行记录存储完成状态。

The completed `1.1.1.dev0` CPU acceptance below is historical evidence. It must not be relabeled as final `1.2.0` package acceptance. No new paid API, rental or GPU training belongs to release preparation.

下方已完成的1.1.1.dev0 CPU验收为历史证据，不能改称最终1.2.0安装包验收；发布准备不新增付费API、租机或GPU训练。

## Historical local966 demo acceptance — 2026-10-12 / 本地966演示验收

This completed owner-authorized local demo integration keeps published `v1.1.0` and historical evidence intact. [DEMO966_CPU](DEMO966_CPU.md) records the fixed policy and controls. The second real CPU attempt passes 80/80 checks and the independent actual-receipt audit passes 36 checks. Owner controls are externally preserved and actual manager/API/browser activation passes. The [acceptance summary](../reports/evidence/demo966-cpu-20261012/acceptance-summary.json) preserves both attempts; the [activation summary](../reports/evidence/demo966-cpu-20261012/demo-activation-summary.json) binds the real default/compare/BM25 checks.

所有者另行授权的本地演示接入已完成，保留已发布 `v1.1.0` 及历史证据。新接入文档记录固定配置及工具；第二次真实 CPU 运行通过 80 项检查，独立实际回执审核通过 36 项检查。所有者工具已外置保存，真实管理器／接口／浏览器激活通过。验收概要保留两次运行，激活概要绑定真实默认／对照／BM25 检查。

- [x] Prepare an immutable 966 seed 42 candidate with fixed fusion 0.5/k60; preserve old releases and the 202-question benchmark.
  准备不可变 966 次更新 seed 42 候选，固定融合 0.5／60，保留旧发布和 202 题基准。
- [x] Implement separate CPU demo controls with rerank initially selected, explicit BM25/compare modes and guarded process ownership.
  实现独立 CPU 演示工具，初始选择重排，保留 BM25／对照及进程归属检查。
- [x] Verify actual installed model loading, all 81 API/offline full pools and fused orders, warm sequential latency and five injected failure paths on CPU; second attempt 80/80 checks pass.
  验证实际安装包模型加载、81 题接口／离线完整候选池及融合顺序、预热串行延迟及五项故障注入；第二次运行 80 项检查通过。
- [x] Independently review the exact CPU acceptance receipt: 36 checks pass and the public summary binds the audit hash.
  独立审核准确 CPU 回执：36 项检查通过，公开概要绑定审核哈希。
- [x] Verify external manager/API/UI activation: exact process identities, real browser default/compare/BM25 states and live omitted-mode API checks pass; 966 is the local demo default with BM25 baseline/fallback retained.
  外置管理器／接口／界面激活通过：准确进程身份、真实浏览器默认／对照／BM25 状态及未指定模式接口检查通过；966 为本地演示默认，保留 BM25 对照与降级。
- [x] Preserve the owner controls and accepted evidence externally; 32 files / 527, 219 bytes are verified, with reusable start/status/stop commands and actual measured limitations.
  将所有者工具和通过的证据保存外置盘：32 个文件／527, 219 字节已核验，提供可复用启动／状态／停止命令及实测范围。

The 966 checkpoint has one training seed 42. Question-bootstrap intervals do not establish multi-seed reproducibility; original R5's three-seed mean stays attached to its own historical model family. No additional GPU training, paid API use or test/holdout tuning is included.

966 断点只有一个训练种子 42。问题 Bootstrap 区间不能证明多种子复现；原 R5 三种子均值仍归属历史模型系列。本次不追加 GPU 训练、付费 API 或测试／留出集调参。

## Published v1.1.0 scope / 已发布v1.1.0范围

**2026-10-10.** The owner authorized public release `v1.1.0` under Apache-2.0. The main experiment conclusion is repeated R5+RRF improvement on the same 202-question validation benchmark; the new 59-question holdout did not replicate that gain. BM25 remains the formal default, with R5 seed 42 available explicitly as a candidate. Exact-source CI, packaging and publication identities belong to the versioned release manifest. No further GPU training is part of this release.

**2026-10-10。**所有者授权 Apache-2.0 公开版 `v1.1.0`。主要实验结论为同一 202 题验证基准上的 R5+RRF 多种子复现提升，新 59 题留出集未复现该提升。BM25 保持正式默认，R5 种子 42 可显式作为候选启用。准确源码 CI、打包和发布标识以版本发布清单为准；本次发布不新增 GPU 训练。

## Completed / 已完成

- [x] Native PyTorch training, model export/reload, GPU recovery and reproducible dependency checks.
  原生 PyTorch 训练、模型导出／重载、GPU 恢复及依赖复现检查。
- [x] First formal held-out campaign with negative findings retained and BM25 selected by validation.
  首轮正式留出评测完成，保留负面结果，按验证结果选择 BM25。
- [x] Real-corpus CPU API demo of the formal BM25 release: search, empty, invalid filters and explicit fallback.
  正式 BM25 发布包的真实语料 CPU 接口演示：检索、空结果、非法筛选与明确降级。
- [x] Validation-v4 labels sealed and all 25 historical systems reevaluated on the same version.
  验证 v4 标签已封存，历史 25 系统使用同一版本完成重评。
- [x] Six matched-loss runs completed; conditional bounds independently checked and pairwise direction stopped.
  六次匹配损失运行完成，条件界限经独立核验，排序损失方向停止。
- [x] Current project entry, model/data cards, scoped results and sanitized evidence consolidated on the documentation branch.
  当前项目入口、模型／数据说明、注明范围的结果及脱敏证据已汇总到文档分支。

## Historical version-1 delivery / 第一版历史交付

| ID | Work / 工作 | Acceptance / 验收 | Status / 状态 |
| --- | --- | --- | --- |
| V1-01 | Review and merge code/docs dependency chain / 审阅合并代码文档依赖链 | PR #2 → #3 → #4 → #5, current CI checked, no unreviewed code change / 核对当前 CI 与代码范围 | Reviewed chain; merge records linked in PR #2–#5 / 已评审依赖链，合并记录见 PR #2～#5 |
| V1-02 | Freeze delivery identity / 冻结交付标识 | Source commit, evaluated release/candidate role, configs, artifact hashes and restoration instructions / 源码、版本角色、配置、哈希与恢复说明 | 1.0.0; exact source/checks/tag bound by release manifest / 1.0.0，准确源码／检查／标签由发布清单关联 |
| V1-03 | Render an honest real-corpus demonstration / 渲染真实语料演示 | UI screenshot/video, source links, empty/error/fallback; label baseline vs experimental mode / 截图或录像、来源及异常流程，并标明模式 | Complete: actual API and browser evidence in release-demo-20261009 / 已完成真实接口与浏览器证据 |
| V1-04 | Durable artifact handoff / 持久产物交付 | Selected artifacts restored and hashes checked at owner-approved durable location / 在所有者认可存储恢复并核验选定产物 | Selected R5 inference assets restored externally; native training checkpoints retained on server, full checkpoint backup not complete / 选定 R5 推理资产已外置恢复；原训练断点保留服务器，全部断点备份未完成 |
| V1-05 | Personal project explanation / 个人项目讲解 | Owner explains one training modification and one failure analysis; AI assistance recorded / 所有者解释一次训练修改与失败分析，并如实记录 AI 协助 | To demonstrate / 待演示 |

## Optional R5 product demonstration / 可选 R5 产品演示

**2026-10-10 completion:** seed 42/epoch 3 and fusion 0.5/k60 were frozen before question generation. Real export/reload and shared offline/online API ordering pass; all 59 server/Mac rankings agree. The new 59-question/59-group synthetic holdout has sealed agent labels and complete first scoring. R5 MRR change is −0.008024 and pooled nDCG change is −0.065773; both quality gates fail. Warm sequential Mac CPU p95 is 622.065 ms. [Independent report](../reports/r5-independent-holdout-2026-10-10.md) · [implementation](R5_INTEGRATION.md).

**2026-10-10 完成：**问题生成前已冻结上述权重及融合。真实导出／重载与离在线共用接口排序通过，服务器与 Mac 的 59 题排序一致。新 59 题／59 组为合成留出集，Agent 标签已封存并完成首次计分。R5 MRR 差值为 −0.008024，池化 nDCG 为 −0.065773，两个质量门槛均未通过；Mac CPU 串行预热后 p95 为 622.065 毫秒。准确范围见独立报告与接入说明。

| ID | Work / 工作 | Status and acceptance / 状态与验收 |
| --- | --- | --- |
| R5-01 | Fix the serving checkpoint/configuration / 固定服务断点配置 | Complete: seed 42/epoch 3 fixed by existing policy; three-seed mean stays a family result / 已完成：按原策略固定断点，不混用三种子均值 |
| R5-02 | Share offline/online rank fusion / 共用离在线排名融合 | Complete: natural top-50, alpha 0.5 / k 60, ties and identity checked; zero ordering mismatches / 已完成：自然候选、融合、同分及身份核验，排序零差异 |
| R5-03 | Verify real-model API/UI and latency / 验证真实模型接口界面及延迟 | Complete: CPU reload, real comparison/empty/filter rejection and browser capture; fixed 150-request warm p95 target passes locally / 已完成：真实重载、对照／空结果／筛选拒绝及浏览器截图；固定 150 请求本机延迟达标 |
| R5-04 | Independent held-out evaluation / 独立留出评测 | Complete: 59 questions retained, 789 pairs sealed after A789/B165/C12; both quality gates fail / 已完成：保留 59 题，789 对经 A789/B165/C12 后封存；两个质量门槛失败 |

These completed tasks used CPU inference only: no GPU startup, retraining, new rental or paid API. The R5 candidate API sorts by fixed fusion and displays raw logits separately. It requires `--allow-candidate`; failed holdout gates do not promote it or alter the published BM25 default. No tuning follows on this holdout.

已完成的任务仅使用 CPU 推理，没有启动 GPU、重训、新租机或付费 API。R5 候选接口按固定融合排序，原始 logit 单独展示，启动须显式允许候选。留出集门槛失败不晋升候选，也不改变已发布 BM25 默认；不据此留出集继续调参。

## Research backlog / 研究待办

No further pairwise sweep is planned. The 316-pair shared-pool extension is required only if complete scores for the new runs are requested. Future research should first justify representative training questions, multi-positive labels or reviewed negatives using train/validation evidence, and predeclare one controlled change. Test data does not guide those choices.

不计划继续排序损失扫参。仅在需要新运行完整成绩时，才需 316 对共享池扩展。未来研究先依据训练／验证证据论证有代表性的问题、多正例或经审负例，再预定单一受控改动；测试数据不参与这些选择。

## Historical v1.0.0 release evidence / v1.0.0 历史发布证据

Package versions are `1.0.0`; isolated wheel identity checks and clean CPU reproduction pass. The real-corpus UI has actual normal/empty/error/fallback screenshots. [Release procedure](RELEASE.md), [packaging verification](../reports/release-packaging-verification-2026-10-09.md) and [independent audit](../reports/release-independent-audit-2026-10-09.md) describe scope. Final source, CI runs and asset hashes are tracked by the [versioned release](https://github.com/Hector-sau/devsearch/releases/tag/v1.0.0) manifest. Packaging/audit reports retain their preparation-time scope; historical GPU receipts keep their original source identity.

包版本为 `1.0.0`，隔离安装标识及独立 CPU 复现通过；真实语料界面已保存正常／空结果／错误／降级截图。发布流程、打包核验与独立审核说明范围；最终源码、CI 与资产哈希以版本发布清单为准。打包／审核报告保留准备时范围，历史 GPU 回执保留原源码归属。

## Public v1.1.0 delivery / 公开 v1.1.0 交付

Follow [RELEASE](RELEASE.md): review PR #6, preserve and segregate private draft assets, validate the final main commit, build and verify the Apache-2.0 wheel, publish immutable `v1.1.0`, then verify public downloads. [CPU validation replay](VALIDATION_REPLAY.md) recomputes all 25 historical systems from owner-held frozen inputs without a GPU. The original `v1.0.0` tag and historical CUDA receipts remain unchanged.

依发布流程审阅 PR #6、保存并隔离私有草稿附件、验证最终主分支、打包核验许可证，再发布不可变 `v1.1.0` 并复查公开下载。CPU 验证重放用所有者持有的冻结输入重算历史 25 系统，无需 GPU；原 `v1.0.0` 标签与历史 CUDA 回执保留不变。
