# Release checklist / 发布清单

**2026-10-10.** The owner authorized public release `v1.1.0` under Apache-2.0. The main experiment conclusion is repeated R5+RRF improvement on the same 202-question validation benchmark; the new 59-question holdout did not replicate that gain. BM25 remains the formal default, with R5 seed42 available explicitly as a candidate. Exact-source CI, packaging and publication identities belong to the versioned release manifest. No further GPU training is part of this release.

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

**2026-10-10 completion:** seed42/epoch3 and fusion0.5/k60 were frozen before question generation. Real export/reload and shared offline/online API ordering pass; all 59 server/Mac rankings agree. The new 59-question/59-group synthetic holdout has sealed agent labels and complete first scoring. R5 MRR change is −0.008024 and pooled nDCG change is −0.065773; both quality gates fail. Warm sequential Mac CPU p95 is 622.065 ms. [Independent report](../reports/r5-independent-holdout-2026-10-10.md) · [implementation](R5_INTEGRATION.md).

**2026-10-10 完成：**问题生成前已冻结上述权重及融合。真实导出／重载与离在线共用接口排序通过，服务器与 Mac 的 59 题排序一致。新 59 题／59 组为合成留出集，Agent 标签已封存并完成首次计分。R5 MRR 差值为 −0.008024，池化 nDCG 为 −0.065773，两个质量门槛均未通过；Mac CPU 串行预热后 p95 为 622.065 毫秒。准确范围见独立报告与接入说明。

| ID | Work / 工作 | Status and acceptance / 状态与验收 |
| --- | --- | --- |
| R5-01 | Fix the serving checkpoint/configuration / 固定服务断点配置 | Complete: seed42/epoch3 fixed by existing policy; three-seed mean stays a family result / 已完成：按原策略固定断点，不混用三种子均值 |
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
