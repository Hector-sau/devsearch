# Release checklist / 发布清单

**2026-10-09.** The original engineering experiment can close with the evaluated BM25 default. The real-corpus BM25 UI is verified; a live R5 fusion demonstration is a separate unfinished feature. The owner authorized the private version-1 integration/publication on 2026-10-09. Final source/check/asset identities are recorded in the versioned release manifest; public visibility and additional paid training are outside this delivery.

**2026-10-09。**原工程实验可按已评测 BM25 默认结项；真实语料 BM25 界面已核验；R5 融合演示是独立的未完成功能。所有者已于 2026-10-09 授权私有第一版整合与发布。准确源码／检查／资产以版本发布清单为准，公开状态和额外付费训练不属于本次交付。

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

## Minimum version-1 delivery / 第一版最低交付

| ID | Work / 工作 | Acceptance / 验收 | Status / 状态 |
| --- | --- | --- | --- |
| V1-01 | Review and merge code/docs dependency chain / 审阅合并代码文档依赖链 | PR #2 → #3 → #4 → #5, current CI checked, no unreviewed code change / 核对当前 CI 与代码范围 | Reviewed chain; merge records linked in PR #2–#5 / 已评审依赖链，合并记录见 PR #2～#5 |
| V1-02 | Freeze delivery identity / 冻结交付标识 | Source commit, evaluated release/candidate role, configs, artifact hashes and restoration instructions / 源码、版本角色、配置、哈希与恢复说明 | 1.0.0; exact source/checks/tag bound by release manifest / 1.0.0，准确源码／检查／标签由发布清单关联 |
| V1-03 | Render an honest real-corpus demonstration / 渲染真实语料演示 | UI screenshot/video, source links, empty/error/fallback; label baseline vs experimental mode / 截图或录像、来源及异常流程，并标明模式 | Complete: actual API and browser evidence in release-demo-20261009 / 已完成真实接口与浏览器证据 |
| V1-04 | Durable artifact handoff / 持久产物交付 | Selected artifacts restored and hashes checked at owner-approved durable location / 在所有者认可存储恢复并核验选定产物 | Small code/report/demo/receipt assets accompany v1; large transfer waits for external drive / 小型代码／报告／演示／回执随版本交付，大文件待外置盘 |
| V1-05 | Personal project explanation / 个人项目讲解 | Owner explains one training modification and one failure analysis; AI assistance recorded / 所有者解释一次训练修改与失败分析，并如实记录 AI 协助 | To demonstrate / 待演示 |

## Optional R5 product demonstration / 可选 R5 产品演示

**2026-10-09 authorized continuation:** seed42/epoch3 and fusion0.5/k60 are frozen before new question generation. Shared offline/online fusion, hash-bound candidate policy, explicit fallback and CPU acceptance tooling are implemented. Historical606ranking replay has zero mismatches. A new59-question/59-group holdout is independently agent-reviewed and sealed, with one exclusion before retrieval. Real R5 export, inference, pooled labeling and latency remain pending access to remote-only weights; no GPU started. [Implementation and exact continuation](R5_INTEGRATION.md).

**2026-10-09 已授权继续：**问题生成前已冻结上述权重及融合。共用排序、候选哈希关联、明确降级及 CPU 验收工具已实现；历史 606 组排名重放零差异。新 59 题／59 组经独立 Agent 审核封存，检索前排除一组。真实 R5 导出、推理、汇总候选补标及延迟仍须访问仅在远端的权重；未启动 GPU。准确继续步骤见上述说明。

| ID | Work / 工作 | Acceptance / 验收 |
| --- | --- | --- |
| R5-01 | Fix the serving checkpoint/configuration / 固定服务断点配置 | Choose by the recorded development policy; retain three-seed mean as a family result / 按已有开发策略选择，不把三种子均值写成单个模型效果 |
| R5-02 | Share offline/online rank fusion / 共用离在线排名融合 | Same top-50, alpha 0.5, k 60, tie-breaking, source filters and model identity / 候选、参数、同分、筛选与模型标识一致 |
| R5-03 | Verify real-model API/UI and latency / 验证真实模型接口界面及延迟 | Reload parity, example queries, fallback and measured serving workload / 重载一致、演示查询、降级与实测负载 |
| R5-04 | Any stronger held-out claim / 更强的留出集结论 | Lock model and protocol before evaluation; assess whether a fresh independent holdout is needed / 评测前锁定模型协议，评审是否需要新独立留出集 |

These are inference/integration/evaluation tasks, not a request to retrain. Current serving uses raw logits; R5 fusion's offline score cannot be used as evidence of a deployed fused service.

以上是推理、集成与评测任务，不要求重训。当前服务使用原始 logit，不能把 R5 离线融合成绩当作融合服务已部署的证据。

## Research backlog / 研究待办

No further pairwise sweep is planned. The 316-pair shared-pool extension is required only if complete scores for the new runs are requested. Future research should first justify representative training questions, multi-positive labels or reviewed negatives using train/validation evidence, and predeclare one controlled change. Test data does not guide those choices.

不计划继续排序损失扫参。仅在需要新运行完整成绩时，才需 316 对共享池扩展。未来研究先依据训练／验证证据论证有代表性的问题、多正例或经审负例，再预定单一受控改动；测试数据不参与这些选择。

## Release evidence / 发布证据

Package versions are `1.0.0`; isolated wheel identity checks and clean CPU reproduction pass. The real-corpus UI has actual normal/empty/error/fallback screenshots. [Release procedure](RELEASE.md), [packaging verification](../reports/release-packaging-verification-2026-10-09.md) and [independent audit](../reports/release-independent-audit-2026-10-09.md) describe scope. Final source, CI runs and asset hashes are tracked by the [versioned release](https://github.com/Hector-sau/devsearch/releases/tag/v1.0.0) manifest. Packaging/audit reports retain their preparation-time scope; historical GPU receipts keep their original source identity.

包版本为 `1.0.0`，隔离安装标识及独立 CPU 复现通过；真实语料界面已保存正常／空结果／错误／降级截图。发布流程、打包核验与独立审核说明范围；最终源码、CI 与资产哈希以版本发布清单为准。打包／审核报告保留准备时范围，历史 GPU 回执保留原源码归属。
