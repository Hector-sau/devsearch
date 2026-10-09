# Release checklist / 发布清单

**2026-10-09.** The original engineering experiment can close with the evaluated BM25 default. A live R5 fusion demonstration is a separate unfinished delivery feature. This checklist does not authorize a merge, public publication or additional paid training.

**2026-10-09。**原工程实验可按已评测 BM25 默认结项；R5 融合真实演示是尚未完成的交付功能。本清单不构成合并、公开发布或额外付费训练的授权。

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
| V1-01 | Review and merge code/docs dependency chain / 审阅合并代码文档依赖链 | PR #2 → #3 → #4, current CI checked, no unreviewed code change / 核对当前 CI 与代码范围 | Open; merge not performed / 待完成，尚未合并 |
| V1-02 | Freeze delivery identity / 冻结交付标识 | Source commit, evaluated release/candidate role, configs, artifact hashes and restoration instructions / 源码、版本角色、配置、哈希与恢复说明 | Existing receipts present; final delivery reference pending / 已有回执，最终交付引用待定 |
| V1-03 | Render an honest real-corpus demonstration / 渲染真实语料演示 | UI screenshot/video, source links, empty/error/fallback; label baseline vs experimental mode / 截图或录像、来源及异常流程，并标明模式 | CPU API verified; rendered formal UI pending / CPU 接口已验，正式界面待验 |
| V1-04 | Durable artifact handoff / 持久产物交付 | Selected artifacts restored and hashes checked at owner-approved durable location / 在所有者认可存储恢复并核验选定产物 | Latest small results local; large transfer waits for external drive / 最新小结果已本地保存，大文件待外置盘 |
| V1-05 | Personal project explanation / 个人项目讲解 | Owner explains one training modification and one failure analysis; AI assistance recorded / 所有者解释一次训练修改与失败分析，并如实记录 AI 协助 | To demonstrate / 待演示 |

## Optional R5 product demonstration / 可选 R5 产品演示

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
