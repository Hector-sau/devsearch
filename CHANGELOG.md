# Changelog / 变更记录

## 1.1.1.dev0 — local integration, 2026-10-12 / 本地接入

Provisional local service integration for the owner-authorized seed42/966-update CPU demo. This does not change the published `v1.1.0` tag, historical R5 three-seed findings or frozen evaluation decisions. The current demo's operating status and exact acceptance evidence are in [DEMO966_CPU](docs/DEMO966_CPU.md).

暂定本地服务接入用于所有者授权的 seed42／966 次更新 CPU 演示；不改写已发布 `v1.1.0` 标签、历史 R5 三种子结果或冻结评测决策。当前运行状态及准确验收证据见上述文档。

- Check the deadline immediately after the final scoring batch; discard complete but late neural results and return the existing explicit BM25 timeout fallback.
  最后一批评分返回后再次检查时限；完整但已超时的神经结果仍须丢弃，并返回原有的明确 BM25 超时降级。
- Make the unavailable-MPS CLI test independent of the host hardware through an explicit simulated unavailable backend.
  通过明确模拟不可用后端，让 CLI 的 MPS 不可用测试不依赖实际主机硬件。
- Keep the new single-seed development comparison and local demo evidence separate from the original R5 model family and public release.
  将新模型的单一种子开发对比及本地演示证据与原 R5 模型系列、公开发布分开记录。

## 1.1.0 — 2026-10-10

Public portfolio release of the completed R5 inference extension, with Apache-2.0 licensing for original DevSearch code and documentation. The original `1.0.0` tag and its experiment identities are preserved.

完成 R5 推理扩展并整理为公开作品集版本；DevSearch 原创代码与文档采用 Apache-2.0。保留原 `1.0.0` 标签及实验标识。

- Reproduced a validation MRR gain across seeds42/43/44 on the same202-question v4 benchmark: BM250.901416, R5 fused0.934076/0.944059/0.945050, mean0.941062. This is repeated adaptive-validation evidence.
  在同一202题 v4 验证基准上，三个种子均复现 MRR 提升；均值0.941062，高于 BM250.901416，属于反复使用验证集上的重复实验。
- Integrated the actual fixed seed42 MiniLM candidate and shared BM25/neural rank fusion into the CPU API/UI, preserving explicit candidate opt-in and BM25 default.
  真实种子42模型及共用排名融合接入 CPU 接口／界面，保留显式候选开关及 BM25 默认。
- Retained the separately sealed59-question evaluation, including its failed quality gates, and the verified warm Mac CPU timing scope.
  保留另行封存59题评测的门槛未通过结果，以及已核验的 Mac CPU 预热耗时范围。
- Added explicit public reproduction boundaries, upstream license notices and exact-source release verification. Private datasets and review checkpoints are stored outside the public repository.
  明确公开复现范围、上游许可及准确源码发布核验，私有数据与审查备份保存在公开仓库之外。

## 1.0.0 — 2026-10-09

Private publication authorized by the owner on 2026-10-09. The release attachments record the exact final source, successful checks and asset hashes. This version covers the BM25 application and reproducible PyTorch tooling; the R5 fused candidate has offline validation evidence and is not the default serving model.

所有者已于 2026-10-09 授权私有发布，附件记录准确最终源码、通过的检查和资产哈希。本版包含 BM25 应用及可复现 PyTorch 工具；R5 融合候选已有离线验证证据，不作为默认服务模型。

- English documentation ingestion, source-linked chunks, natural BM25 retrieval and filters.
  英文文档采集、可追溯段落、自然 BM25 检索及筛选。
- Native PyTorch MiniLM training, checkpoint recovery, export/reload, bounded CPU/CUDA operator workflows and platform dependency locks.
  原生 PyTorch MiniLM 训练、断点恢复、导出／重载、限时 CPU/CUDA 流程及平台依赖锁。
- FastAPI and Streamlit interfaces with explicit BM25 default and fallback behavior.
  FastAPI 与 Streamlit 界面，明确 BM25 默认和降级行为。
- Shared-label ranking evaluation, group-aware uncertainty, agent-review provenance and six human-assisted resolutions.
  共享标签排序评测、分组不确定性、Agent 审查来源与六条人工辅助确认。
- Preserved original held-out findings, R5 adaptive-validation results and the completed six-run matched-loss stopping decision.
  保留首轮留出集结果、R5 反复使用验证集的结果及六次匹配损失实验的停止决定。
- Project guide, data/model cards, CPU quick start, evidence index and release handoff.
  项目导读、数据／模型说明、CPU 快速开始、证据索引及发布交接。

Full private experiment artifacts require owner access; large checkpoint handoff is deferred until external storage is available. No public repository visibility change or redistribution license is granted by this release.

完整私有实验产物需要所有者授权访问，大型断点交接等待外置存储。本版本不改变仓库公开状态，也不授予新的再分发许可证。
