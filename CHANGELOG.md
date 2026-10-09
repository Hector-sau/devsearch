# Changelog / 变更记录

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
