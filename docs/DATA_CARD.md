# Data card / 数据说明

## Source and task / 来源与任务

The measured corpus is the English `pytorch/tutorials` snapshot at `11512db7cbbcc4b530ecfc63205d3fed13fa190d`. The task is question–passage relevance for documentation search. The question set was generated and reviewed through agent workflows; it is not a set of real-user search logs.

实测语料来自上述固定提交的英文 `pytorch/tutorials`，任务是文档检索中的问题—段落相关性判断。问题由 Agent 工作流生成并审查，不是真人搜索日志。

| Partition / 划分 | Questions / 问题数 | Use / 用途 |
| --- | ---: | --- |
| Train / 训练 | 515 | Fine-tuning; 2,575 pairs for one positive and four negatives each / 微调，每题一正四负，共 2,575 对 |
| Validation / 验证 | 202 | Selection, controlled comparisons and error analysis / 选择、受控对照与错例分析 |
| Test / 测试 | 197 | Original locked formal campaign only / 原锁定正式实验 |

Evidence-group ownership governs splits and training-only negative mining. Candidates are natural BM25 top-50 results; known positives are never inserted into retrieval. Exact source/configuration and frozen membership checks are defined in [02](02-data-and-evaluation.md) and the private package manifests.

按证据组归属划分数据，负采样仅使用训练归属内容。候选来自自然 BM25 前 50，已知正例不补入检索。固定来源、配置与成员校验遵循第 02 号规格和私有数据包清单。

## Label versions / 标签版本

| Label set / 标签集 | Scope and provenance / 范围与来源 |
| --- | --- |
| Original validation v1 / 原验证 v1 | One known positive per question; incomplete alternative-answer coverage / 每题一个已知正例，其他有效答案覆盖不完整 |
| Formal test bank / 正式测试审查库 | 9,852 finalized items, 11,627 agent responses, zero human judgments in this bank / 9,852 项确定，11,627 次 Agent 回答，该库无人工判定 |
| Expanded validation v4 / 扩展验证 v4 | 4,571 determinate pairs across 202 questions: 4,345 new and 226 reused pilot pairs / 202 题、4,571 对确定标签，含 4,345 新项和 226 试点复用项 |

The v4 release `validation-v2-sampled202-human6-v4` uses full primary review, required sampled/targeted second review and dispute resolution. New-item A review is 4,345/4,345; required B review is 1,452/1,452. The random audit found 7 issues in 435 pairs, below its 22-item expansion threshold. Six specific resolutions were confirmed by the owner after agent recommendations and are labeled **human-assisted**, not independent human annotations.

v4 发布采用全量初审、必需的抽样／重点复审与分歧处理。新项 A 审 4,345/4,345，必需 B 审 1,452/1,452；随机审计 435 对中有 7 项问题，低于 22 项扩审阈值。所有者按 Agent 建议确认了六条具体裁决，标记为**人工辅助确认**，不冒充独立人工标注。

The older optional 30-card human exercise remains separate from these six confirmations and is not an independent accuracy estimate. Aggregate provenance and release hashes are in [validation-v4-summary.json](../reports/evidence/project-20261009/validation-v4-summary.json).

早期可选的 30 卡人工练习与六条确认独立记录，不能用作独立准确率估计。汇总来源及发布哈希见上述证据。

## Coverage and interpretation / 覆盖与解释

The historical 25 systems have 100% judged top-ten coverage under v4, but only about 45.25% of their top-50 query–passage pairs are judged. Pooled nDCG and recall remain conditional on known positives. Relabeling validation changes measurement; it does not train better model weights by itself.

历史 25 个系统在 v4 下前十判定覆盖率为 100%，但前 50 问题—段落对仅约 45.25% 已判定。池化 nDCG 和召回仍以已知正例为条件。补充验证标签改变衡量方式，不会自动改善模型权重。

The later six-run comparison adds 316 unjudged top-ten pairs across raw and fused rankings (104 for fused alone). Its full new-model quality report is incomplete. Conditional bounds already reject further pairwise-loss experiments; labeling those pairs is optional if the goal is only that stopping decision. Complete comparative metrics require a shared successor label version across all compared systems.

后续六次对照在原始与融合排名中新增 316 对未判定前十候选，其中融合部分 104 对；完整新模型质量报告尚未完成。条件界限已足以拒绝继续排序损失方向，仅为停止决定无需补标；若需完整对比指标，则所有系统须使用共享的后续标签版本。

Raw text, query text, review records, private salt, qrels and runtime indexes are excluded from Git. Full reproduction requires owner-provided private artifacts. This repository does not package or relicense upstream documentation or model weights.

原文、问题、审查记录、私有盐值、qrels 和运行索引不进入 Git。完整复现需要所有者提供私有产物；本仓库不打包或重新授权上游文档和模型权重。
