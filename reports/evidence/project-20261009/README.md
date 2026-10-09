# Project evidence summaries / 项目证据摘要

These JSON files contain selected aggregate fields from verified private runtime artifacts. [provenance.json](provenance.json) records source hashes and the exact bytes/hash of each published JSON summary. They contain no query/passage text, per-pair labels, private salt, host access details, weights or optimizer states.

这些 JSON 从已核验的私有运行产物提取汇总字段。来源清单记录原文件哈希及各摘要的字节数／哈希，不包含问题／段落文本、逐对标签、私有盐、主机接入信息、权重或优化器状态。

- [validation-v4-summary.json](validation-v4-summary.json): all 25 historical systems on the same sealed labels; per-seed aggregates, descriptive intervals and label provenance. / 同一标签下的历史 25 系统、种子汇总、描述性区间与标签来源。
- [r5-candidate-checkpoints.json](r5-candidate-checkpoints.json): three selected R5 checkpoints and configuration hashes; no weights. / 三个 R5 选中断点及配置哈希，不含权重。
- [grouped-campaign-summary.json](grouped-campaign-summary.json): six-run completion, coverage gaps and conditional stopping bounds. The v1 monitoring scores inside each run are not complete v4 quality scores. / 六次运行完成、覆盖缺口和条件停止界限；各运行内 v1 观察指标不等于完整 v4 成绩。

The old evaluated BM25 release and live R5 fusion are different delivery states. Follow [RESULTS](../../RESULTS.md) and [TASKS](../../../docs/TASKS.md) for interpretation.

旧已评测 BM25 发布包与 R5 融合服务属于不同交付状态，解释以结果和任务清单为准。
