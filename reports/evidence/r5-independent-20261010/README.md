# Real R5 candidate demonstration / 真实 R5 候选演示

This is an actual rendered-browser screenshot from the installed `b76e349` implementation using frozen R5 seed42/epoch3, natural top50 and fixed rank fusion. The model runs on Apple M4 **CPU**, with the candidate explicitly allowed. It is separate from the published BM25 v1.0.0 demonstration.

本图来自已安装的 `b76e349` 实现及真实浏览器，使用固定 R5 种子 42／第三轮、自然前 50 候选和固定融合；模型在 Apple M4 **CPU** 上运行，显式允许实验候选，与正式 BM25 第一版演示区分。

![Actual BM25 versus R5 fused comparison / 实际 BM25 与 R5 融合对照](comparison.jpg)

The displayed operator example is not one of the scored holdout questions. Its550ms single-request timing is illustrative; the fixed150-request benchmark establishes the reported622.065ms warm p95. Separate live API checks passed a normal comparison, empty result and unknown-source HTTP422. Intermediate operator-check assumptions were corrected and preserved in the private receipts; no application code was changed to pass them.

图中为操作演示问题，不属于计分留出集；单次约 550 毫秒只作演示，正式 p95 来自固定的 150 次基准。另行核验了正常对照、空结果及未知来源 HTTP422。中间操作检查的错误假设已更正并保留在私有回执，没有为通过它们修改应用代码。

Screenshot SHA-256 `65f4658fef2ca0ad7966c9fea9a013f7317420a48f3d18aa905dcf48561fdf4a`; source attribution and terms are in [THIRD_PARTY_NOTICES](../../../THIRD_PARTY_NOTICES.md). [Summary](summary.json) contains compact metrics and provenance without private query/label cards or runtime paths. [Result and limitations](../../r5-independent-holdout-2026-10-10.md): quality gates failed, so BM25 remains the formal default. This agent-operated demonstration is not a human user study or a production deployment.

截图哈希及第三方来源条款如上；摘要保存小型指标与溯源，不包含私有问题／标签卡片或运行路径。本次效果门槛未通过，正式默认仍为 BM25；Agent 操作演示不是真人研究或生产部署。
