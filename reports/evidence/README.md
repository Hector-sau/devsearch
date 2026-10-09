# Evidence artifacts / 证据文件

> **Current additions / 当前新增：** [gpu-20261008](gpu-20261008/evidence-manifest.json) contains actual first-campaign GPU and follow-up summaries; [project-20261009](project-20261009/provenance.json) contains expanded validation and the completed grouped-loss comparison. The older description below refers only to the earlier CPU evidence groups. / 新目录分别保存实际 GPU 实验摘要、扩展验证与完成的分组损失对照；下方较早说明仅指原 CPU 证据分组。

> Index of all evidence, including CI runs and private release assets: [EVIDENCE_INDEX](../EVIDENCE_INDEX.md). 全部证据索引。

Raw artifacts behind [the evidence report](../accelerator-readiness-evidence.md). Everything here comes from **CPU runs in one container**: `receipts/` and `rehearsal/` use synthetic or diagnostic data, `real-source/` and `real-data/` use a real public repository prepared with a stand-in tokenizer. Ranking numbers inside `rehearsal/` come from a planted synthetic cue and are not results; the BM25-only numbers in `real-data/` are on synthetic, agent-reviewed questions and say nothing about a reranker or about real users.

| Path | What it is |
| --- | --- |
| `receipts/` | Sealed receipts from `scripts/record_receipt.py` (two full-suite runs with zero skips, one clean-wheel offline campaign), bound to the code snapshot `04a7caf3…`; the portable one installs only from the hashed lock and wheelhouse and records the dependency evidence |
| `rehearsal/` | Doctor report, execution plan, selection, final report, acceptance report, campaign summary and status of the diagnostic campaign (recorded with the receipts above) |
| `real-source/` | `sources.toml` and the preparation report of the **first** real run (word-level stand-in tokenizer, reader before the RST fix); history only |
| `real-data/` | The current real-corpus run: `sources.toml`, preparation report, group audit, selection, carry-over and draft summaries, `question-set-summary.json` (counts, rejection reasons and agent-review verdicts, **no question text**), prepared-data, pairs and validation-candidate manifests, BM25 validation metrics, latency and source-link checks, and `data.toml` with its private salt removed, plus `tokenizer-comparison-standin.json` (configuration and encoding comparison of the stand-in vocabulary). No corpus text, question text or test label is committed |
| `review-pilot/` | The candidate-review pilot: sample composition (strata sizes, prompt hash) and the agreement and cost report of two agent judge runs over 80 blinded items. No item text, no judgment and no item-to-stratum map is committed |
| `cpu-demo-20261008/` | CPU demo of the evaluated BM25 release of the first formal campaign (2026-10-08): commands, exit codes, HTTP status and short responses; no passage text |
| `wheelhouse-*.log`, `install-nvidia-cuda-lock.log` | Output of the wheelhouse downloads and the offline lock install (first version; the locks did not change) |

本目录为[证据报告](../accelerator-readiness-evidence.md)背后的原始文件，全部来自单个容器中的 CPU 运行：`receipts/` 与 `rehearsal/` 使用合成或诊断数据，`real-source/` 与 `real-data/` 使用真实公开仓库及替代分词器。`rehearsal/` 中的排序数字来自植入的合成线索，不是结果；`real-data/` 中仅 BM25 的数字基于合成、经 Agent 审查的问题，不说明重排，也不代表真实用户。`real-data/` 不含语料文本、问题文本或测试标签。
