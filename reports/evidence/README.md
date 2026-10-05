# Evidence artifacts / 证据文件

Raw artifacts behind [the evidence report](../accelerator-readiness-evidence.md). Everything here comes from **synthetic or diagnostic data on CPU** except `real-source/`, which is a preparation report for a real public repository prepared with a word-level stand-in tokenizer (no length claim). Ranking numbers inside `rehearsal/` come from a planted synthetic cue and are not results.

| Path | What it is |
| --- | --- |
| `receipts/` | Sealed receipts from `scripts/record_receipt.py` (two full-suite runs with zero skips, one clean-wheel offline campaign), bound to the code snapshot `d029ee34…` |
| `rehearsal/` | Doctor report, execution plan, selection, final report, acceptance report, campaign summary and status of the diagnostic campaign |
| `real-source/` | `sources.toml` and the preparation report for `pytorch/tutorials` at a pinned commit (no corpus text is committed) |
| `wheelhouse-*.log`, `install-nvidia-cuda-lock.log` | Output of the wheelhouse downloads and the offline lock install |

本目录为[证据报告](../accelerator-readiness-evidence.md)背后的原始文件。除 `real-source/`（真实公开仓库的准备报告，使用词级替代分词器，不构成长度结论）外，全部来自 CPU 上的合成或诊断数据；`rehearsal/` 中的排序数字来自植入的合成线索，不是结果。
