# Validation-v4 arithmetic replay / 验证 v4 算术重放

An actual local CPU run on 2026-10-10 verified the byte identities and structure of the owner's sealed validation inputs, then recalculated the metrics from all 25 frozen ranking orders. No model was loaded or scored, no new label was created, and no GPU or paid API was used.
2026-10-10 的真实本地 CPU 运行校验了所有者封存验证输入的文件身份与结构，再根据全部 25 个冻结排序序列重新计算指标。未加载模型或重新打分、未生成新标签，也未使用 GPU 或付费 API。

- [Aggregate result](aggregate.json): input SHA-256 values, counts, coverage, all-system macro metrics, R5 three-seed means and paired group intervals. No private question, label, ranking order, identifier or filesystem path is exported.
  [汇总结果](aggregate.json)：输入 SHA-256、计数、覆盖率、各系统平均指标、R5 三种子均值及配对组区间。不包含私有问题、标签、排序序列、标识或文件系统路径。
- [Verification receipt](receipt.json): script and output identities, the focused test result, and comparison with the [previous aggregate report](../project-20261009/validation-v4-summary.json). All 25 systems' three metrics and the R5 intervals agree within `1e-12`; the largest absolute difference is `7.771561172376096e-16`.
  [校验回执](receipt.json)：脚本与输出身份、针对性测试结果，以及与[原汇总报告](../project-20261009/validation-v4-summary.json)的比较。全部 25 个系统的三项指标及 R5 区间在 `1e-12` 内一致；最大绝对差为 `7.771561172376096e-16`。
- [Portable replay instructions](../../../docs/VALIDATION_REPLAY.md): required files, command, checks and evidence boundaries.
  [可迁移重放说明](../../../docs/VALIDATION_REPLAY.md)：所需文件、命令、检查与证据范围。

The verified denominator is 202 questions and 202 groups, with 4,571 unique binary labels, 498 positive pairs and no excluded question. Every system has the same 50 candidates per question and fully judged top-10 results. BM25 MRR@10 is `0.901416`; R5 fused seeds 42/43/44 are `0.934076`, `0.944059`, `0.945050`, with mean `0.941062`.
验证分母为 202 个问题和 202 个组，共 4,571 个独立二元标签、498 个正例对，没有排除问题。各系统在每题使用同样的 50 个候选，且前 10 名完全已标注。BM25 的 MRR@10 为 `0.901416`；R5 融合种子 42／43／44 为 `0.934076`、`0.944059`、`0.945050`，均值 `0.941062`。

This is **sealed-input arithmetic evidence**. It does not re-audit the original review ledger, verify model-generated scores, or constitute a fresh holdout evaluation. The benchmark has synthetic questions and was reused during development; intervals remain descriptive. The historical test and the separate holdout regression remain in the [results report](../../RESULTS.md). BM25 remains the application default.
本目录提供**封存输入的算术证据**，不重新审核原审查账本、不验证模型生成的分数，也不构成新的留出集评测。验证问题为合成问题且开发时曾反复使用，区间仍是描述性的。历史测试及独立留出集的退化结果保留在[结果报告](../../RESULTS.md)，应用默认模式仍为 BM25。
