# Experiment results / 实验结果

**Compiled 2026-10-09.** The first formal campaign is complete; BM25 is the evaluated default. R5 with fixed rank fusion is the stronger existing validation candidate. The later matched-loss experiment is complete and rejected continuation of pairwise loss. [Evidence provenance](evidence/project-20261009/provenance.json).

**汇总于 2026-10-09。**首轮正式实验完成，BM25 保持已评测默认；固定排名融合的 R5 是现有较好的验证候选。后续匹配损失实验已完成，并停止排序损失方向。证据来源见上述清单。

## 1. Original formal campaign / 首轮正式实验

Source `e3168ca6fef9b64ea0688ea059332c729125e30f`; selection locked on validation before test scoring. The evaluated release is `release-ce3160fc579a52ae2ec30d15412853eb`, with no neural model. All 197 test questions are synthetic and the test labels are agent judgments.

源码提交如上，先按验证集锁定选择，再评测测试集。已评测发布包默认 BM25、不包含神经模型；197 个测试问题为合成问题，测试标签为 Agent 判定。

| System / 系统 | Test MRR@10 | Test nDCG@10 |
| --- | ---: | ---: |
| BM25 | 0.892519 | 0.790862 |
| Frozen encoder / 冻结编码器 | 0.202427 | 0.203488 |
| Full fine-tuning / 完整微调 | 0.873469 | 0.754642 |

Full tuning did not establish improvement: the MRR difference interval includes zero and nDCG declined. Original selection thresholds were MRR gain ≥ 0.02, nonnegative nDCG change and p95 ≤ 2,000 ms; no threshold was weakened to accept the later experiments. [Formal report](evidence/gpu-20261008/formal-final-report.json) · [selection](evidence/gpu-20261008/formal-validation-selection.json).

完整微调未证明提升：MRR 差值区间包含零，nDCG 下降。原选择门槛为 MRR 提升至少 0.02、nDCG 不下降及 p95 不超过 2,000 毫秒；没有为后续实验放宽门槛。

## 2. Earlier controlled development / 早期受控开发

The historical v1 validation set labeled one positive per question. On that version, R4 fused MRR averaged 0.893871; R5 changed the negative mix from 2 hard + 2 random to 3 + 1 and reached 0.900565. The incremental interval included zero. Replacing eight reviewed negatives (R6) and lowering BCE positive weight (R7) did not meet their adoption rules. [Full dated development report](gpu-experiment-results-2026-10-08.md).

原 v1 验证集每题仅标一个正例。该版本下 R4 融合 MRR 均值为 0.893871，R5 将负例组成从两个困难加两个随机改为 3+1，达到 0.900565，但增量区间包含零。八条负例替换（R6）和降低 BCE 正例权重（R7）均未达到采纳条件。完整记录见带日期的发展报告。

## 3. Expanded validation labels / 扩展验证标签

No model was retrained for this reevaluation. All 25 historical systems use the same sealed v4 labels: **202 questions, 4,571 labeled pairs**, including six human-assisted resolutions. MRR values from v1 and v4 are different measurements and must not be presented as an improvement caused by training.

本次重评没有重新训练模型。历史 25 个系统使用同一封存 v4 标签：**202 个问题、4,571 对标签**，含六条人工辅助确认。v1 与 v4 的 MRR 属于不同标注口径，不能把差值包装成训练提升。

| System / 系统 | MRR@10 | Pooled nDCG@10 |
| --- | ---: | ---: |
| BM25 | 0.901416 | 0.827758 |
| R5 fused, three-seed mean / R5 融合，三种子均值 | 0.941062 | 0.846782 |
| Difference / 差值 | +0.039645 | +0.019024 |
| Descriptive paired 95% interval / 描述性配对 95% 区间 | [0.018028, 0.063353] | [-0.001527, 0.039868] |

R5 fused MRR per seed: **42: 0.934076; 43: 0.944059; 44: 0.945050**; sample SD 0.006070. The prespecified 199-question sensitivity has MRR difference +0.041918 and nDCG difference +0.018843. Seeds are averaged inside evidence groups before bootstrap; they are not additional independent questions.

R5 融合 MRR 每种子结果如上，样本标准差 0.006070。预定 199 题敏感性分析的 MRR 差值为 +0.041918、nDCG 为 +0.018843。Bootstrap 前在证据组内平均种子，不能把种子视为新增独立问题。

The questions were repeatedly used in development. These descriptive intervals are not corrected for repeated model selection; nDCG's interval includes zero. No new test evaluation, real-user study or R5 serving promotion is established. [Complete aggregate evidence](evidence/project-20261009/validation-v4-summary.json).

问题已反复用于开发，描述性区间未校正多次模型选择，nDCG 区间包含零。现有证据不代表新增测试评测、真人研究或 R5 服务晋升。

## 4. Matched BCE versus pairwise loss / 匹配 BCE 与排序损失

Source `cc0e7f0c6898af8fc7635c5ce9ffd4a4df8c4f89`, plan SHA-256 `1984f21a76cf98f0d031e155c35bf392fd4df68832fb915b16d19c053f8050d5`. Fixed reviewed R6 data and MiniLM; two losses × seeds 42/43/44; fresh initialization, fixed epoch 3, 387 updates each. All six completed on RTX 4090. From probe start to the final run receipt: **661.29 seconds**; this is not total provider-billed time. All 24 new epoch checkpoint metadata records and the preserved 60 prior checkpoints were verified in the recorded workflow.

源码和计划哈希如上。固定经审核 R6 数据与 MiniLM，两个损失乘三个种子，重新初始化、固定第三轮，每组 387 次更新；六次 RTX 4090 运行全部完成。从短测开始到最后运行回执为 **661.29 秒**，不等同平台总计费时间。流程已核验 24 个新轮次断点元数据及保留的 60 个旧断点。

The new rankings introduce 316 unjudged top-ten pairs across raw/fused modes. Full quality evaluation is therefore pending. However, keeping existing labels fixed and enumerating all possible shared binary assignments for missing labels bounds the paired MRR difference:

新排名在原始／融合模式下引入 316 对未判定前十候选，因此完整质量评测仍待完成。但在固定已有标签、枚举缺失标签共享二元判定的条件下，可界定配对 MRR 差值：

| Necessary continuation condition / 必要继续条件 | Observed bound / 已核验界限 |
| --- | --- |
| Mean gain ≥ +0.005 over the new matched BCE arm / 相对本轮匹配 BCE 均值提升至少 +0.005 | Fused upper bound **+0.000826**; raw upper bound **−0.013326** / 融合与原始上限均未达到 |
| Positive gain in ≥ 2 of 3 seeds / 至少三个种子中两个改善 | At most **1 of 3** in either mode / 两种模式均最多 **1 个** |

**Decision: stop this pairwise-loss direction.** These are mathematical bounds conditional on frozen labels, not completed point metrics, bootstrap intervals or promotion evidence. Additional judgments are unnecessary solely to make this stopping decision. [Completion, coverage and bounds](evidence/project-20261009/grouped-campaign-summary.json).

**决定：停止当前排序损失方向。**这里是冻结标签条件下的数学界限，不是完整点估计、Bootstrap 区间或晋升证据；仅为停止决定，无需新增补标。

## 5. Performance and product evidence / 性能与产品证据

Original formal RTX 4090 p95 latency was approximately **6.6 ms BM25 / 147–150 ms neural reranking**, measured under its recorded fixed workload. A separate earlier candidate passed reload parity and measured p95 **144.145 ms** at concurrency 1. Neither is an R5+fusion API benchmark or a production SLA. [Formal latency](evidence/gpu-20261008/formal-latency.json) · [earlier candidate](evidence/gpu-20261008/followup-candidate-verification.json).

原正式 RTX 4090 固定负载的 p95 约为 **BM25 6.6 毫秒／神经重排 147～150 毫秒**；另一较早候选通过重载一致性，在并发 1 下 p95 为 **144.145 毫秒**。这些不是 R5 加融合接口基准，也不是生产 SLA。

The formal BM25 release has verified real-corpus CPU API and rendered-browser demonstrations. Older diagnostic screenshots remain marked as diagnostic; the new [release demo](evidence/release-demo-20261009/README.md) uses the actual formal BM25 release. Actual R5+fusion serving remains unfinished. [CPU demo evidence](evidence/cpu-demo-20261008/README.md) · [release tasks](../docs/TASKS.md).

正式 BM25 发布包已有真实语料 CPU 接口与浏览器演示。旧诊断截图继续标为诊断；新的发布演示使用实际正式 BM25 发布包。真实 R5 加融合服务仍未完成。
