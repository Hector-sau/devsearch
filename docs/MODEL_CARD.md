# Model card / 模型说明

**Final configuration update, 2026-10-13:** the selected family uses966 optimizer updates/two epochs with fixed fusion0.5/k60 and now has three fresh seeds 42/43/44. The prior verified seed 42 CPU demo remains the serving endpoint; the three-seed mean is neither an ensemble nor the score of that deployed checkpoint. A fresh single-seed 1, 932-update trial failed its duration-improvement rule, so it is not promoted. [Final replication report](../reports/final966-replication-2026-10-13.md).

**最终配置更新，2026-10-13：**选定模型系列为966次优化器更新／两轮及固定融合0.5／60，现已有三个全新种子42／43／44。已有已验收seed42 CPU演示继续作为服务模型；三种子均值不代表集成或已部署断点成绩。全新单一种子1, 932次试验未通过时长改进规则，不予晋升。

This family uses720 training questions and 3, 600 unique weighted-BCE pairs. Fixed epoch 2 endpoints independently pass the47-question development targets; fused useful-MRR mean 0.860934 versus BM25 0.769706, graded-nDCG mean 0.690759 versus 0.612417. All 81 secondary and partial202 historical views are reported separately. Questions are source-derived synthetic development material and relevance labels are frozen agent judgments; no human-user or independent-final-test benefit is claimed.

该系列使用720个训练问题及3, 600对唯一加权BCE样本；固定第二轮模型通过47题开发目标，融合useful-MRR均值0.860934，BM25为0.769706，graded-nDCG均值0.690759，BM25为0.612417。全81题次范围及原202题部分标签视图分别报告；问题为来源文档合成开发材料，相关性标签为冻结Agent判定，不声称真人或独立最终测试收益。

The proposed `1.2.0` source release includes the public model-free application and explicit candidate controls; private trained weights and frozen new execution wrappers are not bundled in the clone. The preferred owner demo uses the accepted966 endpoint, while BM25 remains usable without a model and returns complete fallback results. Exact-source package/service validation belongs to release acceptance, separately from existing `1.1.1.dev0` receipts and private checkpoint archival.

拟发布1.2.0源码包含公开无模型应用及显式候选控制，克隆不附私有权重或新冻结执行包装器；所有者首选演示使用已验收966模型，BM25不依赖模型并提供完整降级。准确源码安装包／服务验证属于发布验收，与已有1.1.1.dev0回执及私有断点归档区分。

Version 1.1.0 publishes the application source under Apache-2.0; trained weights, sealed questions and private review records remain owner-held. [CPU validation replay](VALIDATION_REPLAY.md) checks frozen input hashes and recomputes historical metrics without retraining. Upstream content keeps its own license.

1.1.0 以 Apache-2.0 发布应用源码；训练权重、封存问题和私有审查记录由所有者保存。CPU 验证重放核验冻结输入哈希并重算历史指标，无需重训；上游材料保留原许可证。

**Historical local demo snapshot, 2026-10-12:** the owner-authorized demo uses the separate 966-update seed 42 endpoint and fixed fusion 0.5/k60 after CPU service acceptance. This endpoint has only one training seed; the original R5 three-seed results below remain attached to the original family. [Current demo identity and operation](DEMO966_CPU.md) · [Matched development comparison](../reports/demo966-development-2026-10-12.md). Published `v1.1.0` remains a historical release with its own default and receipts.

**历史本地演示快照，2026-10-12：**所有者授权演示在 CPU 服务验收后使用单独的 966 次更新 seed 42 模型及固定融合 0.5／60。此模型只有一个训练种子；下文原 R5 三种子结果仍归属于原模型系列。当前身份、操作及匹配开发对比见上述链接；已发布 `v1.1.0` 保留自己的默认配置与回执。

## Role and intended use / 角色与用途

The historical `v1.1.0` evaluated release and public model-free example use **BM25 without a neural model**. Neural candidates score the relevance of an English technical question and a documentation passage; they do not generate answers or calibrated correctness probabilities.

历史v1.1.0已评测发布包及公开无模型样例使用 **BM25，不含神经模型**。神经候选模型对英文技术问题与文档段落的相关性评分，不生成答案，也不输出经过校准的答案正确概率。

| Item / 项目 | Value / 值 |
| --- | --- |
| Encoder / 编码器 | `microsoft/MiniLM-L12-H384-uncased` |
| Upstream revision / 上游版本 | `44acabbec0ef496f6dbc93adadea57f376b7c0ec` |
| Trainable model / 可训练模型 | Cross-encoder with a scoring head; 33, 212, 545 parameters / 交叉编码器及评分头 |
| Training engine / 训练实现 | Native PyTorch, AdamW, gradient accumulation, checkpoint recovery / 原生 PyTorch、AdamW、梯度累积、断点恢复 |
| Input limits / 输入长度 | Question 64 tokens; combined pair 256 tokens / 问题 64、联合输入 256 token |
| Measured accelerator / 实测设备 | RTX 4090 24 GB, FP32, TF32 disabled / 关闭 TF32 |
| Verified backends / 已验证后端 | CPU checks and NVIDIA CUDA; MPS remains unverified / CPU 检查和 CUDA，MPS 未验收 |

## R5 experimental candidate family / R5 实验候选组

R5 keeps 515 training positives and uses three BM25-hard plus one random negative per question, weighted BCE, learning rate `5e-5`, and seeds 42/43/44. Selected epochs are **3 / 3 / 2**, using the historical validation selection policy. The three-seed result is not the performance of an ensemble or a particular deployed checkpoint.

R5 保留 515 个训练正例，每题使用三个 BM25 困难负例与一个随机负例，采用加权 BCE、学习率 `5e-5` 和种子 42/43/44；按当时验证选择策略，选中轮次为 **3 / 3 / 2**。三种子均值不代表模型集成或某个已部署断点的成绩。

Selected checkpoint and configuration SHA-256 values are in [r5-candidate-checkpoints.json](../reports/evidence/project-20261009/r5-candidate-checkpoints.json). The real candidate API/UI now serves frozen seed 42/epoch 3 with `alpha=0.5`, `rank_constant=60` on the same natural BM25 top-50. API and offline ordering agree; raw model logits and fusion scores are displayed separately. Candidate activation requires `--allow-candidate`; the published BM25 default is unchanged.

选中断点与配置的 SHA-256 见候选清单。真实候选接口／界面现已使用冻结种子 42、第三轮模型，按上述参数融合同一批自然 BM25 前 50 个候选。接口与离线排序一致，原始 logit 和融合分数分开展示；启动须显式允许候选，已发布 BM25 默认保持不变。

On the expanded v4 validation labels, fused MRR@10 is **0.9410616** averaged across three seeds, compared with **0.9014164** for BM25. The 202 questions were reused for development; this is not fresh held-out or real-user evidence. nDCG uncertainty, per-seed values and sensitivity results are in [RESULTS](../reports/RESULTS.md).

扩展 v4 标签下，融合 MRR@10 三种子均值为 **0.9410616**，BM25 为 **0.9014164**。202 个问题已反复用于开发，因此不属于新留出集或真人证据。nDCG 不确定性、每种子结果及敏感性分析见结果报告。

The frozen seed 42 checkpoint was separately evaluated once on 59 new synthetic questions with sealed agent labels. BM25 MRR@10/nDCG@10 were 0.581302/0.592521; R5 fused values were 0.573278/0.526748. MRR change was −0.008024 (paired 95% interval [−0.072713, 0.057439]); nDCG change was −0.065773 ([−0.125972,−0.005525]). Both unchanged quality gates failed. This candidate is not promoted or tuned on that holdout. [Independent report](../reports/r5-independent-holdout-2026-10-10.md).

冻结的种子 42 断点另外在 59 个新合成问题及封存 Agent 标签上完成一次评测。BM25 的 MRR@10／nDCG@10 为 0.581302／0.592521，R5 融合为 0.573278／0.526748；MRR 差值 −0.008024，配对 95% 区间为 [−0.072713, 0.057439]，nDCG 差值 −0.065773，区间为 [−0.125972,−0.005525]。两个原质量门槛均失败，不晋升候选，也不使用该留出集调参。

Measured warm local CPU API p95 is 622.065 ms, median 572.126 ms: Apple M4 / 16 GiB, four threads, batch 8, concurrency 1, 50 queries × 3 passes and 10 excluded warmups. This passes the point target of two seconds for that sequential in-process workload, including model/fusion/serialization; it excludes cold startup, network transit, concurrent load and GPU performance.

本机 CPU 接口预热后 p95 为 622.065 毫秒，中位数 572.126 毫秒：Apple M4／16 GiB、四线程、批次 8、并发 1、50 题三遍及不计入的 10 次预热。在包含模型／融合／序列化的串行进程内负载下通过两秒点估计目标；不包括冷启动、网络、并发负载或 GPU 性能。

## Later matched-loss experiment / 后续匹配损失对照

The `cc0e7f0` campaign compared grouped BCE against pairwise logistic on fixed reviewed R6 pairs. Six runs completed at fixed epoch 3 and 387 updates each. The pairwise direction failed necessary continuation conditions in both raw and fused modes; it is not the selected candidate. [Completion and bounds](../reports/evidence/project-20261009/grouped-campaign-summary.json).

`cc0e7f0` 实验在固定、经审核的 R6 训练对上比较分组 BCE 与 pairwise logistic。六次运行均完成固定第三轮和 387 次更新；排序损失在原始与融合模式下均未达到必要继续条件，不作为选定候选。

## Assets and limitations / 资产与限制

The selected R5 inference archive is 113, 941, 262 bytes, SHA-256 `1abe8c743bc2e852e61165a98aecb79d642b6badc8e1c860eddda11fed65bdf8`, restored and verified on owner-approved external storage. Native training checkpoints, including optimizer state, remain on the server; this is not a complete checkpoint backup. Restoration preserves the trained model identity and does not replace the original campaign assets below.

选定 R5 推理压缩包为 113, 941, 262 字节，哈希如上，已在获认可的外置盘恢复并核验。含优化器状态的原训练断点保留服务器，这不代表全部断点已备份；恢复保留训练模型身份，不替换下方原正式实验资产。

Original GPU model manifest: `dea5ea7128c636c7cde9a91fe94326fc232095066727b0d712e85f1931da7c40`; original `model.safetensors`: `8c27febef873412d9783b47f39a3d7b86dd6da18f42db32633c4b97f6c66787d`. The CI-converted manifest `6860def4…` describes different serialized bytes; the campaign rejects substitution. Matching an upstream revision alone is insufficient.

原 GPU 模型清单与权重哈希如上。CI 转换版本 `6860def4…` 对应不同的序列化字节，正式流程拒绝替换；仅上游版本相同并不足够。

Weights and optimizer states remain outside Git. Source/model redistribution must retain applicable upstream terms; this card does not grant a new license. The candidate is limited to the measured corpus, English questions and available labels. No production SLA, calibrated answer confidence or human-user benefit has been measured.

权重与优化器状态保存在 Git 外。源码／模型再分发须保留适用的上游条款，本说明不授予新许可证。候选结论限于已测语料、英文问题和现有标签，尚无生产 SLA、校准答案置信度或真人使用收益数据。
