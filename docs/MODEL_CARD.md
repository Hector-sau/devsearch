# Model card / 模型说明

## Role and intended use / 角色与用途

The evaluated application release uses **BM25 without a neural model**. Neural candidates score the relevance of an English technical question and a documentation passage; they do not generate answers or calibrated correctness probabilities.

已评测应用发布包使用 **BM25，不含神经模型**。神经候选模型对英文技术问题与文档段落的相关性评分，不生成答案，也不输出经过校准的答案正确概率。

| Item / 项目 | Value / 值 |
| --- | --- |
| Encoder / 编码器 | `microsoft/MiniLM-L12-H384-uncased` |
| Upstream revision / 上游版本 | `44acabbec0ef496f6dbc93adadea57f376b7c0ec` |
| Trainable model / 可训练模型 | Cross-encoder with a scoring head; 33,212,545 parameters / 交叉编码器及评分头 |
| Training engine / 训练实现 | Native PyTorch, AdamW, gradient accumulation, checkpoint recovery / 原生 PyTorch、AdamW、梯度累积、断点恢复 |
| Input limits / 输入长度 | Question 64 tokens; combined pair 256 tokens / 问题 64、联合输入 256 token |
| Measured accelerator / 实测设备 | RTX 4090 24 GB, FP32, TF32 disabled / 关闭 TF32 |
| Verified backends / 已验证后端 | CPU checks and NVIDIA CUDA; MPS remains unverified / CPU 检查和 CUDA，MPS 未验收 |

## R5 experimental candidate family / R5 实验候选组

R5 keeps 515 training positives and uses three BM25-hard plus one random negative per question, weighted BCE, learning rate `5e-5`, and seeds 42/43/44. Selected epochs are **3 / 3 / 2**, using the historical validation selection policy. The three-seed result is not the performance of an ensemble or a particular deployed checkpoint.

R5 保留 515 个训练正例，每题使用三个 BM25 困难负例与一个随机负例，采用加权 BCE、学习率 `5e-5` 和种子 42/43/44；按当时验证选择策略，选中轮次为 **3 / 3 / 2**。三种子均值不代表模型集成或某个已部署断点的成绩。

Selected checkpoint and configuration SHA-256 values are in [r5-candidate-checkpoints.json](../reports/evidence/project-20261009/r5-candidate-checkpoints.json). Rank fusion uses `alpha=0.5`, `rank_constant=60` on the same natural BM25 top-50. It was evaluated offline. Current serving sorts raw model logits; the live R5 fusion path remains on the [release checklist](TASKS.md).

选中断点与配置的 SHA-256 见候选清单。排名融合使用 `alpha=0.5`、`rank_constant=60`，处理同一批自然 BM25 前 50 个候选，已完成离线评测。当前服务按原始模型 logit 排序，R5 融合的服务接入仍列于发布清单。

On the expanded v4 validation labels, fused MRR@10 is **0.9410616** averaged across three seeds, compared with **0.9014164** for BM25. The 202 questions were reused for development; this is not fresh held-out or real-user evidence. nDCG uncertainty, per-seed values and sensitivity results are in [RESULTS](../reports/RESULTS.md).

扩展 v4 标签下，融合 MRR@10 三种子均值为 **0.9410616**，BM25 为 **0.9014164**。202 个问题已反复用于开发，因此不属于新留出集或真人证据。nDCG 不确定性、每种子结果及敏感性分析见结果报告。

## Later matched-loss experiment / 后续匹配损失对照

The `cc0e7f0` campaign compared grouped BCE against pairwise logistic on fixed reviewed R6 pairs. Six runs completed at fixed epoch 3 and 387 updates each. The pairwise direction failed necessary continuation conditions in both raw and fused modes; it is not the selected candidate. [Completion and bounds](../reports/evidence/project-20261009/grouped-campaign-summary.json).

`cc0e7f0` 实验在固定、经审核的 R6 训练对上比较分组 BCE 与 pairwise logistic。六次运行均完成固定第三轮和 387 次更新；排序损失在原始与融合模式下均未达到必要继续条件，不作为选定候选。

## Assets and limitations / 资产与限制

Original GPU model manifest: `dea5ea7128c636c7cde9a91fe94326fc232095066727b0d712e85f1931da7c40`; original `model.safetensors`: `8c27febef873412d9783b47f39a3d7b86dd6da18f42db32633c4b97f6c66787d`. The CI-converted manifest `6860def4…` describes different serialized bytes; the campaign rejects substitution. Matching an upstream revision alone is insufficient.

原 GPU 模型清单与权重哈希如上。CI 转换版本 `6860def4…` 对应不同的序列化字节，正式流程拒绝替换；仅上游版本相同并不足够。

Weights and optimizer states remain outside Git. Source/model redistribution must retain applicable upstream terms; this card does not grant a new license. The candidate is limited to the measured corpus, English questions and available labels. No production SLA, calibrated answer confidence or human-user benefit has been measured.

权重与优化器状态保存在 Git 外。源码／模型再分发须保留适用的上游条款，本说明不授予新许可证。候选结论限于已测语料、英文问题和现有标签，尚无生产 SLA、校准答案置信度或真人使用收益数据。
