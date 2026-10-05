# Estimates and Bounded Experiment Plan / 工作量估算与有界实验方案

Proposed starting recipe, 2026-10-05. No throughput, training outcome or ranking improvement has been measured. This document estimates work from explicit counts; it is not a promise about a particular GPU or project completion date. The target remains an Apple MPS or NVIDIA CUDA profile to verify on the supplied machine.

拟定起始方案，2026-10-05。尚无吞吐、训练结果或排序提升实测。本文从明确数量估算工作量，不承诺特定 GPU 的速度或项目完成日期；目标为用户提供机器上待验证的 Apple MPS 或 NVIDIA CUDA 配置。

## 1. Fixed scope and starting parameters / 固定范围与起始参数

Use `microsoft/MiniLM-L12-H384-uncased`: the official card specifies 12 layers, hidden size 384 and approximately 33 million parameters, requiring downstream fine-tuning. Pin its revision and tokenizer. [Official model card](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)

采用 `microsoft/MiniLM-L12-H384-uncased`：官方说明为 12 层、384 隐藏维度、约 3,300 万参数，需要下游微调。固定模型及 tokenizer 版本。

The maximum planned domain set is 1,000 training questions, 200 validation questions and 300 test questions, subject to independent evidence-group availability. Keep every unique confirmed training positive and up to four training-owned negatives per query: two BM25-hard and two random, deduplicated and excluding training-known positives. Audit assumed negatives; record shortages instead of drawing from held-out groups. The arithmetic below assumes exactly one positive and four negatives per question.

领域最大规划为 1,000 条训练问题、200 条验证问题和 300 条测试问题，以独立证据组数量为准。每题保留全部唯一且已确认的训练正例，以及最多四条训练归属负例：两条 BM25 困难负例、两条随机负例，去重并排除训练已知正例。抽查假设负例，数量不足时记录，不从留出组补充。下文算术假设每题恰好一正四负。

| Condition / 条件 | Training recipe / 训练方案 | Required runs / 必需运行 |
| --- | --- | --- |
| BM25 | Shared fixed candidates; no neural training / 共享固定候选，不训练神经模型 | One reusable retrieval baseline / 一份可复用基线 |
| Frozen encoder / 冻结编码器 | Train scoring head; encoder stays in eval mode / 仅训练评分层，编码器保持 eval | Seed 42, up to 3 epochs / 种子 42，最多 3 轮 |
| Full fine-tuning / 全量微调 | Train encoder and scoring head / 训练编码器及评分层 | Seed 42, up to 3 epochs / 种子 42，最多 3 轮 |

Both conditions use the same pinned pretrained encoder, matched head-initialization seed, split, materialized pairs and validation candidates. Proposed shared settings: FP32, dropout 0.1, AdamW weight decay 0.01, gradient clipping norm 1, effective batch 16, linear decay with 10% optimizer-update warmup, and `BCEWithLogitsLoss` positive weight equal to the actual sampled negative/positive ratio. Proposed learning rates are `1e-3` for the frozen-head condition and `2e-5` for full fine-tuning.

两条件使用同一固定预训练编码器、配对评分层初始种子、划分、实体样本对及验证候选。共用起始设置为 FP32、dropout 0.1、AdamW weight decay 0.01、梯度裁剪范数 1、有效 batch 16、按优化器更新数的 10% warmup 后线性衰减，以及按实际采样负／正比例设置的 `BCEWithLogitsLoss` 正例权重。冻结评分层学习率拟为 `1e-3`，全量微调为 `2e-5`。

Make optimizer defaults explicit: AdamW `betas=(0.9, 0.999)`, `eps=1e-8`, the same stated decay on all trainable parameters, no hidden differential groups, and zero gradients at each accumulation-group boundary. Use the 256 pair-token/64 query-token preparation policy in document 10. Any changed choice is recorded before formal comparisons; it is not a source patch made during execution.

明确优化器默认值：AdamW 的 `betas=(0.9, 0.999)`、`eps=1e-8`，全部可训练参数使用已声明衰减，不暗设不同参数组，每个累积组边界清梯度。采用第 10 号的联合 256／问题 64 token 准备规则；设置改变须在正式比较前记录，不在执行中改源码。

These are unverified starting parameters, not claimed optimal values. Different learning rates make this a comparison of two practical recipes, not a pure one-factor causal test of freezing. Select the completed-epoch checkpoint by validation group-macro MRR@10, then group-macro nDCG@10, then the earlier epoch. Default serving-model selection follows [the execution contract](10-accelerator-execution-contract.md); no test-based choice is allowed. The default matrix plans exactly three epochs per run; an interrupted or budget-stopped run remains incomplete, rather than a completed early-stopping result. Any smaller budget is declared before formal comparisons.

这些是未经验证的起始参数，不声称最优。学习率不同，因此比较的是两套实际训练方案，不是仅冻结这一因素的因果实验。完整 epoch 的模型按验证集组平均 MRR@10、再按组平均 nDCG@10、最后按较早 epoch 选择。默认模型遵循执行契约，不使用测试结果。默认每次计划完整三轮，中断或预算停止仍属未完成，不冒充已完成的早停；较小预算须在正式比较前声明。

Seeds 43 and 44 are an optional paired extension: if included, run both conditions for both seeds, expanding two runs to six. Decide this extension before formal comparisons; never add seeds to rescue an unfavorable result or report only the best seed. Report all runs and variability separately from query/evidence-group uncertainty. NQ expansion and hyperparameter grids are outside this initial budget.

种子 43、44 是可选的配对扩展：纳入时，两种条件均运行这两个种子，正式运行从两次变六次。正式比较前决定是否扩展，不为挽救差结果追加种子或仅报告最佳种子。报告全部运行，种子波动与问题／证据组不确定性分开。初始预算不包含 NQ 扩量或超参数网格搜索。

## 2. Test-review work is part of the budget / 测试审查也是预算组成

For the recommended fixed-candidate profile, before formal training an isolated review workflow judges each test query's fixed BM25 top 50 plus its known positives, preserving the complete judgment bank and first seal. Training/model selection cannot access test queries, candidates or judgments. After systems are locked, stage two verifies that the bank covers the fixed candidates and preserves the final seal; it does not reopen model-guided judging or tuning. Known positives supplement the judgment bank only, never the retrieved candidates.

推荐的固定候选方案在正式训练前，由隔离审查流程判断每条测试问题固定 BM25 前 50 条及已知正例，保留完整判断库与首次封存。训练／模型选择不可访问测试问题、候选或标注。系统锁定后，第二阶段检查判断库覆盖固定候选并保留最终封存，不重新开展模型引导的审查或调参。已知正例只补入判断库，不补入检索候选。

With `T` test queries, at most `K` candidates and `G_i` known positives for query `i`, review work is at most `sum_i(K + G_i)` before within-query deduplication. At `T=300`, `K=50`, one known positive each, this is **15,300 query–passage judgments**, not 300 decisions. Additional positives change that ceiling. Measure review throughput on development material; report agent and human review separately. Reused passages still require query-specific judgments.

设测试问题数为 `T`、候选上限为 `K`、每题已知正例数为 `G_i`，题内去重前审查量最多为 `sum_i(K + G_i)`。`T=300`、`K=50`、每题一条正例时，为 **15,300 条问题—段落判断**，不是 300 次判断；正例增加会改变上限。在开发资料上测量审查吞吐，分别记录 Agent 与人工审查；相同段落面对不同问题仍需分别判断。

## 3. Work units and accumulation / 工作单位与梯度累积

Let `P = sum_i(positive_i + negative_i)` be actual training pairs, `E` epochs, `b` microbatch pairs, `A` accumulation microbatches, `V/T` validation/test queries, and `K` candidate depth. On one device, nominal effective batch is `b*A`. Training pair passes are `E*P`; microbatches are `E*ceil(P/b)`; optimizer updates are `E*ceil(ceil(P/b)/A)` when accumulation is flushed at each epoch boundary.

令 `P = sum_i(positive_i + negative_i)` 为实际训练对数，`E` 为轮数，`b` 为单次 microbatch 对数，`A` 为累积 microbatch 数，`V/T` 为验证／测试问题数，`K` 为候选深度。单设备名义有效 batch 为 `b*A`。训练处理对数为 `E*P`，microbatch 数为 `E*ceil(P/b)`，每轮末尾提交不足累积组时，优化器更新数为 `E*ceil(ceil(P/b)/A)`。

Weight partial accumulation by actual pair count: accumulate summed per-pair losses normalized by the accumulation group's actual total, or an equivalent sample-weighted mean. Do not divide a short last group by the full planned `A`. Clip after accumulation; update optimizer/scheduler once per completed group. Accumulation preserves memory limits but does not remove forward/backward work. Precision/scaling behavior must match the installed backend; the initial recipe uses FP32. [PyTorch accumulation guidance](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html)

不足累积组按实际样本对数加权：将逐样本损失之和除以当前累积组实际总数，或使用等价的按样本数加权平均。不能把最后短组仍除以完整预定 `A`。累积后裁剪，每组仅更新一次优化器／调度器。累积控制内存，不会省掉前后向计算；精度及缩放行为以安装后端为准，初始使用 FP32。

At `P=5,000`, `E=3`, `b=4`, `A=4`, one run processes **15,000 training pairs**, **3,750 microbatches**, and **939 optimizer updates**. Each epoch ends with an effective batch of eight pairs. Define warmup updates as `ceil(0.10*planned_updates)`, hence 94 here. A different calibrated microbatch must still honor effective batch 16 and actual partial-group normalization.

`P=5,000`、`E=3`、`b=4`、`A=4` 时，单次运行处理 **15,000 对训练样本、3,750 个 microbatch、939 次优化器更新**；每轮最后有效 batch 为八对。warmup 更新数定义为 `ceil(0.10*planned_updates)`，本例为 94。校准后换 microbatch 仍须保持有效 batch 16，并按实际不足组归一化。

| Work / 工作 | One trained model / 单训练模型 | Two core models / 两个核心模型 |
| --- | ---: | ---: |
| Training: 1k queries × 5 pairs × 3 epochs / 训练 | 15,000 pair passes / 对次 | 30,000 |
| Validation: 200 queries × 50 candidates × 3 epochs / 每轮验证 | 30,000 scores / 评分 | 60,000 |
| Final test: 300 queries × 50 candidates / 最终测试 | 15,000 scores / 评分 | 30,000 |
| All neural evaluation / 全部神经评测 | 45,000 scores / 评分 | 90,000 |

Counts are maxima under the example assumptions; fewer candidates, fewer negatives or fewer completed epochs require actual counts. With all three paired seeds, trained-model counts multiply by three. BM25 retrieval is shared CPU work and is not included as neural GPU pair processing. Final scoring uses the locked systems and the same judgment bank; compute additional deployment latency benchmarks separately.

这些是示例假设下的最大工作量；候选、负例或完成轮数较少时使用实际数量。启用全部三个配对种子，训练模型工作量乘三。BM25 是可复用的 CPU 检索，不计作神经模型 GPU 样本处理；最终计分使用锁定系统和相同判断库。额外部署延迟测试另算。

## 4. Conditional time arithmetic / 条件耗时算术

For measured per-model rates `r_train` and `r_eval` in pairs/second, neural processing time is `sum_models(E*P/r_train + (E*V*K + T*K)/r_eval)`. Use actual candidate totals. State whether these rates include tokenization/loading, so overhead is not double-counted. Frozen and full training need separate measurements.

若每个模型实测训练及推理吞吐为 `r_train`、`r_eval`，单位对／秒，则神经处理时间为 `sum_models(E*P/r_train + (E*V*K + T*K)/r_eval)`，候选数使用实际总量。说明吞吐是否包含分词／加载，以免重复计入开销；冻结与全量训练分别测量。

| Hypothetical rates train/eval (pairs/s) / 假设吞吐 | 30k training / 训练 | 90k evaluation / 评测 | Two-model total / 双模型合计 |
| --- | ---: | ---: | ---: |
| 5 / 20 | 100 min / 分钟 | 75 min / 分钟 | 175 min / 分钟 |
| 25 / 100 | 20 min / 分钟 | 15 min / 分钟 | 35 min / 分钟 |
| 100 / 400 | 5 min / 分钟 | 3.75 min / 分钟 | 8.75 min / 分钟 |

These rates are illustrative assumptions applied equally to both models solely for arithmetic; they describe no measured Mac or NVIDIA GPU. Actual time additionally includes installation/downloads, parsing, review, retrieval, checkpoint/export, failures and transfers. Machine processing time is not calendar completion time or rental billing duration. A review workflow at `r_review` judgments/hour adds approximately `N_review/r_review` review hours, with its agent/human source disclosed.

这些吞吐仅为方便算术而同时套用两模型的假设，不对应任何实测 Mac 或 NVIDIA GPU。实际时间还包括安装下载、解析、审查、检索、保存导出、失败重跑与传输。机器处理时间不等于项目日历时间或租赁计费时长。若审查吞吐为每小时 `r_review` 条判断，则约需 `N_review/r_review` 审查小时，并披露 Agent／人工来源。

## 5. Storage and memory arithmetic / 存储与内存算术

For approximately `N=33,000,000` parameters in FP32, weights are about `4N=132 MB`. Full training's weights, gradients and two Adam state tensors total about `16N=528 MB` before activations, operator workspaces, allocator overhead, data and the system. A resumable weights-plus-Adam checkpoint is approximately `12N=396 MB`; these decimal-MB estimates exclude serialization and metadata overhead. They are not peak memory measurements or device-fit guarantees.

约 `N=33,000,000` 参数、FP32 时，权重约 `4N=132 MB`。全量训练的权重、梯度及两份 Adam 状态约 `16N=528 MB`，尚未包含激活、算子临时区、分配器、数据及系统。权重加 Adam 的可恢复断点约 `12N=396 MB`。这里使用十进制 MB，不含序列化与元数据开销，不是峰值内存实测或设备可容纳保证。

Bound retention to epoch zero plus up to three completed-epoch checkpoints: at most four resumable files per run, plus one 132 MB inference export. `best` and `latest` are references, not duplicate weight files. Conservatively using 396 MB even for epoch zero and frozen-head checkpoints gives `4*396+132 = 1,716 MB/run`: **3.432 GB for two runs**, or **10.296 GB for six runs**. Actual frozen-head optimizer state and epoch-zero state are smaller. Temporary writes, original model cache, data/indexes, environment and independent backups add space.

保留上限为 epoch zero 加最多三个完整 epoch：每次运行最多四份可恢复文件，另加约 132 MB 推理导出。`best`、`latest` 为引用，不重复复制权重。即便对 epoch zero 和冻结模型也保守套用 396 MB，则每次 `4*396+132 = 1,716 MB`：**两次约 3.432 GB，六次约 10.296 GB**。实际冻结优化器及初始状态更小。临时写入、原始模型缓存、数据索引、环境和独立备份额外占空间。

Set disk headroom from measured installed/cache/data sizes, retained checkpoints, the largest atomic-write/export temporary copy and backup layout; do not equate these model-only estimates with a complete disk requirement. Keep heavy assets under the verified external or permitted target-host runtime root. Apple 24 GB unified memory is shared with the system; CUDA VRAM and host RAM are separate budgets.

根据实测环境／缓存／数据体积、保留断点、最大原子写入／导出临时副本和备份布局确定磁盘余量，不能把模型文件估算当作完整磁盘需求。大体积产物位于经验证的外置或目标主机允许目录。Apple 24 GB 统一内存与系统共享，CUDA 显存和主机 RAM 则分别预算。

## 6. Calibration and readiness / 校准与就绪条件

Before formal runs, use the bounded calibration protocol in [document 10](10-accelerator-execution-contract.md): three safety-probe optimizer updates per batch candidate; on the selected setting, five warmup plus 50 measured optimizer updates for each training recipe, and five warmup plus 50 inference batches. Optimizer updates and microbatches are different units; record both. Check typical/near-limit lengths, device placement, finite updates, memory and save/reload. Include all calibration time in the budget and synchronize timed regions. [MPS synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html) · [CUDA synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html)

正式运行前遵守第 10 号文档的有界校准：每候选微批次三次安全探测更新；选定后，各训练配方五次预热、50 次实测优化器更新，推理五次预热、50 个实测 batch。优化器更新与 microbatch 是不同单位，均须记录。检查典型近上限长度、设备、有限更新、内存及保存重载；计时前后同步设备，全部校准耗时计入预算。

Then freeze the backend/precision, encoding limits, microbatch/accumulation, software/model revisions, paired-seed plan, actual counts and maximum run budget. Discard diagnostic training weights as a formal starting point. Record measured epoch + validation + save time and backup reserve; do not start an epoch that cannot finish within the execution window. Hardware failures may change an explicitly versioned configuration, never the sealed test membership or the definition of success.

随后冻结后端／精度、编码长度、microbatch／累积、软件模型版本、配对种子方案、实际数量与最大运行预算。诊断训练权重不作为正式起点。记录实测完整 epoch＋验证＋保存时间及备份余量，剩余执行窗口不足时不启动下一轮。硬件问题可触发明确版本化的配置变更，不能改封存测试成员或成功定义。

Quality targets are decision thresholds declared before formal training and applied during validation, not forecasts. No numerical MRR range or guaranteed uplift is justified before data and experiments exist. The deliverable is a reproducible comparison with limitations and a working baseline; a reranker becomes default only under the predeclared selection rule. Keep engineering results, learning evidence and actual user studies distinct.

质量目标是在正式训练前声明、在验证阶段应用的决策门槛，不是预测。数据与实验完成前，没有依据给出 MRR 范围或保证提升。交付目标是可复现对照、局限说明和可用基线；重排器仅按预定规则成为默认。工程结果、学习证据与真实用户研究分别记录。
