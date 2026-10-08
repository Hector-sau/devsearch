# Optional grouped loss comparison / 可选分组损失对照

This is an opt-in implementation contract for one matched comparison. The implementation has offline CPU fixture tests and exact local Round 6 input reconstruction; actual-host model/checkpoint verification and real-model CUDA smoke remain separate gates. It does not authorize a GPU restart or declare a quality improvement.

本文是一次匹配对照的可选实现契约。实现具备离线 CPU 样例测试，并已精确重建本地第 6 轮输入；目标机模型／断点实体核验及真实模型 CUDA 短测仍须另行完成。本文不代表启动 GPU 或已改善排序质量。

## Question and scope / 问题与范围

Does a pairwise logistic objective improve on weighted BCE when query groups, examples, initialization, dropout exposures, updates and checkpoint selection match? Keep the reviewed R6 data and MiniLM fixed. Do not use the previous flat-batch R6 run as the matched control: its batch order and update count differ. Both arms start fresh.

在问题分组、样本、初始化、dropout 曝光、更新次数及模型选择一致时，成对 logistic 是否优于加权 BCE？固定已审核 R6 数据及 MiniLM。原 R6 平铺批次的顺序和更新次数不同，不能直接充当匹配对照；两组均重新初始化。

Maximum proposal: two arms × seeds 42/43/44 × three epochs. FP32, TF32 disabled, head seed 1234, dropout 0.1, LR 5e-5, AdamW weight decay 0.01, betas 0.9/0.999, epsilon 1e-8, clip norm 1, warmup 0.1, query/pair token limits 64/256, validation batch 8. These campaign values must be frozen by the runner; the reusable trainer also permits diagnostic fixtures with other learning rates and token lengths.

方案上限：两组 × 种子 42/43/44 × 三轮。FP32、关闭 TF32、head seed 1234、dropout 0.1、学习率 5e-5、AdamW 权重衰减 0.01、betas 0.9/0.999、epsilon 1e-8、梯度裁剪 1、warmup 0.1、问题／联合长度 64/256、验证批次 8。正式实验执行器必须冻结这些值；复用训练器允许诊断样例使用不同学习率及长度。

## Inputs / 输入

Build groups from canonical `PairRecord` identities before converting to text-only tensors. Every supplied query must be accepted and train-owned, with exactly one confirmed positive and four distinct negatives: three `bm25`, one `random`. Missing pairs are an error, never a reason to drop a query silently. All passages need train ownership derived from the verified merged-group assignment. Reject held-out ownership, duplicate pair/chunk IDs, invalid labels, conflicting qrels, positive-as-negative conflicts and an altered mix. Exclude a negative whenever its verified merged group contains any relevance-1 training qrel, including a not-yet-confirmed positive, matching the miner's exclusion rule. Bind those merged group IDs into checkpoint identity. Check containment of the pair manifest and all three child files, including symlink resolution. These structural checks do not certify absence of semantic leakage.

先保留规范 `PairRecord` 编号构建分组，再转为张量。每个问题必须已接受且属于训练集，包含一个确认正例及四个不同负例（三个 `bm25`、一个 `random`）。样本不足明确报错，不静默删题。段落的训练归属由已校验的合并证据组划分推导。拒绝跨集合归属、重复编号、错误标签、冲突 qrels、正例充当负例及比例改变。负例还须排除所有 relevance=1 训练 qrel 所属的合并证据组（包括尚未确认的正例），与采样规则一致，并在断点身份中绑定这些组编号。清单及三个子文件均检查路径归属和符号链接。这些结构检查不能证明不存在语义泄漏。

`load_training_inputs(..., grouped=True)` verifies the prepared manifest and its referenced artifacts, then retains pair IDs, query IDs, text, labels, evidence groups, ownership and source hashes. The checkpoint binds these in `grouped_data`; source hashes include prepared, pairs, queries, qrels, split and assignment files.

输入加载器先验证准备清单及引用文件，再保留样本／问题编号、原文、标签、证据组、归属和源哈希。断点通过 `grouped_data` 绑定这些内容，覆盖准备清单、训练对、问题、qrels、划分及 assignment 文件。

R6 pairs are a separate immutable artifact, not the original prepared directory's pairs. The standalone training TOML may opt into a replacement only with **both** fields below. The loader verifies its exact SHA256, file hashes, corpus ID, index ID and original train query/qrel hashes; the manifest must remain inside the runtime root. It neither edits nor re-seals the original prepared bundle. Overrides are refused for legacy flat BCE.

R6 训练对是独立的不可变产物，不是原准备目录中的训练对。独立训练配置必须同时指定下列路径和哈希才可选择它。加载器校验精确哈希、文件哈希、语料编号、索引编号及原训练问题／qrels 哈希；清单必须位于运行根目录内。不会修改或重新封存原数据包；平铺 BCE 拒绝这一覆盖参数。

```toml
[data]
prepared_manifest = "../bundle/prepared/prepared-data-manifest.json"
pairs_manifest = "../experiments/round6-reviewed-negatives-20261008/pairs/pairs-manifest.json"
pairs_manifest_sha256 = "f9a091365f8cba75cfc8ee5e2973049692941a3114c19a0ed55f73106d773bcb"
```

Paths above illustrate the config contract, not a verified host layout. Resolve them against the config directory. The actual R6 manifest and replacement-review receipt must be verified before creating the six runtime configs; never invent replacement data to make this hash pass.

以上路径仅展示配置契约，并非已验证的主机目录；相对配置文件目录解析。生成六个运行配置前，必须验证真实 R6 清单及替换审核回执，不能重造数据绕过该哈希。

## Objectives and updates / 损失与更新

Use `[loss] kind = "group_bce_weight4"` or `"pairwise_logistic_scale_4_over_5"`, with explicit `pos_weight = 4` for both. Omitted `kind` retains `flat_bce` and the previous config serialization/hash. Grouped jobs require `mode = "full"`, `epochs = 3`, `microbatch = 5`, `effective_batch = 20`, and `diagnostic = false`.

通过上述两个 `kind` 选择分组损失，两组均明确设置 `pos_weight = 4`。不填 `kind` 时维持原 `flat_bce` 及旧配置序列化／哈希。分组任务要求全量微调、三轮、microbatch 5、effective batch 20，且不开启允许额外轮次的 diagnostic 模式。

For a positive logit `p`, four negatives `n`, and `sp(x) = log(1 + exp(x))`, implemented with stable `softplus`:

设正例 logit 为 `p`、四个负例为 `n`，使用数值稳定的 `softplus`：

```text
group_bce_weight4                 = [4 sp(-p) + sum sp(n)] / 5
pairwise_logistic_scale_4_over_5   = sum sp(n - p) / 5
```

Both have zero-logit gradient `[-0.4, 0.1, 0.1, 0.1, 0.1]`; this matches the initial gradient scale, not the whole objectives. Each query has one five-record forward (positive first, negatives in source order); the same positive graph is reused. Order queries by SHA256 of `group-order-v1|seed|zero_based_epoch|query_id`, with query ID as tie-breaker. Accumulate four complete queries, divide summed query losses by the **actual query count**, then clip and update once. The 515-query input has 129 updates/epoch, 387 total, 39 warmup updates; its final update divides by three, not four or fifteen.

两者在零 logit 下梯度均为 `[-0.4, 0.1, 0.1, 0.1, 0.1]`；只匹配初始梯度尺度，不代表目标等价。每题一次前向计算五条记录，正例在前、负例保持源顺序，共用该正例计算图。按上述哈希排序，编号打破并列。累积四题后除以**实际题数**，再裁剪和更新。515 题每轮更新 129 次，共 387 次、warmup 39 次；末次更新除以三。

## Selection and recovery / 选择与恢复

Epoch metrics are monitoring only. The primary checkpoint is always epoch three; `best` is absent until then. Epoch-zero and complete-epoch snapshots retain optimizer, scheduler, RNG and counters. A partial run resumes only from a complete epoch. Checkpoint identity additionally binds objective/config, grouping, order, reduction and fixed-selection versions. Changed input hashes or objective refuse resume. Legacy flat BCE still selects by its existing validation rule.

每轮指标仅用于观察，主要断点固定为第三轮，此前没有 `best`。初始及完整轮断点保留优化器、调度器、随机状态与计数，部分运行只从完整轮恢复。身份额外绑定目标／配置、分组、顺序、归一化及固定选择版本。输入哈希或损失改变则拒绝续跑；原平铺 BCE 继续原验证选择规则。

## Evaluation and launch gates / 评估与启动条件

Finish A/B independent validation review and adjudication; keep v1 immutable. Full v2 uses the same 202 adaptively reused validation questions, not a fresh holdout. A future model's top-ten results require a blinded shared-pool extension if any are unjudged; reevaluate all compared systems on the same version. Human confirmation is optional under the recorded addendum and remains pending. All current labels are agent labels.

完成 A/B 独立验证审核及裁决，保留 v1 不变。全量 v2 使用已反复分析的同一批 202 题，不能当作新留出集。新模型前十若出现未标注候选，需要扩展共享盲审池，并让所有对照在同一版本重评。按已记录补充协议，人工确认可选且仍待完成，现有标注均为 Agent 标注。

Predeclared exploratory continuation rule: paired seed-mean group-macro MRR gain at least +0.005 over the **new matched BCE arm**, nonnegative nDCG change, positive MRR gain in at least two of three seeds, and a paired evidence-group bootstrap 95% MRR interval with lower bound above zero (10,000 resamples, seed 20261008). Report adaptive-validation uncertainty. Otherwise stop this loss direction without a sweep. BM25 remains the formal default and the formal +0.02 gate does not change. No test-set scoring or tuning is authorized here.

预定探索性继续条件：相对**新匹配 BCE 组**的种子均值组宏 MRR 至少 +0.005、nDCG 不降、至少两种子改善，按证据组配对 bootstrap 的 95% MRR 区间下界大于零（10,000 次、种子 20261008）。披露反复使用验证集的局限，否则停止此损失方向，不扫参。正式默认仍为 BM25，+0.02 门槛不变，本次不授权测试集评分或调参。

Before starting the six-run campaign, freeze source/configs, verify real R6 identities and the existing CUDA lock, and pass a bounded real-model smoke for both objectives. CPU preparation permits starting that hardware smoke; it does not claim the GPU smoke has already passed. Full-v2 labeling is needed for full-v2 claims, but is not a technical prerequisite to a preregistered frozen-v1 exploratory comparison. The proposed GPU-active window is 45 minutes plus two minutes for termination; it is an execution bound, not measured runtime or a provider spending cap. Preserve all 60 existing server checkpoints. No large transfers, new rental or paid API calls are part of CPU preparation.

启动六次完整训练前冻结源码及配置，核实真实 R6 身份和现有 CUDA 锁，并通过两目标的有限真实模型短测。CPU 准备允许开始该硬件短测，不能据此声称短测已通过。完整 v2 标注是完整 v2 成果声明的前提，并非预注册、冻结 v1 探索对照的技术前提。建议 GPU 活跃窗口 45 分钟，另留两分钟终止；这是执行边界，不是实测耗时或平台费用上限。保留服务器既有 60 个断点，CPU 准备不包含大文件回传、新租机或付费 API。

## Bounded operator / 限时执行器

`scripts/grouped_experiment.py plan/check/status` are CPU operations. `plan` writes six immutable configurations and binds exact data, source, tests, this specification, CUDA lock, CUDA profile bytes and its parsed settings. Receipt-only local checks must explicitly use `--defer-host-assets --defer-host-checkpoints`; their JSON marks model/history byte verification and actual-host readiness false. A plan is tied to its absolute runtime paths: create a separate target-host plan with verified existing inputs, never edit a local plan to relocate it.

上述三个命令只使用 CPU。计划绑定精确数据、源码、测试、本规格、CUDA 锁及 CUDA 配置文件和解析结果，并生成六份不可变配置。本地仅凭回执检查必须明确使用两个 defer 参数，输出会标记模型／历史文件实体与目标机就绪状态为 false。计划绑定绝对运行路径，目标机应使用核实过的既有输入另建计划，不能编辑本地计划搬迁。

`run --authorize-cuda` requires the existing exact Linux/Python 3.12 lock environment, all model/tokenizer bytes, hashes of all 60 retained checkpoints, at least 16 GiB free disk and exclusive access to the existing single GPU. It never starts or rents a machine. It checks actual FP32/TF32 settings, supervises one child process group at a time, and preserves the original 45-minute deadline across interruption/resume. Termination also kills descendants if their leader has already exited; inability to confirm cleanup is a failure.

运行命令要求现有 Linux／Python 3.12 精确锁环境、模型及分词器实体、60 份历史断点哈希、至少 16 GiB 空间及单 GPU 独占访问，不负责开机或租机。它核验实际精度设置、串行监管子进程组，中断恢复不能重置原 45 分钟时限。即使领头进程已退出，仍清理其后代；无法确认清理则报错。

Before the six full-data runs, both losses execute a **diagnostic-only 32-group recovery probe** using the production training loop: three continuous epochs versus one epoch, saved production checkpoint, fresh model, perturbed RNG and two resumed epochs. Require matching model/optimizer/scheduler/RNG/counters/LR trace under the frozen parity tolerances. Keep all probe checkpoints separate. Time measured training, loading, saving and full validation scoring; a conservative estimate must fit the remaining deadline. Formal runs still initialize fresh pretrained weights, use all 515 groups and perform exactly 387 updates each. Probe results never become formal candidate models.

六次全量训练前，两种损失均通过生产训练循环执行**仅诊断的 32 组恢复短测**：连续三轮，与一轮后保存生产断点、重新创建模型、扰动随机状态、再恢复两轮进行比较。模型、优化器、调度器、随机状态、计数及学习率轨迹须满足冻结容差，短测断点单独保留。根据训练、加载、存盘和完整验证评分实测做保守估算，剩余时限容得下才开始全量运行。正式运行仍各自重新加载预训练权重、使用全部 515 组、更新恰好 387 次；短测模型不充当正式候选。
