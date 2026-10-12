# Final 966-update development replication — 2026-10-13 / 最终 966 次更新开发复现

The owner selected **966 updates, two epochs and fixed fusion** as the final project configuration after three fresh training runs and one controlled longer-duration comparison. The existing verified seed 42 demo remains the serving endpoint; its weights were not silently replaced by a fresh replication checkpoint. BM25 remains the model-free control and complete failure fallback. No further GPU training is required for this delivery.

所有者在三次全新训练及一次受控延长训练对比后，选择 **966 次更新、两轮训练及固定融合**作为最终项目配置。服务继续使用已有且通过验收的 seed 42 演示模型，不将其权重悄然替换为新复现断点；BM25 保留为无模型对照及完整故障降级方案。本次交付无需继续 GPU 训练。

## Fixed experiment / 固定实验

Each fresh run uses the same 720 training questions, 3,600 unique pairs, weighted BCE (`pos_weight=4`), AdamW at `5e-5`, microbatch 4, effective batch 16 and 15,450 pair exposures. Seeds 42/43/44 vary training order and dropout; scoring-head initialization stays at 1234. All runs finish exactly 966 updates and select fixed epoch 2, independently of the best-checkpoint pointer. A single RTX 4090 24 GB runs them sequentially with FP32 and TF32 disabled; the audited supervisor duration is 335.667 seconds.

每次全新运行使用相同的 720 个训练问题、3,600 对唯一样本、加权 BCE、AdamW、上述批次设置及 15,450 次样本曝光。种子42／43／44改变训练顺序和 dropout，评分头初始化保持1234；均完成966次更新并固定选第二轮，不使用最佳断点指针选择。单张 RTX 4090 24 GB 顺序执行，使用 FP32并关闭TF32；已审核的监督执行耗时为335.667秒。

CPU evaluation uses natural BM25 top 50, pair/query limits 256/64 tokens, inference batch 8 and unchanged fusion `alpha=0.5`, `rank_constant=60`. The graded pool has 4,060 labels: 4,050 natural candidates and 10 outside-pool anchors. Grades 0/1/2 denote not useful, useful partial material and complete material. Useful MRR@10 finds the first grade≥1; graded nDCG@10 uses gains 0/1/3. Outside anchors stay in the ideal-ranking denominator; no positive is injected into retrieval.

CPU评测使用自然 BM25前50、联合／问题长度256／64 token、推理批次8及固定融合。分级池共有4,060项标签，包括4,050个自然候选和10个池外锚点；0／1／2分别表示无用、有用但不完整、完整资料。Useful MRR@10计首个等级≥1结果，graded nDCG@10收益为0／1／3；池外锚点保留理想排序分母，召回不注入正例。

## Three fresh seeds / 三个全新种子

| Development scope / 开发范围 | Metric / 指标 | BM25 | Seed 42 | Seed 43 | Seed 44 | Mean / 均值 | Mean difference / 均值差 | Conditional paired 95% interval / 条件配对95%区间 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh 47 primary / 新47题主范围 | `useful_mrr@10` | 0.769706 | 0.860816 | 0.865248 | 0.856738 | **0.860934** | +0.091228 | [+0.029905, +0.162656] |
| Fresh 47 primary / 新47题主范围 | `graded_ndcg@10` | 0.612417 | 0.686580 | 0.689657 | 0.696040 | **0.690759** | +0.078342 | [+0.038323, +0.123643] |
| All 81 secondary / 全81题次范围 | `useful_mrr@10` | 0.772546 | 0.833642 | 0.853395 | 0.837380 | **0.841472** | +0.068927 | [+0.021788, +0.119818] |
| All 81 secondary / 全81题次范围 | `graded_ndcg@10` | 0.576686 | 0.660440 | 0.661872 | 0.672854 | **0.665056** | +0.088370 | [+0.055972, +0.122533] |
| Retained 34 secondary / 原34题次范围 | `useful_mrr@10` | 0.776471 | 0.796078 | 0.837010 | 0.810621 | **0.814570** | +0.038099 | [-0.029902, +0.103649] |
| Retained 34 secondary / 原34题次范围 | `graded_ndcg@10` | 0.527293 | 0.624306 | 0.623463 | 0.640804 | **0.629524** | +0.102232 | [+0.052986, +0.155234] |

The fixed fused policy passes the unchanged fresh 47 development rule: mean useful-MRR improvement ≥0.02, mean graded-nDCG change ≥0, and positive useful-MRR differences in all three seeds. The fresh 47 useful-MRR sample standard deviation across seeds is 0.004257. The mean is a descriptive model-family statistic, not an ensemble or the score of the deployed checkpoint. Recall@50 is unchanged because reranking preserves the same candidate pool. Retained 34 useful-MRR uncertainty spans zero; raw reranking on that slice has mean difference −0.006742. No claim is made that every policy or subset improves.

固定融合通过原有新47题开发规则：平均 useful-MRR提升≥0.02、平均 graded-nDCG不下降、三个种子的 useful-MRR差值均为正。该范围种子间 useful-MRR样本标准差为0.004257；均值描述模型系列，不代表集成或已部署断点。候选池不变，因此Recall@50不变；原34题useful-MRR区间跨零，纯重排在该范围平均差值为−0.006742，不能宣称所有模式及子集均提升。

## Original 202 questions / 原202题

The original binary benchmark is preserved separately. Known-label fused MRR@10 for seeds 42/43/44 is 0.943089/0.933758/0.947654, mean 0.941500 versus BM25 0.901416 (difference +0.040084); pooled binary nDCG mean 0.866777 versus 0.827758. These new 966 values differ from original R5’s mean 0.941062 and must not be merged with it.

原二元基准单独保留。新966三种子的已知标签融合MRR@10为0.943089／0.933758／0.947654，均值0.941500，BM25为0.901416，差值+0.040084；池化二元nDCG均值0.866777，BM25为0.827758。新966结果不同于原R5均值0.941062，不能混写。

Only 4,570 of 10,100 natural pairs are judged (45.2475%), plus one judged outside-pool pair; 5,530 remain `UNKNOWN`. The known-label projection counts confirmed positives and retains every query/rank denominator; it does not turn unknown records into confirmed negatives. Exact missing-label bounds are held in the private report. This retrospective partial-label regression check neither supplies full acceptance nor replaces the graded primary scope.

自然池10,100对中仅4,570对已判定，覆盖45.2475%，另有1项池外标签；5,530项仍为`UNKNOWN`。已知标签投影计算确认正例并保留全部问题及排序分母，不把未知项当成确认负例；私有报告保留精确缺标界限。该回顾性检查不能提供完整验收，也不替代分级主范围。

## Controlled four-epoch trial / 受控四轮试验

The additional trial starts fresh with seed 42, 4 epochs and 1,932 updates, holding data/model/loss/fusion fixed. Its full learning-rate horizon changes warmup from 97 to 194 updates; this is not a resumed 966-update run or a claim of an identical first 966 updates. Fixed epoch 4 is compared with the matched fresh 966 seed 42 endpoint, not with the best seed or the three-seed average.

额外试验从种子42重新开始，运行4轮、1,932次更新，固定数据／模型／损失／融合。完整学习率周期将预热从97次改为194次，因此不是从966断点续训，也不声称前966次相同；固定第四轮与匹配的新966 seed 42比较，不与最佳种子或三种子均值比较。

| Scope / 范围 | Useful-MRR difference / 差值 | Conditional 95% interval / 条件区间 | Graded-nDCG difference / 差值 | Conditional 95% interval / 条件区间 |
| --- | ---: | --- | ---: | --- |
| Fresh 47 / 新47题 | +0.000887 | [-0.031915, +0.039894] | -0.001980 | [-0.034722, +0.032820] |
| All 81 / 全81题 | +0.004218 | [-0.021399, +0.031893] | -0.003167 | [-0.026068, +0.021189] |

The predeclared additional-duration rule requires positive useful-MRR and nonnegative graded-nDCG change against the matched 966 endpoint. It fails: nDCG decreases, and both primary intervals span zero. On the partial original 202 check, fused known-label MRR also moves from 0.943089 to 0.940270. The owner retains 966; no longer-duration promotion or further training follows automatically.

预先声明的延长训练规则要求相对匹配966模型useful-MRR为正且graded-nDCG不下降；本次未通过，nDCG下降且两项主区间均跨零。原202题部分标签检查中，融合已知标签MRR也从0.943089降至0.940270；所有者保留966，不自动晋升四轮模型或追加训练。

## Evidence and release boundaries / 证据及发布范围

The 47 newer questions were authored after earlier model outcomes; all 81 are source-derived synthetic development questions with frozen agent relevance labels. Known training groups are excluded as question sources, while the retrieval corpus is shared across document splits. These are development replication results, not a preregistered independent final test, unseen-document guarantee or human-user benefit. The historical 197- and 59-question negative findings remain in [RESULTS](RESULTS.md). No test/holdout questions or labels guided this replication or duration choice.

47个较新问题是在早期模型结果之后编写，全部81题属于来源文档合成开发题，使用冻结Agent标签。问题来源排除已知训练组，但检索语料由文档划分共用；结论为开发复现，不等于预注册独立最终测试、未见文档保证或真人使用收益。历史197题与59题负面结果继续保留；本轮复现及训练时长选择未使用测试／留出题或标签。

Intervals use 10,000 paired evidence-group bootstrap draws with seed 20261005, conditional on the realized models and frozen labels. They are not training-seed confidence intervals, annotation-accuracy estimates or multiplicity-adjusted claims. Three seeds provide descriptive variation only.

区间使用种子20261005的10,000次配对证据组Bootstrap，并以已运行模型及冻结标签为条件；不代表训练种子置信区间、标注准确率或多重比较调整。三个种子只提供描述性波动。

[Aggregate summary](evidence/final966-20261013/summary.json) and [hash commitments](evidence/final966-20261013/evidence-manifest.json) publish selected verified statistics. Exact new GPU reproduction additionally requires private frozen execution wrappers, inputs and checkpoint evidence; those wrappers are not asserted to be included in this public clone. The native training source snapshot SHA-256 is `5543091183ebbf59681662ea8e4b496a10da116524c2e958372d883c60297410`, with wrapper/data identities bound by the experiment plans.

公开概要及哈希承诺仅发布核验后的指定统计。精确重现本轮GPU实验还需私有冻结执行包装器、输入及断点证据，不声称公开克隆包含这些包装器；原生训练源码绑定上述源码快照摘要，包装器及数据由实验方案关联。

The existing CPU demo passes 80 checks plus 36 independent checks on installed `1.1.1.dev0`; warm sequential real-TCP client p95 is 635.736 ms for 150 requests on Apple M4. This historical receipt remains unchanged and is not relabeled as `1.2.0` package acceptance. Exact-source CI, isolated installation and public download verification for the proposed release must be completed separately. Complete native checkpoint archival is tracked privately and is not implied by this public result.

已有CPU演示在安装版`1.1.1.dev0`上通过80项及36项独立检查，Apple M4的150次预热串行真实TCP请求客户端p95为635.736毫秒。历史回执保持不变，不改称`1.2.0`安装包验收；拟发布版仍须单独完成准确源码CI、隔离安装及公开下载核验，原生断点完整归档单独私下跟踪。
