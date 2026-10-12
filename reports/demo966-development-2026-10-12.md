# Fixed-model development comparison / 固定模型开发对比

The original R5, 483-update and 966-update endpoints were compared on CPU using the same frozen questions, natural BM25 top-50 pools and graded labels. Each endpoint in this comparison uses training seed42. The new 966 endpoint is not the original three-seed R5 family.

原 R5、483 次更新及 966 次更新模型在 CPU 上使用相同冻结问题、自然 BM25 前 50 候选及分级标签对比；本次每个模型都使用训练种子 42。新 966 模型不是原 R5 三种子系列。

There are 47 new synthetic development questions and 34 retained questions, totaling 81. All 4,050 natural candidate pairs and ten outside-pool source answers have frozen grades. The original 1,704 grades remain unchanged; new review comprises A=2,356, B=500, C=24, with no unresolved items. The original 202-question benchmark remains separate. [Aggregate evidence, all five cohorts and all comparisons](evidence/demo966-cpu-20261012/development-comparison-summary.json).

新增 47 个合成开发问题，保留 34 题，共 81 题。全部 4,050 对自然候选及十个池外来源答案均有冻结分级标签；原 1,704 条标签保持不变，新增审核为初审 2,356 对、复审 500 对、裁决 24 对，无未决。原 202 题基准单独保留。上述摘要提供五个分组及全部对照结果。

| Fixed seed42 system / 固定种子42系统 | New47 useful MRR@10 / 新47题有用MRR | New47 nDCG@10 | Expanded81 useful MRR@10 / 扩充81题有用MRR |
| --- | ---: | ---: | ---: |
| BM25 | 0.7697 | 0.6124 | 0.7725 |
| Original R5, raw / 原R5直接排序 | 0.7730 | 0.5965 | 0.7307 |
| Original R5, fused / 原R5融合 | 0.7975 | 0.6291 | 0.7610 |
| 483, raw / 483直接排序 | 0.8447 | 0.6904 | 0.8409 |
| 483, fused / 483融合 | 0.8304 | 0.6701 | 0.8249 |
| 966, raw / 966直接排序 | 0.8258 | 0.6854 | 0.8014 |
| 966, fused / 966融合 | 0.8608 | 0.6866 | 0.8336 |

On new47, 966 plus fixed fusion improves useful MRR by **+0.091109**, with exploratory paired group-bootstrap 95% interval **[+0.018414, +0.170643]**. Its nDCG difference is **+0.074163**, interval **[+0.024986, +0.125163]**. This meets the existing development point targets (MRR difference at least +0.02 and no nDCG decline); the MRR interval's lower bound is not at least +0.02.

在新 47 题上，966 加固定融合的有用 MRR 提升 **+0.091109**，探索性配对分组 Bootstrap 的 95% 区间为 **[+0.018414, +0.170643]**；nDCG 提升 **+0.074163**，区间为 **[+0.024986, +0.125163]**。满足原开发集点值门槛（MRR 提升至少 +0.02、nDCG 不下降），但 MRR 区间下界没有达到 +0.02。

The 966 fused versus 483 fused MRR interval crosses zero; 483 raw also has the highest new47 nDCG and expanded81 useful MRR in the table. The owner chose 966 plus fixed fusion for the demo after reviewing the comparisons, conditional on separate CPU service acceptance. This is not a claim that 966 wins every metric or that more training always helps.

966 融合相对 483 融合的 MRR 差值区间跨过零；表中 483 直接排序的新 47 题 nDCG 及扩充 81 题有用 MRR 也最高。所有者阅览对比后选择 966 加固定融合作为演示，并要求另行通过 CPU 服务验收。这不表示 966 在每个指标都最好，也不表示训练越多越好。

These questions were authored after earlier development outcomes and use the existing indexed corpus, with source groups excluded from known training groups. Labels come from agents. The 10,000-resample intervals are conditional on the frozen labels, exploratory and unadjusted for multiple comparisons. They describe variation across question groups, not across training seeds. Historical test/holdout results, old labels and prediction files were preserved; no test or holdout was used for this comparison's tuning. [Local demo operation](../docs/DEMO966_CPU.md).

这些问题在已有开发结果后生成，使用既有索引语料，来源组排除了已知训练组；标签来自 Agent。10,000 次重采样区间以冻结标签为条件，属于探索性结果，未作多重比较校正；它们衡量问题组变动，不衡量训练种子变动。历史测试／留出结果、旧标签及预测文件保持原样，本次对比不据测试或留出集调参。演示操作见上述文档。
