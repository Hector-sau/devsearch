# Frozen R5 holdout and CPU acceptance / 冻结 R5 留出集与 CPU 验收

**R5 did not pass the independent quality gates. Keep BM25 as the formal default and the real R5 + fusion application as an experimental candidate.** Inference, reload, API/UI integration and CPU timing are verified; they do not establish a ranking improvement. No new training or GPU startup occurred.

**R5 未通过独立效果门槛。正式默认保留 BM25，真实 R5 加融合应用保留为实验候选。**推理、重载、接口／界面及 CPU 耗时已核验，不能据此宣称排序提升。本次没有新增训练或启动 GPU。

## Independent results / 独立结果

Both systems use all **59 questions / 59 evidence groups**, the same natural BM25 top50 candidates and the same sealed multi-positive labels. This is a single fixed seed42/epoch3 model, not the earlier three-seed validation mean. [Compact evidence](evidence/r5-independent-20261010/summary.json).

两系统共用全部 **59 题／59 个证据组**、自然 BM25 前 50 候选及同一封存多正例标签。本次仅评测固定的种子 42／第三轮模型，不是此前三种子验证均值。

| Metric / 指标 | BM25 | R5 + fixed fusion / R5 加固定融合 | Difference / 差值 |
| --- | ---: | ---: | ---: |
| MRR@10 | 0.581302 | 0.573278 | −0.008024 |
| Pooled nDCG@10 / 汇总标注 nDCG@10 | 0.592521 | 0.526748 | −0.065773 |
| Known-positive Recall@50 / 已知正例召回 | 0.943503 | 0.943503 | 0 |

Paired group bootstrap: 10,000 repetitions, seed20261005, 95% intervals. MRR difference interval **[−0.072713, +0.057439]** includes zero. nDCG difference interval **[−0.125972, −0.005525]** is negative on this benchmark. The frozen **MRR gain ≥0.02** and **nonnegative nDCG change** gates both fail. Equal Recall@50 follows from permuting the same candidate sets; it is not evidence that reranking improves retrieval.

按证据组配对 Bootstrap，重复 10,000 次、种子 20261005、95% 区间。MRR 差值区间包含零；本基准 nDCG 差值区间为负。预先固定的 **MRR 至少增加 0.02** 和 **nDCG 不下降** 两项门槛均未通过。召回相同源于两系统只排列同一批候选，不能证明重排改善检索。

The earlier validation-v4 improvement remains historical adaptive-development evidence. Its questions and labels differ from this holdout, so absolute scores are not directly comparable. The new result does not support claiming that the validation gain generalized. Per-type scores are descriptive, with only3–16 questions per type; they do not justify selecting a favorable subset or changing serving rules.

此前验证 v4 的提升继续保留为反复使用的开发证据；问题及标签与本次不同，绝对分数不能直接比较。新结果不支持声称验证提升已泛化。各类型仅有 3～16 题，分项成绩只能描述，不能据此选择有利子集或改变服务规则。

## Frozen procedure and review / 冻结流程与审查

Model and fusion were fixed before question generation: R5 seed42/epoch3, checkpoint SHA-256 `74b7fd525d75a0aab2a126e49eb759704cd0cf51a2c955122db938d4cd9f2a3c`, fusion alpha0.5/k60, candidate50, batch8, pair/query budgets256/64, no truncation. Sixty unused eligible groups were sampled; one answerability exclusion was recorded before retrieval, with no replacement. All59 retained questions have a confirmed positive after final review, including retrieval misses in the scoring denominator.

问题生成前已固定上述模型、融合及长度策略。抽样 60 个未使用的合格证据组，检索前记录一项不可回答排除且不补选。保留的 59 题最终均有确认正例，检索漏失仍计入分母。

The blinded union of both top10 lists and independently checked source passages contains **789 pairs**. Reviewers saw question/passage cards without system, rank, score, source-positive identity or prior answers. Full A covered789 pairs; random B covered158. The original sample had5 issues (**3.16%**), below the strict >5% expansion threshold. Seven additional targeted B reviews completed the165 required B judgments. A fresh distinct C reviewer resolved12 disputes/unsure cases. Final labels contain148 positives and641 negatives. These are **agent judgments**, not a human user study.

两系统前十与独立核实的来源段落合并去重后为 **789 对**。盲审隐藏系统、排名、分数、来源正例身份及其他审查答案。全量 A 为 789 对，随机 B 为 158 对；原抽样有 5 项问题（**3.16%**），未超过严格大于 5% 的扩审阈值。另补七项重点 B，必需 B 共 165 对。全新独立 C 裁决 12 项分歧／不确定，最终有 148 个正例、641 个负例。这些是 **Agent 判定**，不是真人研究。

Labels and every accepted packet/judgment were sealed and hash-anchored before the first score. The original sample issue count stays5 after adjudication. No question, label, model, seed or fusion parameter was changed after opening results. Full cards, private salt, rankings and per-query records remain outside Git.

首次计分前已封存并独立固定全部标签、材料及判定哈希；裁决后原抽样问题数仍为 5。打开结果后未修改问题、标签、模型、种子或融合参数。完整卡片、私有盐值、排名及逐题记录不进 Git。

A separate audit independently reconstructed all final labels, all354 per-query metric values, group/type aggregates and the10,000 bootstrap resamples from sealed evidence. All22 checks passed, with no calculation or provenance blockers. This is an audit pass and a quality-gate failure. [Sanitized independent receipt](evidence/r5-independent-20261010/independent-audit.json).

另一位审核者从封存证据独立重建全部最终标签、354 个逐题指标值、组／类型汇总及 10,000 次 Bootstrap。22 项检查均通过，没有计算或来源阻塞；这是审核通过、效果门槛未通过，脱敏回执见上述链接。

## Real inference, interface and timing / 真实推理、界面与耗时

Authoritative quality rankings came from the existing AutoDL CPU host, selected before outcomes. All59 queries passed API/offline full-order parity. The independently restored Apple M4 CPU reproduced all59 complete rankings; maximum absolute logit difference was0.000053406. The real browser shows BM25 and fused results side by side, with separate logits/fusion scores. Live API checks passed search, an empty result and an unknown-source filter returning HTTP422. [Actual screenshot and demo scope](evidence/r5-independent-20261010/README.md).

正式计分使用打开结果前选定的既有 AutoDL CPU 排名，59 题接口／离线完整顺序全部一致。Apple M4 CPU 独立恢复后也复现全部顺序，logit 最大绝对差为 0.000053406。真实浏览器并排展示 BM25 与融合结果，分别显示 logit／融合分数；检索、空结果和未知来源返回 HTTP422 的接口检查通过。实际截图及范围见上述记录。

Warm CPU benchmark: **Apple M4, 16GiB RAM, four Torch threads, batch8, concurrency1**, fixed50 queries ×3 passes,150 measured requests after10 excluded warmups. Median **572.126ms**, nearest-rank p95 **622.065ms**; peak process RSS922,124,288bytes. This meets the original **p95 ≤2000ms** point target on this Mac. It measures the in-process real API including admission, model, fusion and response serialization; it excludes cold startup, network transit and concurrent load. It is not an RTX4090 measurement or production SLA.

预热 CPU 基准采用 **Apple M4、16GiB 内存、四个 Torch 线程、批次 8、并发 1**，固定 50 题三遍、150 次计时，排除十次预热。中位数 **572.126 毫秒**，最近秩 p95 **622.065 毫秒**，进程峰值 RSS 为上述字节数。本机达到原定 **p95 不超过 2000 毫秒** 的点估计目标；范围含进程内真实接口、模型、融合及序列化，不含冷启动、网络及并发负载，不能当作 RTX4090 或生产 SLA 成绩。

## Identity, delivery and decision / 标识、交付与决定

Implementation `b76e349adcbb46a73c430889632810cbaebbd66a`; installed source identity `43facc5693d58b9aea875345e4ad40dd95c51f928c32fdb110e914cd28e64c6a`. [CI37958581548](https://github.com/Hector-sau/devsearch/actions/runs/37958581548) passed four jobs; Python3.11/3.12 ML suites each ran936 tests with zero failures/errors/skips. Later report-only commits do not become the source of these inference measurements. Exact input, label-seal and original-receipt hashes are in the compact evidence.

实现及安装源码标识如上；四项 CI 均通过，两个 Python 版本的 ML 全量各 936 项、零失败／错误／跳过。后续仅报告提交不能改写本次推理源码归属。输入、标签封存及原回执准确哈希见小型证据。

The selected inference archive (**113,941,262bytes**) and evaluation/review state are restored on the owner's external drive. Native training checkpoints remain on the original server; this is not a claim of full checkpoint backup. The published private v1.0.0 and its BM25 default are unchanged. [PR#6](https://github.com/Hector-sau/devsearch/pull/6) remains draft and unmerged. Old raw-logit reports cannot authorize fused promotion, and the new failed quality gates do not authorize it either.

选定推理包（**113,941,262 字节**）及评测／审核状态已恢复到所有者外置盘；原生训练断点仍保留服务器，不宣称全部断点已备份。已发布私有 v1.0.0 及其 BM25 默认保持原状态；PR#6 仍为未合并草稿。旧原始排序报告不能授权融合晋升，本次未通过的效果门槛同样不能授权。

The current inference/evaluation work is complete without another GPU run. Any future research needs a separately justified train/validation change and a new preregistered evaluation plan. These59 opened questions cannot guide tuning. The benchmark uses synthetic questions in the same pinned indexed snapshot and pooled top10 labels; it does not establish real-user performance, unseen-repository generalization or exhaustive corpus relevance.

本次推理／评测工作无需再跑 GPU。未来研究须另外依据训练／验证证据论证改动，并预先登记新评测方案；这 59 条已打开的问题不能指导调参。本基准使用同一固定索引快照内的合成问题及前十汇总标签，不证明真人效果、新仓库泛化或语料库完整相关性。
