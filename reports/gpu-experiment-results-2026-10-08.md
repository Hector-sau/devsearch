# GPU experiment results / GPU 实验结果

> **Dated development record / 带日期的开发记录：** this report covers original labels and the pilot. Expanded v4 labels and the latest grouped-loss comparison are in [RESULTS](RESULTS.md). / 扩展 v4 标签与最新分组损失实验见统一结果报告。

**Training through Round 7 and the label pilot are complete:** rank fusion reached mean validation MRR@10 **0.893871** versus BM25 **0.850878** across three seeds. The 3+1 negative mix reached **0.900565**, but its incremental interval includes zero. Replacing eight reviewed negatives (Round 6) and reducing BCE positive weight (Round 7) failed their adoption rules, reaching **0.891845** and **0.884804**. A separate 20-question blinded pilot confirmed 31 additional positives; all 25 included configurations were reevaluated. BM25 remains the original evaluated default; the formal +0.02 gate is unchanged, and no follow-up test evaluation or release promotion has occurred.

**第七轮训练及补标试点均已完成：**排名融合三种子验证 MRR@10 均值为 **0.893871**，BM25 为 **0.850878**。困难／随机负例从 2+2 改为 3+1 后达到 **0.900565**，但增量区间包含零。第六轮八条负例替换与第七轮 BCE 正例权重降低均未通过采纳门槛，分别为 **0.891845**、**0.884804**。独立 20 题盲审新增 31 条正例，已重评纳入比较的全部 25 个配置；BM25 正式默认及 +0.02 门槛保持不变，未重新评测测试集或提升发布版本。

## Reproducible setup / 可复现配置

- Executed source commit: `e3168ca6fef9b64ea0688ea059332c729125e30f`. Later report-only commits do not change the measured code identity.
- RTX 4090, 24 GB; NVIDIA driver 580.105.08; Python 3.12.3; torch `2.14.1+cu130`; FP32, TF32 disabled.
- Exact CUDA lock SHA-256: `596b9dfe7608e62f811f3931de5895035794b8973736f8b26159d8e06906c27a`; 85 packages verified.
- `microsoft/MiniLM-L12-H384-uncased` at `44acabbec0ef496f6dbc93adadea57f376b7c0ec`; 33,212,545 parameters including the scoring head.
- Corpus: pinned `pytorch/tutorials@11512db7cbbcc4b530ecfc63205d3fed13fa190d`; 515 training, 202 validation and 197 test queries; 2,575 training pairs. Questions are synthetic. The completed test-candidate review bank has 9,852 final items and 11,627 agent responses; **zero human judgments**.
- Effective batch 16, microbatch 4, gradient accumulation 4, head seed 1234, three epochs per run, dropout 0.1, weight decay 0.01, maximum pair/query lengths 256/64. Inference latency uses batch 4 and 50 BM25 candidates.

以上为实际运行配置；测试候选审查项、审查回答次数、训练样本对和问题数量是不同单位。合成问题与 Agent 审查不属于真人用户测试或人工标注。

## First formal campaign / 首轮正式实验

Plan `plan-449bfed3cf340b4c84fe1e7aec025caf`; final report `report-f62b55a0779c99ca399d1802e3626022`. Model selection and the default mode were locked on validation before test scoring. The original thresholds were MRR@10 gain at least 0.02, nonnegative nDCG@10 change, and p95 at most 2,000 ms.

正式计划先根据验证集锁定模型和默认模式，随后才计算测试成绩。预设门槛为 MRR@10 至少提升 0.02、nDCG@10 不下降、p95 不超过 2,000 毫秒。

| System / 系统 | Validation MRR@10 | Validation nDCG@10 | Test MRR@10 | Test nDCG@10 | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.850878 | 0.882728 | 0.892519 | 0.790862 | 6.636 |
| Frozen encoder, LR 1e-3, seed 42 | 0.147299 | 0.210161 | 0.202427 | 0.203488 | 150.141 |
| Full tuning, LR 2e-5, seed 42 | 0.847118 | 0.880812 | 0.873469 | 0.754642 | 147.252 |

All test systems scored the same 197 groups/queries. Known-positive Recall@50 was 0.993909 for each: reranking changes order within the same candidate set, so it cannot improve candidate recall. Latency was measured separately on the fixed validation workload, not on the test questions. The formal `run` took approximately 6 min 24 sec, including approximately 4 min in the two training stages; installation, downloads, CI, separate doctor and recovery checks are excluded.

测试系统使用相同的 197 个分组／问题，已知正例 Recall@50 均为 0.993909。重排只改变已有候选的顺序，不能提高候选召回率。延迟单独在固定验证负载上测量。正式运行约 6 分 24 秒，其中两组训练合计约 4 分钟；不包含环境准备、下载、CI 和单独诊断。

## Validation-only follow-up / 仅验证集后续实验

The user requested further experiments after the first run. Two learning rates were recorded before execution; all other settings and data stayed fixed. Both started from the original pretrained encoder and seeded head. No test inputs were used by these training/evaluation commands. The original test results had already been observed, so these follow-ups are explicitly exploratory and are not a new independent test result.

用户要求继续试验后，先记录两个待比较的学习率，再执行训练，其他配置和数据保持不变，每组从原始预训练编码器及固定随机头开始。后续命令只读训练／验证数据；鉴于原测试结果已被看到，本轮明确属于探索实验，不作为新的独立测试结果。

| Learning rate / 学习率 | Seed | Best epoch / 最佳轮次 | Validation MRR@10 | Validation nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| 1e-5 | 42 | 3 | 0.686040 | 0.734600 |
| 2e-5 (original / 原方案) | 42 | 3 | 0.847118 | 0.880812 |
| 5e-5 | 42 | 3 | 0.871920 | 0.898479 |
| 5e-5 (replication / 复测) | 43 | 2 | 0.869055 | 0.898768 |
| 5e-5 (replication / 复测) | 44 | 3 | 0.861822 | 0.893638 |

The two replication seeds, 43 and 44, were declared before those runs. The deployment candidate remained seed 42; no seed was selected after seeing the replication scores. All four new training commands exited 0, each completing three epochs in 171–176 seconds (11 min 37 sec combined).

复测前已固定种子 43、44，部署候选保持为种子 42，没有看完分数再挑种子。四组新增训练均退出 0，各完成三轮，耗时 171–176 秒，训练命令合计约 11 分 37 秒。

For LR 5e-5, three-seed mean validation MRR@10 is **0.867599 ± 0.005204** (sample standard deviation), and nDCG@10 is **0.896962 ± 0.002882**. All three exceed BM25, but mean MRR gain is **0.016721**, below 0.02. Seed 42 improved 32 validation queries and regressed 23, with 147 ties. Its descriptive paired group-bootstrap 95% interval for MRR gain is **[-0.017456, 0.060033]**, which includes zero. This interval does not correct for repeated selection on validation; it cannot establish a stable generalization gain.

学习率 5e-5 的三种子验证 MRR@10 均值为 **0.867599 ± 0.005204**，nDCG@10 为 **0.896962 ± 0.002882**，其中 ± 表示样本标准差。三次均超过 BM25，但平均 MRR 提升 **0.016721**，未达到 0.02。种子 42 有 32 个问题变好、23 个变差、147 个不变；提升的描述性 95% 分组自助法区间包含零，且没有校正反复使用验证集的选择效应，不能据此宣称稳定泛化提升。

## Candidate export and next decision / 候选导出与下一步

Candidate `model-20b8e52fcdd5220ff80604446d9c1483` comes from seed 42, epoch 3, checkpoint SHA-256 `aea6c647312b5fa777451f95842b9cefe6ad303108c52dc685134b1822af8848`. Reload parity passed on eight validation pairs with maximum logit difference **0.0** and identical ranks. The shared search pipeline measured **144.145 ms p95** for reranking versus **6.428 ms** for BM25: 50 validation queries, 10 warm-ups, three passes, 150 measured requests, concurrency 1, no response cache, zero degraded requests. This is an inference bundle, not a promoted evaluated release.

候选模型来自种子 42 的第 3 轮。八组验证样本导出重载后分数最大误差为 **0.0**，排序一致。共用检索链路的重排 p95 为 **144.145 毫秒**，BM25 为 **6.428 毫秒**；使用 50 个验证问题、10 次预热、三遍共 150 次计时请求、并发 1、无响应缓存、零降级。该模型是候选推理包，未替换正式发布版本。

Next useful work is targeted data and relevance review, not an unbounded learning-rate search. Inspect the recorded validation wins/regressions, especially exact API/version questions; audit whether other relevant passages are missing from the sparse validation labels. Any label correction needs a new version and rerunning every compared system. Improve hard negatives using training-owned groups only. Before claiming independent improvement, freeze the next candidate and its protocol, then evaluate a fresh held-out set with documented reviewer provenance. Human review, if performed, must be recorded separately from existing agent judgments.

下一步优先针对数据和相关性做改进，而不是无限搜索学习率：检查已保存的验证集正反例，尤其精确 API／版本问题，并审核稀疏标签是否遗漏其他相关段落。修改标签必须形成新版本并重评全部对照系统。困难负例只能来自训练归属分组；宣称独立提升前，应先固定下一候选和协议，再使用新的未见留出集评测。若引入人工审核，须与既有 Agent 判定分别记录。

## Round 4: lexical/neural rank fusion / 第四轮：词法与神经排名融合

Weights stayed fixed. On the same 202 validation groups and 50 fixed candidates per query, compare `score=(1-alpha)/(60+bm25_rank)+alpha/(60+neural_rank)`, with three preregistered new weights and two endpoint controls. Both endpoint controls reproduced the existing metrics exactly. Seed 42 remained fixed as the candidate seed; weights were compared by mean group MRR across seeds 42/43/44, with nDCG and then smaller alpha as tie breakers.

模型权重保持不变，在相同的 202 个验证分组、每题 50 个固定候选上比较上述公式。预先固定三个新权重及两个端点对照；端点精确复现原成绩。按三种子平均 MRR 选权重，候选种子仍固定为 42。

| Neural weight / 神经权重 | Mean MRR@10 | Mean nDCG@10 | Mean MRR gain over BM25 |
| --- | ---: | ---: | ---: |
| 0 (BM25 control) | 0.850878 | 0.882728 | 0.000000 |
| 0.25 | 0.882700 | 0.907228 | 0.031822 |
| **0.50** | **0.893871** | **0.917516** | **0.042993** |
| 0.75 | 0.889184 | 0.913710 | 0.038306 |
| 1 (neural control) | 0.867599 | 0.896962 | 0.016721 |

At alpha .5, the three MRR values were 0.895509, 0.890512 and 0.895592 (sample SD 0.002910). Seed 42 had 31 wins and 8 losses versus BM25; 15 of the previous 23 neural regressions were fully recovered, and 22 improved relative to pure neural reranking. The descriptive group-bootstrap interval for the three-seed mean gain was [0.018841, 0.068822]. It does not account for selection on validation and is not independent test evidence. GPU scoring covered 10,100 pairs per seed in approximately 16 seconds per seed. All 24 earlier epoch checkpoints remain on the server.

权重 .5 时，三个种子的 MRR 分别为 0.895509、0.890512、0.895592。种子 42 对基线有 31 个问题改善、8 个退步；原纯神经模型的 23 个退步中，15 个恢复到基线，22 个较纯神经改善。描述性区间虽不跨零，仍未校正验证集选择效应，不能当成独立测试结论。每个种子在 GPU 上评分 10,100 对约需 16 秒；原有 24 个断点保留。

## Round 5: controlled negative mix / 第五轮：训练负例组成对照

Before running, fix one treatment: 3 BM25 hard negatives + 1 random negative instead of 2+2, preserving 515 positives, 2,575 total pairs, the training-owned pool, LR 5e-5, three epochs, seeds 42/43/44, and fusion alpha .5. First re-mine the original configuration and require exact pair-record equality; reject any pool, count or positive-set mismatch. Unreviewed negatives remain assumed negatives. Use the native installed training loop with a new pair manifest and new checkpoint identities. Adoption requires mean fused MRR gain >= .005 over round 4 and nonnegative nDCG change. No test inputs, original-file edits, checkpoint deletion, model export or large-file return.

第五轮预先固定单一改动：从 2 个困难负例 + 2 个随机负例改为 3+1。515 个正例、2,575 对总样本、训练归属池、学习率、三轮、三个种子及融合权重全部保持固定。先要求原负采样逐条精确复现，再构建独立新样本清单；拒绝分组、数量或正例集合变化。未知负例仍标假定负例，不冒充新标注。采用原生训练循环和独立断点身份；不读测试、不删除断点、不导出或回传大文件。

All three runs completed. Fused MRR values for seeds 42/43/44 were **0.895063 / 0.902872 / 0.903760**; mean **0.900565**, sample SD **0.004785**. Mean nDCG was **0.922885**. The mean increment over round 4 was **+0.006694 MRR / +0.005370 nDCG**. The preset exploratory adoption rule passed, but the descriptive MRR interval **[-0.002573, 0.016183]** includes zero; seed 42 slightly regressed. These are validation-only, seed-conditional intervals without multiple-selection correction. Training took 158.1, 160.7 and 162.7 seconds; all checkpoints remain on the server.

三个种子均完成，融合 MRR 分别为 **0.895063 / 0.902872 / 0.903760**，均值 **0.900565**、样本标准差 **0.004785**；nDCG 均值 **0.922885**。相对第四轮平均增加 **0.006694 MRR / 0.005370 nDCG**，达到探索实验预设门槛，但描述性 MRR 区间 **[-0.002573, 0.016183]** 包含零，种子 42 略有退步。区间仅描述固定三个种子下的验证问题不确定性，未校正多次选择。三次训练共约八分钟，断点全部保留。

## Round 6 review and controlled change / 第六轮评审与受控改动

Three agents separately owned data/error analysis, training design and independent evaluation review. All agreed on a narrow training-only intervention: exclude eight specifically reviewed answer-bearing negatives and replace their slots with next eligible BM25 negatives from the same frozen training pool. Keep the 3+1 mix, 515 positives, 2,575 pairs, every other pair slot, random-negative membership, BCE, model, optimizer, LR, seeds, epoch selection, validation v1 and fusion fixed. Do not promote the excluded passages to confirmed positives. Review replacement context before GPU execution; no broader mask or loss-function sweep is authorized by this experiment.

三名 Agent 分别负责数据错例、训练方案与独立评测，文件范围互不重叠；一致选择八条明确含答案的训练负例，按原位置替换为同一冻结训练池中后续合格 BM25 候选。其余样本及位置、随机负例、BCE、模型、优化器、学习率、种子、轮次选择、验证标签及融合权重全部保持固定。排除项不新增为已确认正例，替代段落须先审核；本轮不扩展过滤规则或损失函数搜索。

The audit found no silent truncation: maximum train pair length 229 tokens and validation pair length 227, below 256; query/passage limits also passed. Independent CPU reconstruction matched all 15 round-4 seed/weight scores and round-5 statistics. However, validation has just **202 positive judgments, one per query, and no judged negatives**. Multiple answer-bearing passages can therefore be scored as misses. Labels remain unchanged; any broader relevance correction needs a separately versioned, balanced review of all compared systems. Eight identified training negatives are **0.388% of 2,060 negatives**, not an estimate of the dataset error rate.

审核未发现静默截断：训练样本对最大 229 token、验证最大 227，均低于 256，问题／段落限制也通过。独立 CPU 重算与第四轮 15 组成绩及第五轮统计一致。但验证集仅有 **202 条正例标注，每题一条，没有已审负例**，其他正确段落可能被计为失败。因此保持原标签不变；若扩充相关性标注，须对全部对照系统做平衡审核并形成独立版本。八条已识别训练负例占 2,060 条负例的 **0.388%**，不代表数据错误率估计。

## Round 6 and 7 outcomes / 第六、七轮结果

Round 6 replaced exactly eight hard-negative slots; independent reconstruction confirmed all other pair positions and settings identical. Mean fused MRR fell by 0.008720 versus Round 5 (95% descriptive paired-group interval −0.018346 to +0.000774); nDCG fell by 0.007623. The preregistered 199-query sensitivity also declined. The eight false-negative findings remain valid data-quality evidence, but this replacement policy is not a measured performance upgrade.

第六轮恰好替换八个困难负例位置，其余样本和设置经独立重算确认相同。融合 MRR 较第五轮下降 0.008720，描述性分组区间为 [−0.018346, +0.000774]；nDCG 下降 0.007623，预设 199 题诊断也下降。八条负例矛盾仍是数据质量证据，但当前替换策略不能当作实测性能升级。

Round 7 changed only BCE positive weight 4 → 1, reusing the exact Round 6 pairs. Fused MRR for seeds 42/43/44 was 0.896169 / 0.894513 / 0.863730, mean **0.884804**, sample SD **0.018270**. Against Round 6, MRR changed **−0.007041** (95% descriptive interval [−0.019625, +0.005413]); nDCG changed **−0.006820** to **0.908442**. Raw MRR also declined to 0.850913. All seeds completed 483 updates/three epochs and selected epoch 3; three training commands totaled about 8 min 1 sec. Independent CPU recalculation found zero metric discrepancies and verified all 12 new checkpoint hashes. No extra weighting sweep was launched.

第七轮仅把 BCE 正例权重 4 改为 1，复用第六轮完全相同的样本。三个种子的融合 MRR 为 0.896169 / 0.894513 / 0.863730，均值 **0.884804**、样本标准差 **0.018270**。较第六轮 MRR 下降 **0.007041**，描述性区间 [−0.019625, +0.005413]；nDCG 下降 **0.006820** 至 **0.908442**。纯模型 MRR 也降至 0.850913。三个种子均完成 483 次更新、三轮训练并选择第 3 轮，训练命令合计约 8 分 1 秒；独立 CPU 重算零指标差异，12 个新增断点哈希通过，不追加权重搜索。

Both are recorded negative experiments. Intervals are descriptive and do not correct repeated validation selection or fully represent future seed variability. Frozen v1 labels remain unchanged. Three preidentified answer-span overlap cases motivate the 199-query diagnostic; the remaining set is not certified leakage-free. The subsequent v2 label pilot cannot retroactively change either training decision.

两轮均保留为负面实验。区间未校正验证集反复选模，不能完整代表未来种子波动。原 v1 标签不变；三个预先发现的答案片段重叠项促成了 199 题诊断，但其余数据不能称为已完全排除泄漏。后续 v2 补标试点不追溯改变这两轮训练决策。

## Blinded validation label pilot / 验证标签盲审试点

Freeze 20 validation questions before judging: 7 improved, 7 regressed and 6 tied under Round 5 mean fusion versus BM25, using declared SHA256 ordering. Pool the top5 union from BM25 and rounds4–7 × three seeds × raw/fused, plus hidden original-positive anchors. Fresh A/B agents independently reviewed all 226 query–passage pairs with system/rank/old-label information hidden. They agreed on 223/226 (98.67%); C independently judged three disputes before reading anonymous rationales. Final labels are **51 positive / 175 negative**, including **31 additional positives across 15 questions**; all 20 original positives were confirmed. These are agent judgments, not human verification, and agreement does not establish accuracy.

判定前按第五轮融合相对 BM25 固定抽取 7 题改善、7 题退步、6 题持平，使用预定 SHA256 排序。合并 BM25 及第四至七轮 × 三种子 × 纯模型／融合的前五名，并加入隐藏原正例。新的 A、B Agent 在不看系统、排名或旧标签的条件下独立审查 226 对，一致 223 对（98.67%）；C 先独立盲判三项争议，再阅读匿名理由裁决。最终 **51 正、175 负**，其中 **15 题新增 31 条正例**，原 20 正例全部确认。全部是 Agent 判定，不是人工核实，一致率也不是准确率。

All 25 frozen configurations were reevaluated on the same 20 questions with both label versions; independent CPU recomputation found zero discrepancies. The table below is the **20-query pilot**, not the full 202-query validation. Model rows equally average the three seeds. Weights and rankings did not change; v1 remains byte-identical.

25 个固定配置都用同样的 20 题按两版标签重算，独立 CPU 复算零差异。下表是 **20 题试点**，不是完整 202 题；模型行等权汇总三个种子。模型与排序未变，v1 字节未变。

| System / 系统 | Pilot v1 MRR | Pilot v2 MRR | Pilot v2 nDCG |
|---|---:|---:|---:|
| BM25 | 0.657976 | 0.755833 | 0.714970 |
| Round 4 fusion | 0.686944 | 0.824167 | 0.777504 |
| Round 5 fusion | 0.694048 | 0.810000 | 0.765947 |
| Round 6 fusion | 0.691111 | 0.800556 | 0.757624 |
| Round 7 fusion | 0.701362 | 0.842222 | 0.761882 |

All included top5 results are judged, but top10 coverage is only 73.5%–83.0% across individual configurations; top50 coverage is 22.5%. Known-positive Recall@50 stays .95, including one unretrieved original positive. The small sample deliberately oversamples regressions, has no human audit and does not eliminate train/validation answer overlap. Label expansion changes nDCG's ideal denominator; gains between label versions are not training gains. Do not select a new winner or overturn Round 7's full202 decision from this pilot. Broader relevance review is the next useful step before another frozen training comparison.

纳入比较的前五名已全部审查，但各配置前十已审比例仅 73.5%–83.0%，前五十为 22.5%；已知正例 Recall@50 保持 .95，包含一个未召回的原正例。小样本刻意多抽退步项、未经人工核实，也未消除训练／验证答案重叠。扩充标签会改变 nDCG 的理想分母，两版成绩变化不能当作训练提升。不根据该试点挑新模型或推翻第七轮完整 202 题结论，下一步优先扩大相关性审查。

Pilot version: `validation-v2-pilot20`; qrels SHA256 `ffe8b20dd6c367e32f1c78029aa3c9ba27016d5dc108f95bf0e1bb6cd880fbd3`. All 60 epoch checkpoints remain on the GPU server and passed SHA256 verification. Only small labels/reports were transferred, with no new model download, paid API or hardware rental.

试点版本及标签哈希如上；服务器 60 个轮次断点均保留且 SHA256 验证通过，仅传输小型标签和报告，没有新增模型下载、付费 API 或租机。

## Evidence / 证据

- [Regular CI](https://github.com/Hector-sau/devsearch/actions/runs/37760709410): four jobs passed on the executed source commit.
- [Evidence CI](https://github.com/Hector-sau/devsearch/actions/runs/37760704700): exact-lock suite **848 tests, zero failures/errors/skips**, Linux isolation proof, and pinned-model CPU recovery. The actual GPU host separately passed **848 tests, zero failures/errors/skips** and real-model CUDA crash/resume (137 then 0). CI isolation is not claimed for the restricted AutoDL container.
- [Small evidence manifest](evidence/gpu-20261008/evidence-manifest.json): checksums for the formal report/selection/latency/summary and follow-up comparison/stability/export measurements. No raw questions, labels, model weights or credentials are committed with this report.
- Private formal delivery: evidence archive SHA-256 `dcc4bd6367ae46f9be0de0864541498d3d4c87309e014db24cbcdb9f9f1ae812`; backup archive `b8ee3cd6817a3fcf97d60d87193b76d485743ecc4737c010f74d2fd489bb0e60`.
- Private follow-up delivery: evidence archive **88,671 bytes**, SHA-256 `1949a3478aca56f1ae233129e1458c82c56afd89fc1a3c1233b12d6b56b8337f`; candidate archive **116,534,717 bytes**, SHA-256 `230d08999270864e96b3f703b9b708808eab62e0bf0ac8cd69102e612f05ae44`.

上述小型记录可随代码审查；完整模型、数据和运行日志保存在私有运行／交付目录。MPS 仍未完成本项目的实机验收；没有真人用户研究，也没有新增测试集成绩。
