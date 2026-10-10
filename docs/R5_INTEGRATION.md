# R5 fusion integration / R5 融合接入

This is an inference and independent-evaluation extension of private v1.0.0. Existing training code, checkpoints, data seals and historical results are preserved. **Completed2026-10-10:** real CPU export/reload, API/UI, parity, pooled review and scoring are verified. The59-question independent quality gates fail; BM25 remains formal default and R5 remains experimental. No new GPU training occurred. [Final result and scopes](../reports/r5-independent-holdout-2026-10-10.md).

本工作扩展私有 v1.0.0 的推理及独立评测，保留已有训练代码、权重、封存及历史结果。**2026-10-10 已完成：**真实 CPU 导出／重载、接口界面、一致性、汇总审查与计分均核验；新 59 题效果门槛未通过，正式默认保留 BM25，R5 保留实验候选。未新增 GPU 训练，结果及范围见上述报告。

## Frozen candidate / 冻结候选

Use R5 deployment seed **42**, historical best **epoch-0003**, checkpoint SHA-256 `74b7fd525d75a0aab2a126e49eb759704cd0cf51a2c955122db938d4cd9f2a3c`. Seed42 follows the existing deployment policy; seed44 is not substituted because of its higher validation score. The three-seed mean remains a family result, not this checkpoint's score.

使用上述 R5 种子 42、第三轮权重及哈希，遵循原部署种子策略，不因种子 44 验证分数较高而替换。三个种子均值继续作为系列结果，不归到本单个模型。

Keep natural BM25 **top50**, MiniLM revision `44acabbec0ef496f6dbc93adadea57f376b7c0ec`, joint token budget256/query64, no truncation, inference batch8. Fusion: `(1-alpha)/(60+bm25_rank) + alpha/(60+neural_rank)` with `alpha=0.5`, one-based ranks, descending score then ascending chunk_id. No positive injection, new training or parameter sweep.

保持自然 BM25 前 50、上述固定 MiniLM、联合 256／问题 64 token、无截断及推理批次 8。融合公式如上，参数 0.5，排名从 1 开始，同分按 chunk_id 升序。不插入正例，不重训或扫参。

## Export and use / 导出与使用

Copy [export config](../configs/export-r5-fusion.toml) into the existing runtime and resolve paths to its model-assets and search config. Keep candidate_k50. Use CPU profile while GPU is off. Export is an immutable candidate; real-model reload parity and fused smoke must pass before publication to the local runtime. The model export is approximately133MB; the native training checkpoint (including optimizer) is398,812,867bytes. Neither belongs in Git.

将上述导出配置放入已有运行目录，填写 model-assets 与搜索配置路径，保持候选 50。GPU 关机时使用 CPU。导出为不可变候选，真实模型重载一致性及融合重放通过后才放入运行目录。推理权重约 133MB，含优化器的训练断点为 398,812,867 字节，均不进 Git。

```bash
python -m devsearch.export --config "$DEVSEARCH_RUNTIME_ROOT/export-r5-fusion.toml" \
  --index "$DEVSEARCH_RUNTIME_ROOT/bundle/prepared/index/index-manifest.json" \
  --checkpoint "$DEVSEARCH_RUNTIME_ROOT/experiments/round5-hard-negatives-20261008/full-hard3-random1-seed42/checkpoints/epoch-0003.json" \
  --output "$DEVSEARCH_RUNTIME_ROOT/r5-fusion-candidate" --runtime-root "$DEVSEARCH_RUNTIME_ROOT"
python -m devsearch.serve --config "$DEVSEARCH_RUNTIME_ROOT/serve-cpu.toml" \
  --manifest "$DEVSEARCH_RUNTIME_ROOT/r5-fusion-candidate" --allow-candidate --check-only \
  --runtime-root "$DEVSEARCH_RUNTIME_ROOT"
```

`serve-cpu.toml` follows [serve schema](../configs/serve.toml), explicitly uses cpu-diagnostic and batch8. After checks, remove `--check-only` to run the loopback API. Existing modes baseline/rerank/compare remain. The candidate requires `--allow-candidate`, and no candidate performance is silently attached to the evaluated BM25 v1 release. On any inference failure, the API returns complete BM25 results with a warning, discarding partial scores.

CPU 服务配置遵循上述 schema，明确使用 cpu-diagnostic 和批次 8。检查后移除 check-only 可启动本机接口；仍保留 baseline／rerank／compare 模式。候选须显式允许，候选成绩不自动归到已评测 BM25 第一版。推理失败时丢弃部分评分，返回完整 BM25 结果和提示。

## New independent holdout / 新独立留出集

The original197-question test was exposed before R5 development. It cannot establish an untouched R5 evaluation. A read-only audit excludes every saved historical generation selection across train/validation/test, all original test question evidence groups and current train/validation owners. The remaining65 eligible groups have136 passages of at least200characters. Exhaustive exact-text, shingle-Jaccard≥0.9 and source-span checks against other chunks found no violations. This is new evidence groups and questions within the same indexed snapshot, not unseen repositories.

原 197 题测试结果在 R5 开发前已公开于开发过程，不能证明 R5 的未触碰测试效果。只读审核排除了所有历史生成选择、旧测试问题证据及训练／验证归属，留下 65 组、136 段不少于 200 字符的候选证据。逐一精确文本、Jaccard≥0.9 和来源位置检查未发现冲突。这里是同一索引快照中的新证据组和问题，不是新仓库。

Before question generation, freeze the model and protocol. Sample60groups with seed20261009; generate one independently answerable English question per group. Record answerability exclusions before any retrieval, without replacement. All questions and reviewer labels are synthetic/agent-origin and retain evidence quotations. Hash and seal the accepted question set before ranking; failed retrievals stay in the denominator.

问题生成前冻结模型及协议。按固定种子选 60 组，每组生成一个独立可回答的英文问题。检索前记录不可回答的排除项，不补选。问题及审查标签注明合成／Agent 来源并保留原文依据。排名前封存问题哈希，漏检保持在分母内。

Run only the two frozen systems. Blind the deduplicated union of their top10 passages plus the independently checked source positives; reviewers see no system identity or rank. Complete primary review; independently sample20% and also review all unsure/disputed cases; expand to complete secondary review if the sampled issue rate exceeds5%. Unresolved items block final labels and metrics. Use the same sealed labels for both systems and bootstrap evidence groups, never treating additional seeds as independent queries.

仅运行两个冻结系统。合并两者前十与独立确认的来源正例、去重并隐藏系统与排名。全量初审，独立抽样 20% 并复审不确定／分歧；抽样问题率超过 5% 则全量复审。未决项阻止封存及计分。两系统共用同一封存标签，以证据组 Bootstrap，不把额外种子当独立问题。

Report MRR@10, pooled nDCG@10, known-positive Recall@50, per-question/type results and paired95% intervals. Preserve original +0.02 MRR / nonnegative nDCG / p95≤2000ms criteria. Do not tune after opening new outcomes. Wide intervals limit the strength of claims rather than licensing another model-selection cycle on this holdout. A raw-logit formal report cannot promote a fused candidate; promotion requires a new explicitly fusion-aware reviewed evaluation record.

报告上述指标、逐题／类型结果及配对 95% 区间，保留原 +0.02／nDCG 不下降／p95≤2000ms 门槛。打开新结果后不调参；宽区间限制结论力度，不能据此在本留出集继续选模型。原始排序正式报告不能晋升融合候选，须另有明确记录融合的新评测及审核。

## Runtime and latency / 运行与延迟

The owner restored access to the existing no-card instance and authorized the minimal frozen-input transfer and external storage. The real native checkpoint was exported and reload-checked remotely on CPU. The113,941,262byte inference archive was downloaded, hash-verified and restored on the external drive; original native training checkpoints remain on the server. The no-card CPU ranked all59 questions in691.612seconds. A diagnostic300second request deadline permits this slow host to finish; it does not change the2000ms latency criterion or production configuration.

所有者已恢复既有无卡实例访问，并授权最小冻结输入传输及外置存储。真实原生断点已在远端 CPU 导出并核验重载；113,941,262 字节推理包已下载、验 hash 并恢复外置盘，原生训练断点保留服务器。无卡 CPU 在 691.612 秒内完成全部 59 题；诊断单请求 300 秒只让慢机器完成流程，不改变原 2000 毫秒门槛或生产配置。

Both the authoritative server CPU and the independent Apple M4 CPU produced identical full rankings on all59 sealed questions, with zero API/offline mismatches. The real installed API benchmark on Apple M4/16GiB/four Torch threads/batch8 used concurrency1, fixed50queries,10excluded warmups and3passes: median572.126ms/p95622.065ms across150measured requests. It includes admission, BM25, model, fusion and serialization; it excludes cold startup, network transit and concurrent load. It meets this Mac's latency point target, not an RTX4090 or production SLA claim. The completed label review and quality score are in the final report; both quality gates fail.

正式计分的服务器 CPU 与独立 Apple M4 CPU 在全部 59 题上完整顺序一致，接口／离线零差异。M4／16GiB／四线程／批次 8 的已安装真实接口按并发 1、固定 50 题、排除十次预热、三遍共 150 次测量，中位数 572.126 毫秒／p95 622.065 毫秒。范围含准入、BM25、模型、融合和序列化，不含冷启动、网络与并发负载；达到本机点估计门槛，不宣称 RTX4090 或生产 SLA。标签审核及效果计分已完成，两项效果门槛均未通过。

The CPU receipt records host-visible cores/RAM, cgroup mount-root files and process peak RSS. Mount-root files do not establish the process quota in a nested cgroup; effective container limits are explicitly unknown. Missing hardware fields are null rather than guessed. Visible host memory does not mean that memory is allocated to this instance.

CPU 回执记录宿主机可见核心／内存、cgroup 挂载根目录的原始限额及进程峰值 RSS。嵌套 cgroup 中根目录值不能证明进程配额，因此实际容器配额明确记为未知。无法读取的硬件字段留空，不猜测；可见宿主机内存不代表本实例获得同等配额。

The new59-question set is sealed under questions SHA-256 `098bf6a9e624038b816fb3aef71a1051aba302fbc4f2a509366fc78cce03df21`; query lengths≤29WordPiece tokens and source-positive pair lengths≤216 were checked with the pinned reference tokenizer. One unanswered helper-only group was excluded before retrieval. No replacement was drawn. The protocol was frozen before generation, SHA-256 `cb27eccffb91e27f2e4ade2e45dbe649b2cfe26088efa0863bc58521120da6f2`. These private JSON inputs remain outside Git.

新 59 题已按上述哈希封存，固定参考 tokenizer 检查问题最多 29 token、来源正例组合最多 216 token。一组仅含未定义辅助函数调用的材料在检索前排除，未补选。协议在生成前按上述哈希冻结。私有 JSON 输入不进入 Git。

After copying the sealed input directory to the existing runtime, use [CPU acceptance runner](../scripts/r5_cpu_acceptance.py). Both fixed hashes below are required: the preregistered plan and the post-generation input binding. The latter pins the question seal and question bytes, as well as the corpus/index, tokenizer fingerprint and encoding limits. It was frozen before any new retrieval. Changing questions and their seal together cannot preserve acceptance. The runner checks the **actual loaded** inner model and inputs against these identities, plus the outer checkpoint, fusion policy, CPU device and batch, before any API request. It refuses accelerator profiles. Rank phase calls the real in-process API, saves all natural candidate responses, verifies offline/online rank equality and computes **no quality metrics**. Benchmark phase measures the full API path and records CPU-only scope. Output directories are new; failures never overwrite old evidence. A partial rank folder is incomplete evidence and must be rerun into a new folder.

将封存输入复制到已有运行目录后使用上述 CPU 验收脚本。两个固定哈希均必需：生成前预注册计划及生成后、检索前冻结的输入绑定。后者固定问题封存文件、问题内容、语料／索引、分词器指纹及长度限制，问题与封存文件同时改写也不能通过。脚本在请求前检查**实际加载**的内部模型及输入，并核对外层断点、融合配置、CPU 设备及批次，拒绝加速器配置。排名阶段调用真实进程内接口，保存自然候选及排序一致性，**不计算效果指标**。测速阶段记录完整接口和 CPU 范围。输出目录须新建，失败不覆盖旧证据；部分排名目录不算验收，须另建目录重跑。

```bash
python scripts/r5_cpu_acceptance.py rank --inputs "$DEVSEARCH_RUNTIME_ROOT/r5-holdout-inputs" \
  --plan-sha256 cb27eccffb91e27f2e4ade2e45dbe649b2cfe26088efa0863bc58521120da6f2 \
  --binding-sha256 da8ecc36109b820065ec0d4eb19c8a40cf090fb98ba56e7c4189d7121783342d \
  --runtime-root "$DEVSEARCH_RUNTIME_ROOT" --release "$DEVSEARCH_RUNTIME_ROOT/r5-fusion-candidate" \
  --config "$DEVSEARCH_RUNTIME_ROOT/serve-cpu.toml" --output "$DEVSEARCH_RUNTIME_ROOT/r5-holdout-rankings"
python scripts/r5_cpu_acceptance.py benchmark --inputs "$DEVSEARCH_RUNTIME_ROOT/r5-holdout-inputs" \
  --plan-sha256 cb27eccffb91e27f2e4ade2e45dbe649b2cfe26088efa0863bc58521120da6f2 \
  --binding-sha256 da8ecc36109b820065ec0d4eb19c8a40cf090fb98ba56e7c4189d7121783342d \
  --runtime-root "$DEVSEARCH_RUNTIME_ROOT" --release "$DEVSEARCH_RUNTIME_ROOT/r5-fusion-candidate" \
  --config "$DEVSEARCH_RUNTIME_ROOT/serve-cpu.toml" --output "$DEVSEARCH_RUNTIME_ROOT/r5-cpu-latency"
```
