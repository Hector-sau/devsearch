# DevSearch experiment report — DRAFT / 实验报告草稿

> **Draft, 2026-10-08.** GPU host: AutoDL. Codex reports `e3168ca` deployed, its installation passed and the full implementation suite
> running inside `prepare`; no result of that suite, no GPU diagnosis and no training is reported yet (see [the handoff](../docs/HANDOFF.md)). The experiment itself has **not run**: there is no reranker result, no test-set score and no on-device
> latency. This draft fixes the structure, the frozen protocol and the evidence that already exists; every result cell without evidence
> says **pending**. Sources: [evidence index](EVIDENCE_INDEX.md). The final report is produced by the campaign (`gpu.py run`) and must
> answer the questions in [docs/14 §6](../docs/14-target-verification-checklists.md).
>
> **草稿：**实验尚未运行，无重排结果、测试集成绩或目标设备延迟。GPU 主机为 AutoDL：Codex 报告已部署 `e3168ca`、安装通过、`prepare` 内全量测试正在运行，尚无测试结果、GPU 诊断或训练。本稿确定结构、冻结协议与已有证据，其余一律标“待验证”。

## 1. Question / 研究问题

On English PyTorch documentation, does a locally fine-tuned MiniLM cross-encoder that reranks BM25's top 50 improve relevance enough to
justify its added latency? Two training recipes are compared with BM25: a frozen encoder with a trained scoring head, and full fine-tuning.
"No, BM25 stays the default" is a valid outcome and is reported the same way as "yes".

## 2. Frozen setup / 冻结设置

| Item | Value | Source |
| --- | --- | --- |
| Corpus | `pytorch/tutorials@11512db7…`: 230 documents, 4,259 passages (≤189 passage tokens, none truncated) | prepare report |
| Questions | Synthetic, agent-written and agent-reviewed: 515 train, 202 validation, 197 test; median 16–17 words | question-set summary |
| Candidates | BM25 top 50 per question, fixed before training (plus known positives for review) | v2 package |
| Model | `microsoft/MiniLM-L12-H384-uncased` @ `44acabbe…`, 33,212,545 parameters | CPU check receipt |
| Length policy | 256 joint tokens, 64 for the question | `configs/protocol.toml` |
| Recipes | `frozen` (lr 1e-3, encoder in eval mode); `full` (lr 2e-5); both: FP32, dropout 0.1, AdamW (0.9, 0.999, 1e-8), weight decay 0.01, clip 1.0, 10% warmup then linear decay, effective batch 16, `pos_weight` = sampled neg/pos ratio | `configs/protocol.toml` |
| Matrix | seed 42 (deployment seed, declared), head seed 1234, up to 3 epochs | `configs/protocol.toml` |
| Checkpoint selection | per recipe: highest validation group-macro MRR@10, then group nDCG@10, then earlier epoch | docs/10 §6 |
| Default-mode rule | rerank only if validation group-MRR gain ≥ 0.02 absolute over BM25, group nDCG not lower, and warmed sequential p95 ≤ 2,000 ms at top 50 on the target device; otherwise BM25 | `configs/protocol.toml` |
| Test metrics | MRR@10, nDCG@10, Recall@50, query-macro and group-macro; paired group bootstrap, 95%, 10,000 repetitions, seed 20261005 | docs/02, `configs/protocol.toml` |
| Latency | 50 queries, 10 warm-up, 3 passes, seed 20261005; retrieval, reranking and total p50/p95 | `configs/protocol.toml` |

The learning rates differ, so the comparison is between two practical recipes, not a causal test of freezing alone.

## 3. Test labels / 测试标注

The 197 test questions' fixed candidates (9,852 items) were judged blind under review protocol revision 5 before any training:

| Step | Count |
| --- | ---: |
| Primary pass | 9,852 (9,823 resolved, 29 unsure) |
| Seeded stratified second review (label × rank band, all 197 questions covered) | 924 |
| Strata escalated to full second review (unrounded disagreement > 10%) | 4 positive strata: 18.56% (rank 1–5), 32.26% (6–10), 59.38% (11–20), 51.52% (21+); negatives ≤ 2.04% not escalated |
| Second reviews in total | 1,477 |
| Adjudications (disputes + unsure primaries) | 284; 13 left unsure went to a second adjudicator, 1 to a third; none forced |
| Final labels | 542 positive, 9,310 negative; every test question keeps ≥1 positive; 197/197 earlier positives confirmed, 345 new |

Final positives by candidate-rank band: rank 1–5: 331 of 985; 6–10: 79 of 985; 11–20: 65 of 1,970; 21+: 65 of 5,910; known positives
outside the list: 2 of 2.

**Limits.** All judgments are agent judgments (11,627 responses; 0 human). Agreement between agent reviewers measures consistency, not
validity. Of the 253 adjudicated items whose primary label was positive, the adjudicator's final label was negative for 222 and positive for 31;
one primary negative became positive. The final labels in the positive strata therefore rest mostly on the second review and
adjudication, still by agents. Test questions are synthetic.

## 4. Results available now / 现有结果

### 4.1 BM25 on validation (synthetic questions) / 验证集 BM25

| Metric (group-macro, 202 groups) | Value |
| --- | ---: |
| MRR@10 | 0.851 |
| nDCG@10 | 0.883 |
| Recall@50 | 0.995 |

Same corpus and index as the frozen package (`corpus-0a887589…`, `index-29c7419e…`). Labels are the drafted positives of the synthetic
questions, not the reviewed test bank. BM25 is the baseline, not a failure.

### 4.2 BM25 latency, proxy workload / BM25 延迟（代理负载）

50 leaf section headings as queries, 10 warm-up, 3 passes, 150 requests, 1 worker, Linux x86_64 container, Python 3.11: p50 0.86 ms,
p95 2.00 ms, max 2.66 ms. Not the target device and not user questions.

### 4.3 Engineering verification on CPU / CPU 工程验证

On `577b0ce` in CI: exact CUDA lock installs offline with proven isolation; 847 tests, 0 skipped in that environment; the real MiniLM passes 10 CPU checks;
an injected crash in `train:full:42` (exit 137) resumes to a complete campaign (attempt 2). These show the procedure works; they are not
results of the experiment.

## 5. Results pending / 待产出结果

No GPU campaign has run, so every value below is **pending**. Each cell names where the value will be read from, so it can be filled
without interpretation. Paths are relative to the campaign run root (`<runtime root>/campaign/run/`); a stage's files live in
`stages/<stage id>/<attempt id>/`, and the committed attempt is the one named by `stages/<stage id>/stage-manifest.json`. Field names were
taken from the CPU diagnostic rehearsal's outputs (`reports/evidence/rehearsal/`) and `src/devsearch/execution/stages.py`; the rehearsal's
own numbers are diagnostic and must not be copied here.

各项结果均待填写；下表给出每个数值的来源产物与字段，路径相对于 `<runtime root>/campaign/run/`。字段名取自 CPU 诊断演练产物，演练数值不可填入。

### 5.1 Test effectiveness (sealed test set) / 测试集效果

| System | MRR@10 | nDCG@10 | Recall@50 | Source |
| --- | --- | --- | --- | --- |
| BM25 | pending | pending | pending | `score` → `final-report.json`: `systems[]` entry with `system` = `bm25` (`role` = `baseline`) → `group_macro` and `query_macro` (`mrr@10`, `ndcg@10`, `recall@50`) |
| Frozen encoder, seed 42 | pending | pending | pending | same file, `systems[]` entry `frozen-seed42` (`recipe_id` = `frozen`, `seed` = 42) |
| Full fine-tuning, seed 42 | pending | pending | pending | same file, `systems[]` entry `full-seed42` (`recipe_id` = `full`, `seed` = 42) |
| Denominator | pending | | | `final-report.json` → `denominator` (`groups_scored`, `queries_scored`, `queries_without_confirmed_positive`) |

### 5.2 Difference to BM25 with confidence intervals / 相对 BM25 的差值与置信区间

| Comparison | Δ group MRR@10 [95% CI] | Δ group nDCG@10 [95% CI] | Source |
| --- | --- | --- | --- |
| Frozen − BM25 | pending | pending | `final-report.json` → `comparisons[]` (`system` = `frozen-seed42`, `baseline` = `bm25`) → `differences.mrr@10` / `differences.ndcg@10`: `observed_difference`, `interval`, `interval_excludes_zero`, `outcome`, `groups`, `repetitions`, `seed` |
| Full − BM25 | pending | pending | same, `system` = `full-seed42` |
| Bootstrap settings | 95%, 10,000 repetitions, seed 20261005 (declared) | | `final-report.json` → `bootstrap` must equal `configs/protocol.toml` `[bootstrap]` |

### 5.3 Validation selection / 验证集选择

| Item | Value | Source |
| --- | --- | --- |
| BM25 validation group MRR / nDCG | pending | `select` → `validation-selection.json` → `baseline_validation.group_mrr`, `.group_ndcg` |
| Per recipe: best epoch, per-epoch validation MRR/nDCG | pending | `validation-selection.json` → `recipes[].seeds[]` → `best_epoch`, `epochs[]` (`group_mrr`, `group_ndcg`) |
| Per recipe: gain over BM25, gates passed | pending | `recipes[]` → `mean_mrr_gain`, `mean_ndcg_change`, `mean_quality_ok`, `deployment_seed_quality_ok`, `latency_ok`, `qualifies` |
| Default mode and reason | pending | `validation-selection.json` → `decision` (`default_mode`, `recipe_id`, `seed`, `checkpoint_id`, `reason`); thresholds in `thresholds` |

### 5.4 Latency on the GPU host / 目标设备延迟

| Mode | p50 ms | p95 ms | Source |
| --- | --- | --- | --- |
| BM25 | pending | pending | `latency` → `latency.json` → `baseline` (`p50_ms`, `p95_ms`, `valid`, `requests`) |
| Rerank, frozen | pending | pending | `latency.json` → `recipes.frozen` |
| Rerank, full | pending | pending | `latency.json` → `recipes.full`; the gate reads `p95_ms` (≤ 2,000 ms) |
| Device and workload | pending | | `latency.json` → `device`, `workload`, `inference_batch`, `candidate_k` |

### 5.5 Memory, throughput and training time / 显存、吞吐与训练耗时

| Item | Value | Source |
| --- | --- | --- |
| Calibrated microbatch, accumulation, pairs/s (training) | pending | `<runtime root>/campaign/doctor-report-<UTC time>.json` (the one the plan names) → `calibration.microbatch`, `calibration.training.frozen` / `.full` (`pairs_per_second`, `accumulation_steps`, `seconds_per_update`) |
| Peak GPU memory, calibration | pending | same doctor report → `calibration.training.<mode>.memory.device_peak_bytes`, `calibration.inference.memory.device_peak_bytes` |
| Per-epoch training time, loss and memory | pending | `train:<recipe>:42` → `run-summary.json` → `epochs[]` (`seconds`, `train_loss`, `memory`) |
| Stage wall time vs estimate | pending | `summary` → `campaign-summary.json` → `estimates_vs_observed_seconds.<stage id>.observed` |

### 5.6 Release, reload and backup verification / 发布、重载与备份核验

| Item | Value | Source |
| --- | --- | --- |
| Reload parity of the exported release | pending | `reload_verify` → `reload-report.json` (`checks`, `failed`) |
| Server backup verified (on the rental's volume) | pending | `backup` → `backup-receipt.json`; `campaign-summary.json` → `backup` (`status` = `verified`, `files`, `bytes`, `receipt_sha256`) |
| Local delivery verified (operator's Mac, local project directory) | pending | a SHA-256 manifest written on the server and re-checked after download; not produced by the campaign itself |
| Gate verdicts | pending | `summary` → `acceptance-report.json` → `gates`, `accelerator_run_ready[nvidia-cuda]` |
| Recovery events | pending | `status` output and `state` (per stage `attempt`, `error`); `campaign/run` logs |

A value is entered only from these files of the final campaign, with the file's SHA-256 recorded in §7.

## 6. Threats to validity / 有效性风险

- Synthetic questions written from the same passages may favor lexical overlap (BM25) or paraphrase (reranker); they are not user queries.
- Labels are agent judgments only; no human calibration exists (not required before the GPU run; reported as a limitation).
- One seed; no seed variance is measured unless the paired extension (43, 44) is declared before the campaign.
- Single corpus and single domain; no claim beyond `pytorch/tutorials@11512db7…`.
- Latency on the rented GPU says nothing about other devices.

## 7. Reproducibility / 可复现性

Code: `e3168ca` as deployed on AutoDL (reported by Codex; `577b0ce` was the previously verified commit); dataset package `06e7ae62…`; initial seal `9a59905c…`; model
`44acabbe…`; lock `596b9dfe…`; protocol `configs/protocol.toml` at that commit. The campaign writes the execution plan, stage manifests,
final report, per-query outputs and a backup receipt; their hashes go into the final version of this report.
