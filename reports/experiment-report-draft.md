# DevSearch experiment report — DRAFT / 实验报告草稿

> **Draft, updated 2026-10-08 (evening).** The **first formal GPU campaign on `e3168ca` is complete** (AutoDL, RTX 4090). Its
> predeclared validation rule kept **BM25 as the default**: neither fine-tuned recipe beat BM25 on validation, and on the sealed test set
> full fine-tuning did not beat BM25 either. This negative result is the result of record. Values in §5 were **verified here**, read
> directly from the extracted evidence files ([evidence index §9](EVIDENCE_INDEX.md)); the delivery archives are downloaded, with the evidence and backup archives' SHA-256 verified here.
> A second round is an exploratory experiment whose **complete evidence is pending delivery**; it does not replace
> any number below. Questions follow [docs/14 §6](../docs/14-target-verification-checklists.md).
>
> **草稿（2026-10-08 晚更新）：**`e3168ca` 的第一轮正式 GPU 实验已完成。按验证集预先规定的规则，默认模式仍为 BM25：两种微调方案在验证集上都未超过 BM25，完整微调在封存测试集上也未超过 BM25。此负面结果即正式结论。第 5 节数值由本文直接读取证据文件核对；交付包已下载完成，证据包与备份包的 SHA-256 已核对。第二轮为探索实验，完整证据待交付，不替换以下任何数字。

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

## 4. Results available before the GPU campaign / GPU 实验前已有结果

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

## 5. First formal campaign results / 第一轮正式实验结果 (`e3168ca`)

Paths are relative to the campaign run root on the server; the copies read here sit under `evidence-extracted/reports/…`. Field names
follow `src/devsearch/execution/stages.py`. Labels: 197 synthetic test questions, agent judgments only; one seed (42).

数值均读取自证据文件；测试集为 197 条合成问题、仅 Agent 标注、单一随机种子。

### 5.1 Validation selection (decides the default) / 验证集选择（决定默认模式）

| System | Best epoch | Group MRR@10 | Group nDCG@10 | Δ MRR vs BM25 | Gates |
| --- | --- | --- | --- | --- | --- |
| BM25 | — | 0.8509 | 0.8827 | — | — |
| Frozen encoder (lr 1e-3) | 2 | 0.1473 | 0.2102 | −0.7036 | quality failed; latency passed |
| Full fine-tuning (lr 2e-5) | 3 | 0.8471 | 0.8808 | −0.0038 | quality failed; latency passed |

Per-epoch validation MRR@10: frozen 0.1461 / 0.1473 / 0.1473; full 0.6553 / 0.8432 / 0.8471. **Decision: `default_mode = baseline`**,
"no recipe met every predeclared validation gate; BM25 stays the default". Source: `select` → `validation-selection.json`
(`baseline_validation`, `recipes[].seeds[].epochs`, `decision`).

### 5.2 Sealed test set / 封存测试集

| System | MRR@10 | nDCG@10 | Recall@50 |
| --- | --- | --- | --- |
| BM25 (default) | 0.8925 | 0.7909 | 0.9939 |
| Full fine-tuning, seed 42 | 0.8735 | 0.7546 | 0.9939 |
| Frozen encoder, seed 42 | 0.2024 | 0.2035 | 0.9939 |

Group-macro equals query-macro here (197 groups, 197 questions, none without a confirmed positive). Recall@50 is identical because every
system reorders the same BM25 top 50. Source: `score` → `final-report.json` → `systems[]`, `denominator`.

| Difference to BM25 | Observed | 95% interval | Outcome (from the report) |
| --- | --- | --- | --- |
| Full − BM25, MRR@10 | −0.0190 | [−0.0611, 0.0227] | inconclusive: interval spans zero; no stable gain |
| Full − BM25, nDCG@10 | −0.0362 | [−0.0692, −0.0040] | worse: interval excludes zero |
| Frozen − BM25, MRR@10 | −0.6901 | [−0.7445, −0.6333] | worse |
| Frozen − BM25, nDCG@10 | −0.5874 | [−0.6363, −0.5380] | worse |

Paired group bootstrap, 95%, 10,000 repetitions, seed 20261005. Source: `final-report.json` → `comparisons[].differences`, `bootstrap`.

### 5.3 Latency on the GPU host / 目标设备延迟

| Mode | p50 ms | p95 ms | Requests |
| --- | --- | --- | --- |
| BM25 | 5.00 | 6.64 | 150 |
| Rerank, frozen | 142.41 | 150.14 | 150 |
| Rerank, full | 140.11 | 147.25 | 150 |

`cuda:0`, 50 validation questions, 10 warm-up, 3 passes, concurrency 1, top 50, inference batch 4; no degraded request. Both rerank modes
meet the 2,000 ms gate, so latency did not decide the outcome. Source: `latency` → `latency.json`.

### 5.4 Memory, throughput and training time / 显存、吞吐与训练耗时

| Item | Frozen | Full | Source |
| --- | --- | --- | --- |
| Training pairs, epochs, updates | 2,575 pairs, 3 epochs, 483 updates (effective batch 16 = microbatch 4 × 4) | same | `execution-plan.json` → `data.counts`, `training`; `run-summary.json` → `plan` |
| Calibrated training throughput | 444.6 pairs/s | 82.9 pairs/s | doctor report → `calibration.training.<mode>.pairs_per_second` |
| Peak GPU memory allocated in training | 181,717,504 bytes (≈ 0.17 GiB) | 870,730,752 bytes (≈ 0.81 GiB) | `run-summary.json` → `epochs[].memory.device_peak_allocated_bytes` |
| Train time per epoch (validation ≈ 16 s per epoch) | ≈ 7.5–8.0 s | ≈ 36.1–37.1 s | `run-summary.json` → `epochs[].seconds` |
| Training stage wall time | 75.2 s | 165.0 s | `campaign-summary.json` → `estimates_vs_observed_seconds` |
| Training loss, epochs 1→3 | 1.100 → 1.068 | 0.972 → 0.423 | `run-summary.json` → `epochs[].train_loss` |

### 5.5 Release, reload and delivery / 发布、重载与交付

| Item | State | Source |
| --- | --- | --- |
| Gates | all four passed; `accelerator_run_ready[nvidia-cuda]` true | `acceptance-report.json` |
| Release | `release-ce3160fc…`, evaluated, defaults to BM25, ships no model | `campaign-summary.json` |
| Reload verification | hashes/activation, smoke replay and API smoke passed; model parity not applicable | `reload-report.json` |
| Server backup | **verified** on the rental: 497 files, 549,641,178 bytes, receipt `68e56163…` | `campaign-summary.json` → `backup` |
| Download to the operator's Mac | **complete** (all four archives equal the manifest sizes) | delivery manifest; file sizes |
| Local SHA-256 check | **evidence and backup archives verified**; frozen and full archives not checked here | `delivery-manifest.json` |

### 5.6 Interpretation within the evidence / 在证据范围内的解读

- **Result of record:** BM25 stays the default; full fine-tuning was not shown to beat BM25 (MRR inconclusive, nDCG worse on test).
- **Frozen recipe:** the trained head alone ordered candidates far worse than BM25. The recorded training loss barely moved (1.100 → 1.068),
  so a likely cause is under-fitting of a 385-parameter head on a frozen `[CLS]` vector; this is a hypothesis, not tested.
- **What is not shown:** any effect of other learning rates, epochs, seeds or more data; any human-judged quality; any user benefit.

### 5.7 Second round (exploratory) / 第二轮（探索）

Exploratory experiment after the formal campaign; **complete evidence pending delivery**, so nothing from it is filled in here. Any later gain must be reported as a
validation result of a new, separately declared experiment, never as a replacement of the §5.2 test numbers.

### 5.8 Results-fill checklist: fields still needed from Codex / 结果填写清单（待 Codex 提供）

| Field | For | Status |
| --- | --- | --- |
| SHA-256 check of the frozen and full archives (backup and evidence verified) | §5.5 | pending |
| Second round: experiment configuration (recipe, learning rate, epochs, data, seeds), declared before running | §5.7 | pending |
| Second round: best checkpoint id and SHA-256 | §5.7 | pending |
| Second round: validation metrics (group MRR@10, nDCG@10) | §5.7 | pending |
| Whether a new, independent test set exists for the second round (the sealed test set must not be reused for tuning) | §5.7 | pending |
| Second round: latency p50/p95 on the device, if a model would be deployed | §5.7 | pending |
| Second round: export and reload verification, backup receipt, local SHA-256 check | §5.7 | pending |

## 6. Threats to validity / 有效性风险

- Synthetic questions written from the same passages may favor lexical overlap (BM25) or paraphrase (reranker); they are not user queries.
- Labels are agent judgments only; no human calibration exists (not required before the GPU run; reported as a limitation).
- Only one training seed and one learning rate per recipe were run; the negative result holds for these settings only.
- One seed; no seed variance is measured unless the paired extension (43, 44) is declared before the campaign.
- Single corpus and single domain; no claim beyond `pytorch/tutorials@11512db7…`.
- Latency on the rented GPU says nothing about other devices.

## 7. Reproducibility / 可复现性

Code `e3168ca6fef9b64ea0688ea059332c729125e30f`; dataset package `06e7ae62…`; initial seal `9a59905c…`; model `44acabbe…`; lock `596b9dfe…`;
plan `plan-449bfed3cf340b4c84fe1e7aec025caf`; protocol `protocol-c0706f3117ee60d98ca04f646e535ef2`; selection `selection-c38b654a3b3b8ddf649adcee8d03d051`;
final report `report-f62b55a0779c99ca399d1802e3626022`; release `release-ce3160fc579a52ae2ec30d15412853eb`.

SHA-256 of the evidence files read here (first 16 hex digits): `final-report.json` `cf29b10c84b5cfff`, `per-query.jsonl` `04b5f78f67104f56`,
`validation-selection.json` `723b78c4c5dc9495`, `campaign-summary.json` `b14cdeff6ad1c45f`, `acceptance-report.json` `659d5eabc8ceb68d`,
`latency.json` `6acdb7dec4e483ff`, `reload-report.json` `6a3f88282b32b323`, `run-summary.json` full `26550f31b12ab460` / frozen `0bc0b4d0358089b0`,
`execution-plan.json` `93a94491af5ed19f`.
