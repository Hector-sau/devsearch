# DevSearch experiment report — DRAFT / 实验报告草稿

> **Draft, 2026-10-08.** The experiment itself has **not run**: there is no reranker result, no test-set score and no on-device
> latency. This draft fixes the structure, the frozen protocol and the evidence that already exists; every result cell without evidence
> says **pending**. Sources: [evidence index](EVIDENCE_INDEX.md). The final report is produced by the campaign (`gpu.py run`) and must
> answer the questions in [docs/14 §6](../docs/14-target-verification-checklists.md).
>
> **草稿：**实验尚未运行，无重排结果、测试集成绩或目标设备延迟。本稿确定结构、冻结协议与已有证据，其余一律标“待验证”。

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

Exact CUDA lock installs offline with proven isolation; 847 tests, 0 skipped in that environment; the real MiniLM passes 10 CPU checks;
an injected crash in `train:full:42` (exit 137) resumes to a complete campaign (attempt 2). These show the procedure works; they are not
results of the experiment.

## 5. Results pending / 待产出结果

| System | Test MRR@10 | Test nDCG@10 | Recall@50 | Δ vs BM25 (95% CI) | p50 / p95 total latency |
| --- | --- | --- | --- | --- | --- |
| BM25 | pending | pending | pending | — | pending (target device) |
| Frozen encoder (seed 42) | pending | pending | pending | pending | pending |
| Full fine-tuning (seed 42) | pending | pending | pending | pending | pending |

| Pending item | Produced by |
| --- | --- |
| Validation curves and selected epochs per recipe | campaign `select` stage |
| Default-mode decision under the frozen rule | campaign `select` stage |
| Test scores, paired bootstrap intervals, per-query results | `rank_test`, `score`, `seal_final` |
| On-device latency p50/p95 for both modes | `latency` stage |
| Memory, batch size and throughput actually used | doctor calibration |
| Failures, retries, crash recoveries | campaign status and logs |

## 6. Threats to validity / 有效性风险

- Synthetic questions written from the same passages may favor lexical overlap (BM25) or paraphrase (reranker); they are not user queries.
- Labels are agent judgments only; no human calibration exists.
- One seed; no seed variance is measured unless the paired extension (43, 44) is declared before the campaign.
- Single corpus and single domain; no claim beyond `pytorch/tutorials@11512db7…`.
- Latency on the rented GPU says nothing about other devices.

## 7. Reproducibility / 可复现性

Code: the approved commit (`577b0ce` or `22db6f4`, owner decision pending); dataset package `06e7ae62…`; initial seal `9a59905c…`; model
`44acabbe…`; lock `596b9dfe…`; protocol `configs/protocol.toml` at that commit. The campaign writes the execution plan, stage manifests,
final report, per-query outputs and a backup receipt; their hashes go into the final version of this report.
