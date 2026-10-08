# Interview guide / 面试问答提纲

**Reference commit `e3168ca`.** An outline of likely questions with short, evidence-backed answers and the code to point at
([walkthrough](CODE_WALKTHROUGH.md), [decisions](04-integration-and-decisions.md), [evidence index](../reports/EVIDENCE_INDEX.md)).
**Every result question is answered "pending" until the GPU campaign's receipts exist** — no metric, latency, memory or training time is
quoted from estimates or from the CPU diagnostic rehearsal.

参考提交 `e3168ca`。以下为可能问题、基于证据的简短回答及对应代码；所有结果类问题在 GPU 回执到位前一律回答“待验证”，不引用估算或诊断演练数字。

## 0. Ground rules for answers / 回答原则

- **Ownership.** The code was implemented with AI agents. Per [01 §5](01-product-and-evidence.md), personal mastery is claimed only after you
  can explain section 7 below without notes and have made and interpreted one agreed change yourself. Say what you designed, decided,
  reviewed or verified; do not present generated code as your own work.
- **Labels.** All relevance labels are **agent judgments**; questions are **synthetic**. Never call them human labels, user queries or a user study.
- **Results.** Until the campaign finishes: "The comparison is designed and the test set is sealed; the results are not in yet."
  "BM25 stays the default" is a valid answer if that is what the frozen rule decides.

## 1. Product / 产品

| Question | Answer outline | Point to |
| --- | --- | --- |
| What does it do? | Given an English question, it returns passages from the pinned PyTorch tutorials with links to the exact source lines; BM25 retrieves 50 candidates, an optional MiniLM cross-encoder reranks them, 5 are shown. No generated answers. | [README](../README.md), [`SearchPipeline.search`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/serving/pipeline.py#L86) |
| Why evidence instead of generated answers? | Answers must be traceable to the documentation; the first release is about whether reranking improves finding the right passage. | [00](00-architecture-and-roadmap.md) |
| Who is it for, and has anyone used it? | Developers reading PyTorch documentation. **No user study exists**; demo questions are demonstration inputs. | [01](01-product-and-evidence.md) |

## 2. Retrieval and model / 检索与模型

| Question | Answer outline | Point to |
| --- | --- | --- |
| How does BM25 work here? | Okapi BM25, Lucene IDF, k1 1.2, b 0.75, a fixed lexical tokenizer without stemming, over 4,259 chunks; values are untuned starting points. | [`Bm25Index.score`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/retrieval/bm25.py#L125), `configs/index.toml` |
| Why a cross-encoder? | It reads question and passage together, so it can judge relevance beyond shared words; it is too slow for the whole corpus, so it only reranks BM25's top 50. | [`CrossEncoder`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L18) |
| What exactly is the model? | `microsoft/MiniLM-L12-H384-uncased` pinned at commit `44acabbe…` (33,212,545 parameters), plus dropout and one linear layer on the `[CLS]` vector giving one logit. | [`CrossEncoder.forward`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L52) |
| How are long passages handled? | They are not truncated: chunks are built to fit 256 tokens per pair (64 reserved for the question); anything longer is an error, so training and serving see identical input. | [`encode_pair`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/policy.py#L59) |

## 3. Data and labels / 数据与标注

| Question | Answer outline | Point to |
| --- | --- | --- |
| Where do questions come from? | 914 synthetic questions written and reviewed by an agent workflow (515 train / 202 validation / 197 test). No human reviewed them. | `reports/evidence/real-data/question-set-summary.json` |
| How do you avoid leakage between splits? | Evidence that overlaps is merged into groups; whole groups are assigned to a split; an independent audit recomputes overlaps; training negatives come only from train-owned chunks. | [`assign_groups`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/groups.py#L153), [`audit_assignment`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/groups.py#L193), [`mine_pairs`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/pairs.py#L112) |
| How were test labels made? | Every fixed test candidate (9,852 items) was judged blind before training under review protocol revision 5: primary pass, stratified second review (924), escalation of strata above 10% disagreement (+553), independent adjudication (284). Final: 542 positives. All agent judgments. | [`evaluate`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/review_protocol.py#L117), [experiment report §3](../reports/experiment-report-draft.md) |
| What does the high disagreement tell you? | In the four primary-positive strata the second review disagreed 18.6–59.4% of the time, so those strata were fully re-reviewed; agreement between agents measures consistency, not correctness. | [experiment report §3](../reports/experiment-report-draft.md) |
| Why not label 0 when unsure? | An unresolved item treated as irrelevant would bias every metric; unresolved items block sealing instead. | [`write_bank`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/seals.py#L67) |

## 4. Training / 训练

| Question | Answer outline | Point to |
| --- | --- | --- |
| Why a native loop instead of a trainer library? | Every decision (order, accumulation, clipping, schedule, validation, checkpoints) is visible and testable in one file. | [`train`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/loop.py#L113) |
| What loss? | Binary cross-entropy on logits with `pos_weight` = sampled negatives / positives; AdamW; linear warm-up over 10% of updates then linear decay. | [`train`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/loop.py#L113), [`lr_multiplier`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/loop.py#L57) |
| Frozen versus full fine-tuning? | Frozen: only the 385-parameter head trains (384 weights + bias), encoder in eval mode with no gradients; full: all parameters. Different learning rates (1e-3 / 2e-5), so it compares two recipes. | [`CrossEncoder.set_encoder_frozen`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L36), `configs/protocol.toml` |
| How do you recover from a crash? | Epoch-boundary, hash-verified checkpoints hold model, optimizer, scheduler, RNG and counters; resume replays the interrupted epoch. Verified on CPU with the real model: injected crash (exit 137), resume, stage attempt 2. GPU recovery: **pending**. | [`save_checkpoint`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/checkpoint.py#L46), [evidence index §3](../reports/EVIDENCE_INDEX.md) |

## 5. Evaluation / 评测

| Question | Answer outline | Point to |
| --- | --- | --- |
| How do you prevent tuning on the test set? | Selection uses validation only; the campaign cannot read test inputs until the `select` stage has committed. | [`select`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/selection.py#L74), [`World.evaluation`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/execution/world.py#L176) |
| When does the reranker become the default? | Only with ≥ 0.02 validation group-MRR gain over BM25, no nDCG loss and measured p95 ≤ 2,000 ms on the device, all declared before training. | `configs/protocol.toml` |
| Why group-macro metrics and a paired group bootstrap? | Questions about the same evidence are not independent; groups are the resampling unit and both systems share each resample. | [`paired_group_bootstrap`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/metrics.py#L79) |
| What are the results? | **Pending.** Today's only ranking numbers are BM25 on synthetic validation questions (group MRR@10 0.851, nDCG@10 0.883, Recall@50 0.995); they are a baseline, not a result. | [experiment report §4–5](../reports/experiment-report-draft.md) |

## 6. Engineering / 工程

| Question | Answer outline | Point to |
| --- | --- | --- |
| What happens if the reranker fails or is slow? | The whole reranked list is discarded and the complete BM25 list is returned with a warning code; nothing partial is shown. | [`SearchPipeline._rerank`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/serving/pipeline.py#L132) |
| How does the service handle load? | One search at a time with no queue: a second request gets HTTP 429 at once; the deadline starts at admission. | [`SearchService.submit`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/serving/api.py#L112) |
| How do you make a rented-GPU run resumable? | A stage commits atomically through its manifest; committed stages are reused only while their hashes verify; exit codes 0–6. | [`Campaign._execute`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/execution/controller.py#L322) |
| How do you know the environment is the tested one? | Hashed lock, offline install, receipts with a zero-skip full suite. The rented container refuses network namespaces, so portability is imported from CI for the same commit, while the host still runs its own full suite inside `prepare`. | [`step_receipts`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/scripts/gpu.py#L617), [`imported_evidence`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/scripts/gpu.py#L345) |
| A problem you found and fixed? | The first full test suite on the GPU host stopped with 2 failures (local test fixtures probing GitHub) and 2 skips (CUDA-unavailable tests on a CUDA machine); tests were made portable without touching production checks (reported by Codex; commit `e3168ca`). Another: a check script's receipt referenced a variable before assignment; a regression test now runs its failure path. | [HANDOFF §2](HANDOFF.md), `tests/test_real_model_resume.py` |

## 7. What the learner must explain personally / 本人须独立讲解

From [01 §5](01-product-and-evidence.md); practise each with the code open.

| Topic | Facts to explain | Code |
| --- | --- | --- |
| Tensor shapes | `input_ids`, `attention_mask`, `token_type_ids`: [batch, ≤256]; encoder output [batch, seq, 384]; `[CLS]` vector [batch, 384]; head → [batch, 1] → squeeze → [batch] logits | [`CrossEncoder.forward`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L52), [`score_pairs`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/scoring.py#L21) |
| Loss and labels | labels 1/0 per pair; `BCEWithLogitsLoss(reduction="none", pos_weight=…)`; per-group sum divided by the actual group size | [`train`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/loop.py#L113) |
| Gradient flow | frozen: the encoder runs under `no_grad`, only the head gets gradients (the CPU check confirms 0 encoder tensors receive gradients); full: all 199 tensors | [`CrossEncoder.set_encoder_frozen`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L36) |
| Train/eval behavior | dropout active only in training; the frozen encoder stays in eval mode even during training; scoring switches to eval and restores the previous mode | [`score_pairs`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/scoring.py#L21) |
| Checkpoint state | model, optimizer, scheduler, RNG states, epoch and step counters, best epoch, history; resumable only at epoch boundaries | [`load_checkpoint`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/checkpoint.py#L81) |
| One agreed change | **pending**: choose, implement and interpret one change yourself before claiming mastery | — |

## 8. Questions to answer honestly with "not yet" / 须如实回答“尚未”

Test metrics and confidence intervals; whether reranking beats BM25; on-device latency; GPU memory and training time; any human evaluation;
any user benefit. Each is pending until its receipt exists ([experiment report §5](../reports/experiment-report-draft.md) names where each
value will come from).
