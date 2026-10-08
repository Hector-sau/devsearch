# Interview guide / 面试问答提纲

**Reference commit `e3168ca`.** An outline of likely questions with short, evidence-backed answers and the code to point at
([walkthrough](CODE_WALKTHROUGH.md), [decisions](04-integration-and-decisions.md), [evidence index](../reports/EVIDENCE_INDEX.md)).
Results quote only the **first formal GPU campaign** (`e3168ca`), verified from its evidence files ([evidence index §9](../reports/EVIDENCE_INDEX.md));
the second round is an exploratory experiment whose complete evidence is pending delivery, so nothing from it is quoted. Nothing is quoted from estimates or the CPU diagnostic rehearsal.

参考提交 `e3168ca`。以下为可能问题、基于证据的简短回答及对应代码；结果只引用已核对的第一轮正式 GPU 实验，第二轮为探索实验，完整证据待交付，暂不引用；不引用估算或诊断演练数字。

## 0. Ground rules for answers / 回答原则

- **Ownership.** The code was implemented with AI agents. Per [01 §5](01-product-and-evidence.md), personal mastery is claimed only after you
  can explain section 8 below without notes and have made and interpreted one agreed change yourself. Say what you designed, decided,
  reviewed or verified; do not present generated code as your own work.
- **Labels.** All relevance labels are **agent judgments**; questions are **synthetic**. Never call them human labels, user queries or a user study.
- **Results.** "The first formal campaign kept BM25 as the default; full fine-tuning was not shown to beat it." Say it plainly: a negative
  result decided by a predeclared rule is a valid result. Later validation gains, if any, are validation results of a new experiment and
  never replace the first campaign's test numbers.

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
| How do you recover from a crash? | Epoch-boundary, hash-verified checkpoints hold model, optimizer, scheduler, RNG and counters; resume replays the interrupted epoch. Verified with the real model on CPU and on the RTX 4090 (`diagnose`): injected crash (exit 137), resume, stage attempt 2. | [`save_checkpoint`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/checkpoint.py#L46), [evidence index §3](../reports/EVIDENCE_INDEX.md) |

## 5. Evaluation / 评测

| Question | Answer outline | Point to |
| --- | --- | --- |
| How do you prevent tuning on the test set? | Selection uses validation only; the campaign cannot read test inputs until the `select` stage has committed. | [`select`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/selection.py#L74), [`World.evaluation`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/execution/world.py#L176) |
| When does the reranker become the default? | Only with ≥ 0.02 validation group-MRR gain over BM25, no nDCG loss and measured p95 ≤ 2,000 ms on the device, all declared before training. | `configs/protocol.toml` |
| Why group-macro metrics and a paired group bootstrap? | Questions about the same evidence are not independent; groups are the resampling unit and both systems share each resample. | [`paired_group_bootstrap`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/metrics.py#L79) |
| What are the results? | Sealed test set, 197 synthetic questions, agent labels: BM25 MRR@10 0.8925 / nDCG@10 0.7909; full fine-tuning 0.8735 / 0.7546 (MRR difference −0.019, 95% interval [−0.061, 0.023], inconclusive; nDCG worse); frozen encoder 0.2024 / 0.2035. Validation-based rule: **BM25 stays the default**. | [experiment report §5](../reports/experiment-report-draft.md) |

## 6. Engineering / 工程

| Question | Answer outline | Point to |
| --- | --- | --- |
| What happens if the reranker fails or is slow? | The whole reranked list is discarded and the complete BM25 list is returned with a warning code; nothing partial is shown. | [`SearchPipeline._rerank`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/serving/pipeline.py#L132) |
| How does the service handle load? | One search at a time with no queue: a second request gets HTTP 429 at once; the deadline starts at admission. | [`SearchService.submit`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/serving/api.py#L112) |
| How do you make a rented-GPU run resumable? | A stage commits atomically through its manifest; committed stages are reused only while their hashes verify; exit codes 0–6. | [`Campaign._execute`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/execution/controller.py#L322) |
| How do you know the environment is the tested one? | Hashed lock, offline install, receipts with a zero-skip full suite. The rented container refuses network namespaces, so portability is imported from CI for the same commit, while the host still runs its own full suite inside `prepare`. | [`step_receipts`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/scripts/gpu.py#L617), [`imported_evidence`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/scripts/gpu.py#L345) |
| A problem you found and fixed? | The first full test suite on the GPU host stopped with 2 failures (local test fixtures probing GitHub) and 2 skips (CUDA-unavailable tests on a CUDA machine); tests were made portable without touching production checks (reported by Codex; commit `e3168ca`). Another: a check script's receipt referenced a variable before assignment; a regression test now runs its failure path. | [HANDOFF §2](HANDOFF.md), `tests/test_real_model_resume.py` |

## 7. The first campaign's result: harder questions / 第一轮结果：深入问题

Each answer separates what **the project implemented** (verifiable in code and evidence) from what **you must understand yourself**.
Numbers: [experiment report §5](../reports/experiment-report-draft.md). Statements marked *hypothesis* were not tested.

每题区分“项目已实现”（可由代码与证据核对）与“需要本人理解”；标注 *hypothesis* 的为未经检验的推测。

**Q1. BM25 was already strong. Why might reranking not improve it? / BM25 已经很强时，为什么重排可能没有提升？**
- *Project implemented:* the reranker replaces BM25's order completely, using only its own scores ([`SearchPipeline._rerank`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/serving/pipeline.py#L132), [`order_by_score`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/ranking.py#L18)); it never adds new candidates, so Recall@50 (0.9939) is the same for every system.
- *Measured:* BM25's test MRR@10 is already 0.8925, so the right passage is usually ranked first; full fine-tuning reached 0.8735 (difference inconclusive), and nDCG@10 dropped (0.7909 → 0.7546).
- *Hypotheses:* the synthetic questions were written from the passages and may share their wording, which favors BM25; 2,575 training pairs from 515 questions may be too few; agent label noise in the positive strata (second-review disagreement 18.6–59.4%) limits how much any ranker can show.
- *You must understand:* why a strong first stage leaves little room at the top, why replacing (rather than combining) scores can only hurt when the new scorer is weaker, and why "inconclusive" is not "equal".

**Q2. Frozen encoder versus full fine-tuning? / 冻结编码器与全量微调有什么区别？**
- *Project implemented:* [`CrossEncoder.set_encoder_frozen`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L36) turns off gradients for the encoder and keeps it in eval mode; only the 385-parameter head (384 weights + bias) trains, at lr 1e-3. Full fine-tuning trains all 33.2 M parameters at lr 2e-5.
- *Measured:* frozen — training loss 1.100 → 1.068, validation MRR@10 0.147, test 0.2024, peak GPU memory ≈ 0.17 GiB, 444.6 pairs/s; full — loss 0.972 → 0.423, validation 0.6553 → 0.8471, test 0.8735, ≈ 0.81 GiB, 82.9 pairs/s.
- *Hypothesis:* a linear head on a frozen, not-yet-adapted `[CLS]` vector cannot express relevance for this task; the barely moving loss is consistent with under-fitting.
- *You must understand:* where gradients stop in each mode, why the frozen mode is cheaper, and why its result says nothing about full fine-tuning.

**Q3. How do learning rate, epochs and training data affect the result? / 学习率、训练轮数和训练数据量如何影响结果？**
- *Project implemented:* one learning rate per recipe, 3 epochs, 2,575 pairs, effective batch 16, 483 updates, 10% linear warm-up then linear decay ([`lr_multiplier`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/loop.py#L57)); the best epoch is chosen on validation ([`best_epoch`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/selection.py#L49)).
- *Measured:* only one setting each, so **their effect was not measured**. Full fine-tuning's validation MRR was still rising at epoch 3 (0.8432 → 0.8471).
- *You must understand:* too high a learning rate can destroy pretrained features, too low barely moves them; more epochs can under- or over-fit; more and cleaner data usually matters most. Any change is a new, declared experiment judged on validation, not on the sealed test set.

**Q4. Why select on validation and never tune on the test set? / 为什么用验证集选模型，不能反复用测试集调参？**
- *Project implemented:* selection reads validation only ([`select`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/selection.py#L74)); the campaign cannot load test inputs before `select` has committed ([`World.evaluation`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/execution/world.py#L176)); the thresholds were declared in `configs/protocol.toml` before training.
- *You must understand:* every look at test scores that influences a choice makes the test score optimistic (selection bias); after enough tuning the "test" set is just another validation set. A second round that wants a clean test needs a new, independent test set.

**Q5. Why must agent labels never be called human labels? / 为什么 Agent 标注不能写成人工标注？**
- *Project implemented:* every response records `reviewer.type`; the final report counts 11,627 agent and 0 human judgments; questions are recorded as synthetic.
- *You must understand:* agent agreement shows consistency between models, not that a person would agree; calling them human labels misstates the evidence and is not allowed by [01 §5](01-product-and-evidence.md).

**Q6. What does the current result prove, and what not? / 当前结果能证明什么，不能证明什么？**
- *Engineering, shown:* exact-lock installation and the full suite (848 tests, 0 skipped) on the GPU host; a real-model crash and resume on CUDA; an unattended campaign with all 16 stages committed; validation-only selection with predeclared gates; a sealed test evaluation with paired bootstrap intervals; a release that reloads and a verified server backup.
- *Effectiveness, not shown:* that any trained reranker beats BM25 here; any human-judged quality; any user benefit or time saved; behaviour on other corpora, other devices or other settings.

## 8. What the learner must explain personally / 本人须独立讲解

From [01 §5](01-product-and-evidence.md); practise each with the code open.

| Topic | Facts to explain | Code |
| --- | --- | --- |
| Tensor shapes | `input_ids`, `attention_mask`, `token_type_ids`: [batch, ≤256]; encoder output [batch, seq, 384]; `[CLS]` vector [batch, 384]; head → [batch, 1] → squeeze → [batch] logits | [`CrossEncoder.forward`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L52), [`score_pairs`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/scoring.py#L21) |
| Loss and labels | labels 1/0 per pair; `BCEWithLogitsLoss(reduction="none", pos_weight=…)`; per-group sum divided by the actual group size | [`train`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/loop.py#L113) |
| Gradient flow | frozen: the encoder runs under `no_grad`, only the head gets gradients (the CPU check confirms 0 encoder tensors receive gradients); full: all 199 tensors | [`CrossEncoder.set_encoder_frozen`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L36) |
| Train/eval behavior | dropout active only in training; the frozen encoder stays in eval mode even during training; scoring switches to eval and restores the previous mode | [`score_pairs`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/scoring.py#L21) |
| Checkpoint state | model, optimizer, scheduler, RNG states, epoch and step counters, best epoch, history; resumable only at epoch boundaries | [`load_checkpoint`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/checkpoint.py#L81) |
| One agreed change | **pending**: choose, implement and interpret one change yourself before claiming mastery | — |

## 9. Questions to answer honestly with "not yet" / 须如实回答“尚未”

Whether any reranker beats BM25 on this data; any second-round result; any human evaluation; any user benefit; latency on a CPU or Mac.
