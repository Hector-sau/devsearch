# DevSearch — project presentation draft / 项目展示草稿

> **Draft, 2026-10-08.** Slide-by-slide outline with speaker notes. Slides 9–10 hold places for results that **do not exist yet**
> (pending the GPU campaign); do not fill them with estimates. Numbers come from [the evidence index](../reports/EVIDENCE_INDEX.md).
> When the deck is built, keep the "agent judgments, synthetic questions" labels on every slide that shows labels or scores.
>
> **草稿：**逐页提纲与讲稿；第 9–10 页为待 GPU 实验产出的结果占位，不以估计值填充。

## 1. Title / 标题

**DevSearch: evidence search over PyTorch documentation, with a reranker that has to earn its place.**
Notes: one sentence on the goal — a developer asks an English question and gets the exact passage and source link, not a generated answer.

## 2. Problem / 问题

- Documentation answers are scattered; keyword search finds the words, not always the explanation.
- A neural reranker may help, but costs latency and must be proven on held-out data.
Notes: frame the project as a measured decision, not "AI makes search better".

## 3. System / 系统

Visual: [DIAGRAMS.md](DIAGRAMS.md) section 4 (online path).
- Pinned snapshot of `pytorch/tutorials` → 4,259 passages → BM25 top 50 → optional MiniLM cross-encoder → top passages with source links.
- API and UI with explicit fallback: if reranking is unavailable or too slow, BM25 results are shown and the UI says so.
Notes: UI screenshots in `reports/evidence/ui-demo/` (diagnostic model; ranking meaningless by design — say so when showing them).

## 4. Data / 数据

- 230 documents, 4,259 passages at a pinned commit; real MiniLM tokenizer.
- 914 synthetic questions (515 train / 202 validation / 197 test), agent-written, agent-reviewed; **no human review**.
Notes: explain why synthetic: no user query log exists; this limits every claim.

## 5. Test labels without peeking / 不看结果的测试标注

Visual: [DIAGRAMS.md](DIAGRAMS.md) section 2.
- All 9,852 fixed test candidates judged blind **before** training; seeded stratified second review; 4 strata escalated over 10% disagreement;
  284 adjudications; 542 positives; every question has at least one.
Notes: agent judges only; agreement shows consistency, not truth. Emphasize the protocol was frozen before judging.

## 6. Experiment design / 实验设计

- BM25 vs frozen-encoder head vs full fine-tuning; seed 42; up to 3 epochs; checkpoint chosen on validation only.
- Rerank becomes default only with ≥ 0.02 validation group-MRR gain, no nDCG loss, and p95 ≤ 2 s on the device.
Notes: thresholds are product gates declared in advance (`configs/protocol.toml`), not expected results.

## 7. Engineering for an unattended run / 无人值守运行的工程保障

Visual: [DIAGRAMS.md](DIAGRAMS.md) section 1.
- One entry point (`gpu.py check → prepare → diagnose → run → status`); frozen identities; named prerequisites instead of silent fallbacks.
- Verified on CPU: exact dependency lock installed offline (isolation proven), 847 tests with zero skips, real MiniLM crash at mid-training
  (exit 137) and resume to completion.
Notes: CPU checks reduce risk on the rented host but do not remove it. The first AutoDL full test run stopped on 2 failures and 2 skips
(environment assumptions in tests, reported by Codex) and needed a correction; say so, and show GPU-host results only once they exist.

## 8. Baseline today / 当前基线

- BM25 on the 202 synthetic validation questions: MRR@10 0.851, nDCG@10 0.883, Recall@50 0.995.
- BM25 latency on a proxy workload (CPU container): p50 0.86 ms, p95 2.00 ms.
Notes: high recall@50 means the reranker can only reorder what BM25 already found; the bar for improvement is high.

## 9. Results — PENDING / 结果（待产出）

Placeholder table: test MRR@10 / nDCG@10 / Recall@50 for BM25, frozen and full, with paired 95% bootstrap intervals.
Notes: filled only from the campaign's final report.

## 10. Decision and latency — PENDING / 决策与延迟（待产出）

Placeholder: on-device p50/p95 for both modes and the frozen rule's decision (rerank default, or BM25 stays default).
Notes: present "BM25 stays default" with the same weight as an improvement.

## 11. Limitations / 局限

Synthetic questions; agent-only labels; one seed; one corpus; latency only on the rented device; no user study.

## 12. Next steps / 下一步

Done (CPU): all non-GPU preparation. On AutoDL (reported by Codex): `check`, exact 85-package installation, a CUDA tensor check; inside
`prepare` the first full test suite stopped on 2 failures and 2 skips; the test-only correction `e3168ca` is deployed in a fresh runtime
and its full suite is running inside `prepare`. Next (Codex): finish `prepare` → `diagnose` → `run` (server backup) → local delivery with
SHA-256 re-check, then fill slides 9–10 and the experiment report.
Optional later, not a prerequisite: human spot-check of labels, more seeds, MPS.
