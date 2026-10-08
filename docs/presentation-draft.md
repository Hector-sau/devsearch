# DevSearch — project presentation draft / 项目展示草稿

> **Draft, updated 2026-10-08 (evening).** Slide-by-slide outline with speaker notes. Slides 9–10 now carry the **first formal GPU
> campaign's** results (`e3168ca`), verified from its evidence files ([evidence index §9](../reports/EVIDENCE_INDEX.md)). The second round is an
> exploratory experiment whose complete evidence is pending delivery; nothing from it is shown. Keep the "agent judgments, synthetic questions" labels on every slide that shows labels or scores;
> show screenshots only once they are captured from the actual releases.
>
> **草稿（晚间更新）：**第 9–10 页已填入第一轮正式 GPU 实验结果（已核对证据文件）；第二轮为探索实验，完整证据待交付，暂不展示；截图须取自实际发布包。

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
- On the RTX 4090: exact-lock installation, the full suite inside `prepare` (848 tests, 0 skipped), a real-model crash (exit 137) and
  resume, then all 16 campaign stages committed on the first attempt; server backup verified.
Notes: CPU checks reduced risk but did not remove it: the first AutoDL full test run stopped on 2 failures and 2 skips (test environment
assumptions) and needed a correction before the campaign. Say so.

## 8. Baseline / 基线

- BM25 on the 202 synthetic validation questions: MRR@10 0.851, nDCG@10 0.883, Recall@50 0.995.
- BM25 latency on a proxy workload (CPU container): p50 0.86 ms, p95 2.00 ms.
Notes: high recall@50 means the reranker can only reorder what BM25 already found; the bar for improvement is high.

## 9. Results on the sealed test set / 封存测试集结果

| System | MRR@10 | nDCG@10 | Recall@50 |
| --- | --- | --- | --- |
| BM25 | 0.8925 | 0.7909 | 0.9939 |
| Full fine-tuning | 0.8735 | 0.7546 | 0.9939 |
| Frozen encoder | 0.2024 | 0.2035 | 0.9939 |

Full − BM25: MRR@10 −0.019 [−0.061, 0.023] (inconclusive), nDCG@10 −0.036 [−0.069, −0.004] (worse). 197 synthetic questions, agent labels.
Notes: say plainly that fine-tuning was not shown to beat BM25; the frozen recipe failed outright (training loss barely moved).

## 10. Decision and latency / 决策与延迟

- Decided on validation before the test was read: no recipe met the gates (full −0.0038 MRR@10 vs BM25), so **BM25 stays the default**.
- Latency on the RTX 4090 (p50 / p95): BM25 5.0 / 6.6 ms; rerank 140 / 147 ms (full), 142 / 150 ms (frozen) — within the 2 s gate, so
  latency did not decide it.
Notes: present the negative result with the same weight as a gain would have had; it is what the predeclared rule decided.

## 11. Limitations / 局限

Synthetic questions; agent-only labels; one seed and one learning rate per recipe; one corpus; latency only on the RTX 4090; no user study.

## 12. Next steps / 下一步

Done: all non-GPU preparation, the first formal GPU campaign (BM25 stays the default), the download of its delivery and a CPU demo of
the formal release. Open: SHA-256 check of the frozen and full archives (Codex); the rendered page and screenshots (needs Streamlit); a
second round is an exploratory experiment whose complete evidence is pending delivery — its validation results, if any, will be reported
as such and never replace the first campaign's test numbers. Optional later, not a prerequisite: human spot-check of labels, more seeds, MPS.
