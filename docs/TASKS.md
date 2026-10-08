# Task list / 任务清单（current / 当前）

Consolidated 2026-10-08 from [06 execution checklist](06-execution-checklist.md) (T01–T12), [11 readiness review](11-readiness-and-review.md)
(G01–G10), [14 pre-GPU checklists](14-target-verification-checklists.md) and [the GPU release acceptance](../reports/gpu-release-acceptance.md).
Those documents keep their wording (contracts as reference, dated status sections marked history); this list records the **current** state. Status words:
**done (CPU)** = verified on CPU with evidence; **done** = verified, no device involved;
**reported** = stated by Codex or the owner from the GPU host, receipts not yet inspected; **pending (Codex)** = Codex's next work;
**pending (owner)** = needs a decision by the repository owner; **pending** = not started or unknown. Evidence: [index](../reports/EVIDENCE_INDEX.md).

状态：done（CPU）为 CPU 上已验证；done 为与设备无关且已验证；reported 为 Codex 或所有者报告、回执尚未核对；pending（Codex）为 Codex 待办；pending（owner）需所有者决定；pending 为未开始或未知。

## A. Before the GPU / GPU 之前

| # | Task | Status | Evidence / note |
| --- | --- | --- | --- |
| A1 | Package, configs, CLI commands, CPU fixtures, code identity (T01, G01) | done (CPU) | CI on `577b0ce` and `22db6f4` |
| A2 | Pinned sources, real tokenizer, token-valid corpus (T03, G03) | done | 230 documents, 4,259 passages; real-tokenizer comparison and membership audit |
| A3 | BM25 index, baseline retrieve and evaluate (T04) | done (CPU) | BM25-only validation metrics (synthetic questions) |
| A4 | Labels, groups, splits, pairs, fixed validation candidates, initial test seal (T05, G03) | done | v2 package; initial seal `9a59905c…` |
| A5 | Review bank for all fixed test candidates and known positives (G04) | done | revision 5, audit exit 0, 9,852/9,852 final; agent judgments only |
| A6 | Campaign controller, plan, state, resume, exit codes (G05) | done (CPU) | diagnostic campaigns; crash/resume rehearsals |
| A7 | Hashed locks and offline portable install outside the checkout (G02) | done (CPU) for the CUDA lock | evidence run on `577b0ce`: 85/85 packages, isolation proven, 847 tests 0 skipped |
| A8 | Real MiniLM pinned, verified and usable on CPU | done (CPU) | CPU check 10/10; export/reload parity |
| A9 | Real-model crash and resume through the controller | done (CPU) | `real-model` evidence job: exit 137, resume, attempt 2 |
| A10 | API/UI behavior, fallbacks, release export and reload (T09, G09 on CPU) | done (CPU) | UI demo (diagnostic model; ranking meaningless by design) |
| A11 | CI-receipt import for GPU containers without `unshare -rn` | done (CPU) on `577b0ce` | PR #3; AutoDL preflight with it reported passed (B2) |
| A12 | Durable backup of evidence into the private release: CI/evidence artifacts of `577b0ce` and `e3168ca` (the deployed commit), and every GPU-host receipt | pending (Codex) | artifacts expire 90 days after the runs; `22db6f4`'s copy exists |
| A13 | Test portability on GPU hosts (CUDA-unavailable negative tests, local fixture reachability) | done (CPU) on `e3168ca`: CI 4/4, evidence 2/2, 848 tests 0 skipped | on the host: B6 |

## B. On the GPU host (AutoDL) / GPU 主机（AutoDL）

Order follows `scripts/gpu.py`: `check`, then `prepare` (B5–B9 are its internal steps), `diagnose`, `run`, `status`.

| # | Task | Status | Evidence / note |
| --- | --- | --- | --- |
| B1 | Host and mode: AutoDL, CI-import mode, external host configuration | reported done for `e3168ca` (fresh source and runtime) | GPU_QUICKSTART §0.2; configuration lives on the host |
| B2 | `gpu.py check` | reported passed (`577b0ce`, then `e3168ca`) | receipt not inspected |
| B3 | Private dataset package on the host | reported done: SCP, SHA-256 re-verified | receipt not inspected |
| B4 | 85 dependency wheels and the pinned model downloaded and hash-verified | reported done (reused for `e3168ca`; model via `model.source = "local"`) | receipts not inspected |
| B5 | `prepare` › exact-lock installation and environment inspection | reported passed for `577b0ce` and `e3168ca` | — |
| B6 | `prepare` › implementation receipt: full test suite, 0 failures, 0 skips | `577b0ce`: **stopped, 847 tests, 2 failures, 2 skips** (reported); `e3168ca`: 6 targeted tests passed, **full suite started, result pending (Codex)** | — |
| B7 | `prepare` › model, dataset, tokenizer comparison | pending (Codex) | — |
| B8 | `prepare` › data rebuild, review bank and initial seal reproduced (`06e7ae62…` / `9a59905c…`) | pending (Codex) | — |
| B9 | `prepare` › bundle | pending (Codex) | — |
| B10 | `diagnose`: real MiniLM on CUDA, injected crash and resume, export, reload (T02, G06, G07) | pending (Codex) | CPU version passed (A9) |
| B11 | `run` › doctor calibration: memory, batch size, throughput (G06) | pending (Codex) | — |
| B12 | `run` › training of the frozen and full recipes (seed 42, up to 3 epochs) and validation-only selection against BM25 (T07, G08) | pending (Codex) | no result exists |
| B13 | `run` › sealed test evaluation, paired bootstrap, latency p50/p95 on the device (T10, T11, G10) | pending (Codex) | fields: experiment report §5 |
| B14 | `run` › release export, offline reload, **server backup** with receipt (T12, G10) | pending (Codex) | `backup-receipt.json` |
| B15 | **Local delivery**: model, reports, receipts, logs and state downloaded to the local project directory, SHA-256 manifest re-checked on the Mac; external drive later | pending (Codex) | distinct from B14 |
| B16 | Optional domain-adaptation experiment (T08) | pending | only with evidence from B12 |
| B17 | MPS profile on a Mac | pending (optional) | not blocking |

## C. Decisions and documentation / 决策与文档

| # | Task | Status |
| --- | --- | --- |
| C1 | Deployment commit and external host configuration | reported: `e3168ca` deployed with its own host configuration (Codex) |
| C2 | Merge order of PR #2 / #3 / #4 | pending (owner); all stay unmerged |
| C3 | Server access to the private package | **settled: SCP** (a read-only token stays untested) |
| C4 | Human review of labels | **not a GPU prerequisite** (owner, 2026-10-08); labels stay described as agent judgments |
| C5 | Root README updated with current status and documentation map | done on `codex/claude-docs` (the candidate's README is unchanged) |
| C6 | Fill the experiment report and presentation drafts with GPU results | pending (Codex results) |
| C7 | Consolidated handoff, task list, evidence index, diagrams, report and presentation drafts; dated status sections in older documents marked as history | done on `codex/claude-docs` ([PR #4](https://github.com/Hector-sau/devsearch/pull/4)) |

## D. Explicitly not claimed / 明确不宣称

Any CUDA or MPS run, any reranker quality or latency number, any human judgment, any user study or time saved by people, and any
generalization beyond the `pytorch/tutorials` corpus and its synthetic questions.
