# Task list / 任务清单（current / 当前）

Consolidated 2026-10-08 from [06 execution checklist](06-execution-checklist.md) (T01–T12), [11 readiness review](11-readiness-and-review.md)
(G01–G10), [14 pre-GPU checklists](14-target-verification-checklists.md) and [the GPU release acceptance](../reports/gpu-release-acceptance.md).
Those documents keep their wording (contracts as reference, dated status sections marked history); this list records the **current** state. Status words:
**done (CPU)** = verified on CPU with evidence; **done** = verified, no device involved;
**reported** = stated by Codex or the owner from the GPU host, not read here; **verified here** = read from the extracted evidence files ([evidence index §9](../reports/EVIDENCE_INDEX.md)); **pending (Codex)** = Codex's next work;
**pending (owner)** = needs a decision by the repository owner; **pending** = not started or unknown. Evidence: [index](../reports/EVIDENCE_INDEX.md).

状态：done（CPU）为 CPU 上已验证；done 为与设备无关且已验证；reported 为 Codex 或所有者报告、未经本文核对；verified here 为本文直接读取证据文件核对；pending（Codex）为 Codex 待办；pending（owner）需所有者决定；pending 为未开始或未知。

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

Order follows `scripts/gpu.py`: `check`, then `prepare` (B5–B9 are its internal steps), `diagnose`, `run`, `status`. The first formal
campaign on `e3168ca` is complete.

| # | Task | Status | Evidence / note |
| --- | --- | --- | --- |
| B1 | Host and mode: AutoDL, CI-import mode, external host configuration | done for `e3168ca` (reported) | configuration lives on the host |
| B2 | `gpu.py check` | passed (reported) | — |
| B3 | Private dataset package on the host | done (reported): SCP, SHA-256 re-verified | — |
| B4 | 85 dependency wheels and the pinned model downloaded and hash-verified | done (reported; model via `model.source = "local"`) | — |
| B5 | `prepare` › exact-lock installation and inspection | **done, verified here**: 85/85 locked packages, `pip check` 0 | `state/target-install-check.json` |
| B6 | `prepare` › implementation receipt: full test suite, 0 failures, 0 skips | `577b0ce`: stopped (2 failures, 2 skips; reported); **`e3168ca`: 848 tests, 0 failures, 0 skipped — verified here** | implementation receipt |
| B7 | `prepare` › model, dataset, tokenizer comparison | **done, verified here** | `state/model.json`, `state/dataset.json`, `state/tokenizer.json` |
| B8 | `prepare` › data rebuild, review bank and initial seal | **done, verified here**: initial seal `9a59905c…` | `state/data.json`, `state/review.json`, final report seals |
| B9 | `prepare` › bundle | **done, verified here**: `97f548cd…` | `state/prepare.json` |
| B10 | `diagnose` on CUDA: real MiniLM, injected crash and resume | **passed, verified here**: exit 137, status 0, attempt 2 | `state/diagnose.json` |
| B11 | `run` › doctor calibration | **done, verified here**: microbatch 4; 444.6 / 82.9 training pairs/s (frozen / full) | doctor report |
| B12 | `run` › training of the frozen and full recipes and validation-only selection | **done, verified here: default stays BM25**; neither recipe passed the quality gate | `validation-selection.json` |
| B13 | `run` › sealed test evaluation and latency | **done, verified here**: full fine-tuning not better than BM25 (MRR inconclusive, nDCG worse) | `final-report.json`, `latency.json` |
| B14 | `run` › release export, reload, **server backup** | **done, verified here**: release `release-ce3160fc…` (BM25, no model); server backup verified (497 files) | `campaign-summary.json`, `reload-report.json` |
| B15 | **Download** of the delivery archives to the operator's Mac | **in progress**: evidence archive complete; backup, frozen, full downloading | delivery manifest |
| B16 | **Local SHA-256 check** of every downloaded archive | evidence archive **verified**; three pending until their downloads finish | delivery manifest |
| B17 | Durable copy of the delivery and receipts (private release or external drive) | pending (Codex) | owner: local project directory first, external drive later |
| B18 | Second round of exploratory experiments | **in progress (Codex); no result delivered, no conclusion** | fields: experiment report §5.8 |
| B19 | MPS profile on a Mac | pending (optional) | not blocking |

## C. Decisions and documentation / 决策与文档

| # | Task | Status |
| --- | --- | --- |
| C1 | Deployment commit and external host configuration | reported: `e3168ca` deployed with its own host configuration (Codex) |
| C2 | Merge order of PR #2 / #3 / #4 | pending (owner); all stay unmerged |
| C3 | Server access to the private package | **settled: SCP** (a read-only token stays untested) |
| C4 | Human review of labels | **not a GPU prerequisite** (owner, 2026-10-08); labels stay described as agent judgments |
| C5 | Root README updated with current status and documentation map | done on `codex/claude-docs` (the candidate's README is unchanged) |
| C6 | Fill the experiment report and presentation with the first formal campaign's results | **done** (verified here); second-round fields pending (experiment report §5.8) |
| C7 | Consolidated handoff, task list, evidence index, diagrams, report and presentation drafts; dated status sections in older documents marked as history | done on `codex/claude-docs` ([PR #4](https://github.com/Hector-sau/devsearch/pull/4)) |
| C8 | Code walkthrough, five-minute demo script, implemented-decision notes (docs/04), interview guide — verified against `e3168ca`; demo screenshots and results pending | done on `codex/claude-docs` |

## D. Explicitly not claimed / 明确不宣称

Any CUDA or MPS run, any reranker quality or latency number, any human judgment, any user study or time saved by people, and any
generalization beyond the `pytorch/tutorials` corpus and its synthetic questions.
