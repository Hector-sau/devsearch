# Task list / 任务清单（current / 当前）

Consolidated 2026-10-08 from [06 execution checklist](06-execution-checklist.md) (T01–T12), [11 readiness review](11-readiness-and-review.md)
(G01–G10), [14 pre-GPU checklists](14-target-verification-checklists.md) and [the GPU release acceptance](../reports/gpu-release-acceptance.md).
Those documents keep their wording (contracts as reference, dated status sections marked history); this list records the **current** state. Status words:
**done (CPU)** = verified on CPU with evidence; **done** = verified, no device involved;
**reported** = stated by the owner from the GPU host, receipts not yet in the repository or release; **pending (Codex)** = Codex's next work;
**pending (owner)** = needs a decision by the repository owner; **pending** = not started or unknown. Evidence: [index](../reports/EVIDENCE_INDEX.md).

状态：done（CPU）为 CPU 上已验证；done 为与设备无关且已验证；pending（GPU）需 GPU 主机；pending（owner）需所有者决定；pending 为未开始或未知。

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
| A12 | Durable backup of evidence: the `577b0ce` CI/evidence artifacts and every GPU-host receipt, into the private release | pending (Codex) | artifacts expire 90 days after the runs; `22db6f4`'s copy exists |

## B. On the GPU host / GPU 主机上

| # | Task | Status | Evidence / note |
| --- | --- | --- | --- |
| B1 | Host chosen and reachable: AutoDL; candidate `577b0ce` (CI-import mode, GPU_QUICKSTART §0.2) | done (owner decision, 2026-10-08) | — |
| B2 | Preflight on AutoDL | reported passed | receipt pending (Codex) |
| B3a | Private dataset package to the host | reported done: transferred by SCP | hash check on the host pending (Codex) |
| B3b | All 85 dependency wheels and the pinned model downloaded and hash-verified | reported done | receipts pending (Codex) |
| B3c | Rest of `gpu.py prepare`: environment installation and its receipts, tokenizer comparison, data rebuild, bank and seal reproduced (`06e7ae62…` / `9a59905c…`) | pending (Codex) | — |
| B4 | `gpu.py diagnose`: real MiniLM on CUDA, injected crash and resume, export, reload, backup (T02, G06, G07) | pending (Codex) | CPU version passed (A9); CUDA unverified |
| B5 | Doctor calibration: memory, batch size, throughput (G06) | pending (Codex) | — |
| B6 | Campaign: BM25 vs frozen encoder vs full fine-tuning, seed 42, up to 3 epochs, validation-only selection (T07, G08) | pending (Codex) | no result exists |
| B7 | Sealed test evaluation with shared qrels, paired bootstrap, latency p50/p95 on the device (T10, T11, G10) | pending (Codex) | — |
| B8 | Release export, offline reload, verified backup copied off the server (T12, G10) | pending (Codex) | — |
| B9 | Optional domain-adaptation experiment (T08) | pending | only with evidence from B6 |
| B10 | MPS profile on a Mac | pending (optional) | not blocking |

## C. Decisions and documentation / 决策与文档

| # | Task | Status |
| --- | --- | --- |
| C1 | GPU candidate | **settled 2026-10-08: `577b0ce`** (frozen; not modified) |
| C2 | Merge order of PR #2 / #3 / #4 | pending (owner); all stay unmerged |
| C3 | Server access to the private package | **settled: SCP** (a read-only token stays untested and is not needed) |
| C4 | Human review of labels | **not a GPU prerequisite** (owner, 2026-10-08); labels stay described as agent judgments |
| C5 | Root README updated with current status and documentation map | done on `codex/claude-docs` (the candidate's README is unchanged) |
| C6 | Fill the experiment report and presentation drafts with GPU results | pending (Codex results) |
| C7 | Consolidated handoff, task list, evidence index, diagrams, report and presentation drafts; dated status sections in older documents marked as history | done on `codex/claude-docs` ([PR #4](https://github.com/Hector-sau/devsearch/pull/4)) |

## D. Explicitly not claimed / 明确不宣称

Any CUDA or MPS run, any reranker quality or latency number, any human judgment, any user study or time saved by people, and any
generalization beyond the `pytorch/tutorials` corpus and its synthetic questions.
