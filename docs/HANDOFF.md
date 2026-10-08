# DevSearch handoff / 交接（current / 当前）

Consolidated 2026-10-08 on the documentation branch `codex/claude-docs` (base `577b0ce`). This page is the **current** handoff. Older
handoffs keep their contracts as reference; only their dated status sections are history ([05](05-machine-handoff.md), [08](08-ai-developer-handoff.md),
[11](11-readiness-and-review.md); each says which sections). The repository [README](../README.md) on this branch carries the short status and
the documentation map. Every number below has a source in [the evidence index](../reports/EVIDENCE_INDEX.md); facts reported by the owner
from the GPU host are marked *reported*, with their receipts pending from Codex.

本页为当前交接；旧交接文档的契约仍为参考，仅日期状态段落为历史。GPU 主机上的事项标为“所有者报告”，回执待 Codex 提供。

## 1. What DevSearch is / 项目是什么

An English technical-documentation search application: BM25 retrieves up to 50 candidates from pinned documentation snapshots, and a
locally fine-tuned MiniLM cross-encoder (PyTorch) can rerank them; the default mode is chosen on validation data only. Results are evidence
passages with source links; no generated answers. The first corpus is `pytorch/tutorials@11512db7…` (230 documents, 4,259 passages).
Architecture: [00](00-architecture-and-roadmap.md) and [diagrams](DIAGRAMS.md).

## 2. Current status / 当前状态（2026-10-08）

| Area | State | Evidence |
| --- | --- | --- |
| Code | **GPU candidate `577b0ce` (decided 2026-10-08), frozen**: PR #3, stacked on PR #2 (`22db6f4`); both PRs open, unmerged | CI 37748100772 and evidence 37748094266 on `577b0ce`, all green |
| Dependencies | Exact CUDA lock installs offline; whole suite 847 tests, 0 skipped in that environment (on `577b0ce`) | evidence run 37748094266 |
| Real model | MiniLM pinned at `44acabbe…`; CPU check 10/10; crash (exit 137) and resume through the controller on CPU | cp02 receipts; evidence runs |
| Data | Real corpus with the real pinned tokenizer; 197 frozen test questions; 9,852 review items | v2 package |
| Review bank | Complete under protocol revision 5: audit exit 0, 9,852/9,852 final, 542 positives; **agent judgments only, no human review** | cp04; bank manifest |
| Package | Audited v2 (`06e7ae62…`, seal `9a59905c…`) in the private draft release; pinned in `configs/gpu.toml` | release 406041484 |
| GPU host (AutoDL) | *Reported:* private dataset package transferred by SCP; preflight passed; all 85 dependency wheels and the pinned model downloaded and hash-verified. **Pending from Codex:** environment installation and receipts, `prepare` completion, `diagnose` on the GPU, calibration, training, test evaluation, on-device latency | owner report 2026-10-08; receipts pending from Codex |
| Ranking quality | **No reranker result exists.** The only ranking numbers are BM25 on synthetic validation questions | experiment report draft |
| MPS | Profile and lock exist; never run; optional | — |

GPU 候选为 `577b0ce`；无需 GPU 的前提均已有证据；AutoDL 上数据包已 SCP 传输、预检查通过、依赖包与模型已下载校验（所有者报告）；安装、诊断、训练与评测待 Codex；尚无重排效果结果；所有标注均为 Agent 判定，人工审核不是 GPU 前置条件。

## 3. Branches and pull requests / 分支与 PR

| Branch | Head | PR | Role |
| --- | --- | --- | --- |
| `main` | `30927a4` | — | default branch; does **not** contain the implementation |
| `claude/youthful-turing-uc1lkv` | `3d2a846` (earlier candidate `22db6f4`) | #2 open | implementation, GPU release tooling, pins |
| `codex/gpu-ci-receipts` | `577b0ce` (**frozen GPU candidate**) | #3 open (base: PR #2's branch) | CI portability-evidence import for restricted containers |
| `codex/claude-docs` | this branch | [#4](https://github.com/Hector-sau/devsearch/pull/4) draft, open (base: `codex/gpu-ci-receipts`) | documentation only: README, this handoff, task list, evidence index, diagrams, report and presentation drafts |
| `codex/accelerator-readiness-review` | `f533990` | #1 merged (2026-10-05) into `claude/youthful-turing-uc1lkv`, not into `main` | history: readiness review |
| `codex/non-gpu-handoff-20261008` | `5753cf8` | — | history: 2026-10-08 non-GPU handoff note |

No PR is merged by an AI. Deploy a reviewed **code** commit, never this documentation branch (the CI-import mode checks every tracked file).

## 4. How to continue on a GPU host / 在 GPU 主机上继续

Follow [GPU_QUICKSTART](GPU_QUICKSTART.md) sections 0.1 (pre-rental checklist), 0.2 (CI-receipt import for hosts without `unshare -rn`)
and 2 (commands). On AutoDL the steps up to the downloads are reported done (§2); the remaining ones are Codex's:

```bash
git checkout 577b0ce21b94e07b8ca54f6f9bd3a1a3a978dfac   # the frozen GPU candidate; never this docs branch
python3 scripts/gpu.py check    --config configs/gpu.toml
python3 scripts/gpu.py prepare  --config configs/gpu.toml
python3 scripts/gpu.py diagnose --config configs/gpu.toml
python3 scripts/gpu.py run      --config configs/gpu.toml   # after a time-limit stop: --extend-hours 6
python3 scripts/gpu.py status   --config configs/gpu.toml
```

The dataset package came from the private release by SCP (`gh release download devsearch-runtime-checkpoint-20261007 --repo Hector-sau/devsearch
--pattern 'devsearch-dataset-v2.*'`, verify the `.sha256`, `scp` to `<root>/inbox/`); `dataset.url` with a read-only token stays untested.
Copy `campaign/run`, `backup/`, `receipts/`, `logs/` and `state/` off the server before the rental ends.

## 5. Restoring the review state / 恢复审查状态

Download cp01, cp02 and cp04 from the private release, verify each archive and its `MANIFEST.sha256`, then follow cp04's `RESTORE.md`
(cp01 formal-review → overlay cp02 → overlay cp04). Expected: `devsearch.review audit` exit 0, final 9,852, merged responses SHA-256
`1212305a…`. cp03 is superseded. Never concatenate or re-import response files.

## 6. Rules that still apply / 仍须遵守

- Agent judgments stay labelled `agent`; nothing is described as a human test. Simulated questions stay labelled synthetic. Additional
  human review is not a prerequisite for the GPU run.
- Frozen data, review protocol, thresholds, seeds and model pin are not changed for nicer results; a change is a recorded protocol revision.
- Test data is scored only inside the campaign, after validation-only selection.
- No paid API, no hardware rental and no PR merge without the owner's decision. Private data (salt, test questions, labels, corpus text)
  stays out of Git and in the private release.
- "BM25 stays the default" is a valid, reportable outcome.

## 7. Open items / 待办与待确认

| # | Item | Owner |
| --- | --- | --- |
| 1 | Report installation, `prepare`, `diagnose`, calibration and training results from AutoDL, with receipts | Codex |
| 2 | Durable backup of evidence artifacts in the private release: the `577b0ce` CI/evidence artifacts (expire 90 days after the runs) and the GPU-host receipts | Codex |
| 3 | Merge order of PR #2 / #3 / #4 (all stay unmerged until the owner decides) | owner |
| 4 | If PR #4 is ever merged into the candidate branch, the README and docs change the code identity (`gpu.py` code key, wheel long description, CI-import inventory): a new candidate would need fresh CI and evidence runs | owner |

Settled on 2026-10-08: GPU candidate `577b0ce`; package transfer by SCP; human review is not a GPU prerequisite; README updated on this branch.
