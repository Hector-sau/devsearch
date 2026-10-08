# DevSearch handoff / 交接（current / 当前）

Consolidated 2026-10-08 on the documentation branch `codex/claude-docs` (base `577b0ce`). This page is the **current** handoff. Older
handoffs are kept unchanged as history: [05 machine handoff](05-machine-handoff.md) (Mac-only era), [08 AI developer handoff](08-ai-developer-handoff.md)
(implementation era), [11 readiness review](11-readiness-and-review.md) (gap list of 2026-10-05). The repository [README](../README.md) still
describes 2026-10-06 and is deliberately not edited here: it is part of the release's code identity (see "Open questions").
Every number below has a source in [the evidence index](../reports/EVIDENCE_INDEX.md); nothing without one is claimed.

本页为当前交接；旧交接文档保留为历史。仓库 README 仍描述 2026-10-06 状态，因其属于发布代码身份，本分支不修改。

## 1. What DevSearch is / 项目是什么

An English technical-documentation search application: BM25 retrieves up to 50 candidates from pinned documentation snapshots, and a
locally fine-tuned MiniLM cross-encoder (PyTorch) can rerank them; the default mode is chosen on validation data only. Results are evidence
passages with source links; no generated answers. The first corpus is `pytorch/tutorials@11512db7…` (230 documents, 4,259 passages).
Architecture: [00](00-architecture-and-roadmap.md) and [diagrams](DIAGRAMS.md).

## 2. Current status / 当前状态（2026-10-08）

| Area | State | Evidence |
| --- | --- | --- |
| Code | Implemented and CPU-verified. Latest candidate `577b0ce` (PR #3, stacked on PR #2 at `22db6f4`); both PRs open, unmerged | CI and evidence runs on both commits, all green |
| Dependencies | Exact CUDA lock installs offline; whole suite 847 tests, 0 skipped in that environment (on `577b0ce`) | evidence run 37748094266 |
| Real model | MiniLM pinned at `44acabbe…`; CPU check 10/10; crash (exit 137) and resume through the controller on CPU | cp02 receipts; evidence runs |
| Data | Real corpus with the real pinned tokenizer; 197 frozen test questions; 9,852 review items | v2 package |
| Review bank | Complete under protocol revision 5: audit exit 0, 9,852/9,852 final, 542 positives; **agent judgments only, no human review** | cp04; bank manifest |
| Package | Audited v2 (`06e7ae62…`, seal `9a59905c…`) in the private draft release; pinned in `configs/gpu.toml` | release 406041484 |
| GPU | **Not run.** No CUDA diagnosis, calibration, training, test evaluation or on-device latency | — |
| Ranking quality | **No reranker result exists.** The only ranking numbers are BM25 on synthetic validation questions | experiment report draft |
| MPS | Profile and lock exist; never run; optional | — |

无需 GPU 的前提均已有证据；GPU 上的诊断、训练、评测与延迟均未运行；尚无重排效果结果；所有标注均为 Agent 判定，无真人审查。

## 3. Branches and pull requests / 分支与 PR

| Branch | Head | PR | Role |
| --- | --- | --- | --- |
| `main` | `30927a4` | — | default branch; does **not** contain the implementation |
| `claude/youthful-turing-uc1lkv` | `3d2a846` (code candidate `22db6f4`) | #2 open | implementation, GPU release tooling, pins |
| `codex/gpu-ci-receipts` | `577b0ce` | #3 open (base: PR #2's branch) | CI portability-evidence import for restricted containers |
| `codex/claude-docs` | this branch | draft PR (base: `codex/gpu-ci-receipts`) | documentation only: this handoff, task list, evidence index, report and presentation drafts |
| `codex/accelerator-readiness-review` | `f533990` | #1 merged (2026-10-05) into `claude/youthful-turing-uc1lkv`, not into `main` | history: readiness review |
| `codex/non-gpu-handoff-20261008` | `5753cf8` | — | history: 2026-10-08 non-GPU handoff note |

No PR is merged by an AI. Deploy a reviewed **code** commit, never this documentation branch (the CI-import mode checks every tracked file).

## 4. How to continue on a GPU host / 在 GPU 主机上继续

Follow [GPU_QUICKSTART](GPU_QUICKSTART.md) sections 0.1 (pre-rental checklist), 0.2 (CI-receipt import, only for hosts without
`unshare -rn`, available on `577b0ce`) and 2 (commands):

```bash
git checkout <the approved code commit>           # 577b0ce or 22db6f4, see Open questions
python3 scripts/gpu.py check    --config configs/gpu.toml
python3 scripts/gpu.py prepare  --config configs/gpu.toml
python3 scripts/gpu.py diagnose --config configs/gpu.toml
python3 scripts/gpu.py run      --config configs/gpu.toml   # after a time-limit stop: --extend-hours 6
python3 scripts/gpu.py status   --config configs/gpu.toml
```

The dataset package comes from the private release (`gh release download devsearch-runtime-checkpoint-20261007 --repo Hector-sau/devsearch
--pattern 'devsearch-dataset-v2.*'`, verify the `.sha256`, copy to `<root>/inbox/`), or through `dataset.url` with `DEVSEARCH_PACKAGE_TOKEN`.
Copy `campaign/run`, `backup/`, `receipts/`, `logs/` and `state/` off the server before the rental ends.

## 5. Restoring the review state / 恢复审查状态

Download cp01, cp02 and cp04 from the private release, verify each archive and its `MANIFEST.sha256`, then follow cp04's `RESTORE.md`
(cp01 formal-review → overlay cp02 → overlay cp04). Expected: `devsearch.review audit` exit 0, final 9,852, merged responses SHA-256
`1212305a…`. cp03 is superseded. Never concatenate or re-import response files.

## 6. Rules that still apply / 仍须遵守

- Agent judgments stay labelled `agent`; nothing is described as a human test. Simulated questions stay labelled synthetic.
- Frozen data, review protocol, thresholds, seeds and model pin are not changed for nicer results; a change is a recorded protocol revision.
- Test data is scored only inside the campaign, after validation-only selection.
- No paid API, no hardware rental and no PR merge without the owner's decision. Private data (salt, test questions, labels, corpus text)
  stays out of Git and in the private release.
- "BM25 stays the default" is a valid, reportable outcome.

## 7. Open questions for the owner / 待确认问题

1. Which code commit is deployed: `577b0ce` (needed if the GPU container refuses `unshare -rn`) or `22db6f4`? Merge order of PR #2/#3?
2. The repository README still describes 2026-10-06. Updating it changes `gpu.py`'s code key and the wheel (README is the package's long
   description), so it needs a new code-candidate commit with fresh CI and evidence runs. When should that happen?
3. Server access to the private package: copy by `scp`, or a read-only token (not yet tested)?
4. Should the `577b0ce` evidence artifacts (expire 90 days after 2026-10-08) get a durable copy in the private release like `22db6f4`'s?
5. Is any human review of labels planned? Until then every relevance claim rests on agent judgments.
