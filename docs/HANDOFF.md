# DevSearch handoff / 交接（current / 当前）

Consolidated 2026-10-08 on the documentation branch `codex/claude-docs` (created from `577b0ce`, later merged with PR #3's head `e3168ca`). This page is the **current** handoff. Older
handoffs keep their contracts as reference; only their dated status sections are history ([05](05-machine-handoff.md), [08](08-ai-developer-handoff.md),
[11](11-readiness-and-review.md); each says which sections). The repository [README](../README.md) on this branch carries the short status and
the documentation map. Every number below has a source in [the evidence index](../reports/EVIDENCE_INDEX.md); facts reported by the owner
or Codex from the GPU host are marked *reported* until their receipts are inspected.

本页为当前交接；旧交接文档的契约仍为参考，仅日期状态段落为历史。所有者或 Codex 报告的 GPU 主机事项标为“报告”；本文直接读取证据文件核对的标为“verified here”。

## 1. What DevSearch is / 项目是什么

An English technical-documentation search application: BM25 retrieves up to 50 candidates from pinned documentation snapshots, and a
locally fine-tuned MiniLM cross-encoder (PyTorch) can rerank them; the default mode is chosen on validation data only. Results are evidence
passages with source links; no generated answers. The first corpus is `pytorch/tutorials@11512db7…` (230 documents, 4,259 passages).
Architecture: [00](00-architecture-and-roadmap.md) and [diagrams](DIAGRAMS.md).

## 2. Current status / 当前状态（2026-10-08）

### Versions / 版本

| Role | Commit | State | Source |
| --- | --- | --- | --- |
| Documentation branch | head of `codex/claude-docs` ([PR #4](https://github.com/Hector-sau/devsearch/pull/4)) | documentation only; never deployed | this PR |
| Previous verified commit | `577b0ce21b94e07b8ca54f6f9bd3a1a3a978dfac` | CI 4/4 and evidence 2/2: exact lock 85/85, offline isolation proven, 847 tests 0 skipped, real-model CPU crash/resume | runs 37748100772, 37748094266 (receipts read) |
| Current correction | `e3168ca6fef9b64ea0688ea059332c729125e30f` (PR #3 head) | changes tests only (CUDA-unavailable negative tests run in a CUDA-hidden subprocess; local fixtures stub external reachability, with a separate refusal test) plus docs; CI 4/4 and evidence 2/2 green (848 tests, 0 skipped on CI hosts); target full suite inside `prepare` on the RTX 4090: **848 tests, 0 failures, 0 skipped** (verified here) | runs 37760709410, 37760704700 (receipts read) |
| Deployed on AutoDL | `e3168ca` | deployment reported by Codex; **first formal campaign complete, verified here**: default stays BM25 | [evidence index §9](../reports/EVIDENCE_INDEX.md) |

### Execution on AutoDL / AutoDL 执行情况

Rows saying **verified here** were read from the extracted evidence files ([evidence index §9](../reports/EVIDENCE_INDEX.md)); the others are
as reported by Codex. Server paths and access details stay in Codex's handoff, not in this repository.

| Step | Commit | Reported state |
| --- | --- | --- |
| Dataset package | — | transferred by SCP; SHA-256 `06e7ae62…` re-verified on the host |
| `gpu.py check` (CI-import mode) | `577b0ce`, then `e3168ca` | passed (network to GitHub through the provider's documented accelerator) |
| Downloads | — | 85 dependency wheels, 3,149,538,097 bytes, each SHA-256 equal to the CI inventory (re-used for `e3168ca`); pinned MiniLM, all hub hashes verified |
| Model route | — | after Hugging Face TLS timeouts, the documented `model.source = "local"` route points to the downloaded pinned model with its fetch manifest; repo, commit and bytes unchanged |
| First run | `577b0ce` | exact-lock installation passed; inside `prepare` the full implementation suite stopped the gate: **847 tests, 2 failures, 2 skips**; runtime kept unchanged as failed-run evidence |
| Correction on the host | `e3168ca` | 6 targeted tests passed, 0 skipped (323.6 s) |
| CI import for `e3168ca` | `e3168ca` | authenticated on the Mac and transferred; origin SHA-256 `030a9039…`, CI wheel `f8281f60…` |
| Second run | `e3168ca` | fresh source and runtime; installation passed; full implementation suite inside `prepare`: **848 tests, 0 failures, 0 skipped (verified here)** |
| Rest of `prepare`, `diagnose` (crash 137, resume), calibration, campaign, server backup | `e3168ca` | **completed, verified here**: BM25 stays the default; full fine-tuning not better than BM25 on test; server backup verified |
| Download to the Mac / local SHA-256 | `e3168ca` | download **complete**; SHA-256 verified here for the evidence and backup archives; frozen and full not checked here |
| Second round | — | exploratory experiment; complete evidence pending delivery (Codex) |

### Order of work inside the stages / 阶段内部顺序

`prepare` itself runs: prerequisite check → environment installation (exact lock, offline wheelhouse) and inspection → **implementation
receipt = the full test suite in that environment, zero failures and zero skips required** (in CI-import mode the portable receipt is the
imported one) → model → dataset → tokenizer comparison → data rebuild → review bank and initial seal → bundle. Then `diagnose` (real
MiniLM on CUDA, injected crash, resume), then `run` (doctor calibration, training, validation-only selection, sealed test evaluation,
latency, export, reload, **server backup**), then `status`. Source: `stage_prepare` and `step_receipts` in `scripts/gpu.py`.

`prepare` 内部依次为：前置检查 → 锁定环境安装与检查 → 全量实现测试（零失败、零跳过）→ 模型 → 数据包 → 分词器对比 → 数据重建 → 审查库与封存 → 运行包；随后 `diagnose`、`run`（含服务器备份）与 `status`。

### Other areas / 其他

| Area | State | Evidence |
| --- | --- | --- |
| Real model | MiniLM `44acabbe…`; CPU check 10/10; crash (exit 137) and resume on CPU | cp02 receipts; evidence runs |
| Data | real corpus, real tokenizer; 197 frozen test questions; 9,852 review items | v2 package |
| Review bank | complete under protocol revision 5: audit exit 0, 9,852/9,852 final, 542 positives; **agent judgments** | cp04; bank manifest |
| Package | audited v2 `06e7ae62…`, initial seal `9a59905c…`; private draft release; on AutoDL via SCP | release 406041484 |
| Ranking quality | first formal campaign: **BM25 stays the default**; full fine-tuning not better than BM25 on the sealed test set (MRR inconclusive, nDCG worse); frozen far worse | [experiment report §5](../reports/experiment-report-draft.md) |
| MPS | profile and lock exist; never run; optional | — |

此前验证版本 `577b0ce`；当前修正版 `e3168ca` 已部署。第一轮正式 GPU 流程已完成（本文直接核对证据文件）：`prepare` 内全量测试 848 项零失败零跳过，`diagnose` 通过，按验证集预选默认仍为 BM25，完整微调在测试集上未优于 BM25；服务器备份已校验，下载已完成，本地 SHA-256 已核对证据包与备份包。第二轮为探索实验，完整证据待交付。

## 3. Branches and pull requests / 分支与 PR

| Branch | Head | PR | Role |
| --- | --- | --- | --- |
| `main` | `30927a4` | — | default branch; does **not** contain the implementation |
| `claude/youthful-turing-uc1lkv` | `3d2a846` (earlier candidate `22db6f4`) | #2 open | implementation, GPU release tooling, pins |
| `codex/gpu-ci-receipts` | `e3168ca` (previously `577b0ce`) | #3 open (base: PR #2's branch) | CI-receipt import for restricted containers; test-portability correction |
| `codex/claude-docs` | this branch | [#4](https://github.com/Hector-sau/devsearch/pull/4) draft, open (base: `codex/gpu-ci-receipts`) | documentation only |
| `codex/accelerator-readiness-review` | `f533990` | #1 merged (2026-10-05) into `claude/youthful-turing-uc1lkv`, not into `main` | history: readiness review |
| `codex/non-gpu-handoff-20261008` | `5753cf8` | — | history: 2026-10-08 non-GPU handoff note |

No PR is merged by an AI. Never deploy this documentation branch: the CI-import mode compares every Git-tracked file, documents included.

## 4. How to continue on AutoDL / 在 AutoDL 上继续

AutoDL refuses `unshare -rn`, so it runs in the CI-import mode of [GPU_QUICKSTART §0.2](GPU_QUICKSTART.md) with a verified **external
host configuration** (receipt source `ci`, the transferred CI import directory and its pinned SHA-256, the runtime root, `model.source =
"local"`), not the repository's `configs/gpu.toml` as shipped. Codex reports that this configuration and its launcher exist on the host for
`e3168ca`; they are not in this repository, so no runnable command is given here. The first formal campaign has run through all stages; further runs are new,
separately declared experiments (Codex).

Results leave the rental in three separate states, each recorded on its own:

| State | What it is | Evidence | First campaign |
| --- | --- | --- | --- |
| Server backup | the campaign's `backup` stage copies outputs to `<runtime root>/backup` (same volume unless `backup.dir` names another) | `backup-receipt.json`; `campaign-summary.json` → `backup.status` | **verified** (497 files, 549,641,178 bytes) |
| Download | the delivery archives copied to the local project directory on the operator's Mac (external drive later) | archive sizes equal the delivery manifest | **complete** (all four archives) |
| Local SHA-256 check | every downloaded archive's SHA-256 recomputed on the Mac and compared with the delivery manifest | matching hashes | **evidence and backup archives verified here**; frozen and full not checked here |

A verified server backup is not a delivery: it disappears with the rental.

AutoDL 采用 CI 证据导入模式及外部主机配置（不在本仓库，故不给出命令）。第一轮正式实验已跑完全部阶段。结果离开租用机分三种状态分别记录：服务器备份（已校验）、下载到本机（已完成）、本地 SHA-256 核对（证据包与备份包已核对）；服务器备份不等于交付。

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
| 1 | SHA-256 check of the frozen and full archives against the delivery manifest (downloads complete; evidence and backup archives already verified) | Codex |
| 2 | Durable copy of the delivery and receipts (private release or external drive), plus the `577b0ce` and `e3168ca` CI/evidence artifacts | Codex |
| 3 | Second round: the fields in [experiment report §5.8](../reports/experiment-report-draft.md) (configuration, best checkpoint, validation metrics, whether an independent test set exists, latency, export/reload, local backup check) | Codex |
| 4 | Merge order of PR #2 / #3 / #4 (all stay unmerged until the owner decides) | owner |

Settled on 2026-10-08: package transfer by SCP; deployment of `e3168ca`; first formal campaign complete with BM25 as the default (negative
result kept); human review is not a GPU prerequisite; labels stay agent judgments; results go to the local project directory first.
