# DevSearch handoff / 交接（current / 当前）

Consolidated 2026-10-08 on the documentation branch `codex/claude-docs` (created from `577b0ce`, later merged with PR #3's head `e3168ca`). This page is the **current** handoff. Older
handoffs keep their contracts as reference; only their dated status sections are history ([05](05-machine-handoff.md), [08](08-ai-developer-handoff.md),
[11](11-readiness-and-review.md); each says which sections). The repository [README](../README.md) on this branch carries the short status and
the documentation map. Every number below has a source in [the evidence index](../reports/EVIDENCE_INDEX.md); facts reported by the owner
or Codex from the GPU host are marked *reported* until their receipts are inspected.

本页为当前交接；旧交接文档的契约仍为参考，仅日期状态段落为历史。所有者或 Codex 报告的 GPU 主机事项在核对回执前标为“报告”。

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
| Current correction | `e3168ca6fef9b64ea0688ea059332c729125e30f` (PR #3 head) | changes tests only (CUDA-unavailable negative tests run in a CUDA-hidden subprocess; local fixtures stub external reachability, with a separate refusal test) plus docs; CI 4/4 and evidence 2/2 green (848 tests, 0 skipped on CI hosts); target full suite pending | runs 37760709410, 37760704700 (receipts read) |
| Deployed on AutoDL | **pending from Codex**: the final deployment commit and the verified external host configuration | — | — |

### Execution on AutoDL, reported by Codex / AutoDL 执行情况（Codex 报告）

Not yet checked against receipts; recorded as reported.

| Step | Reported state |
| --- | --- |
| Dataset package | transferred by SCP |
| Preflight (CI-import mode, using `577b0ce`'s CI export) | passed |
| Downloads | all 85 dependency wheels (3,149,538,097 bytes) matched the CI inventory; pinned model downloaded and hash-verified |
| Installation of `577b0ce` | succeeded: 85/85 locked packages and the installed project snapshot matched, `pip check` 0; locked torch `2.14.1+cu130` ran a CUDA tensor operation on an RTX 4090 (ABI sanity only) |
| Full test suite on the host | **stopped the gate: 847 tests, 2 failures** (external GitHub preflight inside local fixture workflows) **and 2 skips** (CUDA-unavailable negative tests on a CUDA host) |
| Correction `e3168ca` on the host | 6 targeted tests passed, 0 skipped (323.6 s) |
| Full suite on the final commit, `prepare`, `diagnose`, calibration, training, evaluation | **not done** |

### Other areas / 其他

| Area | State | Evidence |
| --- | --- | --- |
| Real model | MiniLM `44acabbe…`; CPU check 10/10; crash (exit 137) and resume on CPU | cp02 receipts; evidence runs |
| Data | real corpus, real tokenizer; 197 frozen test questions; 9,852 review items | v2 package |
| Review bank | complete under protocol revision 5: audit exit 0, 9,852/9,852 final, 542 positives; **agent judgments** | cp04; bank manifest |
| Package | audited v2 `06e7ae62…`, initial seal `9a59905c…`; private draft release; on AutoDL via SCP | release 406041484 |
| Ranking quality | **no reranker result**; only BM25 on synthetic validation questions | experiment report draft |
| MPS | profile and lock exist; never run; optional | — |

此前验证版本 `577b0ce`；当前修正版 `e3168ca` 仅改测试；实际部署版本与外部主机配置待 Codex 提供。AutoDL 上（Codex 报告，尚未核对回执）：数据包已 SCP，预检查通过，依赖与模型校验通过，`577b0ce` 安装成功；全量测试 2 项失败、2 项跳过；修正版 6 项针对性测试通过。GPU 诊断与训练均未进行。

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

AutoDL refuses `unshare -rn`, so it runs in the CI-import mode of [GPU_QUICKSTART §0.2](GPU_QUICKSTART.md): the host uses a verified
**external host configuration** (receipt source `ci`, the transferred CI import directory and its pinned SHA-256, the runtime root), not the
repository's `configs/gpu.toml` as shipped. **The final deployment commit and that configuration are pending from Codex**, so no runnable
command is given here. Once Codex supplies them, the stage order stays `check → prepare → diagnose → run → status` (GPU_QUICKSTART §2), and
for that commit: regular CI and both evidence jobs green, a fresh CI export, then the target's full suite with zero failures and zero skips
before `prepare` and the real-model GPU diagnosis. Copy `campaign/run`, `backup/`, `receipts/`, `logs/` and `state/` off the host before the
rental ends.

AutoDL 不支持 `unshare -rn`，采用 CI 证据导入模式，使用已校验的外部主机配置；最终部署提交与该配置待 Codex 提供，此处不给出可执行命令。

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
| 1 | Final deployment commit and the verified external host configuration for AutoDL | Codex |
| 2 | For that commit: regular CI and both evidence jobs, a fresh CI export, the target's full suite with 0 failures and 0 skips | Codex |
| 3 | `prepare`, `diagnose` (real MiniLM on CUDA), calibration, training, evaluation on AutoDL, with receipts | Codex |
| 4 | Durable backup in the private release: CI/evidence artifacts of `577b0ce`, `e3168ca` and the deployed commit (they expire 90 days after the runs) and every GPU-host receipt | Codex |
| 5 | Make the AutoDL receipts available so the *reported* facts in §2 can be checked | Codex |
| 6 | Merge order of PR #2 / #3 / #4 (all stay unmerged until the owner decides) | owner |

Settled on 2026-10-08: package transfer by SCP; human review is not a GPU prerequisite; labels stay agent judgments.
