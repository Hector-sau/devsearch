# GPU release acceptance / GPU 发布验收

> **2026-10-09 current pointer / 当前入口：** the target preparation history below predates the completed GPU campaigns. See [RESULTS](RESULTS.md), [EVIDENCE_INDEX](EVIDENCE_INDEX.md) and [TASKS](../docs/TASKS.md) for current completion and release work. Original table rows retain their historical scope. / 下表保留原就绪历史，当前完成状态与发布待办见上述入口。

## Restricted-container compatibility change / 受限容器兼容改动

On `codex/gpu-ci-receipts`, `scripts/gpu.py` adds explicit CI portability-evidence import for Linux CUDA hosts that refuse
`unshare -rn`. `scripts/ci_receipts.py` exports/fetches the exact tested wheel, checks authenticated workflow provenance, pins the
artifact and full deployment inventory, and revalidates inputs on resume. The host still checks installed dependencies, runs
zero-skip implementation tests and performs its own real-model CUDA diagnosis/calibration. Imported evidence is attributed to CI,
never to target-host network isolation. Existing local-recording behavior remains the default.

Predecessor `577b0ce21b94e07b8ca54f6f9bd3a1a3a978dfac` passed [all four regular CI jobs](https://github.com/Hector-sau/devsearch/actions/runs/37748100772)
and [both evidence jobs](https://github.com/Hector-sau/devsearch/actions/runs/37748094266): exact-lock receipt **847 tests, zero failures/errors/skips**,
proved Linux CI isolation, and real-model CPU crash/resume. The actual authenticated Mac fetch passed (including Linux errno interpretation
on macOS), and AutoDL preflight accepted that export. All 85 dependency wheels (3,149,538,097 bytes) matched its CI inventory. The target's
fresh installation matched all 85 locked packages and the installed project snapshot; `pip check` exited 0. Locked torch `2.14.1+cu130`
also completed a real CUDA tensor operation on RTX 4090 (ABI sanity only).

**The target full-suite gate then correctly stopped:** 847 tests, two failures caused by external GitHub preflight inside local fixture
workflows, and two unavailable-CUDA negative tests skipped because this host has CUDA. This follow-up makes the local fixture workflow's
reachability preflight a test double, retains a separate network-refusal assertion, and hides CUDA only in the two negative tests' fresh
subprocesses so their actual error assertions execute on GPU hosts. No production network check, training code, model/data pin or readiness
gate changes. Six focused tests passed on AutoDL in 323.612 seconds with no skips, including the complete local workflow. **This follow-up
still needs fresh CI/evidence and the target's entire zero-skip suite before model preparation and real-model CUDA diagnosis proceed.**

本分支的上一提交 `577b0ce` 已通过常规 CI 和两项证据验证，锁定环境 847 项测试零失败、零跳过；Mac 真实导入及 AutoDL 前置检查也通过。
85 个依赖包及实机安装均已核验，锁定 PyTorch 在 RTX 4090 上完成了基础 CUDA 张量运算。随后实机全量测试因两项样例外网依赖失败及
两项“无 CUDA”测试跳过而停止。本次仅修正这些测试的环境假设，六项实机专项测试零跳过通过；仍须在此新版本完成 CI 与实机全量测试。
真实模型 GPU 诊断、正式训练和排序结果尚未完成；下方更早候选的记录仅为历史证据。

## Previous candidate evidence / 上一候选版本证据

Updated 2026-10-08. Branch `claude/youthful-turing-uc1lkv`, PR #2 (unmerged). **Release-candidate identity:** the commit that last changed
code, tests, configuration or CI (`git log -1 --format=%H -- src scripts tests configs locks .github pyproject.toml`); later commits on the
branch change only documents.

**Current release candidate: `22db6f4776a11566bd95797b0420f0b81c985ab1`.** On that exact commit:
- CI [run 37734103569](https://github.com/Hector-sau/devsearch/actions/runs/37734103569): `cpu-baseline` 3.11/3.12 and `ml-cpu` 3.11/3.12 all
  **success** (`ml-cpu`: whole suite and operator-tooling tests with zero skips; wheel build and installed-wheel check in all four jobs).
- Evidence [run 37734098060](https://github.com/Hector-sau/devsearch/actions/runs/37734098060) (`.github/workflows/non-gpu-evidence.yml`), both jobs **success**:
  - `locked-env` (job 113169617034): the exact `locks/nvidia-cuda-cp312.lock` (SHA-256 `596b9dfe…`) downloaded with hashes into an offline
    wheelhouse; portable receipt: new environment outside the checkout, 85/85 locked packages exactly, no unlocked package, `pip check` 0,
    install and campaign inside `unshare -rn` with the isolation **proven** by a positive control (outside: connected; inside: only `lo`, errno 101),
    wheel matches the checkout, diagnostic rehearsal complete; implementation receipt in a second environment installed only from that lock:
    **835 tests, 0 skipped, 0 failures**; installed wheel checked outside the checkout.
  - `real-model` (job 113169616819), same exact-lock environment, devsearch imported from the installed wheel (`site-packages`, outside the
    checkout; required by the check): MiniLM `44acabbec0ef496f6dbc93adadea57f376b7c0ec` fetched and verified (`pytorch_model.bin` 133,483,028 bytes
    `bcc275fe…`), converted, registered (pretrained, pinned), `check_model_cpu.py --require-pretrained` 10/10, then the `gpu.py diagnose` steps
    with these assets: run killed by the fault `mid_stage:train:full:42` (**exit 137**), `run --resume` exit 0, `status` exit 0 and complete,
    `train:full:42` **attempt 2** (every other stage attempt 1), export, reload verification and backup succeeded. CPU torch from the CUDA lock
    (`2.14.1+cu130`), synthetic rehearsal data: plumbing evidence, not CUDA, memory, speed or ranking.
- Receipts: workflow artifacts (90 days) and a durable copy in the private draft release 406041484
  (`devsearch-actions-evidence-20261008-22db6f4.tar.gz`, SHA-256 `ea711a6834543033a2758b317ab2a14dfaa42f76fb48b275cc9a7afffa8d58a5`, downloaded back and verified).
- Locally (macOS, Python 3.12.14, no ML packages installed): 835 tests run, OK, **272 of them skipped** (the ML and API tests); the zero-skip
  results are the CI `ml-cpu` jobs and the exact-lock implementation receipt above, not the local run.

**All non-GPU prerequisites now have evidence; what remains needs the NVIDIA host** (driver/CUDA compatibility, the CUDA environment's receipts
on that host, `diagnose` with MiniLM on the GPU, device memory calibration, the campaign, on-device latency and reload). No ranking result exists.

更新于 2026-10-08。候选发布提交 `22db6f4`：常规 CI 四项成功；证据工作流两项成功（准确 CUDA 依赖锁离线安装并证明隔离、锁内环境 835 项测试零跳过、源码目录外安装包、真实 MiniLM 固定版本校验与 CPU 检查 10/10、故障退出 137 后恢复成功且 `train:full:42` 第 2 次尝试完成）。本机 835 项中 272 项因未装 ML 依赖而跳过，零跳过证据来自 CI 与锁定环境回执。无需 GPU 的前提均已有证据，剩余为 GPU 主机上的检查与正式运行；尚无排序效果结果。

## Handoff status / 交接状态

> **History (PR #2 candidate `22db6f4`).** Current status: the first section of this report and [docs/TASKS.md](../docs/TASKS.md). 历史状态，当前状态见本报告首节及 TASKS。

| Required evidence | Status | Evidence |
| --- | --- | --- |
| **Code accepted** | release candidate `22db6f4` ready; **awaiting your approval** (PR #2 stays unmerged) | CI run 37734103569 and evidence run 37734098060 on the candidate (above) |
| **Real assets verified** | **done (CPU)** | MiniLM `44acabbe…` pinned in `configs/gpu.toml`; CPU check 10/10, export/reload parity, crash and resume through the controller (rows 9, 12, 21) |
| **Data sealed** | **done** | audited v2 `06e7ae62…` with initial seal `9a59905c…`; byte-reproducible; a fresh root rebuilt it from a fresh download (row 22) |
| **Review coverage complete** | **done** | audit exit 0, 9,852/9,852 final under revision 5; agent judgments only (rows 3, 24) |
| **Package retrievable** | **done** | private draft-release asset 620811833, pinned with `dataset.url`/`token_env`; downloaded with `gh release download` and through gpu.py's authenticated request with the owner's token; **a separate read-only token was not tested** |

Only after these does the server work reduce to installation, hardware diagnosis, training, evaluation and export through existing commands.
Genuinely hardware-dependent checks that remain after that: driver/CUDA compatibility, the CUDA environment's receipts on the host, `diagnose`
with MiniLM on the GPU, the doctor's device calibration, the campaign itself, latency on the device, and reload of the exported release on the device.

Columns: **Implemented** = code and documentation exist. **CPU verified** = exercised by tests or recorded commands on CPU. **Real assets** = run
with the real pinned model, the real corpus or the real review bank. **GPU** = run on an NVIDIA host. "unverified" means not run, not failed.
Commits: `72ef5df` isolation, pilot, review workflow; `80e455a` `gpu.py`; `c5d56bb` progress, fetch-model, rehearsal config, UI evidence; `1812191`
`gpu.py cmd`; `0aeb6ed` campaign-lock message, `rehearsal-judge`; `16bfc6d` budget wording; the release candidate (PR description) adds the API judge runner,
the CPU model check, reproducible seals, the package-shipped bank and authenticated package download.

> **Status note 2026-10-08 (documentation branch):** the **GPU** column below records the state before the first formal campaign on `e3168ca`; that campaign has since run on an RTX 4090 (all four gates passed, BM25 stays the default). GPU evidence: [EVIDENCE_INDEX §9](EVIDENCE_INDEX.md). 下表 GPU 列为首轮正式实验前的状态，GPU 证据见证据索引第 9 节。

| # | Requirement | Implemented | CPU verified | Real assets | GPU |
| --- | --- | --- | --- | --- | --- |
| 1 | Network isolation proven by a positive control; a timeout or unreachable host never counts | `72ef5df` | yes: 3 regression tests (`tests/test_scripts.py`), proof ran on this host | n/a | unverified (`gpu.py check` tests `unshare -rn` on the rented host) |
| 2 | Pilot wording: 100% on the 77 jointly resolved items of 80, 3 unresolved | `72ef5df` | yes (`reports/evidence/review-pilot/pilot-report.json`) | real pool sample, agent judges | n/a |
| 3 | Review import, validation, coverage audit, adjudication; missing judgments block sealing and evaluation; two-pass rule (now the `double_primary` mode; see row 24) | `72ef5df` (+ `c5d56bb` progress) | yes: `tests/test_review_workflow.py`, `tests/test_prepare.py`; fixture and clean-clone rehearsal: one pass of two seals nothing, builds no bundle, and `run` is refused | **complete (2026-10-08)**: audit exit 0, 9,852/9,852 final under revision 5 (row 24); merged bank 11,627 responses `1212305a…`; agent judgments only | n/a |
| 4 | Second pilot, minimal context, measured usage | `72ef5df` | measured: 40 items, two runs, 1,738 tokens and 0.85 s per judgment (`reports/evidence/review-pilot/pilot2-compact-report.json`) | real pool sample, agent judges | n/a |
| 5 | Review workflow ready, not launched: blinded batches, two-pass import commands, adjudication, resumable progress, batch sizing by an estimate (not a spending cap), pilot kept apart | `c5d56bb` (`devsearch.review progress`, `docs/15-review-workflow.md`) | yes: progress derived from imports, `--budget-tokens` (an estimate for choosing batches, not a cap) and `--limit`, pilot reviewers refused by `audit` and by directory (tests) | not launched (needs your decision: paid usage) | n/a |
| 6 | One entry point with check / prepare / diagnose / run / status, logs, exit codes, resume, time limit | `80e455a`, `0aeb6ed` | yes: `tests/test_gpu.py` (20 tests); clean-clone rehearsal below | real data rebuilt through it (row 11) | unverified |
| 7 | Check GPU and driver, persistent storage, space, permissions, network | `80e455a` | storage, permissions, network, package checks run; GPU branch not exercised | n/a | **unverified** (driver parsing never ran on a real driver) |
| 8 | Pinned dependencies into a new environment; implementation and portable receipts | `80e455a` | yes: on the candidate, GitHub Linux x86_64, exact nvidia-cuda lock: portable receipt (85/85 locked, offline proven, outside the checkout, rehearsal complete) and implementation receipt (835 tests, 0 skips), source snapshot `5477b4db…` (evidence run 37734098060) | n/a | unverified (CUDA torch on a GPU) |
| 9 | Model at an immutable commit, integrity verified, provenance recorded, `.bin` converted once | `80e455a` | yes against a local fake hub: resolve-once, LFS SHA-256 and git-blob SHA-1 checks, mismatch refusal, conversion with both hashes, credential refusal named | **yes**: `44acabbec0ef496f6dbc93adadea57f376b7c0ec` fetched with `gpu.py fetch-model` from a host that reaches the hub (2026-10-07/08); `pytorch_model.bin` 133,483,028 bytes `bcc275fe…`, converted and registered; pinned in `configs/gpu.toml` | unverified |
| 10 | Remote-download procedure when the server cannot reach the model host | `c5d56bb` (`gpu.py fetch-model`, quickstart section 10) | yes against the fake hub: verified fetch, manifest, tamper and wrong-commit refusal, conversion on the server | **yes**: the procedure was used for the real model (row 9); receipts in the private checkpoints | n/a |
| 11 | Tokenizer configuration and encodings compared; data rebuilt; frozen membership audited before sealing | `80e455a` | yes (fixture: a cased tokenizer stops with `protocol-revision`) | **yes with the stand-in tokenizer**: real corpus rebuilt from `pytorch/tutorials@11512db7` in three clean runs; every frozen value reproduced (corpus `corpus-4367f673…`, 197 test questions `22818c5c…`, groups `aabc29bb…`, 9,852 review items, training files, pairs); package byte-reproducible (`3d361709…`). **With the real pinned tokenizer (2026-10-08): audited v2 rebuilt twice and once more in a fresh root from a fresh download (sources re-fetched): byte-identical `06e7ae62…`; the 14 frozen membership values equal v1's** | n/a |
| 12 | Small real-model training, inference, recovery check before formal training | `80e455a` (`diagnose`, `rehearsal --assets`) | yes with a stand-in encoder: crash injected in `train:full:42`, resumed, status complete | **yes on CPU** (evidence run 37734098060, `scripts/check_real_model_resume.py`): pinned MiniLM, fault exit 137, resume 0, status complete, `train:full:42` attempt 2, export/reload/backup succeeded; synthetic data | **unverified** |
| 13 | Matrix, validation-only selection, sealed test evaluation, export, reload verification, backup | earlier commits + `80e455a` (`run`) | yes as a diagnostic run on CPU; refusals tested (no pretrained assets, no receipts, no diagnosis) | **unverified** (needs the review bank) | **unverified** |
| 14 | Never change model, dataset, precision or backend silently | `80e455a` (`state/frozen.json`) | yes: a changed dataset hash or model kind is refused (exit 1) | n/a | n/a |
| 15 | Missing credentials or review artifacts stop at a named prerequisite, no fake output | `80e455a`, `0aeb6ed` | yes: `review-bank`, `prepare`, `diagnose`, `dataset-package`, `protocol-revision`, `model-credentials`, `campaign-lock`; scripted judgments refused on a GPU profile | real data stops at `review-bank` | n/a |
| 16 | Installed wheel outside the checkout and a full diagnostic workflow from small fixtures | `16bfc6d` | yes: CI step "Verify the installed wheel outside the checkout" in all four jobs; fixture workflow `tests/test_gpu.py`; on the candidate also from the exact-lock environment (portable receipt, real-model check imports the installed package and fails otherwise) | n/a | n/a |
| 17 | **Clean-environment rehearsal of the documented procedure** (fresh clone, fresh root, quickstart commands) | `0aeb6ed` | **yes, CPU diagnostic**, transcript `reports/evidence/clean-rehearsal.md`: check, prepare stops at `review-bank`, one pass blocks sealing and `run`, two scripted passes, prepare, diagnose (crash 137, resume), `run` hard-killed (all processes) after training started, stale lock taken over, resume to complete, status 0. Found and fixed four gaps | real data, stand-in encoder, scripted judgments | not run |
| 18 | User-facing API and UI, baseline and diagnostic model | earlier commits | **yes, real browser**, `reports/evidence/ui-demo/`: search with source links to the pinned commit, empty result, rejected requests (422), rerank asked of a baseline release degrades with a warning, deadline fallback to BM25, API down. The model is a random encoder: ranking meaningless by design | real corpus index | not run |
| 20 | Real review bank through a platform with a hard spend limit, protocol unchanged, resumable, pilot kept apart | RC (`scripts/judge_api.py`, docs/16) | yes against a fake Batches API: refusal without confirmation or key, retries of refused/errored/incomplete requests never filled, budget guard stops before a chunk, resume without resending, rerun with other settings refused; outputs import through `devsearch.review` | **not run: needs budget approval and an API key** | n/a |
| 21 | Real model usable on CPU before rental (hashes, loading, tokenizer, forward/backward) | RC (`scripts/check_model_cpu.py`, run by `prepare`) | yes on fixture and stand-in assets (a changed file fails; diagnostic assets never pass `--require-pretrained`) | **yes**: `check_model_cpu.py --require-pretrained` 10/10 on the real assets (hashes, pinned, load 33,212,545 parameters, tokenizer, vocabulary, forward, backward, update, eval determinism, frozen encoder) plus export and separate-process reload parity (max diff 0.0); receipts in private checkpoint cp02 | n/a |
| 22 | Sealed data and bank shipped in a private, server-downloadable package; fresh environment reproduces the seal | RC (`gpu.py package --review-responses`, `dataset.token_env`) | yes: fresh root at another path seals with the package's frozen seal hash without any review step; authenticated download with refusal of missing or wrong tokens and of a hash mismatch | **yes**: `devsearch-dataset-v2.tar.gz` (1,645,198 bytes, `06e7ae62c3c78cc14d06407005dc91117d869cd73b8e28c45491744e9c488b1f`, initial seal `9a59905c…`) uploaded as private draft-release asset 620811833; downloaded with `gh release download` and with gpu.py's authenticated `dataset.url` request (owner token; a separate read-only token not tested); fresh-root rebuild from the download reproduced the bytes and the seal | n/a |
| 23 | Paid runner: resumable state bound to the review pool, every batch file and every request's parameters; outputs verified by hash before reuse; submission intent persisted before the create call, uncertain submissions reconciled, a stop (exit 7) when duplicate-free recovery cannot be shown | RC (`scripts/judge_api.py`) | yes against the fake Batches API, no real calls: changed pool / batch file / parameters refused before sending; missing or corrupt output restored from the recorded results without a new request, unrestorable or foreign files stop; a crash after remote creation before the local commit is adopted without resubmitting; a lost create answer is adopted in the same run; a create that never reached the API is resent once after the window; several candidates or a mismatched adoption stop for the operator (`tests/tooling/test_judge_api.py`, 21 tests) | not run (no paid usage authorized) | n/a |
| 24 | Review protocol revision 5 (`primary_audit_v1`): complete primary pass, seeded stratified second review (label × rank band, rates 0.25/0.08, minimum 30, question coverage, seed 20261007), escalation over 10% disagreement, independent adjudication; declared in `data.toml` and hashed into every seal | RC (`devsearch.data.review_protocol`; audit, bank stage, `gpu.py`, `devsearch.review batches --queue second-review`) | yes: `tests/test_review_protocol.py` (no sampling before the primary pass is complete; strata, minimums and question coverage; seed determinism; escalation at 3 of 28 but not 2 of 28; adjudicator independence; an adjudication never replaces a missing primary judgment, an incomplete primary pass, a required (sampled or escalated) second review or the escalation decision, in the audit and at sealing; the bank seals only with the sample second-reviewed; CLI audit and second-review batches) | **run on the frozen pool**: sample 924 (incl. question coverage, 197/197 questions); unrounded disagreement over 10% in all four positive strata (18.56%, 32.26%, 59.38%, 51.52%) → 553 escalated second reviews, 1,477 in all; negative strata ≤2.04% not escalated; 284 adjudications (13 left unsure by the first adjudicator went to a second, the last one to a third independent adjudicator; none forced) | n/a |
| 19 | MPS | profile and lock exist | not run | not run | **unverified, optional, not blocking** |

**Private artifacts preserved (2026-10-07, before any session change):** `private-preserve-20261007.tar.gz`, SHA-256
`27b45140bc4b0a9792a1fda13971b2956bee04f06ae4778de1733a732298e4b7` (3,311,783 bytes, 336 entries), delivered to you as a chat attachment
and kept in the runtime root. It holds what cannot be regenerated from the repository or upstream sources: the package source with
the private review salt and protocol revision 5 (`pkg-src/`), dataset package v1 and its checksum, the question drafts and generation
work (`real/drafts`, `real/qwork*`), the pilot judgments (`real/review-pilot*`) and the data, source and index configurations. Keep it private.

## What I need from you, in order / 需要您做的事（按顺序）

> **History (PR #2 period).** Since then the private package went to the GPU host by SCP; the deployment commit and host configuration are tracked in [docs/HANDOFF.md](../docs/HANDOFF.md). 已过时，见 HANDOFF。

1. **Review and approve the release candidate** `22db6f4` (PR #2; it stays unmerged until you decide).
2. **Before renting:** decide how the server gets the private package: copy it (`gh release download … --pattern 'devsearch-dataset-v2.*'`, then
   `scp`), or create a token that can read this private repository's releases and export it as `DEVSEARCH_PACKAGE_TOKEN` (test it once first;
   only the owner's token was tested). Then check the pre-rental list (`docs/GPU_QUICKSTART.md` section 0.1: driver 580+, 16 GiB+, persistent
   60 GiB, Python 3.12, `unshare -rn`, network during `prepare`).
3. **On the GPU host:** `gpu.py check → prepare → diagnose → run → status` (quickstart section 2). Optional, later: a Mac for MPS.

1. 审阅并批准候选提交 `22db6f4`（PR #2 保持未合并直至您决定）。2. 租机前决定私有数据包的取得方式（下载后 scp，或创建可读取本私有仓库发布附件的令牌并先测试；目前只测试过所有者令牌），并核对租机前检查表。3. 在 GPU 主机依次运行 check → prepare → diagnose → run → status。
