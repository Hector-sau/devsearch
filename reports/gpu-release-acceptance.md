# GPU release acceptance / GPU 发布验收

Updated 2026-10-07. Branch `claude/youthful-turing-uc1lkv`, PR #2 (unmerged). **Release-candidate identity:** the commit that last changed
code, tests, configuration or CI (`git log -1 --format=%H -- src scripts tests configs locks .github pyproject.toml`); later commits on the
branch may change only documents. The exact hash and its CI run are recorded in the PR #2 description, which is updated without new commits.

This is engineering evidence from Linux containers without a GPU. **The project is not GPU-ready, and the remaining work is not "only the
GPU":** two prerequisites that need no GPU are still open (the real model, blocked by network access; the review bank, waiting for budget
approval), and the real run package depends on both.

更新于 2026-10-07。候选发布提交为最后一次修改代码、测试、配置或 CI 的提交，准确哈希与 CI 运行记录在 PR #2 描述中（不为更新提交号再建文档提交）。**项目尚未达到 GPU 就绪，也不是“只差 GPU”**：真实模型（网络被阻止）与审查库（待批准预算）两个无需 GPU 的前提仍未完成，正式运行包依赖这两项。

## Handoff status / 交接状态

| Required evidence | Status | Evidence |
| --- | --- | --- |
| **Code accepted** | release candidate ready; **awaiting your approval** | PR #2 head; CI on the candidate (PR description): `cpu-baseline` and `ml-cpu` on Python 3.11 and 3.12, the whole suite with zero skips, the operator-tooling tests with zero skips, wheel build and installed-wheel check; no open review threads |
| **Real assets verified** | **blocked (no GPU needed)** | Hugging Face hosts return no connection from both of your cloud environments (`Pytroch`, `小程序`). Needed change: environment network access **Custom**, add `huggingface.co`, `cas-bridge.xethub.hf.co`, `cdn-lfs.huggingface.co`, `cdn-lfs-us-1.hf.co` (keep the default package-manager list). Then `gpu.py fetch-model` and `scripts/check_model_cpu.py --require-pretrained` run here; both are tested against a fake hub and the stand-in |
| **Data sealed** | **not yet**: depends on the two rows above and below | rebuild with the stand-in tokenizer reproduces every frozen value; seals are now reproducible across hosts (a fresh root at another path reproduced the package's initial-seal hash in `tests/test_gpu.py`) |
| **Review coverage complete** | **not started: budget approval needed** | 0 of 9,852 items. Proposal with platform, model, billing basis, estimate (~$39 central, Opus 5.5 batch), platform-enforced hard cap and steps: `docs/16-review-execution-proposal.md`; runner `scripts/judge_api.py` tested against a fake Batches API |
| **Package retrievable** | mechanism done and tested; **the real package waits on the rows above** | `gpu.py package --review-responses` ships the audited bank and freezes manifest and seal hashes; `dataset.url` + `dataset.token_env` fetch from a private release asset with a read-only token, SHA-256 pinned (tests with a local authenticated server); the current v1 package (`3d361709…`, no bank) was delivered and retrieved in a clean-clone rehearsal |

Only after these does the server work reduce to installation, hardware diagnosis, training, evaluation and export through existing commands.
Genuinely hardware-dependent checks that remain after that: driver/CUDA compatibility, the CUDA environment's receipts on the host, `diagnose`
with MiniLM on the GPU, the doctor's device calibration, the campaign itself, latency on the device, and reload of the exported release on the device.

Columns: **Implemented** = code and documentation exist. **CPU verified** = exercised by tests or recorded commands on CPU. **Real assets** = run
with the real pinned model, the real corpus or the real review bank. **GPU** = run on an NVIDIA host. "unverified" means not run, not failed.
Commits: `72ef5df` isolation, pilot, review workflow; `80e455a` `gpu.py`; `c5d56bb` progress, fetch-model, rehearsal config, UI evidence; `1812191`
`gpu.py cmd`; `0aeb6ed` campaign-lock message, `rehearsal-judge`; `16bfc6d` budget wording; the release candidate (PR description) adds the API judge runner,
the CPU model check, reproducible seals, the package-shipped bank and authenticated package download.

| # | Requirement | Implemented | CPU verified | Real assets | GPU |
| --- | --- | --- | --- | --- | --- |
| 1 | Network isolation proven by a positive control; a timeout or unreachable host never counts | `72ef5df` | yes: 3 regression tests (`tests/test_scripts.py`), proof ran on this host | n/a | unverified (`gpu.py check` tests `unshare -rn` on the rented host) |
| 2 | Pilot wording: 100% on the 77 jointly resolved items of 80, 3 unresolved | `72ef5df` | yes (`reports/evidence/review-pilot/pilot-report.json`) | real pool sample, agent judges | n/a |
| 3 | Review import, validation, coverage audit, adjudication; missing judgments block sealing and evaluation; two-pass rule | `72ef5df` (+ `c5d56bb` progress) | yes: `tests/test_review_workflow.py`, `tests/test_prepare.py`; fixture and clean-clone rehearsal: one pass of two seals nothing, builds no bundle, and `run` is refused | batches built for the real pool (9,852 items, two passes, 247 batches each); **coverage 0%** | n/a |
| 4 | Second pilot, minimal context, measured usage | `72ef5df` | measured: 40 items, two runs, 1,738 tokens and 0.85 s per judgment (`reports/evidence/review-pilot/pilot2-compact-report.json`) | real pool sample, agent judges | n/a |
| 5 | Review workflow ready, not launched: blinded batches, two-pass import commands, adjudication, resumable progress, batch sizing by an estimate (not a spending cap), pilot kept apart | `c5d56bb` (`devsearch.review progress`, `docs/15-review-workflow.md`) | yes: progress derived from imports, `--budget-tokens` (an estimate for choosing batches, not a cap) and `--limit`, pilot reviewers refused by `audit` and by directory (tests) | not launched (needs your decision: paid usage) | n/a |
| 6 | One entry point with check / prepare / diagnose / run / status, logs, exit codes, resume, time limit | `80e455a`, `0aeb6ed` | yes: `tests/test_gpu.py` (20 tests); clean-clone rehearsal below | real data rebuilt through it (row 11) | unverified |
| 7 | Check GPU and driver, persistent storage, space, permissions, network | `80e455a` | storage, permissions, network, package checks run; GPU branch not exercised | n/a | **unverified** (driver parsing never ran on a real driver) |
| 8 | Pinned dependencies into a new environment; implementation and portable receipts | `80e455a` | yes, in this container: wheelhouse verified, offline install, 797 tests 0 skips, portable receipt (offline proven, rehearsal complete) at source snapshot `6cf3fe89…` (commit `80e455a` era; `src/` has changed since, so a server records its own) | n/a | unverified (CUDA torch on a GPU) |
| 9 | Model at an immutable commit, integrity verified, provenance recorded, `.bin` converted once | `80e455a` | yes against a local fake hub: resolve-once, LFS SHA-256 and git-blob SHA-1 checks, mismatch refusal, conversion with both hashes, credential refusal named | **unverified** (`huggingface.co` blocked here; no real MiniLM file seen) | unverified |
| 10 | Remote-download procedure when the server cannot reach the model host | `c5d56bb` (`gpu.py fetch-model`, quickstart section 10) | yes against the fake hub: verified fetch, manifest, tamper and wrong-commit refusal, conversion on the server | **blocked** (host unreachable from this container) | n/a |
| 11 | Tokenizer configuration and encodings compared; data rebuilt; frozen membership audited before sealing | `80e455a` | yes (fixture: a cased tokenizer stops with `protocol-revision`) | **yes with the stand-in tokenizer**: real corpus rebuilt from `pytorch/tutorials@11512db7` in three clean runs; every frozen value reproduced (corpus `corpus-4367f673…`, 197 test questions `22818c5c…`, groups `aabc29bb…`, 9,852 review items, training files, pairs); package byte-reproducible (`3d361709…`). **Real MiniLM unverified** | n/a |
| 12 | Small real-model training, inference, recovery check before formal training | `80e455a` (`diagnose`, `rehearsal --assets`) | yes with a stand-in encoder: crash injected in `train:full:42`, resumed, status complete | **unverified** with MiniLM | **unverified** |
| 13 | Matrix, validation-only selection, sealed test evaluation, export, reload verification, backup | earlier commits + `80e455a` (`run`) | yes as a diagnostic run on CPU; refusals tested (no pretrained assets, no receipts, no diagnosis) | **unverified** (needs the review bank) | **unverified** |
| 14 | Never change model, dataset, precision or backend silently | `80e455a` (`state/frozen.json`) | yes: a changed dataset hash or model kind is refused (exit 1) | n/a | n/a |
| 15 | Missing credentials or review artifacts stop at a named prerequisite, no fake output | `80e455a`, `0aeb6ed` | yes: `review-bank`, `prepare`, `diagnose`, `dataset-package`, `protocol-revision`, `model-credentials`, `campaign-lock`; scripted judgments refused on a GPU profile | real data stops at `review-bank` | n/a |
| 16 | Installed wheel outside the checkout and a full diagnostic workflow from small fixtures | `16bfc6d` | yes: CI step "Verify the installed wheel outside the checkout" in all four jobs; fixture workflow `tests/test_gpu.py` | n/a | n/a |
| 17 | **Clean-environment rehearsal of the documented procedure** (fresh clone, fresh root, quickstart commands) | `0aeb6ed` | **yes, CPU diagnostic**, transcript `reports/evidence/clean-rehearsal.md`: check, prepare stops at `review-bank`, one pass blocks sealing and `run`, two scripted passes, prepare, diagnose (crash 137, resume), `run` hard-killed (all processes) after training started, stale lock taken over, resume to complete, status 0. Found and fixed four gaps | real data, stand-in encoder, scripted judgments | not run |
| 18 | User-facing API and UI, baseline and diagnostic model | earlier commits | **yes, real browser**, `reports/evidence/ui-demo/`: search with source links to the pinned commit, empty result, rejected requests (422), rerank asked of a baseline release degrades with a warning, deadline fallback to BM25, API down. The model is a random encoder: ranking meaningless by design | real corpus index | not run |
| 20 | Real review bank through a platform with a hard spend limit, protocol unchanged, resumable, pilot kept apart | RC (`scripts/judge_api.py`, docs/16) | yes against a fake Batches API: refusal without confirmation or key, retries of refused/errored/incomplete requests never filled, budget guard stops before a chunk, resume without resending, rerun with other settings refused; outputs import through `devsearch.review` | **not run: needs budget approval and an API key** | n/a |
| 21 | Real model usable on CPU before rental (hashes, loading, tokenizer, forward/backward) | RC (`scripts/check_model_cpu.py`, run by `prepare`) | yes on fixture and stand-in assets (a changed file fails; diagnostic assets never pass `--require-pretrained`) | **blocked: model not obtainable here** | n/a |
| 22 | Sealed data and bank shipped in a private, server-downloadable package; fresh environment reproduces the seal | RC (`gpu.py package --review-responses`, `dataset.token_env`) | yes: fresh root at another path seals with the package's frozen seal hash without any review step; authenticated download with refusal of missing or wrong tokens and of a hash mismatch | waits on rows 20-21 | n/a |
| 19 | MPS | profile and lock exist | not run | not run | **unverified, optional, not blocking** |

## What I need from you, in order / 需要您做的事（按顺序）

1. **Network access (no cost).** In the `Pytroch` environment's settings: Network access **Custom**, add `huggingface.co`,
   `cas-bridge.xethub.hf.co`, `cdn-lfs.huggingface.co`, `cdn-lfs-us-1.hf.co`, keeping the default package managers. A new session then
   downloads and verifies MiniLM in the runtime directory (nothing reaches your Mac), runs the CPU check and the tokenizer comparison, and
   confirms whether the frozen pool (review request `e872f940…`) is reproduced. Steps: https://code.claude.com/docs/en/cloud-environments#network-access
2. **Budget approval (paid).** Approve an amount and the judge model for the review bank (docs/16: proposed hard cap $100 on a dedicated
   Console workspace, `claude-opus-5-5`, central estimate ~$39, or `claude-sonnet-5-5` at about half). Then add that workspace's key as the
   `ANTHROPIC_API_KEY` environment variable in the environment settings (never in chat). The judging starts with a calibration of under $2.
3. **After both:** I build package v2 with the sealed bank, you upload it once as a private release asset (or copy it by `scp`), and the
   handoff is complete.
4. **Then rent the CUDA host** using the pre-rental checklist (`docs/GPU_QUICKSTART.md` section 0.1). Optional, later: a Mac for MPS.

1. 网络（无费用）：在 `Pytroch` 环境设置中把网络访问设为 Custom，加入上列 Hugging Face 域名。2. 预算（付费）：批准审查库金额与模型（建议硬上限 100 美元，`claude-opus-5-5`，中位估计约 39 美元；或约半价的 `claude-sonnet-5-5`），并在环境设置中以 `ANTHROPIC_API_KEY` 提供专用工作区密钥（不要在聊天中粘贴）。3. 两项完成后我生成带封存审查库的 v2 数据包，由您以私有 release 附件上传或 `scp` 复制。4. 最后按租机前检查表租用 CUDA 主机；MPS 以后再说。
