# GPU release acceptance / GPU 发布验收

Date 2026-10-06. Branch `claude/youthful-turing-uc1lkv`. **Final code commit: `0aeb6ed`** (the last commit that changes code, tests or
configuration; every later commit changes only this report, evidence and documentation). Full ML CPU CI on that commit, zero skips:
GitHub Actions run [37452797304](https://github.com/Hector-sau/devsearch/actions/runs/37452797304), `workflow_dispatch` on the branch, four jobs
(`cpu-baseline` and `ml-cpu`, each on Python 3.11 and 3.12): all four jobs succeeded, including `Run the whole suite and fail on any skip` (ml-cpu 3.11 and 3.12), `Build wheel` and `Verify the installed wheel outside the checkout`. The same suite also ran locally on `0aeb6ed` with the ML and serve extras
installed: 808 tests, OK.

This is engineering evidence from one Linux container without a GPU. **The project is not labelled GPU-ready.** Nothing below ran on a GPU, with the
real MiniLM files, or with a judged review bank, and no ranking gain or human test is claimed.

日期 2026-10-06。最终代码提交 `0aeb6ed`（此后提交只改本报告、证据与文档）。该提交上的完整 ML CPU CI（零跳过）见上；本地同一提交含 ML 与服务依赖共 808 项测试通过。**项目未标记为 GPU 就绪**：以下内容均未在 GPU、真实 MiniLM 文件或已判定的审查库上运行，不声称任何排序提升或真人测试。

Columns: **Implemented** = code and documentation exist. **CPU verified** = exercised by tests or recorded commands on CPU. **Real assets** = run
with the real pinned model, the real corpus or the real review bank. **GPU** = run on an NVIDIA host. "unverified" means not run, not failed.
Commits: `72ef5df` isolation, pilot, review workflow; `80e455a` `gpu.py`; `c5d56bb` progress, fetch-model, rehearsal config, UI evidence; `1812191`
`gpu.py cmd`; `0aeb6ed` campaign-lock message, `rehearsal-judge`.

| # | Requirement | Implemented | CPU verified | Real assets | GPU |
| --- | --- | --- | --- | --- | --- |
| 1 | Network isolation proven by a positive control; a timeout or unreachable host never counts | `72ef5df` | yes: 3 regression tests (`tests/test_scripts.py`), proof ran on this host | n/a | unverified (`gpu.py check` tests `unshare -rn` on the rented host) |
| 2 | Pilot wording: 100% on the 77 jointly resolved items of 80, 3 unresolved | `72ef5df` | yes (`reports/evidence/review-pilot/pilot-report.json`) | real pool sample, agent judges | n/a |
| 3 | Review import, validation, coverage audit, adjudication; missing judgments block sealing and evaluation; two-pass rule | `72ef5df` (+ `c5d56bb` progress) | yes: `tests/test_review_workflow.py`, `tests/test_prepare.py`; fixture and clean-clone rehearsal: one pass of two seals nothing, builds no bundle, and `run` is refused | batches built for the real pool (9,852 items, two passes, 247 batches each); **coverage 0%** | n/a |
| 4 | Second pilot, minimal context, measured usage | `72ef5df` | measured: 40 items, two runs, 1,738 tokens and 0.85 s per judgment (`reports/evidence/review-pilot/pilot2-compact-report.json`) | real pool sample, agent judges | n/a |
| 5 | Review workflow ready, not launched: blinded batches, two-pass import commands, adjudication, resumable progress, usage limits, pilot kept apart | `c5d56bb` (`devsearch.review progress`, `docs/15-review-workflow.md`) | yes: progress derived from imports, `--budget-tokens`/`--limit`, pilot reviewers refused by `audit` and by directory (tests) | not launched (needs your decision: paid usage) | n/a |
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
| 16 | Installed wheel outside the checkout and a full diagnostic workflow from small fixtures | `0aeb6ed` | yes: CI step "Verify the installed wheel outside the checkout" in all four jobs; fixture workflow `tests/test_gpu.py` | n/a | n/a |
| 17 | **Clean-environment rehearsal of the documented procedure** (fresh clone, fresh root, quickstart commands) | `0aeb6ed` | **yes, CPU diagnostic**, transcript `reports/evidence/clean-rehearsal.md`: check, prepare stops at `review-bank`, one pass blocks sealing and `run`, two scripted passes, prepare, diagnose (crash 137, resume), `run` hard-killed (all processes) after training started, stale lock taken over, resume to complete, status 0. Found and fixed four gaps | real data, stand-in encoder, scripted judgments | not run |
| 18 | User-facing API and UI, baseline and diagnostic model | earlier commits | **yes, real browser**, `reports/evidence/ui-demo/`: search with source links to the pinned commit, empty result, rejected requests (422), rerank asked of a baseline release degrades with a warning, deadline fallback to BM25, API down. The model is a random encoder: ranking meaningless by design | real corpus index | not run |
| 19 | MPS | profile and lock exist | not run | not run | **unverified, optional, not blocking** |

## Remaining user actions and blockers / 剩余事项与阻塞项

1. **Review bank (decision; paid usage).** 9,852 items need two primary judgments each plus adjudication. Measured agent cost with compact context:
   1,738 tokens per judgment; two passes are about 19,700 judgments, roughly 34 million tokens before adjudication (**extrapolated, not run**).
   Choose agent judges, human judges, or a mix with a human spot-check (`docs/15-review-workflow.md`). Until then the real run is blocked at
   `PREREQUISITE review-bank` by design.
2. **Rent the CUDA host** (not done, not allowed here): `docs/GPU_QUICKSTART.md` section 1 (driver 580+ for CUDA 13, 16 GiB GPU memory, Python 3.12
   with venv, 60 GiB persistent storage, unprivileged user namespaces, HTTPS to PyPI, Hugging Face and GitHub).
3. **Give the server read access to the private repository** (deploy key or token; quickstart section 2) and **copy the dataset package**
   (`devsearch-dataset-v1.tar.gz`, SHA-256 `3d36170959a1932b6d222bd363a80dbc6a270f4c797d67e1c606c1340bd8d903`, delivered with the PR, not in Git).
4. **Model access (blocked here).** `huggingface.co` and its file hosts are unreachable from this container (HTTP 000 on every host tried); the
   environment's network allow-list would have to change. The server downloads the model itself; if it cannot, run `gpu.py fetch-model` on any
   connected machine (quickstart section 10). The first real download is also the first test of the hub-hash checks.
5. Run `check`, `prepare`, import judgments, `diagnose`, `run` as in the quickstart, then bring back the summary. Optional, later: a Mac for MPS.

No further access is needed from this side for the CPU-verifiable work.

1. 审查库（需您决定，涉及付费用量）：9,852 项各两份主判定并裁决；按精简上下文实测每条约 1,738 token，两轮约 3,400 万（外推，未运行）。2. 租 CUDA 主机（此处未做，也不被允许）。3. 为服务器提供私有仓库只读权限并复制数据包。4. 模型访问在本容器仍被阻止（所有 Hugging Face 域名无法连接）；服务器自行下载，下载不了时用 `gpu.py fetch-model` 在任意联网机器完成。5. 按快速上手运行并带回结果；MPS 之后再说。
