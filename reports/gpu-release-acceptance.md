# GPU release acceptance / GPU 发布验收

Date 2026-10-06. Branch `claude/youthful-turing-uc1lkv`. Code under test: commit `{CODE}` (the last commit that changes code, tests or
configuration; later commits change only this report and the evidence). CPU CI with the ML dependencies and zero skips: {CI}.
This is engineering evidence from one Linux container without a GPU. **The project is not labelled GPU-ready:** nothing below ran on a
GPU, with the real MiniLM files, or with a judged review bank, and no ranking gain or human test is claimed.

日期 2026-10-06。被测代码为提交 `{CODE}`（其后提交只改本报告与证据）。CPU CI（含 ML 依赖、零跳过）见上。这是在无 GPU 的单个 Linux 容器中得到的工程证据；**项目未标记为 GPU 就绪**：以下内容均未在 GPU、真实 MiniLM 文件或已判定的审查库上运行，不声称任何排序提升或真人测试。

Columns: **Implemented** = code and documentation exist (commit). **CPU verified** = exercised by recorded tests or commands on CPU
(commit, evidence). **Real assets** = run with the real pinned model, the real corpus or the real review bank. **GPU** = run on an
NVIDIA host. "unverified" means it has not been run, not that it failed.

| # | Requirement | Implemented | CPU verified | Real assets | GPU |
| --- | --- | --- | --- | --- | --- |
| 1 | Network isolation proven by a positive control; a timeout or unreachable host never counts | `72ef5df` (`scripts/record_receipt.py`) | yes, `{CODE}`: 3 regression tests (`tests/test_scripts.py` `PortableReceiptTests`); proven on this host (control connects, isolated process sees only `lo`, `ENETUNREACH`) | n/a | unverified (`unshare -rn` on the rented host is checked by `gpu.py check`) |
| 2 | Pilot report wording: 100% on the 77 jointly resolved items of 80, 3 unresolved | `72ef5df` | yes (`reports/evidence/review-pilot/pilot-report.json` `overall`) | real pool sample, agent judges | n/a |
| 3 | Review import, validation, coverage audit, adjudication queue; missing judgments block sealing and evaluation; two-pass rule enforced | `72ef5df` (`devsearch.review`, `data/review_workflow.py`, `[review] min_primary_reviews`) | yes, `{CODE}`: `tests/test_review_workflow.py`, `tests/test_prepare.py`; the fixture workflow stops at one pass and seals after two (`tests/test_gpu.py`) | batches built for the real pool (9,852 items, two passes); **coverage 0%**, no judgment | n/a |
| 4 | Second, bounded pilot with minimal context and measured usage | `72ef5df` | measured: 40 items, two runs, 1,738 tokens and 0.85 s per judgment (`reports/evidence/review-pilot/pilot2-compact-report.json`) | real pool sample, agent judges | n/a |
| 5 | One entry point with separate stages, logs, exit codes, resumability, time limit, process-group stop | `{CODE}` (`scripts/gpu.py`, `configs/gpu.toml`, `docs/GPU_QUICKSTART.md`) | yes: `tests/test_gpu.py` (11 tests: whole fixture workflow, prerequisites, frozen identity, timeout, deadline extension) | partly: real data (row 9) | unverified |
| 6 | Check GPU and driver, persistent storage, space, permissions, network | `{CODE}` (`gpu.py check`) | storage, permissions, network and package checks run here; GPU branch not exercised (no `nvidia-smi`) | n/a | **unverified** (driver parsing never ran on a real driver) |
| 7 | Pinned dependencies installed into a new environment | `{CODE}` (`gpu.py prepare` step `environment`) | {LOCKENV} | n/a | unverified (CUDA torch import on a GPU) |
| 8 | Implementation and portable receipts on the host | `{CODE}` (step `receipts`) | {RECEIPTS} | n/a | unverified |
| 9 | Model at an immutable commit, integrity verified, provenance recorded, `.bin` converted once | `{CODE}` (step `model`) | yes, against a local fake hub: resolve-once, LFS SHA-256 and git-blob SHA-1 checks, mismatch refusal with nothing registered, conversion with both hashes, refused credential named (`tests/test_gpu.py` `DownloadTests`) | **unverified**: `huggingface.co` is blocked in this container; no real MiniLM file seen | unverified |
| 10 | Real tokenizer configuration and encodings compared; a difference stops before sealing | `{CODE}` (step `tokenizer`) | yes: a cased fixture tokenizer stops with `protocol-revision` and builds nothing | stand-in vocabulary only; **real MiniLM unverified** | n/a |
| 11 | Data rebuilt and frozen test membership audited before sealing | `{CODE}` (steps `dataset`, `data`; `gpu.py package`) | yes (fixture) | **yes with the stand-in tokenizer**: the server path rebuilt the real corpus from `pytorch/tutorials@11512db7` and reproduced every frozen value (corpus `corpus-4367f673…`, 197 test questions `22818c5c…`, test groups `aabc29bb…`, 9,852 review items, training files); the dataset package is byte-reproducible (`3d361709…`) | n/a |
| 12 | Small real-model training, inference and recovery check before formal training | `{CODE}` (`gpu.py diagnose`; `rehearsal --assets`) | yes with the fixture encoder: crash injected in `train:full:42`, resumed, status complete, export, reload, backup | **unverified** with MiniLM | **unverified** |
| 13 | Matrix, validation-only selection, sealed test evaluation, export, reload verification, backup | earlier commits (campaign) + `{CODE}` (`gpu.py run`) | yes as a diagnostic run (fixture, `cpu-diagnostic`); formal refusals tested (no pretrained assets, no receipts, no diagnosis) | **unverified** (needs the review bank) | **unverified** |
| 14 | Never change model, dataset, precision or backend silently | `{CODE}` (`state/frozen.json`) | yes: a changed dataset hash or model kind is refused with exit 1 | n/a | n/a |
| 15 | Missing credentials or review artifacts stop at a named prerequisite, no fake outputs | `{CODE}` | yes: `review-bank`, `prepare`, `diagnose`, `dataset-package`, `protocol-revision`, `model-credentials` | real data stops at `review-bank` | n/a |
| 16 | Installed wheel outside the checkout; full diagnostic workflow from small fixtures | `{CODE}` | {WHEEL} | n/a | n/a |
| 17 | MPS | profile and lock exist | not run | not run | **unverified, optional, not blocking** |

## Remaining user actions / 需要您完成的事项

1. **Review bank (decision, paid usage).** 9,852 items need two primary judgments each plus adjudication. Measured agent cost with
   compact context: about 1,738 tokens per judgment; two passes are about 19,700 judgments, roughly 34 million tokens plus adjudication
   (extrapolated). Decide: agent judges (paid usage), human judges, or a mix with a human spot-check. Nothing has been launched.
2. **Rent the CUDA host** meeting `docs/GPU_QUICKSTART.md` section 1 (driver 580+ for CUDA 13, 16 GiB GPU memory, Python 3.12 with
   venv, 60 GiB persistent storage, unprivileged user namespaces, outbound HTTPS to PyPI, Hugging Face and GitHub).
3. **Copy the dataset package** `devsearch-dataset-v1.tar.gz` (SHA-256 `3d36170959a1932b6d222bd363a80dbc6a270f4c797d67e1c606c1340bd8d903`,
   delivered with this release, not in Git) to `<root>/inbox/` on the server.
4. Run `check`, `prepare`, (import judgments), `diagnose`, `run` as in the quickstart, then bring back the short summary.
5. Optional, later: a Mac for the separate MPS verification.

No other access is needed from this side. Allowing `huggingface.co` in this development container would only let the model download
be rehearsed here; the server does it itself.

1. 审查库（需决定，涉及付费用量）：9,852 项各需两份主判定并裁决分歧；按精简上下文实测约每条 1,738 token，两轮约 3,400 万 token（外推，另加裁决）。请选择 Agent 判定、人工判定或混合（建议人工抽检）。尚未启动。2. 租用符合快速上手第 1 节要求的 CUDA 主机。3. 将数据包复制到服务器 `<root>/inbox/`。4. 按快速上手依次运行并带回简要结果。5. 可选：之后另行验证 MPS。
