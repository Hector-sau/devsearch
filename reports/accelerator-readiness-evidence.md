# Accelerator-readiness evidence against G01–G10 / 加速器就绪证据（G01–G10）

Date 2026-10-05. Branch `claude/youthful-turing-uc1lkv`. Code snapshot under test: `source_snapshot_sha256` `d029ee3474a52e0b507c107743c9f0da4b760a948b9f60aceae2d3e5186b17c5` (the `src/devsearch` tree of commit `993c2f6`; this report and `reports/evidence/` were added afterwards without touching `src/`). Commits of this effort, oldest first: `900854d` (model, devices, training loop) … `b9fe7d6` (failure-injection rehearsal), `2031de2` (train, predict, evaluate, export, serve), `b9c0da6` (prepare), `7ae3514` (review fixes, receipts, ML CI job, operator reference), `b86b2ca` (locks, wheelhouse and bootstrap tooling), `993c2f6` (asset registration, scoring-batch binding, tool tests). No pull request was opened.

日期 2026-10-05，分支 `claude/youthful-turing-uc1lkv`。被测代码快照 `source_snapshot_sha256` 见上（提交 `993c2f6` 的 `src/devsearch`；本报告及 `reports/evidence/` 之后新增，未改动 `src/`）。未开启 pull request。

## 0. Read this first / 先读这里

**This is engineering evidence on synthetic and diagnostic data, on CPU, in one Linux container. It is not accelerator readiness and not a ranking result.** No MPS or CUDA run, no pinned real MiniLM or tokenizer, no real labelled question set, no review bank and no human review exist. The phrase "only the GPU is left" is **false**: the open items in section 4 include assets, labelled data, review, target-machine lock installation and CI, none of which a GPU provides. Estimates (section 3) are kept apart from measured values; every ranking metric in the attached files comes from a planted synthetic cue and says nothing about real quality, so none is quoted here.

**这是在单个 Linux 容器的 CPU 上、基于合成与诊断数据的工程证据，不是加速器就绪，也不是排序结果。**尚无 MPS／CUDA 运行、固定的真实 MiniLM 与分词器、真实标注问题集、审查库或真人审查。“只差 GPU”的说法**不成立**：第 4 节的未决项还包括资产、标注数据、审查、目标机依赖锁安装和 CI，GPU 都不能提供。估算（第 3 节）与实测分开；附件中的排序指标来自合成的植入线索，不代表真实质量，此处不引用。

## 1. Gate status / 门禁状态

Vocabulary: **verified (CPU, synthetic)** = implemented and exercised by recorded commands on synthetic data, not on real data or an accelerator; **partial** = some of the required artifact exists; **open** = not done. Test files show where a behavior is exercised (all are in the 739-test run of section 2).

| ID | Required (docs/11) | Status | Evidence | Still open |
| --- | --- | --- | --- | --- |
| G01 | Real CLIs with strict configs; help is not success | **verified (CPU, synthetic)** | All 16 commands implemented, no stub left (`tests/test_scaffold.py` inventory and help audit, `scripts/check_wheel.py`); strict-config and refusal tests in `tests/test_cli_*.py`, `test_config.py`, `test_ingestion.py`, `test_prepare_sources.py`; full diagnostic workflow produced validated artifacts (section 2.5) | Example configs carry placeholder paths by design |
| G02 | MPS and CUDA profiles, compatible hashed locks, offline assets; installed wheel completes the diagnostic flow outside the repo | **partial** | Both profiles exist and say `unverified`. Hashed locks resolved for each target (2.7). Hash-verified wheelhouses downloaded: `apple-mps` 65 wheels, 191 MB; `nvidia-cuda` 85 wheels, 3.0 GB. The CUDA lock installed **offline** into a new Python 3.12 environment (83 s, `pip check` clean) and the whole suite passed there (739, 0 skips). Portable receipt: wheel installed outside the checkout, network namespace, full diagnostic campaign (2.4) | The `apple-mps` lock was never installed (needs macOS arm64) and pins torch 2.11.0, not the 2.14.1 tested here; no real model/tokenizer assets (`bundle register-assets` now registers them without a Python call; tested on dummy files only); no real accelerator install |
| G03 | Pinned licences, source map, tokenizer; real accepted group and split counts; train-only negatives and unique pairs; automatic audits | **partial** | Real run: `pytorch/tutorials` at commit `11512db7…` fetched, verified by git HEAD, read as data (2.6): 230 documents, 3,815 passages, licence file hashed, deterministic rerun (identical corpus id and report hash). Groups, training-only mining, unique pairs, cross-split audits implemented and tested on synthetic data (`test_groups`, `test_pairs`, `test_prepare`, `test_candidates`) | Real tokenizer missing (the real run used a word-level stand-in, so no length claim); no real questions, so no real group/split counts; the data pipeline was never run on the real corpus |
| G04 | Pre-review all fixed test candidates plus positives; missing coverage blocks sealing; never inject positives | **open for real data** (mechanism **verified (CPU, synthetic)**) | Blinded pool, judgment bank, two-stage seals, coverage failure blocks sealing, positives never added to retrieval (`test_review`, `test_prepare`, `test_campaign` incl. the post-rank pool); exercised with a **scripted oracle**, labelled as such | No real judgments exist; the bank for 300 test questions is up to 15,300 judgments (doc 12): a reviewer and cost decision |
| G05 | Immutable plan, atomic stage manifests, single-owner lock, status/resume, exit codes, hash-valid reuse | **verified (CPU, synthetic)** | `test_controller`, `test_lock`, `test_records`, `test_plan`, `test_campaign` and `test_controller` (real child processes killed with `os._exit` at the documented crash points the tests cover, lease recovery, host-binding chain, fatal vs resumable failures); source-edit-after-planning refusal (regression test) | Nothing known |
| G06 | Isolated device probes with the real model; length/memory/time; unavailable backend fails explicitly | **open** (mechanism **verified (CPU, synthetic)**) | Doctor with fresh-process probes, caps, timeouts on CPU with a tiny diagnostic encoder (`test_doctor`, `test_cli_campaign_tools`); requesting `apple-mps` on this host fails with the reason, releases the lease and never falls back to CPU (`test_cli_serve`, `AcceleratorTests`) | Backend gates `unverified`: needs the pinned MiniLM on the real MPS or CUDA device |
| G07 | Sample-weighted accumulation; continuous vs epoch-resumed equivalence; no diagnostic-weight reuse | **verified (CPU, synthetic)** for the loop; backend **open** | `test_training` (weighted accumulation, resume equivalence, freeze/update boundaries, finite-loss checks, no silent OOM/NaN recovery), `test_campaign` (OOM, NaN, crash recovery) with a tiny random encoder | Same on real MiniLM and an accelerator |
| G08 | Frozen matrix, seeds, selection and latency policies; default chosen before test; honest no-gain and failed-seed outcomes | **verified (CPU, synthetic)** | `test_selection`, `test_latency`, `test_report`, `test_campaign` (both the "reranker ships" and the "BM25 stays default" paths; the test side cannot be opened before `select` commits) | Real validation data and real latency |
| G09 | Same encoding/scoring online and offline; deterministic ties; full-result fallback; health/busy/deadline; corrupted bundle cannot activate | **verified (CPU, synthetic)** | `test_pipeline`, `test_api`, `test_release`, `test_model_io`, `test_cli_serve` (a real loopback server process, graceful SIGTERM, lease handling, batch-size binding, baseline release served without importing torch) | Latency on the real model and device; the UI was not exercised in a browser |
| G10 | Measured report, per-query rows, lineage, offline reload, independent backup checksum receipt; a missing stage cannot summarise complete | **verified (CPU, synthetic)**; independence **unverified** | `test_report`, `test_backup`, `test_campaign`, `test_cli_export`; the diagnostic campaign wrote report, per-query rows, evaluated release, reload report and a verified backup receipt (2.5) | The backup is a local copy: an independent device and a real transport were not exercised |

## 2. Commands, exit status and results / 命令、状态与结果

All commands ran from `/home/user/devsearch` in a Linux x86_64 container (kernel 6.18). Tested versions: dev environment Python 3.11.15 with torch 2.14.1 (CPU), numpy 2.4.6, transformers 5.18.0, tokenizers 0.23.2, safetensors 0.8.0, fastapi 0.142.2, uvicorn 0.54.0, httpx 0.28.1, streamlit 1.65.0; lock-installed environment Python 3.12.3 with the `nvidia-cuda` lock's versions (numpy 2.5.3, the others as above). The container has no GPU.

### 2.1 Unit tests

| Command | Exit | Result |
| --- | --- | --- |
| `python scripts/record_receipt.py implementation --output <dir> --python <dev venv python>` (runs `PYTHONPATH=src:tests python -m unittest discover -s tests`) | 0 | **739 tests, 0 failures, 0 errors, 0 skipped.** Receipt `reports/evidence/receipts/implementation_verified-d029ee3474a5-2026-10-05T154734Z.json`, `manifest_sha256` `b61e007e059365683ba1d0e994b7deda3caed17c3185e6b5043c9366bc73c82b` |
| the same with the lock-installed Python 3.12 environment | 0 | **739 tests, 0 skipped.** Receipt `…T155440Z.json`, `manifest_sha256` `96f282a000a2fa5d6be4d3b0fddb5e6573b5f1df36bcbdc9381d73aa20d1be71` |
| `PYTHONPATH=src:tests python3 -m unittest discover -s tests` with no extras installed | 0 | 739 run, **244 skipped** (every ML/API test is reported as a skip, never as a pass), none failed |

The receipt rule treats any skip as a failure of the gate. One earlier full run failed once because source files were edited while it ran; the preflight correctly refused the changed code, and a regression test now covers that refusal (`test_campaign.py`, `test_changing_the_source_after_planning_is_refused_by_preflight_before_any_work`).

### 2.2 Installed wheel

| Command | Exit | Result |
| --- | --- | --- |
| `python scripts/check_wheel.py dist/devsearch-0.0.0-py3-none-any.whl` | 0 | PASS: wheel installed without dependencies outside the checkout, built, searched and validated a fixture index; every command prints its help and refuses a bare invocation. Fixture-level only |

### 2.3 CI

`.github/workflows/scaffold-check.yml` now has an `ml-cpu` job (CPU torch wheel, the extras, the suite through the receipt script so any skip fails, wheel build and check). **It has not run on GitHub**: no pull request or dispatch was made. The earlier dependency-free job ran once on GitHub (manual dispatch, recorded in `reports/baseline-verification.md`).

### 2.4 Portable-bundle receipt (G02)

`python scripts/record_receipt.py portable --wheel dist/devsearch-0.0.0-py3-none-any.whl --output <dir>` → exit 0. In a new environment outside the checkout, with `unshare -rn` (a network namespace, so no network was reachable), the installed wheel ran `rehearsal` → `doctor` → `plan` → `run` → `status`, every step exit 0, status `complete`. The receipt binds to the **installed** code's snapshot hash and the script verified it equals the checkout's (`wheel_matches_checkout: true`). Receipt `reports/evidence/receipts/portable_bundle_verified-d029ee3474a5-2026-10-05T154852Z.json`, `manifest_sha256` `2d0f6032fe37cf295ce303eb5fbec99d3b390e03101223de0ed4798ed1be8ca7`. Limitation: third-party libraries were shared read-only from the dev environment through a `.pth` file; only `devsearch` came from the wheel.

### 2.5 Diagnostic campaign with receipts (G01, G05, G08–G10)

Commands (from the checkout, `PYTHONPATH=src`), all synthetic CPU:

| Step | Command | Exit | Wall time (measured) |
| --- | --- | --- | --- |
| tree | `python -m devsearch.rehearsal --output R --topics 60` | 0 | 5 s |
| doctor | `python -m devsearch.doctor --profile cpu-diagnostic --host R/host-example.toml --output R/doctor-report.json` | 0 | 31 s (real worker processes) |
| plan | `python -m devsearch.plan --host … --doctor-report … --output R/plan/execution-plan.json --receipts <3 receipts> --wheel-sha256 5ff76357…` | 0 | – |
| run | `python -m devsearch.run --plan R/plan/execution-plan.json` | 0 | 8 s |
| status | `python -m devsearch.status --plan …` | 0 | – |

Plan gates: `implementation_verified` **passed** and `portable_bundle_verified` **passed** (both from the receipts above), `data_ready` `unverified` (diagnostic data), `backend_verified[cpu-diagnostic]` `not_applicable`; `accelerator_run_ready[cpu-diagnostic]` **false**, `evidence_label` `fixture`. 16 stages succeeded, 48 events. Artifacts (all under `reports/evidence/rehearsal/`):

| Artifact | Identity |
| --- | --- |
| plan | `plan-0dacb55f8ed155a8464867db7f9ae243` |
| bundle manifest | `3d8fbdd9afa766f33502ae3f748060a323f50564ed8cc586ded89624ab8e164f` (`diagnostic: true`) |
| doctor report | `2541f8e582e11201be462a4b73fe6274d09af247318ed73d1e66eec10313a3ea` |
| validation selection | `selection-32460585f9d13fe45a68aaf1233f63c4` |
| final report | `report-e3e014d9f411b1d2fc411e933d6a825f` (stamped DIAGNOSTIC REHEARSAL; synthetic planted cue) |
| evaluated release | `release-9ece73873e61604a994ed7a71605511b` |
| acceptance report | `0b91738d5eaa3bc851d9dd85948f97fed5d729ec1370970882336098fda5fe24` |
| backup receipt | `d485740e5044731e05a738668f777c338035a0caae9bca9c044b718bb46868a2` (verified, 164 files, 1,341,630 bytes) |

### 2.6 Real source ingestion (T03, G03 partial)

`python -m devsearch.prepare --config real/sources.toml --output real/corpus3 --offline --runtime-root real` → exit 0, 5 s (the snapshot had been fetched earlier by the same module: `git init` + depth-1 `git fetch` of the exact commit, 3.3 s, `verified-git-head`). Source `pytorch/tutorials` at `11512db7cbbcc4b530ecfc63205d3fed13fa190d`, declared licence BSD-3-Clause (licence file hash recorded, text not verified against SPDX). 790 files seen, 234 included by pattern, **230 documents** produced passages, 4 produced none, 0 quarantined (after reader fixes that this very run exposed: unsanitised heading labels, docstrings with line continuations, layout containers such as `tab-set` being dropped with their content), 1 symlink skipped; **3,815 passages**, 1,424 of them code fragments, 16 duplicate, 43 headings without body, 8 over-long code lines, 7 short passages excluded; **0 truncated**. Corpus id `corpus-20f00a6d9f5d55ff360e1240d95edd21`, report `manifest_sha256` `2cc11e9ec227e1c84e4132ed8eda5c9e88caf0e7791e0c08f65e61cc4a2f00dc` (`reports/evidence/real-source/`); two runs gave the identical corpus id and report hash.

**Caveat that matters:** the tokenizer was a word-level stand-in built from the snapshot's own vocabulary (report: `asset_kind: null`, special tokens 3). Passage lengths (budget 189, median 155, p95 188) are therefore **not** token lengths of the pinned MiniLM tokenizer, and no length or truncation-rate claim is made. This step must be rerun with the real tokenizer.

### 2.7 Locks and offline installation (G02 partial)

| Command | Exit | Result |
| --- | --- | --- |
| `python scripts/lock_profile.py --profile apple-mps` (uv 0.12.23, resolved for `aarch64-apple-darwin`, CPython 3.12) | 0 | 65 packages, sha256 `075e6a4760d4e199a268e7d1925a1d39d6ced332dbb6588507d39b27c1e937a8`; torch 2.11.0; no CUDA package |
| `python scripts/lock_profile.py --profile nvidia-cuda` (`x86_64-manylinux_2_28`) | 0 | 85 packages, sha256 `596b9dfe7608e62f811f3931de5895035794b8973736f8b26159d8e06906c27a`; torch 2.14.1 (PyPI default CUDA build) |
| `python scripts/bootstrap_env.py download --lock locks/apple-mps-cp312.lock --dest …` | 0 | 65 wheels, 191 MB, 12 s, every hash verified by `pip --require-hashes` |
| `… download --lock locks/nvidia-cuda-cp312.lock …` | 0 | 85 wheels, 3.0 GB, 61 s, hashes verified. (A first attempt failed because pip does not expand manylinux tags to the older glibc tags the vendor wheels use; `bootstrap_env.py` now lists them) |
| `/usr/bin/python3.12 scripts/bootstrap_env.py install --lock locks/nvidia-cuda-cp312.lock --wheelhouse … --wheel … --venv …` | 0 | new environment, offline, 83 s, 6.0 GB on disk, `pip check`: no broken requirements. (Installed with the wheel of an earlier commit; the lock does not depend on it.) |

The lock was resolved at the newest versions PyPI offered that day; the unit tests ran against the versions listed above, which differ for numpy in the dev environment and for torch in the macOS lock. Whether transformers 5.x loads `microsoft/MiniLM-L12-H384-uncased` and whether either torch build runs on the target is **unverified**.

## 3. Estimates kept apart from measurements / 估算与实测分开

**Measured here (CPU diagnostic only):** the wall times in 2.1–2.7; storage: real snapshot 184 MB, prepared corpus 16 MB (3,815 passages), diagnostic rehearsal tree 8.5 MB, backup 1.3 MB, `apple-mps` wheelhouse 191 MB, `nvidia-cuda` wheelhouse 3.0 GB, CUDA-lock environment 6.0 GB.

**Estimates (labelled `estimate` in the plan):** the diagnostic plan estimated 7.1 s of work for stages it can estimate (`train:frozen` 0.33 s vs 0.42 s observed, `train:full` 0.88 s vs 0.74 s, `latency` 3.7 s vs 0.44 s, `rank_test` 0.72 s vs 0.10 s; stages without a measured rate carry `estimate: null`). This only shows the estimator's mechanics on a tiny CPU model. **No accelerator time, no real-model time, no review time and no real-data cost was measured**; the figures in docs/12 remain illustrative assumptions.

## 4. Open blockers and next action / 未决阻塞项与下一步

Nothing below is a GPU task except item 5.

1. **Allow the pinned model and tokenizer** (`huggingface.co` and its CDN hosts; this container's proxy blocks them) or supply the files: pin `microsoft/MiniLM-L12-H384-uncased` at a 40-hex commit, copy the files, run `python -m devsearch.bundle register-assets --directory D --repo … --revision …`, then rerun `prepare` with the real tokenizer and `prepare_data` with `[lengths]` so every passage and question is measured.
2. **Decide on real labelled data and review (cost/time decision for you):** write and review a question set from the real corpus (targets in doc 02; the review bank for 300 test questions is up to 15,300 judgments, doc 12). Questions stay labelled synthetic and agent review stays labelled agent unless a human judges them. Until then G03/G04 stay open and `data_ready` stays `unverified`.
3. **Install and test the `apple-mps` lock on a Mac** (and the `nvidia-cuda` lock on the CUDA host), keep the `record_receipt.py implementation` receipt from there, then run `doctor`.
4. **Run `ml-cpu` CI on GitHub** (open a pull request or dispatch); not done without your instruction.
5. **Run the campaign on the real accelerator** (backend gates G06/G07/G09 latency/G10 independent backup). Only after items 1–4 does this produce meaningful evidence.
6. Remaining engineering notes, all listed in `docs/13-operator-reference.md` section 4: backup is a local copy; probe scratch uses the system temp directory; a binding revision copies every host field; the UI was not exercised in a browser; `retrieve --mode rerank` is intentionally not implemented (use `evaluate`, `predict`, `serve`).

Reviewer notes: a sub-agent read the code and listed about two dozen possible defects and doc/code mismatches; each was checked against the code before acting. Fixed, with regression tests where behaviour changed: the `plan` error text lost its fallback, an empty-command receipt crashed the gate, `serve` imported torch for baseline releases, `status` could rewrite the event log, `prepare_data` seal paths were only name-checked, test-question lengths were unchecked before the campaign (optional `[lengths]` now), rehearsal data with real-looking assets was not diagnostic, the interval level was hard-coded, the portable receipt did not bind the installed code, the rehearsal tree lacked `runtime/run`, the scoring batch size was not bound to a release, `configs/data.toml` was missing, and several READMEs were stale. The rest are documented deviations in `docs/13` section 4. That review is an agent review, not a human one.

## 5. 中文摘要

本报告在单个 Linux 容器的 CPU 上，用合成与诊断数据记录工程证据：全部 16 个命令已实现；全量测试 739 项通过且无跳过（开发环境与依据 `nvidia-cuda` 锁离线安装的 Python 3.12 环境各一份回执）；断网命名空间中、由 wheel 安装在检出目录之外完成完整诊断流程（回执）；真实公开仓库 `pytorch/tutorials` 固定提交被解析为 230 篇文档、3,815 个段落（使用词级替代分词器，**不构成长度结论**）；两个加速器配置的带哈希依赖锁已解析，离线包哈希校验通过，CUDA 锁已离线安装。**未完成且不宣称：**任何 MPS／CUDA 运行、固定的真实 MiniLM 与分词器、真实标注问题与审查库、真人审查、苹果锁的实际安装、GitHub 上的 `ml-cpu` CI；“只差 GPU”不成立。估算与实测在第 3 节分开，附件中的排序数字来自合成线索，不代表真实质量。下一步见第 4 节（先放行模型与分词器域名或提供文件，再决定真实问题与审查的投入，之后才是目标机安装与加速器运行）。
