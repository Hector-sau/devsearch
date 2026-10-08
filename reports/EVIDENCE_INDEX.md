# Evidence index / 证据索引

Compiled 2026-10-08 on the documentation branch `codex/claude-docs` (created from `577b0ce`, merged with PR #3's head `e3168ca`). Every claim in [the handoff](../docs/HANDOFF.md),
[the task list](../docs/TASKS.md) and [the experiment report draft](experiment-report-draft.md) points here. **Status words:** *current* =
the latest evidence for the release that would be deployed; *superseded* = true when recorded, replaced by later evidence;
*history* = context only; *reported* = stated by Codex or the owner from the GPU host; receipts not yet inspected. Historical receipts keep the commit they were recorded on. Anything not listed has no
evidence and is **pending**.

本索引汇总全部证据来源。“当前”指将部署版本的最新证据；“已被取代”指记录时属实但已有更新证据；“历史”仅作背景；“报告”指 Codex 或所有者报告、回执尚未核对。历史回执保留其原始提交号。未列出的内容视为待验证。

## 1. Where evidence lives / 证据存放位置

| Location | Access | Retention | Holds |
| --- | --- | --- | --- |
| This repository, `reports/` and `reports/evidence/` | anyone with repo access | Git history | Committed receipts and summaries **without** corpus text, question text, test labels or the review salt |
| GitHub Actions runs (links below) | repo access | logs and artifacts **90 days** | CI logs; evidence receipts as artifacts |
| Private draft release `406041484` (tag `devsearch-runtime-checkpoint-20261007`) | repository owner; a read-only token is **untested** | until deleted | Private data package, review checkpoints, durable copies of evidence receipts |
| Operator's Mac, `~/Documents/AI foundation/devsearch-runtime/` | local only | not a backup | Working copies; every authoritative item is also in the release above |

## 2. Current release identity / 当前版本标识

| Item | Value | Evidence | Status |
| --- | --- | --- | --- |
| Previous verified commit | `577b0ce21b94e07b8ca54f6f9bd3a1a3a978dfac` (PR #3 before the correction): PR #2 plus an explicit CI-receipt import for GPU containers that refuse `unshare -rn` | §3 | verified in CI; installed on AutoDL (reported); its target full suite failed (§8) |
| Current correction | `e3168ca6fef9b64ea0688ea059332c729125e30f` (PR #3 head): test-only portability changes plus docs | §3 | CI 4/4 and evidence 2/2 green; target full suite pending |
| Deployed commit and host configuration | — | §8 | **pending from Codex** |
| Earlier code candidate | `22db6f4776a11566bd95797b0420f0b81c985ab1` (`claude/youthful-turing-uc1lkv`, PR #2) | §3 | superseded; contained in `577b0ce` and `e3168ca` |
| Model | `microsoft/MiniLM-L12-H384-uncased` @ `44acabbec0ef496f6dbc93adadea57f376b7c0ec`; `pytorch_model.bin` 133,483,028 bytes, SHA-256 `bcc275fe3e183a68e629c55355be77ef51c2ee780990173ca8f8edc65152fae7` | §4 | current |
| Dataset package | `devsearch-dataset-v2.tar.gz`, 1,645,198 bytes, SHA-256 `06e7ae62c3c78cc14d06407005dc91117d869cd73b8e28c45491744e9c488b1f` | §5 | current |
| Initial test seal | `9a59905c685cec354ed4f3eca39dcf3ee54685f54f6ce6c87d2ac98c7660e124` | §5 | current |
| Dependency lock (CUDA) | `locks/nvidia-cuda-cp312.lock`, SHA-256 `596b9dfe7608e62f811f3931de5895035794b8973736f8b26159d8e06906c27a` | §3 | current |

**Never deploy this documentation branch.** In CI-import mode the GPU host compares every Git-tracked file (documents included) with the
inventory recorded by CI for the deployed commit (`scripts/ci_receipts.py`), so a checkout of `codex/claude-docs` matches no commit's receipts.
Which code commit is deployed, with which host configuration, is pending from Codex.

不要部署本文档分支：CI 证据导入模式会核对全部受 Git 跟踪文件（含文档）。部署的代码提交与主机配置待 Codex 提供。

## 3. Code: CI and non-GPU evidence runs / 代码 CI 与无 GPU 证据

| Commit | Run | Result | What it shows | Status |
| --- | --- | --- | --- | --- |
| `e3168ca` | [CI 37760709410](https://github.com/Hector-sau/devsearch/actions/runs/37760709410), [Evidence 37760704700](https://github.com/Hector-sau/devsearch/actions/runs/37760704700) | 4/4 and 2/2 success | receipts read: exact-lock environment **848 tests, 0 skipped, 0 failures**; 85/85 locked packages, offline isolation proven; real MiniLM fault exit 137, resume, `train:full:42` attempt 2. CI hosts only: the target full suite on this commit is pending (§8) | current correction |
| `577b0ce` | [CI 37748100772](https://github.com/Hector-sau/devsearch/actions/runs/37748100772) | 4/4 success | `cpu-baseline` 3.11/3.12; `ml-cpu` 3.11/3.12 whole suite and tooling tests with zero skips; wheel build; installed-wheel check outside the checkout | previous verified commit |
| `577b0ce` | [Evidence 37748094266](https://github.com/Hector-sau/devsearch/actions/runs/37748094266) | 2/2 success | `locked-env`: exact CUDA lock, 85/85 packages, offline install with proven isolation, **847 tests, 0 skipped, 0 failures** in the exact-lock environment; `real-model`: pinned MiniLM CPU check 10/10, fault exit 137, resume 0, `train:full:42` attempt 2, package imported from outside the checkout | previous verified commit |
| `22db6f4` | [CI 37734103569](https://github.com/Hector-sau/devsearch/actions/runs/37734103569), [Evidence 37734098060](https://github.com/Hector-sau/devsearch/actions/runs/37734098060) | all success | same checks; 835 tests, 0 skipped | current for PR #2's head |
| `32a2b43` | [Evidence 37732271741](https://github.com/Hector-sau/devsearch/actions/runs/37732271741) | locked-env success; real-model **failure** | an `UnboundLocalError` in the new check script (fixed in `22db6f4`) | history |
| `3ace614`, `f774871`, `df44454` | CI 37586439085, 37587859075, 37592058531 | success | earlier candidates | history |

Durable copy of the `22db6f4` receipts: release asset `devsearch-actions-evidence-20261008-22db6f4.tar.gz` (id 620909657, SHA-256
`ea711a6834543033a2758b317ab2a14dfaa42f76fb48b275cc9a7afffa8d58a5`). **The `577b0ce` and `e3168ca` artifacts have no durable copy yet** (pending,
assigned to Codex; they expire 90 days after the runs). Local test runs on macOS without the ML extras report their skips separately (e.g. 835 run, 272 skipped):
they are never zero-skip evidence.

## 4. Real model / 真实模型

| Evidence | Where | Status |
| --- | --- | --- |
| Fetch at the pinned commit with hub-hash verification, conversion, registration, `check_model_cpu.py --require-pretrained` 10/10, export and separate-process reload parity (max difference 0.0) | private checkpoint cp02 (`model-cpu-receipts/`), recorded 2026-10-07/08 | current (CPU) |
| Real-model crash and resume through the controller with the installed wheel | Evidence runs in §3 (`real-model` job; `scripts/check_real_model_resume.py`) | current (CPU) |
| Real tokenizer comparison and membership audit | `reports/evidence/real-model/` | current |
| Pinned model downloaded and hash-verified on the GPU host | §8 (reported) | reported |
| GPU diagnosis, memory calibration, throughput, latency on the device | — | **pending (Codex)** |

## 5. Data, review bank and package / 数据、审查库与数据包

| Evidence | Where | Status |
| --- | --- | --- |
| Corpus `pytorch/tutorials@11512db7…`: 230 documents, 4,259 passages, 0 truncated | `reports/evidence/real-data/prepare-report.json` | current |
| Question set: 515 train / 202 validation / 197 test synthetic questions, agent-written and agent-reviewed, no human review | `reports/evidence/real-data/question-set-summary.json` | current |
| Review protocol revision 5 (`primary_audit_v1`) | `docs/02-data-and-evaluation.md`; declared in the package's `data.toml` | current |
| Review pilots (80 items, two agent runs; 40-item compact pilot, 1,738 tokens and 0.85 s per judgment) | `reports/evidence/review-pilot/` | history (method calibration) |
| Complete review bank: audit exit 0, 9,852/9,852 final; responses 11,627 (agent 11,627, human 0), SHA-256 `1212305af32455cc9aa4ffde5b44aef9459b1538585e5f056e46fceb424221d1` | private: cp04 and inside the v2 package | current |
| Bank outcome: 542 positives, 9,310 negatives; 197/197 questions keep at least one positive; 197/197 v1 positives confirmed; 345 new positives | judgment-bank manifest `5b5f36a6…` in the v2 build (private) | current |
| v2 package byte-reproducible (two builds) and rebuilt byte-identically in a fresh root from a fresh download; 14 frozen membership values equal v1 | cp04 `v2-evidence/` | current |
| Earlier packages and stand-in-tokenizer rebuilds (`3d361709…` v1) | `reports/gpu-release-acceptance.md` row 11 | superseded |

## 6. Private release assets / 私有发布附件 (release 406041484)

| Asset | Id | Bytes | Role | Status |
| --- | --- | ---: | --- | --- |
| `devsearch-dataset-v2.tar.gz` (+ `.sha256`, `.MANIFEST.sha256`, `.RESTORE.md`) | 620811833 | 1,645,198 | the audited dataset package | current |
| `devsearch-runtime-checkpoint-20261008-cp04.tar.gz` (+ sidecars) | 620813988 | 4,643,721 | review state after completion; SHA-256 `e97878a269f1b34634bbc1c4477dfad8c22b6b9b27c0eeb313115097ade6c121`; needs cp01 + cp02 | current |
| `devsearch-runtime-checkpoint-20261008-cp03.tar.gz` | 620724798 | 4,053,440 | review state at the quota stop | superseded by cp04 |
| `devsearch-runtime-checkpoint-20261008-cp02.tar.gz` | 620429525 | 1,143,132 | delta: accepted responses, real-model CPU receipts | current (required base) |
| `devsearch-runtime-checkpoint-20261007-cp01.tar.gz` | 619452486 | 10,355,394 | frozen inputs, package source with the private salt, real tokenizer, v1 package | current (required base) |
| `devsearch-actions-evidence-20261008-22db6f4.tar.gz` | 620909657 | 15,550 | receipts of evidence run 37734098060 and history run 37732271741 | current for `22db6f4` |

## 7. Engineering evidence kept as history / 历史工程证据

| Evidence | Where | Note |
| --- | --- | --- |
| BM25-only validation metrics and latency (synthetic questions, proxy workload) | `reports/evidence/real-data/` | still the only ranking numbers; see the experiment report draft |
| Diagnostic campaign with a random tiny encoder | `reports/evidence/rehearsal/`, `reports/evidence/receipts/` | planted synthetic cue; not a result |
| API and UI demonstration (CPU, diagnostic) | `reports/evidence/ui-demo/` | behavior only; no ranking quality |
| Clean-clone rehearsal transcript | `reports/evidence/clean-rehearsal.md` | procedure check |
| Earlier verification reports | `reports/scaffold-verification.md`, `baseline-verification.md`, `execution-review-verification.md`, `accelerator-readiness-evidence.md` | superseded status statements; kept unchanged |

## 8. Reported from the GPU host / GPU 主机报告事项 (2026-10-08)

Stated by Codex (PR #3 description and acceptance report at `e3168ca`) or the owner. The receipts are not in this repository or the private
release and were **not inspected** here; archiving them is Codex's task ([TASKS](../docs/TASKS.md) A12).

| Fact | Commit | Source | Status |
| --- | --- | --- | --- |
| Private dataset package transferred by SCP | — | owner | reported |
| Preflight passed in CI-import mode, accepting `577b0ce`'s CI export | `577b0ce` | Codex | reported |
| 85 dependency wheels (3,149,538,097 bytes) matched the CI inventory; pinned MiniLM downloaded and hash-verified | `577b0ce` | Codex, owner | reported |
| Exact installation: 85/85 locked packages and installed project snapshot matched, `pip check` 0; locked torch `2.14.1+cu130` CUDA tensor check on an RTX 4090 (ABI sanity only) | `577b0ce` | Codex | reported |
| Target full suite: 847 tests, **2 failures** (external GitHub preflight in local fixture workflows), **2 skips** (CUDA-unavailable negative tests) — the zero-failure, zero-skip gate stopped | `577b0ce` | Codex | reported |
| 6 targeted tests of the correction passed on the host, 0 skipped, 323.6 s | `e3168ca` | Codex | reported |
| Final deployment commit, external host configuration, target full suite on it, `prepare`, `diagnose`, calibration, training, evaluation | — | — | **pending (Codex)** |

以上为 Codex 或所有者报告的 GPU 主机事项，回执未经本文核对；`577b0ce` 实机全量测试未通过门禁，修正版仅完成 6 项针对性测试，GPU 诊断与训练尚未进行。
