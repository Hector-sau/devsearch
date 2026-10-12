# Evidence index / 证据索引

## Final966 registry — 2026-10-13 / 最终966登记

The selected966 configuration now has three fresh same-host training seeds 42/43/44, separately audited GPU receipts and shared frozen CPU evaluation. This supersedes only the earlier one-seed status; original receipts, historical model families and labels remain unchanged. The existing accepted seed 42 CPU demo remains preferred with BM25 control/fallback. Proposed `v1.2.0` publication still requires exact-source checks; private native checkpoint archival is tracked separately.

选定966配置现已有同机三个全新种子42／43／44，GPU回执另经审核并使用统一冻结CPU评测；本登记仅更新此前单种子状态，原回执、历史模型系列及标签保持不变。已有已验收seed42 CPU演示保持首选，保留BM25对照／降级；拟发布v1.2.0仍须准确源码检查，私有原生断点归档单独跟踪。

| Evidence / 证据 | Public artifact / 公开产物 | Scope / 范围 |
| --- | --- | --- |
| Three-seed 966 replication / 三种子966复现 | [Final report](final966-replication-2026-10-13.md) · [Aggregate](evidence/final966-20261013/summary.json) | Fixed47 primary/81 secondary graded development queries; exact per-seed scores and conditional group intervals / 固定47题主范围及81题次范围分级开发问题、每种子成绩及条件组区间 |
| Original202 regression view / 原202题回顾视图 | [Aggregate](evidence/final966-20261013/summary.json) | 4, 571 judged pairs, 5, 530 natural UNKNOWN records; partial binary view, not complete acceptance / 部分二元视图，不作完整验收 |
| Controlled1, 932 duration comparison / 受控1, 932时长对照 | [Final report](final966-replication-2026-10-13.md) | One freshseed42; matched 966 endpoint; declared rule fails / 一个新seed42，与匹配966比较，预设规则未通过 |
| Independent audit and identity / 独立审核及身份 | [Hash manifest](evidence/final966-20261013/evidence-manifest.json) | Hash commitments to owner-held audited evidence; private contents are not published / 所有者审核证据哈希承诺，不公开私有内容 |
| Historical CPU demo / 历史CPU演示 | [CPU acceptance](evidence/demo966-cpu-20261012/acceptance-summary.json) | `1.1.1.dev0`, 80 checks +36 independent checks, 150 warm sequential real-TCP requests; no newversion claim / 保留历史版本及实测范围 |

## Historical local demo registry — 2026-10-12 / 本地演示登记

The owner-authorized local demo uses the single-seed 42 966-update endpoint with fixed fusion 0.5/k60 after CPU acceptance. This registry adds current evidence without rewriting the published `v1.1.0` or original R5 experiment receipts.

所有者授权的本地演示在 CPU 验收后使用单一种子 42 的 966 次更新模型及固定融合 0.5／60。本登记增加当前证据，不改写已发布 `v1.1.0` 或原 R5 实验回执。

| Evidence / 证据 | Artifact / 产物 | Scope / 范围 |
| --- | --- | --- |
| Matched fixed-model CPU comparison / 匹配固定模型CPU对比 | [Report](demo966-development-2026-10-12.md) · [All-system aggregate](evidence/demo966-cpu-20261012/development-comparison-summary.json) | 47 new + 34 retained development questions; all seven systems; one training seed each / 新47＋原34开发题，七个系统，每个只有一种子 |
| Actual CPU API acceptance / 真实CPU接口验收 | [Summary](evidence/demo966-cpu-20261012/acceptance-summary.json) | Exact installed code/model; real TCP latency, full-pool parity, controlled faults and startup refusals / 准确安装代码及模型，真实接口延迟、完整池一致性、受控故障及启动拒绝 |
| Local demo controls / 本地演示工具 | [Operator guide](../docs/DEMO966_CPU.md) | CPU-only candidate, rerank default, BM25 baseline/fallback; owner-held private assets / 仅CPU候选，默认重排、BM25对照降级及所有者私有资产 |

## Current registry — 2026-10-10 / 当前证据登记

The sections numbered 1–10 below preserve Claude's 2026-10-08 snapshot. Their words “current”, “pending” and “no human judgments” refer to that snapshot and its particular bank. The registry here and [RESULTS](RESULTS.md) govern the latest state; six human-assisted validation resolutions do not alter the earlier agent-only test bank.

下方编号 1～10 节保留 Claude 的 2026-10-08 快照；其中“当前”“待完成”“无人工判定”指当时状态及对应标签库。最新状态以本登记和结果报告为准；六条人工辅助验证裁决不改变此前纯 Agent 测试库。

| Evidence / 证据 | Artifact / 产物 | Scope / 范围 |
| --- | --- | --- |
| First formal campaign and early follow-up / 首轮正式实验与较早后续实验 | [gpu-20261008](evidence/gpu-20261008/evidence-manifest.json) | Historical source `e3168ca`; original hashes preserved / 保留原哈希 |
| Expanded validation v4 / 扩展验证 v4 | [aggregate report](evidence/project-20261009/validation-v4-summary.json) | 25 systems, 202 queries, 4, 571 labeled pairs; adaptive validation / 25 系统、202 题、4, 571 对，反复使用的验证集 |
| R5 candidates / R5 候选 | [checkpoint/config identities](evidence/project-20261009/r5-candidate-checkpoints.json) | Three seeds and their historical selected epochs / 三种子及历史选中轮次 |
| Latest grouped CUDA comparison / 最新分组 CUDA 对照 | [completion and bounds](evidence/project-20261009/grouped-campaign-summary.json) | Six completed runs on `cc0e7f0`; stop pairwise direction / 六次完成，停止排序损失方向 |
| Sanitized aggregate provenance / 脱敏汇总来源 | [provenance manifest](evidence/project-20261009/provenance.json) | Original private source hashes and committed summary hashes / 私有原记录与提交摘要的哈希 |
| Ordinary CPU CI on `cc0e7f0` / 普通 CPU CI | [37877524865](https://github.com/Hector-sau/devsearch/actions/runs/37877524865) | All four jobs succeeded / 四作业通过 |
| Exact-lock and real-model evidence on `cc0e7f0` / 准确依赖及真实模型证据 | [37877521227](https://github.com/Hector-sau/devsearch/actions/runs/37877521227) | Both jobs succeeded; locked suite 906, zero failures/errors/skips / 两作业通过，锁环境 906 项零失败错误跳过 |
| Frozen R5 implementation CI / 冻结 R5 实现检查 | [37958581548](https://github.com/Hector-sau/devsearch/actions/runs/37958581548) | `b76e349`; four jobs passed, Python3.11/3.12 ML suites936each with0skips / 四项通过，两版本 ML 各936项零跳过 |
| Independent R5 holdout and CPU acceptance / 独立 R5 留出集及 CPU 验收 | [Final report](r5-independent-holdout-2026-10-10.md) · [compact summary](evidence/r5-independent-20261010/summary.json) · [independent audit](evidence/r5-independent-20261010/independent-audit.json) | 59queries/789sealedpairs; A789/B165/C12; both quality gates fail, BM25default / 59题789封存对，两项效果门槛未通过 |
| Real R5 candidate API/UI / 真实 R5 候选接口界面 | [Actual screenshot and scope](evidence/r5-independent-20261010/README.md) | Frozen seed 42/epoch 3 + fusion on M4CPU; warm p95622.065 ms, no GPU or SLA claim / 固定候选 M4CPU，预热p95622.065毫秒 |

Small GPU results were downloaded and checksum-verified locally. External storage is now connected: the selected R5 inference archive, corpus/index and independent evaluation/review state are restored and checked there. Native training checkpoint tensors remain on the existing server; full checkpoint backup is not claimed. No claim is made that all later artifacts were copied to the older GitHub draft release. Repository summaries verify selected facts, not full private-data reproduction.

GPU 小型结果已下载并本地校验。外置存储现已接入，选定 R5 推理包、语料／索引及独立评测／审核状态已恢复核验；原生训练断点仍保留既有服务器，不宣称全部断点已备份。不声称所有后续产物已复制到旧 GitHub 草稿发布；仓库摘要只能验证部分事实，不能替代完整私有数据复现。

Release-candidate preparation evidence: [actual BM25 UI/API](evidence/release-demo-20261009/README.md), [package verification](release-packaging-verification-2026-10-09.md), and [independent audit](release-independent-audit-2026-10-09.md). These do not change historical GPU or test identities. Candidate CI and final-main CI are distinct; see [release procedure](../docs/RELEASE.md).

发布候选准备证据见真实 BM25 界面／接口、打包核验与独立审核，不改变历史 GPU 或测试标识。候选 CI 与最终主分支 CI 分开，详见发布流程。

## Historical registry — 2026-10-08 / 历史登记

Compiled 2026-10-08 on the documentation branch `codex/claude-docs` (created from `577b0ce`, merged with PR #3's head `e3168ca`). Every claim in [the handoff](../docs/HANDOFF.md),
[the task list](../docs/TASKS.md) and [the experiment report draft](experiment-report-draft.md) points here. **Status words:** *current* =
the latest evidence for the release that would be deployed; *superseded* = true when recorded, replaced by later evidence;
*history* = context only; *reported* = stated by Codex or the owner from the GPU host, not read here; *verified here* = read directly from the extracted evidence files on the operator's Mac (§9). Historical receipts keep the commit they were recorded on. Anything not listed has no
evidence and is **pending**.

本索引汇总全部证据来源。“当前”指将部署版本的最新证据；“已被取代”指记录时属实但已有更新证据；“历史”仅作背景；“报告”指 Codex 或所有者报告、回执尚未核对。历史回执保留其原始提交号。未列出的内容视为待验证。

Final version-1 source, checks and asset identities are supplied by the [versioned release](https://github.com/Hector-sau/devsearch/releases/tag/v1.0.0) manifest and original CI receipt attachments. They verify the final integrated source on CPU; they do not relabel historical GPU measurements as new runs.

第一版最终源码、检查及附件标识由版本发布清单与原始 CI 回执提供，验证最终整合源码的 CPU 流程，不把历史 GPU 测量改写成新运行。

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
| Current correction | `e3168ca6fef9b64ea0688ea059332c729125e30f` (PR #3 head): test-only portability changes plus docs | §3 | CI 4/4 and evidence 2/2 green; target full suite inside `prepare`: 848 tests, 0 skipped (§9) |
| Deployed commit and host configuration | `e3168ca`, fresh runtime with its own external host configuration | §8, §9 | deployment reported by Codex; its first formal campaign verified here (§9) |
| Earlier code candidate | `22db6f4776a11566bd95797b0420f0b81c985ab1` (`claude/youthful-turing-uc1lkv`, PR #2) | §3 | superseded; contained in `577b0ce` and `e3168ca` |
| Model | `microsoft/MiniLM-L12-H384-uncased` @ `44acabbec0ef496f6dbc93adadea57f376b7c0ec`; `pytorch_model.bin` 133, 483, 028 bytes, SHA-256 `bcc275fe3e183a68e629c55355be77ef51c2ee780990173ca8f8edc65152fae7` | §4 | current |
| Dataset package | `devsearch-dataset-v2.tar.gz`, 1, 645, 198 bytes, SHA-256 `06e7ae62c3c78cc14d06407005dc91117d869cd73b8e28c45491744e9c488b1f` | §5 | current |
| Initial test seal | `9a59905c685cec354ed4f3eca39dcf3ee54685f54f6ce6c87d2ac98c7660e124` | §5 | current |
| Dependency lock (CUDA) | `locks/nvidia-cuda-cp312.lock`, SHA-256 `596b9dfe7608e62f811f3931de5895035794b8973736f8b26159d8e06906c27a` | §3 | current |

**Never deploy this documentation branch.** In CI-import mode the GPU host compares every Git-tracked file (documents included) with the
inventory recorded by CI for the deployed commit (`scripts/ci_receipts.py`), so a checkout of `codex/claude-docs` matches no commit's receipts.
Codex reports `e3168ca` as the deployed commit, with an external host configuration kept on the host.

不要部署本文档分支：CI 证据导入模式会核对全部受 Git 跟踪文件（含文档）。Codex 报告实际部署 `e3168ca`，外部主机配置保存在主机上。

## 3. Code: CI and non-GPU evidence runs / 代码 CI 与无 GPU 证据

| Commit | Run | Result | What it shows | Status |
| --- | --- | --- | --- | --- |
| `e3168ca` | [CI 37760709410](https://github.com/Hector-sau/devsearch/actions/runs/37760709410), [Evidence 37760704700](https://github.com/Hector-sau/devsearch/actions/runs/37760704700) | 4/4 and 2/2 success | receipts read: exact-lock environment **848 tests, 0 skipped, 0 failures**; 85/85 locked packages, offline isolation proven; real MiniLM fault exit 137, resume, `train:full:42` attempt 2. CI hosts; the target full suite inside `prepare` on the RTX 4090 also passed: 848 tests, 0 skipped (§9) | current correction |
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
| GPU diagnosis, memory calibration, throughput, latency on the device | §9 (`state/diagnose.json`, doctor report, `latency.json`) | **done on `e3168ca`, verified here** |

## 5. Data, review bank and package / 数据、审查库与数据包

| Evidence | Where | Status |
| --- | --- | --- |
| Corpus `pytorch/tutorials@11512db7…`: 230 documents, 4, 259 passages, 0 truncated | `reports/evidence/real-data/prepare-report.json` | current |
| Question set: 515 train / 202 validation / 197 test synthetic questions, agent-written and agent-reviewed, no human review | `reports/evidence/real-data/question-set-summary.json` | current |
| Review protocol revision 5 (`primary_audit_v1`) | `docs/02-data-and-evaluation.md`; declared in the package's `data.toml` | current |
| Review pilots (80 items, two agent runs; 40-item compact pilot, 1, 738 tokens and 0.85 s per judgment) | `reports/evidence/review-pilot/` | history (method calibration) |
| Complete review bank: audit exit 0, 9, 852/9, 852 final; responses 11, 627 (agent 11, 627, human 0), SHA-256 `1212305af32455cc9aa4ffde5b44aef9459b1538585e5f056e46fceb424221d1` | private: cp04 and inside the v2 package | current |
| Bank outcome: 542 positives, 9, 310 negatives; 197/197 questions keep at least one positive; 197/197 v1 positives confirmed; 345 new positives | judgment-bank manifest `5b5f36a6…` in the v2 build (private) | current |
| v2 package byte-reproducible (two builds) and rebuilt byte-identically in a fresh root from a fresh download; 14 frozen membership values equal v1 | cp04 `v2-evidence/` | current |
| Earlier packages and stand-in-tokenizer rebuilds (`3d361709…` v1) | `reports/gpu-release-acceptance.md` row 11 | superseded |

## 6. Private release assets / 私有发布附件 (release 406041484)

| Asset | Id | Bytes | Role | Status |
| --- | --- | ---: | --- | --- |
| `devsearch-dataset-v2.tar.gz` (+ `.sha256`, `.MANIFEST.sha256`, `.RESTORE.md`) | 620811833 | 1, 645, 198 | the audited dataset package | current |
| `devsearch-runtime-checkpoint-20261008-cp04.tar.gz` (+ sidecars) | 620813988 | 4, 643, 721 | review state after completion; SHA-256 `e97878a269f1b34634bbc1c4477dfad8c22b6b9b27c0eeb313115097ade6c121`; needs cp01 + cp02 | current |
| `devsearch-runtime-checkpoint-20261008-cp03.tar.gz` | 620724798 | 4, 053, 440 | review state at the quota stop | superseded by cp04 |
| `devsearch-runtime-checkpoint-20261008-cp02.tar.gz` | 620429525 | 1, 143, 132 | delta: accepted responses, real-model CPU receipts | current (required base) |
| `devsearch-runtime-checkpoint-20261007-cp01.tar.gz` | 619452486 | 10, 355, 394 | frozen inputs, package source with the private salt, real tokenizer, v1 package | current (required base) |
| `devsearch-actions-evidence-20261008-22db6f4.tar.gz` | 620909657 | 15, 550 | receipts of evidence run 37734098060 and history run 37732271741 | current for `22db6f4` |

## 7. Engineering evidence kept as history / 历史工程证据

| Evidence | Where | Note |
| --- | --- | --- |
| BM25-only validation metrics and latency (synthetic questions, proxy workload) | `reports/evidence/real-data/` | still the only ranking numbers; see the experiment report draft |
| Diagnostic campaign with a random tiny encoder | `reports/evidence/rehearsal/`, `reports/evidence/receipts/` | planted synthetic cue; not a result |
| API and UI demonstration (CPU, diagnostic) | `reports/evidence/ui-demo/` | behavior only; no ranking quality |
| Clean-clone rehearsal transcript | `reports/evidence/clean-rehearsal.md` | procedure check |
| Earlier verification reports | `reports/scaffold-verification.md`, `baseline-verification.md`, `execution-review-verification.md`, `accelerator-readiness-evidence.md` | superseded status statements; kept unchanged |

## 8. Reported from the GPU host / GPU 主机报告事项 (2026-10-08)

Stated by Codex (PR #3 description; Codex's local handoff `devsearch-autodl-preflight-2026-10-08.md`, 18: 27) or the owner. The receipts are
not in this repository or the private release and were **not inspected** here; archiving them is Codex's task ([TASKS](../docs/TASKS.md) A12).
Server paths, access details and proxy settings are deliberately not copied into the repository.

| Fact | Commit | Status |
| --- | --- | --- |
| Private dataset package transferred by SCP; SHA-256 `06e7ae62…` re-verified on the host | — | reported |
| `gpu.py check` passed in CI-import mode | `577b0ce`, then `e3168ca` | reported |
| 85 dependency wheels (3, 149, 538, 097 bytes), each SHA-256 equal to the CI inventory; pinned MiniLM, all hub hashes verified, used through `model.source = "local"` after Hugging Face TLS timeouts | — | reported |
| Exact-lock installation and environment inspection passed; locked torch `2.14.1+cu130` CUDA tensor check on an RTX 4090 (ABI sanity only) | `577b0ce` | reported |
| Inside `prepare`, the full implementation suite stopped the gate: 847 tests, **2 failures** (external GitHub preflight in local fixture workflows), **2 skips** (CUDA-unavailable negative tests); that runtime is kept as failed-run evidence | `577b0ce` | reported |
| 6 targeted tests of the correction passed on the host, 0 skipped, 323.6 s | `e3168ca` | reported |
| CI import authenticated on the Mac and transferred: origin SHA-256 `030a9039dbc6bb0753caa4025bca2368313ed01beec678c5f14318a9d5903221`, CI wheel SHA-256 `f8281f60d88ca6e95cfd37adf98279f4f385408f0b81b246af9fcbd2a205786e` | `e3168ca` | reported |
| Fresh source and runtime; exact-lock installation passed (103 s); environment inspection passed; full implementation suite inside `prepare`: **848 tests, 0 failures, 0 errors, 0 skipped** | `e3168ca` | suite result **verified here** (§9); the rest reported |
| `prepare`, `diagnose`, calibration, training, test evaluation, server backup | `e3168ca` | **completed, verified here (§9)** |
| Local download of the archives and their SHA-256 check | `e3168ca` | download complete; SHA-256 verified for the evidence and backup archives (§9) |

以上标“报告”者为 Codex 或所有者报告；`577b0ce` 的 `prepare` 内全量测试未通过门禁；`e3168ca` 的第一轮正式 GPU 流程已完成，见第 9 节（本文直接核对证据文件）。

## 9. First formal GPU campaign / 第一轮正式 GPU 实验 (`e3168ca`, 2026-10-08)

**Verified here:** read on 2026-10-08 from the evidence files Codex extracted on the operator's Mac
(`devsearch-runtime/autodl-results-e3168ca/evidence-extracted/`, not in Git). The evidence archive they come from,
`devsearch-e3168ca-evidence.tar.gz`, has a local SHA-256 equal to the delivery manifest (`dcc4bd63…f1ae812`). Nothing here was re-run.
Host: NVIDIA GeForce RTX 4090 (24 GiB), CUDA 13.0, torch `2.14.1+cu130`, Python 3.12.3, profile `nvidia-cuda`.

| Evidence | Value | Source file |
| --- | --- | --- |
| `prepare` | finished 10: 42: 27 UTC; bundle `97f548cd…`, not diagnostic | `state/prepare.json` |
| Target full suite (inside `prepare`) | 848 tests, 0 failures, 0 errors, 0 skipped (verdict OK); install matched 85/85 locked packages, `pip check` 0, no network-isolation claim on the host | `receipts/…/implementation_verified-…json`, `state/target-install-check.json` |
| Portable receipt | the imported Linux CI receipt (CI host, isolation proven there) | `receipts/…/portable_bundle_verified-…json` |
| `diagnose` on CUDA | passed: fault `mid_stage:train:full:42` exit 137, status 0, `train:full:42` attempt 2 (synthetic data, plumbing only) | `state/diagnose.json` |
| Campaign | `run` exit 0, `status` 0; all 16 stages succeeded on attempt 1; plan `plan-449bfed3…` | `state/run-latest.json`, `campaign/run-state.json` |
| Gates | `implementation_verified`, `data_ready`, `portable_bundle_verified`, `backend_verified[nvidia-cuda]` passed; `accelerator_run_ready[nvidia-cuda]` true | `reports/summary/acceptance-report.json` |
| Validation selection | **default mode BM25**: no recipe met every predeclared gate (`selection-c38b654a…`) | `reports/select/validation-selection.json` |
| Test report | `report-f62b55a0…`, complete; 197 synthetic questions, agent judgments only (11, 627 agent / 0 human); paired group bootstrap 95%, 10, 000 repetitions | `reports/score/final-report.json` |
| Seals | initial `9a59905c…`, bank `5b5f36a6…`, ranked outputs `e9d139c1…`, final `8e7f1716…` | `reports/score/final-report.json` |
| Release | `release-ce3160fc…`, evaluated, defaults to BM25, ships no model; reload checks passed (model parity not applicable) | `reports/summary/campaign-summary.json`, `reports/reload_verify/reload-report.json` |
| Server backup | **verified** on the rental: 497 files, 549, 641, 178 bytes, receipt `68e56163…` | `reports/summary/campaign-summary.json` |
| Local download | **complete**: all four archives equal the manifest sizes (checked 22: 23 local time, unchanged since 20: 50) | delivery manifest; file sizes |
| Local SHA-256 check | **evidence and backup archives: match** (verified here; Codex's own record for the backup agrees); **frozen and full archives: not checked here** | `delivery-manifest.json`, `backup-download-parts/verified.json` |
| Second round | exploratory experiment after the formal campaign; **complete evidence pending delivery** | — |

Results are in [the experiment report](experiment-report-draft.md) §5. Three delivery states are recorded separately: server backup
(done), download (complete), local SHA-256 check (evidence and backup archives; frozen and full not checked here).

第一轮正式 GPU 流程已完成（本文直接读取证据文件核对）：验证集预选的默认模式仍为 BM25；完整微调未证明优于 BM25；服务器备份已校验，下载已完成，本地 SHA-256 已核对证据包与备份包（冻结与全量模型包未在本文核对）；第二轮为探索实验，完整证据待交付。

## 10. CPU demo of the formal release / 正式发布包 CPU 演示 (2026-10-08)

The evaluated BM25 release `release-ce3160fc…` from the SHA-verified backup archive was served from `e3168ca` on an Apple M4 by Claude.
`serve --check-only`, normal search, empty result, invalid filter (HTTP 422) and the rerank fallback on a model-less release all behaved as
designed; the page's display functions were checked; **the rendered page and screenshots were not executed** (no Streamlit). Record:
[reports/evidence/cpu-demo-20261008](evidence/cpu-demo-20261008/README.md). Demo timings are not performance results.

已在本机 CPU 上演示正式 BM25 发布包：五项接口场景通过，界面渲染与截图未执行；演示耗时不是性能结果。

The [installed 1.2.0 source CPU check](evidence/final966-20261013/installed-source-cpu-check.json) passes preserved966 loading, model parity and recorded smoke replay outside the checkout. This preparation receipt does not substitute for final-source CI or final release-wheel identity.

[安装版1.2.0源码CPU检查](evidence/final966-20261013/installed-source-cpu-check.json)在检出目录之外通过已有966模型加载、模型一致性及已记录短测重放；这项准备回执不能代替最终源码CI或最终发布安装包标识。
