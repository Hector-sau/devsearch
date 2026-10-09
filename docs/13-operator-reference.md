# Operator reference and schemas / 操作员参考与结构说明

Revision 1, 2026-10-05. This page documents what the implemented execution flow does today: `src/devsearch/execution/`, `serving/release.py`, `modeling/`, `training/` and the command modules `prepare`, `prepare_data`, `doctor`, `plan`, `run`, `status`, `bundle`, `rehearsal`, `train`, `predict`, `evaluate`, `export`, `serve`, `retrieve`, `index`, `validate_manifest`. An operator who configures only machine, paths and deadline should be able to run preflight, calibration (doctor), locked plan, training, validation selection, final evaluation, export, reload verification and backup from it. [Documents 10](10-accelerator-execution-contract.md), [11](11-readiness-and-review.md) and [12](12-estimates-and-experiment-plan.md) stay the requirements; where the code differs, the code is described here and the difference is listed in section 4.

第 1 版，2026-10-05。本文记录当前已实现的执行流程实际做什么：`execution/`、`serving/release.py`、`modeling/`、`training/` 及上述命令模块。操作员只需配置机器、路径和截止时间，即可据此完成：自检、校准（doctor）、锁定计划、训练、验证选择、最终评测、导出、重载验证、备份。第 10、11、12 号文档仍是需求；代码与之不同处，本文按代码描述，并在第 4 节列出差异。

Evidence and limits. Every statement comes from reading the working tree on 2026-10-05 (first at HEAD `b9c0da6`; the source then moved while this page was written, and the lead agent patched the statements that later fixes made stale: `serve` without torch for baseline releases, the `status` event-log read, the `prepare_data` seal path, the rehearsal `runtime/run`, the optional `[lengths]` check, `configs/data.toml`, diagnostic detection from rehearsal data. A plan identifies code by `source_snapshot_sha256`, never by commit). No `devsearch` command and no test was run to write this page (only file reads and read-only import checks), and no measured value is quoted; every number in the JSON examples is an illustrative placeholder, not a measurement. Test names show where a behavior is exercised, not that it passed. Tests that need `torch`, `transformers`, `tokenizers` and `safetensors` skip without them (`tests/mltest.py`), and a skip is not a pass. This is an agent reading of the code, not a test report and not human review. Nothing here marks a gate `passed`; see section 5.

证据与局限。所有内容来自 2026-10-05 对工作区代码的阅读（起点 HEAD `b9c0da6`；撰写期间源码继续变化，后续修复使若干表述过期，已由主 Agent 更正：基线发布 `serve` 不再导入 torch、`status` 读取事件日志、`prepare_data` 封存路径、演练自带 `runtime/run`、可选 `[lengths]` 检查、`configs/data.toml`、由演练数据判定诊断；计划以 `source_snapshot_sha256` 而非提交号识别代码）。撰写本文未运行任何 `devsearch` 命令或测试（仅读文件与只读导入检查），也不引用任何实测值；JSON 示例中的数字均为示意占位，不是实测。测试名只说明该行为由哪个测试覆盖，不代表已通过；缺少 `torch` 等库时相关测试跳过，跳过不算通过。本文是 Agent 对代码的阅读，不是测试报告，也不是人工审查。本文不将任何门禁标为 `passed`，见第 5 节。

## 0. Operator path and manual steps / 操作路径与须手工完成的步骤

| # | Step / 步骤 | Command or action / 命令或操作 | Produces / 产物 |
| --- | --- | --- | --- |
| 1 | Install / 安装 | Built wheel plus the `ml` extra (and `serve` for HTTP). `locks/<profile>-cp312.lock` (hashed, resolved for the target platform by `scripts/lock_profile.py`; see `locks/README.md`) and `scripts/bootstrap_env.py download|install` build an offline wheelhouse and a new environment from it; these are unverified on any target host. `scripts/check_wheel.py` installs a wheel without dependencies and checks the baseline commands plus every command's help and bare-invocation refusal, nothing more. Point `TMPDIR` at the runtime volume before `doctor` (probe scratch uses the system temp dir). / 安装 wheel 与 `ml`（HTTP 另需 `serve`）；仓库提供按配置的哈希依赖锁 `locks/` 与 `scripts/bootstrap_env.py`（建离线包、装新环境），均未在任何目标机验证；运行 `doctor` 前把 `TMPDIR` 指向运行卷 | environment |
| 2 | Corpus, encoder / 语料与编码器 | Corpus: `prepare --config sources.toml --output EMPTY_DIR` (pinned snapshots, needs the pinned tokenizer files and, unless `--offline`, network and `git`). Encoder: no command; copy the pinned files into `model-assets/` and call `devsearch.modeling.assets.register_assets` (a Python function; pinned means asset kind `pretrained` with a 40-hex `source.revision`). / 语料用 `prepare`；编码器文件须手工放入并用 Python 函数登记 | corpus artifact + report; `model-assets/` + `assets-manifest.json` |
| 3 | Prepare data / 准备数据 | `prepare_data --stage groups`, `materialize`, then `bank` (prejudged) or `seal-initial` (postrank) | `<bundle>/prepared/`, evaluation dir, `initial-seal.json` |
| 4 | Bundle / 运行包 | Put `protocol.toml` in the bundle, run `bundle assemble --bundle B [--lock F]` | `bundle-manifest.json` |
| 5 | Host / 主机 | Create the volume marker and every directory named in `host.toml` (the code never creates `run_root`, the plan directory, `backup_dir` or any parent), then write `host.toml` | `host.toml` |
| 6 | Receipts, formal only / 凭据（仅正式） | Not a `devsearch` module: `scripts/record_receipt.py implementation --output DIR` and `portable --wheel W --lock L --wheelhouse H --output DIR` (2.13) wrap `devsearch.execution.acceptance.make_receipt`. Keep the `--receipts` directory free of other JSON. / 由 `scripts/record_receipt.py` 生成；回执目录勿放其他 JSON | `*.json` receipts |
| 7 | `doctor` | `doctor --profile P --host H [--bundle B] --output R` | `doctor-report.json` |
| 8 | `plan` | `plan --host H --doctor-report R --output PLAN [--receipts D] [--wheel-sha256 S]` | `execution-plan.json`, `host-binding-0.json` |
| 9 | `run`, `status` | `run --plan PLAN`; after any stop `run --plan PLAN --resume`; `status --plan PLAN` | `run-state.json`, `events.jsonl`, `stages/`, `work/` |
| 10 | Serve / 服务 | `serve --config S --manifest <release> --check-only`, then without `--check-only` | loopback HTTP |

Rehearsal sequence (CPU, synthetic; proves mechanics only): `rehearsal --output /abs/empty`, then `mkdir /abs/empty/plan` (the builder creates `runtime/run` but not the plan directory), then steps 7 to 9 with `--profile cpu-diagnostic --host /abs/empty/host-example.toml`.

上表为操作顺序。演练序列（CPU、合成数据，只证明流程机制）：先 `rehearsal`，再手工创建 `plan` 目录（构建器会创建 `runtime/run`，不创建计划目录），然后以 `cpu-diagnostic` 执行第 7 至 9 步。

### 0.0 One entry point for a rented NVIDIA server / 租用 NVIDIA 服务器的统一入口

`scripts/gpu.py check|prepare|diagnose|run|status --config configs/gpu.toml` (standard library, Python 3.11+) automates steps 1 to 9
below on a Linux NVIDIA host: a new environment from the hashed lock, both receipts, the pinned model downloaded at an immutable commit
with every file verified and its provenance recorded, the pinned sources, the tokenizer comparison, the data rebuild with a
frozen-membership audit, review batches and the coverage audit under the dataset's declared review protocol, the bank and the bundle, a small real-model diagnosis with
an injected crash and resume, and the campaign. It stops with `PREREQUISITE <name>` (exit 3) instead of substituting anything and
refuses (exit 1) any change of model, dataset, backend, lock or protocol after the first `prepare`. Commands, inputs, storage,
restart and cleanup: [GPU_QUICKSTART.md](GPU_QUICKSTART.md). Verified on CPU with fixtures only (`tests/test_gpu.py`).

`scripts/gpu.py` 在 Linux NVIDIA 主机上自动完成下表第 1 至 9 步，缺少前置条件时以 `PREREQUISITE <名称>`（退出码 3）停止，不替代任何内容；首次 `prepare` 后拒绝更换模型、数据、后端、依赖锁或协议。详见 GPU 快速上手；目前仅用样例在 CPU 上验证。

### 0.1 Real-data helper scripts / 真实数据辅助脚本

Not part of the `devsearch` package and not required by any command; they live in the checkout under `scripts/`. They produce the question and qrel drafts that `prepare_data` reads, plus two measurements. Every file they write stays outside the repository (under the runtime root); only counts and hashes belong in `reports/`.

| Script | Purpose / 用途 |
| --- | --- |
| `realdata/select_passages.py --split-manifest M --output-dir EMPTY` | One passage per eligible evidence group (explanatory prose, not a code fragment, 100–1,800 characters, no boilerplate), written as private `selection.jsonl` plus agent batches that carry only `ref`, `title`, `section`, `text` |
| `realdata/carry_generations.py --old-work-dir A --new-work-dir B` | After a corpus rebuild, reuses a generated question only for the same chunk id **and** the same passage text; the rest go to `batches-todo/` |
| `realdata/validate_generations.py --work-dir W --tokenizer-assets D` | Mechanical checks of agent-written questions (one trimmed question, 6–35 words, no hidden-context phrases, no six-word copy, query-token reserve, exact evidence quote, no near duplicate in a split) and the independent review batches (question and passage only) |
| `realdata/build_drafts.py --work-dir W --split-manifest M --output-dir D` | Only questions with an independent review `answerable: yes`, `self_contained: true`, `generic: false` become drafts. Records say `origin: synthetic`, the generator prompt hash and an `agent` review; **nothing here can pass agent review off as human review**. Never overwrites |
| `realdata/prompts/generate-v1.md`, `review-question-v1.md` | The two agent prompts; their hashes are recorded in each draft |
| `realdata/check_source_links.py` | Checks that the source links of sampled BM25 results name existing files and line ranges that contain the passage's words |
| `compare_tokenizers.py --candidate D [--reference D2] --texts F ...` | Compares a tokenizer directory's complete effective configuration and the actual encodings of real texts (alone and paired) with another directory, Transformers' independent pure-Python BERT tokenizer and `AutoTokenizer`; exit 0 only when all are identical. The gate for pinning MiniLM (docs/14 section 1) |
| `realdata/review_pilot.py sample\|convert\|report` | A stratified, blinded pilot of the candidate review: judge batches, conversion into bank responses checked by the bank's parser, agreement per stratum and measured cost per judgment extrapolated to the whole pool. Pilot judgments never enter the bank |
| `realdata/prompts/judge-candidate-v1.md` | The candidate-judging prompt (binary relevance with an exact evidence quote, or `unsure`) |
| `judge_api.py estimate\|run` | Candidate judging through the Claude Message Batches API: the judge-candidate-v1 criteria verbatim, one request per existing 40-item batch, JSON-schema answers, resumable state bound to the pool, batch-file and request-parameter hashes with hash-checked outputs, submission intents recorded before each create and reconciled (exit 7 when duplicate-free recovery cannot be shown), a budget guard (`--budget-usd`, estimate-based) under a platform spend limit; runs only with `--confirm-paid-usage` and credentials (docs/16). Its tests are in `tests/tooling/` because it needs the `anthropic` SDK, which the training lock does not carry |
| `check_model_cpu.py --assets DIR [--require-pretrained]` | CPU check of a registered model-assets directory: hashes, strict weight loading, tokenizer pair encoding and vocabulary, finite forward/backward, an update, eval determinism and a frozen encoder; `gpu.py prepare` runs it after registering the model |
| `measure_baseline.py --index I --search S --queries Q` | BM25-only latency with the declared protocol (warm-up, seeded sequential passes, concurrency 1, no cache); a stand-in workload must be labelled with `--label` |

Questions generated this way are synthetic, and a review by an agent is an agent review. A labelled set belongs to one corpus build: a rebuilt corpus reshuffles split ownership of many groups. / 这样生成的问题是合成问题，Agent 审查仍是 Agent 审查；标注集属于某一次语料构建，语料重建会使许多证据组换划分。

## 1. Command reference / 命令参考

Run everything as `python -m devsearch.<command>`; there are no console scripts. These commands need the `ml` extra (torch, transformers, tokenizers, safetensors) when they run: `doctor`, `plan`, `run`, `status`, `export`, `serve`, `train`, `predict`, `rehearsal` (its encoder builder) and `evaluate --mode rerank`. `serve` needs it only for a release that ships a model (a baseline release is served without importing torch; `tests/test_cli_serve.py` asserts that), and `status` needs it because it builds the whole campaign. `index`, `retrieve`, `validate_manifest`, `prepare_data`, `bundle` and `evaluate --mode baseline` import no ML library (checked by importing the modules they use); `prepare` needs only the `tokenizers` library (no weights). All seventeen commands are implemented (`review` added 2026-10-06); none is a help-only stub any more (`prepare` was one until commit `b9c0da6`).

所有命令以 `python -m devsearch.<命令>` 运行，无控制台脚本。上列命令运行时需要 `ml` 额外依赖；`serve` 仅在发布包含模型时需要（基线发布不导入 torch，有测试断言），`status` 因需构建完整实验对象也需要。`index`、`retrieve`、`validate_manifest`、`prepare_data`、`bundle` 及基线 `evaluate` 不导入机器学习库；`prepare` 只需 `tokenizers` 库（不需权重）。十六个命令均已实现，不再有仅帮助的占位（`prepare` 在提交 `b9c0da6` 之前仍是占位）。

### 1.1 Exit codes / 退出码

| Code / 码 | Standalone commands / 独立命令 | Campaign commands `doctor` `plan` `run` `status` / 实验命令 |
| --- | --- | --- |
| 0 | success; `serve` clean stop or `--check-only` passed / 成功 | `doctor` no failed check; `plan` written; `run` complete; `status` complete and every artifact re-verified |
| 1 | invalid input or failure, reported as `devsearch.<cmd>: error: ...`, no stack trace / 非法输入或失败 | invalid plan, artifact, manifest or config; `doctor` bundle check failed / 计划产物非法 |
| 2 | usage error or unsupported operation / 用法错误或未支持 | `run` usage (state exists without `--resume`, or `--resume` without state); `doctor --profile` differs from the host file; `plan` missing arguments |
| 3 | not used: a missing prerequisite, such as a held lease or missing device for `serve`, is exit 1 / 不使用 | prerequisite unavailable: mount, device, space, lease, calibration; `plan` refused by a failed gate or short window |
| 4 | not used / 不使用 | `run` execution failure or interruption; `status` failed, interrupted, pending or incomplete |
| 5 | not used | awaiting review (`run`, `status`) |
| 6 | not used | budget stop (`run`, `status`) |

A failed standalone command prints `devsearch.<cmd>: error: ...` on stderr without a stack trace and nothing on stdout; apart from the checkpoints of an interrupted `train` and snapshots already acquired by `prepare`, it leaves no published artifact. 独立命令失败时仅在标准错误输出信息，除中断训练的断点和 `prepare` 已获取的快照外，不留已发布产物。

### 1.2 Commands / 各命令

**`prepare`** `--config sources.toml --output EMPTY_DIR [--report FILE] [--offline] [--runtime-root]`: pinned source snapshots in, an immutable corpus artifact and a preparation report out.

- Reads `sources.toml` (strict: `schema_version`, `parser_version`, `chunker_version`; `[policy]` `max_pair_tokens`, `max_query_tokens`, `min_chunk_tokens`, `overlong_code` = `split` or `exclude`, `max_file_bytes`, `max_files`; `[tokenizer] assets`; one or more `[[source]]` with `source_id`, `revision`, `snapshot`, `license`, `include` and optional `git_url`, `url_template`, `license_file`, `exclude`, `snapshot_sha256`), the tokenizer files in `[tokenizer] assets` (no weights; `assets-manifest.json` is read only to report the asset kind and `pinned`, its file hashes are not verified here) and each snapshot as data: Markdown or MDX, reStructuredText, notebook JSON, tutorial Python. Nothing is imported or executed and symbolic links are not followed.
- Acquisition happens only for a missing snapshot and never with `--offline`: `git init` and a depth-1 `git fetch` of the exact commit into a new directory, hooks and LFS disabled, then verification (needs `git`, `git_url` and network; a failed fetch removes the new directory). An existing snapshot is verified and never changed: `verified-git-head`, `verified-content-hash` (with `snapshot_sha256`) or `declared-unverified`.
- Writes the corpus artifact into `--output` (must exist; never created; an identical rerun returns the existing corpus, a different corpus is refused) and a sealed `prepare-report` at `--report`, default `<output>.prepare-report.json` beside the output directory (identical existing report: no-op; different: refused). The report lists quarantined files with reasons, dropped constructs, excluded and duplicate passages, licence evidence (declared, not verified against upstream text) and token statistics with `truncated: 0`; it holds no absolute paths or timestamps. Passages are packed to `max_pair_tokens - max_query_tokens - measured special tokens` and never truncated; over-long code is split at line boundaries or excluded.
- Refuses: `revision` that is not a full 40-hex commit; a `git_url` that is not credential-free `https://` or `file://`, a `url_template` that is not credential-free `https://` with only `{revision}` and `{path}`; paths (output, report, snapshots, tokenizer assets) outside the runtime root; a checkout whose HEAD differs from the pin or with modified tracked files; a content-hash mismatch; more than `max_files` matching files; a `license_file` that is not a regular file; no document yielding any passage. Exit 0 written or identical, 1 invalid input or any failure (including a missing `git` or network), 2 usage. `configs/sources.toml` ships with a deliberately invalid placeholder revision.

**`prepare`：** 把固定版本的来源快照转成不可变语料及准备报告。缺失快照仅在非 `--offline` 时用 git 按精确提交获取（禁用钩子与 LFS）；已有快照只校验、不改动；绝不截断，放不下的内容被切分或排除并写入报告。要求 40 位提交号，路径须在运行根内。需要 `tokenizers` 库，不需模型权重。

**`prepare_data`** `--config data.toml [--stage groups|materialize|bank|seal-initial] --output M [--corpus] [--split-manifest] [--evaluation-dir] [--evaluation-root] [--prepared-manifest] [--evaluation-manifest] [--reviews] [--runtime-root]`

| Stage / 阶段 | Needs / 需要 | Writes / 写入 |
| --- | --- | --- |
| `groups` (default) | `--corpus` | `<out dir>/split-manifest.json`, `assignment.json`, `audit.json`, `corpus/` (ownership-annotated) |
| `materialize` | `--split-manifest` beside `--output`, `--evaluation-dir` | `<prepared>/{data,index,pairs,candidates}/`, `prepared-data-manifest.json`; evaluation dir: `evaluation-manifest.json`, `queries-test.jsonl`, `qrels-test.jsonl`, `candidates/`, and `review/` for the prejudged protocol |
| `bank` | `--prepared-manifest`, `--evaluation-manifest`, `--reviews` | `<evaluation>/bank/`, `<prepared>/initial-seal.json` (prejudged protocol only) |
| `seal-initial` | `--prepared-manifest`, `--evaluation-manifest` | `<prepared>/initial-seal.json`, no bank (postrank protocol only) |

- Refuses: an `--output` file name other than the fixed one (`split-manifest.json`, `prepared-data-manifest.json`, `initial-seal.json`); any path outside the runtime root (`--runtime-root`, else `DEVSEARCH_RUNTIME_ROOT`, else the output's directory; the evaluation dir must lie inside `--evaluation-root`); an evaluation dir that is not separate from the prepared dir; `bank` under the postrank protocol and `seal-initial` under the prejudged one; unresolved or missing review items (never turned into label 0; nothing is written). Calls no generator, reviewer or network. Artifacts are immutable: a different existing manifest is refused.
- For `bank` and `seal-initial`, `--output` must be `initial-seal.json` inside the prepared directory of `--prepared-manifest` (checked before any work), so the printed path is the real one. With an optional `[lengths]` table in `data.toml`, `materialize` first measures every passage and every question of every split (the held-out split included) with the pinned tokenizer and refuses, writing nothing, if any does not fit; the output reports `lengths_checked`. Exit 0, 1, 2.
- 拒绝：固定文件名不符、路径越出运行根、测试评测目录与训练目录未分离、协议与阶段不匹配、审查项缺失或未裁定（不会变成 0，且不写任何产物）。`bank`/`seal-initial` 的 `--output` 必须正是 prepared 目录内的封存文件；`data.toml` 的可选 `[lengths]` 使 `materialize` 在写任何产物前用固定分词器度量所有段落及各划分（含测试）的问题。

**`review`** `batches|import|audit` (standard library): the production candidate review around the blinded `review-request.jsonl`.
`batches --request R --output-dir EMPTY --pass NAME [--batch-size 40] [--seed] [--items FILE|audit.json]` writes compact blinded judge
batches in a seeded order per pass; `import --request R --judgments F... --reviewer-id ID --reviewer-type agent|human --role
primary|adjudicator [--revision REV] [--reviewed-at UTC] --output responses/X.jsonl` turns judgment lines into bank responses checked
by the bank's parser (`unsure`, an unlocatable quote or a malformed line stays unresolved, never 0; never overwrites; agents need
`--revision`, humans have none); `audit --request R --responses-dir D --output audit.json [--data-config data.toml [--hidden F]] [--min-primary-reviews 2] [--merged F]`
applies the data config's declared review protocol (`devsearch.data.review_protocol`): `primary_audit_v1` (revision 5, shipped)
needs one complete primary pass by `primary_reviewer`, the seeded stratified second review it lists as `second_review_queue` (with
escalation of a stratum over the disagreement threshold) and an independent adjudicator for its `adjudication_queue`;
`double_primary` needs agreeing resolved judgments from `min_primary_reviews` distinct primary reviewers or an adjudication.
`batches --items audit.json --queue second-review|adjudication` cuts either queue. The merged file for `prepare_data --stage bank`
is written only when coverage is complete. Exit 0, 1 invalid, 2 usage, 3 coverage incomplete. The bank stage applies the same rule
from the same `data.toml` `[review]` table. 生产审查流程：盲评批次、导入校验、覆盖审计与裁决队列；覆盖不完整时退出 3，不生成供封存的合并文件。

**`index`** `--config index.toml --corpus M --output M [--runtime-root]`: builds a deterministic BM25 index beside `--output` (directory must exist); prints `built` or `unchanged`; identical inputs are a no-op. Exit 0, 1, 2. 建立确定性 BM25 索引，输出目录须已存在。

**`retrieve`** `--config search.toml --mode baseline --index M --query Q [--output F] [--runtime-root]`: only `--mode baseline` without `--checkpoint` runs; `rerank`, `compare` or `--checkpoint` exit 2 (reranked output comes from `evaluate`, `predict` and `serve`). `--output` must resolve inside the runtime root, is never overwritten and its directory is never created. 仅基线；重排请用 `evaluate`、`predict`、`serve`。

**`validate_manifest`** `--manifest M [--runtime-root]`: kinds `corpus` and `index` only (index is rebuilt and compared); anything else exits 1; writes nothing. 仅校验语料与索引，不写文件。

**`train`** `--config train.toml --run-dir DIR [--resume latest|<checkpoint id>] [--runtime-root]`

- Reads `train.toml` (sections `run`, `optimizer`, `batch`, `model`, optional `loss`, `validation`, plus job sections `data`, `assets`, `device`, `policy`; see `configs/train.toml`), the prepared-data manifest (training-visible files only) and the model assets. Writes `<run-dir>/checkpoints/epoch-NNNN.pt|.json`, `latest.json`, `best.json` (pointers, not copies), `events.jsonl`, and `training-summary.json` on completion. Same loop as a campaign, without a deadline, stop hook or lease.
- Refuses: a run dir that does not exist or lies outside the runtime root (it is never created); existing checkpoints without `--resume`; `--resume` of anything but the latest complete-epoch checkpoint (`latest` or its id); a resume whose identity differs in config hash, pair count, `pos_weight`, validation set, length policy, pairs manifest, tokenizer, model assets or code snapshot (a changed recipe or code is a new run); mid-epoch `diagnostic-*` snapshots are never resumable; more than 3 epochs unless `run.diagnostic = true` (then up to 300, with no checkpoint pruning); an unavailable requested device (never CPU; `PYTORCH_ENABLE_MPS_FALLBACK=1` is refused for MPS); OOM, non-finite loss or gradient stop the run with an error and change nothing silently.
- Exit 0 only when status is `completed`, else 1; 2 usage. 退出 0 仅当训练完成。拒绝项见上；训练不自动创建运行目录，也不持有加速器租约。

**`predict`** `--config model.toml --checkpoint <inference bundle dir|checkpoint .json> --pairs P.jsonl --output O.jsonl [--runtime-root]`: input lines `{"pair_id","query","passage"}` (unique ids); output lines `{"pair_id","logit","model_id"}` in input order; a logit is an uncalibrated score. All-or-nothing: over-long pair (nothing is truncated), non-finite logit or any failure writes no output. Output is a new file inside the runtime root. Exit 0, 1, 2. 全部成功才写输出，不截断。

**`evaluate`** `--config model.toml --mode baseline|rerank --index <prepared>/index/index-manifest.json [--checkpoint REF] [--split validation|development|train] --output O.json [--runtime-root]`

- Scores BM25 or a reranker on a training-visible split over the same fixed candidate lists and denominator (queries with at least one confirmed positive; misses score zero). `validation` uses the stored frozen candidates; `development` and `train` regenerate deterministic BM25 depth-50 candidates and need `data/queries-<split>.jsonl` in the prepared data. Output (not sealed): `schema_version`, `kind: "evaluation"`, `mode`, `split`, `corpus_id`, `index_id`, `candidate_source`, `candidate_depth`, `config_sha256`, model fields, `queries_scored`, `groups_scored`, `queries_without_positive`, `positives_missing_from_candidates`, `query_macro`, `group_macro`, `per_query`, `label_provenance`, `recall_note`, `seconds`.
- **`--split test` or any `--stage` exits 2** with an explanation before any file is read: the final-test stages (`rank`, `pool`, `score`) exist only inside a campaign, after `select` has committed, so no test result can exist before models and default mode are locked on validation alone (`World.evaluation`). The output is never overwritten; `rerank` needs `--checkpoint`. Exit 0, 1, 2.
- 最终测试阶段仅在实验流程内、`select` 提交之后运行，`--split test` 与 `--stage` 一律退出 2 且不读任何文件。

**`export`** three forms, each with `--config export.toml --output OUT [--runtime-root]`: `--mode baseline --index M`; `--index M --checkpoint REF`; `--candidate RELEASE --final-report final-report.json`.

- `baseline`: no model, default BM25. `--checkpoint` (an `epoch_boundary` checkpoint with at least one completed epoch, or an inference bundle) gives a `candidate` release (serving needs `--allow-candidate`). The evaluated form promotes a baseline or candidate release with a campaign's `final-report.json` (with `per-query.jsonl` and `report.md` beside it); a report that chose `baseline` drops the model.
- Never overwrites: the release is staged beside `OUT`, **reloaded and verified (hashes, activation rules, model parity, smoke replay) before it is renamed into place**; an identical rerun is a no-op (`published: false`), a different release at the same path is refused. `OUT` must lie inside the runtime root and its parent must exist. Smoke queries: `[smoke] queries` (1 to 8, never test material) for candidates; the candidate's own for evaluated.
- Refuses an already evaluated candidate, a report that does not verify, a report measured on another corpus or index, a report whose selected checkpoint is not the shipped model, or a report choosing `rerank` for a release without a model. Usage errors exit 2 before any work; others 1. 不覆盖；先重载验证再原子发布。

**`serve`** `--config serve.toml --manifest <release dir|release-manifest.json> [--allow-candidate] [--check-only] [--runtime-root]`

- Order of checks: config; `host` is loopback (`127.0.0.1`, `::1`, `localhost`, no authentication exists); release inside the runtime root; `verify_release` (every hash; a `candidate` needs `--allow-candidate`); a model release on an accelerator profile **must hold `accelerator.lease`** (`[serve] lease_path`, relative to the config file and inside the runtime root, normally `<run_root>/accelerator.lease`; set `--runtime-root` or `DEVSEARCH_RUNTIME_ROOT` high enough to contain both); the requested device resolves (never CPU); the pipeline loads; the reranker's recorded parity self-check passes (a failure is never downgraded to BM25); the recorded smoke replays. Only then does it bind `host:port`.
- `--check-only` runs every check, prints the `startup_checks_passed` report and exits 0 without binding. A held lease, a missing device or a damaged file exits 1; a failure after this process took the lease releases it. `[model] batch_size` must equal the batch size recorded in the release (`provenance.inference_batch`; the plan's `training.inference_batch` for a campaign release); another value is refused because the batch size is part of the scoring identity. 仅回环；加速器上的模型发布须持有租约；`--check-only` 不绑定端口。

**`bundle`** `assemble --bundle B [--lock F]` or `verify --bundle B`: `assemble` verifies `prepared/` (all hash chains), `prepared/initial-seal.json` (same corpus, index and protocol), `protocol.toml` and `model-assets/`, then writes `bundle-manifest.json` exclusively; a different existing manifest (also one made under a different code snapshot) is refused, an identical one is returned. `verify` re-checks everything against the manifest. Exit 0, 1, 2. The thin command itself is not exercised by a test (section 3.5). 装配并校验运行包；已有不同清单则拒绝。

**`rehearsal`** `--output EMPTY_DIR [--topics 60] [--seed 7] [--signal planted|none] [--protocol prejudged_fixed_candidates_v1|postrank_pool_v1]`: builds a synthetic diagnostic tree (`bundle/`, `evaluation/`, `backup/`, `runtime/`, `corpus-in/`, `drafts/`, `host-example.toml` for `cpu-diagnostic`, `rehearsal-summary.json`) with a scripted oracle as reviewer and a random tiny encoder. It creates `runtime/run` (the host example's run root) but not the plan directory. The directory must exist and be empty; at least 18 topics. Everything it makes says `diagnostic`. 合成诊断演练，永不满足真实门禁。

**`doctor`** `--profile P --host H [--bundle B] --output R`

- Verifies the bundle and assets, measures the real tokenizer's pair special tokens, checks every passage against the length budget, checks the host binding, then calibrates on the profile's explicit device in fresh worker processes (timeouts and cleanup from `[limits]`). Opens no test file and downloads nothing. Writes only `R` (exclusive; an existing file is refused with exit 1), a one-byte write probe in `run_root` that it removes, and probe scratch under the system temp dir.
- The report is written whenever host file and bundle could be read, also on failure. Exit 0 if no check failed (`unverified` items do not fail, and the doctor's `implementation_verified` and `portable_bundle_verified` are always `unverified`), 1 if the bundle check failed, an artifact or the host file is invalid, or `--bundle` differs from `paths.bundle`, 3 if any other check failed or the host is unhealthy, 2 if `--profile` differs from the host file's profile. The report is bound to the exact host file: after any edit of `host.toml` (even the deadline), `plan` refuses the old report and the doctor must be rerun.

**`plan`**

- Create: `--host H --doctor-report R --output PLAN [--bundle B] [--receipts D] [--wheel-sha256 S]`. Writes `PLAN` and `host-binding-0.json` beside it (the directory must exist; both are immutable). Refuses, writing nothing: with exit 1, an unsealed or edited report, or a report made for another profile, host file, bundle or code snapshot; with exit 3, a failed check or unhealthy host, an uncalibrated profile, a formal plan whose four gates are not all `passed` or `not_applicable`, or too short a window for the estimated work. A diagnostic bundle yields only a plan with `rehearsal: true`. `--receipts` should name a directory holding only receipts (2.13). `run` never re-reads `host.toml`; it uses the binding files.
- Revise: `--revise-binding --plan PLAN --host H [--authorization TEXT]` writes `host-binding-<n+1>.json` beside the plan after fresh mount, hash and access checks (exit 3 on failure). The device profile cannot change (see section 4 for what else can). Exit 0, 1, 2, 3.

**`run`** `--plan PLAN [--resume] [--reviews F]`

- Sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE` to 1 and `TOKENIZERS_PARALLELISM` to false, then opens the campaign: plan hash, bundle, protocol and binding chain are verified before any state is touched (exit 1, or 3 for a missing prerequisite); the code snapshot, tokenizer, profile hash, offline variable, device, caps, mount and window are re-checked by the `preflight` stage. `run_root` must already exist. Without `--resume`, an existing `run-state.json` exits 2; `--resume` without one exits 2. Changes only run-state and artifacts, never the plan or source. Prints nothing on success; exit codes as in 1.1.
- `--reviews F` (postrank protocol only): JSONL, one object per pooled item with exactly `item_id`, `role` (`primary` or `adjudicator`), `relevance` (0, 1, or null when unresolved), `status` (`resolved` or `unresolved`), `evidence_spans` (`[{"start","end"}]`; required for a positive, empty otherwise), `reviewer` (`{"type":"agent"|"human","id","revision"}`; revision null for humans), `reviewed_at`, `notes`. Any unresolved or missing item blocks sealing.

**`status`** `--plan PLAN`: read-only view that re-verifies each committed stage's outputs instead of trusting `run-state.json`; prints JSON (`plan_id`, `status`, `next_action`, `last_error`, `active_binding`, per-stage `status`, `attempt`, `artifact_check`, `error`, `events`, `complete`). Exit 0 complete and verified, 5 awaiting review, 6 budget stop, 4 in every other case (also when an artifact check failed), 1 invalid plan, bundle or unreadable state.

### 1.3 Config files / 配置文件

All TOML, strict (unknown or missing keys fail, nothing defaulted); relative paths resolve against the config file. Examples: `configs/*.toml` (`sources`, `train`, `eval`, `predict`, `export`, `serve`, `protocol`, `search`, `index`); all but `index` and `search` carry placeholder paths or an invalid placeholder revision and cannot run unchanged. `configs/data.toml` ships with a deliberately invalid placeholder `review.salt` (and optional `[lengths]`); its format is also in the `data/config.py` docstring. Model-job files (`predict`, `evaluate`, `export`, `serve`) share `schema_version`, `[device] profile`, `[model] batch_size` (optional `dropout`) and optional `[assets] directory` and `[policy]` (needed only to rebuild a model from a training checkpoint); `export` adds `[search] path` and `[smoke] queries`; `serve` adds `[serve] host port deadline_seconds http_grace_seconds [lease_path]`. `protocol.toml` is frozen before training and hashed into the plan (matrix, epochs at most 3, selection gates, latency, bootstrap, `[inference] batch_candidates`).

配置均为严格 TOML，路径相对配置文件解析。示例见 `configs/`；`configs/data.toml` 含故意无效的占位盐值。`protocol.toml` 在训练前冻结并计入计划哈希。

## 2. File schemas / 文件结构

### 2.1 Conventions / 约定

Sealed JSON files carry `schema_version: 1`, `kind` and `manifest_sha256`, the SHA-256 of the canonical JSON (sorted keys, compact separators) of everything else; loaders reject unknown or missing fields, duplicate keys and NaN or infinity. Files are written to a temporary file on the destination filesystem, fsynced and renamed; "never overwritten" means exclusive creation. `run-state.json`, `validation-selection.json`, `preflight.json` and most other stage outputs are plain JSON; their integrity comes from the stage manifest (hash and size) and, for the selection, from `selection_id`. A few carry their own seal: `ranked-output-manifest.json`, `test-seal-final.json`, `acceptance-report.json`, `backup-receipt.json`. Identifiers: `plan_id = "plan-" + manifest_sha256[:32]`, `binding_id = "binding-" + manifest_sha256[:32]`. Labels in reports: `measured` (taken on this host with these assets), `fixture` (diagnostic assets or data), `declared` (stated, not observed), `estimate` (derived from measured rates, never a ranking result). Gate statuses: `passed`, `failed`, `unverified`, `not_applicable`.

密封 JSON 含 `schema_version`、`kind`、`manifest_sha256`（其余内容规范 JSON 的 SHA-256），严格拒绝未知或缺失字段。文件先写同文件系统临时文件、刷盘再重命名。运行状态和各阶段输出本身不密封，完整性靠阶段清单（哈希与大小）。标签：`measured` 实测、`fixture` 样例、`declared` 声明、`estimate` 估算。门禁状态：`passed`、`failed`、`unverified`、`not_applicable`。

### 2.2 Where files live / 文件位置

```text
<mount.path>/                        volume; <marker> file holds expected_id
  <runtime_root>/                    checked (exists, inside the mount); the campaign writes under run_root
  <bundle>/                          paths.bundle, inside the mount
    bundle-manifest.json  protocol.toml
    model-assets/   config.json model.safetensors vocab.txt|tokenizer.json [tokenizer_config.json special_tokens_map.json added_tokens.json] assets-manifest.json
    prepared/       split-manifest.json assignment.json audit.json corpus/ index/ data/ pairs/ candidates/
                    prepared-data-manifest.json initial-seal.json
  <run_root>/                        paths.run_root, inside the mount, must exist
    run-state.json  events.jsonl  campaign.lock  accelerator.lease (accelerator profiles; removed on release)
    stages/<stage_id>/stage-manifest.json
    stages/<stage_id>/<attempt_id>/<outputs>            outputs: section 3.1
    stages/<stage_id>/.abandoned-<attempt_id>/          uncommitted attempts, kept as evidence
    work/<stage_id with ':' as '-'>/checkpoints/        epoch-NNNN.pt|json latest.json best.json; training-summary.json
<evaluation_root>/                   paths.evaluation_root (a path, not required to be inside the mount)
    evaluation-manifest.json queries-test.jsonl qrels-test.jsonl candidates/ review/ bank/
<backup_dir>/<plan_id>/              copies plus backup-receipt.json
<plan directory>/                    any existing directory: execution-plan.json host-binding-0.json [host-binding-<n>.json]
```


Tokenizer files: `tokenizer.json` carries the complete pipeline. Without it, `vocab.txt` builds a BERT WordPiece pipeline that honours `do_lower_case`, `strip_accents` and `tokenize_chinese_chars` from `tokenizer_config.json` and **refuses** any setting it cannot reproduce (`do_basic_tokenize: false`, `never_split`, special tokens other than the BERT defaults, additional special tokens, entries in `added_tokens.json`, an `added_tokens_decoder` entry that is not the vocabulary entry at that id); all present tokenizer files form the tokenizer fingerprint. `scripts/compare_tokenizers.py` proves the encodings against independent references (docs/14 section 1). / 分词文件：无 `tokenizer.json` 时，`vocab.txt` 构建的 BERT 流程遵循 `do_lower_case`、`strip_accents`、`tokenize_chinese_chars`，对无法复现的设置一律拒绝；所有分词文件都计入指纹。

Stage ids contain colons, so directories such as `stages/train:full:42/` exist on disk. 阶段 ID 含冒号，磁盘上目录名也含冒号。

### 2.3 `host.toml`

```toml
schema_version = 1
[device]
profile = "apple-mps"             # apple-mps | nvidia-cuda | cpu-diagnostic; equals `doctor --profile`
[paths]                           # absolute, must already exist, no symbolic links
runtime_root    = "/Volumes/X/devsearch"
bundle          = "/Volumes/X/devsearch/bundle"
run_root        = "/Volumes/X/devsearch/runs/c1"
evaluation_root = "/Volumes/E/devsearch-eval"
backup_dir      = "/Volumes/B/devsearch-backup"
[mount]
path = "/Volumes/X"               # runtime_root, bundle and run_root must lie inside it
marker = ".devsearch-volume"      # plain file name inside `path`; its stripped content must equal expected_id
expected_id = "x-2026-10"
require_mount_point = true        # false is accepted but recorded as an `unverified` mount check
[deadline]
at = "2026-10-06T08:00:00Z"       # UTC, exactly this form
reserve_seconds = 1800            # 0..604800, kept free for export, reload and backup
[resources]
min_free_bytes = 30000000000
host_memory_cap_bytes = 17179869184    # must be > 0 (doctor fails a missing cap)
device_memory_cap_bytes = 17179869184  # accelerator: > 0 and <= observed capacity; 0 only for cpu-diagnostic
[limits]
process_timeout_seconds = 300     # 1..86400
doctor_budget_seconds = 1800      # 1..604800
cleanup_grace_seconds = 30        # 0..3600
io_retries = 2                    # 0..2
[backup]
require_independent_device = true # false is recorded as `unverified`
```

All keys are required; there is no credential field. 所有键必填，无凭据字段。

### 2.4 `host-binding-<n>.json` and the binding chain / 主机绑定与链

Sealed, `kind: "host-binding"`. Keys: `revision`, `parent_sha256` (null for revision 0), `paths` (the five paths), `mount` (`path`, `marker`, `expected_id`, `require_mount_point`), `deadline`, `reserve_seconds`, `limits` (the four limits plus `require_independent_backup`), `resources` (three byte values), `device` (`{"profile"}`), `authorization` (text or null). The plan holds only `host_binding.{id,sha256}` of revision 0 and no host paths. At every `run`, `status` and `plan --revise-binding`, the active binding is the highest revision among the `host-binding-*.json` files beside the plan, and its chain is verified: every parent present, hashes valid, revisions consecutive from 0, revision 0 equal to the plan's hash. `run-state.json` records the active `{id, revision, sha256}`; deadline, reserve, paths and caps used at run time are those of the active revision. 计划只引用第 0 版绑定哈希；运行取最高修订版并核验整条链。

### 2.5 `doctor-report.json`

Sealed, `kind: "doctor-report"`. Illustrative, abridged:

```json
{
  "schema_version": 1, "kind": "doctor-report", "profile": "apple-mps",
  "host_binding_sha256": "<sha256 of host-binding revision 0>", "bundle_manifest_sha256": "<sha256>", "diagnostic": false,
  "code": {"package": "devsearch", "version": "0.0.0", "source_snapshot_sha256": "<sha256>", "source_file_count": 100},
  "checks": [
    {"name": "bundle", "status": "passed", "label": "measured", "detail": "every part verifies; corpus <id>"},
    {"name": "model_assets_pinned", "status": "passed", "label": "declared", "detail": "a real pretrained model at a pinned 40-hex revision"},
    {"name": "mount", "status": "passed", "label": "measured", "detail": "/Volumes/X is a mount point with the expected volume identity"}
  ],
  "calibration": {
    "attempts": [{"microbatch": 2, "ok": true, "reason": null, "error": null, "memory": {"host_peak_rss_bytes": 0, "device_peak_bytes": 0}}],
    "microbatch": 2, "inference_batch": 4,
    "training": {"full": {"ok": true, "kind": "train_timing", "mode": "full", "microbatch": 2, "accumulation_steps": 8, "updates": 55, "warmup": 5, "measured": 50,
                          "trainable_parameters": 0, "finite": true, "seconds_per_update": {"mean": 0.0, "min": 0.0, "max": 0.0},
                          "windows": [0.0, 0.0, 0.0, 0.0, 0.0], "pairs_per_second": 0.0, "memory": {"host_peak_rss_bytes": 0, "device_peak_bytes": 0}, "memory_cap_problem": null}},
    "inference_attempts": [{"batch": 4, "ok": true, "reason": null}],
    "inference": {"ok": true, "batch": 4, "pairs_per_second": 0.0, "windows": [0.0, 0.0, 0.0, 0.0, 0.0]},
    "io": {"ok": true, "checkpoint_save_seconds": 0.0, "checkpoint_load_seconds": 0.0, "export_seconds": 0.0, "reload_seconds": 0.0, "reload_parity": {"passed": true}, "scratch_bytes": 0},
    "stopped": null, "unhealthy_host": false,
    "backend": {"profile": "apple-mps", "device": "mps", "torch": "<version>", "python": "<version>", "platform": "<platform>", "machine": "arm64", "tf32": false}
  },
  "gates": {
    "implementation_verified": {"status": "unverified", "label": "declared", "detail": "evidence comes from the recorded test receipt"},
    "data_ready": {"status": "passed", "label": "measured", "detail": "the bundle verifies and every passage fits the budget"},
    "portable_bundle_verified": {"status": "unverified", "label": "declared", "detail": "needs the clean-install rehearsal receipt"},
    "backend_verified[apple-mps]": {"status": "passed", "label": "measured", "detail": "real pinned assets trained, timed, saved and reloaded", "fingerprint": {"profile": "apple-mps"}}
  },
  "estimates": {
    "label": "estimate",
    "train": {"full": {"pairs_per_second": 0.0, "trainable_parameters": 0, "peak_memory": {}}},
    "inference_pairs_per_second": 0.0,
    "io": {"checkpoint_save_seconds": 0.0, "checkpoint_load_seconds": 0.0, "export_seconds": 0.0, "reload_seconds": 0.0, "scratch_bytes": 0},
    "bundle_bytes": 0, "note": "rates were measured on diagnostic-length synthetic pairs; derived times are estimates, not results",
    "capacity": {"label": "estimate", "status": "estimated", "runs": 2, "checkpoints_bytes": 0, "exports_bytes": 0, "release_bytes": 0, "backup_bytes": 0, "temporary_bytes": 0, "margin": 1.2, "bytes": 0, "detail": "models only plus bundle copies"}
  },
  "limits": {"process_timeout_seconds": 300, "doctor_budget_seconds": 1800, "cleanup_grace_seconds": 30, "io_retries": 2},
  "elapsed_seconds": 0.0, "created_at": "2026-10-05T09:00:00Z", "unhealthy_host": false, "manifest_sha256": "<sha256>"
}
```

Check names and rules (every check is `{name, status, detail, label}`):

| name | passed | failed | unverified |
| --- | --- | --- | --- |
| `bundle` | bundle verifies | any manifest error (calibration is then skipped) | n/a |
| `model_assets_pinned` | pretrained at a 40-hex revision | n/a | diagnostic or unpinned assets |
| `code_identity` | installed code equals the bundle's snapshot | n/a | differs (the plan binds the running code) |
| `corpus_length_budget` | every chunk fits `max_pair_tokens - max_query_tokens - measured special tokens` | any chunk over | n/a |
| `host_memory_cap` | 0 < cap <= physical memory | cap is 0 or above physical | physical memory unreadable |
| `device_memory_cap` (emitted only on failure), `device_memory_cap_vs_capacity` | cap <= observed capacity | accelerator with cap 0, or cap above capacity | n/a (omitted when the capacity is unknown; the run's `preflight` then refuses a nonzero cap) |
| `mount`, `path:<root>` (five), `containment:<root>` (emitted only on failure), `writable:run_root`, `free_space:run_root`, `backup_independent`, `deadline` | the mount point, marker, directories and a probe write are fine; free space is at least `max(min_free_bytes, capacity estimate)`; the backup is on another device; time remains after the reserve | missing directory or marker, symlink, not a mount point, unwritable, free space too low, backup on the run root's device when independence is required, no time left after the reserve | `require_mount_point = false`; independence not required; free space unreadable |
| `calibration` | a microbatch and an inference batch were chosen and nothing stopped | otherwise | n/a |

`pair_special_tokens` is informational. Calibration: each microbatch of the profile's declared order (`apple-mps` `[2,1,4,8]`, `nvidia-cuda` `[4,2,1,8,16]`, `cpu-diagnostic` `[4,2,1,8]`) is probed once, in a fresh process, for three optimizer updates at the maximum length in `full` mode (learning rate of the last recipe); the first that runs and stays under the caps wins. Then, per recipe, 5 warmup plus 50 measured updates (five windows of ten), the inference batch from the protocol's `batch_candidates` with its own probe, 5 plus 50 inference batches, and a save, load, export, reload and parity timing. A worker that cannot be confirmed gone after terminate and kill marks the host unhealthy and no further worker starts. `data_ready` is `unverified` for a diagnostic bundle and otherwise `passed` when the bundle verifies and the length budget holds (structural only; see section 5). `backend_verified[<profile>]`: `not_applicable` for `cpu-diagnostic`, `failed` when no worker reached the device or probes failed, `unverified` for diagnostic or unpinned assets on an accelerator, `passed` only for pinned assets that trained, timed, saved and reloaded on the requested device, and only when **every** measurement is successful and finite: each recipe's training timing, the inference timing, and the save/load/export/reload probe with a passing reload parity (a positive finite `pairs_per_second` over at least one measured unit, finite non-negative timings, `finite: true`). A failed or missing inference timing fails the gate and the detail names what is missing. 报告永远不是批准：通过门禁需要计划阶段的凭据与真实固定资产。

### 2.6 `execution-plan.json`

Sealed, `kind: "execution-plan"`; contains hashes and policies, no host paths and no future checkpoint, selection or report hashes (`validation-selection.json` and the final seal bind those later). Illustrative, abridged (`stages` is listed in section 3.1):

```json
{
  "schema_version": 1, "kind": "execution-plan", "rehearsal": false,
  "code": {"identity": {"package": "devsearch", "version": "0.0.0", "source_snapshot_sha256": "<sha256>", "source_file_count": 100}, "wheel_sha256": null, "lock_sha256": null},
  "bundle": {"manifest_sha256": "<sha256>", "protocol_id": "protocol-<32 hex>", "protocol_sha256": "<sha256>", "diagnostic": false},
  "profile": {"name": "apple-mps", "sha256": "<sha256 of the packaged profile>", "device": "mps", "accelerator": true, "declared_verification": "unverified"},
  "backend": {"fingerprint": {"profile": "apple-mps", "device": "mps", "torch": "<version>", "tf32": false}, "doctor_backend_gate": "passed"},
  "model_assets": {"manifest_sha256": "<sha256>", "asset_kind": "pretrained", "pinned": true, "tokenizer_fingerprint": "<sha256>", "pair_special_tokens": 3},
  "data": {"corpus_id": "<id>", "index_id": "<id>", "prepared_manifest_sha256": "<sha256>", "split_manifest_sha256": "<sha256>", "pairs_manifest_sha256": "<sha256>",
           "sampling_manifest_id": "<id>", "validation_candidates_manifest_sha256": "<sha256>", "initial_seal_sha256": "<sha256>", "evaluation_commitment_sha256": "<sha256>",
           "evaluation_protocol": "prejudged_fixed_candidates_v1",
           "counts": {"train": {"queries": 0, "confirmed_positives": 0}, "validation": {"queries": 0, "confirmed_positives": 0}, "test": {"queries": 0, "confirmed_positives": 0}, "pairs": 0, "validation_candidates": 0}},
  "initialization": {"encoder": "re-loaded from the pinned assets for every run; diagnostic weights are discarded", "head_seed": 1234, "diagnostic_weights_discarded": true, "resume": "epoch_boundary checkpoints only"},
  "matrix": [{"stage_id": "train:frozen:42", "recipe_id": "frozen", "mode": "frozen", "seed": 42, "epochs": 3, "learning_rate": 0.001}],
  "training": {"precision": "fp32", "workers": 0, "effective_batch": 16, "microbatch": 2, "accumulation_steps": 8, "max_pair_tokens": 256, "max_query_tokens": 64, "passage_budget": 189,
               "inference_batch": 4, "shared": {"dropout": 0.1, "weight_decay": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "max_grad_norm": 1.0, "warmup_fraction": 0.1, "effective_batch": 16, "pos_weight": "auto", "validation_batch": 8}},
  "policies": {"protocol_id": "protocol-<32 hex>", "selection": {"min_group_mrr_gain": 0.02, "max_p95_ms": 2000}, "latency": {"warmup": 10, "passes": 3, "queries": 50, "seed": 20261005},
               "bootstrap": {"confidence": 0.95, "repetitions": 10000, "seed": 20261005}, "judging": {"protocol": "prejudged_fixed_candidates_v1", "no_test_based_selection": true},
               "search": {"candidate_k": 50, "top_k": 5, "max_candidate_k": 100, "max_query_chars": 1000}},
  "stages": [{"stage_id": "preflight", "needs": []}, {"stage_id": "data_check", "needs": ["preflight"]}],
  "retries": {"setup_io": 2, "training": 0, "note": "OOM, non-finite values and device loss stop with evidence; a changed recipe needs a new plan"},
  "budget": {"initial_deadline": "2026-10-06T08:00:00Z", "initial_reserve_seconds": 1800,
             "estimates": {"label": "estimate", "uncertainty_margin": 1.25, "stage_seconds": {"train:frozen:42": 0.0, "latency": null}, "total_seconds": 0.0, "unverified_stages": ["latency"], "note": "derived from measured rates"},
             "window_seconds_at_planning": 0.0},
  "capacity": {"label": "estimate", "status": "estimated", "bytes": 0},
  "host_binding": {"id": "binding-<32 hex>", "sha256": "<sha256>"},
  "gates": {"data_ready": {"status": "passed", "label": "measured", "detail": "...", "evidence": []}},
  "doctor_report_sha256": "<sha256>", "manifest_sha256": "<sha256>"
}
```

Host-binding chain: `host_binding.sha256` is revision 0, written as `host-binding-0.json` beside the plan; later revisions are separate files that point to their parent (2.4). Seed extension (`seeds = [42, 43, 44]` in `protocol.toml`) expands `matrix` and the train and export stages; latency always uses the deployment seed. The `gates` object holds all four gates after receipts were applied (2.13).

### 2.7 `run-state.json` and `events.jsonl`

`run-state.json` is not sealed; exact keys are validated and it is replaced atomically.

```json
{
  "schema_version": 1, "kind": "run-state", "plan_id": "plan-<32 hex>", "plan_sha256": "<sha256>", "status": "running",
  "stages": {
    "preflight": {"status": "succeeded", "attempt": 1, "attempt_id": "attempt-<16 hex>", "inputs": {}, "outputs": {"preflight.json": {"sha256": "<sha256>", "bytes": 0}},
                  "started_at": "2026-10-05T09:00:00Z", "finished_at": "2026-10-05T09:00:03Z", "error": null},
    "train:full:42": {"status": "failed", "attempt": 1, "attempt_id": "attempt-<16 hex>", "inputs": {"data_check": "<sha256>"}, "outputs": {},
                      "started_at": "2026-10-05T09:01:00Z", "finished_at": "2026-10-05T09:10:00Z", "error": {"kind": "interrupted", "message": "interrupted"}}
  },
  "active_binding": {"id": "binding-<32 hex>", "revision": 0, "sha256": "<sha256>"},
  "lease": {"host": "<hostname>", "pid": 4242, "start": "<process start token>", "token": "<32 hex>", "purpose": "campaign", "plan_id": "plan-<32 hex>"},
  "next_action": "running stage train:full:42", "last_error": null, "updated_at": "2026-10-05T09:10:00Z"
}
```

Stage and campaign statuses: `pending`, `running`, `succeeded`, `failed`, `awaiting_review`, `stopped_budget`. Stage `error.kind`: `interrupted` (resumable), `invalid_input`, `fatal`, `budget`. `last_error` is `{message, exit_code}`. `events.jsonl` is append-only, fsynced per line, one canonical JSON object per line with `seq`, `ts`, `event` plus fields; an incomplete last line from a crash is discarded when the log is opened. Events: `campaign_started`, `stage_started`, `stage_succeeded|failed|awaiting_review|stopped_budget` (with `attempt_id`, `seconds`, outputs or error), `stage_reconstructed`, `stage_interrupted_found`, `stage_not_started_budget`, `budget_stop`, `campaign_stopped`, `campaign_succeeded`, and training events (`resumed`, `checkpoint_saved`, `epoch_start`, `epoch_end`, `stopped`) limited to `epoch`, `global_step`, `train_loss`, `reason`: no test content. 事件日志追加写，训练事件不含测试内容。

### 2.8 `stages/<stage_id>/stage-manifest.json`

Sealed, `kind: "stage"`; the commit record of one attempt.

```json
{
  "schema_version": 1, "kind": "stage", "plan_id": "plan-<32 hex>", "stage_id": "select", "attempt_id": "attempt-<16 hex>", "status": "succeeded",
  "inputs": {"baseline_validation": "<manifest_sha256 of that stage>", "latency": "<sha256>", "train:full:42": "<sha256>"},
  "outputs": {"validation-selection.json": {"sha256": "<sha256>", "bytes": 0}},
  "notes": {"default_mode": "baseline", "selection_id": "selection-<32 hex>"}, "manifest_sha256": "<sha256>"
}
```

`status` is `succeeded` or `awaiting_review` (the only committed values). `inputs` maps each needed stage to that stage's `manifest_sha256`; `outputs` are files under `<attempt_id>/` (regular files, no symlinks). A fresh `attempt_id` (`attempt-` plus 16 hex) is made for every execution of a stage. Reuse rules: section 3.2.

### 2.9 `bundle-manifest.json` and `assets-manifest.json`

```json
{
  "schema_version": 1, "kind": "bundle",
  "code": {"package": "devsearch", "version": "0.0.0", "source_snapshot_sha256": "<sha256>", "source_file_count": 100},
  "prepared": {"manifest_path": "prepared/prepared-data-manifest.json", "manifest_sha256": "<sha256>", "corpus_id": "<id>", "index_id": "<id>", "protocol": "prejudged_fixed_candidates_v1"},
  "initial_seal": {"path": "prepared/initial-seal.json", "manifest_sha256": "<sha256>"},
  "protocol": {"path": "protocol.toml", "sha256": "<sha256>", "protocol_id": "protocol-<32 hex>"},
  "model_assets": {"manifest_path": "model-assets/assets-manifest.json", "manifest_sha256": "<sha256>", "asset_kind": "pretrained", "pinned": true},
  "diagnostic": false, "lock": {"sha256": "<sha256>", "name": "<lock file name>"}, "manifest_sha256": "<sha256>"
}
```

`lock` is null when `--lock` was not given. `diagnostic` is true exactly when the model assets are `diagnostic`. The test side is not in the bundle: only the hash commitment (`evaluation_commitment` in the prepared manifest, the stage-1 seal) is. `assets-manifest.json` (sealed, `kind: "model-assets"`): `asset_kind` (`pretrained` or `diagnostic`), `source` (free-form object; `revision` decides `pinned`), `pinned`, `files` (`{name: {sha256, bytes}}`, regular files only), `encoder` (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `vocab_size`, `max_position_embeddings`), `tokenizer_files`. Weights are `model.safetensors` only; pickled `.bin` is never loaded. 评测侧文件不在运行包内，只有哈希承诺。

### 2.10 `final-report.json`, `per-query.jsonl`, `report.md`

Written by the `score` stage; `final-report.json` is JSON with `final_report_id = "report-" + digest(body)[:32]` (no `manifest_sha256`). Abridged:

```json
{
  "schema_version": 1, "kind": "final-report", "status": "complete", "plan_id": "plan-<32 hex>", "protocol_id": "protocol-<32 hex>", "selection_id": "selection-<32 hex>",
  "default_mode": "rerank", "selected": {"recipe_id": "full", "seed": 42, "checkpoint_id": "epoch-0002", "checkpoint_sha256": "<sha256>"},
  "denominator": {"queries_scored": 0, "groups_scored": 0, "queries_without_confirmed_positive": [], "rule": "queries with at least one confirmed positive in the final qrels; a retrieval miss scores zero and stays in"},
  "systems": [{"system": "bm25", "role": "baseline", "queries_scored": 0, "groups_scored": 0,
               "query_macro": {"mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}, "group_macro": {"mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}},
              {"system": "full-seed42", "role": "locked", "recipe_id": "full", "seed": 42, "checkpoint_id": "epoch-0002", "checkpoint_sha256": "<sha256>", "model_id": "model-<32 hex>",
               "queries_scored": 0, "groups_scored": 0, "query_macro": {"mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}, "group_macro": {"mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}}],
  "comparisons": [{"system": "full-seed42", "baseline": "bm25", "recipe_id": "full", "seed": 42, "primary": true, "deployment_seed_comparison": true,
                   "differences": {"mrr@10": {"observed_difference": 0.0, "interval": [0.0, 0.0], "confidence": 0.95, "repetitions": 10000, "seed": 20261005, "groups": 0, "interval_excludes_zero": false, "outcome": "inconclusive: the interval for the group-macro difference spans zero; a stable gain is not supported"},
                                   "ndcg@10": {"observed_difference": 0.0, "interval": [0.0, 0.0], "confidence": 0.95, "repetitions": 10000, "seed": 20261005, "groups": 0, "interval_excludes_zero": false, "outcome": "inconclusive: ..."}}}],
  "recipe_means": [], "bootstrap": {"kind": "paired group bootstrap", "confidence": 0.95, "repetitions": 10000, "seed": 20261005},
  "provenance": {"query_origins": {"synthetic": 0}, "reviewers": {"agent": 0}, "diagnostic_rehearsal": false, "evaluation_protocol": "prejudged_fixed_candidates_v1", "corpus_id": "<id>", "index_id": "<id>"},
  "seals": {"initial": "<sha256>", "final": "<sha256>", "bank": "<sha256>", "ranked_outputs": "<sha256>"},
  "per_query_sha256": "<sha256>", "limitations": ["Recall@50 is recall of known positives inside the fixed natural BM25 top-50; it is not exhaustive relevance."],
  "final_report_id": "report-<32 hex>"
}
```

`per-query.jsonl`: canonical JSONL, one row per system and scored query, sorted by system then query: `{"system": "bm25", "query_id": "<id>", "group_id": "<id>", "mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0, "top10": ["<chunk_id>"]}`. `recipe_means` is filled only when more than one seed is declared. `report.md` is the short human summary of the same numbers. The limitations list states synthetic queries, agent review and diagnostic rehearsal when they apply. 报告 ID 为正文哈希，任何编辑可被发现；系统名为 `bm25` 与 `<recipe>-seed<seed>`。

### 2.11 `release-manifest.json` and `smoke/smoke.json`

Release layout: `release-manifest.json`, `corpus/`, `index/`, `model/` (`model-manifest.json`, `encoder.safetensors`, `head.safetensors`, `encoder-config.json`, `parity-examples.json`, `tokenizer/`; absent for baseline), `config/search.toml`, `evaluation/` (`final-report.json`, `per-query.jsonl`, `report.md`; evaluated only), `smoke/smoke.json`. In a campaign it is `stages/export_release/<attempt_id>/release/`.

```json
{
  "schema_version": 1, "kind": "release", "status": "evaluated", "default_mode": "rerank",
  "corpus": {"path": "corpus/corpus-manifest.json"}, "index": {"path": "index/index-manifest.json"},
  "model": {"path": "model/model-manifest.json", "model_id": "model-<32 hex>"},
  "search_config": {"path": "config/search.toml"},
  "evaluation": {"report_path": "evaluation/final-report.json", "final_report_id": "report-<32 hex>", "per_query_sha256": "<sha256>", "selection_id": "selection-<32 hex>"},
  "smoke": {"path": "smoke/smoke.json"},
  "files": {"smoke/smoke.json": {"sha256": "<sha256>", "bytes": 0}},
  "provenance": {"plan_id": "plan-<32 hex>", "selection_id": "selection-<32 hex>", "rehearsal": false, "validation_only_smoke": true,
                 "code": {"package": "devsearch", "version": "0.0.0", "source_snapshot_sha256": "<sha256>", "source_file_count": 100}},
  "manifest_sha256": "<sha256>", "release_id": "release-<32 hex>"
}
```

`status`: `baseline` (no model, default BM25), `candidate` (a model without final evaluation; serving needs `--allow-candidate`), `evaluated` (final report bound to the shipped corpus, index and, for `rerank`, the shipped checkpoint). `model` and `evaluation` are null when absent. `files` lists every other file with SHA-256 and size; any added, changed or missing file, or a symlink, blocks activation. `smoke.json` (listed in `files`, not sealed):

```json
{
  "schema_version": 1, "kind": "smoke", "top_k": 5, "logit_atol": 0.0001, "note": "validation queries only; never test material",
  "queries": [{"query": "How does gradient accumulation work?", "sources": [], "versions": [["<source_id>", "<revision>"]],
               "candidate_set_id": "<id>", "baseline_ids": ["<chunk_id>"], "rerank": {"ids": ["<chunk_id>"], "logits": [0.0]}}]
}
```

`rerank` is null when the release has no ready reranker. Replay rule: BM25 candidate set and top ids must match exactly; reranked ids and logits must match within `logit_atol` (near ties within the tolerance may reorder). The campaign uses the first 8 validation queries; `export` uses the operator's `[smoke] queries`.

### 2.12 `test-seal` stage 2, `reload-report.json`, `validation-selection.json`

`test-seal-final.json` (sealed, `kind: "test-seal"`, `stage: 2`): `protocol`, `corpus_id`, `index_id`, `initial_seal_sha256`, `selection_id`, `systems` (`{system: sha256 of ranked file}`), `bank_manifest_sha256`, `qrels_v2_sha256`, `coverage` (`required_pairs`, `judged_pairs`, and `pool_depth` for postrank). `validation-selection.json` (output of `select`, bound by `selection_id` and the stage manifest): `baseline_validation`, `thresholds`, `recipes[]` (per seed `best_epoch`, `checkpoint_id`, `checkpoint_sha256`, `group_mrr`, `group_ndcg`, `mrr_gain`, `ndcg_change`, `epochs[]`; per recipe means, `mean_quality_ok`, `deployment_seed_quality_ok`, `deployment_p95_ms`, `latency_ok`, `qualifies`), `decision` (`default_mode`, `recipe_id`, `seed`, `checkpoint_id`, `checkpoint_sha256`, `reason`), `selection_id`, `latency_sha256`. `reload-report.json`: `release_id`, `checks` (`hashes_and_activation_rules`, `model_parity_after_reload`, `rerank_ready`, `smoke_replay`, `api_smoke`, each with a `status`), `failed`.

### 2.13 Receipts and `acceptance-report.json` / 回执与验收报告

Gates `implementation_verified` and `portable_bundle_verified` cannot come from the doctor. They come from sealed receipts in the `--receipts` directory (every `*.json` there is read strictly):

```json
{
  "schema_version": 1, "kind": "receipt", "gate": "implementation_verified", "code_source_snapshot_sha256": "<sha256 of the code that ran>",
  "commands": [{"argv": ["python", "-m", "unittest", "discover"], "exit_status": 0, "stdout_sha256": "<sha256>"}],
  "summary": {"tests_run": 0, "failures": 0, "errors": 0, "skipped": 0}, "environment": {"python": "<version>"}, "created_at": "2026-10-05T09:00:00Z", "manifest_sha256": "<sha256>"
}
```

Rules (`execution/acceptance.py`): no receipt for the gate gives `unverified`; receipts only for other code give `unverified`; the newest receipt for this code with a command whose `exit_status` is not 0 gives `failed`. `implementation_verified` needs `tests_run > 0`, no `failures` or `errors`, and `skipped == 0` (a skip is not a pass). `portable_bundle_verified` needs `summary.outside_checkout`, `summary.rehearsal_complete` and `summary.offline` all true **and** dependency evidence in `summary.dependencies`: a 64-hex `lock_sha256`, `pip_check_exit` 0, `installed_matches_lock` true (the installed packages are exactly the locked ones), `borrowed_dependencies` false with no `foreign_sys_path_entries`, and a non-zero `wheelhouse_wheels`. A receipt without that evidence is `failed` ("the install must come from a hashed lock and an offline wheelhouse"). When the plan's bundle names a lock (`bundle assemble --lock`), the receipt's `lock_sha256` must equal it; a **formal** plan also requires the bundle to name a lock, the receipt's `lock_profile` to equal the host profile and `install_host.system`/`machine` to be the lock's platform (`apple-mps` Darwin/arm64, `nvidia-cuda` Linux/x86_64), so a receipt recorded on another platform or with another lock cannot vouch for a run. A receipt is sealed against edits, not signed.

`scripts/record_receipt.py` (a script in the checkout, not part of the wheel or the `devsearch` package) writes these receipts into an existing directory: `implementation --output DIR [--python PY]` runs `python -m unittest discover -s tests` with the given interpreter (it needs the `ml` and `serve` extras, or tests skip and the gate fails) and records the command, exit status, output hashes, test counts and the checkout's source-snapshot hash; `portable --wheel W --lock L --wheelhouse H --output DIR [--python PY]` creates a new virtual environment outside the checkout with the interpreter `--python` (its minor version must be the lock's, so run it with e.g. `python3.12` for the `-cp312` locks), installs **everything** from the hashed lock and the offline wheelhouse (`pip install --no-index --find-links H --require-hashes -r L`, then the wheel with `--no-deps`), runs `pip check`, and inspects the environment: the installed packages must equal the lock's pins, and no `sys.path` entry or imported third-party module may lie outside the environment (nothing is shared with the developer's environment, so a package missing from the wheelhouse fails the install instead of being found elsewhere). Only then does it run `rehearsal`, `doctor`, `plan`, `run`, `status` (all `cpu-diagnostic`, `-I`) from it. The install and the campaign both run under `unshare -rn` when available; without network isolation it records `offline: false` and the gate fails rather than claiming it. The receipt records the lock's name, hash, profile and CPython, the install host (system, machine, python) and the wheelhouse size. File name: `<gate>-<12 hex of the snapshot>-<time>.json`. It records engineering facts only and cannot turn rehearsal evidence into a real-data or accelerator result. The receipt's snapshot hash is the checkout's: the script checks that the wheel's package is imported from outside the checkout, not that the wheel was built from that same snapshot. The portable receipt shows that the lock and wheelhouse install cleanly and run the CPU rehearsal on the host that recorded it; a receipt for the `apple-mps` lock has to be recorded on a Mac, and none exists.

`scripts/record_receipt.py`（检出目录中的脚本，不属于 wheel）生成回执：`implementation` 运行完整单元测试并记录命令、退出状态、输出哈希与计数（缺 `ml`／`serve` 依赖则测试跳过、门禁不通过）；`portable` 用与锁一致的解释器在检出目录外建新虚拟环境，完全从带哈希的锁和离线包安装（`--no-index --require-hashes`），运行 `pip check`，核对已装包恰为锁定集合且没有任何路径或第三方模块来自环境之外，再运行完整 CPU 演练；安装与演练都在网络命名空间中，无网络隔离时记录 `offline: false`、门禁不通过。回执记录锁的名称、哈希、配置、解释器与安装主机；正式计划只接受与其运行包所指同一依赖锁、同一平台的回执。回执中的代码快照哈希是检出目录的，脚本不核对 wheel 是否由同一快照构建。

`acceptance-report.json` (output of `summary`, sealed, `kind: "acceptance-report"`): `plan_id`, `profile`, `evidence_label` (`fixture` for a rehearsal, else `measured`), `gates` (the verdicts frozen in the plan, with `evidence` hashes), `accelerator_run_ready[<profile>]` (false for a diagnostic plan and unless all four gates are `passed`; a `not_applicable` backend gate is therefore not ready), `backend_fingerprint`, `stages` (`{stage_id: {manifest_sha256, status}}` of manifests existing when it was written), `commands` (always `[]`), `artifacts` (`final_report_id`, `release_id`, `backup_receipt_sha256`, `selection_id`), `created_at`, `statement`. `campaign-summary.json` (same stage): `status`, `rehearsal`, `recipes_completed`, `recipes_missing`, `validation_decision`, `selection_id`, `final_report_id`, `release_id`, `release_status`, `default_mode`, `estimates_vs_observed_seconds`, `backup`, `limitations`.

### 2.14 `backup-receipt.json`

Sealed, `kind: "backup-receipt"`, at `<backup_dir>/<plan_id>/backup-receipt.json`; the stage also records `backup-location.json` (absolute destination, file and byte counts).

```json
{
  "schema_version": 1, "kind": "backup-receipt", "plan_id": "plan-<32 hex>",
  "files": {"release/release-manifest.json": {"sha256": "<sha256>", "bytes": 0}, "evidence/events.jsonl": {"sha256": "<sha256>", "bytes": 0}},
  "sources": ["best-checkpoints/train_full_42-epoch-0002.json", "evaluation", "evidence", "plan", "release", "reload", "selection"],
  "totals": {"files": 2, "bytes": 0}, "manifest_sha256": "<sha256>"
}
```

The backup holds: `release/`, `evaluation/` (final report), `selection/`, `reload/`, `evidence/` (every stage manifest as `stage-manifests/<stage_id with ':' as '_'>.json`, plus `events.jsonl`), `plan/` (the whole plan directory, including every `host-binding-*.json`), and each run's best checkpoint as `best-checkpoints/<stage_id with '_'>-<checkpoint_id>.pt|.json`. Files are copied to a staging directory, re-read and compared by SHA-256, the receipt is sealed and the staging directory is renamed into place; an existing backup is never overwritten (identical is a no-op). The destination parent must exist; only a local path is supported. 备份先暂存并逐文件校验哈希，再整体重命名；不覆盖，仅本地路径。

### 2.15 Other stage outputs / 其他阶段输出

Plain JSON unless stated; keys as written by `execution/stages.py`.

| File / 文件 | Stage | Keys |
| --- | --- | --- |
| `preflight.json` | `preflight` | `checks[]`, `device`, `limits_applied`, `remaining_seconds`, `estimated_seconds`, `offline_env`, `rehearsal` |
| `data-check.json` | `data_check` | `train_pairs`, `positives`, `negatives`, `auto_pos_weight`, `validation_queries_scored`, `validation_queries_without_positive`, `validation_groups`, `longest_pair_tokens`, `limit`, `pairs_manifest_sha256` |
| `baseline-validation.json` | `baseline_validation` | `group_macro`, `query_macro`, `queries_scored`, `groups_scored`, `queries_without_positive`, `group_mrr`, `group_ndcg` |
| `run-summary.json` | `train:*` | `stage_id`, `recipe_id`, `seed`, `mode`, `status`, `completed_epochs`, `global_step`, `plan` (`microbatches_per_epoch`, `updates_per_epoch`, `planned_updates`), `best_epoch`, `best_checkpoint_id`, `epochs[]` (`epoch`, `group_mrr`, `group_ndcg`, `checkpoint_id`, `checkpoint_sha256`, `train_loss`, `seconds`, `memory`), `lr_trace_sha256`, `config_sha256`, `run_dir` |
| `export.json` | `export:*` | `model_id`, `recipe_id`, `seed`, `checkpoint_id`, `checkpoint_sha256`, `parity` |
| `latency.json` | `latency` | `workload[]`, `inference_batch`, `device`, `candidate_k`, `protocol`, `baseline`, `recipes{}`, `p95_ms{}`; each measurement: `queries`, `workload_id`, `warmup`, `passes`, `requests`, `concurrency` (1), `response_cache` (false), `seed`, `p50_ms`, `p95_ms`, `max_ms`, `mean_ms`, `candidate_count{min,median,max}`, `degraded_requests`, `valid` |
| `ranked/ranked-output-manifest.json` | `rank_test` | sealed, `kind: "ranked-outputs"`: `candidates_manifest_sha256`, `selection_id`, `protocol_id`, `systems{name: {file, sha256, meta, queries}}`; each `ranked-<system>.jsonl` line is `{query_id, candidate_set_id, ranking: [{chunk_id, rank, score}]}`, a permutation of that query's frozen candidates with one finite score each |
| `review-request.json` | `review_gate` | `protocol`, `plan_id`, `selection_id`, `ranked_outputs_manifest_sha256`, `review_pool_manifest_sha256`, `pool_id`, `missing_reviewed_records`, `request_file`, `instructions`; give reviewers only the blinded `review/review-request.jsonl` of the `pool` stage, never `hidden-provenance.jsonl` |

各阶段输出的键如上，均出自 `execution/stages.py`；盲审只交付 `review-request.jsonl`，不得交付 `hidden-provenance.jsonl`。

## 3. Stage graph, recovery and failures / 阶段图、恢复与失败

### 3.1 Stages / 阶段

The plan stores the list; the controller runs it in list order and requires each stage's needs to come earlier. `T` = every train stage, `X` = every export stage, `D` = the export stages of the deployment seed.

| Stage / 阶段 | Needs / 依赖 | Commits (files in `<attempt_id>/`) / 提交内容 |
| --- | --- | --- |
| `preflight` | none | `preflight.json`: re-checks code snapshot, bundle, protocol, tokenizer, profile hash, `HF_HUB_OFFLINE=1`, device and fingerprint against the doctor's; applies the device memory cap; re-checks mount, space and window |
| `data_check` | `preflight` | `data-check.json`: pair counts, `auto_pos_weight`, longest pair; fails if any train or validation pair breaks the length policy |
| `baseline_validation` | `data_check` | `baseline-validation.json`: BM25 group and query macro metrics on validation |
| `train:<recipe>:<seed>` | `data_check` | `run-summary.json` (epochs, best epoch, checkpoint ids and hashes, `lr_trace_sha256`); checkpoints persist in `work/` |
| `export:<recipe>:<seed>` | its `train` | `model/` (inference bundle) and `export.json` (reload parity checked) |
| `latency` | `D`, `baseline_validation` | `latency.json`: seeded sequential workload, warmup, passes, no cache; p50, p95, max, mean per system; invalid if any request degraded |
| `select` | `baseline_validation`, `latency`, `T` | `validation-selection.json` (the validation lock) |
| `rank_test` | `select`, `X` | `ranked/` (`ranked-output-manifest.json`, `ranked-<system>.jsonl`); first stage that may open the evaluation root |
| `pool` (postrank only) | `rank_test` | `review/` (`review-request.jsonl` blinded, `hidden-provenance.jsonl` private, `review-pool-manifest.json`), `pool.json` |
| `review_gate` (postrank only) | `pool` | without `--reviews`: `review-request.json` with status `awaiting_review`; with them: `bank/` |
| `seal_final` | `rank_test` (prejudged) or `review_gate` (postrank) | `test-seal-final.json` |
| `score` | `seal_final`, `select`, `rank_test` | `final-report.json`, `per-query.jsonl`, `report.md` |
| `export_release` | `score`, `select`, `X` | `release/` (2.11) |
| `reload_verify` | `export_release` | `reload-report.json`: hashes, activation rules, parity, smoke replay, in-process API check |
| `backup` | `reload_verify`, `score` | `backup-receipt.json`, `backup-location.json` |
| `summary` | `backup` | `campaign-summary.json`, `acceptance-report.json` |

Only `train`, `export`, `latency` and `rank_test` carry a time estimate; a stage whose estimate exceeds the remaining window is not started (exit 6). The test side is reachable only through `World.evaluation`, which first requires the `select` manifest to be committed. 测试侧只能经 `World.evaluation` 打开，且须 `select` 已提交。

### 3.2 Commit, reuse and crash windows / 提交、复用与崩溃窗口

A stage writes into `stages/<id>/<attempt_id>/`, then (1) the outputs are hashed, (2) `stage-manifest.json` is committed atomically, (3) `run-state.json` is updated. On `--resume` every stage is reconciled in order:

- Crash before (2): the attempt directory is ignored and renamed `.abandoned-<attempt_id>`; the stage re-runs (training continues from the last `epoch_boundary` checkpoint in `work/`; diagnostic mid-epoch snapshots are never resumed and the interrupted epoch is replayed).
- Crash between (2) and (3): the committed manifest is verified (plan id, stage id, attempt id, input hashes, output hashes) and the state is rebuilt as `succeeded` (event `stage_reconstructed`); a committed manifest from a different attempt than `run-state.json` expects is refused, never overwritten.
- A `succeeded` stage is reused only if its recorded `inputs` equal the current manifest hashes of its needs and every output still exists with the recorded size and SHA-256. Any mismatch is a failed gate (exit 1); nothing is regenerated over it. A changed plan file fails its own hash; `run-state.json` for another plan is refused.

阶段先写尝试目录，再提交清单，最后更新状态；恢复时按序核验。提交前崩溃的尝试被改名保留；提交后状态更新前崩溃则核验后重建为成功；复用要求输入、输出哈希仍然一致，不一致即退出 1，不会覆盖重造。

After a crash the only stage with partial progress inside an attempt is `train:*` (its epoch checkpoints live in `work/`); every other stage restarts from scratch in a new attempt. A `backup` interrupted before its final rename leaves a hidden `.<plan_id>.staging-*` directory in `backup_dir`, which is ignored; an identical complete backup is a no-op. 崩溃后仅训练阶段有尝试内进度（`work/` 中的完整轮断点），其余阶段在新尝试中从头重做。

### 3.3 Failure classes / 失败分类

| Cause / 原因 | Stage state / 阶段状态 | Exit | `--resume` |
| --- | --- | --- | --- |
| Deadline minus reserve passed; a stage estimate above the remaining window; the epoch gate (`1.25 x` expected epoch plus validation and save); window closed during `rank_test` (partial rankings discarded) | `stopped_budget` | 6 | continues after a binding revision extends the deadline; training resumes at the last complete epoch |
| Stage returns `awaiting_review` (postrank `review_gate`) | `awaiting_review` | 5 | re-runs `review_gate` with `--reviews` |
| `PrerequisiteError`: missing mount, marker or directory, device unavailable, lease held, `HF_HUB_OFFLINE` not 1, calibration changed; OS errors ENOSPC, EROFS, ENOENT, EIO, EACCES | `failed`, `interrupted` | 3 | resumable once fixed |
| Ctrl-C, `Interrupted`, or an OS error other than the listed ones | `failed`, `interrupted` | 4 | resumable |
| `ManifestError` or `ConfigError`: tampered or missing artifact, hash mismatch, length violation in `data_check`, bundle or plan mismatch | `failed`, `invalid_input` | 1 | stage re-runs once inputs are restored; committed outputs are never regenerated over |
| Fatal: out of memory, non-finite loss, gradient or score, host memory cap exceeded, `SelectionBlocked`, `LengthError` in `rank_test`, device loss, any unexpected exception | `failed`, `fatal` | 4 | refused (exit 4 again); a changed recipe needs a new plan and run root |

Usage errors exit 2 before any lease is taken. 致命失败不会被静默重试，改配方须新建计划和运行根。

### 3.4 Leases / 锁与租约

`campaign.lock` (always) and `accelerator.lease` (accelerator profiles) are files in `run_root`, created with `O_CREAT|O_EXCL`, holding `{host, pid, start, token, purpose, plan_id}`. Only `run` and `serve` take them (`serve` writes `purpose: "serve"` and puts the release id in `plan_id`); standalone `train`, `predict`, `evaluate` and `export` do not. Same host and the process is gone, or the pid belongs to a process with a different start time: stale, moved aside as `<name>.stale-<12 hex>` (evidence kept) and re-created once. Same host and a live process: held, a second owner is refused even inside one process. Another host or an unreadable file: held, nothing is broken. Release removes the file only if it still carries this owner's token. A held lease is exit 3 for `run` and exit 1 for `serve`. 过期判定看主机与进程身份，不只看时间。

### 3.5 Fault injection and tests / 故障注入与测试

`DEVSEARCH_FAULT=<point>:<stage_id>` (or `<point>:*`; the first colon ends the point, so `mid_stage:train:full:42` works) makes the process `os._exit(137)` at that point with no cleanup; `DEVSEARCH_FAULT_AFTER=<n>` lets the first `n` matching hits pass. Production runs set neither.

| Point / 点 | Window | Exercised by (coverage, not results) |
| --- | --- | --- |
| `stage_outputs_written` | outputs written, manifest not committed | `tests/test_controller.py` (fake stages in real processes, `tests/controller_driver.py`); `tests/test_campaign.py` (`:export:full:42`, `:preflight`) |
| `stage_manifest_committed` | manifest committed, state not updated | `tests/test_controller.py`; `tests/test_campaign.py` (`:select`) |
| `state_updated` | everything committed | `tests/test_controller.py`; `tests/test_campaign.py` (`:select`, `:preflight`) |
| `mid_stage` | inside a stage, once per training microbatch | `tests/test_controller.py`; `tests/test_campaign.py` (`train:full:42` with `FAULT_AFTER` set to one epoch's microbatches plus 3) |

Other coverage: `tests/test_doctor.py` (calibration order, memory caps, timeouts, unhealthy host, bundle failure, no test file opened), `tests/test_plan.py` (refusals, receipts, gates, stage graph), `tests/test_hostfile.py`, `tests/test_lock.py`, `tests/test_backup.py`, `tests/test_release.py`, `tests/test_cli_baseline.py` (`index`, `retrieve`, `validate_manifest`), `tests/test_cli_prepare.py`, `tests/test_cli_ops.py` (`train`, `predict`, `evaluate`), `tests/test_cli_export.py`, `tests/test_cli_serve.py` (including loopback and accelerator refusals), `tests/test_ingestion.py` and `tests/test_prepare_sources.py` (`prepare`), `tests/test_scaffold.py` (command inventory and help). Not covered as commands: the thin `bundle` and `rehearsal` mains, the `plan` success path and `plan --revise-binding`, and `doctor` beyond a failing-device case; the full doctor, plan, run chain with real workers is not covered by any test I found (campaign tests use a scripted doctor worker). 以上为覆盖范围，不是通过结果。

### 3.6 After exit 0 / 正常结束后

- `status --plan PLAN` returns 0 only when every stage's artifacts re-verify; it does not make any gate `passed`.
- Read `stages/summary/<attempt_id>/campaign-summary.json` (decision, `final_report_id`, `release_id`, estimates against observed seconds, backup status) and `acceptance-report.json` (`evidence_label`: `fixture` means rehearsal; gate verdicts; `accelerator_run_ready[<profile>]`).
- Read `stages/reload_verify/<attempt_id>/reload-report.json`: `api_smoke` may be `unverified` (no `serve` extra) while the stage still succeeded.
- The release to serve is `stages/export_release/<attempt_id>/release/` (`serve --check-only` first, with `--runtime-root` covering the release and, on an accelerator, the lease); the evidence copy is `<backup_dir>/<plan_id>/` and is checked against its `backup-receipt.json`.

退出 0 后：`status` 须为 0；阅读 `campaign-summary.json` 与 `acceptance-report.json`（`fixture` 表示演练）；`reload-report.json` 中 `api_smoke` 可能为 `unverified`；先 `serve --check-only` 再启动；备份以其回执核对。

### 3.7 Common refusals / 常见拒绝信息

| Message fragment / 信息片段 | Exit | Meaning and action / 含义与处理 |
| --- | --- | --- |
| `run root ... does not exist; the controller never creates it` | 3 | Mount the volume or create `run_root` yourself; nothing is relocated. |
| `run-state.json already exists; use --resume` / `nothing to resume` | 2 | Use `--resume` only for an existing campaign. |
| `run-state.json belongs to a different plan` | 1 | A changed plan needs a new run root. |
| `a second owner is refused` | 3 | Another live process holds the lease; a dead owner on this host is recovered automatically. |
| `failed fatally ... not a retry` | 4 | OOM, non-finite values or a device error; keep the evidence, make a new plan to change the recipe. |
| `the doctor report was made for a different host file`; `the code changed since the doctor ran` (plan) | 1 | Rerun `doctor` on the current host file and code, then `plan`. |
| `the running code is not the code this plan identifies` (preflight) | 1 | Run the code snapshot that was planned, or make a new plan. |
| `a formal plan needs every gate passed` (plan) | 3 | Add matching receipts (2.13) and a real pinned-asset backend run; a rehearsal bundle only makes a `rehearsal: true` plan. |
| `insufficient time: ... against a ...s window` (plan) | 3 | Extend the deadline in `host.toml`, rerun `doctor`, then `plan`. |
| `the ... differs from the doctor's calibration ... recalibrate` (preflight) | 3 | The device, software or platform changed; rerun `doctor` and make a new plan. |
| `the execution deadline minus the export/backup reserve has passed`; `needs about ...s but only ...s remain` | 6 | `plan --revise-binding` with a later deadline, then `run --resume`. |
| `the host-binding chain does not start at the binding the plan froze` | 1 | A stray or edited `host-binding-*.json` beside the plan; restore the chain. |
| `review item(s) unresolved or missing; sealing is blocked` | 1 | Judge every pooled item (never as label 0), then `run --resume --reviews`. |
| `serving a model on an accelerator must share the campaign's accelerator lease` (serve) | 1 | Set `[serve] lease_path` to `<run_root>/accelerator.lease` and `--runtime-root` to a directory containing it and the release. |
| `candidate release ... --allow-candidate` (serve) | 1 | A model without a final evaluation is served only on request. |

表中为代码实际输出的信息片段、对应退出码和处理办法；过期预算须经新的主机绑定修订后用 `--resume` 继续，配方或代码改变须新建计划。

## 4. Deliberate deviations and clarifications / 有意差异与说明

Only items confirmed in code are listed. 仅列出已在代码中确认的项目。

| # | Topic / 主题 | Code behavior / 代码行为 | Where / 位置 | Versus / 对照 |
| --- | --- | --- | --- | --- |
| 1 | Final-test stages / 最终测试阶段 | Run only inside a campaign. `evaluate --split test` and any `--stage` exit 2; the evaluation root is opened only after `select` committed. | `evaluate.main`; `World.evaluation` | doc 03 lists standalone `evaluate --stage rank\|pool\|score` and `prepare_data --seal-final`; code has neither |
| 2 | Evaluated export / 已评测导出 | A report is bound to the shipped `corpus_id`, `index_id` and (for `rerank`) checkpoint by hash; it is hash-sealed, not signed, so a forged self-consistent report passes. The checkpoint binding is checked at export, later `verify_release` rechecks inventory, report hash, default mode, corpus and index. The campaign's `export_release` stage is the authoritative path. | `export_evaluated`; `export_release`; `verify_release`; `verify_report` | doc 10 §6 "immutable evaluated bundle" |
| 3 | Batch size / 批大小 | `batch_size` is part of the scoring identity: the campaign uses the calibrated `plan.training.inference_batch` everywhere, `export` and `serve` use `[model] batch_size`. The release manifest records it as `provenance.inference_batch` (campaign: the plan's calibrated value; `export`: `[model] batch_size`) and `serve` refuses a model release whose recorded value differs from `[model] batch_size`. | `modeling/jobs.py`; `stages.py`; `release.export_release` | doc 10 §4: inference batch is probed separately |
| 4 | Diagnostic rehearsal / 诊断演练 | Never satisfies real gates: `diagnostic` follows the asset kind; backend gate `unverified` (accelerator) or `not_applicable` (CPU); `data_ready` `unverified`; `accelerator_run_ready` false; a diagnostic bundle can only make `rehearsal: true` plans. The flag is true when the model assets are `diagnostic` or when any document's `parser_version` starts with the reserved prefix `rehearsal` (the rehearsal generator's; a real reader must not use it), so fixture data paired with real-looking assets stays diagnostic (`tests/test_doctor.py`). It is a naming convention against accidents, not a defence against forgery. | `bundle.assemble_bundle`; `doctor._gates`; `plan.build_plan`; `acceptance.accelerator_run_ready` | doc 10 §1 |
| 5 | `data_ready` / 数据就绪 | `passed` means the bundle verifies and every passage fits the length budget; no license, count, real-data or reviewer-type audit. | `doctor._gates` | doc 11 G03 |
| 6 | Host binding revision / 绑定修订 | A revision copies every field of the edited host file (paths, mount identity, limits, caps, deadline, reserve); only a profile change is refused. | `hostfile.revise_binding` | doc 10 §5: paths, deadline, reserve only |
| 7 | Estimates / 估算 | Cover `train`, `export`, `latency`, `rank_test` only, and the total is multiplied by 1.25; stages without a measured rate are listed `unverified`; doctor probe time is not added; the window is measured from the doctor's `created_at`. | `plan.estimate_stages`, `plan.build_plan` | doc 10 §4 "include all probe time" |
| 8 | Evaluation root / 评测根 | A path with an existence check and the code-level gate; not required to be a separate mount and not read-restricted. | `hostfile.run_checks`; `World.evaluation` | doc 10 §5 |
| 9 | Mount and backup device / 挂载与备份设备 | `require_mount_point = false` and `require_independent_device = false` are accepted and recorded `unverified`; the plan does not refuse them. | `hostfile.run_checks`; `plan.build_plan` | doc 10 §4 "real writable mount" |
| 10 | Backup transport / 备份传输 | Local filesystem copy only, fixed destination `<backup_dir>/<plan_id>`. | `execution/backup.py` | doc 10 §4 "configured backup transport" |
| 11 | Leases / 租约 | Taken by `run` and `serve` only. | `controller.Campaign._run`; `execution/serving.py` | doc 10 §5 "training and serving share a lease" |
| 12 | Receipts / 凭据 | The `implementation_verified` and `portable_bundle_verified` gates pass only with sealed receipts; no `devsearch` command writes them, only the checkout script `scripts/record_receipt.py`. | `acceptance.make_receipt`; `scripts/record_receipt.py` | doc 11 §2 acceptance evidence |
| 13 | Acceptance report / 验收报告 | `commands` is always `[]`; evidence is the receipt hashes inside `gates`. | `stages.Runner.summary` | doc 10 §5 "actual commands" |
| 14 | API and UI smoke / 接口界面检查 | `reload_verify` uses an in-process ASGI client; without the `serve` extra `api_smoke` is `unverified` and the stage still succeeds; the UI is not exercised. | `stages.Runner._api_smoke` | doc 10 §6 |
| 15 | Latency record / 耗时记录 | Records candidate counts, p50, p95, max, mean; no token lengths. | `evaluation/latency.measure` | doc 10 §6 |
| 16 | Search policy / 搜索策略 | `top_k = 5`, `max_candidate_k = max(100, depth)` and `max_query_chars = 1000` are fixed in the plan; `candidate_k` is the validation candidate depth. | `plan.build_plan` | not configurable per host |
| 17 | `status` exit / 状态退出码 | A failed artifact check is reported inside the output and exits 4, not 1. | `controller.status_report`; `status.main` | doc 10 §5 "1 invalid artifact" |
| 18 | Offline / 离线 | `run` sets offline variables and `preflight` requires `HF_HUB_OFFLINE=1`; assets load from local files. No network sandbox exists. | `run.main`; `Runner.preflight` | doc 10 §4 |
| 19 | Probe settings / 探测设置 | Probes use weight decay 0.01 and clip norm 1.0 regardless of the protocol; the microbatch probe uses `full` mode and the last recipe's learning rate. | `execution/probe.py`; `doctor.calibrate` | doc 10 §4 |
| 20 | Profile fields / 配置字段 | `effective_batch` and `inference_batches` in the profile TOML are parsed but unused; the protocol's values drive calibration. | `modeling/devices.py`; `doctor.calibrate` | doc 10 §4 |
| 21 | Parser version / 解析器版本 | `parser_version` is a string the operator sets in `sources.toml`; no code derives it from the reader's behaviour. A changed reader needs a new string by hand (`static-extract-v2` after the RST short-underline fix), otherwise two corpora with different content would carry the same version. | `ingestion/config.py`; `configs/sources.toml` | doc 02 provenance |

## 5. What this does NOT prove / 本文不能证明的内容

- A CPU or synthetic rehearsal proves mechanics only: the stages commit, recover, refuse and export. It says nothing about ranking quality. Synthetic documents, a scripted oracle reviewer and a random tiny encoder are plumbing, and every rehearsal report is stamped `diagnostic`.
- No MPS or CUDA run, no real MiniLM or real tokenizer assets, no real corpus and no human review were exercised for this page. `backend_verified[apple-mps]` and `backend_verified[nvidia-cuda]` are `unverified` until a real accelerator run records pinned MiniLM assets trained, timed, saved and reloaded on that device. The two accelerator profiles carry `verification = "unverified"`. Docs-only completion is never accelerator readiness (doc 10).
- `prepare` is unit-tested against local `file://` repositories and small fixtures. Separate manual runs (recorded in `reports/accelerator-readiness-evidence.md` 2.6 and 2.8, not by this page) fetched `pytorch/tutorials` at a pinned commit and prepared it, first with a word-level diagnostic tokenizer and then with an unpinned BERT-uncased vocabulary stand-in; `configs/sources.toml` still carries a placeholder revision, and the corpus length check with the real pinned tokenizer has not been run.
- Pinned means a 40-hex `source.revision` in a free-form `source` object; no code checks that the files are `microsoft/MiniLM-L12-H384-uncased`. `data_ready` and the `diagnostic` flag detect rehearsal-generated data only by the reserved `rehearsal` parser-version prefix.
- Estimates in the doctor report and the plan are labelled `estimate`, come from synthetic-length probes, and are not measured ranking or campaign-time results. A plan that fits its window does not guarantee the run fits.
- Test-question token length is measured before the campaign only when `data.toml` carries `[lengths]` (`prepare_data --stage materialize`); without it an over-long test question fails at `rank_test` as a fatal stage error after `select`, because the campaign must not read the test side earlier.
- Offline operation is by configuration and by absence of download code, not by a verified network block. The locks and the bootstrap script were checked only as follows: a resolver run for each target, a hash-verified wheelhouse download, and an offline install of a stand-in wheel; no lock was installed in full, and none on a target host. The installed-wheel check covers the CPU baseline and command help only, and the `portable` receipt covers a CPU rehearsal only.
- Test names in this page identify intended coverage. No test was run to write this page, so no pass is claimed; skipped tests are not passes.

CPU 或合成演练只证明流程机制，不证明排序质量；本页撰写时未运行任何 MPS／CUDA、真实 MiniLM 与分词器、真实语料或人工审查；两种后端门禁在真实加速器运行记录前均为 `unverified`。`prepare` 仅在本地 `file://` 仓库和小样例上测试，未获取任何真实上游来源，真实固定分词器的语料长度检查尚未运行。固定仅指 40 位十六进制版本号，代码不核对是否为指定模型；`data_ready` 与 `diagnostic` 无法识别样例数据配真实资产。估算均标为 `estimate`，不是实测或排序结果。仅当 `data.toml` 含 `[lengths]` 时才在准备阶段度量测试问题长度，否则超长问题要到 `rank_test` 才致命失败。离线靠配置而非网络封锁；无安装脚本、依赖锁或离线包。测试名仅表示预期覆盖，未运行、不声称通过，跳过不算通过。
