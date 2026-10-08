# AI Developer Handoff / AI 开发者交接

> **History.** The current consolidated handoff is [HANDOFF](HANDOFF.md). 当前交接见 HANDOFF。

**Status 2026-10-08 (supersedes the "current state" lists below where they differ):** the non-GPU prerequisites are done. The review bank
is complete under protocol revision 5 (audit exit 0, 9,852/9,852 final, agent judgments only); the audited dataset package v2
(`06e7ae62…`, initial seal `9a59905c…`) is a private release asset, rebuilt byte-identically in a fresh root from a fresh download; MiniLM is
pinned at `44acabbe…` and passed the CPU checks; `configs/gpu.toml` pins both. What remains needs the GPU host. The current evidence and the
remaining hardware checks are in [the GPU release acceptance report](../reports/gpu-release-acceptance.md); the commands are in
[GPU_QUICKSTART](GPU_QUICKSTART.md).

**2026-10-08 状态（与下文“当前状态”不一致处以此为准）：**无需 GPU 的前提已完成：第五版协议下审查库完整（审计退出 0，9,852 项全部确定，均为 Agent 判定）；审计后的 v2 数据包为私有发布附件，并在全新目录从实际下载逐字节重建；MiniLM 固定在 `44acabbe…` 并通过 CPU 检查；`configs/gpu.toml` 已固定两者。剩余工作需要 GPU 主机，见验收报告与快速指南。

**Read the revision-4 addendum before continuing implementation:** [10 Execution contract](10-accelerator-execution-contract.md), [11 Readiness review](11-readiness-and-review.md), [12 Bounded experiments](12-estimates-and-experiment-plan.md). The user now keeps MPS and CUDA options. Implement a prepared no-source-edit run with fixed-candidate pre-review; do not call the present scaffold GPU-ready. Keep your newer implementation work and integrate these requirements instead of replacing it with this older scaffold snapshot.

**继续开发前先读第四版补充：**第 10 号执行契约、第 11 号就绪评审、第 12 号有限实验。用户现保留 MPS／CUDA 两种选择，要求准备完成后无需改源码运行，并采用固定候选预审；当前骨架不能称 GPU 就绪。保留你已做的新实现，整合要求，不用本旧骨架快照覆盖新工作。

## Current implementation state (2026-10-05, after the execution-contract work) / 当前实现状态

**This section supersedes the older paragraphs below, which describe the first slice and are kept as history.** 本节取代下文对首个实现切片的描述，旧段落仅作历史保留。

- Every command in [03](03-implementation.md) is implemented: `prepare`, `prepare_data`, `index`, `retrieve --mode baseline`, `validate_manifest`, `train`, `predict`, `evaluate` (training-visible splits only), `export`, `serve`, plus `doctor`, `plan`, `run`, `status`, `bundle`, `rehearsal`. No help-only stub remains. The full operator and schema reference is [13](13-operator-reference.md); the evidence mapped to gates G01–G10 is in [the evidence report](../reports/accelerator-readiness-evidence.md).
- Checked on CPU only: synthetic and diagnostic data (tiny random encoders, a scripted oracle standing in for a reviewer), plus one real public documentation corpus taken through `prepare`, `prepare_data`, `index` and a BM25-only `evaluate` with 914 synthetic, agent-reviewed questions and an unpinned stand-in vocabulary. Nothing was run on MPS or CUDA; the pinned real MiniLM and tokenizer, a judged review bank (only an 80-item pilot exists) and human review do not exist yet; the only ranking and latency numbers are BM25-only (evidence report 2.8, 2.10). GitHub Actions `ml-cpu` and `cpu-baseline` jobs are green on the branch (no pull request).
- Code map additions: `ingestion/` (readers, chunker, snapshots), `execution/` (controller, leases, doctor, plan, stages, exporter, serving starter, rehearsal), `modeling/`, `training/`, `evaluation/` (protocol, selection, latency, report), `serving/` (pipeline, API, UI, release), `profiles/` (device profiles). Layering is enforced by `tests/test_scaffold.py`.
- Tests: `PYTHONPATH=src:tests python -m unittest discover -s tests` with the `ml` and `serve` extras installed must report **no skips** (`scripts/record_receipt.py implementation` fails on a skip); without the extras ML and API tests are reported as skips, which are not passes.
- Next actions: [14](14-target-verification-checklists.md) gives the order (pin MiniLM and prove the tokenization, freeze the dataset, scale the candidate review after its pilot, then the CUDA and MPS checklists separately); the evidence report section 4 lists each blocker. The project is not GPU-ready.

Handoff prepared on 2026-10-05 against reviewed design revision 3, then updated the same day after the first implementation slice (offline CPU foundation plus a model-free BM25 baseline over tiny fixtures). Three commands are implemented (`index`, `retrieve --mode baseline`, `validate_manifest`) and the other seven remain help-only stubs. See [the baseline verification record](../reports/baseline-verification.md) and the earlier [scaffold record](../reports/scaffold-verification.md). Inspect the checkout again when taking over: fixture-level retrieval is not real-document retrieval, a trained model, an API or a UI. No project training result, measured improvement or user study is available.

本交接基于 2026-10-05 已评审的第三版设计，并在同日首个实现切片（离线 CPU 基础及基于微型样例的无模型 BM25 基线）完成后更新。已实现三个命令（`index`、`retrieve --mode baseline`、`validate_manifest`），其余七个仍是仅帮助占位。详见[基线验证记录](../reports/baseline-verification.md)及此前的[骨架记录](../reports/scaffold-verification.md)。接手时仍须重新检查实际状态；样例级检索不等于真实文档检索、训练模型、接口或界面。目前没有项目训练结果、实测提升或真人研究。

The starter targets Python **3.11+**. The existing bundled Python **3.12.14** is the intended verification interpreter; the system default **3.9.6** is unsupported. The package still has **no runtime dependencies** (standard library only, including `tomllib` for configs), no new dependency has been installed and the future dependency lock is unresolved. Implemented commands are listed above; every other operational CLI action is deliberately unimplemented and must exit nonzero. Inspect the actual check report before treating any behavior as verified.

骨架要求 Python **3.11+**，现有捆绑 Python **3.12.14** 用于验证；系统默认 **3.9.6** 不受支持。包仍**没有运行依赖**（仅标准库，配置用 `tomllib`），未安装新依赖，后续依赖锁尚未确定。已实现的命令见上文，其余实际操作命令有意保持未实现并应非零退出；认定行为已验证前须查看实际检查报告。

`.github/workflows/scaffold-check.yml` (name: CPU baseline checks) runs the unit tests, builds the wheel and runs the installed-wheel check on Python 3.11 and 3.12, on pull requests and manually; it is not a trained-model or full application CI gate. Its GitHub run status is recorded in the [baseline verification record](../reports/baseline-verification.md). Configuration documentation is not a set of runnable experiment configs. The private repository is [Hector-sau/devsearch](https://github.com/Hector-sau/devsearch), with default branch `main`. The local `devsearch/` directory is an independent Git repository with that URL configured as `origin`; parent-workspace Git state is unrelated. The first implementation slice is committed on branch `claude/youthful-turing-uc1lkv`; no pull request has been opened.

`.github/workflows/scaffold-check.yml`（名称 CPU baseline checks）在 Python 3.11 与 3.12 上运行单元测试、构建 wheel 并执行安装后检查，由 pull request 或手动触发；不代表模型或完整应用 CI。其 GitHub 运行状态记录在[基线验证记录](../reports/baseline-verification.md)。配置说明不是可运行的实验配置。私有仓库为 [Hector-sau/devsearch](https://github.com/Hector-sau/devsearch)，默认分支为 `main`。本地 `devsearch/` 已是独立 Git 仓库，`origin` 指向该地址，与父工作区 Git 状态无关；首个实现切片已提交到分支 `claude/youthful-turing-uc1lkv`，尚未创建 pull request。

Status vocabulary: **scaffold-only** means a structure or protocol exists; **implemented** requires working behavior; **verified** additionally requires recorded checks. All real data/index/model/training milestones remain unaccepted. Fixture results are engineering checks, never résumé ranking performance.

状态用语：**仅骨架**表示结构或协议已存在，**已实现**须有工作行为，**已验证**还须有检查记录。真实数据、索引、模型和训练阶段仍未验收。样例测试属于工程检查，不能作为简历排序效果。

## Code map / 代码地图

```text
devsearch/                         # The repository root / 独立仓库根
  AGENTS.md                       # Instructions for the next AI / 接手规则
  pyproject.toml                  # Python 3.11+, src layout, no dependencies yet
  src/devsearch/
    errors.py contracts.py        # Expected failures; frozen internal records
    _cli.py                       # Shared CLI helpers + fail-closed stubs
    artifacts/                    # canonical JSON/SHA-256, path containment, strict
                                  #   field validators, code identity (+ port protocols)
    ingestion/                    # SourceReader protocol (T03 pending)
    data/                         # corpus artifact + source maps (implemented);
                                  #   Chunker, DataPreparer protocols
    retrieval/                    # TOML config, lexical-v1 tokenizer, BM25, index store,
                                  #   baseline response document (implemented)
    modeling/ training/ evaluation/ serving/   # protocols only
    index.py retrieve.py validate_manifest.py  # implemented commands
    prepare.py prepare_data.py train.py evaluate.py
    predict.py serve.py export.py              # help-only stubs, exit 2
  configs/                        # index.toml, search.toml examples; README
  ui/README.md                    # UI boundary; application pending
  tests/                          # unittest suites; fixtures/mini_corpus + its builder
  scripts/check_wheel.py          # Isolated wheel install + baseline commands
  reports/                        # scaffold + baseline verification records
  .github/workflows/scaffold-check.yml
  .github/pull_request_template.md
  docs/                           # Architecture, contracts and assignments
```

Protocol classes describe interfaces only; they do not implement those operations (the `Retriever` protocol is now implemented by `Bm25Retriever`). Keep CPU retrieval independent of encoder imports; a unit test enforces the package layering. Implement one shared pair-encoding policy for training and inference, and one SearchPipeline for CLI and API. The `contracts.py` dataclasses are internal values, not JSON/API schemas or trusted-input validators; persisted records are validated by `data/corpus.py`, `data/sourcemap.py` and `artifacts/schema.py`.

Protocol 仅声明接口，不实现操作（`Retriever` 现由 `Bm25Retriever` 实现）。CPU 检索保持独立于编码器导入，单元测试强制包的分层；训练推理共用样本编码规则，CLI 与 API 共用 SearchPipeline。`contracts.py` 的 dataclass 是内部值对象，不是 JSON／API 结构或输入校验器；持久化记录由 `data/corpus.py`、`data/sourcemap.py` 与 `artifacts/schema.py` 校验。

| Current component / 当前部分 | State / 状态 |
| --- | --- |
| Package imports, CLI help and refusal behavior of the seven stub commands / 包导入及七个占位命令的帮助与拒绝操作 | Implemented and locally checked / 已实现且本地验证 |
| Strict config, corpus/source-map/index artifacts, deterministic IDs, code identity / 严格配置、语料／来源映射／索引产物、确定性 ID、代码身份 | **Implemented and checked on tiny fixtures** (T01 partial) / **已实现，在微型样例上检查**（T01 部分） |
| `lexical-v1` tokenizer, BM25, filters, zero-match, `index`/`retrieve`/`validate_manifest` / 分词、BM25、筛选、无匹配及三个命令 | **Implemented and checked on tiny fixtures only** (T04 fixture level); no real documents, no measured quality / **已实现，仅在微型样例上检查**（T04 样例级），无真实文档、无效果实测 |
| Real-source ingestion/chunking, label and split preparation, metrics / 真实来源采集切块、标注划分、指标 | Planned / 待实现 |
| PyTorch model, training, reranking, API/UI / 模型、训练、重排序、接口及界面 | Planned / 待实现 |
| GitHub workflow / GitHub 工作流 | Updated; remote run status in the verification record / 已更新，远端运行状态见验证记录 |

## 1. Read in this order / 阅读顺序

Read repository `AGENTS.md` first, then follow the sequence below. If an expected file is missing, record the actual state and read the available canonical specification; do not invent its contents. These documents describe the project, not authorization to expand its scope.

先读仓库根目录 `AGENTS.md`，再按下表阅读。预期文件缺失时记录实际状态并读取现有规范，不编造文件内容。文档规定项目方案，不授权扩大范围。

| Order / 顺序 | Read / 文档 | Purpose / 目的 |
| --- | --- | --- |
| 1 | [README](../README.md) | Current outcome, truth and scope / 当前目标、状态与范围 |
| 2 | [00 Architecture](00-architecture-and-roadmap.md) | System boundary, module contracts, lifecycle / 架构、模块契约与生命周期 |
| 3 | [06 Execution checklist](06-execution-checklist.md) | Canonical T01–T12 dependencies and artifacts / T01～T12 依赖与产物 |
| 4 | [07 Review register](07-review-register.md) | Already-resolved design gaps; implementation checks still needed / 已修订设计问题与待执行检查 |
| 5 | [01 Product/evidence](01-product-and-evidence.md) | User behavior and truthful evidence / 使用行为与证据边界 |
| 6 | [02 Data/evaluation](02-data-and-evaluation.md) | Splits, mining ownership and two-stage test sealing / 划分、挖掘归属与两阶段封存 |
| 7 | [03 Implementation](03-implementation.md) | Model, native loop, recovery and CPU/MPS checks / 模型、原生循环、恢复及设备检查 |
| 8 | [04 Integration](04-integration-and-decisions.md) | CLI/API, fallback and release contracts / 命令接口、降级与版本契约 |
| 9 | [05 Machine handoff](05-machine-handoff.md) | Runtime storage, private access and verified backup / 运行存储、私有访问与备份 |
| 10 | [09 Assignments](09-implementation-assignments.md) | File ownership and PR-sized work / 文件分工与 PR 工作单元 |

For conflicts, 00 governs boundaries and milestone order; 02 owns data and evaluation, 03 owns training, and 04 owns API details. Reconcile a genuine conflict explicitly before dependent work; do not silently choose whichever rule makes a test easier.

冲突时，00 管理系统边界与阶段，02 管理数据评测，03 管理训练，04 管理接口细节。真实冲突须在依赖工作前明确统一，不能为了更容易通过测试而静默换规则。

## 2. Non-negotiable project contracts / 必须保持的契约

- English PyTorch Tutorials and Transformers documentation; BM25 first, a MiniLM cross-encoder with a visible native PyTorch loop second. NQ is optional transfer/diagnostic data. No generated answers, Agent system, vector database or new managed service in v1.
- One package and shared SearchPipeline: offline evaluation and serving use the same filtering, tokenization, scoring and tie-breaking logic. The UI calls the API and never loads a second model.
- Train negatives use only `owner_split: train` evidence groups and training queries; persist the mining pool. The serving index may contain all groups, but test queries/qrels never enter training or negative masks. Materialize unique BCE pairs and reconcile positive/negative counts.
- Preserve both test seals: initially freeze questions/groups/corpus/initial qrels; after locking all systems, conduct the specified blind pooled review, freeze final shared qrels, then calculate scores. Never insert gold passages into retrieval, remove retrieval misses, or tune on test outcomes.
- Preserve source versions, code excerpts, evidence locations and label-review provenance. Agent review is not human review. A negative experiment is a valid reported outcome, not a reason to change the test.

- 使用英文 PyTorch Tutorials 和 Transformers 文档；先 BM25，后 MiniLM cross-encoder，并保留清楚可见的原生 PyTorch 训练循环。NQ 是可选迁移／排错数据。首版不增加生成答案、Agent 系统、向量库或托管服务。
- 一个包与共享 SearchPipeline；离线评测和在线服务共用筛选、编码、评分及同分逻辑，界面只调用 API，不另载模型。
- 负例仅使用 `owner_split: train` 的证据组和训练问题，保存挖掘池。服务索引可见全部组，但训练及负例屏蔽不读测试问题／qrels。训练前生成唯一 BCE 样本对并核对正负数量。
- 保留两次测试封存：先冻结问题／组／语料／初始标注；锁定全部系统后按协议盲审汇总候选，封存最终统一标注，再计分。不能补入正确答案、删掉漏检或根据测试调参。
- 保留来源版本、代码摘录、证据位置和审查来源。Agent 审查不是人工审查；无提升也是应报告的实验结果，不能据此换测试。

## 3. GitHub-first, with honest local verification / 以 GitHub 协作为主，验证如实记录

The repository root is **this `devsearch/` directory**, not its parent `AI foundation/`. Verify the Git top-level and remote before committing or pushing. Never include the job-analysis workbook, unrelated personal documents, parent-directory history, credentials, runtime data, model weights, caches or environments by accident. Use small `codex/` branches and reviewable PRs linked to T task IDs; preserve dependency order and record base commits for dependent PRs.

仓库根目录是**本项目 `devsearch/`**，不是上层 `AI foundation/`。提交或推送前核实 Git 根目录及远端，不误带岗位分析表、其他个人资料、父目录历史、凭据、运行数据、模型权重、缓存或环境。使用小范围 `codex/` 分支和关联 T 编号的可审查 PR；有依赖的 PR 保留顺序并记录基准提交。

GitHub carries code, small licensed fixtures, schemas, concise reports and fast CPU CI. CI must not download a production encoder or run costly training. The real pinned tokenizer can be prepared and checked on an authorized CPU environment for M1; MiniLM/MPS and measured target-device experiments run on the supplied Mac. CPU CI success never substitutes for MPS acceptance; a skipped or unavailable check stays marked as such. Record actual commands and results instead of assuming tools, a remote, or CI access exist.

GitHub 承载代码、小型许可合规样例、结构定义、精简报告和快速 CPU CI。CI 不下载正式编码器或运行高成本训练；M1 的真实固定 tokenizer 可在获授权的 CPU 环境准备和检查，MiniLM／MPS 与目标设备实测在用户提供的 Mac 完成。CPU CI 通过不能代替 MPS 验收，被跳过或无法执行的检查如实标注，不假定工具、远端或 CI 已可用。

A private project workflow is intended; public publication, paid cloud services and paid model calls are not automatic. If a repository or required permission is missing, prepare the local reviewable changes and identify the precise remaining prerequisite. Do not fabricate a PR URL, test output, machine access or completed upload. When using Codex and a PR is created, attach that PR to the current task using the available app tool.

默认采用私下项目协作，不自动公开发布、使用付费云服务或付费模型调用。仓库或权限缺失时，先完成可审查的本地改动，再准确说明缺少的前提；不编造 PR 链接、测试输出、机器访问或已完成上传。通过 Codex 创建 PR 后，使用可用的应用工具将 PR 关联到当前任务。

## 4. What can start before the Mac arrives / 机器到位前可以做什么

The user will arrange a future Mac mini M6 with 24 GB memory; it has not been acquired or validated. The assistant owns routine implementation, transfer, dependencies and execution within the authorized resources. Start CPU code, small offline fixtures, schema validation, retrieval and fast tests now where existing tools permit. Do not require the future machine merely to write and review code.

用户将安排未来的 24 GB Mac mini M6，目前尚未取得或验证。助手负责授权资源范围内的常规开发、迁移、依赖和执行。现有工具允许时，可先做 CPU 代码、小型离线样例、结构校验、检索及快速测试；不因未来机器未到而停止代码开发与审查。

Heavy assets and job runtime belong under a verified `DEVSEARCH_RUNTIME_ROOT`, preferring the external drive or the target host's permitted durable storage. Do not install a project environment or download model/data assets onto the current Mac's internal disk by default. Code, documentation and small fixture files may remain in this workspace. An unmounted external volume is not a writable fallback directory. Follow 05 for remote storage and backup; do not publish or transfer unrelated private files.

大体积产物和运行任务使用经验证的 `DEVSEARCH_RUNTIME_ROOT`，优先外置盘或目标主机允许的持久存储。不默认在当前 Mac 内置盘安装项目环境或下载模型／数据；代码、文档和小样例可保存在当前工作区。未挂载卷名不是可写替代目录；远端存储及备份遵循 05，不发布或迁移无关私人文件。

## 5. First PR and reporting / 首个 PR 与交接记录

The first implementation PR should complete the offline T01 foundation and add a minimal BM25 baseline slice over checked-in tiny fixtures: import/config validation, deterministic IDs, filtered retrieval, identifier queries, zero-match behavior and source mapping, all without a model/checkpoint/network. Add the fast CI command after it runs locally. This proves fixture-level code only; T03/T04 real-source ingestion and target-machine checks remain pending until their own gates pass. **Status: this slice is implemented** (see [the record](../reports/baseline-verification.md)); the next work item is T03 (pinned sources, licenses, the real CPU tokenizer and token-valid chunking), which needs a verified `DEVSEARCH_RUNTIME_ROOT`.

首个实现 PR 完成 T01 离线基础，并用仓库内极小样例增加 BM25 基线切片：包导入／配置校验、确定性 ID、筛选检索、标识符问题、零匹配行为和来源映射，均不依赖模型／checkpoint／网络。快速 CI 命令先本地验证再接入。它只证明样例级代码，T03／T04 真实来源采集及目标机器检查，须各自通过门禁才可标完成。**状态：该切片已实现**（见[记录](../reports/baseline-verification.md)）；下一项工作为 T03（固定来源、许可、真实 CPU 分词器及长度合规切块），需要经验证的 `DEVSEARCH_RUNTIME_ROOT`。

For each PR, report task IDs, changed contracts, exact checks with pass/fail/skip, artifact identities, unresolved issues and next action. Keep `engineering_status` separate from `learning_evidence_status`. The user independently explains and modifies training before personal-mastery résumé claims; waiting for that exercise does not block authorized engineering progress.

每个 PR 报告任务编号、契约变化、实际检查及通过／失败／跳过、产物身份、未解决项和下一步。`engineering_status` 与 `learning_evidence_status` 分开；用户须独立解释和修改训练，才能支持本人掌握的简历陈述，等待练习不阻塞授权工程。

## 6. Paste to another AI / 可直接交给其他 AI 的提示词

> Continue DevSearch in the `devsearch/` repository only. Read AGENTS.md, README, docs/00, 06, 07, then 01–05 and 09 in the order above. This is reviewed design revision 3 with a help-only scaffold, not a trained or deployed system. Inspect actual files and tests first. Implement the first small CPU baseline PR described here, using offline fixtures and no model downloads, then follow T01–T12 dependencies. Preserve native PyTorch, English BM25/MiniLM, training-owned negatives, both test seals, shared offline/online logic, honest evidence and the external-runtime storage policy. The user supplies the future Mac; you own routine engineering, and can write/test CPU code before machine access. Keep GitHub scoped to devsearch, use reviewable branches/PRs and fast CPU CI, and reserve real MPS checks for the supplied Mac. Do not publish publicly, buy services, access test labels for tuning, or invent tools/results. Report exact completed checks and the next concrete prerequisite.

> 仅在 `devsearch/` 仓库内继续 DevSearch。按上文顺序先读 AGENTS.md、README、docs/00、06、07，再读 01～05 和 09。当前是第三版已评审设计和仅帮助可用的骨架，不是已训练或部署系统；先检查实际文件与测试。先实现本文规定的小型 CPU 基线 PR，使用离线样例且不下载模型，再按 T01～T12 依赖推进。保持原生 PyTorch、英文 BM25／MiniLM、仅训练归属负例、两阶段测试封存、共享离在线逻辑、真实证据及外置运行存储要求。用户提供未来 Mac，你负责常规工程，接入前可先写测 CPU 代码。GitHub 仅包含 devsearch，用可审查分支／PR 和快速 CPU CI，真实 MPS 检查留给用户提供的 Mac。不公开发布、不购买服务、不用测试标签调参、不编造工具或成果。报告实际完成的检查与下一项具体前提。
