# Implementation Assignments / 实现任务分工

This is a code-ownership and PR plan for reviewed revision 3, not a completion report. The canonical dependency graph and milestone gates remain in [06 Execution checklist](06-execution-checklist.md). Read [08 AI handoff](08-ai-developer-handoff.md) for current scaffold, Python, Git and storage status. A working `--help`, import or protocol stub does not complete M1; real CLI operations remain unimplemented until replaced and verified.

本文是第三版已评审设计的代码职责与 PR 计划，不是完成报告。正式依赖和阶段验收以 [06 实施检查表](06-execution-checklist.md) 为准；当前骨架、Python、Git 和存储状态见 [08 AI 交接](08-ai-developer-handoff.md)。`--help`、导入或协议骨架可用不代表 M1 完成，实际 CLI 操作需实现并验证。

## 1. Four implementation tracks / 四条开发任务线

These are responsibilities within one package, not four services. File paths below are ownership targets; inspect the actual scaffold before creating or renaming files. Each track owns its corresponding tests and config examples. Shared schemas and public signatures require coordination before dependent PRs; do not independently invent competing IDs or duplicate ranking logic.

下表是同一包内的职责，不是四个服务。文件路径表示归属目标，创建或改名前先检查现有骨架。各线负责对应测试与配置样例。共享结构和公共签名须在依赖 PR 前统一，不分别发明 ID 或重复实现排序逻辑。

| Track / 任务线 | Primary scope / 主要范围 | Files owned / 文件归属 | Task IDs / 任务编号 |
| --- | --- | --- | --- |
| A Foundation and retrieval / 基础与检索 | Packaging, small CPU fixtures, ingestion and BM25 / 包结构、小型 CPU 样例、采集与 BM25 | `pyproject.toml`, repository/CI files, `_cli.py`, `ingestion/`, `retrieval/`, `prepare.py`, `index.py`, baseline `retrieve.py`, matching tests / 包、仓库、CI、采集检索入口及测试 | T01, T03, T04 |
| B Data and evaluation / 数据与评测 | Evidence chunks/labels, splits, pair construction, metrics and blind final judging / 证据段落标签、划分、训练对、指标与最终盲审 | `contracts.py`, `data/`, `evaluation/`, `prepare_data.py`, `evaluate.py`, data/eval configs, matching tests / 共享记录、数据评测入口及测试 | T05, evaluation parts of T07/T08, T11 / T05、T07/T08 评测部分、T11 |
| C Model and training / 模型与训练 | Pair encoding/model, native loop, recovery, target-device probes / 样本编码模型、原生循环、恢复和目标设备探测 | `modeling/`, `training/`, `train.py`, `predict.py`, training configs, matching tests / 模型训练预测及测试 | T02, T06, T07, T08 |
| D Serving and release / 服务与交付 | Shared SearchPipeline, API/UI, artifact/release lifecycle, portable export / 检索编排、接口界面、版本生命周期与便携导出 | `artifacts/`, `serving/`, `serve.py`, `export.py`, `validate_manifest.py`, `ui/`, matching configs/tests / 产物服务界面入口及测试 | T09, T10 coordination / 锁定协调, T12 |

Directories are under `src/devsearch/` unless shown otherwise; tests live under `tests/`. Track B owns data contracts, Track C owns the canonical pair encoder, and Track D owns manifest/hash/release helpers. Track A coordinates package and CI changes. Implementers may prepare dependent interfaces early but must not overwrite another track's files or reinterpret its contract without agreement. Any existing shared contract file keeps one designated owner.

未特别注明的模块位于 `src/devsearch/`，测试位于 `tests/`。B 负责数据契约，C 负责唯一的问题—段落编码器，D 负责清单／哈希／发布工具，A 协调包与 CI。可提前准备依赖接口，但不能未经协调覆盖其他线文件或改写契约；已有共享契约文件指定唯一负责人。

Parallel assignment is optional. A single AI can execute these tracks sequentially. A future implementer should only delegate when the user or applicable instructions authorize it; this division is a coordination map, not permission to start unbounded agents.

可以并行分工，也可由一个 AI 顺序执行。后续执行者仅在用户或适用指令授权时委派；这张表用于协调，不代表可以无限启动 Agent。

## 2. T01–T12 ownership and handoffs / 任务与交接

| Task / 任务 | Owner / 负责人 | Required input / 前提 | PR or experiment acceptance / PR 或实验验收 |
| --- | --- | --- | --- |
| T01 Foundation / 基础 | A, with B/D shared-contract input / 协调 B/D 契约 | Reviewed specs, Python 3.11+ existing runtime / 规格与可用解释器 | Package import, invalid-config rejection, small fixtures, code identity; scaffold-only functions are explicit / 导入、非法配置拒绝、小样例、代码身份，未实现处明确 |
| T02 Target probe / 目标探测 | C | Actual machine access and verified runtime storage / 真实机器访问与已验证存储 | Real MiniLM/MPS forward/backward and 50–100-step report; diagnostic weights never become formal starting weights / 真实模型前后向与短测报告，不复用诊断权重作正式起点 |
| T03 Sources and chunks / 来源切块 | A ingestion + B data / 采集与数据 | T01, approved storage, pinned real CPU tokenizer / 存储与真实固定分词器 | Source/license manifest, deterministic mapping, long-code and token-budget evidence / 来源许可、确定映射、长代码与长度证据 |
| T04 BM25 baseline / 检索基线 | A | T03 for real-source acceptance / 真实来源验收依赖 T03 | No model/checkpoint required; source filters, identifier tokens, no-match and source links verified / 无模型依赖，筛选标识符无匹配及来源通过 |
| T05 Data gate / 数据门禁 | B | T03–T04 | Ownership-group audit, unique BCE pairs, training-only negative pool, fixed validation candidates, initial test seal / 归属分组审计、唯一二元对、训练负例池、验证候选与首次封存 |
| T06 Train/resume / 训练恢复 | C | T02 + T05 | Tiny-set correctness, frozen/update checks, continuous-vs-resumed comparison and first export/reload / 极小集正确、冻结更新、连续恢复对照和首次导出 |
| T07 Comparisons / 模型对比 | C + B | T06 | BM25/frozen/full on identical validation candidates, budgets and all runs recorded / 相同验证候选、预算可比、保留全部运行 |
| T08 Domain change / 领域改动 | C + B | T07 | One main factor changed, validation/resource decision, no required positive result / 仅一项主改动、验证资源决定、不强求提升 |
| T09 App candidate / 应用候选 | D | T04; selected T07/T08 artifacts for full acceptance / 完整验收另需所选训练产物 | API/UI modes, offline parity, failures/deadlines, portable candidate and actual private access / 模式、离在线一致、失败时限、便携候选与私有访问 |
| T10 Selection freeze / 选择锁定 | D + B/C sign-off / 确认 | T09 and validation-only decisions / 仅验证集决策 | Immutable model/config/candidate IDs and comparison plan before test outcomes / 测试结果前锁定模型配置候选和计划 |
| T11 Final evaluation / 最终评测 | B | T10 + initial seal / 首次封存 | Blind pooled review, final qrels seal, same denominator, all systems scored, group uncertainty, no retuning / 盲审、最终封存、相同分母、全系统计分、组不确定性、不再调参 |
| T12 Portable release / 便携交付 | D + A/C checks / 配合检查 | T11 | Clean wheel install away from old cache/checkout, load/search/recovery proof, verified backup and demo guide / 脱离旧缓存仓库安装、加载检索恢复、备份及演示指南 |

L01 is tracked separately: the learner explains a batch, independently changes training and interprets a failure. It gates personal skill claims, not T07–T12 engineering. T09 UI development may start on the T04 baseline while training code progresses; final acceptance still requires its listed inputs. Accelerator training and serving do not run simultaneously by default.

L01 单独跟踪：用户解释 batch、独立修改训练并分析失败；它约束个人能力陈述，不阻塞 T07～T12 工程。T09 界面可在 T04 基线上先开发，最终验收仍需对应输入；同一加速设备默认不同时训练和服务。

## 3. First reviewable PR / 首个可审查 PR

Suggested title: **Add offline CPU baseline foundation**. Keep this PR limited to T01 and fixture-level retrieval code supporting T04. Implement import/config validation, a small deterministic fixture corpus, BM25/filter/no-match behavior, and clear baseline CLI output without a checkpoint. Use existing permitted tooling and Python 3.11+; do not install heavy dependencies on the current internal disk. Keep the starter's intentional nonzero exits for operations not implemented by this PR.

建议标题：**Add offline CPU baseline foundation**。范围限于 T01 和支持 T04 的样例级检索代码。实现导入／配置校验、小型确定性语料、BM25／筛选／无匹配行为，以及无 checkpoint 的基线 CLI 输出。使用现有允许工具和 Python 3.11+，不向当前内置盘安装重依赖；本 PR 未实现的操作仍保持明确非零退出。

Proof: record the actual interpreter, exact fast-test/build commands, results and artifact identity. Current manual-only `scaffold-check.yml` is a starting check, not proof of the full feature. Extend CI only after equivalent local CPU checks pass. Fixture retrieval does not certify T03/T04 against real documentation and provides no résumé model-quality metric.

验收证据包括实际解释器、快速测试／构建命令、结果及产物身份。当前仅手动触发的 `scaffold-check.yml` 是起始检查，不证明完整功能；同等本地 CPU 检查通过后再扩展 CI。样例检索不能替代 T03／T04 真实文档验收，也不能提供简历模型质量指标。

**Status (2026-10-05):** the contents of this PR are implemented on branch `claude/youthful-turing-uc1lkv` (the session's designated branch, used instead of a `codex/` name) and checked on tiny fixtures; see [the verification record](../reports/baseline-verification.md). Beyond the list above it adds the corpus artifact/source-map contracts, a stdlib-only TOML config, `validate_manifest` for corpus and index manifests, and an extended wheel check and CPU workflow. The CPU workflow has run once on GitHub (manual dispatch, Python 3.11 and 3.12, both green). Remaining before T01/T04 can pass: real-source ingestion (T03), the pinned real tokenizer, and 30–50 development queries.

**状态（2026-10-05）：** 本 PR 内容已在分支 `claude/youthful-turing-uc1lkv`（本次会话指定分支，代替 `codex/` 命名）上实现，并在微型样例上检查，见[验证记录](../reports/baseline-verification.md)。在上述清单之外，另增语料产物／来源映射契约、仅用标准库的 TOML 配置、针对语料与索引清单的 `validate_manifest`，以及扩展的 wheel 检查与 CPU 工作流。CPU 工作流已在 GitHub 上手动运行一次（Python 3.11 与 3.12，均通过）。T01／T04 通过前仍缺：真实来源采集（T03）、固定真实分词器和 30～50 条开发问题。

## 4. PR boundaries and review protocol / PR 边界与审查

Keep code PRs small enough to establish one contract: baseline retrieval; data ownership and metrics; train/resume; domain experiment configuration/report; service and release. Runtime experiments attach compact provenance/results rather than committing datasets or weights. Multiple experiments may share code, but validation selection and final test evaluation must remain separate stages and artifacts.

代码 PR 每次建立一个可审查契约，例如基线检索、数据归属与指标、训练恢复、领域实验配置报告、服务交付。运行实验只附精简来源与结果，不提交数据集或权重。实验可共用代码，但验证集选择和最终测试评估须保持不同阶段与产物。

Each PR states its T IDs, dependency/base commit, files and schema/API changes, checks actually run, skipped checks and reasons, artifact hashes, remaining work and rollback impact. Review against the owning specification and the R items in 07. Never fix a test failure by changing test membership, importing held-out labels, substituting a fake scorer for required real-model acceptance, or presenting a skip as a pass.

每个 PR 说明 T 编号、依赖／基准提交、文件及结构接口变化、实际检查、跳过项与原因、产物哈希、剩余工作和回滚影响。按所属规格及 07 的 R 问题审查。不能靠改测试成员、读封存标签、用假评分器替代真实模型验收，或把跳过写成通过来消除失败。

GitHub is scoped to `devsearch/` only. The private repository is [Hector-sau/devsearch](https://github.com/Hector-sau/devsearch), with `main` as its default branch and the local `origin` pointing to it. Inspect the checkout and remote before writing; do not accidentally push the parent workspace. Missing account/tool permissions block the remote operation, not preparation of local code and review evidence. Paid services and public publication remain separate user decisions; no new infrastructure is required.

GitHub 范围仅为 `devsearch/`。私有仓库为 [Hector-sau/devsearch](https://github.com/Hector-sau/devsearch)，默认分支为 `main`，本地 `origin` 指向该仓库。写入前检查当前 checkout 和远端，不能误推父工作区。账号／工具权限缺失只阻塞远端操作，不阻塞本地代码及审查证据准备。付费服务与公开发布仍由用户单独决定，不需新增基础设施。

## 5. Status and final proof / 状态与最终证据

Use the task record schema in 06. Mark stub interfaces as scaffold-only, operational code as implemented only when it works, and engineering gates as passed only with recorded evidence. A stage may be partly implemented while awaiting real tokenizer, hardware or storage; report those portions separately. Do not infer completion from a merged PR alone.

使用 06 的任务记录格式。空接口标为仅骨架，工作行为存在才称已实现，工程门禁有证据才称通过。阶段可以部分实现、等待真实分词器／硬件／存储，各部分状态须分开汇报，不能仅凭 PR 合并认定完成。

Final proof consists of the scoped repository, reproducible source/data/version inventory, completed fast CPU checks and target-Mac integration, experiment lineage with both test seals, a locally usable demo, a clean portable install/reload/recovery, verified backups and known limitations. Independent learning evidence is separate. No project metrics, adoption, time savings or model improvement can be claimed before corresponding measurements exist.

最终证据包括限定范围的仓库、可复现来源数据版本清单、快速 CPU 与目标 Mac 集成检查、含两次封存的实验溯源、本地可用演示、干净便携安装／重载／恢复、备份验证及已知局限。独立学习证据另记；没有对应实测前，不宣称项目成绩、用户采用、节省时间或模型提升。
