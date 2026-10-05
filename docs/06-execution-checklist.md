# Execution checklist / 实施检查表

**Revision-4 execution mapping:** [10](10-accelerator-execution-contract.md) defines the new controller/device/protocol contracts and [11](11-readiness-and-review.md) adds G01–G10 evidence gates. T02 now verifies the selected MPS or CUDA backend independently. Under the recommended pre-review protocol, T05 includes all fixed test-candidate judgments; T11 verifies coverage and seals the unchanged bank. T08 expansion is optional before a formal campaign or deferred to a new protocol. Existing T task identities remain; no task becomes passed through this documentation update.

**第四版执行映射：**第 10 号补充控制器设备协议，第 11 号增加 G01～G10 证据门禁。T02 独立验证所选 MPS 或 CUDA；推荐预审模式下 T05 包含全部固定测试候选判断，T11 核对覆盖并封存未变判断库。T08 扩展在正式实验前选择，或延后采用新协议。原任务编号保留，不因更新文档而通过任何任务。

Revision 3, 2026-10-05. T01 (offline CPU foundation) and the fixture-level part of T04 (model-free BM25 slice) are implemented and checked on tiny fixtures ([record](../reports/baseline-verification.md)); both remain `running` because their real-source gates are not passed. All other tasks remain planned. This checklist translates architecture into executable work. Detailed contracts remain in their owning specifications; a checked box requires an artifact and verification result, not a documentation paragraph. See [current scaffold verification](../reports/scaffold-verification.md).

第三版，2026-10-05。T01（离线 CPU 基础）及 T04 中样例级部分（无模型 BM25 切片）已实现，并在微型样例上检查（[记录](../reports/baseline-verification.md)）；两者的真实来源门禁尚未通过，故仍为 `running`。其他任务仍为计划状态。本表把架构转换为可执行工作，细节以各专项规格为准。完成标记必须对应产物和验证结果，当前状态见骨架验证记录。

## 1. Dependency and deliverable map / 依赖与交付表

The assistant owns the engineering tasks. The user supplies machine access; learning demonstrations are tracked separately. M0 has a CPU/package branch and a target-model branch, so model compatibility does not prevent useful corpus and BM25 work. Heavy downloads still require a verified permitted storage root.

工程任务由助手负责，用户提供机器访问，学习演示单独跟踪。M0 分为 CPU／包结构与目标模型两个分支，模型兼容问题不阻塞语料和 BM25 工作；大体积下载仍须已有经验证且允许使用的存储目录。

| ID / Stage 阶段 | Depends on / 依赖 | Work and output / 工作与产物 | Exit evidence / 验收证据 |
| --- | --- | --- | --- |
| T01 / M0 | Reviewed specifications / 已评审规格 | Create package/config schemas and offline CPU fixtures; establish code snapshot identity / 建包、配置结构和离线 CPU 样例，建立代码快照身份 | Package imports and fixture checks; unknown fields/invalid paths fail / 包可导入、样例检查通过，非法字段路径报错 |
| T02 / M0 target branch 目标分支 | Machine access and storage / 机器访问与存储 | Hardware/runtime report and MiniLM 50–100-step probe on independent diagnostic fixtures / 硬件环境报告及独立诊断样例短训练探测 | Device/memory/throughput logged; no test access or diagnostic-weight reuse for formal training / 记录设备内存吞吐，不读测试或将诊断权重复用为正式训练起点 |
| T03 / M1 | T01 + verified storage + pinned real tokenizer on CPU / 存储通过检查，CPU 准备真实固定分词器 | Pin sources/licenses/tokenizer; parse documents and token-valid chunks without encoder weights / 固定来源许可分词器，不需编码器权重即可解析长度合规段落 | Source-map/long-code/special-token checks; coverage and exclusion report / 来源映射、长代码、特殊 token 检查及覆盖排除报告 |
| T04 / M1 | T03 | Build versioned BM25, baseline retrieve/evaluate, 30–50 development cases / 构建 BM25、基线检索评测和开发样例 | Works without model/checkpoint; filter, lexical miss and source links verified / 无模型可运行，筛选、无词项匹配及来源链接通过 |
| T05 / M1 data gate 数据门禁 | T03–T04 | Coverage plan, reviewed labels, evidence groups, split manifest, binary pairs, fixed validation candidates, initial test seal / 覆盖计划、标注分组、划分清单、二元对、验证候选和首次测试封存 | Group-overlap audit, pair-count reconciliation, evidence retention, no test access in training / 分组交叉审计、样本数对账、证据保留、训练不读测试 |
| T06 / M2 | T02 + T05 | Tiny trainable subset and save/resume implementation / 极小可训练子集及保存恢复 | Finite loss/gradients, intended parameter updates, continuous vs resumed run comparison, first export/reload / 损失梯度有限、预期参数更新、连续与恢复运行对照、首次导出重载 |
| T07 / M2 | T06 | BM25 vs frozen-head vs full fine-tuning; record every run / 基线、冻结评分层与全量微调比较，保留所有运行 | Same validation candidates; comparable data/budgets; quality, latency and failures reported / 同验证候选、数据预算可比、效果耗时失败完整报告 |
| T08 / M3 | T07 | One controlled domain change; optionally expand if justified / 一项受控领域改动，有依据再扩量 | One main factor changed; validation and resource decision recorded, including negative results / 仅一个主因素变化，记录验证与资源决定及负面结果 |
| T09 / M4 | T04 and selected T07/T08 artifacts / 基线和已选实验产物 | API/UI, portable candidate, health/readiness, fallback and private remote access / 接口界面、便携候选、健康就绪、降级及私有远程访问 | Offline/online parity, failure/deadline tests, access from user's endpoint, compatible manifest / 离在线一致、失败时限检查、用户访问端可达、清单兼容 |
| T10 / M4 freeze 锁定 | T09 | Lock compared systems, default mode, thresholds and final comparison plan / 锁定比较系统、默认模式、阈值与最终比较计划 | Model/config/candidate hashes fixed; test outcomes unavailable to selection / 模型配置候选哈希固定，选择不使用测试结果 |
| T11 / M5 | T10 + initial seal / 首次封存 | Blind pooled review, final qrels seal, all-model report and group uncertainty / 盲审候选池、最终标签封存、全模型报告和分组不确定性 | Same final qrels and eligible queries; retrieval misses retained; no post-test tuning / 统一最终标签及合格问题，保留漏检，不据测试再调参 |
| T12 / M5 | T11 | Evaluated release, clean wheel install, fixed-query reload, verified backup and demo guide / 已评测版本、干净 wheel 安装、固定查询重载、备份与演示说明 | Start outside the old checkout/cache; independent reload/recovery and return-machine export verified / 离开原 checkout 与缓存启动，独立重载恢复及退机导出通过 |
| L01 / Learning 学习 | T06 onward / T06 起 | Explain one batch, implement a small change and interpret its result / 解释一个 batch，独立修改并分析结果 | Separate learning evidence; required for personal-mastery claims, not to unblock T07–T12 / 单独学习证据，仅约束个人掌握声明，不阻塞后续工程 |

T09 interface work can start from the BM25 slice while training proceeds; accelerator inference and training still run at different times on the same machine. Optional NQ work is recorded as `not_required` with a reason when domain evidence does not justify it. A useful baseline with a negative reranking result is a valid engineering outcome.

T09 界面可在 BM25 切片完成后与训练开发并行，但同机加速器上的推理和训练仍错时运行。领域证据不支持 NQ 扩量时，记录为 `not_required` 并说明原因。可用基线配合无提升的重排实验，仍可构成有效工程成果。

## 2. Artifact and configuration requirements / 产物与配置要求

Keep code, small fixtures, schemas and concise reports in the project. Use `DEVSEARCH_RUNTIME_ROOT` for environments, dependency/model caches, raw snapshots, normalized documents, chunks, labels, indexes, candidates, runs and exports. Portable artifact references are relative to their bundle root; resolved paths must remain within that root. Transfer manifests and reconstruct the environment rather than copying machine-specific virtual environments.

代码、小样例、结构定义和精简报告保存在项目中；环境、依赖及模型缓存、原始快照、规范正文、段落、标注、索引、候选、运行和导出包位于 `DEVSEARCH_RUNTIME_ROOT`。便携产物引用相对包根目录，解析后不能越界。迁移产物清单并重建环境，不复制绑定原机器的虚拟环境。

| Contract / 契约 | Required record / 必须记录 |
| --- | --- |
| Source/corpus / 来源语料 | Repo commit or dataset revision, selected paths, licenses, normalized text and source map, chunking/token rules, content hashes and exclusions / 来源版本、选取路径、许可、正文映射、切块编码、哈希及排除 |
| Labels/splits / 标注划分 | Query task/source/filter metadata, evidence groups, provenance/reviews, ownership and duplicate audit, initial/final test seals / 问题任务来源筛选、证据组、审查来源、归属去重、两次封存 |
| Index/candidates / 索引候选 | Corpus ID, BM25 implementation/tokenizer/parameters, filter semantics, ordered IDs/scores, actual counts and timings / 语料 ID、BM25 实现分词参数、筛选规则、候选排名分数、实际数量耗时 |
| Run/checkpoint / 运行断点 | Code identity, dependency lock, full resolved config, data hashes, seed, device, update count, resume and best-model state, metrics / 代码身份、依赖锁、配置数据哈希、种子设备、更新次数、恢复与最佳模型状态及指标 |
| Report/release / 报告发布 | Compared run/candidate IDs, final qrels, per-query metrics, group counts/uncertainty, failure cases, mode decision, inference assets and hashes / 比较运行候选、最终标签、每题指标、组数不确定性、失败案例、模式决定、推理产物哈希 |
| Runtime handoff / 运行交接 | Machine/OS/access mode, runtime root, availability window, measured stop/backup reserve, last verified export, startup/stop instructions / 机器系统访问、运行根目录、可用窗口、实测停止备份余量、最近有效导出和启停说明 |

Every schema has a version and a validator. Fail on missing required files, duplicate IDs, invalid foreign references, unrecognized fields or inconsistent hashes. Define manifest hashes using one canonical JSON serialization, excluding the hash field itself; artifact bytes have separate SHA-256 values. Record a clean code commit where available, otherwise a reproducible source snapshot hash plus any diff; never invent a commit for untracked work. A final report points to a candidate; a new evaluated release points to that report without editing the candidate.

所有结构有版本与校验器；缺文件、重复 ID、引用失效、未知字段或哈希冲突时失败。manifest 使用统一规范 JSON 序列化计算哈希，排除自身哈希字段；文件字节另算 SHA-256。可用时记录干净代码提交，否则记录可复现源码快照哈希及差异，不给未跟踪代码编造提交。最终报告引用候选，新建的已评测发布清单再引用报告，不修改原候选。

## 3. Verification layers / 验证层次

1. **Offline CPU checks:** small fixed sources, identifier queries, filters/no-match behavior, source maps and long-code excerpts, hand-computable metrics, schema rejection, mock scoring/API failure paths, and tiny checkpoint recovery. No network or downloaded encoder required.
   **离线 CPU 检查：**固定小资料、标识符查询、筛选和无匹配、来源映射与长代码摘录、手算指标、结构拒绝、替代评分器和接口异常、极小断点恢复；不需联网或下载编码器。
2. **Target-machine integration:** real tokenizer/MiniLM, FP32 MPS forward/backward, tiny-set diagnostics, two-epoch recovery comparison, device timing, offline export/reload and CPU/MPS output tolerance. Fix tolerances on small validation fixtures before comparing formal experiments.
   **目标机器集成：**真实 tokenizer／MiniLM、FP32 MPS 前后向、小集诊断、两轮恢复对照、设备计时、离线导出重载及 CPU／MPS 输出容差。正式实验前，先在小验证样例上固定容差。
3. **Application/release checks:** all API modes, invalid/missing assets, partial-score discard, saturation, actual demo access, startup/stop/readiness/rollback, interrupted transfer and clean wheel installation. Benchmark normal and degraded requests separately.
   **应用发布检查：**全部接口模式、非法或缺失产物、丢弃不完整评分、容量耗尽、实际演示访问、启停就绪回滚、传输中断及干净 wheel 安装；正常与降级请求分开计时。

Only relevant tests are required; no giant integration platform is needed. The same fast local test command can later run in CI if the repository is published. An MPS skip is a skip, not a pass; GPU checks must actually run on the supplied machine before the accelerator path is accepted.

只运行与风险相关的检查，不引入大型集成平台。快速本地检查可在未来发布仓库后复用于 CI。MPS 检查被跳过就记作跳过，不能当作通过；加速器路径必须在所提供机器上实际运行才可验收。

## 4. Decisions and failure recovery / 决策与失败恢复

Before model selection, save a validation decision policy: primary metric, family/source coverage, permitted latency, compared training budgets, candidate depth and tie-breaking. Numerical speed limits and maximum run duration are measured on the target Mac and frozen before final test access. Count all completed and failed runs; do not keep only the best seed. Lower throughput changes batch/run size, not the definition of success or test membership.

选模型前保存验证决策策略：主指标、任务来源覆盖、允许延迟、比较训练预算、候选深度和同分规则。速度上限及最长运行时间在目标 Mac 实测后确定，并在读取最终测试前冻结。记录成功及失败运行，不仅保留最佳种子。吞吐较低可以改变 batch 和运行规模，不能因此改写成功定义或测试成员。

| Failure / 失败 | Required response / 处理 |
| --- | --- |
| Unsupported source format or evidence lost / 格式不支持或证据丢失 | Quarantine affected material; report coverage loss; no unsupported labels / 隔离资料、报告覆盖损失、不生成无依据标签 |
| Too few independent groups / 独立组不足 | Shrink the planned count and state the limitation; do not pad with paraphrases / 缩小数量并披露，不用改写凑数 |
| MPS unavailable or memory pressure / MPS 不可用或内存压力 | Diagnose and benchmark an explicit revised config; CPU baseline remains usable / 排查并实测新的明确配置，保留 CPU 基线 |
| Training interrupted / 训练中断 | Restore the last complete epoch, preserve best state, replay the interrupted epoch; label partial snapshots diagnostic-only / 恢复最后完整 epoch 及最佳模型状态，重跑中断轮，中途快照仅诊断 |
| Storage missing or full / 存储丢失或已满 | Stop writes safely, preserve valid artifacts, no silent relocation / 安全停止写入、保留有效产物、不静默迁移 |
| Reranking deadline/error / 重排超时或错误 | Discard partial scores; explicit full-BM25 fallback only when its result is valid / 丢弃部分分数，仅在完整基线有效时明确降级 |
| Backup pending near rental end / 临近退机而备份未完 | Stop new long runs, finish verified export; alert user only if time/storage access requires action / 停止新增长任务，完成验证导出；需要延时或存储访问才通知用户处理 |
| No measured ranking gain / 未获得排序提升 | Retain baseline default if selected on validation; report the failed hypothesis / 按验证选择保留基线默认，如实报告无提升 |

## 5. Task status and final handover / 任务状态与最终交付

Each task record contains `task_id`, `engineering_status`, `learning_evidence_status`, `input_manifest_ids`, `output_manifest_ids`, `check_results`, `open_issues`, and `next_action`. Use `planned`, `running`, `passed`, `failed`, `blocked` or `not_required` as appropriate. T01 and T04 are partially implemented and remain `running` (T01: strict validators, deterministic IDs, configs, corpus/index artifacts and code identity exist; T04: only fixture-level BM25, filters, zero-match and source mapping exist, with T03 real sources, the pinned real tokenizer and 30–50 development queries pending); other tasks remain planned. A missing machine is an execution prerequisite, not a reason to claim a design test passed.

每项任务记录编号、工程状态、学习证据状态、输入输出清单、检查结果、未解决项和下一步。按实际使用 `planned`、`running`、`passed`、`failed`、`blocked` 或 `not_required`。T01 与 T04 部分实现、状态为 `running`（T01：已有严格校验、确定性 ID、配置、语料／索引产物及代码身份；T04：仅有样例级 BM25、筛选、无匹配与来源映射，T03 真实来源、固定真实分词器及 30～50 条开发问题待完成），其他任务仍为计划。机器未提供属于执行前提，不能据此声称设计验证已通过。

Final handover includes runnable source/package, setup and start/stop instructions, versioned source/license inventory, reproducible data/training/evaluation commands, model and index artifacts, experiment and limitation report, backup verification and a short demo walkthrough. Résumé evidence maps claims to these artifacts. The learning record remains separate until the user completes their independent explanation and change.

最终交付包括可运行源码与包、安装启停说明、来源版本许可清单、数据训练评测复现命令、模型索引、实验与局限报告、备份校验及简短演示讲解。简历陈述映射到这些产物；用户独立解释和修改完成前，学习证据保持单独状态。
