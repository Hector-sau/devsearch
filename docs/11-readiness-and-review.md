# Execution readiness review / 执行就绪评审

Reviewed 2026-10-05, starting at `main` commit `30927a4a6b89b53dd6c9f7a78000216970510258`, then incorporating the newly visible development branch `claude/youthful-turing-uc1lkv` at `ee4edea2d53e103119b5f45fe4f1278e4f478499`. That branch adds the CPU baseline; no implementation PR was open at recheck. Later/unpushed changes remain unknown. This review changes documentation, not implementation. Three agents reviewed runtime/recovery, evaluation/data, and resources/estimation; this is agent review, not human testing.

2026-10-05 从上述主分支提交开始评审，再补查新出现开发分支的上述提交；其增加 CPU 基线，复查时尚无实现 PR。更晚或未推送变更仍未知。本评审只改文档，不改实现。三位 Agent 分别评审运行恢复、数据评测、资源估算；这是 Agent 评审，不是真人测试。

## 1. Finding and present state / 结论与当前状态

The architecture is suitable for a bounded single-device project, but is **not accelerator-ready**. The new branch implements `index`, `retrieve --mode baseline`, `validate_manifest`, TOML index/search configs and strict fixture corpus/index contracts. Its seven training/data/evaluation/service/export entry points remain stubs and production ML dependencies are unresolved. An independent temporary-copy run passed 125 tests on macOS/Python 3.12.14; [GitHub run 37307966474](https://github.com/Hector-sau/devsearch/actions/runs/37307966474) succeeded on preceding code-identical commit `fda9eea`. These establish CPU fixture behavior only; see [verification](../reports/execution-review-verification.md).

架构适合有界单设备项目，但**尚不具备直接加速运行条件**。新分支已实现建索引、基线检索、清单校验、TOML 索引搜索配置及严格样例产物契约；七个训练数据评测服务导出入口仍为占位，生产 ML 依赖未解决。本次在临时副本上用 macOS／Python 3.12.14 独立通过 125 项测试，也核实前一代码相同提交的 GitHub 运行成功；这些只证明 CPU 样例行为，详见验证记录。

| Readiness / 就绪项 | Observed state / 当前状态 | Closure evidence / 关闭证据 |
| --- | --- | --- |
| Specification / 规格 | Revision 4 addendum written; implementation pending / 已写第四版补充，待实现 | Consistent 02/10/11/12 contracts and implementer acknowledgment / 数据执行契约一致并被接手 |
| `implementation_verified` | False / 未通过 | All required operational paths and meaningful CPU tests, including integrated orchestration / 实际命令及有效 CPU 测试，含集成编排 |
| `data_ready` | False / 未通过 | Real immutable corpus/pairs/splits/candidates/reviews/seals and coverage audit / 实际不可变数据候选审查封存及覆盖审计 |
| `portable_bundle_verified` | False / 未通过 | Clean offline installation and diagnostic end-to-end run outside source checkout / 脱离源码的干净离线安装及诊断全流程 |
| `backend_verified.apple-mps` | Unverified / 未验证 | Actual MiniLM train/resume/inference run on supported Apple target / 真实模型在目标 Apple 上训练恢复推理 |
| `backend_verified.nvidia-cuda` | Unverified / 未验证 | Separate actual MiniLM train/resume/inference run on supported NVIDIA target / NVIDIA 目标上的独立真实验证 |
| `accelerator_run_ready` | False for both profiles / 两种均未通过 | Conjunction of all applicable gates for exact input/profile identity / 确切输入配置的全部相关门禁同时通过 |

Only the chosen backend must pass to execute its campaign; the unused backend stays unverified rather than blocking it. Both profiles must pass independently before advertising verified support for both. An automated doctor on the supplied host may close the missing backend gate; the implementation must already contain that tested path, with no interactive source patching.

实际运行只要求所选后端通过，未选后端保持未验证，不阻塞已就绪后端；宣称双后端已验证则必须分别通过。所提供主机上的自动自检可补齐该设备门禁，但代码必须事先具有经过测试的流程，不依赖交互式修改源码。

## 2. Blocking gaps and acceptance / 阻塞项及验收

All gates below remain open at the end-to-end level. G01 has partial implementation evidence for three CPU commands, G03 for fixtures only; credit that work and extend it instead of rebuilding it. A documented remedy is not closure. Gate IDs are used by the future `acceptance-report.json`; attach exact commands, artifact hashes and observed results for each.

以下门禁在端到端层面均未关闭。G01 已有三个 CPU 命令的部分实现证据，G03 只有样例证据，应复用扩展而非重建。写明修正方法不等于已关闭，未来验收报告逐项附命令、产物哈希及实际结果。

| ID / Owner | Gap / 缺口 | Required artifact and decisive check / 必需产物及关键验收 |
| --- | --- | --- |
| G01 / A | Stub commands and missing configs / 占位命令与配置缺失 | Implement required CLIs with strict config schemas; invalid fields fail, real tiny workflow produces validated artifacts, help is not success / 严格配置，非法项失败，真实微型流程输出经验证产物 |
| G02 / A+C+D | No platform-locked portable runtime / 无平台锁及便携环境 | MPS and CUDA profiles, compatible hashed locks, bootstrap/offline assets; fresh installed wheel completes diagnostic flow outside repository / 双配置依赖锁安装及离线资产，仓库外安装完成诊断 |
| G03 / A+B | Dataset counts are targets / 数据仍为目标 | Pinned licenses/source map/tokenizer, real accepted group/split counts, train-only negatives and unique pairs; automatic cross-split and evidence audits / 固定来源分词及实际分组数量、训练负例唯一对、交叉划分证据检查 |
| G04 / B | Final judging requires a later input / 最终审查需后续输入 | For no-intervention mode, pre-review all fixed test candidates plus positives; corrupt/missing coverage blocks final sealing; never inject positives into retrieval / 无干预模式预审全候选及正例，缺覆盖阻止封存，不补召回正例 |
| G05 / A+D | No campaign controller/state protocol / 无运行控制器及状态协议 | Immutable plan, atomic stage manifests, single-owner lock, status/resume, clear exit codes; rerun reuses only hash-valid stages / 不可变计划、原子清单锁、状态恢复退出码，仅复用哈希有效阶段 |
| G06 / C | Device feasibility assumed, not measured / 设备可行性未实测 | Isolated diagnostic MiniLM updates, freeze checks, representative/max-length memory and timing, backend unavailable fails explicitly / 独立真实模型更新冻结及长度内存耗时，设备缺失明确失败 |
| G07 / C | Training/recovery and accumulation unverified / 训练恢复累积未验证 | Sample-weighted partial accumulation; identical data order/update/LR sequence continuous vs epoch-resumed within declared tolerances; no diagnostic-weight reuse / 不足组样本加权、连续恢复对照数据更新学习率容差，不复用诊断权重 |
| G08 / B+C+D | Selection/retries rely on live judgment / 选择与重试依赖临场判断 | Frozen matrix/seeds/selection/latency policies; default chosen before test; no-positive-gain and failed-seed paths yield honest outcomes / 固定矩阵种子选择耗时，测试前选默认，无提升及种子失败如实输出 |
| G09 / D | Online/offline and release path not exercised / 离在线发布未演练 | Same pair encoding/scoring, deterministic IDs/ties, full-result fallback, health/busy/deadline behaviors; corrupted bundle cannot activate / 相同编码评分、确定排序、完整降级、健康忙碌超时，损坏包不能激活 |
| G10 / A+D | Reporting/backup can be left manual / 报告备份可能仍靠手动 | Measured report + per-query outputs + artifact lineage + offline reload + independent backup checksum receipt; missing stage cannot be summarized complete / 实测报告逐题来源、离线重载与独立备份凭据，缺阶段不算完整 |

## 3. Concrete work order for the implementing AI / 开发 AI 的具体执行顺序

### Fixes confirmed in the concurrent CPU branch / 已复现的并行开发分支修正项

The existing 125 tests pass, but three independent temporary-fixture probes exposed the following defects at `ee4edea`. These are specific fixes for that implementation, not a request to restart it. No formal training/test leakage has occurred; the ownership issue is a validation gap that must close before training data reuse.

已有 125 项测试通过，但在 `ee4edea` 的独立临时样例探测中复现以下三项缺陷。它们是针对现有实现的修正，不要求重做；尚未发生正式训练测试泄漏，归属问题是进入训练复用前必须关闭的校验缺口。

| ID / Priority / Owner | Verified behavior / 已验证行为 | Fix and acceptance / 修正与验收 |
| --- | --- | --- |
| F01 / P1 / A+B+D | [`write_corpus`](https://github.com/Hector-sau/devsearch/blob/ee4edea2d53e103119b5f45fe4f1278e4f478499/src/devsearch/data/corpus.py#L302) writes its final manifest before validation. An invalid HTTP source URL raises `ManifestError` but leaves `corpus-manifest.json`; retry then fails the immutable-corpus guard / 先提交后校验，非法来源留下正式清单，重试被拒绝 | Stage and validate files before publishing a final manifest, retain content identity, and test invalid input plus both crash windows. Failed writes must be safely retryable / 暂存验证后发布，保留内容身份，覆盖非法输入及崩溃窗口，失败可安全重试 |
| F02 / P1 before training / B | [`load_corpus`](https://github.com/Hector-sau/devsearch/blob/ee4edea2d53e103119b5f45fe4f1278e4f478499/src/devsearch/data/corpus.py#L240) accepts two chunks with the same evidence group but `train` and `test` owners when hashes are valid / 哈希正确时，同证据组的训练及测试冲突归属仍通过 | Require one owner per evidence group across all chunks; allow all-null unassigned groups during preparation, reject mixed null/assigned or different assigned owners; test both conflicts before pair mining / 同组唯一归属，准备阶段可全空，拒绝空与已分配混用或不同划分，挖掘前验证 |
| F03 / P2 / A+D | [`retrieve`](https://github.com/Hector-sau/devsearch/blob/ee4edea2d53e103119b5f45fe4f1278e4f478499/src/devsearch/retrieve.py#L66) validates the index root but writes `--output` outside it; the probe returned 0 and created the file / 索引受根目录限制，输出却可越界，探测退出 0 且文件存在 | Resolve/contain output paths including symlink parents before writing; add outside-root and symlink-escape checks. Intentional small Git reports use an explicit separate export contract / 写前检查输出及父符号链接，补越界测试；小型 Git 报告另走明确导出契约 |

Keep the branch's TOML configuration. Host/config input formats may follow that convention; resolved execution plans, manifests and reports use canonical JSON. Reuse existing canonicalization/path helpers and content-addressed index IDs. Campaign/stage/attempt identity belongs in a separate stage receipt, not in reusable index content identity.

保留开发分支的 TOML 配置，主机配置输入可沿用该格式，解析后执行计划清单报告用规范 JSON。复用现有哈希路径工具及内容寻址索引 ID，实验阶段尝试身份另存阶段凭据，不写进可复用索引的内容身份。

### Delivery sequence / 交付顺序

1. **Agree the execution contract, fix F01–F03, then finish T01/T03/T04.** Keep the CPU baseline implementation. Extend shared schemas and source/tokenizer preparation, then verify natural BM25/filter/source behavior on real data. Keep runtime assets outside Git. Owner A, with B/D for contracts.
   **先对齐契约并修正 F01～F03，再完成基础及检索。** 保留现有 CPU 实现，扩展共享结构及来源分词器准备，在真实数据上验证自然召回、筛选和来源。运行资产不进 Git，由 A 主责、B/D 对齐契约。
2. **Close T05 before formal training.** Prepare actual data and full fixed-candidate review under the chosen protocol; publish a data-readiness report and initial seal. Do not delay labeling work until GPU rental time. Owner B.
   **正式训练前完成数据门禁。** 按所选协议准备实际数据及完整候选审查，输出就绪报告和首次封存，不把标注拖到租用 GPU 时再开始，由 B 负责。
3. **Implement T02/T06 and backend profiles.** Test native training/accumulation/checkpoint logic on tiny CPU models first; package real-model diagnostic probes for both backends. Real backend acceptance still waits for an accessible device. Owner C.
   **完成训练恢复与双设备配置。** 先用 CPU 微型模型验证原生训练、累积和断点，打包两种后端的真实模型诊断，真实设备验收仍需可访问机器，由 C 负责。
4. **Implement the controller and T09 in parallel.** Controller calls existing modules; fake scorers test failure transitions only. API/UI must work in baseline mode and pass parity/fallback checks with a diagnostic model. Owners A+D.
   **并行完成控制器与应用。** 编排调用已有模块，假评分器只测异常状态；接口界面先支持基线，再以诊断模型通过一致与降级检查，由 A/D 负责。
5. **Rehearse all stages with diagnostic data.** Build/install the complete bundle in a clean environment, block network asset acquisition while allowing loopback and the declared backup transport, then execute train→validation→lock→rank→seal→score→export→reload/backup. Inject the failures below. This rehearsal is engineering evidence and never consumes the formal test set. Owners A+B+C+D.
   **用诊断数据演练全流程。** 干净环境安装完整包后阻止联网获取资产，允许回环和声明的备份传输，再跑训练、验证、锁定、排序、封存、计分、导出及重载备份，并注入下述失败。演练属于工程证据，不消耗正式测试集，由四线共同验收。
6. **Deliver the run bundle and exact acceptance report.** The user supplies the host; doctor closes its backend gate, calibration resolves finite profile choices, and plan/run executes the frozen matrix. T07/T10/T11/T12 then run from existing code. T08 expansion is optional and must be declared before the formal campaign or deferred to a new protocol; it is not a live GPU-stage coding task.
   **交付运行包及精确验收记录。** 用户提供主机后，自检补齐设备门禁，校准解析有限配置，再执行锁定矩阵。对比、模型锁定、测试与交付都调用已有代码。T08 扩展为可选，正式实验前声明或留给新协议，不在 GPU 阶段现场补代码。

## 4. Required failure-injection rehearsal / 必需的故障演练

| Scenario / 场景 | Required behavior / 必须行为 |
| --- | --- |
| Crash after stage output but before manifest commit / 写输出后提交清单前崩溃 | Ignore incomplete output, reuse only previously committed valid stages / 忽略不完整输出，仅复用已提交有效阶段 |
| Crash after manifest commit but before state update / 清单提交后状态更新前崩溃 | Verify plan/stage/attempt/input/output identity and reconstruct succeeded state; never overwrite the committed artifact / 核对计划阶段尝试输入输出后重建成功状态，不覆盖提交产物 |
| Interrupt mid-epoch and restart / 轮中中断后重启 | Replay from last epoch boundary with correct optimizer/scheduler/RNG state; diagnose partial file / 正确恢复优化器调度随机状态并重跑，半文件仅诊断 |
| Two invocations, or training plus serving / 双调用或训练服务同时启动 | Second accelerator owner rejected; no simultaneous mutation or device contention / 拒绝第二持有者，避免并发修改争用 |
| OOM, nonfinite score/loss, unsupported operator / 显存不足非有限值算子不支持 | Fail explicit gate; no hidden CPU/precision/length/data change; no completed result / 明确失败，不暗改设备精度长度数据，不产完成结果 |
| Missing mount, disk full, corrupt checksum / 挂载丢失盘满哈希损坏 | Preserve prior valid artifacts, fail safely, do not relocate to internal disk / 保留旧有效产物安全失败，不改存内置盘 |
| Missing review or mismatched candidate universe / 缺审查或候选集合不一致 | No final metrics/evaluated bundle; fixed-prejudge fails, postrank mode waits / 不计最终分不发布，预审模式失败，后审模式等待 |
| Changed source/model/config after sealing / 封存后来源模型配置改变 | Hash mismatch prevents stale-stage reuse; changed recipe needs new identity / 拒绝旧阶段复用，改配方须新身份 |
| Insufficient remaining epoch/export window / 剩余时间不足 | `stopped_budget`, retain resumable state and partial-status report / 预算停止，保留恢复状态及未完成报告 |
| Reranker worse than BM25 / 重排差于基线 | Follow validation-only default rule, retain all final results and failed hypothesis / 遵循验证选择，保留全部结果及失败假设 |
| Missing real backend or model asset / 缺真实后端或模型文件 | `unverified`/failed prerequisite, never fixture success relabeled real / 未验证或前提失败，不以样例通过替代真实验证 |

## 5. Coordination and review acceptance / 协作与本次评审验收

This branch changes documentation only. The implementation AI should integrate the requirements into its branch and map each G gate to tests/actual artifacts; it must not replace its work with the old scaffold. The pre-reviewed candidate protocol is a deliberate extension to document 02; all group ownership, known-positive recall limits and anti-leakage rules remain. Latest device selection supersedes older Mac-only planning statements. Do not merge over newer changes without checking their contracts.

本分支仅改文档。开发 AI 应将要求整合至正在开发的分支，把每个 G 门禁映射到测试和实际产物，不用旧骨架覆盖其代码。预审全候选是对第 02 号文档的明确扩展，分组归属、已知正例召回局限及防泄漏规则继续有效。最新双设备选择替代旧的仅 Mac 规划；合并前检查新代码契约，不能覆盖更新工作。

For this review delivery, verify local links, Markdown/JSON examples, formula arithmetic, diagram-source parity and consistency between 02/10/12. These checks validate documents only. The other AI's code and real data/GPU gates are not marked passed. A finite design plus tested failure handling reduces code intervention; it cannot guarantee unknown third-party hardware, dependencies or model quality.

本次评审交付检查本地链接、Markdown／JSON 示例、算术、图源一致及 02/10/12 契约一致；这些仅验证文档，不将另一位 AI 的代码或实际数据／GPU 门禁标为通过。有界设计及故障处理测试可以减少代码干预，但不能保证未知第三方硬件、依赖或模型效果。
