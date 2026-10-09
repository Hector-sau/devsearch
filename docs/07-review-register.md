# Architecture review register / 架构评审问题记录

## Revision 4 execution review / 第四版执行评审

On 2026-10-05, three agents reviewed operational reliability, evaluation/data readiness and resource estimates against remote scaffold commit `30927a4`. The new [execution contract](10-accelerator-execution-contract.md), [readiness gates](11-readiness-and-review.md) and [bounded estimates](12-estimates-and-experiment-plan.md) address specification gaps; G01–G10 remain open implementation/evidence requirements. A second review corrected optimizer-update versus microbatch units, bounded calibration, network/backup boundaries, future-output hashes, atomic crash recovery, host-binding revisions and multi-seed bootstrap units. No GPU compatibility or quality result was produced by this review.

2026-10-05 三位 Agent 针对远端骨架提交评审运行可靠性、数据评测就绪及资源估算。新增执行契约、就绪门禁及有界估算处理规格缺口，G01～G10 仍是未关闭的实现证据要求。二轮复核修正优化器更新与微批次单位、校准预算、网络备份边界、未来输出哈希、原子提交崩溃恢复、主机绑定修订及多种子区间抽样单位。本次未产生 GPU 兼容或质量实测。

The earlier revision-3 findings below remain historical design records; a specification resolution never certifies implementation completion.

下列第三版问题保留为历史设计记录，规格修正不等于实现通过。

Reviewed 2026-10-05, design revision 3. Three agents independently examined implementation/runtime, data/evaluation and product/delivery. The coordinating review reconciled their findings and added a task-to-artifact checklist. The original architecture remains suitable: one Python package, BM25, a native PyTorch MiniLM reranker, one local API and a thin UI. No new infrastructure is required by this review.

评审日期 2026-10-05，设计修订版 3。三位 Agent 独立检查实现运行、数据评测和产品交付，主评审统一处理问题并补充任务与产物检查表。原架构仍适用：一个 Python 包、BM25、原生 PyTorch MiniLM 重排器、一个本地 API 和轻量界面。本次修订不要求新增基础设施。

The register consolidates overlapping findings into 12 items. **Closed in design** means the governing contract has been corrected; it does not mean code, tests or experiments have passed. All implementation checks below remain pending. The known P1 ambiguity was the final-test judging protocol; other findings close important execution gaps.

相同问题合并后共 12 项。**设计已修订**表示对应契约已经补齐，不代表代码、测试或实验通过。下列实施验证全部待执行。已知 P1 歧义是最终测试标注流程，其余问题补齐关键实施细节。

## Findings and corrections / 发现与修订

| ID / Priority 优先级 | Original gap / 原缺口 | Corrected contract / 修订契约 | Evidence required in implementation / 实施时必需证据 |
| --- | --- | --- | --- |
| R01 / P1 | Test labels both sealed early and revised after final candidates / 提前封存标签与最终补审冲突 | [Data §5](02-data-and-evaluation.md): initial question/qrels seal; locked systems; blind pooled review; final shared qrels before scores / 首次封存、锁定系统、盲审汇总、最终统一标注后计分 | Both seals, pool provenance, hidden model identities, common scoring denominator and no test retuning / 两次封存、池溯源、模型身份隐藏、共同分母、不据测试调参 |
| R02 / P2 | Query count could exaggerate independent evidence / 问题数量可能夸大独立证据 | [Data §§3–5](02-data-and-evaluation.md): source/task coverage, evidence groups, group-level uncertainty / 来源任务覆盖、证据组、按组不确定性 | Raw/accepted/group counts, overlap audit, paired group interval or disclosed descriptive-only result / 生成接受组数、重复审计、配对组区间或描述性局限 |
| R03 / P2 | Triplets and BCE pairs/mining ownership unclear / 三元组与二元样本及挖掘归属不清 | [Data §4](02-data-and-evaluation.md): unique binary pairs and training-owned mining pools / 唯一二元对及训练归属挖掘池 | Pair-count reconciliation, conflict rejection and no held-out input access / 对数对账、冲突拒绝、不读留出输入 |
| R04 / P2 | Long code and unsupported parsing could silently lose evidence / 长代码及不支持解析可能静默丢证据 | [Data §3](02-data-and-evaluation.md), [product §3](01-product-and-evidence.md): static format support, quarantine, source maps and explicit excerpts / 静态格式、隔离、来源映射、明确摘录 | Long-code fixture and coverage-loss report; snippets not claimed executable / 长代码样例及覆盖损失，不声称摘录可运行 |
| R05 / P2 | Model probe blocked the entire baseline path; data readiness implicit / 模型探测阻塞全部基线，数据就绪不明确 | [Architecture §7](00-architecture-and-roadmap.md): separate CPU/model branches and pre-M2 data gate / CPU 与模型分支分开，M2 前设数据门禁 | BM25 can run without model; training waits for both actual model and data checks / 基线无模型可用，训练等待模型及数据实测 |
| R06 / P2 | Missing lifecycle commands and installed-package checks / 生命周期入口和安装包验收缺失 | [Implementation §§4–5](03-implementation.md): full CLI lifecycle; real CPU tokenizer for baseline; portable config paths / 全链路入口、基线真实 CPU 分词器、便携配置路径 | Offline CPU fixtures, MPS integration, clean wheel startup outside the old checkout/cache / 离线 CPU、MPS 集成、离开原目录缓存的干净 wheel 启动 |
| R07 / P2 | Resume could lose best-model state or imply exact mid-epoch recovery / 续训可能丢最佳模型状态或误称精确中途恢复 | [Implementation §2](03-implementation.md): complete-epoch state, best/next-epoch fields, diagnostic-only partial snapshots / 完整轮状态、最佳与下一轮字段、中途仅诊断 | Continuous-vs-resumed comparison, atomic save and valid previous checkpoint / 连续与恢复对照、原子保存、有效旧断点 |
| R08 / P2 | Concurrency, partial scores, errors and readiness underspecified / 并发、部分评分、错误与就绪不足 | [Integration](04-integration-and-decisions.md): one active search, executor lifetime, complete-or-discard ranking, explicit HTTP and readiness / 单活动搜索、执行器生命周期、全成或放弃、明确 HTTP 与就绪 | Cancellation/deadline/failure tests; no overlapping device work or mislabeled ranking / 取消时限失败检查，无设备任务重叠或排序误标 |
| R09 / P2 | Portable release paths and offline reconstruction incomplete / 便携路径和离线重建不足 | [Implementation §3](03-implementation.md): relative paths, explicit head/tokenizer/config, source snapshot identity / 相对路径、评分层分词配置、源码快照身份 | Hash and containment checks, no-network reload, candidate/report hash graph / 哈希越界校验、离线重载、候选报告哈希关系 |
| R10 / P2 | Remote demo and first successful backup left until handback / 远程演示及首次备份缺少提前验证 | [Handoff](05-machine-handoff.md): localhost SSH access, early export test, periodic verified copies and return window / SSH 私有访问、早期导出、周期校验及退机窗口 | Actual browser access, first smoke export, restore report and latest verified copy / 实际浏览器访问、首次小运行导出、恢复报告、最近有效副本 |
| R11 / P2 | User learning gate could halt authorized engineering / 用户学习门禁可能阻塞已授权开发 | [Product §5](01-product-and-evidence.md): separate engineering and learning evidence / 工程与学习证据分别跟踪 | Engineering continues when passed; résumé mastery waits for independent demonstration / 工程通过继续，个人掌握声明等待独立演示 |
| R12 / P2 | No complete dependency/output/acceptance inventory / 缺少完整依赖产物验收表 | [Execution checklist](06-execution-checklist.md): T01–T12, L01, artifact records and failure actions / 工程及学习任务、产物记录、失败处理 | Each completed task links real inputs, outputs and check results; no planned item marked passed / 完成项关联真实输入输出及检查，不将计划标为通过 |

Additional consistency corrections: BM25 candidates require lexical overlap without assuming scores are positive; global corpus statistics remain fixed under filters. UI renders source text/code as escaped content. Final tests separate answerable ranking queries from invalid, unanswerable and unjudged cases. Test top-k pooling never inserts gold passages into retrieval. These rules fit the existing scope.

同时修订一致性细节：BM25 候选必须有词项交集，但不假定分数为正；筛选不改变全语料统计。界面以转义形式展示来源文本代码。最终测试区分有答案的排序问题与非法、无答案、未判断项。测试候选池补审不会把正确段落塞入检索结果。这些规则均属于原范围。

Follow-up cross-review confirmed closure of R01 and identified three remaining clarifications, now incorporated: M0 probes use independent fixtures and their learned weights are discarded before formal comparisons; M1 prepares real tokenizer files on CPU without encoder weights; installed-wheel configuration cannot rely on the old checkout. The final response schema also exposes code-fragment/source-location metadata and structured warning codes so the UI can implement the product contract.

后续交叉复核确认 R01 已在设计上闭合，并补齐三处澄清：M0 用独立样例，正式对照不沿用其训练后权重；M1 在 CPU 上准备真实分词器但不需编码器权重；wheel 安装后的配置不能依赖旧 checkout。响应结构同时补充代码片段与原文位置字段、结构化警告码，使界面能够落实产品契约。

## Remaining execution evidence / 仍需执行的验证

| Before / 时间点 | Still unknown / 待确定 | Resolver / 如何解决 |
| --- | --- | --- |
| Machine setup / 机器接入 | Actual chip/OS, access, persistent storage, available window / 实际硬件系统、访问、持久存储、使用窗口 | User supplies machine; assistant runs T02 and handoff checks / 用户提供机器，助手执行 T02 与交接检查 |
| First corpus / 首次语料 | Exact pinned revisions, applicable licenses, parser coverage, usable independent evidence / 精确版本、适用许可、解析覆盖、可用独立证据 | T03–T05 manifests and audits; reduce targets when needed / 清单审计，不足则缩小目标 |
| Training / 训练前 | Compatible dependencies, actual MiniLM throughput, memory, checkpoint stability / 兼容依赖、模型吞吐内存、断点稳定性 | T02/T06 measured probes and recovery comparison / 实测探测及恢复对照 |
| Selection / 选型前 | Numeric latency limits, training budget, final default mode / 具体延迟限制、训练预算、最终默认模式 | Freeze the validation decision policy before test access / 读取测试前冻结验证决策策略 |
| Release / 交付前 | Measured quality, actual private demo, clean install and restore / 实测效果、私有演示、干净安装与恢复 | T09–T12 artifacts; disclose failed hypotheses and pending backups / 实际产物，披露负面实验及待完成备份 |
| Personal résumé claims / 个人简历声明前 | Independent explanation and code change / 独立讲解与修改 | L01 recorded demonstration / 记录实际演示 |

These are bounded implementation checks, not unanswered product choices. No additional user decision is needed now. New costs, rental extensions, publication or material scope expansion remain concrete decisions to raise only if they arise.

这些是边界明确的实施检查，不是尚未解决的产品方向选择。现在无需用户新增决策。新增费用、续租、公开发布或重大范围扩展实际发生时，再提出具体决定。

## Review limits and references / 评审边界与参考

This is an independent multi-agent design review followed by consolidation, not a guarantee of no future defects. Structural document checks cover local links, code fences, illustrative JSON and Mermaid-source synchronization; runtime acceptance remains the checklist's job. No project code, dependencies, datasets or models were installed or run in this review.

本次是多 Agent 独立设计评审与统一修订，不保证未来没有缺陷。文档结构检查覆盖本地链接、代码围栏、示例 JSON 和 Mermaid 源码同步；实际运行验收由实施检查表落实。本次评审未安装或运行项目代码、依赖、数据或模型。

Recorded static-check result: 9 Markdown documents, 9 parsed JSON records, two synchronized Mermaid source files, and no detected missing local links, unpaired fences or source-copy differences. Diagram checks verify source structure/synchronization, not a rendered visual inspection. Runtime tasks retain planned status.

已记录静态检查结果：9 份 Markdown、9 条成功解析的 JSON、两份同步 Mermaid 源码；未发现本地链接缺失、代码围栏未配对或图源码副本差异。图检查仅验证源码结构与同步，不等于渲染后的视觉检查；运行任务仍为计划状态。

PyTorch's [MPS documentation](https://docs.pytorch.org/docs/2.14/notes/mps.html) supports checking device availability and actually running tensors on the device; the [save/load guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) is the upstream reference for model/optimizer state restoration. [Sentence Transformers reranker examples](https://www.sbert.net/examples/cross_encoder/training/rerankers/README.html) support the reference pattern of scoring text pairs. The project's split rules, two-stage judging, acceptance gates and storage policy are project design decisions, not claims copied from those sources.

PyTorch MPS 文档用于核对设备检测与实际设备运算，保存加载指南作为模型和优化器状态恢复的上游参考；Sentence Transformers 重排示例作为文本对评分的代码参考。本项目的划分、两阶段审查、验收及存储策略属于自己的设计决定，不冒充上游既有结论。
