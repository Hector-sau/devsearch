# DevSearch — Product, Acceptance, and Evidence Specification / DevSearch：产品、验收与能力证据规格

**Status: proposed scope, not completed work.** The user confirmed English queries and English documents on 2026-10-05. The initial sources are PyTorch Tutorials and Transformers documentation. No improvement, user study, or personal contribution is assumed.

**状态：待实施方案，不代表已完成成果。**用户于 2026-10-05 确认英文问题与英文文档，初始来源为 PyTorch Tutorials 和 Transformers 文档。本文不预设性能提升、真人测试或用户本人贡献。

## 1. Problem and intended use / 问题与使用场景

DevSearch helps a developer find the official explanation or code snippet needed for a concrete programming question. It returns evidence, source links, and documentation versions. The learning objective is to understand and modify a complete PyTorch reranker training and evaluation pipeline.

DevSearch 帮助开发者针对具体编程问题找到官方解释或代码片段，返回证据、来源链接和文档版本。学习目标是能够理解并修改完整的 PyTorch 重排序训练与评估流程。

The initial corpus is official PyTorch and Hugging Face Transformers documentation. Before ingestion, record the permitted use, exact release or documentation branch, source URL, collection date, and snapshot identifier. Pin one version per library for the first corpus; never label an unverified page as current.

初始语料来自 PyTorch 与 Hugging Face Transformers 官方文档。采集前记录使用许可、确切发布版本或文档分支、来源 URL、采集日期与快照标识。第一版每个库固定一个版本，未经核实不得标为“最新”。

Typical questions concern evaluation mode, checkpoint recovery, tokenization, and model loading. Answers must remain traceable to the indexed documentation. These are proposed use cases, not findings from interviews.

典型问题涉及评估模式、断点恢复、分词和模型加载。检索结果必须可追溯到入库文档。上述属于设计用例，不是用户访谈结论。

## 2. MVP and boundaries / 最小版本与范围

The MVP accepts a query and library/version filters, retrieves candidates, reranks the same candidates, and displays both rankings side by side. Each result includes a title, passage, preserved code formatting, source link, version label, score, and rank change. Scores are model outputs, not calibrated probabilities.

最小版本支持输入问题、按库及版本筛选、检索候选和重排序，并排展示同一候选集合的排序前后结果。每条结果包含标题、正文、保留格式的代码、来源、版本、评分及名次变化。评分是模型输出，不应解释为经过校准的正确概率。

A version filter selects an exact indexed documentation snapshot. The first corpus has one pinned snapshot per source; this does not establish compatibility with every installed library version or comprehensive multi-version search. Source code is displayed as evidence, not executed or certified as runnable.

版本筛选选择已入库的确切文档快照。首批语料每个来源仅一个固定快照，不能据此宣称适配所有已安装库版本或全面支持多版本检索。来源代码作为证据展示，不执行，也不认证为可运行代码。

Generative answers, agents, tool execution, multi-turn chat, login, and large-scale deployment are outside the MVP. It can demonstrate search and training ability without claiming a complete RAG or Agent system.

生成式回答、Agent、工具执行、多轮对话、登录与大规模部署均不属于第一版范围。项目展示检索与训练能力，不宣称完成整个 RAG 或 Agent 系统。

## 3. User-facing behavior / 使用行为与异常处理

- Empty queries and unsupported versions produce clear validation errors. A valid filter with no matching documents returns an empty state; never silently search another version.
- No lexical match or an empty filtered corpus produces “no results.” Weak relevance must be distinguished from no retrieved candidates; a learned confidence threshold is optional and requires validation.
- Preserve indentation, language labels when available, and parent headings. Identify incomplete code as an excerpt, preserve its source location, and link to the complete pinned context. Do not label an excerpt as a standalone runnable example.
- Parsing, model loading, and inference failures produce actionable errors. Any fallback ranking must be labeled as the baseline.

- 空问题和不支持的版本返回明确参数错误。合法筛选条件没有匹配文档时显示空状态，不偷偷切换到其他版本。
- 关键词无匹配或筛选后语料为空时显示“无结果”。相关性较弱不等于没有候选；如增加可信度阈值，必须先验证。
- 保留代码缩进、可用的语言标记和所属标题；不完整代码明确标为摘录，保留原文位置并链接完整固定版本上下文，不称为可独立运行的示例。
- 解析、模型加载或推理失败时显示可理解的错误；回退到原始检索排序时明确标记。

Long code follows the data specification's deterministic length and splitting policy. The ranked excerpt must retain the evidence needed for its label; the UI may offer full source context separately and must distinguish that context from the text scored by the model. If no valid excerpt retains the necessary evidence, record a coverage gap and exclude the unsupported query–chunk label. M1 includes a long-code fixture that checks formatting, excerpt labeling, source mapping, and evidence retention.

长代码遵循数据规格中确定的长度与切分规则。参与排序的摘录必须保留标签所需证据；界面可另外提供完整来源上下文，并明确区分完整上下文与模型实际评分文本。若无法构造保留必要证据的有效摘录，记录覆盖缺口并排除无依据的问题—段落标签。M1 加入长代码样例，检查格式、摘录标记、来源映射与证据保留。

## 4. Milestones and acceptance / 阶段与验收

| Stage / 阶段 | Functional gate / 必须通过的功能验收 | Evidence / 留存证据 |
|---|---|---|
| M0 Contracts/resources / 契约与资源 | Verify paths, device and dependencies / 验证路径、设备与依赖 | Resource and compatibility report / 资源兼容报告 |
| M1 Usable baseline / 可用基线 | Fixed docs → chunks → BM25 → source results / 固定文档→切块→BM25→来源结果 | Development queries, corpus manifest, candidate records / 开发问题、语料清单、候选记录 |
| M2 Training/comparison / 训练与对照 | Tiny training loop then frozen/full comparison on validation / 小训练闭环后进行验证集冻结与全量对照 | Config, logs, recovery tests and metrics / 配置日志、恢复检查与指标 |
| M3 Domain adaptation / 领域适配 | One controlled data/training change, optional measured expansion / 一项数据或训练对照，按实测决定扩量 | Domain labels, split audits and failed/successful results / 领域标签、划分审查及完整结果 |
| M4 Release candidate / 候选版本 | API/UI, compatible artifacts, explicit fallback / 接口界面、兼容产物与明确降级 | Candidate manifest, source links and latency checks / 候选清单、来源和耗时检查 |
| M5 Final evidence / 最终证据 | Locked test and independently loadable local release / 封存测试与可独立加载的本地发布 | Reproduction commands, report, separately recorded learning evidence / 复现命令、报告、单独记录的学习证据 |

The [overall blueprint](00-architecture-and-roadmap.md) defines milestone dependencies. These are functional gates, not promises of quality gains. Report Recall@k, MRR/nDCG with defined relevance judgments, and inference latency on named hardware. Quality targets and acceptable latency remain undecided until the first baseline and hardware benchmark; no percentage improvement is imposed.

整体蓝图统一阶段依赖。上述是功能验收，不是效果承诺。报告定义明确的 Recall@k、MRR／nDCG，以及指定硬件上的推理耗时。质量目标与可接受延迟，待基线和硬件测试后确定，不预先强加提升百分比。

Track `engineering_status` and `learning_evidence_status` separately. Passing a technical gate allows the authorized engineering work to continue while a learner exercise is pending. Offer a batch explanation and small modification exercise at M2; verify independent understanding before personal PyTorch mastery claims are used in the résumé. Delivering working software does not automatically complete this learning gate.

分别记录 `engineering_status` 与 `learning_evidence_status`。技术门禁通过后，即使学习练习待完成，也可继续获授权的工程工作。M2 提供一次 batch 讲解与小修改练习；在简历使用本人掌握 PyTorch 的陈述前，验证独立理解。软件交付完成不自动代表学习验收完成。

Keep the candidate set fixed for reranker comparisons. In end-to-end results, a missing relevant passage remains a miss. Record warm-up, batch size, query/candidate counts, and measurement repetitions for latency; distinguish model-only time from total search time.

重排序对比固定候选集合；端到端结果中，漏检相关段落必须计为失败。耗时记录预热、批大小、问题及候选数量与测量次数，并区分模型耗时和完整检索耗时。

## 5. Evidence and ownership / 证据与本人职责

Agents may research references, draft code, propose tests, and summarize logs. The learner must independently explain tensor shapes, loss/labels, gradient flow, train/eval behavior, and checkpoint state, then implement one agreed change and interpret its result. Record assistance honestly; generated code alone does not demonstrate personal mastery.

Agent 可以调查资料、起草代码、设计测试和整理日志。学习者须独立解释张量维度、损失与标签、梯度传播、训练／评估模式和检查点状态，再完成一项约定修改并解释结果。如实记录辅助工作；生成代码本身不能证明本人掌握。

Trace every résumé claim through: claim → experiment/run ID → code identity and config → data snapshot and split → logs, metrics, and representative failures. Code identity is a clean commit or a preserved source snapshot hash as defined in the implementation specification. Label synthetic queries, automatic labels, manual reviews, and public-data labels separately. Offline evaluation and scripted tests must never be presented as interviews, real-user studies, or observed adoption.

每条简历陈述都沿“陈述→实验编号→代码身份与配置→数据快照及划分→日志、指标和失败案例”追溯。代码身份按实现规格使用干净提交或已保留源码快照哈希。区分生成问题、自动标注、人工审核与公开数据标注。离线评测和脚本测试不能包装成访谈、真人测试或实际用户使用。

## 6. Résumé draft after completion / 完成后简历占位稿

> Interview preparation, including what must be explained personally before claiming mastery: [INTERVIEW_GUIDE](INTERVIEW_GUIDE.md). 面试准备见 INTERVIEW_GUIDE。

- Implemented a PyTorch document reranker with custom training, validation, and checkpoint recovery on [N] labeled pairs from [dataset/version].
- Compared [strategies] on [Q] held-out queries; measured [metric/result] against [baseline] under fixed candidate conditions.
- Built a version-aware documentation search demo; measured [latency] on [hardware] and documented [failure categories] with reproducible experiments.

- 基于 [数据集／版本] 的 [N] 条标注样本，实现 PyTorch 文档重排序模型，完成训练验证与断点恢复。
- 在 [Q] 条独立测试问题上对比 [策略]，固定候选条件下取得 [指标／结果]，基线为 [基线结果]。
- 构建支持版本筛选的技术文档检索演示，在 [硬件] 上测得 [耗时]，通过可复现实验分析 [失败类型]。

Replace placeholders only with verified evidence. If fine-tuning does not improve quality, describe the measured trade-off and diagnosis instead of claiming an improvement.

占位内容只能替换为已验证证据。若微调没有改善质量，陈述实际取舍与诊断结果，不编造提升。
