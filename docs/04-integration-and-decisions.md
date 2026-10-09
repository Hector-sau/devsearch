# Integration and Decisions 集成与决策

> **The two dated status notes below (2026-10-05) are history.** The decision log was updated on 2026-10-08; implemented decisions with code links (reference commit `e3168ca`) are in *Implemented decisions*. 日期状态段落为历史，决策表已于 2026-10-08 更新。

**Updated user decision, 2026-10-05:** retain both Apple MPS and NVIDIA CUDA profiles; the final machine is undecided. [10 Execution contract](10-accelerator-execution-contract.md) governs automated campaigns and validation-only selection. It supersedes older Mac-only rows below; API/fallback/release invariants remain unchanged. No compute purchase or public deployment is implied.

**2026-10-05 用户更新：**保留 Apple MPS 和 NVIDIA CUDA 两种配置，最终机器未定。第 10 号执行契约管理自动实验和仅验证集选择，替代下文旧的仅 Mac 决策；接口降级发布约束不变，不代表采购算力或公开部署。

This document defines API details and records implementation decisions. The [overall architecture blueprint](00-architecture-and-roadmap.md) owns system boundaries, lifecycle and milestone order; the contracts here implement that blueprint. All endpoints, files and example values below are proposed contracts, not implemented features or measured results.

本文定义接口细节并记录实施决策；整体架构蓝图管理系统边界、生命周期与阶段顺序，本文契约落实该蓝图。以下接口、文件和示例值均为设计契约，不代表已实现功能或实测结果。

## Decision log 决策记录

| Decision 决策 | State 状态 | Impact 影响 |
| --- | --- | --- |
| English queries and English documents / 英文问题与英文文档 | User confirmed 2026-10-05 / 用户已确认 | MiniLM English encoder remains the initial candidate / 保留英文 MiniLM 作为首个候选 |
| Markdown in current AI foundation workspace / 当前工作区 Markdown 文档 | User confirmed 2026-10-05 / 用户已确认 | Save under `devsearch/`; no environment installed internally / 文档保存在 `devsearch/`，不在内置盘安装环境 |
| External storage preferred / 优先外置盘存储 | Applies to the operator's Mac: final model and reports go to the local project directory first, the external drive later (owner, 2026-10-08) / 本机先存项目目录，之后转外置盘 | Verify storage on the actual runtime host; remote machines have separate paths / 检查实际运行主机的存储，远端路径独立 |
| User-provided Mac mini M6, 24 GB / 用户提供 24 GB Mac mini M6 | **Superseded:** the GPU run uses a rented NVIDIA host (AutoDL, RTX 4090); MPS stays optional / 已被取代：改用租用的 NVIDIA 主机（AutoDL），MPS 为可选 | Assistant handles migration, setup and execution after access; no GPU-cloud purchase / 接入后助手负责迁移、配置与运行，无需采购云 GPU |
| No additional paid services or public deployment / 不新增付费服务或公开部署 | Current scope; the GPU rental was arranged by the owner, no paid API used / 当前范围；GPU 租用由所有者安排，未使用付费 API | Private project transfer to the provided Mac is authorized / 已授权向用户提供的 Mac 私下迁移项目 |
| MiniLM plus native PyTorch loop / MiniLM 加原生 PyTorch 循环 | **Implemented**; MiniLM pinned at `44acabbe…`; first GPU campaign run, no gain over BM25 / 已实现；首轮 GPU 实验未优于 BM25 | Verify tokenizer/model/device compatibility at M0 / M0 验证 tokenizer、模型与设备兼容性 |
| Modular monolith, one API worker, cache off / 模块化单体、单 API worker、缓存关闭 | Architecture revision 2 / 架构修订版 2 | Shared CLI/API core; UI does not load a model / CLI 和 API 共用核心，界面不加载模型 |
| Versioned release bundle and restart / 版本化发布包与重启切换 | Architecture revision 2 / 架构修订版 2 | Validate compatibility; preserve previous version for rollback / 校验兼容，保留旧版本用于回滚 |
| BM25 top 50 then top 5 display / BM25 前 50 条重排后展示 5 条 | **Implemented as configured** (`candidate_k` 50, `top_k` 5); not tuned / 已按配置实现，未调优 | Tune only on validation data; persist candidates / 仅用验证集调整，保存候选列表 |
| Pinned PyTorch Tutorials and Transformers English docs / 固定版本英文文档 | First release indexes **PyTorch Tutorials only**, pinned at `11512db7…`; Transformers docs not included / 首版仅收录 PyTorch Tutorials，未纳入 Transformers 文档 | Use one version per source in first release / 首版每个来源只纳入一个版本 |
| Config files are TOML read with stdlib `tomllib` / 配置文件采用标准库 `tomllib` 读取的 TOML | Engineering decision 2026-10-05, replacing the placeholder `.yaml` names / 工程决定，取代占位的 `.yaml` 名称 | No dependency, comments allowed, duplicate keys rejected; the CPU baseline and its offline wheel check stay dependency-free. Revisit only if a real need appears / 无依赖、可注释、拒绝重复键，基线及离线 wheel 检查保持零依赖；有实际需要再重新评估 |
| Baseline tokenizer `lexical-v1`, Okapi BM25 with Lucene IDF, k1 1.2, b 0.75, no stemming / 基线分词 `lexical-v1`、Okapi BM25（Lucene IDF）、k1 1.2、b 0.75、不做词干化 | Engineering decision, **untuned starting values** / 工程决定，**未调优起始值** | Tune and consider stemming only on validation queries over the real corpus (T04); a rule change bumps the tokenizer fingerprint and needs a new index / 仅在真实语料的验证问题上调参并评估词干化（T04）；规则变化会改变分词指纹并需新索引 |


### Implemented decisions / 已实现的技术决策（reference commit `e3168ca`）

Each row was checked against the code at the reference commit. "Why" states the reason recorded in the design documents; "State" adds
the outcome of the first formal GPU campaign where it bears on the decision ([experiment report §5](../reports/experiment-report-draft.md)).

下表每项均已对照参考提交的代码核对，“Why”为设计文档记载的理由，“State”补充第一轮正式 GPU 实验中与该决策相关的结果（BM25 仍为默认）。

| Decision | Why | Code | State |
| --- | --- | --- | --- |
| Fixed BM25 candidate lists; every reranker may only reorder them | the same candidates for every system make the comparison fair and let test labels be complete before training | [`build_candidates`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/candidates.py#L62), [`verify_permutation_output`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/seals.py#L136) | implemented |
| Blind pre-review of all test candidates, then an initial seal (`prejudged_fixed_candidates_v1`) | no label can be influenced by a system's ranking; the campaign can run unattended | [`assert_blinded`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/review.py#L138), [`build_initial_seal`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/seals.py#L151) | bank complete, sealed |
| Review protocol revision 5: one primary pass, stratified second review, escalation above 10% disagreement, independent adjudication | two complete passes were extrapolated at about 34 million judge tokens versus about 19 million for revision 5 before escalation ([15](15-review-workflow.md)); the audit measures disagreement and expands review where it is high | [`evaluate`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/review_protocol.py#L117) | executed: 4 positive strata escalated |
| Unresolved judgments block sealing; never default to 0 | a missing label silently counted as irrelevant would bias every metric | [`write_bank`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/data/seals.py#L67) | implemented |
| Reranking is all or nothing, with a complete BM25 fallback and a warning code | a partially reranked list is neither BM25 nor the model; users must know which ranking they see | [`SearchPipeline._rerank`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/serving/pipeline.py#L132) | implemented, CPU-verified |
| One search at a time, no queue; the deadline starts at admission | one device, predictable latency; a busy service says so (HTTP 429) instead of queuing | [`SearchService.submit`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/serving/api.py#L112) | implemented |
| No truncation: a pair longer than 256 tokens is an error | training, evaluation and serving must see the same input; the corpus is chunked to fit the budget instead | [`encode_pair`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/policy.py#L59) | implemented |
| Deterministic ordering: score, then `chunk_id` | identical rankings online, offline and across reruns | [`order_by_score`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/ranking.py#L18) | implemented |
| Native PyTorch loop; epoch-boundary checkpoints; resume replays the interrupted epoch | every training choice visible in one file; recovery reproduces the uninterrupted run | [`train`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/loop.py#L113), [`save_checkpoint`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/training/checkpoint.py#L46) | implemented; crash/resume verified on CPU and on CUDA (`diagnose`) |
| Two recipes: frozen encoder (lr 1e-3) and full fine-tuning (lr 2e-5), seed 42 | a cheap and a strong option; different learning rates make it a comparison of recipes, not of freezing alone | [`CrossEncoder.set_encoder_frozen`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/modeling/cross_encoder.py#L36), `configs/protocol.toml` | run on AutoDL (`e3168ca`): full fine-tuning not better than BM25; frozen far worse (experiment report §5) |
| Validation-only selection with predeclared gates; test data readable only after selection | prevents tuning on the test set; "BM25 stays the default" is a valid outcome | [`select`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/selection.py#L74), [`World.evaluation`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/execution/world.py#L176) | applied in the first formal campaign: **BM25 stays the default** |
| Paired group bootstrap for confidence intervals | questions about the same evidence are not independent; both systems share every resample | [`paired_group_bootstrap`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/evaluation/metrics.py#L79) | implemented |
| Campaign stages commit atomically and are reused only while their hashes verify | an interrupted rental resumes without trusting log lines or regenerating committed results | [`Campaign._execute`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/src/devsearch/execution/controller.py#L322) | implemented; crash/resume verified on CPU and on CUDA (`diagnose`) |
| Hashed dependency locks, offline install, sealed receipts; CI-import mode for hosts without `unshare -rn` | the same dependencies everywhere; portability proven where isolation can be proven, never claimed for the rented host | [`step_receipts`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/scripts/gpu.py#L617), [`imported_evidence`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/scripts/gpu.py#L345) | implemented; CI evidence green |
| Pinned model fetched once and verified; `model.source = "local"` when the host cannot reach the hub | same weights byte for byte; the local route keeps the fetch manifest and hashes | [`step_model`](https://github.com/Hector-sau/devsearch/blob/e3168ca6fef9b64ea0688ea059332c729125e30f/scripts/gpu.py#L778) | local route used in the first formal campaign (reported by Codex) |

The user arranges the machine and access later; no rental listing or GPU-cloud budget is needed now. The assistant owns routine technical decisions and execution within the provided machine's resource limits. Consult the user for a rental extension or added cost, material scope change, publication or an actual participant study. Sample sizes, package versions and useful batch sizes follow measurements. See the [machine handoff](05-machine-handoff.md) for ownership and transfer acceptance.

用户后续安排机器与访问，目前无需租赁链接或云 GPU 预算。助手在所提供机器的资源限制内负责常规技术决策和执行。租期延长或新增费用、重大范围变化、公开发布或真人参与研究时再与用户沟通。样本量、包版本和合适 batch 根据实测确定。具体分工和迁移验收见机器交接文档。

## Module contracts 模块契约

| Module 模块 | Input 输入 | Output 输出 | Owner specification 规格 |
| --- | --- | --- | --- |
| Ingest 采集 | Allowed repositories and pinned revisions / 允许的仓库与固定版本 | Source manifest, extracted documents / 来源清单、提取文档 | Data 数据 |
| Chunk 分段 | Text, headings, code and source identity / 正文、标题、代码、来源身份 | Stable chunks and source spans / 稳定段落 ID 与原文位置 | Data 数据 |
| Prepare labels 标签准备 | Queries, verified support spans / 问题与已核对证据范围 | Query groups, qrels, training pairs / 问题组、相关性标注、训练样本 | Data 数据 |
| Index 索引 | Frozen corpus snapshot / 固定语料快照 | BM25 index and index manifest / BM25 索引与清单 | Integration 集成 |
| Train 训练 | Split manifest, pairs, config / 划分清单、样本与配置 | Checkpoints, logs, model manifest / 模型、日志与清单 | Implementation 实现 |
| Evaluate 评估 | Model, fixed queries/candidates/qrels / 模型及固定问题、候选、标注 | Per-query scores, aggregates, error report / 每题成绩、汇总和错误分析 | Data 数据 |
| Serve 服务 | Approved index and model pair / 验证后的索引模型组合 | Local search API and comparison UI / 本地搜索接口与对比界面 | Integration 集成 |

No ingestion step executes tutorials, notebooks or repository scripts. Treat source contents as data. Parse supported text formats and extract code as text; retain headings and code boundaries. Respect each source's license and attribution requirements, and record any restrictions before redistribution.

采集阶段不执行教程、Notebook 或仓库脚本，只把内容当作数据。解析支持的文本格式，将代码作为文本提取，保留标题和代码边界。记录各来源的许可、署名及再分发限制。

Documentation and code snippets are untrusted content in the UI as well. Render escaped text and code, never executable source HTML/JavaScript or unsafe HTML interpolation. Source links must use validated HTTPS URLs on the configured upstream source hosts; reject script/data schemes and malformed URLs. This is a rendering/link-validation rule, not an additional approval step.

文档和代码片段在界面中同样属于不可信内容。以转义文本和代码方式展示，不执行来源 HTML／JavaScript，也不使用不安全 HTML 插值。来源链接必须为配置的上游来源主机上的合法 HTTPS URL，拒绝脚本／data 等协议及无效地址。这是渲染和链接校验规则，不增加审批步骤。

## Index and version semantics 索引与版本语义

For the first release, index a frozen snapshot with a recorded tokenizer, BM25 implementation/version and parameters. Keep identifier tokens such as `state_dict` useful for retrieval. Persist corpus and index hashes. Apply source/version filters before retrieval; an empty filtered corpus returns zero results. Rebuild a new snapshot when documentation changes instead of silently changing the index used in an experiment.

首版索引固定语料快照，记录分词、BM25 实现版本与参数，并保留 `state_dict` 等标识符的检索价值。保存语料和索引哈希。来源与版本筛选在召回前生效；筛选后无内容时返回空结果。文档更新时生成新快照，不静默修改已有实验索引。

V1 computes document frequencies, IDF and length statistics once over the complete frozen corpus. A request applies metadata masks before selecting top-k; it does not rebuild statistics over its filtered subset. A passage is eligible only if it passes the filters and shares at least one indexed lexical token with the processed query. Eligibility does not depend on the sign of its BM25 score: some BM25 variants permit zero/negative matching scores. Sort eligible passages by descending score, then ascending `chunk_id`; return fewer than k when necessary. No processed query tokens or no overlapping eligible passages yields an empty result, never arbitrary zero-scored top-k. Persist tokenizer rules and the BM25 variant, including identifier handling.

首版在整个固定语料上一次计算词频、IDF 与长度统计。每次查询先按元数据屏蔽不符合条件的段落，再取 top-k，不在筛选子集上重算统计。候选须通过筛选，且与处理后的问题至少共享一个已入索引的词项。资格与 BM25 分数正负无关，因为部分变体允许匹配段落得零分或负分。候选按分数降序、`chunk_id` 升序排列，不足 k 时按实际返回。处理后无词项或无符合条件的词项交集时返回空结果，不以任意零分段落补满。保存分词规则、BM25 变体和标识符处理方式。

**Implemented baseline definition (tokenizer `lexical-v1`, `devsearch.retrieval`).** Tokenization is Unicode NFKC, then *runs* of word characters that may continue across dots. Each run emits, case-folded and without repeats, its full token, each dot-separated component, and its parts split at `_`, camelCase, acronym and letter/digit boundaries, so `load_state_dict` yields `load_state_dict`, `load`, `state`, `dict` and `AutoTokenizer.from_pretrained` yields `autotokenizer.from_pretrained`, `autotokenizer`, `from_pretrained`, `auto`, `tokenizer`, `pretrained`. Tokens under two characters, tokens without a letter or digit, and a fixed English stopword list are dropped; there is no stemming (so `resume` does not match `resuming`; a known limitation to evaluate on real data). Query terms are unique and weighted equally. Scoring is Okapi BM25 with `idf = ln(1 + (N - n + 0.5) / (n + 0.5))` (`lucene`, always positive) or the classic `ln((N - n + 0.5) / (n + 0.5))` (`robertson`, which can be zero or negative and exists so eligibility-independent-of-sign is exercised), `k1` and `b` from `index.toml`, and scores rounded to 10 decimals before ordering by descending score then ascending `chunk_id`. The index manifest records the tokenizer name and a fingerprint of its rules *and* of its output on fixed probe strings; a loader refuses an index built under different rules, and `validate_manifest` additionally rebuilds the postings from the corpus text and compares them.

**Filter semantics.** `sources` limits results to those source IDs; `versions` maps a source ID to one revision and constrains only that source, so sources not named stay unconstrained. Both are applied to the chunk's document record before ranking. An unknown source or version, a repeated source, or a version for a source that `sources` excludes is an `invalid_request` error; a valid filter that merely has no lexical match returns an empty result. Filters never trigger a silent fallback to another version.

**已实现的基线定义（分词 `lexical-v1`，`devsearch.retrieval`）。** 先做 Unicode NFKC，再取可跨点号延续的单词字符*片段*。每个片段按大小写折叠且不重复地产出：完整词元、各点号分隔成分，以及按 `_`、驼峰、缩写和字母／数字边界切分的部分，例如 `load_state_dict` 产出 `load_state_dict`、`load`、`state`、`dict`，`AutoTokenizer.from_pretrained` 产出 `autotokenizer.from_pretrained`、`autotokenizer`、`from_pretrained`、`auto`、`tokenizer`、`pretrained`。丢弃少于两个字符的词元、不含字母或数字的词元以及固定英文停用词表；不做词干化（`resume` 不匹配 `resuming`，这是已知局限，须在真实数据上评估）。查询词去重且权重相同。评分为 Okapi BM25：`idf = ln(1 + (N - n + 0.5) / (n + 0.5))`（`lucene`，恒为正）或经典 `ln((N - n + 0.5) / (n + 0.5))`（`robertson`，可为零或负，用于检验资格与分数正负无关），`k1`、`b` 取自 `index.toml`，排序前分数舍入到 10 位小数，再按分数降序、`chunk_id` 升序。索引清单记录分词名称及其规则和固定探测字符串输出的指纹；加载时拒绝规则不同的索引，`validate_manifest` 还会用语料文本重建倒排并比较。

**筛选语义。** `sources` 将结果限定为这些来源 ID；`versions` 把来源 ID 映射到一个版本，只约束该来源，未列出的来源不受约束。两者在排序前按段落所属文档记录生效。未知来源或版本、重复来源、或对被 `sources` 排除的来源给出版本，均为 `invalid_request` 错误；合法筛选仅因无词项匹配则返回空结果。筛选绝不会静默回退到其他版本。

A first release with one pinned version per source supports exact snapshot selection, not a claim of comprehensive multi-version search. Near-duplicate pages across versions must remain grouped during any later train/test split.

首版每个来源一个固定版本，只证明精确快照筛选，不声称全面支持多版本检索。后续加入多版本时，近重复页面在训练与测试划分中仍需分组处理。

## Search API 搜索接口

Use a local FastAPI service bound to localhost by default. These are interface contracts, not runnable code. UI choices remain replaceable; a small Streamlit comparison interface is the default implementation option.

默认使用绑定 localhost 的 FastAPI 本地服务。以下是接口契约，不是可运行代码。界面保持可替换，初始可用小型 Streamlit 页面展示排序对比。

For a remote Mac, keep API and UI bound to that host's localhost. The proposed defaults are API port 8000 and UI port 8501, configurable after availability checks. Forward only the UI port through the existing SSH access; the remote UI calls its co-located API. This is private access, not public hosting, and requires no new authentication service. Follow the handoff specification for connection and backup checks.

远程 Mac 的 API 与 UI 都绑定该主机 localhost，拟定默认端口为 8000 和 8501，检查可用性后可配置。通过已有 SSH 接入仅转发 UI 端口，远端 UI 调用同机 API。这是私有访问，不是公开托管，不增加认证服务；连接和备份验收遵循交接规格。

Run one API process and one bounded, process-local search executor; the executor owns model invocation, including on CPU, so synchronous retrieval/scoring does not block the event loop or health checks. V1 admits one active search and no waiting search queue; excess requests receive a busy response. Acquire admission before submitting executor work. Cancellation/disconnection or an HTTP timeout does not free its slot until the worker and already-submitted device work finish. Check a monotonic deadline before retrieval and between scoring batches; stop submitting further batches after expiry. No promise is made to cancel a running device kernel. Training and serving remain mutually exclusive on the accelerator, as specified in the architecture.

运行一个 API 进程及一个有界的进程内搜索执行器；模型调用由执行器统一负责，CPU 路径也如此，避免同步检索／评分阻塞事件循环和健康检查。首版接纳一个活动搜索，不设等待搜索队列；超量请求返回忙碌。提交执行器任务前取得接入许可。客户端取消／断开或 HTTP 超时，都不得在工作线程及已提交设备任务结束前释放许可。检索前及评分 batch 之间检查单调时钟截止时间，过期后不再提交后续 batch，不承诺中断正在执行的设备 kernel。训练和服务仍按架构要求错时使用加速器。

Reranking is all-or-nothing: only after every candidate has one finite score can its ranking be published. Timeout, nonfinite scores or model exceptions discard all partial reranking output. If a complete baseline result already exists, return the explicit degraded result; otherwise return the applicable error below. A timed-out HTTP response may precede worker cleanup, but the occupied slot remains unavailable until cleanup completes. Record observed elapsed time rather than claiming a hard wall-clock bound on accelerator work. A model exception or nonfinite output marks rerank readiness false until an explicit successful self-check or restart; no request reloads the model automatically.

重排采用全部完成或全部放弃：每个候选都得到一个有限分数后才能发布排序。超时、非有限分数或模型异常均丢弃所有部分重排结果；已有完整基线时明确降级，否则返回下述错误。HTTP 超时响应可早于工作线程清理结束，但接入许可持续占用到清理完成。记录真实耗时，不宣称设备计算存在严格墙钟上限。模型异常或非有限输出将重排就绪状态置为 false，直到显式自检成功或重启；请求不自动重新加载模型。

`POST /search` request:

```json
{
  "query": "How do I resume training from a checkpoint?",
  "sources": ["pytorch-tutorials"],
  "versions": {"pytorch-tutorials": "<pinned-revision>"},
  "candidate_k": 50,
  "top_k": 5,
  "mode": "compare"
}
```

Supported modes are `baseline`, `rerank` and `compare`. Validate a nonempty query, `1 <= top_k <= candidate_k <= 100`, and source/version membership. Reject unsupported modes and an overlong query with a readable error instead of silently changing meaning. Training and serving must share pair tokenization and length handling. Limits become configurable but are recorded in each benchmark.

模式限定为 `baseline`、`rerank`、`compare`。检查问题非空、`1 <= top_k <= candidate_k <= 100`、来源版本合法；未知模式和过长问题返回可读错误，不静默改写含义。训练与服务复用问题—段落编码和长度处理方式。限制可配置，但必须记入每次评测。

Response contract:

```text
request_id, release_id, status, mode_requested, mode_effective
corpus_id, index_id, model_id, candidate_set_id
baseline_results[], reranked_results[]
timings_ms: retrieval, rerank, total
warnings[]
```

Each result contains `chunk_id`, `source_id`, `source_url`, `revision`, `title`, `section`, `text`, `code_fragment`, `source_location`, `baseline_rank`, `reranked_rank`, `bm25_score` and `rerank_logit`. `section` is derived from the chunk's heading path; `source_location` resolves the saved source map to the original file/lines or notebook cell. `text` is exactly the passage scored, without silently appended context; the UI labels excerpts using `code_fragment`. Arrays may be shorter than requested. A logit is an uncalibrated relevance score, not a probability of correctness, and is not numerically comparable to a BM25 score.

每条结果包含上述来源、文本、版本、片段标记、原文位置、排名和分数。`section` 来自段落标题路径，`source_location` 通过保存的映射定位原文件行或 Notebook 单元。`text` 必须等于模型实际评分段落，不静默补入其他上下文；界面按 `code_fragment` 标记摘录。实际数量可少于请求数量。logit 是未校准相关性分数，不应展示为正确概率，也不能与 BM25 分数直接比较。

Baseline responses put usable results in `baseline_results`, with `reranked_results: []`, `reranked_rank: null` and `rerank_logit: null`. `model_id` is null when no model is loaded; a fallback may retain the attempted model ID for diagnosis. Do not copy BM25 ranks into fields that imply successful reranking. `status` is `ok` for requested baseline, `degraded` for explicit fallback, and `empty` for a valid empty result.

基线模式把有效结果放在 `baseline_results`，`reranked_results` 为空，重排排名与 logit 为 null。无模型加载时 `model_id` 为 null；降级时可保留尝试调用的模型 ID 用于诊断。不能把 BM25 排名复制为看似成功的重排结果。主动基线模式的状态为 `ok`，明确降级为 `degraded`，合法空结果为 `empty`。

**Baseline CLI response (implemented).** `devsearch.retrieve` prints a subset of this contract: `status`, `mode_requested`, `mode_effective`, `corpus_id`, `index_id`, `model_id` (null), `candidate_set_id`, the echoed `query` and `filters`, `candidate_k`, the actual `candidate_count`, `top_k`, `baseline_results` (the full result contract; `reranked_rank` and `rerank_logit` are null), `reranked_results` (empty), `timings_ms` (`rerank` null), `warnings`, a `schema_version` and the search config's `config_sha256`. It has no `request_id` or `release_id` yet; those arrive with the API (T09). Text is returned exactly as indexed and unescaped; escaping is the renderer's job. Only `mode: baseline` is accepted; `rerank` and `compare` are rejected (CLI exit status 2) until a reranker exists.

**基线 CLI 响应（已实现）。** `devsearch.retrieve` 输出本契约的子集：`status`、`mode_requested`、`mode_effective`、`corpus_id`、`index_id`、`model_id`（null）、`candidate_set_id`、回显的 `query` 与 `filters`、`candidate_k`、实际 `candidate_count`、`top_k`、`baseline_results`（完整结果契约；`reranked_rank` 与 `rerank_logit` 为 null）、`reranked_results`（空）、`timings_ms`（`rerank` 为 null）、`warnings`，以及 `schema_version` 和搜索配置的 `config_sha256`。目前没有 `request_id` 与 `release_id`，随 API（T09）加入。文本按入库内容原样返回、不转义，转义由渲染方负责。仅接受 `mode: baseline`；在重排器存在之前，`rerank` 与 `compare` 被拒绝（CLI 退出状态 2）。

Illustrative response fields; result entries use the full result contract above:

以下为响应示意，结果条目使用上述完整结果契约：

```json
{
  "request_id": "<request>",
  "release_id": "<baseline-release>",
  "status": "ok",
  "mode_requested": "baseline",
  "mode_effective": "baseline",
  "model_id": null,
  "baseline_results": [{"chunk_id": "<chunk>", "baseline_rank": 1, "reranked_rank": null, "rerank_logit": null}],
  "reranked_results": [],
  "warnings": []
}
```

```json
{
  "request_id": "<request>",
  "release_id": "<evaluated-release>",
  "status": "degraded",
  "mode_requested": "rerank",
  "mode_effective": "baseline",
  "model_id": "<attempted-model>",
  "baseline_results": [{"chunk_id": "<chunk>", "baseline_rank": 1, "reranked_rank": null, "rerank_logit": null}],
  "reranked_results": [],
  "warnings": [{"code": "rerank_failed", "message": "Reranking failed; showing BM25 results."}]
}
```

`GET /health` is a liveness response (200 while the process can answer), reporting `alive`, `baseline_ready`, `rerank_ready`, `busy`, release status and manifest identities, without secrets. `GET /ready?mode=baseline|rerank` returns 200 only when that requested capability has passed artifact checks and its startup smoke test, otherwise 503. Temporary admission occupancy is reported as busy, not a failed startup. Activation checks the intended default mode explicitly; a baseline-ready process does not prove reranking works. Search does not download or train missing assets, and local services do not upload queries by default.

`GET /health` 用于存活检查，进程可回答时返回 200，并提供 `alive`、基线／重排就绪、忙碌状态、发布状态及清单身份，不暴露秘密。`GET /ready?mode=baseline|rerank` 仅在指定能力通过产物检查和启动试跑后返回 200，否则 503。临时许可占用显示忙碌，不等同启动失败。激活时明确检查预期默认模式；基线就绪不能证明重排可用。搜索不下载或训练缺失资源，默认不上传查询。

## Failure behavior 失败行为

| Condition 条件 | Response 响应 |
| --- | --- |
| Empty/overlong query, invalid filter or mode / 空问题、过长问题、非法筛选或模式 | HTTP 422, `invalid_request` / 参数错误 |
| Valid filters with no lexical candidates / 合法筛选后无词项匹配候选 | HTTP 200, `status: empty`, empty arrays / 空结果 |
| Reranker absent at runtime, failed, timed out or nonfinite; complete baseline available / 运行时重排不可用、失败、超时或非有限，已有完整基线 | HTTP 200, `status: degraded`, explicit reason, no partial reranking / 明确基线降级，不保留部分重排 |
| Capacity exhausted / 接入容量耗尽 | HTTP 429, `busy`, no unbounded queue / 忙碌，不建立无界队列 |
| Deadline exceeded before a complete baseline / 完整基线生成前超时 | HTTP 504, `deadline_exceeded` / 截止超时 |
| Index unavailable or invalid / 索引不可用或无效 | HTTP 503, `index_unavailable`; no fabricated results / 不可用，不编造结果 |
| Unexpected nonrecoverable pipeline error / 其他不可恢复流程错误 | HTTP 500, `internal_error`; local detailed log / 通用错误，详情仅记本地 |
| Low scores or irrelevant content / 分数低或内容无关 | Show ranked evidence with limitation; no calibrated no-answer claim / 展示证据与局限，不凭原始分数声称确定无答案 |
| Source changed upstream / 上游内容变化 | Link the pinned revision; schedule an explicit rebuild as later work / 链接固定版本，新版本需显式重建 |

Error bodies contain `request_id`, `error.code`, a readable `error.message` and `retryable`; they do not mimic successful search arrays or expose stack traces. Degradation warnings include a stable reason code, such as `rerank_timeout`, alongside readable text. Declared baseline-only releases legitimately omit model artifacts; required artifact corruption blocks activation rather than silently converting a damaged reranking release into a baseline release. During `compare`, both displayed lists come from the same candidate set; a failed reranker leaves only the correctly labeled baseline list.

错误体包含请求 ID、稳定错误码、可读消息及 `retryable`，不伪装成功结果数组，也不暴露堆栈。降级警告除文字外还含 `rerank_timeout` 等稳定原因码。明确基线版本可以没有模型，但必需产物损坏时拒绝激活，不能把损坏的重排版本悄悄当基线启动。对比模式的两组结果来自同一候选集；重排失败时只展示正确标记的基线。

Response caching is disabled in v1; the following keys are requirements if caching is added later. In compare mode both lists originate from exactly the same retrieved candidates. Candidate ranking uses deterministic tie-breaking by chunk ID. Full-response cache keys include release/corpus/index/model/configuration identities and all normalized output-affecting request fields, including query, filters, `candidate_k`, `top_k` and `mode`. If caching in layers, retrieval keys include `candidate_k`; reranking keys include the ordered candidate-ID/text-hash list and model/tokenizer configuration; display settings are applied at the response layer. Benchmark reports state whether caches were enabled, and report degraded requests separately.

首版关闭响应缓存，下述缓存键为后续启用时的要求。对比模式的两组结果必须来自同一候选集，同分按段落 ID 确定排序。完整响应缓存键包括发布版本、语料、索引、模型和配置身份，以及所有影响输出的规范化请求字段：问题、筛选、`candidate_k`、`top_k`、`mode`。分层缓存时，召回键包含候选数，重排键包含有序候选 ID／文本哈希及模型编码配置，展示参数在响应层处理。性能报告说明缓存状态，并单独记录降级请求。

## Integration acceptance 集成验收

- Verify source/version filters before ranking and links back to immutable source revisions.
  验证召回前的来源版本筛选，以及指向固定版本原文的链接。
- Preserve code formatting and source attribution without executing source code.
  保留代码格式和署名，不执行资料中的代码。
- Evaluate baseline and reranker on identical candidate IDs; do not inject known answers.
  基线与重排序使用同一候选 ID，不插入已知答案。
- Check train/serve tokenization parity, finite scores, model-manifest compatibility and deterministic sorting.
  检查训练服务编码一致、分数有限、模型清单兼容与排序确定性。
- Verify checkpoint recovery, invalid-input handling, empty results and explicit fallback.
  验证断点恢复、非法输入、空结果与明确降级。
- Verify no-overlap empty retrieval even when the BM25 implementation returns all-zero scores; include lexical matches with nonpositive scores, global-IDF filter fixtures and deterministic ties.
  验证 BM25 返回全零分时，无词项交集仍为空；覆盖词项匹配但分数非正、全局 IDF 筛选及同分排序样例。
- Inject scorer timeout, nonfinite output, exception and cancellation; assert that no partial ranking leaks, occupied work retains admission, and health remains responsive. Check baseline/rerank readiness separately.
  注入评分超时、非有限输出、异常及取消，确认不泄漏部分排序、未结束任务持续占用许可、健康检查仍响应，并分别检查基线与重排就绪。
- Time retrieval and reranking separately after warmup; synchronize accelerator operations. Report device, query count, length distribution, batch size and concurrency. A single-user local result is not a production SLA.
  预热后分别测量召回和重排，并同步加速器操作。记录设备、问题数、长度分布、batch 与并发；本地单用户结果不等于生产 SLA。

## Collaboration and release 协作与交付

The assistant owns code, project transfer, setup and experiments on the user-provided Mac, result analysis and documentation. The user arranges machine access, reviews a focused set of ambiguous labels and independently demonstrates at least one training modification and failure analysis before using personal-skill claims. Record actual contributions, including AI assistance, without converting generated code into an unsupported personal-skill claim.

助手负责代码、项目迁移、用户所提供 Mac 上的环境与实验、结果分析及文档；用户安排机器访问，集中审查少量有歧义标签，并在陈述个人能力前独立完成至少一项训练修改与失败分析。按事实记录贡献及 AI 辅助情况，不把生成代码自动等同于个人能力。

Before publishing a repository, exclude datasets, caches, credentials and checkpoints by default; include scripts, source manifests, lockfiles, experiment configuration, result summaries and attribution. Whether to redistribute specific data or weights is a separate license and publication decision. This documentation request does not authorize a public upload.

发布仓库前，默认排除数据集、缓存、凭据与模型权重，保留脚本、来源清单、依赖锁、实验配置、结果摘要和署名。具体数据或权重的再分发需单独核对许可和发布决定。本次整理文档不授权公开上传。

## References 参考

- [PyTorch Tutorials](https://github.com/pytorch/tutorials): proposed application corpus / 拟用应用资料来源。
- [Transformers documentation](https://github.com/huggingface/transformers/tree/main/docs): proposed application corpus / 拟用应用资料来源。
- [Natural Questions pairs](https://huggingface.co/datasets/sentence-transformers/natural-questions): general training source; not a technical-domain benchmark / 通用训练来源，不等于技术领域评测。
- [MiniLM model card](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased): initial encoder candidate / 初始编码器候选。

References were checked on 2026-10-05. Resolve immutable revisions and dependency versions during M0; links to default branches are discovery references only.

参考链接于 2026-10-05 核对。M0 阶段解析不可变版本与依赖版本，默认分支链接仅用于定位来源。
