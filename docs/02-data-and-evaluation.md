# DevSearch: Data and Evaluation Specification / DevSearch：数据与评测开发规格

Status: proposed design; no dataset, training result, or user study is claimed complete. Version 1 uses English questions and English documents, as confirmed by the user. MiniLM is the initial model proposal; NQ is an optional diagnostic and transfer-data track. This document defines requirements, not measured achievements.

状态：开发方案，尚未完成数据集、训练或用户研究。用户已确认第一版采用英文问题和英文文档；MiniLM 作为起始模型，NQ 属于可选排错与迁移数据支线。本文规定实现要求，不代表已经取得的成果。

## 1. Two separate evaluation tracks / 1. 两条独立实验线

**General training track:** use the [Sentence Transformers Natural Questions dataset](https://huggingface.co/datasets/sentence-transformers/natural-questions), which exposes a 100k-row train split with query/answer pairs. Use it for training-pipeline development and general reranking experiments. Its custom subsets are not the official NQ benchmark.

**通用训练线：**使用上述 NQ 整理版，其页面提供约 10 万条 query/answer 训练数据。用于训练流程开发和通用重排序实验；自定义子集成绩不属于 NQ 官方成绩。

**Technical-document track:** index pinned versions of [PyTorch Tutorials](https://github.com/pytorch/tutorials) and [Transformers documentation](https://github.com/huggingface/transformers/tree/main/docs/source/en). General NQ gains do not establish gains on these documents. Record each repository commit, selected paths, original URL, acquisition time, and applicable license. Exact revisions remain to be selected; do not use a moving `main` in released experiments.

**技术文档线：**索引固定版本的 PyTorch Tutorials 与 Transformers 文档。NQ 上的提升不能证明技术文档检索提升。记录仓库提交、选取路径、原始链接、获取时间和适用许可；具体版本待选，正式实验不得依赖不断变化的 `main`。

## 2. Data contracts and traceability / 2. 数据格式与溯源

Store UTF-8 JSONL. IDs must be stable; changing source text or chunking creates a new version. Preserve source offsets so every relevant chunk can be checked against its evidence.

使用 UTF-8 JSONL，ID 应稳定；原文或切块策略改变时生成新版本。保留原文位置，使相关段落能够回溯至具体证据。

```json
{"doc_id":"pytorch:tutorials/path","source_id":"pytorch-tutorials","title":"<title>","revision":"<commit>","source_url":"<immutable-url>","license":"<verified>","content_sha256":"<normalized-text-hash>","normalized_text_path":"<artifact-relative-path>","source_map_path":"<artifact-relative-path>","parser_version":"v1","parse_status":"accepted"}
{"chunk_id":"<doc+revision+text+offsets+config hash>","doc_id":"pytorch:tutorials/path","revision":"<commit>","start_char":100,"end_char":600,"heading_path":["Training"],"text":"...","chunker_version":"v1","evidence_group_id":"eg001","owner_split":"train","code_fragment":false}
{"query_id":"q001","text":"...","language":"en","group_id":"g001","evidence_group_ids":["eg001"],"parent_query_id":null,"task_family":"checkpoint-recovery","split":"train","filters":{"sources":["pytorch-tutorials"],"versions":{"pytorch-tutorials":"<commit>"}},"origin":"synthetic","generator":{"model":"<id>","prompt_sha256":"<hash>","seed":null},"review_status":"unreviewed","reviews":[]}
{"query_id":"q001","chunk_id":"<id>","relevance":1,"evidence_spans":[{"doc_id":"pytorch:tutorials/path","revision":"<commit>","content_sha256":"<normalized-text-hash>","start_char":250,"end_char":340}],"label_source":"synthetic","review_status":"unreviewed","reviews":[]}
{"query_id":"q001","positive_chunk_id":"<id>","negative_chunk_id":"<id2>","negative_method":"bm25","label_status":"assumed_negative","seed":42}
{"pair_id":"<query+chunk+label-version-hash>","query_id":"q001","chunk_id":"<id>","label":1,"label_status":"reviewed_positive","sampling_manifest_id":"<id>"}
```

Separate relevance judgments from training pairs. A missing judgment is `unjudged`, not a verified negative. Store experiment manifests with dataset hashes, split IDs, model revision, tokenizer configuration, candidate files, and code identity (clean commit or preserved source-snapshot hash). Never put test judgments in training inputs.

相关性标注与训练样本分开保存。缺少标注表示 `unjudged`，不等于已确认不相关。实验清单保存数据哈希、划分 ID、模型版本、分词参数、候选文件和代码身份（干净提交或已保留源码快照哈希）；测试标注不能进入训练输入。

Source-document identity is `(doc_id, revision)`; the immutable corpus manifest also identifies the normalized-text hash and parser version. Chunk offsets use `[start_char, end_char)` in that saved normalized text. Each source-map entry links a normalized span to the original path and line range, or notebook cell ID and cell-local range; generated heading text is marked synthetic. A parser change creates a new normalized artifact, never an overwrite. Resolve `source_id`, `title`, and `source_url` through the document record. For NQ rows without an original article URL, store the dataset revision and row identity with `source_url: null`; never invent source links. Relevance values in v1 are binary 0/1, with missing judgments kept separate.

原文身份为 `(doc_id, revision)`，不可变语料清单另记录规范化正文哈希和解析器版本。段落位置采用该已保存正文中的 `[start_char, end_char)`。映射项将正文范围关联至原文件路径和行范围，或 Notebook 单元 ID 及单元内位置；生成的标题文字标为合成。解析器变更生成新正文产物，不覆盖旧内容。通过文档记录取得来源、标题与链接。NQ 若缺少原文章 URL，则保存数据集版本与行身份，`source_url` 为 null，不编造链接。首版相关性值为二元 0/1，缺失标注单独表示。

Candidate records persist `query_id`, query/filter hashes, corpus/index IDs and hashes, retrieval-config hash, requested and actual candidate counts, and the ordered list of `chunk_id`, text hash, BM25 score and rank. Record candidate depth separately from displayed result count. Evaluation reuses these records rather than regenerating a potentially different list.

候选记录保存 `query_id`、问题与筛选哈希、语料索引 ID 及哈希、召回配置哈希、请求和实际候选数，以及有序的段落 ID、文本哈希、BM25 分数和排名。候选深度与展示数量分别记录；评测复用该文件，不临时重新生成可能不同的候选列表。

### Corpus artifact v1 (implemented: loader, validator and writer) / 语料产物 v1（已实现：加载、校验与写入）

A corpus is one immutable directory, loaded only through its `corpus-manifest.json` (`kind: corpus`, `schema_version: 1`). Every reference is a relative POSIX path that must stay inside that directory (no absolute paths, backslashes, `..` escapes or symlinks that leave it). The manifest lists `documents.jsonl` and `chunks.jsonl` with SHA-256 and counts, and a sorted `files` inventory (every normalized-text and source-map file with SHA-256; no unreferenced files). `corpus_id` is `corpus-` plus 32 hex characters derived from those hashes; `manifest_sha256` covers the canonical manifest without itself. Loading fails on unknown or missing fields, duplicate or unsorted records, dangling `(doc_id, revision)` references, any hash mismatch, chunk text that differs from `normalized_text[start_char:end_char]`, a `chunk_id` that is not re-derived exactly, a chunk with no non-synthetic source location, a `source_url` that is not credential-free HTTPS (or null), or `parse_status` other than `accepted`. JSONL files are UTF-8 with LF line endings, no blank lines, no duplicate keys and no NaN/Infinity. Documents are sorted by `(doc_id, revision)` and chunks by `(doc_id, revision, start_char, chunk_id)`.

语料是一个不可变目录，只通过其 `corpus-manifest.json`（`kind: corpus`、`schema_version: 1`）加载。所有引用均为相对 POSIX 路径，且必须位于该目录内（不允许绝对路径、反斜杠、`..` 越界或指向目录外的符号链接）。清单列出 `documents.jsonl` 与 `chunks.jsonl` 的 SHA-256 和数量，以及按路径排序的 `files` 清单（每个规范正文和来源映射文件及其 SHA-256，不得有未引用文件）。`corpus_id` 为 `corpus-` 加由这些哈希导出的 32 位十六进制；`manifest_sha256` 覆盖去掉自身后的规范清单。以下情况加载失败：未知或缺失字段、重复或未排序记录、`(doc_id, revision)` 引用失效、任何哈希不符、段落文本与 `normalized_text[start_char:end_char]` 不一致、`chunk_id` 无法精确重新导出、段落没有非合成的来源位置、`source_url` 不是不含凭据的 HTTPS（或 null）、`parse_status` 不是 `accepted`。JSONL 为 UTF-8、LF 换行，无空行、重复键、NaN／Infinity。文档按 `(doc_id, revision)` 排序，段落按 `(doc_id, revision, start_char, chunk_id)` 排序。

**Ownership and publication rules (review fixes F01/F02).** Every chunk of one `evidence_group_id` must carry the same `owner_split` — all `null` while groups are still being prepared, or one identical assigned value; a group that mixes `null` with an assigned owner, or spans two assigned owners, is rejected at load, before any pair mining, because one piece of evidence must never feed both training and held-out evaluation. A corpus is written by staging every file in a private `.staging-*` subdirectory, validating it completely, and only then publishing: data files first, the manifest last by an atomic rename. Invalid input therefore never leaves a manifest, an interrupted write leaves only unpublished remnants (set aside, never deleted, by the next attempt), and a retry is safe. Re-writing identical content is idempotent; different content in a directory that already holds a corpus is refused.

**所有权与发布规则（评审修正 F01／F02）。** 同一 `evidence_group_id` 的全部段落须有相同的 `owner_split`：准备阶段全为 `null`，或全为同一个已分配值；若同组混用 `null` 与已分配归属、或跨两个已分配归属，则在加载时（任何负例挖掘之前）拒绝，因为同一证据绝不能同时用于训练和留出评测。写入语料时先把全部文件放入私有的 `.staging-*` 子目录并完整校验，再发布：先移入数据文件，最后以原子重命名发布清单。因此非法输入不会留下清单，中断的写入只留下未发布的残留（下次尝试将其移开而不删除），重试是安全的。重复写入相同内容是幂等的；目录中已有不同内容的语料则拒绝。

`content_sha256` is the SHA-256 of the normalized text file's exact UTF-8 bytes; offsets count Unicode code points. `chunk_id` is `ch_` plus the first 32 hex characters of the SHA-256 of the canonical JSON of `{doc_id, revision, start_char, end_char, text_sha256, chunker_version}`, so identical text at different offsets, revisions or chunker versions gets distinct IDs and `owner_split` assignments never change an ID.

`content_sha256` 为规范正文文件精确 UTF-8 字节的 SHA-256；偏移按 Unicode 码点计数。`chunk_id` 为 `ch_` 加 `{doc_id, revision, start_char, end_char, text_sha256, chunker_version}` 规范 JSON 的 SHA-256 前 32 位十六进制；因此相同文本在不同位置、版本或切块版本下 ID 不同，`owner_split` 的分配也不会改变 ID。

**Source map v1** (`maps/NNNN.json`: `schema_version`, `doc_id`, `revision`, sorted non-overlapping `entries`). A *line* entry has `start_char`, `end_char`, `path`, `start_line`, `end_line` (1-based, inclusive), `synthetic`; a lookup reports the whole entry's lines, so producers emit one entry per source line or block. A *cell* entry has `path`, `cell_id`, `cell_start_char`, `cell_end_char` mapping the span one-to-one (equal lengths), so partial overlaps resolve exactly. `synthetic: true` marks generated text (for example an added title); it is never reported as a source location. `source_location` in results is a list of merged ranges, `{path, start_line, end_line}` or `{path, cell_id, start_char, end_char}`.

**来源映射 v1**（`maps/NNNN.json`：`schema_version`、`doc_id`、`revision` 及排序且不重叠的 `entries`）。*行*条目含 `start_char`、`end_char`、`path`、`start_line`、`end_line`（从 1 起、含端点）和 `synthetic`；查询返回整条目的行范围，故生产者应每行或每块一个条目。*单元*条目含 `path`、`cell_id`、`cell_start_char`、`cell_end_char`，一一对应映射（长度相等），因此部分重叠也能精确解析。`synthetic: true` 标记生成文字（如补充的标题），不作为来源位置返回。结果中的 `source_location` 是合并后的范围列表，形如 `{path, start_line, end_line}` 或 `{path, cell_id, start_char, end_char}`。

The baseline `retrieve` response carries a `candidate_set_id` (`cands-` plus 32 hex): a hash of `index_id`, the query hash, filters, `candidate_k` and the ordered `(chunk_id, text hash, BM25 score, rank)` list. It identifies the exact candidate list for later reuse and parity checks; the persisted per-query candidate file described above is still to be written by the evaluation task.

基线 `retrieve` 响应带有 `candidate_set_id`（`cands-` 加 32 位十六进制）：由 `index_id`、问题哈希、筛选、`candidate_k` 和有序 `(chunk_id, 文本哈希, BM25 分数, 排名)` 列表求哈希，用于标识确切候选列表，便于后续复用及一致性检查；上文所述按问题持久化的候选文件仍待评测任务写入。

## 3. Construction, chunking, and labeling / 3. 数据构造、切块与标注

Parse headings, explanatory text, and code blocks while removing navigation and repeated boilerplate. Chunk by document structure, then validate the actual paired-token length with the chosen tokenizer. Record exclusions and truncation rates. Do not silently truncate relevance evidence.

保留标题、说明文字和代码块，去除导航及重复模板。先按文档结构切块，再用实际 tokenizer 检查“问题＋段落”的长度，记录排除数量及截断比例；不能悄悄截掉相关性证据。

Inventory source formats before ingestion. Support only tested static extraction paths for the selected Markdown/MDX, reStructuredText, notebook JSON, and tutorial Python sources; never execute code or documentation directives. Preserve raw snapshots. Unsupported syntax or failed extraction is quarantined with a reason and source link; raw fallback is not silently indexed as clean text. A checked plain-text fallback may enter a new corpus with its extraction mode recorded.

采集前清点文件格式；选取的 Markdown／MDX、reStructuredText、Notebook JSON 和教程 Python 文件只使用经过验证的静态提取路径，不执行代码或文档指令。保留原始快照。未知语法或解析失败时隔离记录原因与来源，不把原始回退内容悄悄当作干净正文入库；经检查的纯文本回退可进入新语料版本，并记录提取方式。

To keep train/serve behavior identical, reserve `max_query_tokens` and special-token space when building the passage corpus. Every v1 passage must fit the remaining `max_pair_tokens` budget; each query is checked against its reserve. Exclude or explicitly rechunk invalid records before labels are finalized. Serving and training do not silently truncate these validated pairs.

为保证训练与服务一致，构建段落库时预留 `max_query_tokens` 和特殊 token 空间，首版每个段落须能放入剩余 `max_pair_tokens` 预算；问题也检查预留上限。标签定稿前排除或显式重新切分不合格记录，训练和服务均不静默截断已验证样本。

Keep a code block intact when it fits. An overlong block may be split at recorded line or structural boundaries only when the resulting excerpt retains the labeled evidence; preserve indentation and mark `code_fragment: true` with a complete-source link. Do not imply that a fragment is independently executable. Otherwise exclude the block from model input and record the coverage loss. Questions requiring omitted evidence or several jointly necessary chunks are outside v1's single-passage-positive evaluation; count and disclose them before sealing, rather than converting fragments into false positives.

代码块能放入预算时保持完整。超长块仅在切后仍保留标注证据时，按记录的行或结构边界切成片段，保留缩进、标记 `code_fragment: true` 并链接完整原文，不能声称片段可独立执行。否则将该块排除在模型输入之外并记录覆盖损失。依赖被排除证据或多个必须联合阅读段落的问题，不属于首版单段正例评测；封存前统计披露，不把残缺片段误标为正例。

Long NQ answers cannot pass their positive label to every chunk or truncated prefix. For the initial NQ experiment, retain examples that fit completely after tokenization; disclose the resulting length bias. A later long-text dataset requires evidence localization before assigning chunk labels.

NQ 长答案不能把正标签传给所有切块或截断前缀。首版优先保留分词后可完整编码的样本，并披露长度筛选偏差；以后处理长文时，必须先定位证据再标注切块。

Domain targets are **500–1,000 training questions**, a separate validation set, and **200–300 sealed test questions**; these are targets, not existing assets. Generate each question from an identifiable evidence span. Verify answerability, technical-version consistency, and preservation of evidence after chunking. Store generator provenance and review status; reject unsupported or ambiguous items.

领域数据目标为 **500–1,000 条训练问题**、独立验证集和 **200–300 条封存测试问题**，目前均未完成。每个问题必须对应可定位证据，并检查可回答性、技术版本一致性及切块后证据是否完整。记录生成来源和审查状态，排除无依据或歧义样本。

Plan coverage by source and the three declared task families before generation. Report raw generated questions, accepted questions, unique paraphrase/evidence groups, and counts by source/family/split; do not fill a quota with repeated paraphrases. Preserve the parent question or generation batch and evidence group. If independent groups are insufficient, reduce the target and report the limitation. Synthetic evaluation measures this constructed question distribution, not real developer traffic.

生成前按来源和既定三类任务规划覆盖。分别报告原始生成数、接受问题数、独立改写／证据组数，以及各来源、任务和划分的数量，不靠同义改写凑数。保留父问题或生成批次和证据组；独立组不足就缩小目标并说明。合成评测衡量的是构造的问题分布，不等同于真实开发者流量。

Synthetic questions remain synthetic after review. Store review status separately from reviewer type. Each `reviews[]` record contains `reviewer_type: agent|human`, `reviewer_id`, `status`, `reviewed_at`, and an agent model/workflow revision when applicable. Empty review history means unreviewed. Agent review must not be labeled human review. Participant feedback belongs to a separate study record; no such study has occurred.

合成问题经审查后仍属于合成数据。审查状态与审查者类型分开记录；每条 `reviews[]` 包含 `reviewer_type: agent|human`、身份、结论、时间，以及适用的 Agent 模型或流程版本。空历史表示未审查，Agent 检查不能标为人工审查。参与者反馈另存研究记录，目前没有这样的研究。

Review initial test positives before the first seal; review the final pooled judgments under the second-seal procedure below. Audit a sample of training negatives separately. Record who or what reviewed each item, resolve uncertainty under fixed rules, and disclose agent-only review when that is the evidence available. Human agreement can be measured on a separate audited subset if actually performed, without inventing a participant count.

第一次封存前审查初始测试正例，最终汇总候选按下文第二次封存流程审查；训练负例另行抽查。记录审查来源并按固定规则处理不确定项；如果只有 Agent 审查，就披露这一局限。若实际进行了人工抽查，可另报抽查子集的一致情况，不编造参与人数。

## 4. Splits, corpus boundaries, and negatives / 4. 划分、语料边界与负样本

Assign domain evidence groups to development, training, validation or sealed test before generating their questions. Group duplicate/near-duplicate passages, overlapping answer-bearing excerpts and their question paraphrases together; preserve original-document identity without claiming whole-document separation. Persist normalization rules, grouping method/thresholds, seed and assignments. Audit cross-split question and evidence overlap before formal comparisons. A later cross-split duplicate triggers merging/repartitioning and new affected manifests before training; after test exposure, affected claims become exploratory or need a new independent test.

先把领域证据组分配至开发、训练、验证或封存测试，再生成对应问题。重复／近重复段落、相互重叠的回答证据及同题改写归为一组；保留原文身份，但不声称整篇文档隔离。保存规范化规则、分组方法／阈值、种子和分配记录，正式对照前检查跨集合问题与证据重叠。后续发现跨集合重复时，在训练前合并重划并更新受影响清单；若已接触测试，相关结论只能作为探索结果或另建独立测试。

For NQ, pin the dataset revision and row IDs; normalize and deduplicate answer texts, then join linked query/paraphrase and answer groups before assigning splits. Treat each retained full answer as a passage; its owner is its group split. Its source mapping points to the dataset row and answer field, not invented original-article line numbers. Text-only grouping cannot establish complete separation of original Wikipedia articles. NQ splits, mining pools, candidates and metrics remain separate from domain artifacts.

NQ 固定数据版本和行 ID，规范化并去重答案正文，再合并有关联的问题／改写和答案组后划分。每个保留的完整答案视为段落，归属其组所在集合；来源映射指向数据集行与 answer 字段，不编造原文章行号。仅靠文本分组不能证明维基原文章严格隔离；NQ 的划分、挖掘池、候选和指标与领域产物分开管理。

Primary domain evaluation measures **unseen questions against a fixed shared document index**. Corpus visibility is not permission to train on test queries or labels. A separate unseen-document experiment must hold out source-document groups from all training pairs and mining; report it separately. Avoid claims about document generalization from the primary track.

领域主评测测量**固定共享文档索引上的新问题检索**。索引可见不代表可以使用测试问题或标签训练。若测试新文档泛化，须另行把原文组排除在全部训练样本及挖掘范围之外，并单独报告；主实验不能代替此结论。

Mine random and BM25 hard negatives using training queries and only chunks whose evidence groups have `owner_split: train`. Persist the exact mining-pool IDs/hash; unassigned, development, validation and test groups are not eligible. The serving/evaluation index may contain all corpus groups; maintain a separate train-owned mining view. Exclude positives known from training judgments and their duplicates. Do not read test queries or qrels to build a negative mask. Unknown candidates remain assumed negatives and require auditing. If a query lacks enough eligible negatives, record the shortage and use fewer or exclude it by a preset rule; never silently sample held-out groups. Controlled sampling comparisons use the same eligible questions, pair counts and training steps.

随机与 BM25 困难负例只使用训练问题及 `owner_split: train` 的证据组段落，并保存挖掘池 ID 与哈希；未分配、开发、验证和测试组均不可进入。服务／评测索引可包含全部语料组，训练另设仅包含训练归属段落的挖掘视图。排除训练标注中的已知正例及其重复，不读取测试问题或 qrels 制作负例屏蔽表。未知候选仍是假设负例，需要抽查。合格负例不足时记录短缺，按预设规则减少数量或排除问题，不偷偷改采封存组。负采样对照使用相同合格问题、样本对数量和训练步数。

Triplets describe sampling provenance, not the loss input. Materialize unique `(query_id, chunk_id, label)` binary records for BCE: include each unique positive once, deduplicate negatives, and resolve positive/negative conflicts before training. Four negatives do not imply four copies of the positive. Keep assumed-negative provenance even though its training label is zero. Compute `pos_weight` from the actual materialized sampling distribution, not a nominal negative target. Record unique queries, positive/negative pairs and actual draw counts; any repeated sampling is an explicit configuration.

三元组用于记录采样来源，不直接作为损失输入。训练前展开为去重的 `(query_id, chunk_id, label)` 二元 BCE 记录：每个唯一正例只保留一次，负例去重，先解决正负标签冲突。四条负例不意味着复制四次正例；假设负例即使训练标签为零，仍保留其来源状态。`pos_weight` 按实际训练采样分布计算，不按名义负例目标推算。分别记录独立问题数、正负样本对数和实际抽样次数；重复采样必须显式配置。

## 5. Reproducible evaluation / 5. 可复现评测

**Execution protocol extension, revision 4:** choose and seal exactly one `evaluation_protocol` before formal comparisons. The following original paragraphs describe `postrank_pool_v1`. For the user's run-without-code-intervention goal, prefer `prejudged_fixed_candidates_v1` defined in [10 Execution contract](10-accelerator-execution-contract.md): review every frozen test BM25 top-50 candidate plus known positives before training, with ranks/scores/prior labels hidden. Seal the complete judgment bank with the initial test inputs. Training/selection cannot access it. After validation locks the systems, the second seal verifies ranked-pool coverage and binds the **entire** unchanged bank to the locked outputs; it does not require fresh judgments or crop qrels to model top-10. Known positives never enter retrieval artificially. Unresolved required judgments block pre-review readiness; the original post-rank mode instead waits for reviewed inputs. No silent protocol switching, test-driven selection or weakened group ownership is allowed.

**第四版执行协议扩展：**正式比较前只选择并封存一个评测协议。下列原有段落描述排序后汇总审查模式。为满足无需临场改代码运行的目标，优先使用第 10 号文档的固定候选预审模式：训练前对固定测试 BM25 前 50 候选及已知正例全部盲审，隐藏排名分数和旧标注，完整判断库随首次测试输入封存，训练选择不能访问。验证锁定系统后，第二次封存只核对排序池覆盖并将**整个**未变判断库绑定至锁定输出，不需要新判断，不按模型前 10 裁剪标注。已知正例绝不补入检索。预审模式的必需未裁定项阻塞就绪，原后审模式则等待审查输入；不静默切换协议、不据测试选择、不削弱分组归属。

**Review protocol revision 5 (2026-10-07), `primary_audit_v1`:** the judgment bank of `prejudged_fixed_candidates_v1` keeps the same pool (every frozen test question, the BM25 top-50 per question and the known positives; questions and candidate depth unchanged) and the same blinded request, relevance rule and evidence requirement, but replaces the second complete pass with a declared audit. The rule lives in the data config's `[review]` table, which is hashed into every manifest and seal, and is implemented once in `devsearch.data.review_protocol` for the audit, the bank stage and the controller. (1) One complete primary pass by the declared `primary_reviewer` over all items. (2) Only after that pass covers every item, a seeded random second review by a different reviewer, stratified by primary label (1/0) × hidden candidate-rank band (1–5, 6–10, 11–20, 21+, known positives outside the list): from each stratum `max(audit_min_per_stratum, ceil(rate × size))` items (all when smaller), rate `audit_rate_positive` = 0.25 for positive strata and `audit_rate_negative` = 0.08 for negative strata, minimum 30, taken in `sha256("<audit_seed>|second-review|<item_id>")` order with `audit_seed` = 20261007; then every question without a sampled item gets one (a primary positive where it has one). (3) Expansion: once every sampled item has its second review, a stratum whose disagreement rate (a second review that is unresolved or gives the other label, over the stratum's sampled items) exceeds `escalate_disagreement_rate` = 0.10 is second-reviewed in full. (4) Adjudication of every unresolved primary judgment (`unsure`, an unlocatable quote, an unusable line) and every disputed or unresolved second review, by an adjudicator who is neither that item's primary nor second reviewer; an unsure adjudicator settles nothing. (5) Final: the independent adjudicator's resolved judgment, else the resolved primary judgment confirmed by every second review it has; items outside the sample become final only once the escalation is decided. On the frozen 9,852-item pool (197 questions; 985 / 985 / 1,970 / 5,910 items in the four rank bands, 2 outside) the sample is about 890–910 items for primary-positive shares of 1–25% by band, plus question-coverage additions; an escalated large negative stratum adds up to its full size (5,910 at most for rank 21+). These numbers are fixed before any formal judging and depend only on the seed, the primary labels and the hidden ranks; they must never be changed in response to reranker or baseline test scores, and changing them is a recorded protocol revision with new hashes. Labels stay `agent` or `human` as recorded; agreement between two agent runs measures consistency, not validity. The original `double_primary` rule (two complete passes, `min_primary_reviews`) remains available and is the default when no mode is declared.

**第五版审查协议（2026-10-07）`primary_audit_v1`：**固定候选预审的判断库保持原池（全部冻结测试问题、每题 BM25 前 50 及已知正例，问题与候选深度不变）、盲审请求、相关性规则及证据要求不变，以声明的抽检替代第二遍全量审查。规则写入数据配置 `[review]`（哈希进入所有清单和封存），由 `devsearch.data.review_protocol` 统一实现。（1）声明的主审对全部条目完成一遍主审。（2）主审覆盖全部条目后才抽样：按主审标签（1／0）×隐藏候选排名段（1–5、6–10、11–20、21+、候选外已知正例）分层，每层抽取 `max(30, ceil(比例×层大小))`（不足则全取），正例层比例 0.25、负例层 0.08，层内按 `sha256("20261007|second-review|<item_id>")` 排序；未被抽中的问题再各补一条（有正例时取正例）。由不同审查者复核。（3）扩大复核：样本全部复核后，若某层分歧率（复核不确定或标签不同）超过 0.10，则该层全部复核。（4）所有不确定、无效主审及争议或不确定复核，交由既非主审亦非该条复核者的裁决者裁决；裁决者不确定不算定案。（5）定案：独立裁决者的明确判断，否则为经全部复核确认的主审判断；样本外条目在扩大复核决定后才定案。冻结池 9,852 条（197 题）预计抽检约 890–910 条外加问题覆盖补充；若大负例层被扩大，最多增加该层全部（排名 21+ 层 5,910 条）。上述数值在正式判定前固定，只取决于种子、主审标签与隐藏排名，绝不能依据重排序或基线测试成绩调整；修改即为有记录的协议修订并产生新哈希。标签按实际记录为 `agent` 或 `human`；两次 Agent 运行一致只说明一致性而非有效性。原 `double_primary` 规则仍可用，未声明模式时为默认。

Choose checkpoints, hyperparameters, stopping rules and the default serving mode on validation only. Test sealing has two documented stages; neither permits test-driven tuning. Stage 1, before training comparisons: freeze test groups, queries, filters, corpus snapshot, reviewed initial `qrels-v1`, eligibility/adjudication rules and coverage targets with hashes. Training and selection cannot read these queries or judgments. Do not replace a hard query because models perform poorly on it.

仅用验证集选择模型、超参数、停止条件和默认服务模式。测试采用两阶段封存，均不允许根据测试调参。第一阶段在训练对照前，冻结测试组、问题、筛选、语料快照、已审查初始 `qrels-v1`、纳入／裁定规则和覆盖目标并记录哈希；训练与选择流程不得读取测试问题或标注。不能因模型表现差而替换难题。

Stage 2, after locking the comparison systems, checkpoints, retrieval settings and review protocol: generate and preserve their ranked outputs without exposing aggregate test metrics for selection. For each query, pool the top 10 from BM25 and every locked comparison system, plus known positives from `qrels-v1`. Deduplicate and shuffle the pool; reviewers see the query, filters, passage and source evidence, but not system identity, score, original rank or prior label. Record the hidden provenance separately. Review all pooled records under the same relevance rule and resolve disagreements using the predefined process. Known positives enter the judging pool only; never insert them into retrieved candidate lists.

第二阶段在锁定对比系统、checkpoint、召回设置和审查协议后，生成并保存排序输出，不向模型选择流程提供测试汇总指标。每题汇总 BM25 和全部锁定对比系统的前 10 条，以及 `qrels-v1` 已知正例，去重并打乱顺序。审查者可见问题、筛选、段落和来源证据，不可见系统身份、分数、原排名或原标签；隐藏的来源信息另存。按相同相关性规则审查全部汇总项，依据预定流程解决分歧。已知正例仅进入审查池，不得补入真实检索候选。

Freeze pooled `qrels-v2` and the adjudication log before calculating final comparison metrics. Score every system against this same version and preserve changes from v1. An unresolved judgment remains unjudged; disclose judged coverage. A query that loses all confirmed positives is recorded separately under the preset rule, never silently deleted. After scores are revealed, additional tuning requires a new independent test or an explicitly exploratory report. This two-stage process versions annotations; it does not permit changing queries, improving models or selecting a new winner on the test.

计算最终比较指标之前，冻结汇总审查后的 `qrels-v2` 和裁定日志。所有系统使用同一版本计分，并保留相对 v1 的修改记录。无法裁定的项仍为未标注，报告已审查覆盖率；若某题失去全部已确认正例，按预设规则单独记录，不能静默删除。公布成绩后再调参，需要新独立测试或明确标为探索报告。两阶段流程只允许标注版本化，不允许修改问题、改进模型或在测试上另选赢家。

For every test query, apply its frozen source/version filters to the same corpus and save BM25's natural top 50 (or fewer actual matches). Reuse exactly these candidates for every reranker and retain empty lists as retrieval failures. Never insert a gold passage, remove failures, or change candidates to favor a model. Compare BM25, a trained head with frozen encoder, and full fine-tuning; isolate negative-strategy ablations. Any alternative retrieval depth is chosen on validation, named in its report, and never mislabeled Recall@50.

每条测试问题在相同语料中应用已冻结来源／版本筛选，自然召回 BM25 前 50 条或实际更少的匹配项并保存，所有重排模型使用同一候选列表；空列表作为召回失败保留。不得补入正例、删除失败问题或为不同模型更换候选。比较 BM25、冻结编码器训练评分层、完整微调，并单独对比负样本策略。若采用其他召回深度，必须由验证集决定并写入报告，不得仍标为 Recall@50。

The ranking denominator contains valid queries with at least one confirmed positive in the frozen filtered corpus, regardless of whether retrieval finds it. Invalid requests, verified unanswerable questions and zero-confirmed-judgment questions are reported as separate counts and behavioral fixtures; never divide by zero or score an unjudged question as a verified failure. Keep pre/post-adjudication query and group counts. Missing judgments contribute zero gain by metric convention, not proof of irrelevance; report the fraction of top-10 results judged and label Recall as recall of known positives.

排序指标分母为冻结筛选语料内至少有一个已确认正例的合法问题，不取决于检索是否找到它。非法请求、已确认无答案问题和没有已确认判断的问题分别统计并用于行为检查，不能除以零，也不能把未标注问题当作已确认失败。报告裁定前后问题数与组数。缺失标注按指标约定贡献零收益，但不代表已证明无关；报告前 10 条结果的已审查比例，Recall 明确指已知正例的召回。

| Metric | Definition and scope / 定义与范围 |
|---|---|
| Recall@50 | Known relevant passages retrieved / all known relevant passages, averaged over queries. With one known positive, this is a hit rate. / 对每题计算召回的已知相关段落占全部已知相关段落比例再平均；单正例时等于命中率。 |
| MRR@10 | Mean reciprocal rank of the first known relevant result within 10; otherwise zero. / 前 10 名内首个已知相关结果排名倒数的平均值，未命中为零。 |
| nDCG@10 | Normalized discounted gain using frozen relevance judgments; binary labels in v1. / 按固定相关性标注计算归一化折损收益，首版使用二元标注。 |
| Latency | Warmed-up p50/p95, reported separately for retrieval, reranking, and total request time. / 预热后分别报告检索、重排和完整请求的 p50／p95。 |

Include every eligible test query in end-to-end metrics; retrieval misses score zero. A result restricted to queries with a recalled positive is a conditional diagnostic and must be labeled separately. Incomplete judgments limit every metric even after pooled review; these results are not an official benchmark or a claim of exhaustive relevance labeling.

端到端指标包含全部合格测试问题，召回失败计零。仅统计已召回正例的问题属于条件诊断，必须单独标明。即使完成候选汇总审查，标注不完整仍会影响所有指标；这些结果不等于官方基准，也不意味着全部相关段落均已标注。

Report ordinary query-macro metrics and clearly labeled group-macro metrics: first average a metric within each frozen `group_id`, then give each group equal weight. For a model-minus-baseline change, compute paired group differences and bootstrap groups with replacement, using the same sampled groups for both systems. Prespecify confidence level, repetition count and seed before final scores; report the interval and number of independent groups. Do not treat paraphrases as independent bootstrap units. Sparse source/family slices are descriptive, and an interval spanning zero does not support a stable positive-gain claim. Training-seed variability is reported separately.

同时报告常规按问题平均指标和明确标注的按组平均指标：先在固定 `group_id` 内平均，再让各组等权。模型相对基线的变化使用配对组差值，按组有放回 bootstrap，两系统使用相同抽中的组。最终计分前预定置信水平、重复次数和种子，报告区间与独立组数量，不把同义改写当作独立抽样单元。样本稀疏的来源／任务切片只作描述；区间跨零不能支持稳定提升声明。训练种子波动另行报告。

Fix hardware, device, precision, token limits, batch size, candidate count, and concurrency for latency comparisons. Synchronize accelerators before stopping timers. Report quality and latency together; use repeated seeds when feasible and report variation rather than selecting only the best run.

速度比较固定硬件、设备、精度、长度、batch、候选数量和并发条件，加速设备计时结束前进行同步。效果与耗时一起报告；资源允许时运行多个种子，报告波动而非只选最好结果。

## 6. Milestones and acceptance / 6. 阶段与验收

Follow the [overall roadmap](00-architecture-and-roadmap.md): first verify the domain corpus and BM25 slice, then use a smoke subset to verify loading, labels, training, checkpoint recovery, and evaluation. NQ is an optional diagnostic/transfer track; scale it only when domain validation supports the hypothesis. Expand to **10k → 30k → 50k distinct training queries** only after filtering, group availability, and measured memory/throughput support it. These are experiment tiers, not commitments; query counts differ from question–passage pair counts.

遵循整体路线，先验证领域语料与 BM25 可用切片，再用小样本验证加载、标签、训练、断点续训和评测链路。NQ 属于可选排错与迁移支线，仅在领域验证支持假设时扩大。经过过滤、分组与内存吞吐实测后，再考虑 **1 万→3 万→5 万条独立训练问题**。这些是实验档位而非承诺，问题数量与问题—段落样本对数量必须分开记录。

Acceptance requires: versioned manifests; overlap and evidence-retention reports; fixed candidate files; tests for known metric examples and retrieval misses; reproducible training/resume/evaluation commands; baseline comparisons; reviewed failure cases; and explicit labeling of synthetic data and unjudged candidates. Improvement is a hypothesis, not a pass condition. No official-NQ, real-user, or domain-improvement claim is allowed without its corresponding evidence.

验收要求：版本清单、重复检查和证据保留报告、固定候选文件、已知排名与召回失败的指标测试、可复现训练／恢复／评测命令、基线对比、已审查失败案例，以及合成数据和未标注候选说明。效果提升属于待验证假设，不是必须得到的结论；没有对应证据，不得宣称官方 NQ 成绩、真人效果或领域效果提升。

Reference implementation: [official NQ cross-encoder training example](https://github.com/huggingface/sentence-transformers/blob/main/examples/cross_encoder/training/rerankers/training_nq_bce.py). Reference code informs implementation; the protocol above governs this project's experiments.

实现参考为上述官方 NQ cross-encoder 训练示例。参考代码用于学习实现，本规格定义本项目自己的实验边界。
