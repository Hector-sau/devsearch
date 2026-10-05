# DevSearch: PyTorch implementation and local runtime specification / DevSearch：PyTorch 实现与本地运行开发规格

**Execution addendum:** [10](10-accelerator-execution-contract.md) and [12](12-estimates-and-experiment-plan.md) add the controller, separate MPS/CUDA profiles, fixed-candidate pre-review, calibration and bounded recipes. These supersede older Mac-only planning and underspecified calibration units below; the native PyTorch, checkpoint and existing CLI contracts remain. Added commands are specifications until implemented and tested.

**执行补充：**第 10、12 号文档增加控制器、独立 MPS／CUDA 配置、固定候选预审、校准及有限配方，替代下文旧的仅 Mac 规划和未具体化的校准单位；原生训练、断点及已有 CLI 契约保留。新增命令在实现验证前仅为规格。

**Status: implementation specification. Since 2026-10-05 the model-free CPU baseline commands `index`, `retrieve` (baseline mode) and `validate_manifest` are implemented and checked on tiny fixtures only ([record](../reports/baseline-verification.md)); every other command is still a help-only stub that fails closed. No persistent project environment or project model training exists. The package checks are recorded separately from the earlier tiny MPS host probe; real MiniLM verification remains pending. English queries/documents are confirmed.**

**状态：实施规格。自 2026-10-05 起，无模型的 CPU 基线命令 `index`、`retrieve`（baseline 模式）和 `validate_manifest` 已实现，仅在微型样例上检查（见[验证记录](../reports/baseline-verification.md)）；其余命令仍是仅提供帮助、明确拒绝执行的占位。尚无持久项目环境或项目模型训练。包检查与此前极小 MPS 宿主探测分开记录，真实 MiniLM 验证仍待执行。英文问题与文档已确认。**

## 1. Scope and model / 1. 范围与模型

The first version retrieves 50 passages with BM25, reranks question–passage pairs with MiniLM, and returns the top five passages with source URL, document version, and passage identifier. Generation, agents, managed application services, and vector databases are outside the first milestone. A user-provided Mac mini is the planned runtime, whether physically available or remotely hosted. If fewer than 50 passages exist, return the available candidates and record their count.

第一版采用 BM25 检索 50 个段落，再由 MiniLM 为“问题—段落”联合输入打分，返回前五条及来源链接、文档版本、段落编号。第一里程碑暂不包含答案生成、Agent、托管应用服务或向量数据库；用户提供的 Mac mini 是计划运行主机，可以是本地实体或远程托管机器。语料不足 50 条时使用实际候选数并记录。

Use the pretrained `microsoft/MiniLM-L12-H384-uncased` encoder through Transformers, followed by a dropout layer and a linear layer producing one logit. Start with the first-token representation and document this choice. Implement the wrapper as `nn.Module` and implement the training loop in native PyTorch. The model card describes a 33M-parameter English encoder requiring downstream fine-tuning; it does not establish performance on this computer. [MiniLM model card](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)

通过 Transformers 加载 `microsoft/MiniLM-L12-H384-uncased` 编码器，接 dropout 与输出单个 logit 的线性层。起步使用首个 token 的表示并记录选择依据。模型封装为 `nn.Module`，训练循环用原生 PyTorch 实现。官方模型卡说明它是约 3,300 万参数、需要下游微调的英文编码器；这不能证明本机训练性能。

Tokenize pairs jointly. In v1, prefilter NQ and construct domain chunks so the full passage fits the documented query reserve and pair limit, including special tokens. Reject an overlong query; do not silently truncate a positive passage. Use `truncation=False` with an explicit length check, and dynamic padding in `collate_fn`. Persist tokenizer revision, token limits and exclusion statistics. Never feed an additional sigmoid into `BCEWithLogitsLoss`; expose the raw logit as an uncalibrated relevance score, not a probability.

问题与段落联合编码。首版先筛选 NQ、构造领域切块，使完整段落在预留问题长度、特殊 token 和总长度限制下仍可编码。过长问题明确拒绝，不静默截断正例段落。使用 `truncation=False` 加显式长度检查，`collate_fn` 动态补齐，保存 tokenizer revision、长度限制与排除统计。使用 `BCEWithLogitsLoss` 前不额外 sigmoid；对外展示原始 logit，属于未校准相关性分数，不是概率。

## 2. Training and reproducibility / 2. 训练与复现

Start with FP32 and standard `AdamW`. The loop performs zeroing gradients, forward computation, loss, backward computation, optional gradient clipping, optimizer step, and scheduler step. Validation uses `eval()` and disabled gradients. Compare a frozen encoder with full fine-tuning under the same data split and evaluation protocol. For the frozen baseline, keep the encoder in evaluation mode so its dropout does not vary the representations unintentionally.

先用 FP32 与标准 `AdamW`，训练循环依次完成清梯度、前向、损失、反向、可选梯度裁剪、参数更新与学习率更新。验证使用 `eval()` 并关闭梯度。冻结编码器与全量微调使用同一划分和评测协议；冻结基线将编码器保持在评估模式，避免其 dropout 无意改变特征。

For binary pairs, labels are floating-point zeros or ones matching the logits' shape and device. Configure the positive weight explicitly: an initial option is `N_negative / N_positive` for the training pairs actually sampled. With one positive and four negatives, that option is 4. Compare against weight 1 if needed; do not combine balancing mechanisms silently. Negatives must exclude known positives and documented near duplicates. Treat other unlabeled passages as potentially relevant, and audit difficult examples.

二分类标签使用与 logits 形状、设备一致的浮点 0/1。显式配置正例权重，可先按实际采样训练对的 `负例数/正例数` 设置；一正四负时该选项为 4。必要时对比权重 1，不能悄悄叠加多种平衡策略。负例排除已知正例与近重复，未标注段落仍可能相关，困难负例必须抽查。

Save an initial epoch-zero checkpoint, each completed epoch, and the best completed-epoch validation checkpoint. A resumable checkpoint includes model, optimizer, scheduler, completed epoch, global step, configuration/manifest hashes and random-number states. V1 resumes only from an epoch boundary, with `resume_kind: epoch_boundary`. Periodic mid-epoch snapshots may preserve diagnostic weights and progress, but carry `resume_kind: diagnostic_only` and are rejected by `--resume`; an interruption replays from the last complete epoch. Exact mid-epoch continuation is a later feature requiring sampler and batch-position state. Do not claim bitwise reproducibility across devices.

保存初始 epoch-zero 断点、每个完整 epoch 断点及完整 epoch 验证得到的最佳模型。可续训断点包含模型、优化器、调度器、已完成 epoch、global step、配置和清单哈希及随机数状态。首版只从 epoch 边界恢复，标记 `resume_kind: epoch_boundary`。定期中途快照可保留诊断权重和进度，但标记 `diagnostic_only`，`--resume` 必须拒绝；意外中断后从最近完整 epoch 重放。精确中途恢复是需要采样器与 batch 位置状态的后续能力，不宣称跨设备逐位一致。

The checkpoint also stores `next_epoch`, `best_metric`, `best_checkpoint_id` and the per-epoch sampler/generator seed or state; best fields are null before the first validation. Resume verifies immutable model, data, tokenizer, optimizer and scheduler configurations. Documented output-path and device changes are allowed and logged; hyperparameter changes create a forked run with a parent-checkpoint reference, not a claim of identical continuation. Acceptance compares two same-device epochs continuously against one epoch plus save/reload plus one epoch: data order, update count, learning-rate sequence, best-checkpoint selection and parameters within a predeclared numerical tolerance. Record a determinism limitation if that comparison cannot meet its criterion.

断点还须保存 `next_epoch`、`best_metric`、`best_checkpoint_id`，以及每轮采样器／生成器种子或状态；首次验证前最佳指标与断点字段为 null。恢复时核对不可变模型、数据、tokenizer、优化器和调度器配置。允许已记录的输出路径与设备变更；超参数改变须建立引用父断点的新运行，不宣称完全相同的续训。验收比较同设备连续两轮与“一轮后保存重载再一轮”：核对数据顺序、更新次数、学习率序列、最佳模型选择及预先声明容差内的参数；不能满足时明确记录确定性局限。

Write checkpoints to temporary files in the destination filesystem, flush and verify them, then atomically replace the current pointer while retaining the previous valid checkpoint. Record diagnostic-snapshot intervals, run deadlines and the estimated complete-epoch plus validation/save time. Choose bounded datasets/epochs before a run, and do not start an epoch that cannot fit the remaining window plus export margin. Check stop requests between batches; if stopped mid-epoch, save diagnostics and preserve the prior resumable checkpoint. Any new run using a smaller subset receives a new manifest. Store resolved configuration, dependency lock, model revision, seed, metrics and local logs for every run.

断点先写目标文件系统上的临时文件，刷盘并验证，再原子切换当前指针，保留上一份有效断点。记录诊断快照间隔、运行截止时间，以及完整 epoch 加验证和保存的耗时估算。运行前限定数据与 epoch；剩余窗口不足以完成一个 epoch 并留出导出余量时，不启动下一轮。batch 之间检查停止请求，中途停止时保存诊断快照并保留此前可恢复断点。缩小子集后另开运行并生成新清单。每次运行保存完整配置、依赖锁、模型 revision、随机种子、指标和本地日志。

## 3. Data and configuration contracts / 3. 数据与配置契约

Store documents, queries, relevance judgments, split assignments, and candidate lists separately. Use the data specification's canonical fields: `doc_id`, `chunk_id`, `source_id`, `text`, `source_url`, `revision`, and `content_sha256`. Resolve document metadata through `doc_id` and `revision` rather than introducing competing `passage_id` or `source_version` fields. The manifest records source and license, acquisition date, preprocessing/chunking version, SHA-256 hashes, label origin, split policy, and row counts. BM25 index metadata must reference the same corpus hash. Follow the data specification's two-stage test protocol: freeze queries, groups and initial qrels before formal training comparisons; after models and pooling depth are fixed, conduct blinded pooled-candidate judging and freeze final qrels before final metrics. Never tune models or remove queries based on test performance.

文档、问题、相关性标注、划分与候选列表分开保存，采用数据规格的统一字段，通过文档编号和 revision 解析来源。manifest 记录来源与许可、获取日期、清洗切块版本、SHA-256、标签来源、划分规则和样本数。BM25 索引必须对应相同语料哈希。遵循数据规格的两阶段测试协议：正式训练比较前冻结问题、分组和初始 qrels；模型及候选池深度固定后，盲审汇总候选，并在最终指标计算前冻结最终 qrels。不得依据测试表现调参或删题。

Required configuration fields cover paths, corpus language, model/tokenizer revision, device, precision, pair length, batch size, workers, epochs, learning rate, weight decay, scheduler, positive weight, negative-sampling recipe, seed, retrieval depth, and validation metric. Record both requested and resolved settings. Stop on unknown fields, missing files, unavailable explicitly requested devices, or mismatched data hashes.

配置至少覆盖路径、语言、模型与 tokenizer revision、设备、精度、输入长度、batch、worker、epoch、学习率、权重衰减、调度器、正例权重、负采样、种子、检索深度和验证指标。同时保存请求值与实际值；未知字段、文件缺失、指定设备不可用、数据哈希不匹配时明确报错。

Use the package layout in the [overall blueprint](00-architecture-and-roadmap.md): ingestion, data, retrieval, modeling, training, evaluation, artifacts and serving modules, plus thin `train/evaluate/predict/retrieve` CLI entry points. Keep `configs/`, `tests/` and `reports/`; ignored runtime data, caches, indexes and runs live under the host-specific runtime root, preferring attached external storage. Pin dependency versions and model revisions after a successful compatibility check; do not invent a validated version matrix now.

采用整体蓝图中的采集、数据、检索、模型、训练、评估、产物和服务模块，并保留轻量 CLI 入口及配置、测试和报告目录；Git 忽略的运行数据、缓存、索引及模型放主机专属运行目录，优先使用连接的外置盘。兼容性测试通过后再锁定依赖版本和模型 revision，目前不虚构已验证版本组合。

Portable manifests use relative artifact paths or logical IDs resolved inside the selected bundle/runtime root; they do not embed a rental host's absolute paths. Reject path traversal and mismatched file hashes. Export an offline inference bundle containing encoder configuration and weights, scoring-head configuration and weights, all tokenizer files, pair-encoding policy, model/code revisions, and the dependency lock. Serialize weights as state dictionaries with explicit model construction, not a pickled live model object. Keep training checkpoints separate. After transfer, load locally without Hub access and compare fixed-example logits/ranks within a documented tolerance; when resuming, explicitly restore optimizer tensors to the selected device.

便携清单采用产物相对路径或逻辑 ID，在所选发布包／运行根目录内解析，不嵌入租用机绝对路径；拒绝路径越界及文件哈希不符。离线推理包包含编码器配置与权重、评分层配置与权重、完整 tokenizer 文件、联合编码规则、模型和代码版本及依赖锁。权重采用 state dictionary 并显式构建模型，不序列化运行中的完整模型对象。训练断点独立保存。迁移后不访问 Hub 即可加载，以记录的容差核对固定样本 logits／排名；续训时显式恢复优化器张量到选定设备。

Manifest hashing uses a documented canonical JSON encoding with `schema_version`, stable key ordering and exact field rules; a manifest never includes its own hash in the hashed payload. A clean Git commit may identify code, but is not assumed to exist. For uncommitted or untracked code, export a scoped source snapshot and record its hash and file inventory, excluding runtime assets and credentials. A commit ID alone is insufficient when the working tree is dirty. Preserve that exact snapshot with the run and release.

清单哈希使用写入规范的 canonical JSON 编码，明确 `schema_version`、稳定键排序和字段规则；被哈希内容不包含自身哈希。代码可以用干净 Git 提交标识，但不假设已有提交。未提交或未跟踪代码须导出限定范围的源码快照，记录哈希及文件清单，排除运行资产和凭据。工作区有改动时，仅记录提交 ID 不足以复现；实验与发布须保留该确切快照。

## 4. Command-line interfaces / 4. 命令行接口

The following operational commands are contracts. **Implemented and checked on tiny fixtures:** `index`, `retrieve --mode baseline` and `validate_manifest`; they also accept an optional `--runtime-root <root>`, defaulting to `DEVSEARCH_RUNTIME_ROOT`, then to the manifest's own directory. Every other command, and `retrieve` with `--mode rerank|compare` or `--checkpoint`, exists only as `--help` and deliberately exits nonzero as unimplemented. Paths resolve from an explicit runtime root; no command publishes artifacts or contacts tracking services by default.

以下业务命令为契约。**已实现并在微型样例上检查：** `index`、`retrieve --mode baseline` 与 `validate_manifest`；它们另接受可选的 `--runtime-root <root>`，缺省取 `DEVSEARCH_RUNTIME_ROOT`，再缺省取清单所在目录。其余命令，以及带 `--mode rerank|compare` 或 `--checkpoint` 的 `retrieve`，仅有 `--help`，明确非零退出并提示未实现。路径从显式运行根目录解析，默认不发布产物或连接外部实验日志服务。

Exit status of every command: `0` success; `1` invalid input or a validation/runtime failure, reported as `devsearch.<command>: error: ...` without a stack trace; `2` usage error or an operation that is not implemented. Failures write no artifact and nothing to stdout.

所有命令的退出状态：`0` 成功；`1` 输入无效或校验／运行失败，以 `devsearch.<command>: error: ...` 报告且无堆栈；`2` 用法错误或操作尚未实现。失败时不写产物，也不向标准输出写内容。

```text
python -m devsearch.prepare --config configs/sources.toml --output <corpus-manifest.json>
python -m devsearch.prepare_data --config configs/data.toml --corpus <corpus-manifest.json> --output <split-manifest.json>
python -m devsearch.index --config configs/index.toml --corpus <corpus-manifest.json> --output <index-manifest.json> [--runtime-root <root>]
python -m devsearch.validate_manifest --manifest <manifest.json> --runtime-root <root>
python -m devsearch.evaluate --config configs/eval.toml --mode baseline --index <index-manifest.json> --split validation --output <metrics.json>
python -m devsearch.retrieve --config configs/search.toml --mode baseline --index <index-manifest.json> --query "How does gradient accumulation work?" --output <results.json> [--runtime-root <root>]
python -m devsearch.train --config configs/smoke.toml --run-dir <run>
python -m devsearch.train --config configs/train.toml --run-dir <run> --resume <checkpoint>
python -m devsearch.evaluate --config configs/eval.toml --mode rerank --index <index-manifest.json> --checkpoint <checkpoint> --split validation --output <metrics.json>
python -m devsearch.predict --config configs/predict.toml --checkpoint <checkpoint> --pairs <pairs.jsonl> --output <scores.jsonl>
python -m devsearch.export --config configs/export.toml --index <index-manifest.json> --checkpoint <checkpoint> --output <candidate-bundle>
python -m devsearch.serve --config configs/serve.toml --manifest <candidate-bundle/manifest.json> --allow-candidate
python -m devsearch.evaluate --stage rank --comparison-plan <locked-plan.json> --split test --output <ranked-output-manifest.json>
python -m devsearch.evaluate --stage pool --comparison-plan <locked-plan.json> --ranked-outputs <ranked-output-manifest.json> --output <blinded-review-pool.jsonl>
python -m devsearch.prepare_data --seal-final --comparison-plan <locked-plan.json> --reviews <reviewed-pool.jsonl> --output <final-test-manifest.json>
python -m devsearch.evaluate --stage score --comparison-plan <locked-plan.json> --ranked-outputs <ranked-output-manifest.json> --test-manifest <final-test-manifest.json> --output <report.json>
python -m devsearch.export --config configs/export.toml --candidate <candidate-manifest.json> --final-report <report.json> --output <evaluated-bundle>
python -m devsearch.serve --config configs/serve.toml --manifest <evaluated-bundle/manifest.json>
```

`prepare` pins/acquires allowed sources and writes normalized documents, chunks and provenance; its offline mode consumes an existing snapshot. `index` builds deterministic BM25 artifacts: it validates the corpus artifact ([data §2](02-data-and-evaluation.md)), writes `<output>` plus a sibling `<stem>.postings.json` atomically, never creates the output directory, never overwrites a different index (identical inputs rebuild to the same `index_id` and are left untouched), and requires every artifact path to stay inside the runtime root. `retrieve` reads candidate depth, display count, limits and optional filters from `search.toml`, prints the result document (a subset of the API response, [integration doc](04-integration-and-decisions.md)) to stdout or to a new `--output` file that it never overwrites. Label preparation uses the separately versioned data-specification inputs. `validate_manifest` verifies schema, paths, required artifacts and compatibility without mutating them. Baseline retrieval/evaluation requires no checkpoint and does not instantiate the encoder; reranking requires a checkpoint or exported model referenced by configuration. Evaluation configuration identifies frozen queries, qrels and corpus/candidate manifests. Final-test execution additionally requires the predeclared comparison plan and sealed split hashes.

`prepare` 固定并获取允许的来源，生成正文、切块及溯源；离线模式读取已有快照。`index` 构建确定性 BM25 产物：校验语料产物（[数据规格 §2](02-data-and-evaluation.md)），以原子方式写入 `<output>` 及同目录 `<stem>.postings.json`，不创建输出目录，不覆盖不同的索引（输入相同则重建得到相同 `index_id` 且文件保持不变），并要求所有产物路径位于运行根目录内。`retrieve` 从 `search.toml` 读取候选深度、展示数量、限制和可选筛选，将结果文档（API 响应的子集，见[集成文档](04-integration-and-decisions.md)）输出到标准输出或新的 `--output` 文件，且从不覆盖。标注准备遵循数据规格中单独版本化的输入。`validate_manifest` 只验证 schema、路径、必需产物和兼容性，不修改文件。基线检索／评估不需要 checkpoint，也不实例化编码器；重排需要断点或配置引用的已导出模型。评估配置明确固定问题、qrels、语料与候选清单；最终测试还须提供预先固定的比较计划和封存划分哈希。

`prepare_data` consumes traceable question/label/review drafts supplied by the assistant workflow, audits group assignments and evidence, and materializes splits and unique BCE pairs; it does not silently call a paid generator. Its initial preparation writes the first test seal. Final `evaluate` stages separate ranking, blinded pool export and metric scoring; scoring requires the reviewed, validated final test manifest. `prepare_data --seal-final` verifies review coverage and adjudication rules and writes a new manifest without changing test questions or ranks. These interfaces implement the two-stage protocol in the data specification; they do not automate a claim of human review.

`prepare_data` 接收助手流程提供的可追溯问题、标签和审查初稿，检查分组与证据，生成划分和唯一 BCE 对，不悄悄调用付费生成服务；初次准备写入首次测试封存。最终 `evaluate` 区分排序、盲审池导出和指标计算，计分必须提供已审查验证的最终测试清单。`prepare_data --seal-final` 检查审查覆盖和裁定规则，新建清单且不改变测试问题或排名。这些接口落实数据规格的两阶段协议，不能自动产生人工审查声明。

`export` creates a new immutable candidate or evaluated bundle; baseline export uses `--mode baseline` and omits `--checkpoint`. Candidate inspection requires `--allow-candidate`; completed-release activation requires evaluated status and its final report. `--config` takes an explicit file path, resolved once and logged; examples using `configs/...` assume the caller deliberately selected that directory. The bundle contains required runtime configurations. Paths within a config resolve relative to its file or its explicitly declared runtime root, and artifact references resolve from their manifest root. Packaged defaults are package resources; installed code never assumes the original checkout exists. Commands are idempotent only when hashes/configuration match; otherwise they create a new ID or fail. Apart from the three baseline commands named above, none of these examples has been implemented or executed.

`export` 新建不可变候选或已评测发布包；基线导出使用 `--mode baseline` 并省略 `--checkpoint`。加载候选须明确 `--allow-candidate`，完整发布激活须具备 evaluated 状态及最终报告。`--config` 接收明确文件路径，解析后记录；示例 `configs/...` 假设调用者有意选择对应目录。发布包携带所需运行配置。配置内路径相对配置文件或明确运行根目录解析，产物引用相对清单根目录，内置默认值作为包资源；安装后的程序不能假定原 checkout 仍在。仅哈希配置一致时幂等复用，否则新建 ID 或报错。上述接口除上文已标明的三个基线命令外，尚未实现或执行。

Training emits checkpoints, resolved configuration, and step/epoch metrics. Evaluation distinguishes reranking a fixed candidate set from end-to-end retrieval: report MRR@10, nDCG@10, Recall@50, query counts, and missing positives; never insert the correct passage into an end-to-end result. Prediction preserves pair identifiers and emits logits and ranking scores. Search emits source/version identifiers and timing breakdowns. Invalid inputs produce nonzero exit codes; overwriting an existing run requires an explicit option.

训练输出断点、完整配置及 step/epoch 指标。评估区分固定候选集重排序与端到端检索，记录 MRR@10、nDCG@10、Recall@50、问题数量和未检出正例；端到端评测不得补入正确段落。预测保留样本编号，输出 logits 与排序分数；检索输出来源版本和分段耗时。无效输入非零退出，覆盖已有运行需显式参数。

## 5. Local runtime and acceptance gates / 5. 本地运行与阶段验收

Resolve paths from `DEVSEARCH_RUNTIME_ROOT` on the actual execution host. Prefer `/Volumes/MAC专用/AI-Lab/pytorch/devsearch/` when that drive is physically connected and verified. For a remote Mac, use its own verified storage as described in [machine handoff](05-machine-handoff.md); the current Mac's external drive is not automatically available remotely. Before downloads, verify the mount, free space, filesystem suitability, and write access. Keep the environment, Hub/dataset caches, temporary files, indexes and checkpoints under the selected root. Identify necessary host tools and set cache environment variables before importing libraries. [Hugging Face cache configuration](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)

路径从实际执行主机的 `DEVSEARCH_RUNTIME_ROOT` 解析。外置盘实际连接且验证后，优先使用 `/Volumes/MAC专用/AI-Lab/pytorch/devsearch/`。远程 Mac 使用交接文档中所述的独立存储，当前 Mac 外置盘不会自动在远端出现。下载前检查挂载、空间、文件系统适配与写权限，将环境、模型/数据缓存、中间文件、索引和断点指向选定目录；列明必要宿主工具，导入相关库前设置缓存变量。

Probe CPU, MPS, and CUDA rather than assuming availability. `auto` resolves and records a usable backend; explicit unavailable devices fail clearly. Start with FP32, workers 0, pair length 128 or 256 after checking full-evidence length compliance and exclusion rates, and batch size 2 or 4. Measure 50–100 complete training steps before estimating runtime. Check MPS availability through the official API and test actual forward/backward operations. [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)

实际探测 CPU、MPS、CUDA，不预设可用性。`auto` 选择并记录可用后端，显式指定不可用设备时报错。先用 FP32、worker 0，根据完整证据长度合规与排除比例试用长度 128/256 和 batch 2/4。测量 50～100 个完整训练 step 后再估算耗时。MPS 除官方可用性检测外，还需实际验证前向与反向。

The usable BM25 slice is M1; this training work belongs to M2 and can be prototyped alongside ingestion. Training acceptance progresses from 32–64 pairs that the model can overfit, to approximately 1,000 training queries with validation, then to 10,000 only if measured resources justify it. Tests cover batch shape/masks, finite loss and gradients, parameter updates/freezing, known-answer ranking metrics, no forced-positive retrieval, and checkpoint reload/resume. Benchmark warmed-up repeated queries, synchronize accelerators when timing, and report hardware, token lengths, candidate counts, batch size, median/p95 latency, and memory pressure. Speed and quality are measured outcomes, not promised improvements.

M1 先交付可用 BM25 切片；本节训练工作属于 M2，可与采集并行做诊断。先用 32～64 对样本检验能否过拟合，再做约 1,000 条训练问题和独立验证，资源允许后扩大到 1 万。必要测试覆盖 batch 形状与 mask、有限损失/梯度、参数更新/冻结、手算排序指标、不补正例、断点重载与恢复。性能使用预热后的重复查询，计时同步加速器，报告硬件、长度、候选数、batch、中位/p95 耗时及内存压力。速度和质量均以实测为准。

M0 compatibility/timing probes use independent diagnostic fixtures and cannot read future test questions, evidence or labels. Discard their learned weights as a training starting point; formal frozen/full comparisons reload the same pinned pretrained encoder and controlled head initialization after the first test seal. Probe outputs establish runtime capability only. Separately, prepare the real pinned tokenizer on CPU before M1 chunking/query validation; this needs tokenizer assets but no encoder weights or working MPS. Baseline bundles retain those tokenizer assets and length rules without loading an encoder.

M0 兼容性与计时探测仅用独立诊断样例，不读取未来测试问题、证据或标签；诊断训练后的权重不得作为正式起点。首次测试封存后，冻结与全量正式对照重新加载同一固定预训练编码器和受控评分层初值。探测只证明运行能力。另外，M1 切块和查询检查前，在 CPU 上准备真实固定版本 tokenizer；仅需分词器资产，不需编码器权重或 MPS。基线包保留分词器及长度规则，但不加载编码器。

### Packaging and test layers / 打包与测试分层

`pyproject.toml` declares the `src` package layout, supported Python range after verification, build backend, runtime dependencies and test/UI optional groups. The lockfile records resolved dependencies; a target-host runtime report records OS, architecture and device versions. A wheel must install into a clean target environment and expose all documented module entry points, starting outside the original checkout with no access to its caches. Test the installed wheel, not only an editable checkout; keep hardware-specific setup outside reusable model/data logic.

`pyproject.toml` 声明 `src` 包结构、验证后的 Python 支持范围、构建后端、运行依赖及测试／UI 可选组。依赖锁记录实际解析版本，目标主机报告记录系统、架构和设备版本。wheel 必须可在干净目标环境安装并提供全部已文档化入口，在原 checkout 之外且无法访问旧缓存的位置启动；验收已安装 wheel，不只验证 editable 工作区，硬件配置与通用模型／数据逻辑分离。

The proposed fast gate is `python -m pytest -m "not integration and not mps"`: CPU-only, network disabled, tiny local corpus/tokenizer fixtures and a tiny initialized encoder or fake scorer, without downloading MiniLM. It covers schemas, lexical/no-match retrieval, filters, known ranking metrics, checkpoint boundaries, API fallback/errors and package imports. The separate target-host integration gate loads the pinned real MiniLM from prepared local assets and verifies MPS forward/backward, checkpoint transfer, batch scoring and service parity. Missing MPS is reported as unavailable, not a passed hardware test. Run the same fast gate locally and in CI when CI exists; no remote CI service is required for v1. This section defines future checks, not completed tests.

拟定快速检查为 `python -m pytest -m "not integration and not mps"`：仅 CPU、禁网络、使用极小本地语料／tokenizer 样例及小型初始化编码器或假评分器，不下载 MiniLM。覆盖结构、词项匹配／无匹配检索、筛选、手算指标、断点边界、接口降级错误与包导入。目标机集成检查独立加载已准备的固定版本 MiniLM，验证 MPS 前后向、断点迁移、批量评分及离在线一致。没有 MPS 时记录不可用，不能算硬件测试通过。未来 CI 复用本地快速检查即可，首版不要求远程 CI 服务。本节只定义待执行检查。

**Current gate (implemented):** `PYTHONPATH=src python -m unittest discover -s tests -v`, standard library only (the package has no runtime dependencies yet), CPU-only and network-free, covering schemas and strict rejection, the tokenizer, hand-computed BM25 scores, filters, zero-match and ties, index integrity, and the three baseline commands run as subprocesses outside the checkout. `scripts/check_wheel.py` additionally installs the built wheel into an isolated environment and runs those commands over the fixture corpus. CI runs both on Python 3.11 and 3.12 ([workflow](../.github/workflows/scaffold-check.yml)). The `pytest` gate above remains the target once model-dependent tests exist; tests that need a real tokenizer, MiniLM or MPS stay outside this gate.

**当前检查（已实现）：** `PYTHONPATH=src python -m unittest discover -s tests -v`，仅用标准库（包尚无运行依赖）、仅 CPU、禁网络，覆盖结构与严格拒绝、分词器、手算 BM25 分数、筛选、无匹配与同分、索引完整性，以及在 checkout 之外以子进程运行的三个基线命令。`scripts/check_wheel.py` 另将构建的 wheel 装入隔离环境，并用样例语料运行这些命令。CI 在 Python 3.11 与 3.12 上运行两者（[工作流](../.github/workflows/scaffold-check.yml)）。上文 `pytest` 检查仍是出现依赖模型的测试后的目标；需要真实 tokenizer、MiniLM 或 MPS 的测试不在此检查内。

## 6. Reference boundary and pending decisions / 6. 参考边界与待决事项

The Sentence Transformers NQ example is a design reference, not a copy-and-run recipe: it currently uses ModernBERT, batch size 64, BF16, workers 4, FAISS mining, additional evaluators, and an upload attempt. Borrow its data/evaluation ideas, cite the pinned commit and license, and omit automatic upload and external logging. [Reference script](https://github.com/huggingface/sentence-transformers/blob/main/examples/cross_encoder/training/rerankers/training_nq_bce.py)

Sentence Transformers 的 NQ 示例是设计参考，不能直接照搬：当前采用 ModernBERT、batch 64、BF16、worker 4、FAISS 挖掘、额外评测和上传尝试。借鉴数据与评估思路，记录固定 commit 与许可，取消自动上传及外部日志。

English queries and documents are confirmed. PyTorch Tutorials and Transformers documentation define the initial domain; the data specification defines label provenance and review. A 2026-10-05 check confirmed the current MacBook's Apple M4, 16 GB memory, macOS 26.6.2 and arm64. Existing host PyTorch 2.8.0 reports MPS available outside the restricted sandbox and passed a tiny linear-layer forward/backward/SGD-update test. The restricted sandbox reported MPS unavailable. These results do not validate MiniLM performance or the future machine. The user subsequently selected a Mac mini M6 with 24 GB memory, to be arranged by the user and operated for this project by the assistant after access is available. Source revisions, clean sample counts, throughput and usable training duration remain implementation checks. No GPU-cloud purchase is planned; additional expenses beyond the supplied machine require a concrete user decision.

英文问题与文档已确认。初始领域限定为 PyTorch Tutorials 与 Transformers 文档，标签来源和审查遵循数据规格。2026-10-05 检查确认当前 MacBook 为 Apple M4、16 GB 内存、macOS 26.6.2 与 arm64；现有 PyTorch 2.8.0 在受限沙盒外检测到 MPS，并通过极小线性层前向、反向及 SGD 更新测试，沙盒内报告 MPS 不可用。这不代表 MiniLM 性能或未来机器已验证。用户随后决定自行安排 24 GB Mac mini M6，可访问后由助手负责项目运行。来源版本、清洗数量、吞吐和可用训练时长仍待实测。不再规划云 GPU 采购；所提供机器以外的新增费用需要具体用户决定。
