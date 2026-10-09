# Configuration ownership / 配置职责

Configuration files are **TOML**, read with the standard-library `tomllib` (Python 3.11+): no dependency, comments allowed, and unknown or duplicate keys fail. Every file below ships as an example. `index.toml` and `search.toml` run as they are; the others carry placeholder paths (and `sources.toml` an invalid placeholder revision) that an operator must replace, so none can run a job unchanged. Implement strict schemas and reject missing source/model revisions before real data acquisition or experiments.

配置文件采用 **TOML**，由标准库 `tomllib`（Python 3.11+）读取：无需依赖、可写注释，未知或重复键会报错。已实现并提供示例的有两个；其余是 CLI 计划契约，现在不能启动任务。后续须实现严格结构，真实采集或实验前拒绝缺失来源及模型版本的配置。

| File / 文件 | State / 状态 | Owner / 负责范围 |
| --- | --- | --- |
| `index.toml` | **Implemented** (T01/T04, fixture level) / 已实现 | Lexical tokenizer, BM25 variant and parameters / 词项分词、BM25 变体与参数 |
| `search.toml` | **Implemented** (T01/T04, fixture level) / 已实现 | Candidate depth, display count, limits and optional filters / 候选深度、展示数量、限制与可选筛选 |
| `sources.toml` | **Implemented** (T03, `prepare`) / 已实现 | Pinned revisions, licenses, include/exclude patterns, token budget and tokenizer assets / 固定版本、许可、包含排除规则、长度预算及分词器 |
| `data.toml` | **Implemented** (`prepare_data`) / 已实现 | Evidence groups, splits, labels, sampling and test seals / 证据组、划分、标注、采样和测试封存 |
| `train.toml` | **Implemented** (`train`) / 已实现 | One recipe, prepared data, assets, device profile and length policy; `[run] diagnostic` allows long tiny-set checks only / 单个训练方案、数据、资产、设备与长度策略 |
| `eval.toml`, `predict.toml` | **Implemented** (`evaluate`, `predict`) / 已实现 | Device profile, scoring batch, assets and length policy for loading a model / 加载模型所需的设备、批大小、资产及长度策略 |
| `serve.toml` | **Implemented** (`serve`) / 已实现 | Loopback host and port, per-search deadline, accelerator lease path / 回环地址端口、单次搜索期限、加速器租约路径 |
| `export.toml` | **Implemented** (`export`) / 已实现 | Device, scoring batch, search limits and the fixed smoke questions of a release / 设备、批大小、搜索限制与发布包的固定冒烟问题 |

`index.toml` requires every key (`schema_version`, `tokenizer`, `[bm25]` `idf`/`k1`/`b`); `search.toml` requires `schema_version`, `candidate_k`, `top_k`, `max_candidate_k`, `max_query_chars` and accepts an optional `[filters]` table (`sources`, `[filters.versions]`). Values are explicit rather than defaulted so the resolved settings are always recorded. The BM25 numbers in the examples are common starting values, **not tuned** on this project's data; any tuning uses validation queries only. The examples work with the tiny fixture corpus and carry no real source or model revision.

`index.toml` 要求全部键（`schema_version`、`tokenizer`、`[bm25]` 的 `idf`/`k1`/`b`）；`search.toml` 要求 `schema_version`、`candidate_k`、`top_k`、`max_candidate_k`、`max_query_chars`，并接受可选的 `[filters]` 表（`sources`、`[filters.versions]`）。值须显式给出而非隐式默认，使实际设置总被记录。示例中的 BM25 数值是常见起点，**未**在本项目数据上调优；调优只能使用验证问题。示例适用于微型样例语料，不含真实来源或模型版本。

Use `DEVSEARCH_RUNTIME_ROOT` as the host-specific root (or pass `--runtime-root`). CLI config paths are explicit; values inside a config resolve relative to that file or its declared runtime root, not the original developer's checkout. Export runtime configuration with the release. Future example files must identify unresolved fields and fail validation until filled; no fake revisions or fabricated compatibility locks.

以 `DEVSEARCH_RUNTIME_ROOT`（或 `--runtime-root`）作为主机专属根目录。CLI 明确接收配置路径，内部路径相对配置文件或声明的运行根目录解析，不依赖原开发者 checkout；发布包携带运行配置。未来示例须标识未定字段，填写前不能通过校验，不编造版本或兼容依赖锁。
