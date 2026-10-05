# Configuration ownership / 配置职责

Configuration parsers and production configuration files are not implemented in the scaffold. The names below are planned CLI contracts, not files that can run a training job today. Implement strict schemas and reject missing source/model revisions before real data acquisition or experiments.

骨架尚未实现配置解析或正式配置文件。下列名称是 CLI 计划契约，不代表现在已有可启动训练的配置。后续须实现严格结构，真实采集或实验前拒绝缺失来源及模型版本的配置。

| Planned file / 计划文件 | Owner / 负责范围 |
| --- | --- |
| `sources.yaml` | Pinned repo/path/license selections and parser policy / 固定仓库路径许可及解析策略 |
| `data.yaml` | Evidence groups, splits, labels, sampling and test seals / 证据组、划分、标注、采样和测试封存 |
| `index.yaml` | Lexical tokenizer, BM25 variant and immutable corpus / 词项分词、BM25 变体和不可变语料 |
| `smoke.yaml`, `train.yaml` | Diagnostic-only vs formal data, device, loop and resume / 独立诊断与正式数据、设备、训练和恢复 |
| `eval.yaml`, `predict.yaml` | Fixed candidates/qrels, stages, metrics and score output / 固定候选标签、阶段、指标和评分输出 |
| `search.yaml`, `serve.yaml` | Filters, limits, admission, readiness and loaded release / 筛选限制、接入、就绪和加载版本 |
| `export.yaml` | Portable relative paths, inference files and release lineage / 便携相对路径、推理文件和发布溯源 |

Use `DEVSEARCH_RUNTIME_ROOT` as the host-specific root. CLI config paths are explicit; values inside a config resolve relative to that file or its declared runtime root, not the original developer's checkout. Export runtime configuration with the release. Future example files must identify unresolved fields and fail validation until filled; no fake revisions or fabricated compatibility locks.

以 `DEVSEARCH_RUNTIME_ROOT` 作为主机专属根目录。CLI 明确接收配置路径，内部路径相对配置文件或声明的运行根目录解析，不依赖原开发者 checkout；发布包携带运行配置。未来示例须标识未定字段，填写前不能通过校验，不编造版本或兼容依赖锁。
