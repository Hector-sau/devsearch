# Reproduction guide / 复现指南

## Public release scope / 公开版范围

Version1.1.0 includes source, original miniature fixtures, scoped result summaries, actual demonstration screenshots and a CPU arithmetic-replay tool. Original code uses Apache-2.0; model/documentation dependencies retain their upstream terms. No GPU is needed for the small example or replaying already frozen validation orders.

版本1.1.0提供源码、原创微型样例、注明范围的成绩、真实演示截图及 CPU 算术重放工具。原创代码采用 Apache-2.0，上游模型／文档条款保持原义；小样例和既有验证排序重放无需 GPU。

The private dataset/review draft formerly hosted in this repository is preserved in owner-held storage and removed before public publication. `configs/gpu.toml` expects the original sealed package to be copied locally; its original SHA-256 remains enforced. A public clone alone does not include labels, a real encoder, the real index or native optimizer checkpoints. See [validation replay](VALIDATION_REPLAY.md), [real R5 inference](R5_INTEGRATION.md) and [third-party terms](../THIRD_PARTY_NOTICES.md).

原私有数据／审查草稿在公开前转存到所有者存储，公开仓库不再托管。GPU 配置要求本地提供原封存数据包，并继续校验原 SHA-256；单独公开克隆不包含标签、真实编码器、真实索引或原优化器断点。验证重放、真实推理和上游条款见上述入口。

## Small CPU example / 小型 CPU 示例

Use Python **3.11+** and the repository root. This example uses only the standard library and committed miniature fixtures. It creates a separate temporary runtime; it needs no model download, GPU or credentials. Its output demonstrates the command path, not ranking quality on the real corpus.

在仓库根目录使用 Python **3.11+**。本例只使用标准库与仓库内微型样例，创建独立临时运行目录，无需模型下载、GPU 或凭据；结果用于演示命令流程，不代表真实语料上的排序质量。

```bash
export PYTHONPATH="$PWD/src"
export DEVSEARCH_RUNTIME_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/devsearch-demo.XXXXXX")"
mkdir -p "$DEVSEARCH_RUNTIME_ROOT/index"
cp -R tests/fixtures/mini_corpus "$DEVSEARCH_RUNTIME_ROOT/corpus"
python3.11 -m devsearch.index --config configs/index.toml \
  --corpus "$DEVSEARCH_RUNTIME_ROOT/corpus/corpus-manifest.json" \
  --output "$DEVSEARCH_RUNTIME_ROOT/index/index-manifest.json"
python3.11 -m devsearch.retrieve --config configs/search.toml --mode baseline \
  --index "$DEVSEARCH_RUNTIME_ROOT/index/index-manifest.json" \
  --query "how do I load_state_dict"
python3.11 -m devsearch.validate_manifest \
  --manifest "$DEVSEARCH_RUNTIME_ROOT/index/index-manifest.json"
```

The commands write under `$DEVSEARCH_RUNTIME_ROOT`; index creation prints the manifest path. Keep or remove only this dedicated example directory when finished. The application and CLI reference is [docs/13](13-operator-reference.md); real-release API demonstration commands are in [DEMO_SCRIPT](DEMO_SCRIPT.md).

命令写入 `$DEVSEARCH_RUNTIME_ROOT`，创建索引时打印清单路径。完成后按需保留或仅清理这个独立样例目录。应用命令参考见第 13 号文档，真实发布包接口演示见演示脚本。

## Verification levels / 验证层级

| Level / 层级 | Requirements / 要求 | Evidence / 能证明什么 |
| --- | --- | --- |
| Baseline fixture / 基线样例 | Python 3.11+, source and fixtures / 源码与样例 | Commands and deterministic retrieval / 命令与确定性检索 |
| Standard-library suite / 标准库测试 | `PYTHONPATH=src python3.11 -m unittest discover -s tests` | Available tests; optional ML/API skips remain skips / 可用测试；ML/API 跳过仍须注明 |
| Full CPU CI / 全量 CPU CI | `ml` and `serve` extras; workflow installs its tooling requirements / 安装可选依赖及工作流工具依赖 | Full suite, zero-skip receipt and installed-wheel checks / 全量零跳过与安装包检查 |
| Exact dependency evidence / 准确依赖证据 | Hashed platform lock and separate evidence workflow / 平台锁及独立证据工作流 | Exact packages, isolation and real-model CPU recovery / 准确包、隔离与真实模型恢复 |
| Real GPU campaign / 真实 GPU 实验 | Pinned source plus private runtime artifacts and target GPU / 固定源码、私有产物及目标 GPU | Actual hardware runs and bound outputs / 实机运行及对应产物 |

CI result links are in [EVIDENCE_INDEX](../reports/EVIDENCE_INDEX.md). `cc0e7f0` has a 906-test zero-skip exact-lock result; that count is scoped to that commit and environment. A docs commit does not inherit a new hardware receipt merely because its Python files match.

CI 链接见证据索引。`cc0e7f0` 的准确依赖环境有 906 项、零跳过通过记录，该数量仅适用于对应提交和环境。文档提交即使 Python 文件相同，也不会自动获得新的实机回执。

## Full experiment reproduction / 完整实验复现

1. Choose a **recorded experiment**, its source commit and its original label version. Use the identities in [RESULTS](../reports/RESULTS.md), [MODEL_CARD](MODEL_CARD.md) and the evidence JSON files.
   先选择一个**已有记录的实验**、源码提交及原标签版本，按结果、模型说明和证据 JSON 定位。
2. Restore owner-provided private artifacts under a durable runtime root. The owner supplies the sealed dataset/review state, original model manifest and files, index/configuration, and the selected checkpoint where inference is needed. Verify hashes before use.
   在持久运行目录恢复所有者提供的私有产物，包括封存数据／审查状态、原模型清单与文件、索引／配置，以及推理所需断点；使用前校验哈希。
3. Rebuild the recorded environment from its platform lock. Follow [GPU_QUICKSTART](GPU_QUICKSTART.md) for the existing operator sequence; [docs/16](16-grouped-loss-experiment.md) governs the optional six-run comparison.
   从对应平台依赖锁重建环境，按 GPU 操作指南执行现有入口；六次分组对照遵循第 16 号规格。
4. Keep test access within the locked campaign. Training and exploratory selection use train/validation only. New top-ten candidates without labels require a versioned shared-pool extension before complete new-model quality reporting.
   测试集仅由锁定后的正式流程访问；训练与探索选择只用训练／验证。新模型前十出现未判定候选时，完整质量报告前须扩展并封存共享标签版本。
5. Preserve original identities. A moved or rebuilt environment must pass the corresponding gates; do not substitute another serialized model archive at the same upstream revision.
   保留原标识；迁移或重建环境须通过对应检查，不能因上游版本号相同而替换另一份序列化权重。

A plain repository clone can run the small example and tests; it cannot reproduce private-label experiments without the owner-held artifacts. **No new training is required to complete the current documentation and baseline delivery.**

单独克隆仓库可以运行小样例与测试；完整私有标签实验还需要所有者持有的产物。**完成当前文档和基线交付不需要新增训练。**

## Frozen R5 inference reproduction / 冻结 R5 推理复现

The completed R5 extension uses CPU inference without new training. Restore the owner's verified inference archive, corpus/index, frozen configuration and private evaluation inputs into an approved runtime. The selected archive is 113,941,262 bytes, SHA-256 `1abe8c743bc2e852e61165a98aecb79d642b6badc8e1c860eddda11fed65bdf8`; native optimizer checkpoints are not needed for this inference route and remain on the server.

已完成的 R5 扩展只需 CPU 推理，不需要重新训练。在获认可的运行目录恢复所有者提供的推理压缩包、语料／索引、冻结配置及私有评测输入。选定压缩包为 113,941,262 字节，哈希如上；此推理流程无需原优化器断点，原断点保留服务器。

Use the installed wheel from frozen implementation `b76e349adcbb46a73c430889632810cbaebbd66a`, outside its checkout, with the recorded CPU dependencies. Its [CI 37958581548](https://github.com/Hector-sau/devsearch/actions/runs/37958581548) passed all four jobs, including 936 ML tests per Python 3.11/3.12 job with zero skips. Follow [R5_INTEGRATION](R5_INTEGRATION.md) for model identity, fusion, startup checks and CPU acceptance commands; serve the immutable candidate only with `--allow-candidate`.

使用上述冻结实现的已安装 wheel，在检出目录之外配合实测 CPU 依赖运行。对应 CI 四项作业通过，Python 3.11／3.12 的 ML 作业分别为 936 项、零跳过。模型身份、融合、启动检查及 CPU 验收命令见 R5 接入说明；不可变候选服务须显式使用 `--allow-candidate`。

Existing sealed labels, private review tooling and rank receipts are needed to reproduce the reported 59-question comparison; do not generate substitute labels or rerun model selection on the opened holdout. The first evaluation failed both unchanged quality gates. Preserve the formal BM25 default, original reports and no-tuning boundary. The warm local CPU result does not establish cold-start, network, concurrency, GPU or production-SLA performance. [Independent report and compact evidence](../reports/r5-independent-holdout-2026-10-10.md).

复现已报告的 59 题对比还需要原封存标签、私有审查工具及排名回执，不生成替代标签，也不在已打开的留出集重新选择模型。首次评测未达到两个原质量门槛；保留正式 BM25 默认、原报告及不调参边界。本机 CPU 预热后结果不代表冷启动、网络、并发、GPU 或生产 SLA 性能，详见独立报告及小型证据。
