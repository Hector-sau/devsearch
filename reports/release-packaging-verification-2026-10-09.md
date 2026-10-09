# Version 1 packaging verification / 第一版打包验证

**Status: local release candidate verified, not published.** Both project metadata and `devsearch.__version__` are `1.0.0`. The package description now identifies the implemented BM25 retrieval and optional native PyTorch MiniLM reranker. These edits do not change training, data, model selection, dependency locks or serving behavior.

**状态：本地发布候选已验证，尚未发布。** 项目元数据与 `devsearch.__version__` 均为 `1.0.0`；包描述已说明实现中的 BM25 检索与可选原生 PyTorch MiniLM 重排。本次修改不改变训练、数据、模型选择、依赖锁或服务行为。

## Verification environment / 验证环境

- Starting source: `b23b0f4`, followed by the release-preparation working-tree edits. This report is not a receipt for a final tag or frozen release commit.
  起始源码为 `b23b0f4`，随后加入发布准备的工作区改动；本报告不是最终标签或冻结发布提交的回执。
- macOS arm64; Python `3.11.16`; existing setuptools `84.0.0`, pip `26.2.1` and wheel `0.48.0`.
  macOS arm64，Python `3.11.16`，使用已有的 setuptools、pip 和 wheel。
- New dependencies installed: **0**. Network downloads: **0 bytes**. Build uses `--no-build-isolation --no-deps --no-index`.
  新安装依赖 **0** 个，网络下载 **0 字节**；构建禁止隔离环境安装、依赖下载及索引访问。

## Candidate artifact / 候选产物

The wheel is stored outside Git under the workspace runtime, `devsearch-runtime/release-v1-20261009/assets/`. A build snapshot containing `pyproject.toml`, `README.md` and `src/` was copied to that runtime before building, so build directories and generated package metadata did not enter the checkout.

wheel 保存在 Git 外的工作区运行目录；构建前将 `pyproject.toml`、`README.md` 与 `src/` 复制到该目录，避免构建目录和生成元数据进入仓库。

| Field / 字段 | Verified value / 实际值 |
| --- | --- |
| Filename / 文件名 | `devsearch-1.0.0-py3-none-any.whl` |
| Size / 大小 | `296155` bytes / 字节 |
| SHA-256 | `6c3b77c96b14bf9183a0795ef98bc0e42db410aec359b184215271bb6440742b` |
| Metadata name / 元数据名称 | `devsearch` |
| Metadata and import version / 元数据及导入版本 | `1.0.0` |
| Python requirement / Python 要求 | `>=3.11` |

This candidate embeds the README from the build snapshot. **Rebuild and reverify the wheel after the final source and documentation are frozen**, then use that wheel's identity in the release manifest. Do not reuse this candidate hash for a different artifact.

该候选包嵌入构建快照中的 README。**最终源码与文档冻结后，须重新构建并验证 wheel**，再将其标识写入发布清单；其他产物不能沿用本候选哈希。

## Commands and outcomes / 命令及结果

Commands below show the executed arguments with the absolute runtime path abbreviated as `$RELEASE_RUNTIME`. This directory is the sibling `devsearch-runtime/release-v1-20261009` of the repository checkout.

以下命令保留实际执行参数，运行目录的绝对路径缩写为 `$RELEASE_RUNTIME`，即 checkout 同级的运行目录。

```bash
# Run from the external packaging runtime / 从外部打包运行目录执行
/opt/homebrew/bin/python3.11 -m pip wheel \
  --no-build-isolation --no-deps --no-index \
  --disable-pip-version-check --no-cache-dir \
  --wheel-dir "$RELEASE_RUNTIME/assets" \
  "$RELEASE_RUNTIME/packaging/build-source"

# Run the repository gate; it creates a fresh isolated environment outside Git
# 执行仓库检查，在 Git 外创建全新隔离环境
/opt/homebrew/bin/python3.11 scripts/check_wheel.py \
  "$RELEASE_RUNTIME/assets/devsearch-1.0.0-py3-none-any.whl"

PYTHONPATH=src /opt/homebrew/bin/python3.11 -m unittest discover \
  -s tests -p test_check_wheel.py -v
```

All commands exited **0**. The installed-wheel check installed only the supplied wheel with indexes and dependencies disabled, then used `python -I` from a directory outside the checkout. It verified that the import comes from the fresh environment, distribution metadata and `__version__` agree, and importing the package does not import ML or web packages.

所有命令退出码均为 **0**。安装包检查禁止索引和依赖安装，仅安装指定 wheel，然后在 checkout 外使用 `python -I`；核验导入来自全新环境、版本一致，并确认导入包时不加载 ML 或 Web 依赖。

The existing portability checks passed: build a 20-chunk fixture index, retrieve `Checkpoints > Loading` with the expected source-line reference, obtain an empty response for an unrelated query, validate the manifest, reject reranking without a model, and check help plus bare-invocation refusal for all 17 CLI modules.

既有便携性检查全部通过：构建 20 段样例索引、返回预期的模型加载段落及来源行、无关问题返回空结果、清单验证通过、缺少模型时拒绝重排，以及 17 个命令模块的帮助与缺参拒绝。

The seven new release-identity tests passed with **0 failures and 0 skips**. They reject a stale version, unrelated distribution, missing or changed Python requirement and multiple distribution metadata files. One test performs an actual offline installation of a wheel whose metadata version is current but imported `__version__` is stale; the gate rejects it before running baseline commands.

新增七项发布标识测试 **0 失败、0 跳过**，覆盖旧版本、其他包、缺失或改变的 Python 要求及多套元数据。其中一项实际离线安装“元数据版本正确、导入版本过期”的 wheel，确认在基线命令执行前拒绝。

The three commands in [the small CPU quick start](../docs/REPRODUCIBILITY.md) also ran in a newly created runtime with the documented query. All exited **0**: `built` with 20 chunks, `ok` with the expected first result, then `valid`.

[小型 CPU 示例](../docs/REPRODUCIBILITY.md) 的三个命令也已在新建运行目录按文档问题执行，均退出 **0**：20 段索引 `built`、首条结果符合预期的 `ok`，最后清单 `valid`。

## Preserved records and boundaries / 记录及边界

Local records are under `$RELEASE_RUNTIME/packaging/`: `verification-receipt.json`, `logs/check-wheel.txt`, `logs/check-wheel-command.json` and `logs/quickstart-results.json`. The receipt contains the candidate wheel hash, environment, check counts and hashes of the build snapshot's key inputs.

本地记录位于该运行目录的 `packaging/`：验证回执、安装包日志、命令参数和小样例输出；回执保存候选 wheel 哈希、环境、检查数量及构建快照关键输入哈希。

These are tiny-fixture CPU and packaging checks. They do not add a GPU/MPS receipt, evaluate a production encoder, replace historical experiment evidence, prove real-corpus UI behavior, or publish a release. Final integrated CI and final-commit packaging verification remain the release executor's responsibility.

以上属于微型样例 CPU 与打包检查，不新增 GPU/MPS 回执，不评测正式编码器，不替换历史实验，不证明真实语料界面行为，也未发布版本。最终整合 CI 与冻结提交后的打包验证仍须由发布主执行者完成。
