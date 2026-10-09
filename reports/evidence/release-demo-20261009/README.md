# Real formal-release demonstration / 真实正式发布包演示

The local Mac restored the evaluated BM25 release and exercised its actual API and Streamlit application. This is an agent-operated demonstration using authored D1–D4 inputs from [DEMO_SCRIPT](../../../docs/DEMO_SCRIPT.md), not a human user study, a held-out evaluation or a latency benchmark. No GPU was started and no model/checkpoint was downloaded.

本机恢复已评测 BM25 发布包，并运行其真实 API 与 Streamlit 应用。本目录记录 Agent 使用专门编写的 D1–D4 输入开展的演示，不是真人测试、保留集评测或延迟基准；没有启动 GPU 或下载模型／断点。

| Item / 项目 | Recorded identity / 记录标识 |
| --- | --- |
| Formal release / 正式包 | `release-ce3160fc579a52ae2ec30d15412853eb`; evaluated, BM25 default, no model / 已评测、BM25 默认、无模型 |
| Original archive / 原备份 | 378,786,182 bytes; SHA-256 `b8ee3cd6817a3fcf97d60d87193b76d485743ecc4737c010f74d2fd489bb0e60` |
| Restoration / 恢复 | only `backup/release/`; 479 archive members, 17,447,914 uncompressed bytes / 仅解压正式包 |
| Original export code / 原导出代码 | `e3168ca6fef9b64ea0688ea059332c729125e30f`; unchanged historical attribution / 历史归属不变 |
| Current serving code / 当前服务代码 | individual SHA-256 values for UI, API, pipeline, startup and CLI in `demo-record.json`; checkout head is recorded as an ancestor, with working-tree version/doc preparation explicitly disclosed / 记录服务文件哈希及版本整理工作区范围 |
| API environment / API 环境 | Python 3.11.16; FastAPI 0.141.1, uvicorn 0.52.4, httpx 0.28.1; actual values also in receipt / 回执保存实测版本 |
| UI environment / 界面环境 | Streamlit 1.65.0 in an isolated runtime `pip --target` directory; training packages untouched / 界面依赖独立安装 |
| Service addresses / 服务地址 | `127.0.0.1:8810` API, `127.0.0.1:8811` UI; loopback only / 仅本机 |

- `demo-record.json`: verified archive and release identities; startup checks and five real HTTP request/response summaries. The startup replay passed eight queries; ordinary/empty/fallback requests returned 200; the invalid source returned 422. Passage text and private evaluation queries are omitted.
- `streamlit-components.json`: actual Streamlit AppTest execution calling the live API, with normal results and five source links, empty-result information, invalid-filter error, and compare fallback warning. Four scenarios passed without exceptions. This component evidence does not stand in for a browser screenshot.
- `normal-search.jpg`, `empty-results.jpg`, `invalid-filter.jpg`, `compare-fallback.jpg`: four actual browser scenarios independently executed and captured by the main Codex agent using CUA. All four passed. `source-results.jpg` and `normal-search-preview.jpg` provide additional result views. Source buttons display the pinned upstream URLs.
- `capture-manifest.json`: screenshot and JSON receipt sizes/SHA-256 values, bound to the exact UI/API/serving file hashes. Raw browser DOM was kept outside Git because it includes upstream excerpts.

- `demo-record.json`：发布包与备份标识、八条启动回放、五项真实 HTTP 场景；省略段落正文与私有评测题。
- `streamlit-components.json`：实际运行 Streamlit 脚本并调用真实 API，四项组件场景无异常；该证据不能代替浏览器截图。
- 主模型通过 CUA 独立操作真实浏览器，普通检索、空结果、无效筛选及对照降级四项均通过，并保存实际截图；另外两张截图补充结果与来源视图。
- `capture-manifest.json` 保存截图与回执的大小、哈希及服务实现文件哈希；含上游正文的原始 DOM 留在 Git 外。

The first normal result is “Automatic Mixed Precision > Saving/Resuming”, with a source button pointing to `recipes_source/recipes/amp_recipe.py` at the pinned PyTorch tutorials commit `11512db7cbbcc4b530ecfc63205d3fed13fa190d`. Source locations display lines 225–243. The no-model fallback is expected for this formal release; R5 rank fusion is not part of this app execution.

普通检索首条为混合精度教程的保存／恢复段落，来源按钮指向固定提交中的 `amp_recipe.py`，界面显示 225–243 行。无模型时降级符合正式发布包设计；本次应用执行不包含 R5 排名融合。

Historical [2026-10-08 CPU evidence](../cpu-demo-20261008/README.md) remains unchanged. The [diagnostic UI screenshots](../ui-demo/README.md) belong to a random diagnostic encoder and must not be relabeled as this release. The small evidence files here are suitable for Git; runtime extraction, installation directories and raw logs stay outside Git.

历史 CPU 证据保持原样；诊断界面截图属于随机诊断模型，不能改称本次正式包。本目录只保存适合 Git 的小型证据，解压包、安装目录及原始日志留在 Git 外。
