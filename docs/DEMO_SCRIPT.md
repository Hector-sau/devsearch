# Five-minute demo script / 五分钟演示脚本

> **Current v1 demonstration (2026-10-09):** the evaluated formal BM25 release was restored after its backup SHA-256 matched the delivery manifest. The current release-preparation checkout passed startup replay, five live API scenarios and four actual Streamlit component scenarios. A separate browser pass verified all four rendered states and pinned source-link buttons; see [the current demonstration evidence](../reports/evidence/release-demo-20261009/README.md) for its exact status and captures. R5+fusion remains an offline experimental candidate. / 当前 v1 演示使用已核验并恢复的正式 BM25 发布包；发布整理代码通过启动回放、五项真实 API 场景与四项 Streamlit 组件场景，独立浏览器验收四种状态及固定来源链接全部通过，实际截图见当前证据。R5 融合仍为离线实验候选。

## Current v1 execution / 当前 v1 执行

The release is `release-ce3160fc579a52ae2ec30d15412853eb`, evaluated and model-free. Its original export and evaluation belong to `e3168ca`; the current UI/API implementation is recorded by file SHA-256 in `demo-record.json`. Version and documentation changes in the release-preparation working tree do not change the historical evaluation's source identity. This is an agent-operated local demonstration, not a human usability study or a new ranking/latency benchmark.

演示发布包已评测、默认 BM25、不含模型；原导出与评测仍归属 `e3168ca`。当前界面和 API 的实现文件哈希保存在回执中，版本与文档整理不改写历史评测来源。这是 Agent 执行的本机演示，不是真人可用性研究，也不是新的排序质量或延迟基准。

1. Verify the private backup archive's SHA-256 against the delivery manifest before extraction: `b8ee3cd6817a3fcf97d60d87193b76d485743ecc4737c010f74d2fd489bb0e60` (378,786,182 bytes). Extract only `backup/release/` into an independent runtime root; the extracted release is 17,447,914 bytes. No model or checkpoint download is needed for this demo.
   解压前核对私有备份包完整哈希，再仅将 `backup/release/` 恢复到独立运行目录；本次演示不需要下载模型或训练断点。
2. Use Python 3.11 or 3.12 and the project's `serve` dependencies. The recorded API used the existing Python 3.11 environment; Streamlit 1.65.0 and its UI dependencies were installed in a separate runtime target directory, leaving the training environment unchanged.
   使用 Python 3.11 或 3.12 及项目的 `serve` 依赖。实测 API 使用现有 Python 3.11，Streamlit 及界面依赖单独装在运行目录中，训练环境保持原样。
3. Copy `configs/serve.toml` to the runtime, select an unused loopback port, and keep its CPU profile and deadline unchanged. Start the API first, then the UI with `DEVSEARCH_API_URL` pointing at it.
   复制服务配置到运行目录，仅选择空闲本地端口，保持 CPU 配置与时限不变；先启动 API，再启动指向它的界面。

From the repository root, after restoring the release and installing the serving dependencies:
从仓库根目录执行，前提是已恢复发布包并安装服务依赖：

```bash
export DEVSEARCH_RUNTIME_ROOT=/absolute/path/to/independent/runtime
export PYTHONPATH="$PWD/src"
python -m devsearch.serve --config "$DEVSEARCH_RUNTIME_ROOT/serve-demo.toml" \
  --manifest "$DEVSEARCH_RUNTIME_ROOT/backup/release" \
  --runtime-root "$DEVSEARCH_RUNTIME_ROOT" --check-only
python -m devsearch.serve --config "$DEVSEARCH_RUNTIME_ROOT/serve-demo.toml" \
  --manifest "$DEVSEARCH_RUNTIME_ROOT/backup/release" \
  --runtime-root "$DEVSEARCH_RUNTIME_ROOT"
# A second terminal; use the API port selected in serve-demo.toml.
DEVSEARCH_API_URL=http://127.0.0.1:8810 python -m streamlit run src/devsearch/serving/ui.py \
  --global.developmentMode false --server.address 127.0.0.1 --server.port 8811 \
  --server.headless true --browser.gatherUsageStats false
```

The local verification used ports 8810/8811. When Streamlit is installed with `pip --target`, include that target directory before `src` in the UI's `PYTHONPATH`; this target is a local tooling detail, not a distributable model environment.
本地验证使用端口 8810/8811；若采用 `pip --target` 安装 Streamlit，将目标目录放在界面 `PYTHONPATH` 中 `src` 之前。该目录只是本机界面工具，不属于模型运行环境交付。

| Scenario / 场景 | Expected behavior / 预期行为 | Actual evidence / 实际证据 |
| --- | --- | --- |
| Startup / 启动 | all release files and recorded smoke replay verified / 校验发布文件与回放 | exit 0; 8 replay queries passed / 退出 0、8 条回放通过 |
| D1 baseline / 普通检索 | 50 candidates, 5 visible results, pinned upstream source links / 50 个候选、显示 5 条及固定来源链接 | live HTTP 200 and Streamlit components passed / 真实 API 与组件通过 |
| D3 empty / 空结果 | explain no lexical match / 解释为无词法匹配 | live HTTP 200, `empty`; component info banner / 空状态及提示通过 |
| D4 source `nope` / 无效筛选 | reject the request with a readable error / 拒绝并说明原因 | live HTTP 422, `invalid_request`; component error banner / 接口拒绝及界面提示通过 |
| D2 rerank or compare / 重排或对照 | keep BM25 results, show `rerank_unavailable` / 保留 BM25 并提示不可重排 | live HTTP 200, `degraded`; compare component warning / 降级与提示通过 |
| Rendered browser / 浏览器渲染 | visible results, links and all four states / 真实页面及四种状态 | four CUA scenarios passed; actual screenshots saved / 四项浏览器场景通过、实际截图已保存 |

Present D1–D4 as authored demonstration inputs. Use [RESULTS](../reports/RESULTS.md) for the formal test results and later R5 validation results. Preserve the current BM25 default and state that no R5 fusion is running in this UI.
将 D1–D4 明确称为专门编写的演示输入；正式测试和后续 R5 验证结果以报告为准。界面默认使用 BM25，展示时说明其尚未运行 R5 融合。

## Historical demonstration record / 历史演示记录

The remaining sections preserve the first 2026-10-08 demo and its original limitations. Their “not executed” and “pending” statuses describe that historical run; the current evidence above supersedes them for the v1 demonstration only.
下方保留 2026-10-08 首次演示的原始说明，其中“未执行／待完成”属于当时状态；当前 v1 演示以本页上方和新证据为准。

**Reference commit `e3168ca`.** Commands below use the real arguments of that commit (`python -m devsearch.serve`, `python -m devsearch.export`,
`streamlit run src/devsearch/serving/ui.py`); code behavior is explained in [the code walkthrough](CODE_WALKTHROUGH.md).

Two releases must never be confused:

| Release | What it is | Status |
| --- | --- | --- |
| **Formal default (BM25)** | `release-ce3160fc579a52ae2ec30d15412853eb`, exported by the first formal campaign, **evaluated**, defaults to BM25, ships **no model** | **verified for the demo:** taken from the backup archive after its local SHA-256 matched the delivery manifest (`b8ee3cd6…`); `serve --check-only` passed on CPU (2026-10-08) |
| **Experimental reranker** | a *candidate* release built locally from the campaign's full fine-tuning model bundle; served only with `--allow-candidate` | not built in this demo (needs the `full` archive's model bundle, its SHA-256 check and the `ml` extra); it did **not** beat BM25 and may be shown only as an experiment |

**Screenshots, responses and timings:** only those actually captured from these releases may be shown. API responses of the formal
release were captured on CPU (see *Verified on CPU* below); **no screenshot of the rendered page exists yet** because no compatible
environment has Streamlit installed. The screenshots in [reports/evidence/ui-demo](../reports/evidence/ui-demo/README.md) come from a random
diagnostic encoder (2026-10-06) and may be used only to illustrate failure handling, labelled "diagnostic".

两个发布包必须区分：正式默认（BM25，已评测，无模型，已在 CPU 上完成演示验证）与实验性重排（候选包，须 `--allow-candidate`，未优于 BM25，本次未构建）。界面截图因缺少 Streamlit 环境尚未取得；已有截图来自随机诊断模型，只能用于说明异常处理。

## Demonstration inputs / 演示输入

Written **for this demo**: not test-set questions, not user queries, not evidence of quality.
以下为**演示输入**，专为演示编写，不属于测试集，也不是真实用户问题。

| # | Demonstration input | Purpose |
| --- | --- | --- |
| D1 | `How do I save and later resume a model checkpoint with the optimizer state?` | ordinary BM25 search with source links |
| D2 | `Why does model.eval() change the output of dropout layers?` | compare mode: on the formal release it shows the `rerank_unavailable` fallback; on the experimental release, both orders |
| D3 | `qzxv flarp` | an empty result reported honestly (neither token occurs in the corpus) |
| D4 | the D1 question with the source filter set to `nope` | a rejected request (HTTP 422) |

## Preconditions / 前置条件

1. The code of `e3168ca` (exported with `git archive`, not a worktree) and Python ≥ 3.11 with FastAPI, uvicorn and httpx for the API;
   Streamlit for the page; the `ml` extra only for the experimental reranker. The CPU demo used Python 3.11.16 (FastAPI 0.141.1, uvicorn
   0.52.4, httpx 0.28.1) on an Apple M4; Streamlit was not available.
2. The formal release unpacked under a runtime root of your own: copy the backup archive, check its SHA-256 against the delivery manifest
   (`b8ee3cd6…0e60`), extract `backup/release/`. Never work inside the shared download directory.
3. A copy of `configs/serve.toml` with a free port (the demo used 18080; profile `cpu-diagnostic`, deadline 2.0 s unchanged). Rerank
   latency on a CPU has **not** been measured; the 147 ms p95 is on the RTX 4090 and does not transfer.

## Commands / 命令

```bash
# Verified on CPU (2026-10-08). <demo dir> holds: code/ (git archive of e3168ca), runtime/backup/release/, serve-demo.toml (port 18080)
cd <demo dir>
export PYTHONPATH=$PWD/code/src DEVSEARCH_RUNTIME_ROOT=$PWD/runtime   # serve refuses a release outside the runtime root
python3.11 -m devsearch.serve --config serve-demo.toml --manifest runtime/backup/release --runtime-root runtime --check-only   # exit 0
python3.11 -m devsearch.serve --config serve-demo.toml --manifest runtime/backup/release --runtime-root runtime              # listens on 127.0.0.1:18080
curl -s -X POST http://127.0.0.1:18080/search -H 'Content-Type: application/json' -d '{"query": "<demonstration input>", "mode": "baseline"}'

# Not executed: the page (needs Streamlit in a Python >= 3.11 environment)
DEVSEARCH_API_URL=http://127.0.0.1:18080 streamlit run code/src/devsearch/serving/ui.py --server.address 127.0.0.1 --server.port <free port>

# Not executed: experimental reranker (needs the full archive's model bundle after its SHA-256 check, and the ml extra)
python -m devsearch.export --config <export.toml with device profile cpu-diagnostic> \
  --index runtime/backup/release/index/index-manifest.json --checkpoint <full fine-tuning model bundle directory> \
  --output runtime/<new candidate release directory>
python -m devsearch.serve --config serve-demo.toml --manifest runtime/<new candidate release directory> --allow-candidate
```

`--check-only` runs every startup check (file hashes, activation rules, recorded smoke replay) and exits without binding a port. `export`
reloads the model on the configured device and records a smoke test before it publishes the release; `serve` replays it.

## Verified on CPU / CPU 实测（2026-10-08）

Formal release `release-ce3160fc…` served from `e3168ca` on `127.0.0.1:18080`, Apple M4, by Claude (agent). Record:
[reports/evidence/cpu-demo-20261008](../reports/evidence/cpu-demo-20261008/README.md) (commands, exit codes, HTTP status, short responses; no passage text).

| # | Scenario | Result |
| --- | --- | --- |
| 1 | `serve --check-only` | **pass**: exit 0, `startup_checks_passed`, status `evaluated`, default `baseline`, smoke replay passed (8 queries) |
| 2 | D1, normal search | **pass**: HTTP 200, `ok`, 50 candidates, 5 shown; top result *Automatic Mixed Precision › Saving/Resuming*, link to `amp_recipe.py` lines 225–243 at `11512db7…` |
| 3 | D3, no match | **pass**: HTTP 200, `empty`; page text "No passages matched … no lexical match, not that no answer exists" |
| 4 | D4, invalid source | **pass**: HTTP 422 `invalid_request`, "unknown source 'nope'; available: pytorch-tutorials" |
| 5 | D2 with `rerank` and with `compare` on the BM25 release | **pass**: HTTP 200, `degraded`, effective mode `baseline`, warning `rerank_unavailable` "This release has no reranker; showing BM25 results." |
| 6 | UI shows these states | **partly**: the page's own functions (`describe_response`, `describe_error`, `call_api`) produce the right banner for every state above and "Cannot reach the DevSearch API" for a closed port; **the rendered page and its screenshots were not executed** (no Streamlit) |
| — | Experimental reranker | **not executed** |

Executable steps of the script took 0.74 s of command time on the M4 (a demo record, not a performance result). The spoken five-minute
script has **not** been rehearsed aloud. Observation, not a defect: for D2 BM25 ranks save/load sections first, a lexical match on
"model" — a fair example of what BM25 cannot judge.

正式 BM25 发布包已在本机 CPU 上实际演示：启动检查、正常搜索、无匹配、无效过滤、重排降级五项通过；界面逻辑函数验证通过，但界面渲染与截图因缺少 Streamlit 未执行；实验性重排未执行；演示耗时仅为演示记录。

## Script / 讲稿

| Time | Show | Say (key points) | Evidence |
| --- | --- | --- | --- |
| 0:00–0:30 | title | "DevSearch finds the passage in the official PyTorch tutorials that answers an English question, with a link to the exact source lines. It returns evidence, not generated answers." | — |
| 0:30–1:30 | D1 on the **formal BM25 release** | BM25 over 4,259 passages of `pytorch/tutorials` pinned at one commit; each result links to the file and lines; the response records corpus, index and candidate-set ids. | **verified on CPU** (scenario 2); screenshot **pending** |
| 1:30–2:10 | D2 in compare mode on the formal release, then D3 and D4 | The formal release ships no model, so compare shows the BM25 list with "Reranking was not applied"; empty results mean "no lexical match"; a bad filter is rejected with a readable message. | **verified on CPU** (scenarios 3–5); screenshots **pending** |
| 2:10–3:00 | (optional) D2 on the **experimental** candidate release | "This is the fine-tuned reranker from the first campaign. It is not the default: on validation it did not beat BM25 (MRR@10 0.847 vs 0.851), and on the test set it was not better either." If the 2 s deadline is exceeded on this machine, the page falls back to BM25 with a warning. | release not built yet; output **pending** |
| 3:00–3:50 | review-flow diagram ([DIAGRAMS](DIAGRAMS.md) section 2) | All 9,852 fixed test candidates were judged blind before training: second review, escalation above 10% disagreement, independent adjudication. **Agent judgments only.** | done |
| 3:50–4:30 | the campaign's stages | One launcher ran check, prepare (with the full test suite: 848 tests, 0 skipped on the GPU host), diagnose (real-model crash and resume) and the campaign on an RTX 4090; test data were read only after validation-only selection. | verified (evidence index §9) |
| 4:30–5:00 | results table ([experiment report §5.2](../reports/experiment-report-draft.md)) | Test MRR@10: BM25 0.8925, full fine-tuning 0.8735 (difference −0.019, interval spans zero), frozen 0.2024. **The predeclared rule kept BM25 as the default** — a negative result, reported as such. | verified |

## Failure handling during the demo / 演示中的异常处理

| Symptom | Meaning | What to say / do |
| --- | --- | --- |
| `serve --check-only` exits 1 | a hash, smoke replay or activation rule failed | do not demo that release; it refuses by design |
| "Cannot reach the DevSearch API" | the API is not running or `DEVSEARCH_API_URL` is wrong | start `serve`; the UI never loads a model itself |
| HTTP 429 | another search is running (one slot, no queue) | wait and retry; explain the design |
| "Reranking was not applied" / `rerank_unavailable` | the release has no model (formal release) | expected on the formal default |
| `rerank_timeout` | reranking exceeded the 2 s deadline on this machine | expected possibility on CPU; BM25 results are complete |
| `--allow-candidate` refused or missing | a candidate release cannot be served without it | add the flag only for the experimental release, and say so |

## Rules while presenting / 演示规则

- Say "demonstration input" when typing D1–D4.
- Show only numbers that are in the evidence index; for the second round say "exploratory experiment; complete evidence pending delivery".
- Do not show server paths, credentials or the private dataset.
