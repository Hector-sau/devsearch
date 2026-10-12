# Five-minute demo script / 五分钟演示脚本

**Current local demo — 2026-10-12 completed:** 966-update seed42 + fixed fusion is the owner-authorized local demo default. The second actual CPU attempt passes 80/80 checks; the independent actual-receipt audit passes 36 checks, and the owner controls are preserved externally. Actual external-manager/API/browser activation passes default, compare and explicit BM25 checks. [Operator guide](DEMO966_CPU.md), [acceptance summary](../reports/evidence/demo966-cpu-20261012/acceptance-summary.json) and [activation summary](../reports/evidence/demo966-cpu-20261012/demo-activation-summary.json) bind the exact scopes. BM25 remains an explicit baseline and failure fallback. Local `1.1.1.dev0` is provisional and is not a new public release.

**当前本地演示——2026-10-12 已完成：**966 次更新的 seed42 加固定融合已成为所有者授权的本地演示默认。第二次真实 CPU 运行通过 80 项检查，独立实际回执审核通过 36 项检查，所有者工具已保存外置盘。真实外置管理器／接口／浏览器的默认、对照及显式 BM25 激活检查通过。操作文档、验收概要及激活概要绑定准确范围。BM25 保留为显式对照和故障降级；本地 `1.1.1.dev0` 为暂定版，不是新的公开发布。

Begin in `rerank` mode, then select `compare` to show both lists from the same natural top 50 pool. State that the 966 model uses one training seed42 and development-set evidence; show its own measured receipt instead of the original R5 family mean or historical R5 latency. Use an authored demonstration query, source links, explicit BM25 mode and recoverable timeout/latched-model-failure behavior. No test or holdout input is used for this demonstration.

从 `rerank` 开始，再选择 `compare` 展示同一自然前 50 候选的两组排序。说明 966 使用一个训练种子 42 和开发集证据，展示其自身实测回执，不借用原 R5 系列均值或历史延迟。使用专门编写的演示问题、来源链接、显式 BM25 模式，并解释可恢复超时及模型故障锁定行为；演示不使用测试或留出集输入。

Actual browser captures: [default fused search](../reports/evidence/demo966-cpu-20261012/demo-default.jpg), [two-column comparison](../reports/evidence/demo966-cpu-20261012/demo-compare.jpg), [explicit BM25](../reports/evidence/demo966-cpu-20261012/demo-baseline.jpg). These are agent-operated demonstration records; the UI is restored to `rerank` after checking all modes.

实际浏览器截图见上述默认融合检索、双列对照及显式 BM25 链接。它们是 Agent 操作的演示记录；各模式核验后，界面已恢复为 `rerank`。

For latency, report both 150-request sequential TCP samples: first median 1396.083 ms/p95 1577.089 ms with failed overall schema handling; second median 581.042 ms/p95 635.736 ms with 80/80 checks passing. The second attempt's cold process-to-TCP startup was 58.978 seconds. Neither sample is a production SLA or evidence of deterministic speedup.

说明延迟时保留两组 150 请求串行 TCP 样本：第一次中位数 1396.083 毫秒／p95=1577.089 毫秒，完整运行因结构处理失败；第二次中位数 581.042 毫秒／p95=635.736 毫秒，80 项检查通过。第二次从冷进程到 TCP 监听为 58.978 秒。两组样本均不是生产 SLA，也不证明确定性提速。

The sections below preserve the published `v1.1.0` and earlier demonstration scopes. Their BM25 default, R5 seed42/epoch3 model and three-seed 202 results remain historical facts, not the configuration or score of the new 966 demo.

下文保留已发布 `v1.1.0` 及更早演示范围。其 BM25 默认、R5 seed42／第三轮模型及三种子 202 题结果仍为历史事实，不代表新 966 演示的配置或成绩。

## Published v1.1.0 demonstration / 已发布v1.1.0演示

Demonstrate BM25 as the evaluated default and original R5 seed42 + fixed fusion as an explicitly enabled experimental candidate. The UI calls the shared API, so both modes use the same server scoring, ordering and fallback implementation.

演示已评测的 BM25 默认，以及显式启用的原 R5 种子 42 加固定融合实验候选。界面调用共用接口，两种模式使用同一服务器评分、排序与降级实现。

Lead with the measured finding: **on the same sealed 202-question validation benchmark, each of the three R5 training seeds improves MRR@10 over BM25**. Mean `0.941062` versus `0.901416`, difference `+0.039645`; seeds 42/43/44 score `0.934076 / 0.944059 / 0.945050`. The application shown here uses seed 42, not the three-seed mean. [Results and uncertainty](../reports/RESULTS.md).

先介绍实测结论：**在同一份封存的 202 题验证基准上，三个 R5 训练种子均提高了相对 BM25 的 MRR@10**；均值、差值及各种子成绩如上。这里展示的应用使用种子 42，不能将三种子均值当作应用中单个模型成绩。结果与不确定性见报告。

The questions are synthetic and were repeatedly used during development. Preserve the separate historical 197-question test and new 59-question holdout scopes; the latter failed the frozen R5 candidate's two quality gates. Demonstration inputs are authored examples and the browser captures are agent-operated, with no human-user success claim.

问题为合成且曾反复用于开发；历史 197 题测试与新 59 题留出集分别保留，后者未通过固定 R5 候选的两项效果门槛。演示输入为专门编写的样例，浏览器记录由 Agent 操作，不宣称真人使用成功。

## Suggested five-minute presentation / 五分钟讲解顺序

| Time / 时间 | Show and explain / 展示与说明 |
| --- | --- |
| 0:00–0:40 | A normal BM25 search with exact source links / 普通 BM25 检索及准确来源链接 |
| 0:40–1:30 | Optional R5 candidate comparison; identify seed 42, natural top-50 and fixed fusion / 可选 R5 对照，说明种子 42、自然前 50 与固定融合 |
| 1:30–2:30 | Train-only negative mining, native PyTorch loop, export/reload and shared API / 仅训练集负采样、原生训练循环、导出重载及共用接口 |
| 2:30–3:30 | Same 202-question validation table and all three seed results / 同一 202 题验证表及三个种子成绩 |
| 3:30–4:20 | Historical test, new 59-question holdout failure and retained BM25 default / 历史测试、新 59 题失败与保留 BM25 默认 |
| 4:20–5:00 | Empty/invalid/fallback behavior, public fixture quick start and evidence links / 空结果、非法输入、降级、公开样例及证据链接 |

This is a proposed spoken sequence; its duration is not a timed rehearsal. Original DevSearch code uses [Apache-2.0](../LICENSE), with upstream terms preserved in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md). A public clone can run [the small CPU example](REPRODUCIBILITY.md); real-corpus/model demonstrations additionally require the owner-held runtime inputs and model assets.

上表为讲解安排，并非计时彩排。原创代码采用 Apache-2.0，上游条款由第三方说明保留。公开克隆可运行小型 CPU 示例；真实语料／模型演示还需要所有者持有的运行输入与模型资产。

> **Recorded BM25 demonstration (2026-10-09):** the evaluated formal release was restored after its backup SHA-256 matched the delivery manifest. Startup replay, five live API scenarios, four Streamlit component scenarios and four independently rendered browser states passed. See [actual BM25 demonstration evidence](../reports/evidence/release-demo-20261009/README.md). These receipts keep their original source identities; the separate R5 candidate capture below has its own scope. / **已记录 BM25 演示：**恢复前核对备份哈希，启动回放、五项接口、四项组件及四种独立浏览器状态通过；回执保留原源码标识，下方 R5 候选截图另有范围。

## Real R5 candidate demonstration — 2026-10-10 / 真实 R5 候选演示

Frozen seed 42/epoch3 weights are restored from a verified external archive. The candidate API/UI uses natural BM25 top-50 and fixed rank fusion (`alpha=0.5`, `rank_constant=60`), with raw logits and fusion scores shown separately. Installed-wheel startup/reload and offline/online ordering passed; server and Mac orders match for all 59 sealed queries. It requires `--allow-candidate`; the `1.1.0` application preparation retains BM25 as the default. [Operator configuration](R5_INTEGRATION.md) · [actual comparison capture](../reports/evidence/r5-independent-20261010/comparison.jpg).

冻结种子 42、第三轮权重已从核验后的外置压缩包恢复。候选接口／界面使用自然 BM25 前 50 与固定排名融合，原始 logit 和融合分数分别展示。安装包启动／重载及离在线排序通过，服务器与 Mac 的 59 个封存问题排序一致。服务须显式允许候选，`1.1.0` 应用准备保持 BM25 默认；配置与实际对照截图见上述链接。

Show the authored demonstration query `How do I save and resume a model checkpoint with optimizer state?` in compare mode: both ranked lists, pinned source links and candidate identity are visible. Actual API checks also passed empty lexical retrieval and invalid `sources` rejection (HTTP 422); the browser captured the real comparison view. These are agent-operated demonstrations, not human-user testing or ranking-quality measurements.

在 compare 模式展示专门编写的演示问题，界面可见两组排序、固定来源链接及候选身份。真实接口还通过词法无匹配的空结果与无效 `sources` 拒绝检查（HTTP 422）；浏览器已保存真实对照页面。这些是 Agent 操作演示，不是真人测试或排序效果测量。

State the independent result honestly: on 59 new synthetic questions, R5 fused MRR@10 was 0.573278 versus BM25's 0.581302, and pooled nDCG@10 was 0.526748 versus 0.592521. Both quality gates failed, so BM25 remains the formal default. Warm local CPU p95 was 622.065 ms under a separate fixed 150-request sequential in-process benchmark; it is not GPU latency or a production SLA. [Full result and uncertainty](../reports/r5-independent-holdout-2026-10-10.md).

如实说明独立结果：59 个新合成问题上，R5 融合 MRR@10 为 0.573278，BM25 为 0.581302；池化 nDCG@10 为 0.526748，BM25 为 0.592521。两个质量门槛均失败，正式默认保持 BM25。另一个固定 150 请求串行进程内基准测得本机 CPU 预热后 p95 为 622.065 毫秒，不代表 GPU 延迟或生产 SLA；完整结果与不确定性见报告。

## Evaluated BM25 default demonstration / 已评测 BM25 默认演示

The release is `release-ce3160fc579a52ae2ec30d15412853eb`, evaluated and model-free. Its original export and evaluation belong to `e3168ca`; the UI/API implementation used for that recorded demonstration is identified by file SHA-256 in `demo-record.json`. Version and documentation changes in the release-preparation working tree do not change the historical evaluation's source identity. This is an agent-operated local demonstration, not a human usability study or a new ranking/latency benchmark.

演示发布包已评测、默认 BM25、不含模型；原导出与评测仍归属 `e3168ca`。该次已记录演示所用界面和 API 的实现文件哈希保存在回执中，版本与文档整理不改写历史评测来源。这是 Agent 执行的本机演示，不是真人可用性研究，也不是新的排序质量或延迟基准。

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

Present D1–D4 as authored demonstration inputs. Use [RESULTS](../reports/RESULTS.md) for separately scoped formal-test, R5 validation and independent-holdout results. The evaluated BM25 runtime bundle has no neural model; the R5 candidate shown above is a separate runtime bundle and does not change the application default.
将 D1–D4 明确称为专门编写的演示输入；正式测试、R5 验证及独立留出集结果分开引用报告。已评测 BM25 运行包没有神经模型；上方真实 R5 候选为独立运行包，不改变应用默认模式。

## Historical demonstration record / 历史演示记录

<details>
<summary>Original 2026-10-08 demonstration record / 原始演示记录（历史）</summary>


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

</details>
