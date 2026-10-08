# Five-minute demo script / 五分钟演示脚本

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
