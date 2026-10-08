# Five-minute demo script / 五分钟演示脚本

**Reference commit `e3168ca`.** Commands below use the real arguments of that commit (`python -m devsearch.serve`, `python -m devsearch.export`,
`streamlit run src/devsearch/serving/ui.py`); code behavior is explained in [the code walkthrough](CODE_WALKTHROUGH.md).

Two releases must never be confused:

| Release | What it is | Status |
| --- | --- | --- |
| **Formal default (BM25)** | `release-ce3160fc579a52ae2ec30d15412853eb`, exported by the first formal campaign, **evaluated**, defaults to BM25, ships **no model** | produced and reload-verified on the server (verified here); local copy inside the backup archive, **still downloading** |
| **Experimental reranker** | a *candidate* release built locally from the campaign's full fine-tuning model bundle; served only with `--allow-candidate` | not built; the model bundle is in the `full` archive, **still downloading**. It did **not** beat BM25 and is shown only as an experiment |

**Screenshots, responses and timings:** only those actually captured from these releases may be shown; none exist yet. The screenshots in
[reports/evidence/ui-demo](../reports/evidence/ui-demo/README.md) come from a random diagnostic encoder (2026-10-06) and may be used only to
illustrate failure handling, labelled "diagnostic".

两个发布包必须区分：正式默认（BM25，已评测，无模型）与实验性重排（候选包，须 `--allow-candidate`，未优于 BM25，仅作实验展示）。截图、响应与耗时只使用实际取得的证据；已有截图来自随机诊断模型，只能用于说明异常处理。

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

1. A checkout of `e3168ca` and a Python 3.12 environment with the `serve` extra (FastAPI, uvicorn, httpx, Streamlit); the experimental
   reranker also needs the `ml` extra.
2. The download and the **local SHA-256 check** of the backup archive (formal release) and, for the experiment, the `full` archive
   (model bundle) have finished — **pending**. The runtime root that holds the unpacked release: `<runtime root>` (pending).
3. `configs/serve.toml` as shipped: profile `cpu-diagnostic`, loopback `127.0.0.1:8000`, search deadline 2.0 s. Rerank latency on the
   presenting machine's CPU has **not** been measured; the measured 147 ms p95 is on the RTX 4090 and does not transfer.

## Commands / 命令

```bash
export DEVSEARCH_RUNTIME_ROOT=<runtime root>          # must contain the release (serve refuses anything outside it)

# Formal default: the evaluated BM25 release
python -m devsearch.serve --config configs/serve.toml --manifest <runtime root>/<release-ce3160fc… directory> --check-only
python -m devsearch.serve --config configs/serve.toml --manifest <runtime root>/<release-ce3160fc… directory>
streamlit run src/devsearch/serving/ui.py --server.address 127.0.0.1 --server.port 8501   # second terminal

# Experimental reranker (optional; build once, after the full archive is downloaded and checked)
python -m devsearch.export --config <export.toml with device profile cpu-diagnostic> \
  --index <runtime root>/<…>/index/index-manifest.json --checkpoint <full fine-tuning model bundle directory> \
  --output <runtime root>/<new candidate release directory>
python -m devsearch.serve --config configs/serve.toml --manifest <runtime root>/<new candidate release directory> --allow-candidate
```

`--check-only` runs every startup check (file hashes, activation rules, recorded smoke replay) and exits without binding a port. `export`
reloads the model on the configured device and records a smoke test before it publishes the release; `serve` replays it.

## Script / 讲稿

| Time | Show | Say (key points) | Evidence |
| --- | --- | --- | --- |
| 0:00–0:30 | title | "DevSearch finds the passage in the official PyTorch tutorials that answers an English question, with a link to the exact source lines. It returns evidence, not generated answers." | — |
| 0:30–1:30 | D1 on the **formal BM25 release** | BM25 over 4,259 passages of `pytorch/tutorials` pinned at one commit; each result links to the file and lines; the response records corpus, index and candidate-set ids. | behavior verified; screenshot **pending** |
| 1:30–2:10 | D2 in compare mode on the formal release, then D3 and D4 | The formal release ships no model, so compare shows the BM25 list with "Reranking was not applied"; empty results mean "no lexical match"; a bad filter is rejected with a readable message. | behavior verified on CPU; screenshots **pending** |
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
