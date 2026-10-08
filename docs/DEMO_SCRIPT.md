# Five-minute demo script / 五分钟演示脚本

**Reference commit `e3168ca`.** Code behavior cited here is explained, with links, in [the code walkthrough](CODE_WALKTHROUGH.md).
**Pending until GPU evidence arrives:** the final release to demonstrate, every screenshot of it, every reranked result, every timing and
every quality number. Until then this script can be rehearsed only with a baseline (BM25) release or the diagnostic release; the existing
screenshots in [reports/evidence/ui-demo](../reports/evidence/ui-demo/README.md) come from a **random diagnostic encoder** and may be shown only
for failure-path behavior, labelled as such.

参考提交 `e3168ca`。最终发布包、截图、重排结果、耗时与效果数字均待 GPU 证据到位后填写；在此之前只能用 BM25 发布包或诊断发布包排练，已有截图来自随机诊断模型，只能用于展示异常处理并须注明。

## Demonstration inputs / 演示输入

These questions were **written for this demo**. They are not test-set questions, not user queries and not evidence of quality.
以下问题为**演示输入**，专为演示编写，不属于测试集，也不是真实用户问题，不能作为效果证据。

| # | Demonstration input | Purpose |
| --- | --- | --- |
| D1 | `How do I save and later resume a model checkpoint with the optimizer state?` | ordinary search; source links |
| D2 | `Why does model.eval() change the output of dropout layers?` | compare BM25 and reranked order (rerank part pending) |
| D3 | `qzxv flarp` | an empty result that is reported honestly |
| D4 | the D1 question with the source filter set to `nope` | a rejected request (HTTP 422) |

## Setup (before the audience joins) / 准备

On the presenting machine, from a checkout of the reference commit with the `ml` and `serve` extras installed:

```bash
# <release> is the exported release directory: PENDING (the final release comes from the GPU campaign's export_release stage)
python -m devsearch.serve --config configs/serve.toml --manifest <release> --check-only     # every startup check, then exit
python -m devsearch.serve --config configs/serve.toml --manifest <release>                  # loopback 127.0.0.1:8000
streamlit run src/devsearch/serving/ui.py --server.address 127.0.0.1 --server.port 8501    # second terminal
```

`configs/serve.toml` uses the `cpu-diagnostic` device profile and a 2-second search deadline; a model release must be served on the profile it
was verified for. A release without a final evaluation needs `--allow-candidate` and must be announced as a candidate.

## Script / 讲稿

| Time | Show | Say (key points) | Evidence status |
| --- | --- | --- | --- |
| 0:00–0:30 | title | "DevSearch finds the passage in the official PyTorch tutorials that answers an English question, with a link to the exact source lines. It returns evidence, not generated answers." | — |
| 0:30–1:30 | D1 in **baseline** mode | Results come from BM25 over 4,259 passages of `pytorch/tutorials` pinned at one commit; each result links to the file and line range. Mention that the response records the corpus, index and candidate-set ids. | behavior verified on CPU; final screenshot **pending** |
| 1:30–2:20 | D2 in **compare** mode | The reranker, a MiniLM cross-encoder fine-tuned in native PyTorch, reorders BM25's top 50; both lists are shown with their ranks. | reranked order, logits, timing: **pending (GPU)** |
| 2:20–3:00 | D3, D4, then a rerank request against a release that cannot rerank | Empty result is "no lexical match", not "no answer exists". A bad filter is rejected with a readable message. If reranking is unavailable, times out or produces a bad score, the page shows the complete BM25 list with a warning; nothing partial is ever shown. | behavior verified on CPU (diagnostic screenshots exist, labelled); final screenshots **pending** |
| 3:00–3:50 | the review-flow diagram ([DIAGRAMS](DIAGRAMS.md) section 2) | All 9,852 fixed test candidates were judged blind before any training, under a declared protocol: second review, escalation above 10% disagreement, independent adjudication. All judgments are agent judgments; no human review. | done (audit exit 0) |
| 3:50–4:30 | the status diagram (section 1) | One launcher runs check, prepare (including the full test suite), diagnose (a real-model crash and resume), run and status. Test data can only be read after validation-only selection has committed. | CPU evidence done; GPU host in progress (reported) |
| 4:30–5:00 | results slide | Test MRR@10/nDCG@10 for BM25, frozen and full with confidence intervals, and the default-mode decision. "BM25 stays the default" is a valid outcome. | **pending (GPU)**: show the placeholder, not estimates |

## Rules while presenting / 演示规则

- Say "demonstration input" when typing D1–D4; never present them as user queries or test questions.
- Never show a number that is not in a receipt; for pending items say "pending" and point to the experiment report.
- If the API is down or busy, the page says so (`Cannot reach the DevSearch API`, HTTP 429 when another search is running): explain the
  behavior instead of retrying silently.
- Do not show server paths, credentials or the private dataset.
