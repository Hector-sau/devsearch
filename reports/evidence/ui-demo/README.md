# API and UI demonstration, 2026-10-06 (CPU, diagnostic)

Screenshots of the Streamlit page (`src/devsearch/serving/ui.py`) in a real Chromium session against the local FastAPI service, plus the
API probes behind them. **What this shows:** the user-facing paths work on CPU and degrade explicitly. **What it does not show:** any ranking
quality. The corpus is the real `pytorch/tutorials@11512db7` index (4,259 passages); the "model" is a random tiny encoder from a diagnostic
CPU campaign (`candidate` release, served with `--allow-candidate`), so its reranked order is meaningless by design. No GPU, no MiniLM.

Releases exported with `export` and reload-verified (`smoke_replay: passed`; the model one also `model_parity_after_reload: passed`):
a **baseline** release (`release-bb2956bf…`, no model) and a diagnostic **candidate** release (`release-ae3754d9…`).

| Screenshot | Scenario | Observed |
| --- | --- | --- |
| `01-baseline-search.png` | baseline release, a real question | 50 candidates in 1.7 ms; each result shows title, section, `pytorch-tutorials @ 11512db7… - path lines a-b`, BM25 rank and score, and a source link to that exact commit and file on github.com |
| `02-baseline-empty.png` | nonsense query | "No passages matched ... no lexical match, not that no answer exists" (API: `status: empty`, HTTP 200) |
| `03-baseline-rerank-degraded.png` | `rerank` asked of a baseline-only release | warning "Reranking was not applied ... No partial reranking is shown"; `rerank_unavailable`; shown mode `baseline` (API `status: degraded`) |
| `04-baseline-unknown-source.png` | source filter `nope` | "The search request was rejected: unknown source 'nope'; available: pytorch-tutorials" (HTTP 422, no stack trace) |
| `05-baseline-empty-query.png` | empty query | "query must not be empty" (HTTP 422) |
| `06-model-compare.png` | diagnostic model release, `compare` | baseline and reranked lists side by side with baseline rank, reranked rank, BM25 score and uncalibrated logit; rerank 64 ms on CPU |
| `07-model-rerank.png` | diagnostic model release, `rerank` | reranked list only (random encoder: a navigation fragment from BM25 rank 47 comes first: meaningless, as expected) |
| `08-model-deadline-fallback.png` | model release served with a 0.03 s deadline | reranking exceeded the deadline: `rerank_timeout`, BM25 results shown, "No partial reranking is shown" |
| `09-api-down.png` | API not running | "Cannot reach the DevSearch API ... Start the API service, or set DEVSEARCH_API_URL" |

API probes (curl): malformed JSON, an unknown mode and an unknown source each return HTTP 422 with `{"error": {"code": "invalid_request", ...,
"retryable": false}}`; `/health` reports `rerank_ready` false for the baseline release and true for the model release.

Reproduce: `export`, then `serve` (loopback only), then `streamlit run src/devsearch/serving/ui.py --server.address 127.0.0.1`
with `DEVSEARCH_API_URL`. The browser capture used Playwright in a separate tooling environment, not part of the project dependencies.
