# Review-bank execution proposal (needs budget approval) / 审查库执行方案（需批准预算）

Status 2026-10-06: **proposal, nothing launched, nothing billed.** The tooling is built and tested against a local fake API
(`scripts/judge_api.py`, `tests/tooling/test_judge_api.py`). Approval needed from you: the budget below and the judge model.

状态：**仅为方案，未启动、未产生任何费用。**工具已完成并用本地假接口测试。需要您批准：下述预算与审查模型。

## 1. What is judged / 判定范围（不变）

The declared protocol, unchanged: the blinded pool of **9,852 items** (`review-request.jsonl`, SHA-256 `e872f940…`), **two independent
primary passes** (A and B) over every item, then **adjudication** of every item the two passes do not settle (disagreement or `unsure`).
Nothing is sealed until `devsearch.review audit` reports complete coverage; missing items are never filled in.

- Batches: the existing judge batches, 40 compact blinded items each (item id, question, title, section, text), seeded order per pass;
  247 requests per pass. Pass B sees the items in a different order and grouping from pass A, in separate requests with no shared context.
- Criteria: `scripts/realdata/prompts/judge-candidate-v1.md`, verbatim (the file-handling lines are replaced by the API's input/output).
  Recorded revision per judgment: `judge-candidate-v1-api:3e159a182d56d770|<model>|effort=<effort>` (adjudicator: `judge-adjudicator-v1-api:…`).
- Labels: every judgment is imported as `reviewer.type = agent`. Pilot and calibration judgments use `pilot-*` reviewer ids, which `audit` refuses.

**Precondition:** the pinned MiniLM tokenizer must first be shown to reproduce this exact pool (the frozen review-request hash). If it does
not, the pool changes under a recorded protocol revision and judging it now would be wasted. Model access is therefore resolved first; it
costs nothing (section 6).

## 2. Platform, model and billing basis / 平台、模型与计费依据

| | Proposal |
| --- | --- |
| Platform | Claude API (Anthropic first-party), **Message Batches API**: asynchronous, billed at **50% of list prices** |
| Model | `claude-opus-5-5` (the current default Opus; the agent pilots ran on the same model family), effort `medium` (its default) |
| Alternative (your choice) | `claude-sonnet-5-5`: half the price; agreement not yet measured for this task, so the calibration step would run on it too |
| Billing basis | per token. List prices (cached 2026-09-25, confirm on the pricing page): Opus 5.5 $4 input / $20 output per million tokens; Sonnet 5.5 $2 / $10. Batch: half. Output includes the model's thinking tokens |
| Output format | JSON schema (structured outputs), so no answer is lost to formatting |

Context overhead is the main saving against the agent-based estimate of the earlier pilots (34 million tokens): a direct API request
carries only the criteria (about 500 tokens) and the 40 compact items, instead of an agent session's tools, file reads and replies.

## 3. Estimate / 预计用量（估算，未测量）

From the real pool's character counts (8.7 million characters of item text; 3 characters per token assumed, conservative):

| | Input tokens | Output tokens (assumed) | Cost, Opus 5.5 batch | Cost, Sonnet 5.5 batch |
| --- | --- | --- | --- | --- |
| Pass A (247 requests) | 3.06 M | 1.18 M (120 per item) | $17.9 | $9.0 |
| Pass B | 3.06 M | 1.18 M | $17.9 | $9.0 |
| Adjudication (about 300–600 items, effort `high`) | ~0.2 M | ~0.2 M | ~$3 | ~$1.5 |
| **Central estimate** | **~6.3 M** | **~2.6 M** | **~$39** | **~$20** |
| If output is 400 tokens per item (heavy thinking) | ~6.3 M | ~8.2 M | ~$95 | ~$48 |

The output figure is the uncertain one (thinking at `medium` effort is not yet measured for this task), hence the calibration step.

## 4. Hard limit, guard and checkpoints / 硬上限、保护与进度

1. **Hard cap, enforced by the platform:** create a dedicated workspace in the Claude Console (for example `devsearch-review`) with a
   **spend limit equal to the approved budget**, and an API key from that workspace only. Requests beyond the limit are rejected by
   the platform, whatever the script does. **Proposed limit: $100** (covers the high case for Opus 5.5).
2. **Guard in the runner (second line):** `--budget-usd` per pass. Before each chunk of 50 requests it adds the measured cost of every
   finished request (from the usage the API returns) and an estimate for the chunk, and stops (exit 6) rather than pass it. This is an
   estimate-based guard, not a cap: submitted requests can still finish, which is why item 1 is the hard limit.
3. **Checkpoints:** the runner's state file records every request (pending, submitted with its batch id, done, incomplete) and is rewritten
   after every step; a rerun resumes and never resends a finished request. Each finished request writes one judgment file; import them
   with `devsearch.review import` and watch coverage with `devsearch.review progress` and `audit` (docs/15).
4. **Failures:** a refused, errored, expired or incomplete request is retried up to 3 times, then left missing for the audit to report;
   nothing is filled in.

## 5. Steps after approval / 批准后的步骤

| # | Step | Spend |
| --- | --- | --- |
| 0 | You add `ANTHROPIC_API_KEY` (the workspace key from section 4) to this environment's settings as an environment variable; a new session picks it up. Never paste the key in chat | none |
| 1 | Calibration: `judge_api.py run --max-requests 2` on pass A's first two batches, reviewer id `pilot-api-calibration` (kept out of the bank); report measured tokens per item and recompute section 3 | under $2 |
| 2 | Continue only if the recomputed total fits the approved budget. Pass A: `judge_api.py run --batches-dir <root>/review/batches/A --pass A --budget-usd <A share> --confirm-paid-usage …`, then import as `judge-api-A` | ~$18 |
| 3 | Pass B, same, separate run and state, imported as `judge-api-B` | ~$18 |
| 4 | `devsearch.review audit`; adjudication batches from `adjudication_queue`; `judge_api.py run --adjudicator --effort high`, imported with `--role adjudicator` (or judged by a person) | ~$3 |
| 5 | `audit` complete, then `gpu.py prepare` seals and builds the bundle; the run package is built from it (docs/GPU_QUICKSTART.md) | none |

A human spot-check of a sample (for example 100 items) remains recommended: two runs of one model measure consistency, not validity.

## 6. What this needs from you / 需要您决定的事项

1. **Approve a budget** (proposed: hard cap $100 on the dedicated workspace) **and the model** (proposed `claude-opus-5-5`; or `claude-sonnet-5-5`).
2. **Network:** before the review, allow the Hugging Face hosts in this environment so the pinned tokenizer can be checked first.
3. **API key:** a key of the dedicated workspace, added as the `ANTHROPIC_API_KEY` environment variable in this environment's settings.

1. 批准预算（建议：专用工作区硬上限 100 美元）与模型（建议 `claude-opus-5-5`，或更便宜的 `claude-sonnet-5-5`）。2. 先放开本环境对 Hugging Face 的访问，以便先确认分词器。3. 在环境设置中以 `ANTHROPIC_API_KEY` 环境变量提供该工作区的密钥（不要在聊天中粘贴）。
