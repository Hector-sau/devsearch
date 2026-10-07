# Review-bank execution: paid API option (not authorized) / 审查库执行：付费接口方案（未授权）

Status 2026-10-07: **not authorized, nothing launched, nothing billed.** The user's 2026-10-07 instruction permits only existing
available quota for the review and no new paid API charges, so this page is kept as a costed option, not the plan. The runner
(`scripts/judge_api.py`) is built and tested against a local fake API (`tests/tooling/test_judge_api.py`), including resumable state
bound to its input and reconciliation of uncertain submissions; it refuses to run without `--confirm-paid-usage`, a budget and a key.
The current plan (judging with existing session quota, protocol revision 5) is in docs/15 and the acceptance report.

状态：**未授权、未启动、未产生费用。**2026-10-07 指令只允许使用现有额度，不授权新增 API 费用，故本页仅作为有成本测算的备选方案。

## 1. What would be judged / 判定范围

The declared protocol (docs/02 revision 5, `primary_audit_v1`): the blinded pool of **9,852 items** (`review-request.jsonl`, SHA-256
`e872f940…` for the v1 dataset; to be confirmed with the pinned tokenizer), **one complete primary pass**, a **seeded stratified
second review** of about 900 items (escalated in full for a stratum whose disagreement rate exceeds 10%), and **adjudication** of every
unresolved or disputed item. Nothing is sealed until `devsearch.review audit` reports complete coverage; missing items are never filled in.

- Batches: the existing judge batches, 40 compact blinded items each (item id, question, title, section, text); 247 requests for the
  primary pass; the second review and adjudication are cut from the audit's queues in separate requests with no shared context.
- Criteria: `scripts/realdata/prompts/judge-candidate-v1.md`, verbatim (the file-handling lines are replaced by the API's input/output).
  Recorded revision per judgment: `judge-candidate-v1-api:3e159a182d56d770|<model>|effort=<effort>` (adjudicator: `judge-adjudicator-v1-api:…`).
- Labels: every judgment is imported as `reviewer.type = agent`. Pilot and calibration judgments use `pilot-*` reviewer ids, which `audit` refuses.

**Precondition:** the pinned MiniLM tokenizer must first be shown to reproduce this exact pool. Model access is therefore resolved first.

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
| Primary pass (247 requests) | 3.06 M | 1.18 M (120 per item) | $17.9 | $9.0 |
| Second review (~900 items) | ~0.3 M | ~0.11 M | ~$1.7 | ~$0.9 |
| Adjudication (about 300–600 items, effort `high`) | ~0.2 M | ~0.2 M | ~$3 | ~$1.5 |
| **Central estimate** | **~3.6 M** | **~1.5 M** | **~$23** | **~$11** |
| If output is 400 tokens per item (heavy thinking) | ~3.6 M | ~4.6 M | ~$54 | ~$27 |
| Escalation of the largest stratum (rank 21+, label 0: up to ~5,400 more items) | +~1.7 M | +~0.65 M | +~$10 | +~$5 |

The output figure is the uncertain one (thinking at `medium` effort is not yet measured for this task), hence the calibration step.

## 4. Hard limit, guard and checkpoints / 硬上限、保护与进度

1. **Hard cap, enforced by the platform:** create a dedicated workspace in the Claude Console (for example `devsearch-review`) with a
   **spend limit equal to the approved budget**, and an API key from that workspace only. Requests beyond the limit are rejected by
   the platform, whatever the script does. **Proposed limit: $100** (covers the high case for Opus 5.5).
2. **Guard in the runner (second line):** `--budget-usd` per pass. Before each chunk of 50 requests it adds the measured cost of every
   finished request (from the usage the API returns) and an estimate for the chunk, and stops (exit 6) rather than pass it. This is an
   estimate-based guard, not a cap: submitted requests can still finish, which is why item 1 is the hard limit.
3. **Checkpoints:** the runner's state file records every request (pending, submitting, submitted with its batch id, done, incomplete)
   and is rewritten atomically after every step; a rerun resumes and never resends a finished request. The state is bound to the
   review pool's SHA-256 (`--request`), every batch file's SHA-256 and every request's full parameters; every batch item must be the
   pool's item, and a rerun with any of these changed is refused before anything is sent (exit 1). Each finished request writes one
   judgment file whose SHA-256 is recorded and checked before reuse and at the end: a missing or changed file is restored from the
   recorded batch's results (nothing billed) or the run stops; a file the state did not write stops the run, since the import would take it.
   Import the files with `devsearch.review import` and watch coverage with `devsearch.review progress` and `audit` (docs/15).
4. **No blind resubmission:** each chunk is written to the state as a submission intent (random id, also carried in every
   `custom_id`) before the create call, which is never retried automatically. If the process dies after the API created the batch
   but before the state recorded it, or the call ends without a clear answer, the next step reconciles the intent against the
   workspace's batch list: exactly one unrecorded batch of that size created within the window (`--reconcile-window-seconds`, 900)
   is adopted, and its results are used only if their `custom_id`s are exactly the intent's; none after the window proves nothing was
   created. Several candidates, or an adopted batch with other `custom_id`s, stop the run (exit 7) without sending anything; after
   checking the console the operator reruns with `--adopt-batch INTENT=BATCH_ID` or `--abandon-intent INTENT`. Run one pass at a time
   in the workspace, so another run's batch of the same size cannot make reconciliation ambiguous.
5. **Failures:** a refused, errored, expired or incomplete request is retried up to 3 times, then left missing for the audit to report;
   nothing is filled in.

## 5. Steps, only if a budget is ever approved / 若日后批准预算的步骤

| # | Step | Spend |
| --- | --- | --- |
| 0 | `ANTHROPIC_API_KEY` of a dedicated workspace whose spend limit is the approved budget, added as an environment variable in the environment settings (never pasted in chat) | none |
| 1 | Calibration: `judge_api.py run --max-requests 2` on the primary pass's first two batches, reviewer id `pilot-api-calibration` (kept out of the bank); measured tokens per item replace the estimate above | under $2 |
| 2 | Primary pass: `judge_api.py run --request <root>/evaluation/review/review-request.jsonl --batches-dir <root>/review/batches/A --pass A --budget-usd <share> --confirm-paid-usage …`, imported as `judge-primary` | ~$18 |
| 3 | `devsearch.review audit --data-config …`; second-review batches from `second_review_queue`, a separate run and state with a different reviewer id | ~$2 (+ escalation) |
| 4 | Adjudication batches from `adjudication_queue`; `judge_api.py run --adjudicator --effort high`, imported with `--role adjudicator` (or judged by a person) | ~$3 |
| 5 | `audit` complete, then `gpu.py prepare` seals and builds the bundle; the run package is built from it (docs/GPU_QUICKSTART.md) | none |

A human spot-check of a sample remains recommended: agent runs measure consistency, not validity.

## 6. Status of the decision / 决策状态

Not requested now. The paid option needs a new, explicit budget approval, a dedicated workspace key and the Hugging Face access that
lets the pinned tokenizer confirm the pool first. Until then the review uses existing session quota only (docs/15).

当前不申请。付费方案需另行明确批准预算、专用工作区密钥，并先开放 Hugging Face 以便分词器确认候选池；在此之前审查只使用现有会话额度。
