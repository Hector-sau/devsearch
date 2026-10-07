# Candidate-review workflow / 候选审查流程

Status 2026-10-07: **tooling ready, nothing launched.** Coverage of the real pool is 0 of 9,852 items. This page is the operating
procedure for the formal review bank (docs/02 two-stage sealing and "Review protocol revision 5"; docs/14 section 3). Nothing here
calls a model or spends anything: the tools only prepare batches, validate and import judgments you produce, audit coverage under
the declared protocol and report progress. Formal judging starts only after the pinned MiniLM tokenizer has confirmed the candidate pool
(docs/GPU_QUICKSTART.md P2/P3).

状态：工具已就绪，**尚未启动任何审查**；真实池覆盖率 0／9,852。本文是正式审查库的操作流程（第 02 号文档第五版审查协议）。这些工具不调用任何模型、不产生费用。正式判定须在固定 MiniLM 分词器确认候选池之后开始。

## 1. The declared protocol / 声明的审查协议

The dataset's `data.toml` `[review]` table declares the protocol; it is part of that file's hash, so every manifest and seal binds it,
and `devsearch.review audit`, `prepare_data --stage bank` and `gpu.py` all apply it through `devsearch.data.review_protocol`.
The shipped configuration is **`primary_audit_v1`** (revision 5):

| Parameter | Value | Meaning |
| --- | --- | --- |
| `primary_reviewer` | `judge-primary` | the reviewer id the complete primary pass is imported under |
| `audit_seed` | 20261007 | items inside a stratum are taken in `sha256("<seed>\|second-review\|<item_id>")` order |
| strata | primary label (1, 0) × candidate-rank band (1–5, 6–10, 11–20, 21+, known positive outside the list) | hidden from reviewers |
| `audit_rate_positive` / `audit_rate_negative` | 0.25 / 0.08 | share of each stratum second-reviewed |
| `audit_min_per_stratum` | 30 | at least this many per stratum (all of it when smaller) |
| question coverage | every question gets at least one sampled item (a primary positive when it has one) | |
| `escalate_disagreement_rate` | 0.10 | a stratum whose sampled disagreement rate exceeds this is second-reviewed in full |

1. **Primary pass:** one complete pass over all 9,852 items by `judge-primary`. The sample is drawn only once it covers every item.
2. **Second review:** the audit lists the sample as `second_review_queue`; a different reviewer judges it, blind to the primary label
   and to the strata (the batches are cut from the blinded request like any other). Expected size on the frozen pool: about 890–910
   items plus question-coverage additions.
3. **Expansion:** once every sampled item has its second review, each stratum's disagreement rate (second review unresolved or with
   the other label) is compared with 0.10; an escalated stratum's remaining items join `second_review_queue`.
4. **Adjudication:** every unresolved primary (`unsure`, a quote not in the passage, an unusable line) and every disputed or
   unresolved second review joins `adjudication_queue`. The adjudicator must be neither the primary nor a second reviewer of the
   item (`adjudicator is not independent` otherwise; remove that response). An unsure adjudicator settles nothing; another may.
5. **Final:** the independent adjudicator's resolved judgment, else the resolved primary judgment confirmed by every second review.
   Missing judgments are never turned into "not relevant"; anything not final keeps the audit incomplete and `prepare_data --stage bank` refuses to seal.

The values were fixed before any formal judging and depend only on the seed, the primary labels and the hidden ranks. They must never
be changed in response to reranker or baseline test scores; a change is a recorded protocol revision (new data-config hash, new
seals). The original `double_primary` mode (two complete passes, `min_primary_reviews = 2`) stays available for a dataset that declares it.

**Pilot results are kept apart.** Pilot judgments (`reports/evidence/review-pilot/`, runtime `review-pilot*/`) use reviewer ids
starting with `pilot`, are imported only with `--pilot` outside any `responses/` directory, and `audit` refuses a response from a
pilot reviewer. The formal responses live only in `<root>/review/responses/`.

## 2. Primary-pass batches / 主审批次

`gpu.py prepare` writes them (or run it yourself):

```bash
G="python3 scripts/gpu.py cmd --config configs/gpu.toml --"; R=/workspace/devsearch-runtime
mkdir -p $R/review/batches/A
$G review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/A --pass A --batch-size 40
```

Each line is `{"item_id", "question", "title", "section", "text"}`: 9,852 items at 40 per batch is 247 batches.

## 3. Judging / 判定

Give a reviewer one batch at a time with only the rubric `scripts/realdata/prompts/judge-candidate-v1.md` and that batch, and have it
write one line per item: `{"item_id", "relevance": 1 | 0 | "unsure", "evidence_quote", "note"}`. For `1` the quote must be exact
passage text (whitespace may differ). The second reviewer and the adjudicator work independently: separate sessions, no access to
anyone else's output. For agents, record the prompt revision and model; humans are recorded as `human`. Agent judgments stay `agent`
in every record and report; two agent runs measure consistency, not validity.

`scripts/judge_api.py` can send batches through the Claude Message Batches API (paid; docs/16) with resumable state bound to the pool
and batch hashes, never resending a submission it cannot account for. It runs only with an approved budget and is not part of the
current plan, which uses existing session quota (docs/16).

## 4. Import / 导入

```bash
$G review import --request $R/evaluation/review/review-request.jsonl --judgments judged-A/A-0001.jsonl ... \
    --reviewer-id judge-primary --reviewer-type agent --revision <prompt revision|model> --role primary --output $R/review/responses/A-0001-0050.jsonl
```

`--reviewer-type human` takes no `--revision`. An existing output file is never overwritten: import each group of batches to its own
file name, so an interrupted run is continued by importing only the new batches. A line naming an item twice or an unknown item is an
error; `unsure` or a quote not found in the passage is stored as **unresolved**. Second reviews are imported with `--role primary`
under their own reviewer id; adjudications with `--role adjudicator`.

## 5. Progress, resume and batch sizing / 进度、恢复与批次规模

```bash
$G review progress --request $R/evaluation/review/review-request.jsonl --batches-dir $R/review/batches/A \
    --responses-dir $R/review/responses --reviewer-id judge-primary --pass A --budget-tokens 2000000 --output $R/review/progress-A.json
```

Progress is derived from the imported responses alone, so it can be recomputed at any time and a crash loses nothing that was imported.
It lists per reviewer the items judged, resolved and unresolved, and per batch file who covers it. With `--reviewer-id` and `--pass` it
names the **next batches that reviewer has not covered**, stopping at the size you give: `--budget-tokens N` and/or `--limit N` batches.

**`--budget-tokens` is an estimate used only to choose how many batches to hand out. It is not a spending cap.** The number is the
batch's item count times a per-judgment figure (the 1,738 tokens measured in the second pilot; override with `--tokens-per-judgment`).
Nothing measures, meters or limits what a judging run actually consumes. The real limit is whatever applies where the judging runs
(the platform's usage limit, a session allowance, or a count of human hours); use `--budget-tokens` and `--limit` to keep each session
small. Work a session at a time: judge those batches, import, rerun `progress`, and compare the real usage with the estimate.

Extrapolated usage (agent judges, compact context; second pilot): 1,738 tokens per judgment. Revision 5 needs about 9,852 primary +
about 900 second-review judgments + adjudication (a few hundred), about 11,000 judgments or **about 19 million tokens** before any
escalation (**extrapolated, not run**); the earlier two-pass rule needed about 34 million. An escalated large stratum adds up to its size.
A platform usage limit once stopped 12 parallel agents (2026-10-05): keep parallelism modest.

`--budget-tokens` 只是用于挑选批次数量的估算值，**不是费用上限**。

## 6. Audit, second review and adjudication / 审计、复核与裁决

```bash
$G review audit --request $R/evaluation/review/review-request.jsonl --responses-dir $R/review/responses \
    --data-config $R/dataset/data.host.toml --output $R/review/audit.json --merged $R/review/all-responses.jsonl
```

The audit applies the data config's protocol (`--hidden` defaults to `hidden-provenance.jsonl` beside the request; it holds the ranks
the strata need and is never shown to reviewers). Exit 0 only when every item is final; the merged file is then written for
`prepare_data --stage bank`. Exit 3 lists the states (`awaiting the complete primary pass`, `needs second review`,
`awaiting the second-review sample`, `needs adjudication: unresolved primary`, `needs adjudication: second review disagrees`,
`adjudicator left it unresolved`, `adjudicator is not independent`, `more than one adjudicator`), and `audit.json` holds
`second_review_plan` (strata, sizes, disagreement rates, escalations), `second_review_queue` and `adjudication_queue`:

```bash
mkdir -p $R/review/batches/S1 $R/review/batches/ADJ1
$G review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/S1 --pass S1 --items $R/review/audit.json --queue second-review
$G review import ... --reviewer-id judge-second --role primary --output $R/review/responses/S1.jsonl
$G review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/ADJ1 --pass ADJ1 --items $R/review/audit.json
$G review import ... --reviewer-id adjudicator-1 --role adjudicator --output $R/review/responses/ADJ1.jsonl
```

Rerun the audit after each import (an escalation adds a new second-review queue). Then rerun
`python3 scripts/gpu.py prepare --config configs/gpu.toml`, which audits, seals the initial test stage and builds the bundle.
Formal evaluation (`run`) cannot start without the bundle, and the bundle cannot exist without the seal.

## 7. What is not covered / 未覆盖

Agent review stays labelled `agent` in every report; no step turns it into human review. The 9,852 labels remain synthetic-question
labels. Items outside the second-review sample rest on one primary judgment, checked statistically by their stratum's sample, not
individually. The usage figures above are estimates.
