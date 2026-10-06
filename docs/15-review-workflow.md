# Candidate-review workflow / 候选审查流程

Status 2026-10-06: **tooling ready, nothing launched.** Coverage of the real pool is 0 of 9,852 items. This page is the operating
procedure for the formal review bank (docs/02 two-stage sealing; docs/14 section 3). Nothing here calls a model or spends anything:
the tools only prepare batches, validate and import judgments you produce, and report progress.

状态：工具已就绪，**尚未启动任何审查**；真实池覆盖率 0／9,852。本文是正式审查库的操作流程；这些工具不调用任何模型、不产生费用，只负责准备批次、校验并导入您产出的判定、报告进度。

## 1. What counts as the formal bank / 什么算正式审查库

- The pool is the blinded `evaluation/review/review-request.jsonl` (9,852 items for the frozen dataset): question, passage title,
  section and text under a salted item id. It carries no rank, score, system, provenance or earlier label (`assert_blinded`).
- A formal item is final when it has **an adjudicator's resolved judgment, or resolved agreeing judgments from at least two distinct
  primary reviewers** (`[review] min_primary_reviews = 2`). Anything else (one pass, disagreement, `unsure`, an unlocatable quote)
  keeps the audit incomplete, and `prepare_data --stage bank` refuses to seal. Missing judgments are never turned into "not relevant".
- **Pilot results are kept apart.** Pilot judgments (`reports/evidence/review-pilot/`, runtime `review-pilot*/`) use reviewer ids
  starting with `pilot`, are imported only with `--pilot` outside any `responses/` directory, and `audit` refuses a response from a
  pilot reviewer. The formal responses live only in `<root>/review/responses/`.

## 2. Batches / 盲审批次

`gpu.py prepare` writes them (or run it yourself):

```bash
G="python3 scripts/gpu.py cmd --config configs/gpu.toml --"; R=/workspace/devsearch-runtime
mkdir -p $R/review/batches/A $R/review/batches/B
$G review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/A --pass A --batch-size 40
$G review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/B --pass B --batch-size 40
```

Each line is `{"item_id", "question", "title", "section", "text"}`. Pass A and pass B order the items differently (seeded hash), so the
two reviewers never see them in the same order. 9,852 items at 40 per batch is 247 batches per pass.

## 3. Judging (outside this repository) / 判定

Give a reviewer one batch at a time with the prompt `scripts/realdata/prompts/judge-candidate-v1.md`, and have it write one line per
item: `{"item_id", "relevance": 1 | 0 | "unsure", "evidence_quote", "note"}`. For `1` the quote must be exact passage text (whitespace
may differ). The two passes must be **independent**: separate runs, no access to each other's output. For agents, a different prompt
revision or model for the adjudicator is recommended; humans are recorded as `human`. A person should also spot-check a sample: two
runs of the same model measure consistency, not validity (evidence report 2.10).

## 4. Import (two passes) / 双轮导入

```bash
for P in A B; do
  $G review import --request $R/evaluation/review/review-request.jsonl --judgments judged-$P/*.jsonl \
      --reviewer-id judge-$P --reviewer-type agent --revision judge-candidate-v1 --role primary --output $R/review/responses/$P.jsonl
done
```

`--reviewer-type human` takes no `--revision`. An existing output file is never overwritten: import each group of batches to its own file name
(`A-0001-0050.jsonl`), so an interrupted run is continued by importing only the new batches. (A reviewer who judged an item twice counts once, the first recorded.)
A line naming an item twice or an unknown item is an error; `unsure` or a quote not found in the passage is stored as **unresolved**.

## 5. Progress, resume and batch sizing / 进度、恢复与批次规模

```bash
$G review progress --request $R/evaluation/review/review-request.jsonl --batches-dir $R/review/batches/A \
    --responses-dir $R/review/responses --reviewer-id judge-A --pass A --budget-tokens 2000000 --output $R/review/progress-A.json
```

Progress is derived from the imported responses alone, so it can be recomputed at any time and a crash loses nothing that was imported.
It lists per reviewer the items judged, resolved and unresolved, and per batch file who covers it. With `--reviewer-id` and `--pass` it
names the **next batches that reviewer has not covered**, stopping at the size you give: `--budget-tokens N` and/or `--limit N` batches.

**`--budget-tokens` is an estimate used only to choose how many batches to hand out. It is not a spending cap.** The number is the
batch's item count times a per-judgment figure (the 1,738 tokens measured in the second pilot; override with `--tokens-per-judgment`).
Nothing measures, meters or limits what a judging run actually consumes: a reviewer, prompt or platform that costs more than the pilot
will exceed the estimate, and nothing here stops it. The real spending limit is whatever you set where the judging runs (the agent
platform's own usage limit, a per-session allowance, or a count of human hours); use `--budget-tokens` and `--limit` to keep each
session small enough for that limit to matter. Work a session at a time: judge those batches, import, rerun `progress`, and compare
the real usage with the estimate before the next session. The tools never start any judging themselves.

Measured and extrapolated usage (agent judges, compact context; second pilot, 40 items per run): 1,738 tokens and 0.85 s per judgment.
Two passes over 9,852 items are about 19,700 judgments, about 34 million tokens before adjudication (**extrapolated, not run**).
A platform usage limit once stopped 12 parallel agents (2026-10-05): keep parallelism modest and set the actual spending limit on the platform; `--budget-tokens` only sizes a session.

`--budget-tokens` 只是用于挑选批次数量的估算值，**不是费用上限**：工具不统计也不限制实际用量，实际限额须在运行判定的平台或预算处设置。

## 6. Audit and adjudication / 审计与裁决

```bash
$G review audit --request $R/evaluation/review/review-request.jsonl --responses-dir $R/review/responses \
    --output $R/review/audit.json --merged $R/review/all-responses.jsonl
```

Exit 0 only when every item is final; the merged file is then written for `prepare_data --stage bank`. Exit 3 lists the states
(`needs primary reviews (k of 2)`, `needs adjudication: primaries disagree`, `needs adjudication: unresolved primary`,
`adjudicator left it unresolved`, `more than one adjudicator`) and `audit.json` holds `adjudication_queue` and `needs_primary`.
Adjudication judges only the queue, by a different reviewer (a different prompt revision for agents, or a person):

```bash
mkdir -p $R/review/batches/ADJ
$G review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/ADJ --pass ADJ --items $R/review/audit.json
$G review import ... --reviewer-id adjudicator-1 --role adjudicator --output $R/review/responses/ADJ.jsonl
```

An adjudicator who is unsure settles nothing: the item stays in the queue until another adjudicator resolves it. Then rerun
`python3 scripts/gpu.py prepare --config configs/gpu.toml`, which audits, seals the initial test stage and builds the bundle.
Formal evaluation (`run`) cannot start without the bundle, and the bundle cannot exist without the seal.

## 7. What is not covered / 未覆盖

Agent review stays labelled `agent` in every report; no step turns it into human review. The 9,852 labels remain synthetic-question
labels. The cost above is an estimate; the real figure depends on the platform and prompt.
