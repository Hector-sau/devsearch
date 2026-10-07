#!/usr/bin/env python3
"""Candidate-review judging through the Claude Message Batches API, with a budget guard and resumable state (docs/16).

    judge_api.py estimate --batches-dir DIR [--batches-dir DIR ...] [--model M] [--effort E]
    judge_api.py run      --batches-dir DIR --pass NAME --state-dir DIR --out-dir DIR --budget-usd X --confirm-paid-usage
                          [--model M] [--effort E] [--adjudicator] [--max-requests N] [--chunk-requests 50] [--poll-seconds 60]

Nothing here decides relevance or touches the formal bank: `run` writes judgment-line files (one per batch file) that
`python -m devsearch.review import` validates and imports, exactly like the judgments of any other reviewer. It spends money only
with `--confirm-paid-usage` and a `--budget-usd`, and only with credentials in the environment (`ANTHROPIC_API_KEY` from a
workspace whose spend limit is the platform-enforced hard cap; see docs/16).

Protocol kept as declared: one request per existing judge batch (the same 40 compact, blinded items in the same seeded order per pass),
the judging criteria of `scripts/realdata/prompts/judge-candidate-v1.md` verbatim, every item judged on its own, `unsure` allowed and
never turned into 0. Context overhead is cut without changing that: the criteria sit in a cached system block and the items travel as
compact JSON lines; the answer is constrained by a JSON schema, so no output is lost to formatting.

Budget: the guard is a second line of defence. Before each new chunk it adds the cost of every finished request (from the token usage
the API returns, at the batch price) and a conservative estimate for the chunk, and stops (exit 6) if that would pass `--budget-usd`.
Requests already submitted can still finish, so the workspace spend limit, which the platform enforces, is the hard cap.

State lives in `--state-dir/<pass>.json` and is rewritten after every step: rerunning `run` with the same arguments resumes (finished
requests are never sent again; a failed, expired or refused request is retried up to 3 times, then left for the audit to report).

Exit: 0 every request of the pass finished; 1 invalid input; 2 usage; 3 missing credentials or confirmation; 4 incomplete after retries;
6 the budget guard stopped before the next chunk.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROMPT_FILE = REPO / "scripts" / "realdata" / "prompts" / "judge-candidate-v1.md"
DEFAULT_MODEL = "claude-opus-5-5"
# List prices in USD per million tokens (Anthropic first-party, cached 2026-09-25); the Batches API bills 50% of these. Override with
# --price-input/--price-output if they change. Cache writes bill 1.25x input, cache reads 0.1x input (0.05x on Claude Opus 5.5).
PRICES = {"claude-opus-5-5": (4.00, 20.00, 0.05), "claude-sonnet-5-5": (2.00, 10.00, 0.10), "claude-haiku-4-5": (1.00, 5.00, 0.10)}
BATCH_DISCOUNT = 0.5
CHARS_PER_TOKEN = 3.0            # conservative for English technical text and code (real counts are usually lower)
ESTIMATED_OUTPUT_PER_ITEM = 120  # judgment JSON plus thinking at the chosen effort: an estimate, replaced by measured usage once known
MAX_ATTEMPTS = 3

SCHEMA = {
    "type": "object",
    "properties": {"judgments": {"type": "array", "items": {
        "type": "object",
        "properties": {"item_id": {"type": "string"}, "relevance": {"type": "string", "enum": ["1", "0", "unsure"]},
                       "evidence_quote": {"type": "string"}, "note": {"type": "string"}},
        "required": ["item_id", "relevance", "evidence_quote", "note"], "additionalProperties": False}}},
    "required": ["judgments"], "additionalProperties": False,
}


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


ADJUDICATOR_NOTE = ("\n\nYou are the adjudicator for items that earlier, independent reviewers could not settle. You are not told their answers; "
                    "judge each item from the question and the passage alone, with the same criteria.")


def criteria(adjudicator: bool = False) -> str:
    """The judging criteria of judge-candidate-v1, verbatim, with the file-based input/output sections replaced by the API's."""
    text = PROMPT_FILE.read_text(encoding="utf-8")
    body = text.split("## Output", 1)[0].rstrip()
    body = body.replace("You are given a JSONL file of items; each has an `item_id`, a `query`\n(`text`, `filters`) and a `passage` (`title`, `section`, `text`, `code_fragment`, `source_url`, `source_location`).",
                        "You are given items as JSON lines; each has an `item_id`, a `question` and a passage (`title`, `section`, `text`).")
    body = body.replace("Do not open URLs or files other than the input and output you are given.", "Do not open or request anything; judge only what is given.")
    body = body.replace("`passage.text`", "the passage `text`").replace("using only the question and the passage text", "using only the question and the passage text")
    return body + (ADJUDICATOR_NOTE if adjudicator else "") + ("\n\n## Output\nReturn one judgment per input item, in input order, with the same `item_id`; `relevance` is the string \"1\", \"0\" or "
                   "\"unsure\". Judge every item; do not skip any.\n")


def prompt_revision(adjudicator: bool = False) -> str:
    return ("judge-adjudicator-v1-api:" if adjudicator else "judge-candidate-v1-api:") + hashlib.sha256(criteria(adjudicator).encode()).hexdigest()[:16]


def reviewer_revision(model: str, effort: str, adjudicator: bool = False) -> str:
    """What the import records as the agent reviewer's revision: prompt hash, model and effort, so every judgment is traceable."""
    return f"{prompt_revision(adjudicator)}|{model}|effort={effort}"


def read_batches(directory: Path) -> dict[str, list[dict]]:
    batches = {}
    for path in sorted(Path(directory).glob("*.jsonl")):
        batches[path.stem] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not batches:
        raise SystemExit(f"judge_api: no batch files (*.jsonl) in {directory}")
    return batches


def user_content(items: list[dict]) -> str:
    lines = "\n".join(json.dumps({k: item.get(k) for k in ("item_id", "question", "title", "section", "text")}, ensure_ascii=False) for item in items)
    return f"Judge each of these {len(items)} items on its own.\n\n{lines}"


def request_params(items: list[dict], model: str, effort: str, adjudicator: bool = False) -> dict:
    return {"model": model, "max_tokens": 16000, "system": [{"type": "text", "text": criteria(adjudicator), "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_content(items)}],
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}


def estimate_tokens(items: list[dict]) -> tuple[int, int]:
    """(input, output) token estimates for one request: characters / CHARS_PER_TOKEN, plus a per-item output allowance."""
    inp = int((len(criteria()) + len(user_content(items))) / CHARS_PER_TOKEN)
    return inp, ESTIMATED_OUTPUT_PER_ITEM * len(items)


def cost(model: str, usage: dict, price_in: float | None = None, price_out: float | None = None) -> float:
    pin, pout, read_mult = PRICES.get(model, (price_in or 0, price_out or 0, 0.1))
    pin, pout = price_in or pin, price_out or pout
    total = (usage.get("input_tokens", 0) * pin + usage.get("cache_creation_input_tokens", 0) * pin * 1.25
             + usage.get("cache_read_input_tokens", 0) * pin * read_mult + usage.get("output_tokens", 0) * pout) / 1e6
    return total * BATCH_DISCOUNT


def estimate(args) -> int:
    rows = []
    for directory in args.batches_dir:
        batches = read_batches(directory)
        tin = tout = items = 0
        for batch in batches.values():
            a, b = estimate_tokens(batch)
            tin, tout, items = tin + a, tout + b, items + len(batch)
        rows.append({"batches_dir": str(directory), "requests": len(batches), "items": items, "input_tokens_est": tin, "output_tokens_est": tout,
                     "cost_usd_est_batch": round(cost(args.model, {"input_tokens": tin, "output_tokens": tout}), 2)})
    print(json.dumps({"model": args.model, "effort": args.effort, "reviewer_revision": reviewer_revision(args.model, args.effort), "chars_per_token": CHARS_PER_TOKEN,
                      "output_tokens_per_item_est": ESTIMATED_OUTPUT_PER_ITEM, "passes": rows,
                      "total_cost_usd_est_batch": round(sum(r["cost_usd_est_batch"] for r in rows), 2),
                      "note": "estimates from character counts; replace with measured usage after a calibration run (docs/16)"}, indent=2))
    return 0


# ---------------------------------------------------------------------------------------------------------------- run

class State:
    def __init__(self, path: Path, header: dict):
        self.path = path
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            changed = {k: (self.data["header"].get(k), v) for k, v in header.items() if self.data["header"].get(k) != v}
            if changed:
                raise SystemExit(f"judge_api: {path} was started with different settings {changed}; use a new --state-dir for a different run")
        else:
            self.data = {"header": header, "requests": {}, "batches": {}, "spent_usd": 0.0, "usage": {}}
            self.save()

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


def parse_output(message, expected: list[str]) -> tuple[list[dict], str | None]:
    """Judgment lines for the items this request carried; anything else in the answer is dropped. Returns (lines, problem)."""
    if getattr(message, "stop_reason", None) == "refusal":
        return [], "refusal"
    text = next((block.text for block in message.content if getattr(block, "type", None) == "text"), None)
    if text is None:
        return [], "no text block"
    try:
        judgments = json.loads(text)["judgments"]
    except (ValueError, KeyError, TypeError) as error:
        return [], f"unparseable answer ({error})"
    wanted, lines = set(expected), {}
    for entry in judgments:
        item = entry.get("item_id")
        if item in wanted and item not in lines:
            rel = entry.get("relevance")
            lines[item] = {"item_id": item, "relevance": 1 if rel == "1" else 0 if rel == "0" else "unsure",
                           "evidence_quote": entry.get("evidence_quote") or "", "note": str(entry.get("note") or "")[:500]}
    missing = [item for item in expected if item not in lines]
    return [lines[item] for item in expected if item in lines], (f"{len(missing)} item(s) missing from the answer" if missing else None)


def run(args) -> int:
    if not args.confirm_paid_usage:
        print("judge_api: refusing to spend: pass --confirm-paid-usage together with an approved --budget-usd (docs/16)", file=sys.stderr)
        return 3
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("judge_api: PREREQUISITE api-key: set ANTHROPIC_API_KEY to a key of a workspace whose spend limit is the approved budget", file=sys.stderr)
        return 3
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    batches = read_batches(args.batches_dir)
    if args.max_requests:
        batches = dict(list(batches.items())[:args.max_requests])
    args.state_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.pass_name
    out.mkdir(parents=True, exist_ok=True)
    header = {"pass": args.pass_name, "model": args.model, "effort": args.effort, "prompt_revision": prompt_revision(args.adjudicator), "batches_dir": str(args.batches_dir.resolve()),
              "requests": sorted(batches)}
    state = State(args.state_dir / f"{args.pass_name}.json", header)
    requests = state.data["requests"]
    for name in batches:
        requests.setdefault(name, {"status": "pending", "attempts": 0, "batch_id": None, "problem": None})
    state.save()
    client = anthropic.Anthropic(max_retries=4)

    def log(message: str) -> None:
        print(f"{utc()} {message}", file=sys.stderr, flush=True)

    while True:
        # 1. collect every submitted batch that has ended
        for batch_id, entry in list(state.data["batches"].items()):
            if entry["collected"]:
                continue
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                continue
            for result in client.messages.batches.results(batch_id):
                name = result.custom_id
                record = requests.get(name)
                if record is None or record["status"] != "submitted" or record["batch_id"] != batch_id:
                    continue
                kind = result.result.type
                if kind == "succeeded":
                    message = result.result.message
                    usage = {k: getattr(message.usage, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
                    spent = cost(args.model, usage, args.price_input, args.price_output)
                    state.data["spent_usd"] = round(state.data["spent_usd"] + spent, 6)
                    for key, value in usage.items():
                        state.data["usage"][key] = state.data["usage"].get(key, 0) + value
                    expected = [item["item_id"] for item in batches[name]]
                    lines, problem = parse_output(message, expected)
                    if problem is None or record["attempts"] >= MAX_ATTEMPTS:
                        if lines:
                            (out / f"{name}.jsonl").write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
                        record.update(status="done" if problem is None else "incomplete", problem=problem, written=len(lines))
                    else:
                        record.update(status="pending", problem=problem)
                else:
                    record.update(status="pending" if record["attempts"] < MAX_ATTEMPTS else "failed", problem=kind)
                log(f"{name}: {record['status']}{' (' + str(record['problem']) + ')' if record['problem'] else ''}")
            entry["collected"] = True
            state.save()
        # 2. submit the next chunk if the budget allows it
        pending = [name for name, record in requests.items() if record["status"] == "pending"]
        waiting = [b for b, entry in state.data["batches"].items() if not entry["collected"]]
        if pending:
            chunk = pending[:args.chunk_requests]
            est = sum(cost(args.model, dict(zip(("input_tokens", "output_tokens"), estimate_tokens(batches[n]))), args.price_input, args.price_output) for n in chunk)
            outstanding = sum(cost(args.model, dict(zip(("input_tokens", "output_tokens"), estimate_tokens(batches[n]))), args.price_input, args.price_output)
                              for n, r in requests.items() if r["status"] == "submitted")
            if state.data["spent_usd"] + outstanding + est > args.budget_usd:
                if waiting:
                    log(f"budget guard: next chunk (~${est:.2f}) would pass ${args.budget_usd:.2f} with ${state.data['spent_usd']:.2f} spent and ~${outstanding:.2f} "
                        "outstanding; waiting for submitted batches to finish")
                else:
                    log(f"budget guard: stopping before the next chunk: ${state.data['spent_usd']:.2f} spent, next chunk ~${est:.2f}, budget ${args.budget_usd:.2f}")
                    state.save()
                    return 6
            else:
                batch = client.messages.batches.create(requests=[Request(custom_id=name, params=MessageCreateParamsNonStreaming(**request_params(batches[name], args.model, args.effort, args.adjudicator)))
                                                                 for name in chunk])
                state.data["batches"][batch.id] = {"requests": chunk, "submitted_at": utc(), "collected": False}
                for name in chunk:
                    requests[name].update(status="submitted", batch_id=batch.id, attempts=requests[name]["attempts"] + 1)
                state.save()
                log(f"submitted {len(chunk)} request(s) as {batch.id}; spent so far ${state.data['spent_usd']:.2f}")
                continue
        if not pending and not waiting:
            break
        time.sleep(args.poll_seconds)
    done = sum(1 for r in requests.values() if r["status"] == "done")
    summary = {"pass": args.pass_name, "requests": len(requests), "done": done, "incomplete_or_failed": len(requests) - done, "spent_usd_est": state.data["spent_usd"],
               "usage": state.data["usage"], "reviewer_revision": reviewer_revision(args.model, args.effort, args.adjudicator), "judgments_dir": str(out),
               "next": f"python -m devsearch.review import --request <review-request.jsonl> --judgments {out}/*.jsonl --reviewer-id <id> --reviewer-type agent "
                       f"--revision '{reviewer_revision(args.model, args.effort, args.adjudicator)}' --role {'adjudicator' if args.adjudicator else 'primary'} "
                       f"--output <root>/review/responses/{args.pass_name}.jsonl"}
    print(json.dumps(summary, indent=2))
    return 0 if done == len(requests) else 4


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="judge_api.py", description=__doc__.split("\n\n")[0], epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--batches-dir", type=Path, action="append", required=True)
    r = sub.add_parser("run")
    r.add_argument("--batches-dir", type=Path, required=True)
    r.add_argument("--pass", dest="pass_name", required=True)
    r.add_argument("--state-dir", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--budget-usd", type=float, required=True, help="the approved budget for this pass; the guard stops submitting before passing it")
    r.add_argument("--confirm-paid-usage", action="store_true", help="required: this command spends money on the configured API key")
    r.add_argument("--max-requests", type=int, help="only the first N batch files (calibration)")
    r.add_argument("--chunk-requests", type=int, default=50)
    r.add_argument("--poll-seconds", type=float, default=60)
    r.add_argument("--adjudicator", action="store_true", help="adjudication run: the adjudicator revision of the prompt (still blinded to earlier answers)")
    r.add_argument("--price-input", type=float)
    r.add_argument("--price-output", type=float)
    for p in (e, r):
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    args = parser.parse_args(argv)
    if args.action == "run" and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.pass_name):
        parser.error("--pass must be a simple name such as A, B or ADJ")
    return estimate(args) if args.action == "estimate" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
