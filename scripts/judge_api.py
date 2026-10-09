#!/usr/bin/env python3
"""Candidate-review judging through the Claude Message Batches API, with a budget guard and resumable state (docs/16).

    judge_api.py estimate --batches-dir DIR [--batches-dir DIR ...] [--model M] [--effort E]
    judge_api.py run      --request review-request.jsonl --batches-dir DIR --pass NAME --state-dir DIR --out-dir DIR --budget-usd X
                          --confirm-paid-usage [--model M] [--effort E] [--adjudicator] [--max-requests N] [--chunk-requests 50]
                          [--poll-seconds 60] [--reconcile-window-seconds 900] [--adopt-batch INTENT=BATCH_ID] [--abandon-intent INTENT]

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

Resumable state (`--state-dir/<pass>.json`, rewritten atomically after every step) is bound to its input: the sha256 of the review
pool (`--request`), of every batch file and of every request's full parameters (model, effort, prompt, schema, items). Every batch
item must be the pool's item, unchanged. A rerun with any of that changed is refused before anything is sent (exit 1). Finished
requests are never sent again: each output file's sha256 is recorded, checked before reuse and at the end, and a missing or changed file
is restored from the recorded batch's results (no new request) or the run stops for reconciliation; a file in the output directory that
the state did not write stops the run too, because the import would pick it up.

Submissions are recorded before they happen. Each chunk is first written to the state as a submission intent with a random id that
also tags every custom_id (`<batch file>__<intent>`); only then is the batch created, with no automatic retry of the create call.
If the process dies or the call fails without a clear answer, the intent stays open and is reconciled from the workspace's batch list
before anything is sent again: exactly one unrecorded batch of that size created within `--reconcile-window-seconds` is adopted; none,
once that window has passed, proves nothing was created and the requests are sent again; several stop the run (exit 7). An adopted
batch's results are used only if their custom_ids are exactly the intent's. When duplicate-free recovery cannot be established, the
run stops with exit 7 and names the operator choices (`--adopt-batch`, `--abandon-intent`) to make after checking the console.

Exit: 0 every request of the pass finished; 1 invalid or changed input; 2 usage; 3 missing credentials or confirmation; 4 incomplete
after retries; 6 the budget guard stopped before the next chunk; 7 reconciliation required (nothing was sent again).
"""

import argparse
import hashlib
import json
import os
import re
import secrets
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

STATE_FORMAT = 2
EXIT_RECONCILE = 7
CREATE_TIMEOUT = 300.0           # seconds; a create call is never retried by the SDK, so an uncertain one goes to reconciliation
SEP = "__"                       # custom_id = <batch file name>__<intent id>; batch file names carry no underscore
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,47}")


class Stop(Exception):
    """A condition the runner must not guess past. `code` is the exit status: 1 invalid input, 7 reconciliation required."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def render(lines: list[dict]) -> bytes:
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode()


def load_pool(path: Path) -> tuple[str, dict[str, dict]]:
    """(sha256 of the review-request file, its compact blinded items by id), through the same reader the import uses."""
    sys.path.insert(0, str(REPO / "src"))
    from devsearch.data.review_workflow import compact, load_request
    try:
        request = load_request(path)
    except Exception as error:                                     # noqa: BLE001 (reported as invalid input)
        raise Stop(1, f"judge_api: --request {path}: {error}") from error
    return sha256(Path(path).read_bytes()), {item: compact(record) for item, record in request.items()}


def bind_batches(directory: Path, pool: dict[str, dict], max_requests: int | None) -> dict[str, dict]:
    """{name: {items, batch_sha256}} for the batch files, after checking that every item is the pool's item, unchanged."""
    paths = sorted(Path(directory).glob("*.jsonl"))[:max_requests or None]
    if not paths:
        raise Stop(1, f"judge_api: no batch files (*.jsonl) in {directory}")
    bound = {}
    for path in paths:
        if not NAME.fullmatch(path.stem):
            raise Stop(1, f"judge_api: batch file name {path.name!r} must be letters, digits and '-' (at most 48)")
        data = path.read_bytes()
        items = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        for item in items:
            if pool.get(item.get("item_id")) != {k: item.get(k) for k in ("item_id", "question", "title", "section", "text")}:
                raise Stop(1, f"judge_api: {path.name}: item {item.get('item_id')!r} is not the review pool's item (missing or changed); rebuild the batches")
        bound[path.stem] = {"items": items, "batch_sha256": sha256(data)}
    return bound


class State:
    """Resumable state bound to the review pool, every batch file's content and every request's parameters (the header)."""

    def __init__(self, path: Path, header: dict):
        self.path = path
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            old = self.data.get("header", {})
            changed = sorted(k for k in set(old) | set(header) if k != "requests" and old.get(k) != header.get(k))
            old_req, new_req = old.get("requests", {}), header["requests"]
            for name in sorted(set(old_req) | set(new_req)):
                if old_req.get(name) != new_req.get(name):
                    changed.append(f"request {name} ({'added' if name not in old_req else 'removed' if name not in new_req else 'content or parameters changed'})")
            if changed:
                raise Stop(1, f"judge_api: {path} was started for different input: {', '.join(changed[:10])}{' ...' if len(changed) > 10 else ''}. "
                              "Nothing was sent. Restore the original input, or use a new --state-dir (and --out-dir) for a different run")
        else:
            self.data = {"header": header, "requests": {}, "batches": {}, "intents": {}, "spent_usd": 0.0, "usage": {}}
            self.save()

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.data, indent=1, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
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


def write_output(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def commit_submission(state: State, intent: str, batch_id: str, how: str) -> None:
    """Record that `intent` is the remote batch `batch_id` (created here, found by reconciliation, or named by the operator)."""
    record = state.data["intents"][intent]
    state.data["batches"][batch_id] = {"intent": intent, "requests": record["requests"], "recorded_at": utc(), "how": how, "collected": False}
    record.update(status="submitted", batch_id=batch_id)
    for name in record["requests"]:
        state.data["requests"][name].update(status="submitted", batch_id=batch_id)
    state.save()


def release_intent(state: State, intent: str, status: str) -> None:
    """The intent created no batch (proved, or asserted by the operator): its requests may be sent again and the attempt is not counted."""
    record = state.data["intents"][intent]
    record["status"] = status
    for name in record["requests"]:
        entry = state.data["requests"][name]
        if entry.get("intent") == intent and entry["status"] in ("submitting", "submitted"):
            entry.update(status="pending", batch_id=None, attempts=entry["attempts"] - 1)
    drop_adopted(state, intent, status)
    state.save()


def drop_adopted(state: State, intent: str, reason: str) -> None:
    """Stop using the batch recorded for `intent` (its results are never read); its id stays known so it is never adopted again."""
    batch_id = state.data["intents"][intent].get("batch_id")
    if batch_id and batch_id in state.data["batches"]:
        state.data["batches"][batch_id].update(collected=True, released=reason)


def reconcile_help(intent: str) -> str:
    return (f"Check the Batches page of the workspace console for batches created around that time, then rerun with either "
            f"--adopt-batch {intent}=<batch id> (that batch is this intent; its custom_ids are still verified when it ends) or "
            f"--abandon-intent {intent} (no batch from it exists, or you canceled it; its requests are sent again)")


def run(args) -> int:
    if not args.confirm_paid_usage:
        print("judge_api: refusing to spend: pass --confirm-paid-usage together with an approved --budget-usd (docs/16)", file=sys.stderr)
        return 3
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("judge_api: PREREQUISITE api-key: set ANTHROPIC_API_KEY to a key of a workspace whose spend limit is the approved budget", file=sys.stderr)
        return 3
    try:
        return _run(args)
    except Stop as stop:
        print(str(stop), file=sys.stderr)
        return stop.code


def _run(args) -> int:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    pool_sha, pool = load_pool(args.request)
    batches = bind_batches(args.batches_dir, pool, args.max_requests)
    params = {name: request_params(b["items"], args.model, args.effort, args.adjudicator) for name, b in batches.items()}
    header = {"format": STATE_FORMAT, "pass": args.pass_name, "model": args.model, "effort": args.effort, "adjudicator": bool(args.adjudicator),
              "prompt_revision": prompt_revision(args.adjudicator), "settings_sha256": sha256(canonical(request_params([], args.model, args.effort, args.adjudicator))),
              "pool_sha256": pool_sha, "requests": {name: {"batch_sha256": b["batch_sha256"], "params_sha256": sha256(canonical(params[name]))} for name, b in batches.items()}}
    args.state_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / args.pass_name
    out.mkdir(parents=True, exist_ok=True)
    state = State(args.state_dir / f"{args.pass_name}.json", header)
    requests, intents = state.data["requests"], state.data["intents"]
    for name in batches:
        requests.setdefault(name, {"status": "pending", "attempts": 0, "batch_id": None, "intent": None, "problem": None})
    state.save()
    client = anthropic.Anthropic(max_retries=4)
    creator = client.with_options(max_retries=0, timeout=CREATE_TIMEOUT)   # a retried create could make a second, billed batch

    def log(message: str) -> None:
        print(f"{utc()} {message}", file=sys.stderr, flush=True)

    # operator decisions on intents that reconciliation could not settle
    for spec in args.adopt_batch or []:
        intent, _, batch_id = spec.partition("=")
        if intents.get(intent, {}).get("status") not in ("open", "mismatch") or not batch_id:
            raise Stop(1, f"judge_api: --adopt-batch {spec}: no open or mismatched submission intent {intent!r} "
                          f"(unsettled: {[i for i, r in intents.items() if r['status'] in ('open', 'mismatch')]})")
        if batch_id in state.data["batches"]:
            raise Stop(1, f"judge_api: --adopt-batch {spec}: {batch_id} is already recorded for intent {state.data['batches'][batch_id]['intent']}")
        drop_adopted(state, intent, "replaced by the operator")
        commit_submission(state, intent, batch_id, "operator")
        log(f"intent {intent}: adopted {batch_id} as the operator asked; its custom_ids are verified when it ends")
    for intent in args.abandon_intent or []:
        if intents.get(intent, {}).get("status") not in ("open", "mismatch"):
            raise Stop(1, f"judge_api: --abandon-intent {intent}: not an open or mismatched submission intent")
        release_intent(state, intent, "abandoned")
        log(f"intent {intent}: abandoned as the operator asked; its {len(intents[intent]['requests'])} request(s) will be sent again")
    if any(r["status"] == "mismatch" for r in intents.values()):
        bad = [i for i, r in intents.items() if r["status"] == "mismatch"]
        raise Stop(EXIT_RECONCILE, f"judge_api: RECONCILE: the batch adopted for intent(s) {bad} did not carry their custom_ids; " + reconcile_help(bad[0]))

    def fetch(batch_id: str) -> list:
        return list(client.messages.batches.results(batch_id))

    def check_outputs(restore: bool) -> None:
        """Every finished request's file exists with its recorded hash (restored from the recorded batch's results if not), and no file is unaccounted."""
        for path in sorted(out.glob("*.jsonl")):
            record = requests.get(path.stem)
            if record is None or record["status"] not in ("done", "incomplete", "submitted") or (record["status"] != "submitted" and not record.get("output_sha256")):
                raise Stop(EXIT_RECONCILE, f"judge_api: RECONCILE: {path} is not an output this state recorded; move it out of {out} (it would be imported) and rerun")
        for name, record in requests.items():
            if record["status"] not in ("done", "incomplete") or not record.get("output_sha256"):
                continue
            path = out / f"{name}.jsonl"
            if path.is_file() and sha256(path.read_bytes()) == record["output_sha256"]:
                continue
            problem = "missing" if not path.is_file() else "changed (hash differs)"
            if not restore:
                raise Stop(EXIT_RECONCILE, f"judge_api: RECONCILE: output {path} is {problem}")
            cid = f"{name}{SEP}{record['intent']}"
            try:
                result = next((r for r in fetch(record["batch_id"]) if r.custom_id == cid), None)
            except anthropic.AnthropicError as error:
                result, problem = None, f"{problem}; the results of {record['batch_id']} could not be fetched ({error.__class__.__name__})"
            data = None
            if result is not None and result.result.type == "succeeded":
                data = render(parse_output(result.result.message, [item["item_id"] for item in batches[name]["items"]])[0])
            if data is None or sha256(data) != record["output_sha256"]:
                raise Stop(EXIT_RECONCILE, f"judge_api: RECONCILE: output {path} is {problem} and could not be restored to the recorded hash "
                                           f"{record['output_sha256'][:16]}... from {record['batch_id']}. Nothing was resubmitted. If that file was already imported, "
                                           "restore it from a copy; otherwise start a new --state-dir and --out-dir for the affected batch files")
            write_output(path, data)
            log(f"{name}: output {problem}; restored from {record['batch_id']} (hash verified, nothing billed)")

    check_outputs(restore=True)

    def reconcile() -> bool:
        """Settle every open intent from the workspace's batch list. Returns True while an intent is still too young to settle."""
        young = False
        for intent, record in intents.items():
            if record["status"] != "open":
                continue
            lo, hi = record["epoch"] - args.reconcile_window_seconds, record["epoch"] + args.reconcile_window_seconds
            known = set(state.data["batches"])
            candidates = []
            for batch in client.messages.batches.list(limit=100):
                created = batch.created_at.timestamp()
                if created < lo - 86400:                            # the list is newest first; nothing older can match
                    break
                counts = batch.request_counts
                total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
                if lo <= created <= hi and batch.id not in known and total == len(record["requests"]):
                    candidates.append(batch.id)
            record["checked_at"], record["candidates"] = utc(), candidates
            if len(candidates) == 1:
                commit_submission(state, intent, candidates[0], "reconciled")
                log(f"intent {intent}: the uncertain submission is {candidates[0]} (the only batch of {len(record['requests'])} request(s) created in the window); "
                    "adopted, custom_ids verified when it ends")
            elif candidates:
                state.save()
                raise Stop(EXIT_RECONCILE, f"judge_api: RECONCILE: intent {intent} ({len(record['requests'])} request(s), {record['created_at']}) matches several batches "
                                           f"{candidates}; resubmitting or guessing could duplicate billed work. Nothing was sent. " + reconcile_help(intent))
            elif time.time() - record["epoch"] >= args.reconcile_window_seconds:
                release_intent(state, intent, "not-created")
                log(f"intent {intent}: no batch was created in the window; its {len(record['requests'])} request(s) will be sent again")
            else:
                state.save()
                young = True
        return young

    while True:
        unsettled = reconcile()
        # 1. collect every recorded batch that has ended; its custom_ids must be exactly this intent's
        for batch_id, entry in list(state.data["batches"].items()):
            if entry["collected"]:
                continue
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status != "ended":
                continue
            results = fetch(batch_id)
            got = [result.custom_id for result in results]
            expected = {f"{name}{SEP}{entry['intent']}" for name in entry["requests"]}
            if set(got) != expected or len(got) != len(expected):
                intents[entry["intent"]]["status"] = "mismatch"
                state.save()
                raise Stop(EXIT_RECONCILE, f"judge_api: RECONCILE: batch {batch_id} ({entry['how']}) does not carry the custom_ids of intent {entry['intent']}; "
                                           "nothing from it was used. " + reconcile_help(entry["intent"]))
            for result in results:
                name = result.custom_id.rsplit(SEP, 1)[0]
                record = requests[name]
                if record["status"] != "submitted" or record["batch_id"] != batch_id:
                    continue
                kind = result.result.type
                if kind == "succeeded":
                    message = result.result.message
                    usage = {k: getattr(message.usage, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
                    state.data["spent_usd"] = round(state.data["spent_usd"] + cost(args.model, usage, args.price_input, args.price_output), 6)
                    for key, value in usage.items():
                        state.data["usage"][key] = state.data["usage"].get(key, 0) + value
                    lines, problem = parse_output(message, [item["item_id"] for item in batches[name]["items"]])
                    if problem is None or record["attempts"] >= MAX_ATTEMPTS:
                        data = render(lines) if lines else None
                        if data is not None:
                            write_output(out / f"{name}.jsonl", data)
                        record.update(status="done" if problem is None else "incomplete", problem=problem, written=len(lines),
                                      output_sha256=sha256(data) if data is not None else None)
                    else:
                        record.update(status="pending", problem=problem)
                else:
                    record.update(status="pending" if record["attempts"] < MAX_ATTEMPTS else "failed", problem=kind)
                log(f"{name}: {record['status']}{' (' + str(record['problem']) + ')' if record['problem'] else ''}")
            entry["collected"] = True
            state.save()
        # 2. submit the next chunk if the budget allows it, recording the intent before the call
        pending = [name for name, record in requests.items() if record["status"] == "pending"]
        waiting = [b for b, entry in state.data["batches"].items() if not entry["collected"]]
        if pending:
            def est_cost(names):
                return sum(cost(args.model, dict(zip(("input_tokens", "output_tokens"), estimate_tokens(batches[n]["items"]))), args.price_input, args.price_output) for n in names)
            chunk = pending[:args.chunk_requests]
            est = est_cost(chunk)
            outstanding = est_cost([n for n, r in requests.items() if r["status"] in ("submitted", "submitting")])
            if state.data["spent_usd"] + outstanding + est > args.budget_usd:
                if waiting or unsettled:
                    log(f"budget guard: next chunk (~${est:.2f}) would pass ${args.budget_usd:.2f} with ${state.data['spent_usd']:.2f} spent and ~${outstanding:.2f} "
                        "outstanding; waiting for submitted batches to finish")
                else:
                    log(f"budget guard: stopping before the next chunk: ${state.data['spent_usd']:.2f} spent, next chunk ~${est:.2f}, budget ${args.budget_usd:.2f}")
                    state.save()
                    return 6
            else:
                intent = secrets.token_hex(6)
                intents[intent] = {"requests": chunk, "created_at": utc(), "epoch": time.time(), "status": "open", "batch_id": None}
                for name in chunk:
                    requests[name].update(status="submitting", intent=intent, attempts=requests[name]["attempts"] + 1)
                state.save()                                        # the intent is on disk before anything can be created remotely
                try:
                    batch = creator.messages.batches.create(requests=[Request(custom_id=f"{name}{SEP}{intent}", params=MessageCreateParamsNonStreaming(**params[name]))
                                                                      for name in chunk])
                except anthropic.APIStatusError as error:
                    if error.status_code >= 500 or error.status_code in (408, 409):
                        log(f"intent {intent}: create returned HTTP {error.status_code}; the batch may exist, reconciling before anything is sent again")
                        continue
                    release_intent(state, intent, "rejected")       # a 4xx answer means nothing was created
                    if error.status_code == 429:
                        log("create was rate limited (HTTP 429); nothing was created, waiting before the next try")
                        time.sleep(args.poll_seconds)
                        continue
                    raise Stop(1, f"judge_api: the API rejected the batch (HTTP {error.status_code}): {error}")
                except anthropic.APIConnectionError as error:       # includes timeouts: the request may have reached the API
                    log(f"intent {intent}: create did not answer ({error.__class__.__name__}); the batch may exist, reconciling before anything is sent again")
                    continue
                commit_submission(state, intent, batch.id, "created")
                log(f"submitted {len(chunk)} request(s) as {batch.id} (intent {intent}); spent so far ${state.data['spent_usd']:.2f}")
                continue
        if not pending and not waiting and not unsettled:
            break
        time.sleep(args.poll_seconds)
    check_outputs(restore=False)
    done = sum(1 for r in requests.values() if r["status"] == "done")
    summary = {"pass": args.pass_name, "requests": len(requests), "done": done, "incomplete_or_failed": len(requests) - done, "spent_usd_est": state.data["spent_usd"],
               "usage": state.data["usage"], "pool_sha256": pool_sha, "reviewer_revision": reviewer_revision(args.model, args.effort, args.adjudicator),
               "judgments_dir": str(out),
               "next": f"python -m devsearch.review import --request {args.request} --judgments {out}/*.jsonl --reviewer-id <id> --reviewer-type agent "
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
    r.add_argument("--request", type=Path, required=True, help="the blinded review-request.jsonl the batches were cut from (its hash binds the state)")
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
    r.add_argument("--reconcile-window-seconds", type=float, default=900,
                   help="an uncertain create is matched to batches created within this many seconds of its intent; with none found after it, it is proved not created")
    r.add_argument("--adopt-batch", action="append", metavar="INTENT=BATCH_ID", help="operator reconciliation: that batch is this intent (custom_ids still verified)")
    r.add_argument("--abandon-intent", action="append", metavar="INTENT", help="operator reconciliation: no batch from this intent exists (or it was canceled); send again")
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
