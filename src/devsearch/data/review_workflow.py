"""Production candidate review around the blinded request (docs/02 two-stage sealing, docs/14 section 3).

The review pool and its blinded `review-request.jsonl` come from `prepare_data --stage materialize`. This module turns them into
judge batches, imports judgments, audits coverage and builds the adjudication queue. It never decides relevance and never fills a gap:

* Judge batches carry only what a reviewer needs (item id, question, passage title, section and text), in a seeded order per pass,
  so two passes see the items in different orders. They never carry rank, score, system, provenance or an earlier label.
* A judgment line is `{"item_id", "relevance": 1 | 0 | "unsure", "evidence_quote", "note"}`. An exact quote (whitespace-tolerant)
  becomes a passage-local evidence span; `unsure`, a quote that is not in the passage or a malformed line becomes an UNRESOLVED
  response, never a 0. Every response is checked with the bank's own `parse_response` before anything is written.
* The audit requires, for every item of the pool, either an adjudicator's resolved judgment or resolved, unanimous primary judgments
  from at least `min_primary_reviews` distinct reviewers (the declared protocol: two). Anything else is listed by reason; disagreements
  and unresolved items go to a blinded adjudication queue. Only a complete audit writes the merged responses the bank stage reads.
"""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from devsearch.data.review import RESPONSE_FIELDS, assert_blinded, parse_response
from devsearch.errors import ManifestError

MAX_QUOTE = 400


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Where an exact quote sits in the passage, allowing any run of whitespace to match any other run; None when absent."""
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(word) for word in words), text)
    return (match.start(), match.end()) if match else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError as error:
                raise ManifestError(f"{path}:{number}: not JSON ({error})") from error
    return rows


def load_request(path: Path) -> dict[str, dict[str, Any]]:
    """The blinded request records keyed by item id, after re-checking that they leak nothing."""
    records = read_jsonl(path)
    assert_blinded(records)
    request = {}
    for record in records:
        item_id = record.get("item_id")
        if not isinstance(item_id, str) or item_id in request:
            raise ManifestError(f"{path}: missing or repeated item_id {item_id!r}")
        request[item_id] = record
    return request


def compact(record: Mapping[str, Any]) -> dict[str, Any]:
    passage = record["passage"]
    return {"item_id": record["item_id"], "question": record["query"]["text"], "title": passage.get("title"), "section": passage.get("section"),
            "text": passage["text"]}


def judge_batches(request: Mapping[str, Mapping[str, Any]], *, pass_name: str, batch_size: int, seed: int) -> list[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ManifestError("batch_size must be positive")
    order = sorted(request, key=lambda item: hashlib.sha256(f"{seed}|{pass_name}|{item}".encode()).hexdigest())
    return [[compact(request[item]) for item in order[start:start + batch_size]] for start in range(0, len(order), batch_size)]


def to_response(line: Mapping[str, Any], passage: str, *, reviewer_type: str, reviewer_id: str, revision: str | None, role: str,
                reviewed_at: str) -> tuple[dict[str, Any], str | None]:
    """A bank response record for one judgment line, and why it is unresolved (None when resolved)."""
    relevance, quote = line.get("relevance"), line.get("evidence_quote") or ""
    note = str(line.get("note") or "")[:500]
    record = {"item_id": line["item_id"], "role": role, "relevance": None, "status": "unresolved", "evidence_spans": [],
              "reviewer": {"type": reviewer_type, "id": reviewer_id, "revision": revision}, "reviewed_at": reviewed_at, "notes": note}
    if relevance == 0 and not quote:
        return {**record, "relevance": 0, "status": "resolved"}, None
    if relevance == 1:
        span = locate(quote, passage) if quote and len(quote) <= MAX_QUOTE else None
        if span is None:
            return {**record, "notes": (note + " | evidence quote not found in the passage").strip(" |")}, "quote not found"
        return {**record, "relevance": 1, "status": "resolved", "evidence_spans": [{"start": span[0], "end": span[1]}]}, None
    if relevance == "unsure":
        return record, "unsure"
    return {**record, "notes": (note + f" | unusable judgment line: relevance {relevance!r}").strip(" |")}, "malformed"


def import_judgments(lines: Iterable[Mapping[str, Any]], request: Mapping[str, Mapping[str, Any]], *, reviewer_type: str, reviewer_id: str,
                     revision: str | None, role: str, reviewed_at: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validated response records for one reviewer's lines; an unknown or repeated item is an error, not something to skip."""
    lengths = {item: len(record["passage"]["text"]) for item, record in request.items()}
    responses, reasons, seen = [], defaultdict(int), set()
    for number, line in enumerate(lines, start=1):
        item = line.get("item_id") if isinstance(line, Mapping) else None
        if item not in request:
            raise ManifestError(f"judgment line {number}: item {item!r} is not in the review request")
        if item in seen:
            raise ManifestError(f"judgment line {number}: item {item} is judged twice by {reviewer_id}")
        seen.add(item)
        record, reason = to_response(line, request[item]["passage"]["text"], reviewer_type=reviewer_type, reviewer_id=reviewer_id,
                                     revision=revision, role=role, reviewed_at=reviewed_at)
        parse_response(record, f"judgment line {number}", lengths)
        responses.append(record)
        reasons["resolved" if reason is None else reason] += 1
    return responses, dict(reasons)


def audit(request: Mapping[str, Mapping[str, Any]], responses: Sequence[Mapping[str, Any]], *, min_primary_reviews: int = 2) -> dict[str, Any]:
    """Coverage of the whole pool under the declared rule. `complete` is true only when every item has a final resolved judgment."""
    lengths = {item: len(record["passage"]["text"]) for item, record in request.items()}
    grouped: dict[str, list] = defaultdict(list)
    for number, record in enumerate(responses, start=1):
        if not isinstance(record, Mapping) or set(record) != set(RESPONSE_FIELDS):
            raise ManifestError(f"response {number}: not a response record")
        parsed = parse_response(record, f"response {number}", lengths)
        grouped[parsed.item_id].append(parsed)
    states: dict[str, list[str]] = defaultdict(list)
    reviewers = defaultdict(int)
    for item in sorted(request):
        mine = grouped.get(item, [])
        for response in mine:
            reviewers[f"{response.reviewer_type}:{response.reviewer_id}:{response.role}"] += 1
        adjudicators = [r for r in mine if r.role == "adjudicator"]
        primaries: dict[str, Any] = {}
        for response in mine:
            if response.role == "primary":
                primaries.setdefault(response.reviewer_id, response)       # one vote per reviewer, the first recorded
        if len(adjudicators) > 1:
            states["more than one adjudicator"].append(item)
        elif adjudicators:
            states["final: adjudicated" if adjudicators[0].resolved else "adjudicator left it unresolved"].append(item)
        elif len(primaries) < min_primary_reviews:
            states[f"needs primary reviews ({len(primaries)} of {min_primary_reviews})"].append(item)
        elif not all(r.resolved for r in primaries.values()):
            states["needs adjudication: unresolved primary"].append(item)
        elif len({r.relevance for r in primaries.values()}) != 1:
            states["needs adjudication: primaries disagree"].append(item)
        else:
            states["final: unanimous primaries"].append(item)
    final = sum(len(items) for state, items in states.items() if state.startswith("final"))
    return {"items": len(request), "final": final, "coverage": round(final / len(request), 4) if request else 0.0, "complete": final == len(request),
            "min_primary_reviews": min_primary_reviews, "states": {state: len(items) for state, items in sorted(states.items())},
            "adjudication_queue": sorted(item for state, items in states.items() if state.startswith("needs adjudication") or state == "adjudicator left it unresolved"
                                         for item in items),
            "needs_primary": sorted(item for state, items in states.items() if state.startswith("needs primary") for item in items),
            "responses_by_reviewer": dict(sorted(reviewers.items()))}
