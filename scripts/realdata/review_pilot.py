"""Pilot the candidate review on a small stratified sample before judging the whole pool (docs/02 two-stage sealing).

    review_pilot.py sample  --review-dir evaluation/review --output-dir EMPTY [--positives 16] [--top 32] [--deep 32] [--batch-size 40] [--seed 20261005]
    review_pilot.py convert --pilot-dir PILOT --review-dir evaluation/review
    review_pilot.py report  --pilot-dir PILOT --review-dir evaluation/review [--run-cost BATCH:JUDGE:TOKENS:SECONDS ...]

`sample` draws items from the blinded request in three strata that only the operator sees (hidden provenance): known positives, other
candidates ranked 1-10, other candidates ranked 11-50. The judge batches hold the request records unchanged (blinded: no rank, score,
system or prior label), shuffled together; the strata go to a private file. `convert` turns each judge's lines (`judge-<ID>/<batch>.jsonl`,
written with `prompts/judge-candidate-v1.md`) into bank response records: an exact evidence quote becomes a passage-local span, `unsure` or a
quote that cannot be found becomes an unresolved judgment, and every record is checked with the bank's own `parse_response`. `report`
measures agreement between judges per stratum (raw agreement, Cohen's kappa, unresolved), how often each judge recognises the known
positives, and, from the measured cost of each judge run, extrapolates the cost of judging the whole pool twice plus adjudicating the
expected disagreements. Everything is written to the pilot directory, outside the evaluation tree; nothing here can seal anything, and
pilot judgments are agent judgments, labelled as such.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from devsearch.data.review import parse_response  # noqa: E402

HERE = Path(__file__).resolve().parent
PROMPT = HERE / "prompts" / "judge-candidate-v1.md"
STRATA = ("known_positive", "rank_1_10", "rank_11_50")


def _locate():
    spec = importlib.util.spec_from_file_location("realdata_validate_generations", HERE / "validate_generations.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.locate


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def stratum(provenance: dict) -> str:
    if provenance.get("known_positive"):
        return "known_positive"
    return "rank_1_10" if (provenance.get("candidate_rank") or 99) <= 10 else "rank_11_50"


def load_hidden(review_dir: Path) -> dict[str, dict]:
    return {r["item_id"]: r for r in read_jsonl(review_dir / "hidden-provenance.jsonl") if r["item_id"] != "_pool"}


def sample(args) -> int:
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} must be an existing, empty directory")
    hidden = load_hidden(args.review_dir)
    request = {r["item_id"]: r for r in read_jsonl(args.review_dir / "review-request.jsonl")}
    by_stratum = defaultdict(list)
    for item_id in sorted(request):
        by_stratum[stratum(hidden[item_id]["provenance"])].append(item_id)
    rng = random.Random(args.seed)
    wanted = {"known_positive": args.positives, "rank_1_10": args.top, "rank_11_50": args.deep}
    chosen = {name: rng.sample(by_stratum[name], min(wanted[name], len(by_stratum[name]))) for name in STRATA}
    order = [item for name in STRATA for item in chosen[name]]
    rng.shuffle(order)
    batches = args.output_dir / "batches"
    batches.mkdir()
    for number, start in enumerate(range(0, len(order), args.batch_size), start=1):
        write_jsonl(batches / f"pilot-{number:03d}.jsonl", [request[i] for i in order[start:start + args.batch_size]])
    write_jsonl(args.output_dir / "pilot-strata.private.jsonl", [{"item_id": i, "stratum": stratum(hidden[i]["provenance"])} for i in sorted(order)])
    summary = {"seed": args.seed, "items": len(order), "per_stratum": {n: len(chosen[n]) for n in STRATA},
               "pool_per_stratum": {n: len(by_stratum[n]) for n in STRATA}, "batches": len(list(batches.iterdir())),
               "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest()}
    (args.output_dir / "pilot-sample.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def to_response(line: dict, passage: str, judge: str, revision: str, stamp: str, locate) -> tuple[dict, str | None]:
    """A bank response record for one judge line, and why it is unresolved (None when resolved)."""
    relevance, quote = line.get("relevance"), line.get("evidence_quote") or ""
    note = str(line.get("note") or "")[:500]
    record = {"item_id": line["item_id"], "role": "primary", "relevance": None, "status": "unresolved", "evidence_spans": [],
              "reviewer": {"type": "agent", "id": f"agent-judge-{judge}", "revision": revision}, "reviewed_at": stamp, "notes": note}
    if relevance == 0 and not quote:
        return {**record, "relevance": 0, "status": "resolved"}, None
    if relevance == 1:
        span = locate(quote, passage) if quote and len(quote) <= 400 else None
        if span is None:
            return {**record, "notes": (note + " | evidence quote not found in the passage").strip(" |")}, "quote not found"
        return {**record, "relevance": 1, "status": "resolved", "evidence_spans": [{"start": span[0], "end": span[1]}]}, None
    if relevance == "unsure":
        return record, "unsure"
    return {**record, "notes": (note + f" | unusable judge line: relevance {relevance!r}").strip(" |")}, "malformed"


def convert(args) -> int:
    locate = _locate()
    pilot = args.pilot_dir
    passages = {}
    for path in sorted((pilot / "batches").glob("*.jsonl")):
        for record in read_jsonl(path):
            passages[record["item_id"]] = record["passage"]["text"]
    lengths = {item: len(text) for item, text in passages.items()}
    revision = "judge-candidate-v1:" + hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:16]
    summary = {}
    for judge_dir in sorted(pilot.glob("judge-*")):
        judge = judge_dir.name.removeprefix("judge-")
        responses, reasons, seen = [], Counter(), set()
        for path in sorted(judge_dir.glob("*.jsonl")):
            stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            for line in read_jsonl(path):
                if line.get("item_id") not in passages or line["item_id"] in seen:
                    reasons["unknown or repeated item"] += 1
                    continue
                seen.add(line["item_id"])
                record, reason = to_response(line, passages[line["item_id"]], judge, revision, stamp, locate)
                parse_response(record, f"{path.name}:{line['item_id']}", lengths)          # the bank's own validation
                responses.append(record)
                if reason:
                    reasons[reason] += 1
        missing = sorted(set(passages) - seen)
        write_jsonl(pilot / f"responses-{judge}.jsonl", responses)
        summary[judge] = {"responses": len(responses), "missing": len(missing), "unresolved_reasons": dict(reasons)}
    print(json.dumps(summary, indent=2))
    return 0


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    p_a, p_b = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def report(args) -> int:
    pilot = args.pilot_dir
    strata = {r["item_id"]: r["stratum"] for r in read_jsonl(pilot / "pilot-strata.private.jsonl")}
    judges = {p.stem.removeprefix("responses-"): {r["item_id"]: r for r in read_jsonl(p)} for p in sorted(pilot.glob("responses-*.jsonl"))}
    if len(judges) != 2:
        raise SystemExit(f"the report compares exactly two judges; found {sorted(judges)}")
    (a_name, a), (b_name, b) = sorted(judges.items())
    per = {}
    needs_adjudication = Counter()
    for name in STRATA:
        items = [i for i, s in strata.items() if s == name]
        both = [(a[i]["relevance"], b[i]["relevance"]) for i in items if i in a and i in b and a[i]["status"] == b[i]["status"] == "resolved"]
        unresolved = sum(1 for i in items if a.get(i, {}).get("status") != "resolved" or b.get(i, {}).get("status") != "resolved")
        disagree = sum(1 for x, y in both if x != y)
        needs_adjudication[name] = unresolved + disagree
        per[name] = {"items": len(items), "both_resolved": len(both), "agreement": round(sum(1 for x, y in both if x == y) / len(both), 3) if both else None,
                     "kappa": None if kappa(both) is None else round(kappa(both), 3), "disagreements": disagree, "unresolved_by_either": unresolved,
                     f"positive_rate_{a_name}": round(sum(1 for i in items if a.get(i, {}).get("relevance") == 1) / len(items), 3) if items else None,
                     f"positive_rate_{b_name}": round(sum(1 for i in items if b.get(i, {}).get("relevance") == 1) / len(items), 3) if items else None}
    all_pairs = [(a[i]["relevance"], b[i]["relevance"]) for i in strata if i in a and i in b and a[i]["status"] == b[i]["status"] == "resolved"]
    pool = json.loads((pilot / "pilot-sample.json").read_text())["pool_per_stratum"]
    expected = {name: round(needs_adjudication[name] / per[name]["items"] * pool[name]) if per[name]["items"] else None for name in STRATA}
    out = {"labels": "agent judgments (two runs of the same agent workflow, not independent humans); pilot only, never part of a bank or seal",
           "judges": [a_name, b_name], "per_stratum": per, "overall_kappa": None if kappa(all_pairs) is None else round(kappa(all_pairs), 3),
           "expected_items_needing_adjudication_in_full_pool": expected, "full_pool_items": sum(pool.values())}
    if args.run_cost:
        runs = [dict(zip(("batch", "judge", "tokens", "seconds"), part.split(":"))) for part in args.run_cost]
        sizes = {p.stem: len(read_jsonl(p)) for p in (pilot / "batches").glob("*.jsonl")}
        items = sum(sizes[r["batch"]] for r in runs)
        tokens, seconds = sum(int(r["tokens"]) for r in runs), sum(float(r["seconds"]) for r in runs)
        per_item_tokens, per_item_seconds = tokens / items, seconds / items
        total_items = sum(pool.values())
        adjudications = sum(v for v in expected.values() if v)
        judgements = 2 * total_items + adjudications
        out["cost"] = {"label": "measured on the pilot runs, then EXTRAPOLATED to the full pool", "pilot_runs": len(runs), "pilot_judgements": items,
                       "tokens_per_judgement": round(per_item_tokens), "seconds_per_judgement_per_run": round(per_item_seconds, 2),
                       "full_pool_judgements_two_primaries_plus_adjudication": judgements,
                       "estimated_tokens": round(judgements * per_item_tokens), "estimated_agent_runs_at_40_items": -(-judgements // 40),
                       "estimated_serial_hours": round(judgements * per_item_seconds / 3600, 1)}
    (pilot / "pilot-report.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--review-dir", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--positives", type=int, default=16)
    s.add_argument("--top", type=int, default=32)
    s.add_argument("--deep", type=int, default=32)
    s.add_argument("--batch-size", type=int, default=40)
    s.add_argument("--seed", type=int, default=20261005)
    for name in ("convert", "report"):
        p = sub.add_parser(name)
        p.add_argument("--pilot-dir", type=Path, required=True)
        p.add_argument("--review-dir", type=Path, required=True)
        if name == "report":
            p.add_argument("--run-cost", action="append", help="BATCH:JUDGE:TOKENS:SECONDS of one judge run, from the run's own usage report")
    args = parser.parse_args()
    return {"sample": sample, "convert": convert, "report": report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
