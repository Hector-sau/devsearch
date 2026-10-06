"""`python -m devsearch.review`: judge batches, judgment import and the coverage audit for the candidate review (docs/14 section 3).

  batches  --request R --output-dir EMPTY --pass NAME [--batch-size 40] [--seed 20261005] [--items FILE]
           Compact, blinded judge batches (item id, question, passage title/section/text) in a seeded order per pass.
           `--items` limits them to the item ids listed in FILE (one per line or `audit.json`'s queue), for adjudication.
  import   --request R --judgments FILE [...] --reviewer-id ID --reviewer-type agent|human --role primary|adjudicator
           [--revision REV] [--reviewed-at UTC] --output responses/NAME.jsonl
           Judgment lines -> validated bank responses; `unsure` or an unlocatable quote stays unresolved. Never overwrites.
  progress --request R --batches-dir D --responses-dir D [--reviewer-id ID --pass NAME [--budget-tokens N] [--limit N]] [--output progress.json]
           Resumable progress from the imported responses: per reviewer, per batch; with --reviewer-id the next unjudged batches
           of a pass, cut off at a token budget (estimate: measured pilot cost per judgment) or a batch count.
  audit    --request R --responses-dir DIR --output audit.json [--min-primary-reviews 2] [--merged all-responses.jsonl]
           Coverage of the whole pool. Exit 0 only when every item has a final resolved judgment, and only then is the merged
           file written for `prepare_data --stage bank`; exit 3 lists what is missing or needs adjudication.

Pilot judgments use reviewer ids starting with "pilot" (`import --pilot`), are written outside the formal responses directory and are
refused by `audit`: they never count towards the formal bank.

Exit: 0 success; 1 invalid input; 2 usage; 3 coverage incomplete (sealing and formal evaluation stay blocked).
"""

import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from devsearch._cli import new_parser


def build_parser():
    parser = new_parser("review", "Judge batches, judgment import and the coverage audit for the candidate review.", epilog=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    b = sub.add_parser("batches")
    b.add_argument("--request", type=Path, required=True, help="review-request.jsonl from prepare_data materialize")
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--pass", dest="pass_name", required=True)
    b.add_argument("--batch-size", type=int, default=40)
    b.add_argument("--seed", type=int, default=20261005)
    b.add_argument("--items", type=Path, help="restrict to these item ids (a text file or an audit.json, whose adjudication queue is used)")
    i = sub.add_parser("import")
    i.add_argument("--request", type=Path, required=True)
    i.add_argument("--judgments", type=Path, action="append", required=True)
    i.add_argument("--reviewer-id", required=True)
    i.add_argument("--reviewer-type", choices=("agent", "human"), required=True)
    i.add_argument("--role", choices=("primary", "adjudicator"), required=True)
    i.add_argument("--revision", help="agent reviewers: the prompt/workflow revision (required); humans carry none")
    i.add_argument("--reviewed-at", help="UTC time of the judgments, YYYY-MM-DDTHH:MM:SSZ (default: newest judgment file's time)")
    i.add_argument("--output", type=Path, required=True)
    i.add_argument("--pilot", action="store_true", help="a pilot import: the reviewer id must start with 'pilot'; never counted by `audit`")
    g = sub.add_parser("progress")
    g.add_argument("--request", type=Path, required=True)
    g.add_argument("--batches-dir", type=Path, required=True)
    g.add_argument("--responses-dir", type=Path, required=True)
    g.add_argument("--reviewer-id")
    g.add_argument("--pass", dest="pass_name")
    g.add_argument("--budget-tokens", type=int, help="a usage limit: list only the next batches that fit this token estimate")
    g.add_argument("--limit", type=int, help="a usage limit: at most this many batches")
    g.add_argument("--tokens-per-judgment", type=int, default=1738)
    g.add_argument("--output", type=Path)
    a = sub.add_parser("audit")
    a.add_argument("--request", type=Path, required=True)
    a.add_argument("--responses-dir", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--min-primary-reviews", type=int, default=2)
    a.add_argument("--merged", type=Path, help="where to write the merged responses when the audit is complete")
    return parser


def _items(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        report = json.loads(text)
        return set(report["adjudication_queue"])
    return {line.strip() for line in text.splitlines() if line.strip()}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import DevSearchError
    from devsearch.data import review_workflow as wf
    try:
        request = wf.load_request(args.request)
        if args.action == "batches":
            if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
                raise DevSearchError(f"{args.output_dir} must be an existing, empty directory")
            if args.items:
                wanted = _items(args.items)
                unknown = wanted - set(request)
                if unknown:
                    raise DevSearchError(f"{len(unknown)} listed item(s) are not in the request")
                request = {item: record for item, record in request.items() if item in wanted}
            batches = wf.judge_batches(request, pass_name=args.pass_name, batch_size=args.batch_size, seed=args.seed)
            for number, batch in enumerate(batches, start=1):
                (args.output_dir / f"{args.pass_name}-{number:04d}.jsonl").write_text(
                    "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in batch), encoding="utf-8")
            print(json.dumps({"pass": args.pass_name, "items": len(request), "batches": len(batches)}, indent=2))
            return 0
        if args.action == "import":
            if args.output.exists():
                raise DevSearchError(f"{args.output} exists; imported judgments are never overwritten")
            if args.pilot != args.reviewer_id.startswith(wf.PILOT_PREFIX):
                raise DevSearchError(f"pilot imports (--pilot) need a reviewer id starting with {wf.PILOT_PREFIX!r}, and formal imports must not use one")
            if args.pilot and args.output.parent.name == "responses":
                raise DevSearchError("a pilot import must not be written into a 'responses' directory (the formal bank's)")
            if args.reviewer_type == "agent" and not args.revision:
                raise DevSearchError("an agent reviewer needs --revision (the prompt or workflow revision)")
            if args.reviewer_type == "human" and args.revision:
                raise DevSearchError("a human reviewer carries no --revision")
            stamp = args.reviewed_at or datetime.fromtimestamp(max(p.stat().st_mtime for p in args.judgments), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            lines = [row for path in args.judgments for row in wf.read_jsonl(path)]
            responses, reasons = wf.import_judgments(lines, request, reviewer_type=args.reviewer_type, reviewer_id=args.reviewer_id,
                                                     revision=args.revision, role=args.role, reviewed_at=stamp)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in responses), encoding="utf-8")
            print(json.dumps({"output": str(args.output), "responses": len(responses), "outcomes": reasons}, indent=2))
            return 0
        responses = [row for path in sorted(args.responses_dir.glob("*.jsonl")) for row in wf.read_jsonl(path)]
        if args.action == "progress":
            batches = {path.name: [row["item_id"] for row in wf.read_jsonl(path)] for path in sorted(args.batches_dir.glob("*.jsonl"))}
            report = wf.progress(request, batches, responses, tokens_per_judgment=args.tokens_per_judgment)
            if args.reviewer_id:
                if not args.pass_name:
                    raise DevSearchError("--reviewer-id needs --pass (the batch file prefix, e.g. A)")
                matches = [name for name in report["reviewers"] if name.startswith(args.reviewer_id + " (")]
                reviewer = matches[0] if matches else f"{args.reviewer_id} (primary)"
                report["next"] = wf.next_batches(report, reviewer=reviewer, prefix=args.pass_name, budget_tokens=args.budget_tokens, limit=args.limit)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")       # derived and recomputable: replaced each time
            print(json.dumps({k: v for k, v in report.items() if k != "batches"} | {"batches_total": len(report["batches"]),
                                                                                    "batches_done_by_any": sum(1 for b in report["batches"].values() if b["covered_by"])}, indent=2))
            return 0
        report = wf.audit(request, responses, min_primary_reviews=args.min_primary_reviews)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary = {k: report[k] for k in ("items", "final", "coverage", "complete", "states")}
        if report["complete"] and args.merged:
            if args.merged.exists():
                raise DevSearchError(f"{args.merged} exists; the merged responses are never overwritten")
            args.merged.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in responses), encoding="utf-8")
            summary["merged"] = str(args.merged)
        print(json.dumps(summary, indent=2))
        if not report["complete"]:
            print(f"devsearch.review: coverage incomplete: {report['final']} of {report['items']} items have a final judgment; "
                  "sealing and formal evaluation stay blocked", file=sys.stderr)
            return 3
        return 0
    except (DevSearchError, OSError, ValueError, KeyError) as error:
        print(f"devsearch.review: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
