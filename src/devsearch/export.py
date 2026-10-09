"""`python -m devsearch.export`: build an immutable release bundle that is reload-verified before it is published.

    export --config export.toml --mode baseline --index <index-manifest.json> --output <bundle>
    export --config export.toml --index <index-manifest.json> --checkpoint <checkpoint | model bundle> --output <candidate-bundle>
    export --config export.toml --candidate <candidate bundle | manifest> --final-report <final-report.json> --output <evaluated-bundle>

A baseline bundle holds no model and defaults to BM25. A candidate bundle holds a model without a final evaluation and
needs `--allow-candidate` to serve. An evaluated bundle needs a campaign's final report, measured on this very corpus and
index and (for reranking) selecting this very checkpoint. Everything verifies after a real reload before it appears at
`--output`; an identical rerun changes nothing and a different bundle at the same path is refused. The output must lie in the
runtime root and its parent must exist. Exit: 0 ok, 1 invalid input or failure, 2 usage.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser


def build_parser():
    parser = new_parser("export", "Export an immutable baseline, candidate or evaluated release bundle.")
    parser.add_argument("--config", type=Path, required=True, help="Export TOML (see configs/export.toml).")
    parser.add_argument("--mode", choices=("baseline",), help="baseline: export BM25 only (no checkpoint).")
    parser.add_argument("--index", type=Path, help="<dir>/index/index-manifest.json of a built index (its corpus sits beside it).")
    parser.add_argument("--checkpoint", type=Path, help="A training checkpoint manifest (.json) or an exported inference bundle directory.")
    parser.add_argument("--candidate", type=Path, help="An existing candidate (or baseline) release directory or its release-manifest.json.")
    parser.add_argument("--final-report", type=Path, help="The campaign's final-report.json (per-query.jsonl and report.md beside it).")
    parser.add_argument("--output", type=Path, required=True, help="New bundle directory (never overwritten; parent must exist).")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain every input and the output.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluated = args.candidate is not None or args.final_report is not None
    if evaluated:
        if args.candidate is None or args.final_report is None or args.index is not None or args.checkpoint is not None or args.mode is not None:
            parser.error("an evaluated export takes --candidate and --final-report only")
    elif args.index is None:
        parser.error("--index is required (or --candidate with --final-report)")
    elif (args.mode == "baseline") == (args.checkpoint is not None):
        parser.error("use --mode baseline (no checkpoint) or --checkpoint (a candidate with a model), not both and not neither")
    from devsearch.errors import DevSearchError
    try:
        summary = _run(args, evaluated)
    except (DevSearchError, OSError) as error:
        return fail("export", error)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run(args, evaluated: bool) -> dict:
    from devsearch.artifacts.paths import resolve_root
    from devsearch.execution.exporter import export_candidate, export_evaluated

    root = resolve_root(args.runtime_root, args.output.resolve().parent)
    if evaluated:
        return export_evaluated(config=args.config, candidate=args.candidate, final_report=args.final_report, output=args.output, runtime_root=root)
    return export_candidate(config=args.config, index_manifest=args.index, checkpoint=args.checkpoint, output=args.output, runtime_root=root)


if __name__ == "__main__":
    raise SystemExit(main())
