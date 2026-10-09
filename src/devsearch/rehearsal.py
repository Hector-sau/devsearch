"""`python -m devsearch.rehearsal`: build a complete DIAGNOSTIC bundle for an end-to-end engineering rehearsal.

Synthetic documents and questions, a scripted oracle standing in for a reviewer and a random tiny encoder: it proves
the pipeline runs (prepare, doctor, plan, run, export, reload, backup), never anything about ranking quality, and its
reports are stamped as diagnostic. The output directory must exist and be empty. Exit: 0 built, 1 failed, 2 usage.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import new_parser


def build_parser():
    parser = new_parser("rehearsal", "Build a diagnostic rehearsal tree (synthetic, CPU, no network).")
    parser.add_argument("--output", type=Path, required=True, help="An existing empty directory.")
    parser.add_argument("--topics", type=int, default=60, help="Synthetic documents (3 evidence groups each, one question each); at least 18.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--signal", choices=("planted", "none"), default="planted",
                        help="planted: answer sections carry a cue a reranker can learn (exercises the 'reranker ships' path); none: BM25 stays the default.")
    parser.add_argument("--protocol", choices=("prejudged_fixed_candidates_v1", "postrank_pool_v1"), default="prejudged_fixed_candidates_v1")
    parser.add_argument("--assets", type=Path, help="A registered model-assets directory to copy instead of the random tiny encoder (the target-host check).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import DevSearchError
    try:
        from devsearch.execution.rehearsal import build_rehearsal
        summary = build_rehearsal(args.output, topics=args.topics, seed=args.seed, signal=args.signal, protocol=args.protocol, assets=args.assets)
    except (DevSearchError, OSError) as error:
        print(f"devsearch.rehearsal: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
