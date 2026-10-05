"""`python -m devsearch.run`: execute (or resume) a locked plan. Exit: 0 complete, 1 invalid plan/artifact, 2 usage,
3 prerequisite unavailable, 4 execution failure, 5 awaiting review, 6 budget stop. Never a shell success for an incomplete run.

The run changes only run-state and artifacts, never the plan or the source. Formal compute is offline: no model or data
download, question generation or dependency installation happens here.
"""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import new_parser

OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def build_parser():
    parser = new_parser("run", "Execute a locked execution plan, or resume it from verified state.")
    parser.add_argument("--plan", type=Path, required=True, help="execution-plan.json written by `devsearch.plan`.")
    parser.add_argument("--resume", action="store_true", help="Continue an existing campaign (reuses only hash-valid stages).")
    parser.add_argument("--reviews", type=Path, help="Post-rank protocol only: the reviewed judgments for the pooled review request.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.update(OFFLINE)
    from devsearch.errors import DevSearchError
    try:
        from devsearch.execution.runner import open_campaign
        campaign, initial, active, _plan, _binding = open_campaign(args.plan, reviews=args.reviews)
        return campaign.run(resume=args.resume, initial_binding_sha256=initial, active_binding_sha256=active)
    except (DevSearchError, OSError) as error:
        print(f"devsearch.run: error: {error}", file=sys.stderr)
        from devsearch.errors import PrerequisiteError
        return 3 if isinstance(error, PrerequisiteError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
