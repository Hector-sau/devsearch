"""`python -m devsearch.status`: a verified, read-only view of a campaign. It re-checks committed artifacts; it never
trusts `run-state.json` alone. Exit: 0 complete and verified, 5 awaiting review, 6 stopped on budget, 4 failed or
interrupted or incomplete, 1 invalid plan or artifact.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import new_parser


def build_parser():
    parser = new_parser("status", "Show the verified state of a campaign.")
    parser.add_argument("--plan", type=Path, required=True, help="execution-plan.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import DevSearchError
    try:
        from devsearch.execution.runner import open_campaign
        campaign, *_ = open_campaign(args.plan)
        report = campaign.status_report()
    except (DevSearchError, OSError) as error:
        print(f"devsearch.status: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["complete"]:
        return 0
    return {"awaiting_review": 5, "stopped_budget": 6}.get(report["status"], 4)


if __name__ == "__main__":
    raise SystemExit(main())
