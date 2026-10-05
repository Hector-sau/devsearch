"""`python -m devsearch.bundle`: seal and verify the prepared bundle that a campaign consumes.

    bundle assemble --bundle DIR [--lock FILE]     verify every part, write bundle-manifest.json (never overwrites a different one)
    bundle verify   --bundle DIR                   re-verify every part against the sealed manifest

The bundle holds the training-visible inputs only (prepared data, pinned model assets, frozen protocol, initial seal);
test material lives in the separate evaluation root. Exit: 0 ok, 1 invalid or modified, 2 usage.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import new_parser


def build_parser():
    parser = new_parser("bundle", "Assemble or verify a prepared bundle.")
    parser.add_argument("action", choices=("assemble", "verify"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--lock", type=Path, help="assemble: the platform dependency lock file whose hash is recorded.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import DevSearchError
    try:
        from devsearch.execution.bundle import assemble_bundle, verify_bundle
        if args.action == "assemble":
            manifest = assemble_bundle(args.bundle, lock_file=args.lock)
        else:
            manifest = verify_bundle(args.bundle)["manifest"]
    except (DevSearchError, OSError) as error:
        print(f"devsearch.bundle: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"action": args.action, "status": "ok", "manifest_sha256": manifest["manifest_sha256"], "diagnostic": manifest["diagnostic"],
                      "corpus_id": manifest["prepared"]["corpus_id"], "protocol_id": manifest["protocol"]["protocol_id"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
