"""`python -m devsearch.bundle`: seal and verify the prepared bundle that a campaign consumes.

    bundle assemble --bundle DIR [--lock FILE]     verify every part, write bundle-manifest.json (never overwrites a different one)
    bundle verify   --bundle DIR                   re-verify every part against the sealed manifest
    bundle register-assets --directory DIR --repo OWNER/NAME --revision COMMIT [--license ID]
                                                   hash already-present encoder and tokenizer files into assets-manifest.json

The bundle holds the training-visible inputs only (prepared data, pinned model assets, frozen protocol, initial seal);
test material lives in the separate evaluation root. `register-assets` downloads nothing: the files (`config.json`,
`model.safetensors`, a tokenizer) must already be in DIR, copied from the pinned upstream revision on a connected machine; a
40-hex commit revision is required (a branch name is refused) and the manifest records what the operator declares, because no
code can prove the files really are that upstream model. Exit: 0 ok, 1 invalid or modified, 2 usage.
"""

import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import new_parser


def build_parser():
    parser = new_parser("bundle", "Assemble or verify a prepared bundle.")
    parser.add_argument("action", choices=("assemble", "verify", "register-assets"))
    parser.add_argument("--bundle", type=Path, help="assemble / verify: the bundle directory.")
    parser.add_argument("--lock", type=Path, help="assemble: the platform dependency lock file whose hash is recorded.")
    parser.add_argument("--directory", type=Path, help="register-assets: the existing directory holding the pinned encoder and tokenizer files.")
    parser.add_argument("--repo", help="register-assets: upstream repository the files came from, e.g. microsoft/MiniLM-L12-H384-uncased.")
    parser.add_argument("--revision", help="register-assets: the full 40-hex upstream commit of those files.")
    parser.add_argument("--license", help="register-assets: declared licence identifier (recorded, not verified).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "register-assets":
        if args.directory is None or not args.repo or not args.revision:
            parser.error("register-assets needs --directory, --repo and --revision")
        return _register(args)
    if args.bundle is None:
        parser.error(f"{args.action} needs --bundle")
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


def _register(args) -> int:
    from devsearch.errors import DevSearchError, ManifestError
    from devsearch.modeling.assets import register_assets
    try:
        source = {"repo": args.repo, "revision": args.revision}
        if args.license:
            source["license"] = args.license
        if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
            raise ManifestError("--revision must be a full 40-hex commit id; a branch or tag is not a pinned revision")
        manifest = register_assets(args.directory, asset_kind="pretrained", source=source)
    except (DevSearchError, OSError) as error:
        print(f"devsearch.bundle: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"action": "register-assets", "status": "ok", "manifest_sha256": manifest["manifest_sha256"], "pinned": manifest["pinned"],
                      "asset_kind": manifest["asset_kind"], "files": sorted(manifest["files"]), "source": manifest["source"],
                      "note": "the files are hashed as found; nothing here proves they are the upstream model's files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
