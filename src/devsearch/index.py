"""`python -m devsearch.index`: build a deterministic BM25 index over a corpus artifact.

Model-free and offline. The corpus manifest and the output manifest must live inside
one runtime root (`--runtime-root`, else DEVSEARCH_RUNTIME_ROOT, else the output
manifest's directory). The output directory must already exist; it is never created.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser
from devsearch.artifacts.paths import resolve_root
from devsearch.errors import DevSearchError
from devsearch.retrieval.config import load_index_config
from devsearch.retrieval.index_store import build_index


def build_parser():
    parser = new_parser(
        "index", "Build a versioned BM25 index over a validated corpus manifest.",
        epilog="Identical inputs rebuild to the same index_id and leave existing files untouched.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Index TOML (see configs/index.toml).")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus manifest JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Index manifest JSON to write.")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain every artifact path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_index_config(args.config)
        root = resolve_root(args.runtime_root, args.output.resolve())
        manifest, created = build_index(config, args.corpus, args.output, root)
    except (DevSearchError, OSError) as error:
        return fail("index", error)
    print(json.dumps({
        "status": "built" if created else "unchanged", "index_id": manifest["index_id"],
        "corpus_id": manifest["corpus"]["corpus_id"], "stats": manifest["stats"],
        "manifest": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
