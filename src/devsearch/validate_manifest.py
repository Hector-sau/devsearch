"""`python -m devsearch.validate_manifest`: verify an artifact without changing it.

Implemented kinds: `corpus` (records, hashes, offsets, IDs, source maps) and `index`
(references, hashes, tokenizer rules, and a fresh rebuild compared with the stored
postings). Other kinds, such as release bundles, are not implemented and fail.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser
from devsearch.artifacts.canonical import loads_strict, read_utf8
from devsearch.artifacts.paths import ensure_inside, resolve_root
from devsearch.artifacts.schema import as_object
from devsearch.data.corpus import load_corpus
from devsearch.errors import DevSearchError, ManifestError
from devsearch.retrieval.index_store import load_index, verify_index_deep


def build_parser():
    parser = new_parser("validate_manifest", "Validate a corpus or index manifest and its referenced files.")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSON to validate.")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain every artifact path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = args.manifest.resolve(strict=True)
        root = resolve_root(args.runtime_root, path)
        path = ensure_inside(path, root, what="manifest")
        kind = as_object(loads_strict(read_utf8(path, where="manifest"), where="manifest"), "manifest").get("kind")
        if kind == "corpus":
            corpus = load_corpus(path)
            summary = {"kind": kind, "id": corpus.corpus_id, "documents": len(corpus.docs), "chunks": len(corpus.chunks)}
        elif kind == "index":
            loaded = load_index(path, root)
            verify_index_deep(loaded)
            summary = {"kind": kind, "id": loaded.index_id, "corpus_id": loaded.corpus.corpus_id, **loaded.manifest["stats"]}
        else:
            raise ManifestError(f"unsupported manifest kind {kind!r}; only 'corpus' and 'index' are implemented")
    except (DevSearchError, OSError) as error:
        return fail("validate_manifest", error)
    print(json.dumps({"status": "valid", **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
