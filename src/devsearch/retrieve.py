"""`python -m devsearch.retrieve`: model-free BM25 search over a built index.

Only `--mode baseline` is implemented. `rerank` and `compare`, and any
`--checkpoint`, fail closed with status 2 until the reranker exists. Candidate
depth, display count, filters and limits come from the search config. The result
file must resolve (through symlinks) to a path inside the runtime root, is never
overwritten and its directory is never created.
"""

import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser
from devsearch.artifacts.canonical import sha256_file
from devsearch.artifacts.paths import ensure_inside, resolve_root
from devsearch.errors import DevSearchError
from devsearch.retrieval.config import load_search_config
from devsearch.retrieval.index_store import load_index
from devsearch.retrieval.search import run_baseline


def build_parser():
    parser = new_parser(
        "retrieve", "Search a built BM25 index (baseline mode; reranking is not implemented).",
    )
    parser.add_argument("--config", type=Path, required=True, help="Search TOML (see configs/search.toml).")
    parser.add_argument("--mode", required=True, choices=("baseline", "rerank", "compare"))
    parser.add_argument("--index", type=Path, required=True, help="Index manifest JSON.")
    parser.add_argument("--query", required=True, help="English technical question.")
    parser.add_argument("--output", type=Path, help="Write the result JSON here (inside the runtime root) instead of stdout.")
    parser.add_argument("--checkpoint", type=Path, help="Reranker checkpoint (not implemented).")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain every artifact path.")
    return parser


def _write_new(path: Path, text: str) -> None:
    if not path.parent.is_dir():
        raise DevSearchError(f"output directory does not exist: {path.parent}")
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise DevSearchError(f"{path} already exists; refusing to overwrite") from None
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)      # never leave a truncated result that blocks reruns
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode != "baseline" or args.checkpoint is not None:
        print(
            "devsearch.retrieve: reranking is not implemented; only --mode baseline without "
            "--checkpoint runs. No operation was run. / 重排序尚未实现，仅支持无 checkpoint 的 baseline。",
            file=sys.stderr,
        )
        return 2
    try:
        limits = load_search_config(args.config)
        root = resolve_root(args.runtime_root, args.index.resolve())
        # F03: results are runtime artifacts too. Resolve symlinks (parents and the final
        # component) and require the destination to stay inside the root before any work.
        output = None if args.output is None else ensure_inside(args.output, root, what="result output")
        loaded = load_index(args.index, root)
        response = run_baseline(loaded, limits.to_request(args.query, "baseline"), limits)
        response["config_sha256"] = sha256_file(args.config)
        text = json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if output is None:
            sys.stdout.write(text)
        else:
            _write_new(output, text)
            print(f"wrote {output}: status={response['status']} candidates={response['candidate_count']}")
    except (DevSearchError, OSError) as error:
        return fail("retrieve", error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
