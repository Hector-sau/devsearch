"""Check that the source links a BM25 search returns point at the lines the passage came from.

    check_source_links.py --index <index-manifest.json or directory> --search search.toml --queries queries.txt --snapshot-root DIR
                          [--sample 30] [--seed 20261005] [--runtime-root R]

For a seeded sample of queries (plain text, one per line) the baseline search runs and every returned result is checked: each reported
`source_location` must name a file that exists under the snapshot root, its 1-based line range must lie inside that file, and the words
of the passage (five or more letters) must appear in those lines at the rate `--min-overlap` demands (default 0.9; roles, links and
markup change some spellings, so a perfect match is not required). Prints a JSON report and exits 1 when any result fails. This checks
the link, not the relevance of the result.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from devsearch.artifacts.paths import resolve_root  # noqa: E402
from devsearch.contracts import SearchRequest  # noqa: E402
from devsearch.retrieval.config import load_search_config  # noqa: E402
from devsearch.retrieval.index_store import load_index  # noqa: E402
from devsearch.serving.pipeline import SearchPipeline  # noqa: E402

WORD = re.compile(r"[A-Za-z]{5,}")


def link_problem(text: str, source_lines: list[str], start: int, end: int, min_overlap: float) -> str | None:
    """None when the lines [start, end] (1-based, inclusive) exist and contain the passage's words, else the reason."""
    if not 1 <= start <= end <= len(source_lines):
        return f"lines {start}-{end} are outside a file of {len(source_lines)} lines"
    words = {w.lower() for w in WORD.findall(text)}
    if not words:
        return None
    haystack = " ".join(source_lines[start - 1:end]).lower()
    found = sum(1 for w in words if w in haystack)
    if found / len(words) < min_overlap:
        return f"only {found} of {len(words)} passage words occur in lines {start}-{end}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--min-overlap", type=float, default=0.9)
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    index = args.index / "index-manifest.json" if args.index.is_dir() else args.index
    loaded = load_index(index, resolve_root(args.runtime_root, index.resolve()))
    limits = load_search_config(args.search)
    pipeline = SearchPipeline(loaded, limits)
    queries = [line.strip() for line in args.queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = random.Random(args.seed).sample(queries, min(args.sample, len(queries)))
    checked, failures = 0, []
    for query in chosen:
        response = pipeline.search(SearchRequest(query=query, sources=(), versions=(), candidate_k=limits.candidate_k, top_k=limits.top_k, mode="baseline"))
        for result in response["baseline_results"]:
            for location in result["source_location"]:
                checked += 1
                path = (args.snapshot_root / location["path"]).resolve()
                if not path.is_relative_to(args.snapshot_root.resolve()) or not path.is_file():
                    failures.append({"query": query, "chunk_id": result["chunk_id"], "problem": f"{location['path']} is not a file under the snapshot root"})
                    continue
                problem = link_problem(result["text"], path.read_text(encoding="utf-8", errors="replace").splitlines(), location["start_line"], location["end_line"], args.min_overlap)
                if problem:
                    failures.append({"query": query, "chunk_id": result["chunk_id"], "path": location["path"], "problem": problem})
    report = {"queries": len(chosen), "locations_checked": checked, "failures": len(failures), "examples": failures[:5], "min_overlap": args.min_overlap, "seed": args.seed}
    print(json.dumps(report, indent=2))
    return 0 if checked and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
