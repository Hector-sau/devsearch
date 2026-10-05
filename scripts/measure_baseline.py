"""Measure BM25-only search latency on a built index with the declared protocol (docs/10 §6): warm-up, seeded sequential passes,
concurrency 1, no response cache. No model is loaded. The queries come from a file: either plain text (one per line) or a
query JSONL (a `text` field per line; use validation or development queries, never test ones).

    measure_baseline.py --index <index-manifest.json> --search <search.toml> --queries <file> [--size 50] [--warmup 10] [--passes 3]
                        [--seed 20261005] [--runtime-root R] [--label "what the queries are"]

The result is a CPU measurement on THIS host for the given index; it says nothing about reranking or about any other machine. A
workload made of section headings or other stand-ins must be labelled as such with --label.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from devsearch.artifacts.paths import resolve_root  # noqa: E402
from devsearch.contracts import SearchRequest  # noqa: E402
from devsearch.evaluation.latency import measure, workload  # noqa: E402
from devsearch.retrieval.config import load_search_config  # noqa: E402
from devsearch.retrieval.index_store import load_index  # noqa: E402
from devsearch.serving.pipeline import SearchPipeline  # noqa: E402


def read_queries(path: Path) -> list[str]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        queries.append(json.loads(line)["text"] if line.startswith("{") else line)
    return queries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--label", default="unlabelled")
    args = parser.parse_args()
    root = resolve_root(args.runtime_root, args.index.resolve())
    loaded = load_index(args.index, root)
    limits = load_search_config(args.search)
    pipeline = SearchPipeline(loaded, limits)
    chosen = workload(read_queries(args.queries), size=args.size, seed=args.seed)

    def search(query: str) -> dict:
        return pipeline.search(SearchRequest(query=query, sources=(), versions=(), candidate_k=limits.candidate_k, top_k=limits.top_k, mode="baseline"))

    result = measure(search, chosen, warmup=args.warmup, passes=args.passes, seed=args.seed)
    print(json.dumps({"workload_label": args.label, "chunks": len(loaded.corpus.chunks), "index_id": loaded.index_id, "host": {"machine": platform.machine(), "python": platform.python_version()},
                      "mode": "baseline (BM25 only, no model)", **result}, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
