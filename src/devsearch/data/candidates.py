"""Fixed natural-BM25 candidate lists (docs/02 §2 and §5).

For every query the natural top-`k` under its own source/version filters is persisted once and
reused by every system, so a reranker only ever *permutes* its query's list. Empty lists are
retained as retrieval failures. Nothing here can insert a passage: candidates come straight
from the retriever. `candidate_set_id` is the same identity `devsearch.retrieve` prints, which
lets live and offline results be compared.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import (
    digest_of, loads_strict, pretty_json, read_utf8, seal_manifest, sha256_file, sha256_hex, sha256_text,
    verify_manifest_hash,
)
from devsearch.artifacts.files import jsonl_bytes, read_jsonl, write_atomic
from devsearch.artifacts.schema import as_object, as_sha256, exact_keys
from devsearch.contracts import SearchRequest
from devsearch.data.queries import Query
from devsearch.errors import ManifestError
from devsearch.retrieval.index_store import LoadedIndex
from devsearch.retrieval.search import candidate_set_id

CANDIDATES_SCHEMA_VERSION = 1
RECORD_FIELDS = (
    "query_id", "query_sha256", "filter_sha256", "corpus_id", "index_id", "retrieval_config_sha256",
    "requested", "actual", "candidate_set_id", "candidates",
)
CANDIDATE_FIELDS = ("chunk_id", "text_sha256", "bm25_score", "rank")


def filter_hash(query: Query) -> str:
    return digest_of({"sources": list(query.sources), "versions": [list(pair) for pair in query.versions]})


def retrieval_config_hash(loaded: LoadedIndex, candidate_k: int) -> str:
    return digest_of({"bm25": loaded.manifest["bm25"], "tokenizer": loaded.manifest["tokenizer"], "candidate_k": candidate_k})


def _request(query: Query, candidate_k: int) -> SearchRequest:
    return SearchRequest(query=query.text, sources=query.sources, versions=query.versions, candidate_k=candidate_k, top_k=1)


def candidate_record(loaded: LoadedIndex, query: Query, candidate_k: int) -> dict[str, Any]:
    request = _request(query, candidate_k)
    found = loaded.retriever().retrieve(request)
    return {
        "query_id": query.query_id, "query_sha256": sha256_text(query.text), "filter_sha256": filter_hash(query),
        "corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id,
        "retrieval_config_sha256": retrieval_config_hash(loaded, candidate_k),
        "requested": candidate_k, "actual": len(found),
        "candidate_set_id": candidate_set_id(loaded, request, found),
        "candidates": [
            {"chunk_id": c.chunk.chunk_id, "text_sha256": sha256_text(c.chunk.text), "bm25_score": c.bm25_score, "rank": c.baseline_rank}
            for c in found
        ],
    }


def build_candidates(loaded: LoadedIndex, queries: Sequence[Query], candidate_k: int) -> list[dict[str, Any]]:
    if candidate_k < 1:
        raise ManifestError("candidate_k must be at least 1")
    return [candidate_record(loaded, query, candidate_k) for query in sorted(queries, key=lambda q: q.query_id)]


def write_candidate_artifacts(
    directory: Path, split: str, records: Sequence[Mapping[str, Any]], loaded: LoadedIndex, *, queries_sha256: str,
) -> dict[str, Any]:
    """Write `candidates-<split>.jsonl` then its sealed manifest; never overwrites."""
    if not directory.is_dir():
        raise ManifestError(f"directory does not exist: {directory}")
    data = jsonl_bytes(records)
    name = f"candidates-{split}.jsonl"
    ks = {r["requested"] for r in records}
    manifest = seal_manifest({
        "schema_version": CANDIDATES_SCHEMA_VERSION, "kind": "candidates", "split": split,
        "corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id, "queries_sha256": queries_sha256,
        "candidate_k": next(iter(ks)) if len(ks) == 1 else None,
        "file": {"path": name, "sha256": sha256_hex(data)},
        "counts": {"queries": len(records), "empty": sum(1 for r in records if r["actual"] == 0),
                   "candidates": sum(r["actual"] for r in records)},
    })
    write_atomic(directory / name, data, exclusive=True)
    write_atomic(directory / f"candidates-{split}-manifest.json", pretty_json(manifest).encode("utf-8"), exclusive=True)
    return manifest


def load_candidates(manifest_path: Path, loaded: LoadedIndex) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate: hashes, identities, rank/score order, and every passage against the corpus."""
    where = "candidates manifest"
    base = manifest_path.resolve(strict=True).parent
    manifest = dict(as_object(loads_strict(read_utf8(manifest_path, where=where), where=where), where))
    exact_keys(manifest, ("schema_version", "kind", "split", "corpus_id", "index_id", "queries_sha256", "candidate_k",
                          "file", "counts", "manifest_sha256"), where)
    if manifest["schema_version"] != CANDIDATES_SCHEMA_VERSION or manifest["kind"] != "candidates":
        raise ManifestError(f"{where}: expected kind='candidates', schema_version={CANDIDATES_SCHEMA_VERSION}")
    verify_manifest_hash(manifest, where=where)
    if (manifest["corpus_id"], manifest["index_id"]) != (loaded.corpus.corpus_id, loaded.index_id):
        raise ManifestError(f"{where}: built for a different corpus/index")
    file_ref = as_object(manifest["file"], f"{where}.file")
    exact_keys(file_ref, ("path", "sha256"), f"{where}.file")
    name = file_ref["path"]
    path = base / name if isinstance(name, str) and Path(name).name == name else None
    if path is None or not path.is_file() or sha256_file(path) != as_sha256(file_ref["sha256"], f"{where}.file.sha256"):
        raise ManifestError(f"{where}: candidate file is missing or does not match its recorded SHA-256")
    chunks = {chunk.chunk_id: chunk for chunk in loaded.corpus.chunks}
    records, previous = [], None
    for label, record in read_jsonl(path, path.name):
        exact_keys(record, RECORD_FIELDS, label)
        if previous is not None and record["query_id"] <= previous:
            raise ManifestError(f"{label}: records must be sorted by unique query_id")
        previous = record["query_id"]
        if (record["corpus_id"], record["index_id"]) != (loaded.corpus.corpus_id, loaded.index_id):
            raise ManifestError(f"{label}: built for a different corpus/index")
        items = record["candidates"]
        if record["actual"] != len(items) or record["actual"] > record["requested"]:
            raise ManifestError(f"{label}: actual/requested counts are inconsistent")
        seen = set()
        for position, item in enumerate(items, start=1):
            exact_keys(as_object(item, f"{label}.candidates[{position - 1}]"), CANDIDATE_FIELDS, f"{label}.candidates")
            chunk = chunks.get(item["chunk_id"])
            if chunk is None or sha256_text(chunk.text) != item["text_sha256"]:
                raise ManifestError(f"{label}: candidate {item['chunk_id']!r} is unknown or its text hash differs")
            if item["rank"] != position or item["chunk_id"] in seen:
                raise ManifestError(f"{label}: ranks must be dense from 1 and chunks unique")
            seen.add(item["chunk_id"])
            if position > 1:
                before = items[position - 2]
                if not (-before["bm25_score"], before["chunk_id"]) < (-item["bm25_score"], item["chunk_id"]):
                    raise ManifestError(f"{label}: candidates are not ordered by descending score then ascending chunk_id")
        records.append(record)
    if manifest["counts"]["queries"] != len(records):
        raise ManifestError(f"{where}: query count does not match the file")
    return manifest, records


def verify_candidates_deep(records: Sequence[Mapping[str, Any]], loaded: LoadedIndex, queries: Sequence[Query]) -> None:
    """Regenerate every list from the index and require byte-identical content (determinism check)."""
    by_id = {q.query_id: q for q in queries}
    if [r["query_id"] for r in records] != sorted(by_id):
        raise ManifestError("candidate records and queries cover different query IDs")
    for record in records:
        fresh = candidate_record(loaded, by_id[record["query_id"]], record["requested"])
        if fresh != dict(record):
            raise ManifestError(f"query {record['query_id']}: stored candidates differ from a fresh retrieval")
