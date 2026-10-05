"""Baseline search: request validation and the response document (docs/04 "Search API").

This is the model-free slice used by `devsearch.retrieve`. A later `SearchPipeline`
(T09) composes it with reranking; the result fields below are the shared contract.
The document is a subset of the API response: no `request_id`/`release_id` yet, and
`reranked_results` stays empty because no model is loaded.
"""

import time
from typing import Any

from devsearch.artifacts.canonical import digest_of, sha256_text
from devsearch.contracts import RetrievalCandidate, SearchRequest
from devsearch.errors import InvalidRequestError
from devsearch.retrieval.config import SearchConfig
from devsearch.retrieval.index_store import LoadedIndex

RESPONSE_SCHEMA_VERSION = 1


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_request(request: SearchRequest, limits: SearchConfig, *, modes: tuple[str, ...] = ("baseline",)) -> None:
    """Reject invalid requests readably instead of silently changing their meaning.

    `modes` lists what the caller can serve: the model-free CLI serves only `baseline`; the serving
    pipeline adds `rerank` and `compare` (and degrades them explicitly when no reranker is ready).
    """
    if request.mode not in modes:
        raise InvalidRequestError(
            f"mode {request.mode!r} is not available here; this entry point supports {', '.join(modes)}"
        )
    if not isinstance(request.query, str) or not request.query.strip():
        raise InvalidRequestError("query must not be empty")
    if len(request.query) > limits.max_query_chars:
        raise InvalidRequestError(
            f"query has {len(request.query)} characters; the limit is {limits.max_query_chars}"
        )
    if not (_is_int(request.top_k) and _is_int(request.candidate_k)
            and 1 <= request.top_k <= request.candidate_k <= limits.max_candidate_k):
        raise InvalidRequestError(
            f"need 1 <= top_k <= candidate_k <= {limits.max_candidate_k}; "
            f"got top_k={request.top_k}, candidate_k={request.candidate_k}"
        )


def result_record(loaded: LoadedIndex, candidate: RetrievalCandidate) -> dict[str, Any]:
    chunk = candidate.chunk
    doc = loaded.corpus.doc_of(chunk)
    return {
        "chunk_id": chunk.chunk_id, "source_id": doc.source_id, "source_url": doc.source_url,
        "revision": chunk.revision, "title": doc.title,
        "section": " > ".join(chunk.heading_path) or None,
        "text": chunk.text, "code_fragment": chunk.code_fragment,
        "source_location": loaded.corpus.locate(chunk),
        "baseline_rank": candidate.baseline_rank, "reranked_rank": None,
        "bm25_score": candidate.bm25_score, "rerank_logit": None,
    }


def candidate_set_id(
    loaded: LoadedIndex, request: SearchRequest, candidates: tuple[RetrievalCandidate, ...]
) -> str:
    """Identity of the exact ordered candidate list, for later reuse and parity checks."""
    return "cands-" + digest_of({
        "index_id": loaded.index_id, "query_sha256": sha256_text(request.query),
        "sources": list(request.sources), "versions": [list(pair) for pair in request.versions],
        "candidate_k": request.candidate_k,
        "candidates": [
            [c.chunk.chunk_id, sha256_text(c.chunk.text), c.bm25_score, c.baseline_rank]
            for c in candidates
        ],
    })[:32]


def run_baseline(loaded: LoadedIndex, request: SearchRequest, limits: SearchConfig) -> dict[str, Any]:
    validate_request(request, limits)
    started = time.perf_counter()
    candidates = loaded.retriever().retrieve(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "status": "ok" if candidates else "empty",
        "mode_requested": request.mode, "mode_effective": "baseline",
        "corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id,
        "model_id": None, "candidate_set_id": candidate_set_id(loaded, request, candidates),
        "query": request.query,
        "filters": {"sources": list(request.sources), "versions": dict(request.versions)},
        "candidate_k": request.candidate_k, "candidate_count": len(candidates),
        "top_k": request.top_k,
        "baseline_results": [result_record(loaded, c) for c in candidates[:request.top_k]],
        "reranked_results": [],
        "timings_ms": {"retrieval": round(elapsed_ms, 3), "rerank": None, "total": round(elapsed_ms, 3)},
        "warnings": [],
    }
