"""Candidate ordering and scoring of ranked outputs. One definition shared by validation, final
evaluation and serving, so ties and denominators can never differ between them.

A reranker only ever permutes its query's frozen candidate list. Order is descending score, then
ascending `chunk_id` (deterministic ties). The scoring denominator contains queries with at least
one confirmed positive; retrieval misses score zero rather than leaving the denominator.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from devsearch.data.queries import Qrel
from devsearch.evaluation.metrics import aggregate, query_metrics


def order_by_score(chunk_ids: Sequence[str], scores: Sequence[float]) -> list[tuple[str, int, float]]:
    """(chunk_id, rank, score) ordered by descending score then ascending chunk_id. Scores must be finite."""
    if len(chunk_ids) != len(scores):
        raise ValueError("one score per candidate is required")
    if not all(isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) for s in scores):
        raise ValueError("every candidate needs one finite score")
    ordered = sorted(zip(chunk_ids, scores), key=lambda pair: (-pair[1], pair[0]))
    return [(chunk_id, rank, float(score)) for rank, (chunk_id, score) in enumerate(ordered, start=1)]


def ranked_record(query_id: str, candidate_set_id: str, ordering: Sequence[tuple[str, int, float]]) -> dict[str, Any]:
    return {
        "query_id": query_id, "candidate_set_id": candidate_set_id,
        "ranking": [{"chunk_id": chunk_id, "rank": rank, "score": score} for chunk_id, rank, score in ordering],
    }


def bm25_ranked_records(candidate_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The BM25 baseline as a system: its candidate order, with BM25 scores."""
    return [
        ranked_record(r["query_id"], r["candidate_set_id"], [(c["chunk_id"], c["rank"], c["bm25_score"]) for c in r["candidates"]])
        for r in candidate_records
    ]


def positives_by_query(qrels: Sequence[Qrel]) -> dict[str, frozenset[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels:
        if qrel.confirmed_positive:
            found[qrel.query_id].add(qrel.chunk_id)
    return {q: frozenset(v) for q, v in found.items()}


def score_ranked(
    ranked: Sequence[Mapping[str, Any]], qrels: Sequence[Qrel], group_of: Mapping[str, str],
) -> dict[str, Any]:
    """Per-query metrics and query/group aggregates for one system over the scoring denominator."""
    return score_with_positives(ranked, positives_by_query(qrels), group_of)


def score_with_positives(
    ranked: Sequence[Mapping[str, Any]], positives: Mapping[str, frozenset[str]], group_of: Mapping[str, str],
) -> dict[str, Any]:
    per_query: dict[str, dict[str, float]] = {}
    without_positive: list[str] = []
    for record in ranked:
        query_id = record["query_id"]
        known = positives.get(query_id)
        if not known:
            without_positive.append(query_id)            # reported separately; never divided by zero or counted as a miss
            continue
        per_query[query_id] = query_metrics([item["chunk_id"] for item in record["ranking"]], known)
    if not per_query:
        raise ValueError("no query with a confirmed positive to score")
    result = aggregate(per_query, group_of, excluded_without_positive=without_positive)
    result["per_query"] = {q: per_query[q] for q in sorted(per_query)}
    return result
