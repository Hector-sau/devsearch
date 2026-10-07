"""Blinded review pool and judgment bank for the test set (docs/02 §5, docs/10 "Fixed-candidate pre-review").

Recommended protocol `prejudged_fixed_candidates_v1`: before any training comparison, review
*every* unique (query, passage) pair of `U(q) = C(q) ∪ known positives(q)`, where `C(q)` is the
frozen natural BM25 candidate list. Known positives only join the review pool; they are never
inserted into retrieval. Protocol `postrank_pool_v1` pools the top-10 of every locked system plus
known positives after ranking. Both use the same blinded request/response/bank machinery.

* The reviewer-facing request shows the question, its filters, the passage and its source
  evidence. It never shows ranks, scores, system names, whether a pair was a known positive, or any
  earlier label. Item IDs are salted hashes and the order is a seeded shuffle, so neither leaks rank.
* Provenance lives in a separate private file keyed by `item_id`.
* Every item needs a *resolved* binary judgment. Unresolved or missing items block sealing; they
  are never turned into label 0 and no hard question is dropped.
* Reviews carry `agent` or `human` reviewer type; the bank reports both counts separately.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from devsearch.artifacts.canonical import digest_of
from devsearch.artifacts.schema import as_id, as_int, as_label, as_list, as_object, as_str, exact_keys
from devsearch.contracts import SearchRequest
from devsearch.data.queries import EvidenceSpan, Qrel, Query, Review, derive_review_status
from devsearch.errors import ManifestError
from devsearch.retrieval.index_store import LoadedIndex

PROTOCOLS = ("prejudged_fixed_candidates_v1", "postrank_pool_v1")
RESPONSE_FIELDS = ("item_id", "role", "relevance", "status", "evidence_spans", "reviewer", "reviewed_at", "notes")
LEAKY_KEYS = frozenset({"rank", "baseline_rank", "reranked_rank", "score", "bm25_score", "rerank_logit", "system", "systems",
                        "label", "relevance", "known_positive", "prior_label", "candidate_rank", "chunk_id", "query_id"})


@dataclass(frozen=True)
class ReviewItem:
    item_id: str
    query_id: str
    chunk_id: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ReviewPool:
    protocol: str
    salt: str
    items: tuple[ReviewItem, ...]               # in blinded (shuffled) presentation order

    @property
    def pool_id(self) -> str:
        return "reviewpool-" + digest_of([self.protocol, self.salt, sorted(i.item_id for i in self.items)])[:32]


def item_id_for(salt: str, query_id: str, chunk_id: str) -> str:
    return "it_" + digest_of({"salt": salt, "query_id": query_id, "chunk_id": chunk_id})[:24]


def build_review_pool(protocol: str, salt: str, universe: Mapping[tuple[str, str], Mapping[str, Any]]) -> ReviewPool:
    """Unique (query, chunk) pairs -> salted item IDs, shuffled by a seeded hash order."""
    if protocol not in PROTOCOLS:
        raise ManifestError(f"unknown evaluation protocol {protocol!r}")
    if len(salt) < 8:
        raise ManifestError("the review salt must have at least 8 characters")
    items = [ReviewItem(item_id_for(salt, q, c), q, c, dict(provenance)) for (q, c), provenance in sorted(universe.items())]
    if len({i.item_id for i in items}) != len(items):
        raise ManifestError("review item ID collision")
    items.sort(key=lambda i: hashlib.sha256(f"{salt}|order|{i.item_id}".encode()).hexdigest())
    return ReviewPool(protocol, salt, tuple(items))


def prejudged_universe(
    loaded: LoadedIndex, queries: Sequence[Query], candidate_records: Sequence[Mapping[str, Any]], qrels_v1: Sequence[Qrel],
) -> dict[tuple[str, str], dict[str, Any]]:
    """U(q) = C(q) ∪ known positives(q). Positives outside the filtered corpus are an error."""
    by_query = {q.query_id: q for q in queries}
    cands = {r["query_id"]: r for r in candidate_records}
    if set(cands) != set(by_query):
        raise ManifestError("candidate lists and queries cover different query IDs")
    positives: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels_v1:
        if qrel.query_id not in by_query:
            raise ManifestError(f"qrels-v1 names query {qrel.query_id!r}, which is not a test query")
        if qrel.confirmed_positive:
            positives[qrel.query_id].add(qrel.chunk_id)
    universe: dict[tuple[str, str], dict[str, Any]] = {}
    for query_id, record in sorted(cands.items()):
        for item in record["candidates"]:
            universe[(query_id, item["chunk_id"])] = {"known_positive": False, "candidate_rank": item["rank"], "bm25_score": item["bm25_score"]}
        query = by_query[query_id]
        eligible = loaded.retriever().eligible_chunk_ids(
            SearchRequest(query=query.text, sources=query.sources, versions=query.versions, candidate_k=1)
        )
        for chunk_id in sorted(positives.get(query_id, ())):
            if eligible is not None and chunk_id not in eligible:
                raise ManifestError(f"query {query_id}: known positive {chunk_id} lies outside the query's filtered corpus")
            entry = universe.setdefault((query_id, chunk_id), {"known_positive": True, "candidate_rank": None, "bm25_score": None})
            entry["known_positive"] = True
    return universe


def postrank_universe(
    ranked_by_system: Mapping[str, Sequence[Mapping[str, Any]]], qrels_v1: Sequence[Qrel], depth: int = 10,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Top-`depth` of every locked system plus known positives (post-rank pooling)."""
    universe: dict[tuple[str, str], dict[str, Any]] = {}
    for system, records in sorted(ranked_by_system.items()):
        for record in records:
            for item in record["ranking"][:depth]:
                entry = universe.setdefault((record["query_id"], item["chunk_id"]), {"known_positive": False, "systems": {}})
                entry["systems"][system] = {"rank": item["rank"], "score": item["score"]}
    for qrel in qrels_v1:
        if qrel.confirmed_positive:
            universe.setdefault((qrel.query_id, qrel.chunk_id), {"known_positive": True, "systems": {}})["known_positive"] = True
    return universe


def request_records(pool: ReviewPool, loaded: LoadedIndex, queries: Mapping[str, Query]) -> list[dict[str, Any]]:
    """The blinded reviewer-facing items, in presentation order."""
    chunks = {c.chunk_id: c for c in loaded.corpus.chunks}
    records = []
    for item in pool.items:
        chunk, query = chunks[item.chunk_id], queries[item.query_id]
        doc = loaded.corpus.doc_of(chunk)
        records.append({
            "item_id": item.item_id,
            "query": {"text": query.text, "filters": {"sources": list(query.sources), "versions": dict(query.versions)}},
            "passage": {
                "title": doc.title, "section": " > ".join(chunk.heading_path) or None, "text": chunk.text,
                "code_fragment": chunk.code_fragment, "source_url": doc.source_url, "source_location": loaded.corpus.locate(chunk),
            },
        })
    assert_blinded(records)
    return records


def assert_blinded(records: Iterable[Mapping[str, Any]]) -> None:
    """Fail if any reviewer-facing record carries a key that could reveal rank, score, system or prior label."""
    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, inner in value.items():
                if key in LEAKY_KEYS:
                    raise ManifestError(f"review request leaks {key!r} at {path}")
                walk(inner, f"{path}.{key}")
        elif isinstance(value, list):
            for inner in value:
                walk(inner, path)
    for record in records:
        walk(record, "item")


def hidden_records(pool: ReviewPool) -> list[dict[str, Any]]:
    return [{"item_id": i.item_id, "query_id": i.query_id, "chunk_id": i.chunk_id, "provenance": dict(i.provenance)}
            for i in sorted(pool.items, key=lambda i: i.item_id)]


@dataclass(frozen=True)
class ReviewResponse:
    item_id: str
    role: str
    relevance: int | None
    resolved: bool
    spans: tuple[tuple[int, int], ...]            # passage-local [start, end)
    reviewer_type: str
    reviewer_id: str
    revision: str | None
    reviewed_at: str


def parse_response(record: Mapping[str, Any], where: str, passage_lengths: Mapping[str, int]) -> ReviewResponse:
    exact_keys(record, RESPONSE_FIELDS, where)
    item_id = as_id(record["item_id"], f"{where}.item_id")
    if item_id not in passage_lengths:
        raise ManifestError(f"{where}: item {item_id!r} is not in the review pool")
    if record["role"] not in ("primary", "adjudicator"):
        raise ManifestError(f"{where}.role: must be 'primary' or 'adjudicator'")
    resolved = record["status"] == "resolved"
    if record["status"] not in ("resolved", "unresolved"):
        raise ManifestError(f"{where}.status: must be 'resolved' or 'unresolved'")
    relevance = record["relevance"]
    spans = tuple(sorted(
        (as_int(s["start"], f"{where}.evidence_spans.start"), as_int(s["end"], f"{where}.evidence_spans.end", minimum=1))
        for s in (as_object(x, f"{where}.evidence_spans[]") for x in as_list(record["evidence_spans"], f"{where}.evidence_spans"))
    ))
    length = passage_lengths[item_id]
    if resolved:
        if isinstance(relevance, bool) or relevance not in (0, 1):
            raise ManifestError(f"{where}: a resolved judgment needs relevance 0 or 1")
        if relevance == 1 and not spans:
            raise ManifestError(f"{where}: a positive judgment cites at least one evidence span")
        if relevance == 0 and spans:
            raise ManifestError(f"{where}: a negative judgment cites no evidence span")
        if any(not (0 <= a < b <= length) for a, b in spans):
            raise ManifestError(f"{where}: an evidence span lies outside the passage")
    elif relevance is not None or spans:
        raise ManifestError(f"{where}: an unresolved judgment has no relevance and no spans")
    reviewer = as_object(record["reviewer"], f"{where}.reviewer")
    exact_keys(reviewer, ("type", "id", "revision"), f"{where}.reviewer")
    kind = reviewer["type"]
    if kind not in ("agent", "human"):
        raise ManifestError(f"{where}.reviewer.type: must be 'agent' or 'human'")
    revision = reviewer["revision"]
    if kind == "agent":
        revision = as_label(revision, f"{where}.reviewer.revision", max_length=200)
    elif revision is not None:
        raise ManifestError(f"{where}.reviewer.revision: a human reviewer carries no agent revision")
    if record["notes"] is not None:
        as_str(record["notes"], f"{where}.notes", max_length=2000, allow_empty=True)
    return ReviewResponse(item_id, record["role"], relevance if resolved else None, resolved, spans, kind,
                          as_id(reviewer["id"], f"{where}.reviewer.id"),
                          revision, as_str(record["reviewed_at"], f"{where}.reviewed_at", max_length=20))


@dataclass(frozen=True)
class Bank:
    qrels: tuple[Qrel, ...]
    unresolved: tuple[tuple[str, str], ...]        # (item_id, reason)
    reviewer_counts: Mapping[str, int]

    @property
    def ready(self) -> bool:
        return not self.unresolved


def resolve_bank(
    pool: ReviewPool, responses: Sequence[ReviewResponse], loaded: LoadedIndex, *, min_primaries: int = 1, protocol: Any = None,
) -> Bank:
    """One final judgment per item: the adjudicator's, else the unanimous primaries of at least `min_primaries` distinct reviewers;
    else unresolved. A missing second review is never filled in: the item stays unresolved and blocks sealing.

    With a `primary_audit_v1` protocol (`devsearch.data.review_protocol`) the final rule is that protocol's: an independent
    adjudicator's judgment, else the primary pass confirmed by every second review, with the stratified sample (and any escalated
    stratum) second-reviewed."""
    chunks = {c.chunk_id: c for c in loaded.corpus.chunks}
    if protocol is not None and protocol.mode != "double_primary":
        return _resolve_audited(pool, responses, loaded, protocol, chunks)
    grouped: dict[str, list[ReviewResponse]] = defaultdict(list)
    for response in responses:
        grouped[response.item_id].append(response)
    unresolved, qrels = [], []
    counts = {"agent": 0, "human": 0}
    for response in responses:
        counts[response.reviewer_type] += 1
    for item in sorted(pool.items, key=lambda i: i.item_id):
        mine = grouped.get(item.item_id, [])
        if not mine:
            unresolved.append((item.item_id, "missing"))
            continue
        adjudicators = [r for r in mine if r.role == "adjudicator"]
        if len(adjudicators) > 1:
            raise ManifestError(f"item {item.item_id}: more than one adjudicator")
        primaries = [r for r in mine if r.role == "primary"]
        basis = adjudicators if adjudicators else primaries
        if not basis:
            unresolved.append((item.item_id, "missing"))
            continue
        if not adjudicators and len({r.reviewer_id for r in primaries}) < min_primaries:
            unresolved.append((item.item_id, f"needs {min_primaries} independent primary reviews, has {len({r.reviewer_id for r in primaries})}"))
            continue
        if not all(r.resolved for r in basis):
            unresolved.append((item.item_id, "unresolved"))
            continue
        if len({r.relevance for r in basis}) != 1:
            unresolved.append((item.item_id, "disagreement without adjudication"))
            continue
        chunk = chunks[item.chunk_id]
        doc = loaded.corpus.doc_of(chunk)
        relevance = basis[0].relevance
        spans = tuple(sorted({(a, b) for r in basis for a, b in r.spans}))
        evidence = tuple(EvidenceSpan(doc.doc_id, doc.revision, doc.content_sha256, chunk.start_char + a, chunk.start_char + b) for a, b in spans)
        reviews = tuple(Review(r.reviewer_type, r.reviewer_id, "accepted", r.reviewed_at, r.revision) for r in basis)
        qrels.append(Qrel(item.query_id, item.chunk_id, relevance, evidence, "pooled-review", derive_review_status(reviews), reviews))
    qrels.sort(key=lambda q: (q.query_id, q.chunk_id))
    return Bank(tuple(qrels), tuple(unresolved), counts)


def _resolve_audited(pool: ReviewPool, responses: Sequence[ReviewResponse], loaded: LoadedIndex, protocol: Any, chunks: Mapping[str, Any]) -> Bank:
    from devsearch.data.review_protocol import evaluate

    items = {i.item_id: {"query_id": i.query_id, "provenance": i.provenance} for i in pool.items}
    result = evaluate(protocol, items, responses)
    counts = {"agent": 0, "human": 0}
    for response in responses:
        counts[response.reviewer_type] += 1
    unresolved, qrels = [], []
    for item in sorted(pool.items, key=lambda i: i.item_id):
        state = result["states"][item.item_id]
        if not state.startswith("final"):
            unresolved.append((item.item_id, state))
            continue
        basis = result["basis"][item.item_id]
        chunk = chunks[item.chunk_id]
        doc = loaded.corpus.doc_of(chunk)
        decider = basis[0]                                   # the adjudicator, or the primary pass that every second review confirmed
        spans = tuple(sorted({(a, b) for r in basis if r.relevance == decider.relevance for a, b in r.spans}))
        evidence = tuple(EvidenceSpan(doc.doc_id, doc.revision, doc.content_sha256, chunk.start_char + a, chunk.start_char + b) for a, b in spans)
        reviews = tuple(Review(r.reviewer_type, r.reviewer_id, "accepted", r.reviewed_at, r.revision) for r in basis)
        qrels.append(Qrel(item.query_id, item.chunk_id, decider.relevance, evidence, "pooled-review", derive_review_status(reviews), reviews))
    qrels.sort(key=lambda q: (q.query_id, q.chunk_id))
    return Bank(tuple(qrels), tuple(unresolved), counts)


def qrels_diff(qrels_v1: Sequence[Qrel], qrels_v2: Sequence[Qrel], query_ids: Iterable[str]) -> dict[str, Any]:
    """What the review changed relative to the initial judgments; queries are recorded, never deleted."""
    v1 = {(q.query_id, q.chunk_id): q.relevance for q in qrels_v1 if q.confirmed_positive}
    v2 = {(q.query_id, q.chunk_id): q.relevance for q in qrels_v2}
    kept = sum(1 for k in v1 if v2.get(k) == 1)
    reversed_ = sorted(f"{q}:{c}" for (q, c) in v1 if v2.get((q, c)) == 0)
    new = sorted(f"{q}:{c}" for (q, c), r in v2.items() if r == 1 and (q, c) not in v1)
    positives_v2 = defaultdict(int)
    for (q, _c), r in v2.items():
        positives_v2[q] += r
    ids = sorted(set(query_ids))
    return {
        "v1_positives": len(v1), "v1_positives_confirmed": kept, "v1_positives_reversed": reversed_, "new_positives": new,
        "queries_total": len(ids), "queries_with_positive_v2": sum(1 for q in ids if positives_v2[q] > 0),
        "queries_without_positive_v2": [q for q in ids if positives_v2[q] == 0],
        "queries_losing_all_positives": sorted(q for q in ids if positives_v2[q] == 0 and any(k[0] == q for k in v1)),
    }
