"""Query and relevance-judgment records (docs/02 §2): strict schemas and validation.

* A query belongs to exactly one split, one paraphrase `group_id`, and the evidence
  groups it was generated from. Filters are the same source/version filters search uses.
* A qrel states a binary judgment for one (query, chunk). Missing judgments are *unjudged*,
  never negative. A positive must cite evidence spans inside its chunk; a negative cites none.
* Reviews keep reviewer type (`agent` or `human`) apart. `review_status` is derived from
  `reviews`, never free-standing, so a status cannot claim a review that was not recorded and
  agent review can never be passed off as human review.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from devsearch.artifacts.files import read_jsonl
from devsearch.artifacts.schema import (
    as_id, as_int, as_label, as_list, as_object, as_sha256, as_str, exact_keys,
)
from devsearch.data.corpus import SPLITS, Corpus
from devsearch.errors import ManifestError

QUERY_FIELDS = (
    "query_id", "text", "language", "group_id", "evidence_group_ids", "parent_query_id",
    "task_family", "split", "filters", "origin", "generator", "review_status", "reviews",
)
QREL_FIELDS = (
    "query_id", "chunk_id", "relevance", "evidence_spans", "label_source", "review_status", "reviews",
)
REVIEW_FIELDS = ("reviewer_type", "reviewer_id", "status", "reviewed_at", "revision")
SPAN_FIELDS = ("doc_id", "revision", "content_sha256", "start_char", "end_char")
ORIGINS = ("synthetic", "manual", "public")
LABEL_SOURCES = ("synthetic", "manual", "public", "pooled-review")
REVIEW_STATUSES = ("accepted", "rejected", "ambiguous")
MAX_QUERY_CHARS = 2000
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


@dataclass(frozen=True)
class Review:
    reviewer_type: str
    reviewer_id: str
    status: str
    reviewed_at: str
    revision: str | None


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    group_id: str
    evidence_group_ids: tuple[str, ...]
    parent_query_id: str | None
    task_family: str
    split: str
    sources: tuple[str, ...]
    versions: tuple[tuple[str, str], ...]
    origin: str
    generator: Mapping[str, Any] | None
    review_status: str
    reviews: tuple[Review, ...]
    language: str = "en"

    @property
    def accepted(self) -> bool:
        return self.review_status == "accepted"


@dataclass(frozen=True)
class EvidenceSpan:
    doc_id: str
    revision: str
    content_sha256: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Qrel:
    query_id: str
    chunk_id: str
    relevance: int
    evidence_spans: tuple[EvidenceSpan, ...]
    label_source: str
    review_status: str
    reviews: tuple[Review, ...]

    @property
    def confirmed_positive(self) -> bool:
        return self.relevance == 1 and self.review_status == "accepted"


def derive_review_status(reviews: Sequence[Review]) -> str:
    """`unreviewed` without reviews; the shared status when reviewers agree; else `ambiguous`."""
    if not reviews:
        return "unreviewed"
    statuses = {review.status for review in reviews}
    return statuses.pop() if len(statuses) == 1 else "ambiguous"


def _parse_reviews(value: Any, where: str) -> tuple[Review, ...]:
    reviews = []
    for index, item in enumerate(as_list(value, where)):
        label = f"{where}[{index}]"
        record = as_object(item, label)
        exact_keys(record, REVIEW_FIELDS, label)
        kind = record["reviewer_type"]
        if kind not in ("agent", "human"):
            raise ManifestError(f"{label}.reviewer_type: must be 'agent' or 'human'")
        if record["status"] not in REVIEW_STATUSES:
            raise ManifestError(f"{label}.status: must be one of {', '.join(REVIEW_STATUSES)}")
        stamp = as_str(record["reviewed_at"], f"{label}.reviewed_at", max_length=20, pattern=_TIMESTAMP)
        try:
            datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError as error:
            raise ManifestError(f"{label}.reviewed_at: not a real timestamp: {stamp}") from error
        revision = record["revision"]
        if kind == "agent":
            revision = as_label(revision, f"{label}.revision", max_length=200)   # model/workflow revision is mandatory
        elif revision is not None:
            raise ManifestError(f"{label}.revision: a human review carries no agent revision (use null)")
        reviews.append(Review(kind, as_id(record["reviewer_id"], f"{label}.reviewer_id"), record["status"], stamp, revision))
    return tuple(reviews)


def _check_review_status(status: Any, reviews: tuple[Review, ...], where: str) -> str:
    if status != derive_review_status(reviews):
        raise ManifestError(
            f"{where}.review_status {status!r} does not match its reviews "
            f"(expected {derive_review_status(reviews)!r})"
        )
    return status


def parse_query(record: Mapping[str, Any], where: str) -> Query:
    exact_keys(record, QUERY_FIELDS, where)
    split = record["split"]
    if split not in SPLITS:
        raise ManifestError(f"{where}.split: must be one of {', '.join(SPLITS)}")
    if record["language"] != "en":
        raise ManifestError(f"{where}.language: v1 supports English ('en') only")
    text = as_str(record["text"], f"{where}.text", max_length=MAX_QUERY_CHARS)
    if not text.strip() or text != text.strip() or not text.isprintable():
        raise ManifestError(f"{where}.text: must be non-empty, trimmed, single-line printable text")
    groups = as_list(record["evidence_group_ids"], f"{where}.evidence_group_ids")
    if not groups:
        raise ManifestError(f"{where}.evidence_group_ids: a query needs at least one evidence group")
    group_ids = tuple(as_id(g, f"{where}.evidence_group_ids[{i}]") for i, g in enumerate(groups))
    if len(set(group_ids)) != len(group_ids):
        raise ManifestError(f"{where}.evidence_group_ids: duplicates")
    parent = record["parent_query_id"]
    filters = as_object(record["filters"], f"{where}.filters")
    exact_keys(filters, ("sources", "versions"), f"{where}.filters")
    sources = tuple(as_id(s, f"{where}.filters.sources[{i}]") for i, s in enumerate(as_list(filters["sources"], f"{where}.filters.sources")))
    versions = tuple(sorted(
        (as_id(k, f"{where}.filters.versions key"), as_label(v, f"{where}.filters.versions[{k!r}]", max_length=128))
        for k, v in as_object(filters["versions"], f"{where}.filters.versions").items()
    ))
    origin = record["origin"]
    if origin not in ORIGINS:
        raise ManifestError(f"{where}.origin: must be one of {', '.join(ORIGINS)}")
    generator = record["generator"]
    if origin == "synthetic":
        generator = as_object(generator, f"{where}.generator")
        exact_keys(generator, ("model", "prompt_sha256", "seed"), f"{where}.generator")
        as_label(generator["model"], f"{where}.generator.model", max_length=200)
        as_sha256(generator["prompt_sha256"], f"{where}.generator.prompt_sha256")
        if generator["seed"] is not None:
            as_int(generator["seed"], f"{where}.generator.seed", minimum=-2**63, maximum=2**63)
        generator = dict(generator)
    elif generator is not None:
        raise ManifestError(f"{where}.generator: only synthetic queries carry generator provenance")
    reviews = _parse_reviews(record["reviews"], f"{where}.reviews")
    return Query(
        query_id=as_id(record["query_id"], f"{where}.query_id"), text=text,
        group_id=as_id(record["group_id"], f"{where}.group_id"), evidence_group_ids=group_ids,
        parent_query_id=None if parent is None else as_id(parent, f"{where}.parent_query_id"),
        task_family=as_id(record["task_family"], f"{where}.task_family"), split=split,
        sources=sources, versions=versions, origin=origin, generator=generator,
        review_status=_check_review_status(record["review_status"], reviews, where), reviews=reviews,
    )


def _span(value: Any, where: str) -> EvidenceSpan:
    record = as_object(value, where)
    exact_keys(record, SPAN_FIELDS, where)
    start = as_int(record["start_char"], f"{where}.start_char")
    return EvidenceSpan(
        as_id(record["doc_id"], f"{where}.doc_id"), as_label(record["revision"], f"{where}.revision", max_length=128),
        as_sha256(record["content_sha256"], f"{where}.content_sha256"), start,
        as_int(record["end_char"], f"{where}.end_char", minimum=start + 1),
    )


def parse_qrel(record: Mapping[str, Any], where: str) -> Qrel:
    exact_keys(record, QREL_FIELDS, where)
    relevance = record["relevance"]
    if isinstance(relevance, bool) or relevance not in (0, 1):
        raise ManifestError(f"{where}.relevance: v1 judgments are binary 0 or 1")
    spans = tuple(_span(item, f"{where}.evidence_spans[{i}]") for i, item in enumerate(as_list(record["evidence_spans"], f"{where}.evidence_spans")))
    if relevance == 1 and not spans:
        raise ManifestError(f"{where}: a positive judgment must cite at least one evidence span")
    if relevance == 0 and spans:
        raise ManifestError(f"{where}: a negative judgment cites no evidence span")
    if record["label_source"] not in LABEL_SOURCES:
        raise ManifestError(f"{where}.label_source: must be one of {', '.join(LABEL_SOURCES)}")
    reviews = _parse_reviews(record["reviews"], f"{where}.reviews")
    return Qrel(
        as_id(record["query_id"], f"{where}.query_id"), as_id(record["chunk_id"], f"{where}.chunk_id"),
        relevance, spans, record["label_source"], _check_review_status(record["review_status"], reviews, where), reviews,
    )


def _review_record(review: Review) -> dict[str, Any]:
    return {"reviewer_type": review.reviewer_type, "reviewer_id": review.reviewer_id, "status": review.status,
            "reviewed_at": review.reviewed_at, "revision": review.revision}


def query_record(query: Query) -> dict[str, Any]:
    return {
        "query_id": query.query_id, "text": query.text, "language": query.language, "group_id": query.group_id,
        "evidence_group_ids": list(query.evidence_group_ids), "parent_query_id": query.parent_query_id,
        "task_family": query.task_family, "split": query.split,
        "filters": {"sources": list(query.sources), "versions": dict(query.versions)},
        "origin": query.origin, "generator": None if query.generator is None else dict(query.generator),
        "review_status": query.review_status, "reviews": [_review_record(r) for r in query.reviews],
    }


def qrel_record(qrel: Qrel) -> dict[str, Any]:
    return {
        "query_id": qrel.query_id, "chunk_id": qrel.chunk_id, "relevance": qrel.relevance,
        "evidence_spans": [
            {"doc_id": s.doc_id, "revision": s.revision, "content_sha256": s.content_sha256,
             "start_char": s.start_char, "end_char": s.end_char} for s in qrel.evidence_spans
        ],
        "label_source": qrel.label_source, "review_status": qrel.review_status,
        "reviews": [_review_record(r) for r in qrel.reviews],
    }


def load_queries(path: Path) -> tuple[Query, ...]:
    """Queries sorted by `query_id` (canonical order), unique, one split per file."""
    queries: list[Query] = []
    for label, record in read_jsonl(path, path.name):
        queries.append(parse_query(record, label))
        if len(queries) > 1 and queries[-1].query_id <= queries[-2].query_id:
            raise ManifestError(f"{label}: queries must be unique and sorted by query_id")
    if len({q.split for q in queries}) > 1:
        raise ManifestError(f"{path.name}: one file may hold queries of one split only")
    return tuple(queries)


def load_qrels(path: Path) -> tuple[Qrel, ...]:
    """Judgments sorted by (query_id, chunk_id), unique."""
    qrels: list[Qrel] = []
    for label, record in read_jsonl(path, path.name):
        qrels.append(parse_qrel(record, label))
        if len(qrels) > 1 and (qrels[-1].query_id, qrels[-1].chunk_id) <= (qrels[-2].query_id, qrels[-2].chunk_id):
            raise ManifestError(f"{label}: judgments must be unique and sorted by (query_id, chunk_id)")
    return tuple(qrels)


def validate_queries(queries: Sequence[Query], corpus: Corpus) -> None:
    """Filters, evidence groups and paraphrase groups must exist and be consistent."""
    known = corpus.revisions_by_source()
    groups = {chunk.evidence_group_id for chunk in corpus.chunks}
    by_id = {q.query_id: q for q in queries}
    for query in queries:
        where = f"query {query.query_id}"
        for source in query.sources:
            if source not in known:
                raise ManifestError(f"{where}: unknown source {source!r}")
        seen_sources = set()
        for source, revision in query.versions:
            if source not in known or revision not in known[source]:
                raise ManifestError(f"{where}: unknown version filter {source!r}={revision!r}")
            if query.sources and source not in query.sources:
                raise ManifestError(f"{where}: version given for {source!r}, which its source filter excludes")
            if source in seen_sources:
                raise ManifestError(f"{where}: repeated version filter for {source!r}")
            seen_sources.add(source)
        missing = [g for g in query.evidence_group_ids if g not in groups]
        if missing:
            raise ManifestError(f"{where}: unknown evidence group(s): {', '.join(missing)}")
        if query.parent_query_id is not None:
            parent = by_id.get(query.parent_query_id)
            if parent is None:
                raise ManifestError(f"{where}: parent query {query.parent_query_id!r} is not in this file")
            if parent.group_id != query.group_id or parent.split != query.split:
                raise ManifestError(f"{where}: a paraphrase must share its parent's group_id and split")


def validate_qrels(qrels: Sequence[Qrel], queries: Mapping[str, Query], corpus: Corpus) -> None:
    """Chunks exist, evidence is real text inside the chunk, positives come from the query's groups."""
    chunks = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    for qrel in qrels:
        where = f"qrel ({qrel.query_id}, {qrel.chunk_id})"
        query = queries.get(qrel.query_id)
        if query is None:
            raise ManifestError(f"{where}: unknown query")
        chunk = chunks.get(qrel.chunk_id)
        if chunk is None:
            raise ManifestError(f"{where}: unknown chunk")
        for span in qrel.evidence_spans:
            doc = corpus.docs.get((span.doc_id, span.revision))
            if doc is None or doc.content_sha256 != span.content_sha256:
                raise ManifestError(f"{where}: evidence span does not match a document in the corpus")
            if (span.doc_id, span.revision) != (chunk.doc_id, chunk.revision) or not (
                chunk.start_char <= span.start_char < span.end_char <= chunk.end_char
            ):
                raise ManifestError(f"{where}: evidence span lies outside the judged chunk")
        if qrel.relevance == 1 and qrel.label_source in ("synthetic", "manual") and chunk.evidence_group_id not in query.evidence_group_ids:
            raise ManifestError(f"{where}: positive chunk is outside the query's evidence groups")
