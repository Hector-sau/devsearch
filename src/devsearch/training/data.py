"""Training and validation inputs materialized from prepared artifacts (no test data ever enters here)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from devsearch.artifacts.canonical import digest_of
from devsearch.data.corpus import Corpus
from devsearch.data.pairs import PairRecord
from devsearch.data.queries import Qrel, Query
from devsearch.errors import ManifestError
from devsearch.evaluation.ranking import positives_by_query


@dataclass(frozen=True)
class TrainPair:
    query_text: str
    passage_text: str
    label: int


@dataclass(frozen=True)
class GroupedPair:
    """Canonical identity and train ownership retained alongside model input."""

    record: PairRecord
    pair: TrainPair
    owner_split: str
    evidence_group_id: str
    merged_group_id: str


@dataclass(frozen=True)
class TrainQueryGroup:
    query_id: str
    query_split: str
    known_positive_chunk_ids: frozenset[str]
    known_positive_group_ids: frozenset[str]
    records: tuple[GroupedPair, ...]  # positive first, negatives in frozen source-record order


@dataclass(frozen=True)
class GroupedTrainSet:
    groups: tuple[TrainQueryGroup, ...]
    source_sha256: tuple[tuple[str, str], ...]

    @property
    def pairs(self) -> tuple[TrainPair, ...]:
        return tuple(item.pair for group in self.groups for item in group.records)

    @property
    def identity(self) -> str:
        return digest_of({
            "sources": dict(self.source_sha256),
            "groups": [{"query_id": g.query_id, "query_split": g.query_split,
                        "known_positive_chunk_ids": sorted(g.known_positive_chunk_ids),
                        "known_positive_group_ids": sorted(g.known_positive_group_ids),
                        "records": [{"record": p.record.as_record(), "query_text": p.pair.query_text,
                                     "passage_text": p.pair.passage_text, "label": p.pair.label,
                                     "owner_split": p.owner_split, "evidence_group_id": p.evidence_group_id,
                                     "merged_group_id": p.merged_group_id}
                                    for p in g.records]} for g in self.groups],
        })


def build_grouped_train_set(
    pairs: Sequence[PairRecord], queries: Mapping[str, Query], corpus: Corpus, qrels: Sequence[Qrel],
    chunk_owners: Mapping[str, str], source_sha256: Mapping[str, str],
    *, merged_of: Mapping[str, str],
) -> GroupedTrainSet:
    """Build before dropping IDs; inputs must come from hash-validated prepared artifacts.

    `chunk_owners` must be derived from the verified merged-group assignment, not guessed from IDs.
    Only training queries/qrels may be supplied. File loading/hash checks remain the caller's job.
    """
    from devsearch.training.grouped import validate_grouped_data

    if not queries or any(qid != q.query_id or q.split != "train" or not q.accepted for qid, q in queries.items()):
        raise ManifestError("grouped inputs require accepted training queries only")
    if any(q.query_id not in queries for q in qrels):
        raise ManifestError("grouped qrels must belong only to supplied training queries")
    positives: dict[str, set[str]] = {qid: set() for qid in queries}
    positive_groups: dict[str, set[str]] = {qid: set() for qid in queries}
    chunks = {c.chunk_id: c for c in corpus.chunks}
    seen_qrels: set[tuple[str, str]] = set()
    for qrel in qrels:
        key = (qrel.query_id, qrel.chunk_id)
        if key in seen_qrels:
            raise ManifestError("duplicate or conflicting training qrel")
        seen_qrels.add(key)
        if qrel.confirmed_positive:
            positives[qrel.query_id].add(qrel.chunk_id)
        if qrel.relevance == 1:
            chunk = chunks.get(qrel.chunk_id)
            group = merged_of.get(chunk.evidence_group_id) if chunk is not None else None
            if not isinstance(group, str) or not group:
                raise ManifestError("known positive requires verified merged-group identity")
            positive_groups[qrel.query_id].add(group)
    grouped: dict[str, list[GroupedPair]] = {qid: [] for qid in queries}
    for pair in pairs:
        if type(pair.label) is not int or pair.label not in (0, 1):
            raise ManifestError("grouped pairs require integer binary labels")
        query, chunk = queries.get(pair.query_id), chunks.get(pair.chunk_id)
        if query is None or chunk is None:
            raise ManifestError("grouped pair refers to unknown or non-training query/chunk")
        owner = chunk_owners.get(pair.chunk_id)
        if owner != "train" or chunk.owner_split not in (None, "train"):
            raise ManifestError("grouped pair requires verified train-owned passage")
        merged = merged_of.get(chunk.evidence_group_id)
        if not isinstance(merged, str) or not merged:
            raise ManifestError("grouped pair requires verified merged-group identity")
        grouped[pair.query_id].append(GroupedPair(pair, TrainPair(query.text, chunk.text, pair.label), owner, chunk.evidence_group_id, merged))
    result = GroupedTrainSet(tuple(
        TrainQueryGroup(qid, queries[qid].split, frozenset(positives[qid]), frozenset(positive_groups[qid]),
                        tuple(p for p in grouped[qid] if p.record.label == 1) +
                        tuple(p for p in grouped[qid] if p.record.label == 0))
        for qid in sorted(queries)
    ), tuple(sorted(source_sha256.items())))
    validate_grouped_data(result)
    return result


@dataclass(frozen=True)
class ValidationQuery:
    query_id: str
    text: str
    group_id: str
    candidate_ids: tuple[str, ...]
    candidate_texts: tuple[str, ...]
    positives: frozenset[str]


@dataclass(frozen=True)
class ValidationSet:
    queries: tuple[ValidationQuery, ...]
    excluded_without_positive: tuple[str, ...]

    @property
    def identity(self) -> str:
        return digest_of([[q.query_id, list(q.candidate_ids), sorted(q.positives)] for q in self.queries])

    @property
    def group_of(self) -> dict[str, str]:
        return {q.query_id: q.group_id for q in self.queries}


def build_train_pairs(pairs: Sequence[PairRecord], queries: Mapping[str, Query], corpus: Corpus) -> list[TrainPair]:
    texts = {chunk.chunk_id: chunk.text for chunk in corpus.chunks}
    built = []
    for pair in pairs:
        query = queries.get(pair.query_id)
        if query is None or query.split != "train":
            raise ManifestError(f"pair {pair.pair_id} refers to a query that is not a train query")
        if pair.chunk_id not in texts:
            raise ManifestError(f"pair {pair.pair_id} refers to an unknown chunk")
        built.append(TrainPair(query.text, texts[pair.chunk_id], pair.label))
    return built


def build_validation_set(
    candidate_records: Sequence[Mapping[str, object]], queries: Mapping[str, Query], qrels: Sequence[Qrel], corpus: Corpus,
) -> ValidationSet:
    texts = {chunk.chunk_id: chunk.text for chunk in corpus.chunks}
    positives = positives_by_query(qrels)
    included, excluded = [], []
    for record in candidate_records:
        query = queries[record["query_id"]]
        if query.split != "validation":
            raise ManifestError(f"query {query.query_id} is not a validation query")
        known = positives.get(query.query_id)
        if not known:
            excluded.append(query.query_id)
            continue
        ids = tuple(c["chunk_id"] for c in record["candidates"])
        included.append(ValidationQuery(query.query_id, query.text, query.group_id, ids, tuple(texts[i] for i in ids), known))
    return ValidationSet(tuple(included), tuple(excluded))
