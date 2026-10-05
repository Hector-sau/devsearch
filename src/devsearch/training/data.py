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
