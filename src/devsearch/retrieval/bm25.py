"""Okapi BM25 baseline over a frozen corpus (docs/04 "Index and version semantics").

* N, document frequencies, IDF and average length are computed once over the
  **complete** corpus; a request's filters mask chunks before top-k and never
  change those statistics.
* A chunk is eligible only if it passes the filters and shares at least one
  indexed token with the processed query. Eligibility does not depend on the sign
  of its score: with `idf = "robertson"` a term found in more than half the corpus
  scores below zero yet still matches. No eligible chunk means an empty result,
  never arbitrary zero-scored padding.
* Order is descending score (rounded to `SCORE_DECIMALS` so platform-level float
  noise cannot reorder ties), then ascending `chunk_id`.
* Query terms are unique (binary weight) and summed in first-occurrence order.

IDF variants: `lucene`: ln(1 + (N - n + 0.5) / (n + 0.5)), always positive;
`robertson`: ln((N - n + 0.5) / (n + 0.5)), may be zero or negative.
Per-term gain: idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len)).
"""

import heapq
import math
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

from devsearch.artifacts.schema import as_int, as_list, as_object, as_str, exact_keys
from devsearch.contracts import Chunk, RetrievalCandidate, SearchRequest
from devsearch.data.corpus import Corpus
from devsearch.errors import InvalidRequestError, ManifestError
from devsearch.retrieval.tokenizer import tokenize, unique_query_terms

SCORE_DECIMALS = 10
POSTINGS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Postings:
    chunk_ids: tuple[str, ...]
    doc_lengths: tuple[int, ...]
    terms: Mapping[str, tuple[tuple[int, int], ...]]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": POSTINGS_SCHEMA_VERSION,
            "chunk_ids": list(self.chunk_ids),
            "doc_lengths": list(self.doc_lengths),
            "terms": {term: [list(pair) for pair in pairs] for term, pairs in sorted(self.terms.items())},
        }


def build_postings(chunks: Sequence[Chunk]) -> Postings:
    """Term -> ((chunk_index, term_frequency), ...) in chunk order; deterministic."""
    terms: dict[str, list[tuple[int, int]]] = {}
    lengths: list[int] = []
    for index, chunk in enumerate(chunks):
        counts = Counter(tokenize(chunk.text))
        lengths.append(sum(counts.values()))
        for term, frequency in counts.items():
            terms.setdefault(term, []).append((index, frequency))
    return Postings(
        chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
        doc_lengths=tuple(lengths),
        terms={term: tuple(pairs) for term, pairs in sorted(terms.items())},
    )


def parse_postings(value: Any, *, expected_chunk_ids: Sequence[str], where: str) -> Postings:
    record = as_object(value, where)
    exact_keys(record, ("schema_version", "chunk_ids", "doc_lengths", "terms"), where)
    if record["schema_version"] != POSTINGS_SCHEMA_VERSION:
        raise ManifestError(f"{where}: unsupported postings schema_version")
    chunk_ids = tuple(record["chunk_ids"]) if isinstance(record["chunk_ids"], list) else None
    if chunk_ids != tuple(expected_chunk_ids):
        raise ManifestError(f"{where}: chunk_ids do not match the corpus chunks (order included)")
    lengths = tuple(
        as_int(item, f"{where}.doc_lengths[{i}]")
        for i, item in enumerate(as_list(record["doc_lengths"], f"{where}.doc_lengths"))
    )
    if len(lengths) != len(chunk_ids):
        raise ManifestError(f"{where}: doc_lengths has the wrong length")
    totals = [0] * len(lengths)
    terms: dict[str, tuple[tuple[int, int], ...]] = {}
    for term, pairs_value in as_object(record["terms"], f"{where}.terms").items():
        label = f"{where}.terms[{term!r}]"
        as_str(term, label, max_length=1000)
        pairs, previous = [], -1
        for pair in as_list(pairs_value, label):
            pair = as_list(pair, label)
            if len(pair) != 2:
                raise ManifestError(f"{label}: expected [chunk_index, term_frequency]")
            index = as_int(pair[0], label, minimum=previous + 1, maximum=len(lengths) - 1)
            frequency = as_int(pair[1], label, minimum=1)
            previous, totals[index] = index, totals[index] + frequency
            pairs.append((index, frequency))
        if not pairs:
            raise ManifestError(f"{label}: empty posting list")
        terms[term] = tuple(pairs)
    if tuple(totals) != lengths:
        raise ManifestError(f"{where}: doc_lengths do not equal the summed term frequencies")
    return Postings(chunk_ids, lengths, terms)


class Bm25Index:
    """Immutable scoring structure with fixed global statistics."""

    def __init__(self, postings: Postings, *, idf: str, k1: float, b: float) -> None:
        if idf not in ("lucene", "robertson"):
            raise ValueError(f"unknown idf variant {idf!r}")
        self.postings = postings
        self.chunk_count = len(postings.chunk_ids)
        self.total_tokens = sum(postings.doc_lengths)
        self.average_length = self.total_tokens / self.chunk_count if self.chunk_count else 0.0
        self.vocabulary_size = len(postings.terms)
        self._k1 = k1
        self._norm = [
            k1 * (1.0 - b + (b * length / self.average_length if self.average_length else 0.0))
            for length in postings.doc_lengths
        ]
        self._idf = {}
        for term, pairs in postings.terms.items():
            ratio = (self.chunk_count - len(pairs) + 0.5) / (len(pairs) + 0.5)
            self._idf[term] = math.log1p(ratio) if idf == "lucene" else math.log(ratio)

    def score(self, terms: Sequence[str], allowed: Set[int] | None) -> dict[int, float]:
        """Scores of every eligible chunk: those in `allowed` containing >= 1 query term."""
        gains: dict[int, float] = {}
        for term in terms:
            pairs = self.postings.terms.get(term)
            if pairs is None:
                continue
            idf = self._idf[term]
            for index, frequency in pairs:
                if allowed is not None and index not in allowed:
                    continue
                gain = idf * frequency * (self._k1 + 1.0) / (frequency + self._norm[index])
                gains[index] = gains.get(index, 0.0) + gain
        return {index: round(total, SCORE_DECIMALS) + 0.0 for index, total in gains.items()}


class Bm25Retriever:
    """The `Retriever` port for the model-free baseline."""

    def __init__(self, corpus: Corpus, index: Bm25Index) -> None:
        if index.postings.chunk_ids != tuple(chunk.chunk_id for chunk in corpus.chunks):
            raise ManifestError("index and corpus describe different chunks")
        self._corpus = corpus
        self._index = index
        self._origin = [(corpus.doc_of(chunk).source_id, chunk.revision) for chunk in corpus.chunks]
        self._known = corpus.revisions_by_source()

    def eligible_chunks(self, request: SearchRequest) -> frozenset[int] | None:
        """Validate filters; return the chunk indexes they allow, or None for no filtering.

        An unknown source or version is an invalid request (never silently
        widened to another version); a valid filter may legitimately match nothing
        lexically and then yields an empty result.
        """
        sources = tuple(request.sources)
        versions: dict[str, str] = {}
        for source, revision in request.versions:
            if source in versions:
                raise InvalidRequestError(f"version filter repeats source {source!r}")
            versions[source] = revision
        if len(set(sources)) != len(sources):
            raise InvalidRequestError("source filter contains duplicates")
        for source in (*sources, *versions):
            if source not in self._known:
                raise InvalidRequestError(
                    f"unknown source {source!r}; available: {', '.join(self._known) or 'none'}"
                )
        for source, revision in versions.items():
            if sources and source not in sources:
                raise InvalidRequestError(f"version given for {source!r}, which the source filter excludes")
            if revision not in self._known[source]:
                raise InvalidRequestError(
                    f"source {source!r} has no version {revision!r}; "
                    f"available: {', '.join(self._known[source])}"
                )
        if not sources and not versions:
            return None
        return frozenset(
            index for index, (source, revision) in enumerate(self._origin)
            if (not sources or source in sources) and versions.get(source, revision) == revision
        )

    def eligible_chunk_ids(self, request: SearchRequest) -> frozenset[str] | None:
        """`eligible_chunks` as chunk IDs (None = unfiltered); validates the filters like `retrieve`."""
        view = self.eligible_chunks(request)
        if view is None:
            return None
        ids = self._index.postings.chunk_ids
        return frozenset(ids[i] for i in view)

    def retrieve(self, request: SearchRequest) -> tuple[RetrievalCandidate, ...]:
        return self.retrieve_within(request, None)

    def retrieve_within(
        self, request: SearchRequest, allowed_chunk_ids: Set[str] | None
    ) -> tuple[RetrievalCandidate, ...]:
        """Like `retrieve`, additionally restricted to `allowed_chunk_ids` (a restricted view).

        Used to mine negatives from a train-owned pool: the restriction masks chunks before
        ranking exactly as filters do, so corpus-wide statistics and scores are unchanged.
        """
        if request.candidate_k < 1:
            raise InvalidRequestError("candidate_k must be at least 1")
        allowed = self.eligible_chunks(request)
        if allowed_chunk_ids is not None:
            view = frozenset(i for i, chunk_id in enumerate(self._index.postings.chunk_ids) if chunk_id in allowed_chunk_ids)
            allowed = view if allowed is None else allowed & view
        scores = self._index.score(unique_query_terms(request.query), allowed)
        chunk_ids = self._index.postings.chunk_ids
        best = heapq.nsmallest(
            request.candidate_k, scores.items(), key=lambda item: (-item[1], chunk_ids[item[0]])
        )
        return tuple(
            RetrievalCandidate(chunk=self._corpus.chunks[index], bm25_score=score, baseline_rank=rank)
            for rank, (index, score) in enumerate(best, start=1)
        )
