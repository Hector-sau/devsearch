"""Scoring port. A later adapter loads Transformers/PyTorch lazily."""

from typing import Protocol

from devsearch.contracts import RetrievalCandidate


class PairScorer(Protocol):
    def score(
        self, query: str, candidates: tuple[RetrievalCandidate, ...]
    ) -> tuple[float, ...]:
        """Return one finite logit per input, in input order; never partial output."""
        ...
