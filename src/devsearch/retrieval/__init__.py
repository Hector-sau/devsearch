"""Retrieval port; independent of modeling, training and serving adapters."""

from typing import Protocol

from devsearch.contracts import RetrievalCandidate, SearchRequest


class Retriever(Protocol):
    def retrieve(self, request: SearchRequest) -> tuple[RetrievalCandidate, ...]:
        """Fixed corpus IDF; metadata/lexical eligibility before deterministic top-k."""
        ...
