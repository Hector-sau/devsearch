"""Shared CLI/API orchestration port, not an HTTP server implementation."""

from typing import Protocol

from devsearch.contracts import SearchRequest, SearchResult


class SearchPipeline(Protocol):
    def search(self, request: SearchRequest) -> SearchResult:
        """Compose retrieval/scoring; retain honest baseline and fallback states."""
        ...
