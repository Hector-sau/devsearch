"""Experimental in-process boundary types, not complete persisted/API schemas.

The governing schemas and validation requirements are in docs/02 and docs/04.
Frozen dataclasses express ownership; they do not validate untrusted input.
Collections use tuples so boundary values do not contain mutable lists/dicts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SearchMode = Literal["baseline", "rerank", "compare"]
SplitName = Literal["development", "train", "validation", "test"]


@dataclass(frozen=True)
class Doc:
    doc_id: str
    revision: str
    source_id: str
    title: str
    text: str
    source_url: str | None
    content_sha256: str
    parser_version: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    revision: str
    text: str
    start_char: int
    end_char: int
    heading_path: tuple[str, ...]
    chunker_version: str
    evidence_group_id: str
    owner_split: SplitName | None = None
    code_fragment: bool = False


@dataclass(frozen=True)
class SearchRequest:
    query: str
    sources: tuple[str, ...] = ()
    versions: tuple[tuple[str, str], ...] = ()
    candidate_k: int = 50
    top_k: int = 5
    mode: SearchMode = "baseline"


@dataclass(frozen=True)
class RetrievalCandidate:
    chunk: Chunk
    bm25_score: float
    baseline_rank: int


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: RetrievalCandidate
    rerank_logit: float
    reranked_rank: int


@dataclass(frozen=True)
class SearchWarning:
    code: str
    message: str


@dataclass(frozen=True)
class SearchResult:
    """Core result only; API timing/source-map/error envelopes remain to be built."""

    mode_requested: SearchMode
    mode_effective: SearchMode
    status: Literal["ok", "empty", "degraded"]
    baseline: tuple[RetrievalCandidate, ...]
    reranked: tuple[ScoredCandidate, ...]
    warnings: tuple[SearchWarning, ...] = ()


@dataclass(frozen=True)
class ArtifactRef:
    """A manifest reference; existence, containment and hashes need validation."""

    manifest_path: Path
