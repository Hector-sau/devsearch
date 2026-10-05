"""Chunking and label-preparation ports; no model weights are required here."""

from pathlib import Path
from typing import Protocol

from devsearch.contracts import ArtifactRef, Chunk, Doc


class Chunker(Protocol):
    def chunk(self, document: Doc) -> tuple[Chunk, ...]:
        """Use pinned tokenizer length rules and preserve evidence/source spans."""
        ...


class DataPreparer(Protocol):
    def prepare(self, config: Path, corpus: ArtifactRef) -> ArtifactRef:
        """Audit supplied labels/groups and materialize versioned splits/pairs."""
        ...
