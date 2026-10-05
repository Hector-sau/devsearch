"""Static source extraction port. Never execute source scripts or directives."""

from pathlib import Path
from typing import Protocol

from devsearch.contracts import Doc


class SourceReader(Protocol):
    def read_snapshot(self, source_manifest: Path) -> tuple[Doc, ...]:
        """Read pinned local sources; preserve source mappings and exclusions."""
        ...
