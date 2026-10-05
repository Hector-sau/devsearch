"""Immutable artifact validation/export ports; no filesystem operations yet."""

from pathlib import Path
from typing import Protocol

from devsearch.contracts import ArtifactRef


class ArtifactValidator(Protocol):
    def validate(self, artifact: ArtifactRef, runtime_root: Path) -> None:
        """Reject invalid schemas, escaping paths, missing files and hash mismatch."""
        ...


class BundleExporter(Protocol):
    def export(self, config: Path, destination: Path) -> ArtifactRef:
        """Create a portable offline bundle without mutating previous artifacts."""
        ...
