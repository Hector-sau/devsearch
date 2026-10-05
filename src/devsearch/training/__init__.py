"""Native training boundary. No trainer, optimizer or checkpoint IO is implemented."""

from pathlib import Path
from typing import Protocol

from devsearch.contracts import ArtifactRef


class Trainer(Protocol):
    def train(
        self, config: Path, *, resume: ArtifactRef | None = None
    ) -> ArtifactRef:
        """Return a run manifest; v1 resumes only validated complete-epoch state."""
        ...
