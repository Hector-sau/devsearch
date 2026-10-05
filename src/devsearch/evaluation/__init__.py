"""Separate ranking, blinded pooling and final scoring; never tune on test scores."""

from typing import Protocol

from devsearch.contracts import ArtifactRef


class Evaluator(Protocol):
    def rank(self, comparison_plan: ArtifactRef) -> ArtifactRef:
        """Generate fixed-system ranks without reading test judgments."""
        ...

    def pool(self, plan: ArtifactRef, ranked: ArtifactRef) -> ArtifactRef:
        """Export a blinded review pool at its predeclared depth."""
        ...

    def score(
        self, plan: ArtifactRef, ranked: ArtifactRef, final_test: ArtifactRef
    ) -> ArtifactRef:
        """Compute a report against one reviewed and sealed final qrels version."""
        ...
