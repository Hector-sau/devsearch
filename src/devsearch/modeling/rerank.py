"""The reranker used by serving, offline ranking and the export checks: one object, one scoring path.

It wraps a loaded inference bundle and exposes only `score`: one finite logit per passage, in input
order, or an exception with every partial score discarded. Ordering lives elsewhere
(`devsearch.evaluation.ranking.order_by_score`) so ties break identically online and offline.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from devsearch.modeling.io import LoadedModel, check_parity
from devsearch.modeling.scoring import score_pairs


class Reranker:
    def __init__(self, loaded: LoadedModel, *, device: torch.device, batch_size: int = 8) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.loaded = loaded
        self.device = device
        self.batch_size = batch_size

    @property
    def model_id(self) -> str:
        return self.loaded.model_id

    def score(self, query: str, passages: Sequence[str], *, should_stop: Callable[[], bool] | None = None) -> list[float]:
        pairs = [(query, passage) for passage in passages]
        return score_pairs(self.loaded.model, self.loaded.tokenizer, self.loaded.policy, pairs,
                           device=self.device, batch_size=self.batch_size, should_stop=should_stop)

    def self_check(self, *, atol: float | None = None) -> dict[str, Any]:
        """Re-score the bundle's recorded fixed examples; used as the startup smoke test and for recovery."""
        return check_parity(self.loaded, device=self.device, atol=atol, batch_size=self.batch_size)
