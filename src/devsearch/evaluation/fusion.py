"""Shared offline/online rank fusion. No model loading, labels or training here.

The historical R5 formula is (1-alpha)/(k+BM25 rank) + alpha/(k+neural
rank), with one-based ranks and ascending chunk_id for every score tie.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from devsearch.errors import ConfigError
from devsearch.evaluation.ranking import order_by_score


@dataclass(frozen=True)
class RankingPolicy:
    method: str = "logit"
    alpha: float = 0.5
    rank_constant: int = 60

    def __post_init__(self) -> None:
        if self.method not in ("logit", "rank_fusion"):
            raise ConfigError("ranking.method must be logit or rank_fusion")
        if (isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float))
                or not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1):
            raise ConfigError("ranking.alpha must be finite in 0..1")
        if isinstance(self.rank_constant, bool) or not isinstance(self.rank_constant, int) or self.rank_constant < 1:
            raise ConfigError("ranking.rank_constant must be a positive integer")

    @classmethod
    def from_record(cls, record: Any = None) -> "RankingPolicy":
        if record is None:
            return cls()                         # old releases remain raw-logit releases
        if not isinstance(record, Mapping):
            raise ConfigError("ranking must be an object")
        method = record.get("method")
        keys = {"method", "alpha", "rank_constant"} if method == "rank_fusion" else {"method"}
        if set(record) != keys:
            raise ConfigError(f"ranking requires exactly {sorted(keys)}")
        return cls(method, record.get("alpha", 0.5), record.get("rank_constant", 60))

    def to_record(self) -> dict[str, Any]:
        if self.method == "logit":
            return {"method": "logit"}
        return {"method": self.method, "alpha": self.alpha, "rank_constant": self.rank_constant}

    def order(
        self, chunk_ids: Sequence[str], logits: Sequence[float], bm25_ranks: Sequence[int],
    ) -> list[tuple[str, int, float]]:
        if (len(set(chunk_ids)) != len(chunk_ids) or len(bm25_ranks) != len(chunk_ids)
                or any(isinstance(r, bool) or not isinstance(r, int) for r in bm25_ranks)
                or sorted(bm25_ranks) != list(range(1, len(chunk_ids) + 1))):
            raise ValueError("unique candidates and a permutation of one-based BM25 ranks are required")
        neural = order_by_score(chunk_ids, logits)  # validates all logits even at alpha=0
        if self.method == "logit":
            return neural
        rank_of = {cid: rank for cid, rank, _ in neural}
        fused = [(1 - self.alpha) / (self.rank_constant + lexical)
                 + self.alpha / (self.rank_constant + rank_of[cid])
                 for cid, lexical in zip(chunk_ids, bm25_ranks)]
        return order_by_score(chunk_ids, fused)
