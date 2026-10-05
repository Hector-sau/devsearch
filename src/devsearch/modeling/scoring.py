"""One scoring function for validation, ranking and serving (shared pair encoding, finite or nothing).

Training validation, offline ranking and the online reranker all call `score_pairs`, so there is
one definition of encoding, batching order and finiteness checking. Batches follow input order
(no length sorting), so the same inputs always form the same batches. Scoring is all-or-nothing:
a stop request between batches, a non-finite logit or any exception discards every partial score.
"""

import math
from collections.abc import Callable, Sequence

import torch

from devsearch.errors import DevSearchError
from devsearch.modeling.batching import collate
from devsearch.modeling.cross_encoder import CrossEncoder
from devsearch.modeling.policy import PairEncodingPolicy, PairTokenizer, encode_pair


class DeadlineExceeded(DevSearchError):
    """A deadline or stop request arrived between batches; all partial scores are discarded."""


class NonFiniteScoreError(DevSearchError):
    """The model produced NaN or infinity; no ranking may be published from it."""


@torch.no_grad()
def score_pairs(
    model: CrossEncoder, tokenizer: PairTokenizer, policy: PairEncodingPolicy, pairs: Sequence[tuple[str, str]],
    *, device: torch.device, batch_size: int, should_stop: Callable[[], bool] | None = None,
) -> list[float]:
    """One finite logit per (query, passage), in input order. Raises instead of returning partial output."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    was_training = model.training
    model.eval()
    scores: list[float] = []
    try:
        for start in range(0, len(pairs), batch_size):
            if should_stop is not None and should_stop():
                raise DeadlineExceeded(f"stopped after {len(scores)} of {len(pairs)} scores; partial output discarded")
            batch = collate([encode_pair(tokenizer, policy, q, p) for q, p in pairs[start:start + batch_size]], tokenizer.pad_id)
            logits = model(**{key: value.to(device) for key, value in batch.items()}).detach().float().cpu().tolist()
            if not all(math.isfinite(value) for value in logits):
                raise NonFiniteScoreError("the model produced a non-finite logit")
            scores.extend(logits)
    finally:
        model.train(was_training)
    return scores
