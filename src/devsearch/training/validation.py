"""Validation of a model on frozen candidate lists: group-macro MRR@10 / nDCG@10 (selection metric)."""

import torch

from devsearch.evaluation.ranking import order_by_score, ranked_record, score_with_positives
from devsearch.modeling.cross_encoder import CrossEncoder
from devsearch.modeling.policy import PairEncodingPolicy, PairTokenizer
from devsearch.modeling.scoring import score_pairs
from devsearch.training.data import ValidationSet


def evaluate_validation(
    model: CrossEncoder, tokenizer: PairTokenizer, policy: PairEncodingPolicy, validation: ValidationSet,
    device: torch.device, batch_size: int,
) -> dict[str, object]:
    """Score every validation candidate once (all-or-nothing), rank each list, aggregate."""
    pairs = [(q.text, text) for q in validation.queries for text in q.candidate_texts]
    scores = score_pairs(model, tokenizer, policy, pairs, device=device, batch_size=batch_size)
    ranked, cursor = [], 0
    for query in validation.queries:
        count = len(query.candidate_ids)
        ranked.append(ranked_record(query.query_id, "", order_by_score(query.candidate_ids, scores[cursor:cursor + count])))
        cursor += count
    result = score_with_positives(ranked, {q.query_id: q.positives for q in validation.queries}, validation.group_of)
    return {k: result[k] for k in ("group_macro", "query_macro", "queries_scored", "groups_scored")}
