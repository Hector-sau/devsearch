"""Explicit opt-in matched query objectives; never change the legacy flat-BCE path.

Each group has one positive and four negatives, scored once in one five-record forward.
The coefficient/order/reduction/selection versions are part of the checkpoint identity.
"""

import hashlib
from collections import Counter
from typing import TYPE_CHECKING

from devsearch.errors import ManifestError

if TYPE_CHECKING:
    from devsearch.training.data import GroupedTrainSet

GROUPED_PROTOCOL = "query-1p3h1r-forward-once-v1"
GROUPED_ORDER = "group-order-v1"
GROUPED_REDUCTION = "sum-query-loss/actual-queries-in-update-v1"
GROUPED_SELECTION = "fixed-final-epoch-3-v1"
SOURCE_KEYS = {"pairs_manifest", "train_queries", "train_qrels", "split_manifest", "assignment"}


def validate_grouped_data(data: "GroupedTrainSet") -> None:
    sources = dict(data.source_sha256)
    if (not data.groups or len(sources) != len(data.source_sha256) or not SOURCE_KEYS <= sources.keys()
            or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                   for v in sources.values())):
        raise ManifestError("grouped data needs nonempty groups and exact verified source SHA256 identities")
    query_ids, pair_ids = set(), set()
    for group in data.groups:
        if (not group.query_id or group.query_id in query_ids or group.query_split != "train"
                or len(group.records) != 5 or len(group.known_positive_chunk_ids) != 1
                or not group.known_positive_group_ids):
            raise ManifestError("each unique train query needs exactly one known positive and four negatives")
        query_ids.add(group.query_id)
        chunks, query_texts = set(), set()
        for index, item in enumerate(group.records):
            record, pair = item.record, item.pair
            expected_label = 1 if index == 0 else 0
            if (record.query_id != group.query_id or not record.pair_id or record.pair_id in pair_ids
                    or record.chunk_id in chunks or item.owner_split != "train" or not item.evidence_group_id or not item.merged_group_id
                    or type(record.label) is not int or record.label != expected_label
                    or type(pair.label) is not int or pair.label != expected_label
                    or not isinstance(pair.query_text, str) or not pair.query_text.strip()
                    or not isinstance(pair.passage_text, str) or not pair.passage_text.strip()):
                raise ManifestError("invalid grouped pair identity, ownership, order or label")
            pair_ids.add(record.pair_id)
            chunks.add(record.chunk_id)
            query_texts.add(pair.query_text)
            if expected_label:
                if (record.chunk_id not in group.known_positive_chunk_ids or item.merged_group_id not in group.known_positive_group_ids
                        or record.label_status != "reviewed_positive" or record.negative_method is not None):
                    raise ManifestError("grouped positive must match the confirmed training qrel")
            elif (record.chunk_id in group.known_positive_chunk_ids or item.merged_group_id in group.known_positive_group_ids
                  or record.label_status != "assumed_negative"
                  or record.negative_method not in ("bm25", "random")):
                raise ManifestError("grouped assumed negative has conflicting provenance")
        if len(query_texts) != 1 or Counter(x.record.negative_method for x in group.records[1:]) != {"bm25": 3, "random": 1}:
            raise ManifestError("grouped query text or three-hard/one-random composition differs")


def grouped_order(data: "GroupedTrainSet", seed: int, epoch: int) -> list[int]:
    return sorted(range(len(data.groups)), key=lambda i: (
        hashlib.sha256(f"{GROUPED_ORDER}|{seed}|{epoch}|{data.groups[i].query_id}".encode()).hexdigest(),
        data.groups[i].query_id))


def query_loss(logits, loss_kind: str):
    """Return one differentiable scalar. Torch stays optional outside the ML path."""
    import torch.nn.functional as functional

    if logits.ndim != 1 or logits.shape[0] != 5:
        raise ManifestError("grouped objective expects exactly five logits, positive first")
    positive, negatives = logits[0], logits[1:]
    if loss_kind == "group_bce_weight4":
        return (4 * functional.softplus(-positive) + functional.softplus(negatives).sum()) / 5
    if loss_kind == "pairwise_logistic_scale_4_over_5":
        return functional.softplus(negatives - positive).sum() / 5
    raise ManifestError("unknown grouped objective")
