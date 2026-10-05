"""Dynamic-padding collation for encoded pairs (docs/03 §1: `collate_fn`, no truncation)."""

from collections.abc import Sequence

import torch

from devsearch.modeling.policy import EncodedPair


def collate(pairs: Sequence[EncodedPair], pad_id: int) -> dict[str, torch.Tensor]:
    """Pad to the longest pair in the batch; the attention mask marks real tokens."""
    if not pairs:
        raise ValueError("cannot collate an empty batch")
    width = max(len(p.ids) for p in pairs)
    ids = torch.full((len(pairs), width), pad_id, dtype=torch.long)
    types = torch.zeros((len(pairs), width), dtype=torch.long)
    mask = torch.zeros((len(pairs), width), dtype=torch.long)
    for row, pair in enumerate(pairs):
        n = len(pair.ids)
        ids[row, :n] = torch.tensor(pair.ids, dtype=torch.long)
        types[row, :n] = torch.tensor(pair.type_ids, dtype=torch.long)
        mask[row, :n] = 1
    return {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
