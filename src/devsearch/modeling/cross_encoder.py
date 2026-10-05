"""MiniLM-style cross-encoder: a BERT encoder, dropout, and one linear layer giving one logit.

First-token pooling is the documented starting choice (docs/03 §1). The output is a raw,
uncalibrated logit, never a probability, and training applies `BCEWithLogitsLoss` to it with no
extra sigmoid. `set_encoder_frozen(True)` freezes encoder parameters *and* keeps the encoder in
eval mode, so frozen-head training does not let encoder dropout vary the features.
"""

from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig, BertModel

from devsearch.errors import ManifestError


class CrossEncoder(nn.Module):
    def __init__(self, encoder: BertModel, *, dropout: float = 0.1, head_seed: int = 0) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        self.encoder_frozen = False
        self.init_head(head_seed)

    def init_head(self, seed: int) -> None:
        """Seeded head initialization, so frozen and full runs start from the same scoring layer."""
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(self.head.weight.shape, generator=generator) * 0.02)
            self.head.bias.zero_()

    def set_encoder_frozen(self, frozen: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not frozen)
        self.encoder_frozen = frozen
        if frozen:
            self.encoder.eval()

    def train(self, mode: bool = True) -> "CrossEncoder":
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        if self.encoder_frozen:
            with torch.no_grad():
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        else:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        return self.head(self.dropout(hidden[:, 0])).squeeze(-1)


def build_tiny_encoder(
    vocab_size: int, *, hidden_size: int = 32, layers: int = 2, heads: int = 2, intermediate_size: int = 64,
    max_positions: int = 512, seed: int = 0, hidden_dropout: float = 0.1, attention_dropout: float = 0.1,
) -> BertModel:
    """A small randomly initialized BERT for diagnostics, rehearsals and CPU tests. Never a real model."""
    config = BertConfig(
        vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=layers, num_attention_heads=heads,
        intermediate_size=intermediate_size, max_position_embeddings=max_positions, hidden_dropout_prob=hidden_dropout,
        attention_probs_dropout_prob=attention_dropout,
    )
    torch.manual_seed(seed)
    return BertModel(config, add_pooling_layer=False)


def load_pretrained_encoder(directory: Path) -> BertModel:
    """Load a pinned encoder from a local directory only; this never touches the network."""
    directory = Path(directory)
    if not (directory / "config.json").is_file():
        raise ManifestError(f"{directory} holds no config.json: the pinned model assets are not prepared")
    return BertModel.from_pretrained(str(directory), add_pooling_layer=False, local_files_only=True)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
