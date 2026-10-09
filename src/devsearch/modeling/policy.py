"""Pair-encoding policy shared by training, evaluation and serving (docs/03 §1, docs/10 §4).

One definition of how a (question, passage) pair becomes model input, with no third-party
imports so data preparation can validate lengths before any model library is present:

* `max_pair_tokens` bounds the whole pair including special tokens; `max_query_tokens` is the
  query's reserve. The passage budget is `max_pair_tokens - max_query_tokens - special`, where
  `special` is **measured** from the pinned tokenizer, never assumed.
* Over-long input is an error, never silently truncated: a truncated passage could drop the very
  evidence a label depends on, and training and serving must never disagree about the input.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from devsearch.contracts import Chunk
from devsearch.errors import DevSearchError


class LengthError(DevSearchError):
    """A question or pair exceeds the declared token budget."""


@dataclass(frozen=True)
class EncodedPair:
    ids: tuple[int, ...]
    type_ids: tuple[int, ...]


class PairTokenizer(Protocol):
    pair_special_tokens: int
    pad_id: int
    fingerprint: str

    def encode_pair(self, query: str, passage: str) -> EncodedPair: ...

    def count(self, text: str) -> int:
        """Token count of `text` alone, without special tokens."""
        ...


@dataclass(frozen=True)
class PairEncodingPolicy:
    max_pair_tokens: int = 256
    max_query_tokens: int = 64

    def __post_init__(self) -> None:
        if not (8 <= self.max_query_tokens < self.max_pair_tokens <= 4096):
            raise ValueError("need 8 <= max_query_tokens < max_pair_tokens <= 4096")

    def passage_budget(self, tokenizer: PairTokenizer) -> int:
        budget = self.max_pair_tokens - self.max_query_tokens - tokenizer.pair_special_tokens
        if budget < 1:
            raise ValueError("the policy leaves no room for a passage")
        return budget


def encode_pair(tokenizer: PairTokenizer, policy: PairEncodingPolicy, query: str, passage: str) -> EncodedPair:
    """Encode one pair under the policy or raise LengthError. Never truncates."""
    if tokenizer.count(query) > policy.max_query_tokens:
        raise LengthError(f"question has {tokenizer.count(query)} tokens; the reserve is {policy.max_query_tokens}")
    pair = tokenizer.encode_pair(query, passage)
    if len(pair.ids) > policy.max_pair_tokens:
        raise LengthError(f"pair has {len(pair.ids)} tokens; the limit is {policy.max_pair_tokens}")
    if len(pair.ids) != len(pair.type_ids):
        raise LengthError("tokenizer returned ids and type ids of different lengths")
    return pair


def check_corpus_budget(chunks: Sequence[Chunk], tokenizer: PairTokenizer, policy: PairEncodingPolicy) -> dict[str, object]:
    """Which passages cannot fit the budget. Exclusions are reported, never truncated away."""
    budget = policy.passage_budget(tokenizer)
    excluded, longest = [], 0
    for chunk in chunks:
        tokens = tokenizer.count(chunk.text)
        longest = max(longest, tokens)
        if tokens > budget:
            excluded.append({"chunk_id": chunk.chunk_id, "tokens": tokens})
    return {
        "chunks": len(chunks), "passage_budget": budget, "pair_special_tokens": tokenizer.pair_special_tokens,
        "max_pair_tokens": policy.max_pair_tokens, "max_query_tokens": policy.max_query_tokens,
        "excluded_count": len(excluded), "exclusion_rate": (len(excluded) / len(chunks)) if chunks else 0.0,
        "longest_passage_tokens": longest, "excluded": excluded, "tokenizer_fingerprint": tokenizer.fingerprint,
    }
