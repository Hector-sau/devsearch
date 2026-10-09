"""Structure-aware, token-validated chunking (docs/02 §3).

A document's blocks become one normalized text (blocks separated by a blank line) with an exact source map. Passages are
cut along the document structure: a section is a heading plus the blocks up to the next heading, whole blocks are packed into
a passage while the *measured* token count of the passage text stays within the passage budget, an over-long block is split
only at line (code) or sentence/line (prose) boundaries, and anything that still cannot fit is excluded and counted. Nothing
is ever truncated. Every passage is re-measured on its exact text with the pinned tokenizer before it is accepted.

Passages of one section share an evidence group (a question answerable from one of them must not leak through a sibling), and
a passage holding part of a code block is marked `code_fragment`.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from devsearch.data.corpus import ChunkSpan
from devsearch.data.sourcemap import SourceSegment
from devsearch.ingestion.model import Block, CellSpan, Extraction

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        """Token count of `text` alone, without special tokens."""
        ...


@dataclass(frozen=True)
class ChunkingPolicy:
    passage_budget: int
    min_chunk_tokens: int = 8
    overlong_code: str = "split"          # "split": cut at line boundaries (fragments) | "exclude": drop the block

    def __post_init__(self) -> None:
        if self.passage_budget < 8 or self.min_chunk_tokens < 1 or self.min_chunk_tokens >= self.passage_budget or self.overlong_code not in ("split", "exclude"):
            raise ValueError("invalid chunking policy")


@dataclass
class _Unit:
    start: int
    end: int
    tokens: int
    section: int
    path: tuple[str, ...]
    heading: bool = False
    fragment: bool = False
    barrier: bool = False                 # excluded: passages never span it


@dataclass
class Chunked:
    text: str
    segments: tuple[SourceSegment, ...]
    chunks: tuple[ChunkSpan, ...]
    tokens: tuple[int, ...]               # measured passage tokens, aligned with `chunks`
    excluded: Counter = field(default_factory=Counter)        # reason -> count
    excluded_examples: list[dict] = field(default_factory=list)


def _section_group(doc_id: str, revision: str, section: int) -> str:
    return "eg-" + hashlib.sha256(f"{doc_id}\x00{revision}\x00{section}".encode()).hexdigest()[:24]


def assemble(extraction: Extraction, path: str) -> tuple[str, list[SourceSegment], list[tuple[Block, int, list[int]]]]:
    """Normalized text, its source segments, and per block: (block, start offset, start offset of each kept line)."""
    text, segments, placed = "", [], []
    for block in extraction.blocks:
        lines = list(block.lines)
        while lines and not lines[0].text.strip():
            lines.pop(0)
        while lines and not lines[-1].text.strip():
            lines.pop()
        if not lines:
            continue
        if text:
            text += "\n\n"
        block_start, starts = len(text), []
        for index, line in enumerate(lines):
            if index:
                text += "\n"
            start = len(text)
            starts.append(start)
            text += line.text
            if len(line.text) == 0:
                continue
            if isinstance(line.origin, CellSpan):
                segments.append(SourceSegment(start, len(text), path, False, cell_id=line.origin.cell_id,
                                              cell_start_char=line.origin.start, cell_end_char=line.origin.end))
            else:
                segments.append(SourceSegment(start, len(text), path, False, start_line=line.origin, end_line=line.origin))
        placed.append((Block(block.kind, tuple(lines), block.level, block.title), block_start, starts))
    return text + "\n", segments, placed


def chunk_document(extraction: Extraction, *, doc_id: str, revision: str, path: str, counter: TokenCounter, policy: ChunkingPolicy) -> Chunked:
    text, segments, placed = assemble(extraction, path)
    result = Chunked(text, tuple(segments), (), ())
    units: list[_Unit] = []
    stack: list[tuple[int, str]] = []
    section = 0

    def exclude(reason: str, start: int, end: int, lines: int = 1) -> None:
        result.excluded[reason] += lines
        if len(result.excluded_examples) < 20:
            result.excluded_examples.append({"reason": reason, "path": path, "chars": [start, end]})
        units.append(_Unit(start, end, 0, section, tuple(t for _, t in stack), barrier=True))

    def measure(start: int, end: int) -> int:
        return counter.count(text[start:end])

    for block, block_start, starts in placed:
        block_end = starts[-1] + len(block.lines[-1].text)
        if block.kind == "heading":
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.title))
            section += 1
            units.append(_Unit(block_start, block_end, measure(block_start, block_end), section, tuple(t for _, t in stack), heading=True))
            continue
        path = tuple(t for _, t in stack)
        whole = measure(block_start, block_end)
        if whole <= policy.passage_budget:
            units.append(_Unit(block_start, block_end, whole, section, path))
            continue
        if block.kind == "code" and policy.overlong_code == "exclude":
            exclude("overlong-code-block", block_start, block_end, len(block.lines))
            continue
        for index, line in enumerate(block.lines):
            start, end = starts[index], starts[index] + len(line.text)
            if start == end:
                continue
            tokens = measure(start, end)
            if tokens <= policy.passage_budget:
                units.append(_Unit(start, end, tokens, section, path, fragment=block.kind == "code"))
                continue
            if block.kind == "code":
                exclude("overlong-code-line", start, end)
                continue
            position = start
            for sentence in _SENTENCE.split(line.text):
                s_start = text.index(sentence, position) if sentence else position
                s_end = s_start + len(sentence)
                position = s_end
                if not sentence.strip():
                    continue
                tokens = measure(s_start, s_end)
                if tokens <= policy.passage_budget:
                    units.append(_Unit(s_start, s_end, tokens, section, path))
                else:
                    exclude("overlong-sentence", s_start, s_end)

    chunks: list[ChunkSpan] = []
    token_counts: list[int] = []
    by_section: dict[int, list[_Unit]] = {}
    for unit in units:
        by_section.setdefault(unit.section, []).append(unit)
    for number in sorted(by_section):
        pending = by_section[number]
        index = 0
        while index < len(pending):
            if pending[index].barrier:
                index += 1
                continue
            taken: list[_Unit] = []
            total = 0
            while index < len(pending) and not pending[index].barrier and total + pending[index].tokens <= policy.passage_budget:
                total += pending[index].tokens
                taken.append(pending[index])
                index += 1
            if not taken:                                                         # a single unit already over budget cannot happen; stay safe
                result.excluded["unit-over-budget"] += 1
                index += 1
                continue
            exact = measure(taken[0].start, taken[-1].end)
            while exact > policy.passage_budget and len(taken) > 1:             # token counts are not perfectly additive: give a unit back
                index -= 1
                taken.pop()
                exact = measure(taken[0].start, taken[-1].end)
            if exact > policy.passage_budget:
                result.excluded["passage-over-budget"] += 1
                continue
            if all(unit.heading for unit in taken):                             # a heading with no room for its first block: the path keeps it
                result.excluded["heading-without-body"] += 1
                continue
            body = [unit for unit in taken if not unit.heading]
            if exact < policy.min_chunk_tokens:
                result.excluded["short-passage"] += 1
                if len(result.excluded_examples) < 20:
                    result.excluded_examples.append({"reason": "short-passage", "path": path, "chars": [taken[0].start, taken[-1].end]})
                continue
            chunks.append(ChunkSpan(taken[0].start, taken[-1].end, _section_group(doc_id, revision, number), body[0].path, None,
                                    any(unit.fragment for unit in body)))
            token_counts.append(exact)
    result.chunks, result.tokens = tuple(chunks), tuple(token_counts)
    return result
