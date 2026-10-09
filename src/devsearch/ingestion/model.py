"""The intermediate representation shared by every reader: blocks of lines with exact origins.

A reader turns one source file into `Block`s. Each `Line` carries the text that will enter the normalized document and
where it came from: a 1-based source line (line form) or a verbatim span of a notebook cell (cell form). Dropped
constructs are counted by kind so coverage loss is reported, never hidden.
"""

from collections import Counter
from dataclasses import dataclass, field

from devsearch.errors import DevSearchError

KINDS = ("heading", "text", "code")


class ParseFailure(DevSearchError):
    """The file cannot be extracted safely; it is quarantined with this reason (never indexed as raw text)."""


@dataclass(frozen=True)
class CellSpan:
    cell_id: str
    start: int          # code-point offsets inside the cell's normalized source
    end: int


@dataclass(frozen=True)
class Line:
    text: str
    origin: int | CellSpan          # 1-based source line, or the verbatim cell span (cell form)


@dataclass(frozen=True)
class Block:
    kind: str                        # heading | text | code
    lines: tuple[Line, ...]
    level: int = 0                   # headings only
    title: str = ""                  # headings only: the heading text without markup

    def __post_init__(self) -> None:
        if self.kind not in KINDS or not self.lines:
            raise ValueError("a block has a known kind and at least one line")

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class Extraction:
    fmt: str
    title: str | None = None
    blocks: list[Block] = field(default_factory=list)
    dropped: Counter = field(default_factory=Counter)            # construct kind -> number of source lines dropped
    notes: Counter = field(default_factory=Counter)              # informational counters (e.g. notebook outputs ignored)

    def drop(self, kind: str, lines: int = 1) -> None:
        self.dropped[kind] += lines


def clean_title(text: str, limit: int = 200) -> str:
    """A heading as a one-line label: whitespace collapsed, non-printable characters removed, at most `limit` characters."""
    return " ".join("".join(ch if ch.isprintable() else " " for ch in text).split())[:limit].rstrip()


def normalize_newlines(text: str) -> str:
    """LF only, no BOM; NUL means binary data and is refused."""
    if "\x00" in text:
        raise ParseFailure("the file contains NUL characters (binary data)")
    return text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


MAX_LINE_CHARS = 10_000


def check_lines(lines: list[str]) -> None:
    for number, line in enumerate(lines, start=1):
        if len(line) > MAX_LINE_CHARS:
            raise ParseFailure(f"line {number} has {len(line)} characters (more than {MAX_LINE_CHARS}); minified or generated content is not indexed")
