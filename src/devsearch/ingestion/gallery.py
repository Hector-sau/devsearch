"""Tutorial Python in the sphinx-gallery layout, read as text (docs/02 §3). The file is parsed, never imported or run.

The module docstring is reStructuredText (title, introduction). The rest is cut at `# %%` / `####…` separator lines into
blocks: right after a separator, the run of comment lines is reStructuredText prose (the `# ` prefix removed); everything else
(including the code before the first separator) is a code block kept verbatim. Text before the docstring (licence headers) is boilerplate and is dropped. A file that is not valid Python, or
whose docstring cannot be mapped to exact source lines, is quarantined.
"""

import ast
import re

from devsearch.ingestion.model import Block, Extraction, Line, ParseFailure, check_lines, normalize_newlines
from devsearch.ingestion.rst import parse_rows

_SEPARATOR = re.compile(r"^\s*(?:#\s?%%.*|#{20,}\s*)$")


def _comment_text(line: str) -> str:
    body = line.lstrip()[1:]
    return body[1:] if body.startswith(" ") else body


_QUOTE_OPEN = re.compile(r"""^[rRuU]?('{3}|"{3}|'|")""")


def _column(line: str, byte_offset: int) -> int:
    """`ast` columns count UTF-8 bytes; convert to a character index."""
    return len(line.encode("utf-8")[:byte_offset].decode("utf-8", errors="ignore"))


def _raw_docstring_rows(lines: list[str], node: ast.Expr) -> list[tuple[str, int]]:
    """The docstring as the author wrote it: the source lines of the literal without its opening and closing quotes."""
    first, last = node.lineno, node.end_lineno or node.lineno
    begin = _column(lines[first - 1], node.col_offset)
    opening = _QUOTE_OPEN.match(lines[first - 1][begin:])
    if opening is None:
        raise ParseFailure("the module docstring has an unrecognised quote style")
    start = begin + opening.end()
    stop = _column(lines[last - 1], node.end_col_offset or 0) - len(opening.group(1))
    if first == last:
        return [(lines[first - 1][start:stop].rstrip(), first)]
    rows = [(lines[first - 1][start:].rstrip(), first)]
    rows += [(lines[number - 1].rstrip(), number) for number in range(first + 1, last)]
    rows.append((lines[last - 1][:stop].rstrip(), last))
    return rows


def parse_gallery(text: str) -> Extraction:
    lines = normalize_newlines(text).split("\n")
    check_lines(lines)
    try:
        tree = ast.parse("\n".join(lines))
    except (SyntaxError, ValueError, RecursionError, MemoryError) as error:
        raise ParseFailure(f"not valid Python source: {type(error).__name__}: {error}") from error
    extraction = Extraction("python-gallery")
    styles: dict = {}
    after = 0
    first = tree.body[0] if tree.body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        value_lines = first.value.value.split("\n")
        if (first.end_lineno or first.lineno) - first.lineno == len(value_lines) - 1:
            rows = [(row.rstrip(), first.lineno + offset) for offset, row in enumerate(value_lines)]
        else:                                    # escapes or continuations changed the line count: read the raw source lines instead
            rows = _raw_docstring_rows(lines, first)
            extraction.notes["docstring-read-raw"] += 1
        sub = Extraction("python-gallery")
        parse_rows(rows, sub, styles)
        extraction.blocks.extend(sub.blocks)
        extraction.dropped.update(sub.dropped)
        after = first.end_lineno or first.lineno
    else:
        extraction.drop("no-module-docstring")
    segment: list[tuple[str, int]] = []
    state = {"after_separator": False}

    def close() -> None:
        while segment and not segment[0][0].strip():
            segment.pop(0)
        while segment and not segment[-1][0].strip():
            segment.pop()
        if not segment:
            return
        prose = 0
        if state["after_separator"]:                       # the comment lines right after a separator are reStructuredText prose
            while prose < len(segment) and segment[prose][0].lstrip().startswith("#"):
                prose += 1
        if prose:
            sub = Extraction("python-gallery")
            parse_rows([(_comment_text(row), number) for row, number in segment[:prose]], sub, styles)
            extraction.blocks.extend(sub.blocks)
            extraction.dropped.update(sub.dropped)
        code = segment[prose:]
        while code and not code[0][0].strip():
            code.pop(0)
        if code:
            extraction.blocks.append(Block("code", tuple(Line(row.rstrip(), number) for row, number in code)))
        segment.clear()

    for number in range(after + 1, len(lines) + 1):
        row = lines[number - 1]
        if _SEPARATOR.match(row):
            close()
            state["after_separator"] = True
            extraction.drop("gallery-separator")
            continue
        segment.append((row, number))
    close()
    extraction.title = next((b.title for b in extraction.blocks if b.kind == "heading"), None)
    return extraction
