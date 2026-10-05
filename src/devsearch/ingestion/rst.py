"""reStructuredText, extracted statically (docs/02 §3). No directive is ever executed or resolved.

Supported: title/section headings (overline or underline, levels by first appearance), paragraphs and lists as text,
literal blocks (`::`) and `code-block`/`code`/`sourcecode` directives as code, admonition-style directives as text (their
body is parsed again), and inline roles/literals reduced to their visible text. Every other directive (toctree, image,
include, raw, literalinclude, autodoc ...), comments, link targets and table borders are dropped with a per-kind count; a
dropped directive's indented body goes with it. Layout containers (tab-set, tab-item, dropdown, grid, card) are transparent:
their content is read, the wrapper is not. Normalized lines keep the number of the source line they came from.
"""

import re

from devsearch.ingestion.model import Block, Extraction, Line, check_lines, clean_title, normalize_newlines

_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([A-Za-z][\w:+-]*)::[ \t]*(.*)$")
_COMMENT = re.compile(r"^(\s*)\.\.(?:\s|$)")
_TARGET = re.compile(r"^\s*\.\.\s+(?:_[^:]*:|\[[^\]]+\]|\|[^|]+\|)")
_ADORNMENT = re.compile(r"^([!-/:-@\[-`{-~])\1{2,}\s*$")
_OPTION = re.compile(r"^\s*:[\w-]+:")
_GRID_BORDER = re.compile(r"^\s*\+[-=+]+\+\s*$")
_SIMPLE_BORDER = re.compile(r"^\s*=+(?:\s+=+)+\s*$")
_ROLE = re.compile(r":[A-Za-z][\w.+-]*:`([^`]+)`")
_LINK = re.compile(r"`([^`<]+?)\s*<[^>]+>`_{1,2}")
_LITERAL = re.compile(r"``([^`]+)``")
_INTERPRETED = re.compile(r"(?<![\w`])`([^`]+)`(?![\w`])")
_LABEL_TARGET = re.compile(r"^(.*?)\s*<([^<>]+)>$")
_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")

CODE_DIRECTIVES = {"code-block", "code", "sourcecode", "prompt", "parsed-literal"}
CONTAINER_DIRECTIVES = {"tab-set", "tab-item", "tab-set-code", "dropdown", "grid", "grid-item", "grid-item-card", "card", "container", "div"}
TEXT_DIRECTIVES = {"note", "warning", "tip", "important", "caution", "attention", "hint", "danger", "error", "seealso", "admonition",
                   "versionadded", "versionchanged", "deprecated", "topic", "sidebar", "epigraph", "highlights", "pull-quote"}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _role(match: re.Match) -> str:
    content = match.group(1)
    labelled = _LABEL_TARGET.match(content)
    visible = labelled.group(1) if labelled and labelled.group(1) else (labelled.group(2) if labelled else content)
    return visible.lstrip("~!")


def inline_text(line: str) -> str:
    """The visible text of a line: roles, hyperlink markup and literals reduced, escapes removed."""
    line = _ROLE.sub(_role, line)
    line = _LINK.sub(lambda m: m.group(1), line)
    line = _LITERAL.sub(lambda m: m.group(1), line)
    line = _INTERPRETED.sub(lambda m: m.group(1), line)
    return _ESCAPE.sub(r"\1", line)


def _dedent(rows: list[tuple[str, int]]) -> list[tuple[str, int]]:
    margin = min((_indent(text) for text, _ in rows if text.strip()), default=0)
    return [(text[margin:] if text.strip() else "", number) for text, number in rows]


def _trim(rows: list[tuple[str, int]]) -> list[tuple[str, int]]:
    while rows and not rows[0][0].strip():
        rows = rows[1:]
    while rows and not rows[-1][0].strip():
        rows = rows[:-1]
    return rows


def _body(rows: list[tuple[str, int]], start: int, margin: int) -> int:
    """Index after the indented body that starts at `start` (blank lines belong to it unless the next content is not indented)."""
    end, last = start, start
    while end < len(rows):
        text = rows[end][0]
        if text.strip() and _indent(text) <= margin:
            break
        if text.strip():
            last = end + 1
        end += 1
    return last


def parse_rows(rows: list[tuple[str, int]], extraction: Extraction, styles: dict[tuple[str, bool], int]) -> None:
    index, count = 0, len(rows)
    paragraph: list[tuple[str, int]] = []

    def emit(kind: str, items: list[tuple[str, int]], **extra) -> None:
        if items:
            extraction.blocks.append(Block(kind, tuple(Line(text, number) for text, number in items), **extra))

    def flush() -> bool:
        """Close the paragraph; returns True if it ended with `::` (a literal block follows)."""
        nonlocal paragraph
        if not paragraph:
            return False
        cleaned = inline_text("\n".join(text.strip() for text, _ in paragraph)).split("\n")     # markup may span lines
        numbers = [number for _, number in paragraph]
        items = [(line, numbers[min(position, len(numbers) - 1)]) for position, line in enumerate(cleaned)]
        introduces = items[-1][0].endswith("::")
        if introduces:
            last, number = items[-1]
            items[-1] = ((last[:-2].rstrip() + ":") if last[:-2].strip() else "", number)
            items = [item for item in items if item[0]]
        emit("text", items)
        paragraph = []
        return introduces

    def title_block(title: str, number: int, style: tuple[str, bool]) -> None:
        level = styles.setdefault(style, len(styles) + 1)
        label = clean_title(inline_text(title))
        if label:
            emit("heading", [(inline_text(title.strip()), number)], level=level, title=label)

    while index < count:
        text, number = rows[index]
        stripped = text.strip()
        if not stripped:
            flush()
            index += 1
            continue
        directive = _DIRECTIVE.match(text)
        if directive and not paragraph:
            margin, name, argument = len(directive.group(1)), directive.group(2).lower(), directive.group(3).strip()
            end = _body(rows, index + 1, margin)
            body = rows[index + 1:end]
            if name in CODE_DIRECTIVES:
                options = 0
                while options < len(body) and (_OPTION.match(body[options][0]) or not body[options][0].strip()):
                    options += 1                      # option lines (:linenos: ...) and the blank line before the code
                code = _trim(_dedent(body[options:]))
                emit("code", code)
                extraction.drop("rst-directive-marker")
            elif name in CONTAINER_DIRECTIVES:                  # layout wrappers: the content inside is real and is read, the wrapper is not
                inner = _dedent(body)
                skip = 0
                while skip < len(inner) and (_OPTION.match(inner[skip][0]) or not inner[skip][0].strip()):
                    skip += 1
                parse_rows(_trim(inner[skip:]), extraction, styles)
                extraction.drop("rst-directive-marker")
            elif name in TEXT_DIRECTIVES:
                inner = _dedent(body)
                if argument:
                    inner = [(argument, number), ("", number)] + inner
                parse_rows(_trim(inner), extraction, styles)
                extraction.drop("rst-directive-marker")
            else:
                extraction.drop(f"rst-directive:{name}", end - index)
            index = max(end, index + 1)
            continue
        if _TARGET.match(text) and not paragraph:
            extraction.drop("rst-target")
            index += 1
            continue
        comment = _COMMENT.match(text)
        if comment and not paragraph:
            end = _body(rows, index + 1, len(comment.group(1)))
            extraction.drop("rst-comment", max(end, index + 1) - index)
            index = max(end, index + 1)
            continue
        if _GRID_BORDER.match(text) or _SIMPLE_BORDER.match(text):
            extraction.drop("rst-table-border")
            index += 1
            continue
        if not paragraph:
            below = rows[index + 1][0] if index + 1 < count else ""
            if _ADORNMENT.match(text) and index + 2 < count and rows[index + 1][0].strip() and not _ADORNMENT.match(below) \
                    and _ADORNMENT.match(rows[index + 2][0]) and rows[index + 2][0].strip()[0] == stripped[0] \
                    and len(rows[index + 2][0].strip()) >= len(below.strip()) and len(stripped) >= len(below.strip()):
                title_block(below, rows[index + 1][1], (stripped[0], True))
                index += 3
                continue
            if _ADORNMENT.match(below) and below.strip() and len(below.strip()) >= len(stripped) and not _ADORNMENT.match(text):
                title_block(text, number, (below.strip()[0], False))
                index += 2
                continue
        paragraph.append((text, number))
        index += 1
        next_is_blank = index >= count or not rows[index][0].strip()
        if next_is_blank and paragraph[-1][0].rstrip().endswith("::"):
            margin = _indent(paragraph[0][0])
            flush()
            start = index
            while start < count and not rows[start][0].strip():
                start += 1
            end = _body(rows, start, margin)
            literal = _trim(_dedent(rows[start:end])) if end > start and _indent(rows[start][0]) > margin else []
            emit("code", literal)
            index = end if literal else index
    flush()


def parse_rst(text: str) -> Extraction:
    lines = normalize_newlines(text).split("\n")
    check_lines(lines)
    return parse_rst_rows([(line, number) for number, line in enumerate(lines, start=1)])


def parse_rst_rows(rows: list[tuple[str, int]], fmt: str = "rst") -> Extraction:
    extraction = Extraction(fmt)
    parse_rows([(text.expandtabs(8).rstrip(), number) for text, number in rows], extraction, {})
    extraction.title = next((b.title for b in extraction.blocks if b.kind == "heading"), None)
    return extraction

