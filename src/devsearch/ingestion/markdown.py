"""Markdown and MDX, extracted statically (docs/02 §3).

Supported: ATX and setext headings, fenced code blocks (``` and ~~~), paragraphs, lists, tables and block quotes (kept as
text blocks of their original lines), YAML front matter (dropped; its `title:` may name the document), HTML comments
(dropped) and, for MDX, `import`/`export` statements and lines that are only a JSX/HTML tag (dropped). `[[autodoc]]`
API-stub directives are dropped with their list of members. Every kept line is verbatim, which is what lets notebook cells use
exact cell-span mappings. An unterminated fence or comment quarantines the file.
"""

import re

from devsearch.ingestion.model import Block, CellSpan, Extraction, Line, ParseFailure, check_lines, clean_title, normalize_newlines

_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})\s*([^`]*)$")
_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_TAG_ONLY = re.compile(r"^\s*</?[A-Za-z][\w.:-]*(\s[^<>]*)?/?>\s*$")
_MDX_STATEMENT = re.compile(r"^(import|export)\s")
_AUTODOC = re.compile(r"^\[\[autodoc\]\]")
_TITLE_KEY = re.compile(r"^title:\s*(.+?)\s*$")
_ANCHOR = re.compile(r"\s*\{#[\w.:-]+\}\s*$")


def _clean_heading(text: str) -> str:
    return clean_title(_ANCHOR.sub("", text).replace("`", ""))


def parse_lines(raw_lines: list[str], *, mdx: bool = False, first_line: int = 1, cell_id: str | None = None) -> Extraction:
    """Parse already-split lines. With `cell_id`, origins are verbatim cell spans (the lines are the whole cell source)."""
    extraction = Extraction("mdx" if mdx else "markdown")
    check_lines(raw_lines)
    offsets, position = [], 0
    for line in raw_lines:
        offsets.append(position)
        position += len(line) + 1

    def origin(index: int):
        if cell_id is None:
            return first_line + index
        return CellSpan(cell_id, offsets[index], offsets[index] + len(raw_lines[index]))

    def add(kind: str, indexes: list[int], **extra) -> None:
        extraction.blocks.append(Block(kind, tuple(Line(raw_lines[i], origin(i)) for i in indexes), **extra))

    index, count = 0, len(raw_lines)
    if raw_lines and raw_lines[0].strip() == "---" and cell_id is None:           # YAML front matter (files only)
        end = next((i for i in range(1, count) if raw_lines[i].strip() in ("---", "...")), None)
        if end is None:
            raise ParseFailure("front matter opened at line 1 is never closed")
        for i in range(1, end):
            match = _TITLE_KEY.match(raw_lines[i])
            if match and extraction.title is None:
                extraction.title = match.group(1).strip("'\"")
        extraction.drop("front-matter", end + 1)
        index = end + 1
    paragraph: list[int] = []

    def flush() -> None:
        if paragraph:
            add("text", paragraph)
            paragraph.clear()

    while index < count:
        line = raw_lines[index]
        fence = _FENCE.match(line)
        if fence:
            flush()
            marker, opened = fence.group(2), index
            body = []
            index += 1
            while index < count:
                closing = _FENCE.match(raw_lines[index])
                if closing and closing.group(2)[0] == marker[0] and len(closing.group(2)) >= len(marker) and not closing.group(3).strip():
                    break
                body.append(index)
                index += 1
            else:
                raise ParseFailure(f"the fenced code block opened at line {first_line + opened} is never closed")
            if any(raw_lines[i].strip() for i in body):
                add("code", body)
            extraction.drop("code-fence-marker", 2)
            index += 1
            continue
        if line.lstrip().startswith("<!--"):
            flush()
            end = index
            while end < count and "-->" not in raw_lines[end]:
                end += 1
            if end == count:
                raise ParseFailure(f"the HTML comment opened at line {first_line + index} is never closed")
            extraction.drop("html-comment", end - index + 1)
            index = end + 1
            continue
        if _AUTODOC.match(line):
            flush()
            end = index + 1
            while end < count and raw_lines[end].strip():
                end += 1
            extraction.drop("autodoc-directive", end - index)
            index = end
            continue
        if mdx and (_MDX_STATEMENT.match(line) or _TAG_ONLY.match(line)):
            flush()
            extraction.drop("mdx-statement" if _MDX_STATEMENT.match(line) else "jsx-tag-line")
            index += 1
            continue
        atx = _ATX.match(line)
        if atx:
            flush()
            title = _clean_heading(atx.group(2))
            if title:
                add("heading", [index], level=len(atx.group(1)), title=title)
            else:
                extraction.drop("empty-heading")
            index += 1
            continue
        if not line.strip():
            flush()
            index += 1
            continue
        setext = _SETEXT.match(raw_lines[index + 1]) if index + 1 < count else None
        if setext and not paragraph and not _FENCE.match(line) and not line.startswith(("- ", "* ", "+ ", ">", "|")):
            add("heading", [index, index + 1], level=1 if setext.group(1)[0] == "=" else 2, title=_clean_heading(line))
            index += 2
            continue
        paragraph.append(index)
        index += 1
    flush()
    if extraction.title is None:
        extraction.title = next((b.title for b in extraction.blocks if b.kind == "heading" and b.level == 1), None)
    return extraction


def parse_markdown(text: str, *, mdx: bool = False) -> Extraction:
    return parse_lines(normalize_newlines(text).split("\n"), mdx=mdx)
