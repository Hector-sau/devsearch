"""Jupyter notebook JSON (nbformat 4), read as data (docs/02 §3). Cells are never executed and outputs are never indexed.

Markdown cells go through the Markdown reader and code cells become code blocks; raw cells are dropped. Because the lines of a
cell are kept verbatim, every normalized line maps to an exact span of its cell (cell form of the source map). Cell ids come from
the notebook when they are valid and unique, otherwise `cell-<index>`.
"""

import json
import re

from devsearch.ingestion.markdown import parse_lines
from devsearch.ingestion.model import Block, CellSpan, Extraction, Line, ParseFailure, check_lines, normalize_newlines

_CELL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_CELLS = 5000


def _reject_constant(name: str):
    raise ParseFailure(f"the notebook JSON contains the non-standard constant {name}")


def parse_notebook(text: str) -> Extraction:
    try:
        document = json.loads(normalize_newlines(text), parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        raise ParseFailure(f"not valid JSON: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("cells"), list):
        raise ParseFailure("not a notebook: no `cells` list")
    if not isinstance(document.get("nbformat"), int) or document["nbformat"] < 4:
        raise ParseFailure(f"unsupported notebook format {document.get('nbformat')!r}; only nbformat 4 is read")
    cells = document["cells"]
    if len(cells) > MAX_CELLS:
        raise ParseFailure(f"{len(cells)} cells is more than the {MAX_CELLS}-cell limit")
    extraction = Extraction("notebook")
    seen: set[str] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") not in ("markdown", "code", "raw"):
            raise ParseFailure(f"cell {index} has no known cell_type")
        source = cell.get("source", "")
        if isinstance(source, list) and all(isinstance(part, str) for part in source):
            source = "".join(source)
        if not isinstance(source, str):
            raise ParseFailure(f"cell {index} has a source that is neither text nor a list of text")
        wanted = cell.get("id")
        cell_id = wanted if isinstance(wanted, str) and _CELL_ID.match(wanted) and wanted not in seen else f"cell-{index}"
        if cell_id in seen:
            cell_id = f"cell-{index}-{len(seen)}"
        seen.add(cell_id)
        if cell["cell_type"] == "code":
            extraction.notes["code-cell-outputs-ignored"] += len(cell.get("outputs") or [])
        if cell["cell_type"] == "raw":
            extraction.drop("raw-cell", source.count("\n") + 1 if source.strip() else 0)
            continue
        lines = normalize_newlines(source).split("\n")
        check_lines(lines)
        if not any(line.strip() for line in lines):
            continue
        if cell["cell_type"] == "markdown":
            part = parse_lines(lines, cell_id=cell_id)
            extraction.blocks.extend(part.blocks)
            extraction.dropped.update(part.dropped)
            if extraction.title is None:
                extraction.title = part.title
        else:
            position, items = 0, []
            for line in lines:
                items.append(Line(line, CellSpan(cell_id, position, position + len(line))))
                position += len(line) + 1
            while items and not items[-1].text.strip():
                items.pop()
            extraction.blocks.append(Block("code", tuple(items)))
    return extraction
