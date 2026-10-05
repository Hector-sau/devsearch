"""Source maps: normalized-text spans -> original file lines or notebook cells.

Schema v1 (docs/02 §2). Offsets are Unicode code-point indexes into the saved
normalized text; spans are half-open `[start_char, end_char)`. Entries are sorted
and non-overlapping. Two entry forms:

* line form: `path`, `start_line`, `end_line` (1-based, inclusive). A lookup reports
  a whole entry's lines, so producers emit one entry per source line or block.
* cell form: `path`, `cell_id`, `cell_start_char`, `cell_end_char`, mapping the span
  one-to-one onto cell-local offsets (equal lengths), so partial overlaps are exact.

`synthetic: true` marks generated text (for example an added heading). It is never
reported as a source location, and a chunk covering only synthetic text has no
evidence trace and is rejected by corpus validation.
"""

from dataclasses import dataclass
from typing import Any

from devsearch.artifacts.schema import as_bool, as_id, as_int, as_label, as_list, as_object, exact_keys
from devsearch.errors import ManifestError

SOURCE_MAP_SCHEMA_VERSION = 1

_COMMON = ("start_char", "end_char", "path", "synthetic")
_LINE_FIELDS = (*_COMMON, "start_line", "end_line")
_CELL_FIELDS = (*_COMMON, "cell_id", "cell_start_char", "cell_end_char")


@dataclass(frozen=True)
class SourceSegment:
    start_char: int
    end_char: int
    path: str
    synthetic: bool = False
    start_line: int | None = None
    end_line: int | None = None
    cell_id: str | None = None
    cell_start_char: int | None = None
    cell_end_char: int | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "start_char": self.start_char, "end_char": self.end_char,
            "path": self.path, "synthetic": self.synthetic,
        }
        if self.cell_id is None:
            record.update(start_line=self.start_line, end_line=self.end_line)
        else:
            record.update(
                cell_id=self.cell_id, cell_start_char=self.cell_start_char,
                cell_end_char=self.cell_end_char,
            )
        return record


@dataclass(frozen=True)
class SourceMap:
    doc_id: str
    revision: str
    segments: tuple[SourceSegment, ...]

    def locate(self, start_char: int, end_char: int) -> list[dict[str, Any]]:
        """Original locations of the non-synthetic text overlapping `[start, end)`."""
        lines: dict[str, list[int]] = {}
        cells: dict[tuple[str, str], list[int]] = {}
        for segment in self.segments:
            if segment.synthetic or segment.end_char <= start_char or segment.start_char >= end_char:
                continue
            if segment.cell_id is None:
                low, high = lines.setdefault(segment.path, [segment.start_line, segment.end_line])
                lines[segment.path] = [min(low, segment.start_line), max(high, segment.end_line)]
            else:
                skip = max(start_char - segment.start_char, 0)
                take = min(end_char, segment.end_char) - segment.start_char
                low, high = segment.cell_start_char + skip, segment.cell_start_char + take
                key = (segment.path, segment.cell_id)
                old = cells.setdefault(key, [low, high])
                cells[key] = [min(old[0], low), max(old[1], high)]
        located: list[dict[str, Any]] = [
            {"path": path, "start_line": span[0], "end_line": span[1]}
            for path, span in lines.items()
        ]
        located.extend(
            {"path": path, "cell_id": cell, "start_char": span[0], "end_char": span[1]}
            for (path, cell), span in cells.items()
        )
        return located


def parse_source_map(
    value: Any, *, doc_id: str, revision: str, text_length: int, where: str
) -> SourceMap:
    record = as_object(value, where)
    exact_keys(record, ("schema_version", "doc_id", "revision", "entries"), where)
    if record["schema_version"] != SOURCE_MAP_SCHEMA_VERSION:
        raise ManifestError(f"{where}: unsupported source-map schema_version {record['schema_version']!r}")
    if record["doc_id"] != doc_id or record["revision"] != revision:
        raise ManifestError(f"{where}: source map belongs to a different (doc_id, revision)")
    segments: list[SourceSegment] = []
    previous_end = 0
    for index, entry_value in enumerate(as_list(record["entries"], f"{where}.entries")):
        label = f"{where}.entries[{index}]"
        entry = as_object(entry_value, label)
        is_cell = "cell_id" in entry
        exact_keys(entry, _CELL_FIELDS if is_cell else _LINE_FIELDS, label)
        start = as_int(entry["start_char"], f"{label}.start_char")
        end = as_int(entry["end_char"], f"{label}.end_char", minimum=start + 1)
        if end > text_length:
            raise ManifestError(f"{label}: ends at {end}, beyond the {text_length}-character text")
        if start < previous_end:
            raise ManifestError(f"{label}: entries must be sorted and must not overlap")
        previous_end = end
        path = as_label(entry["path"], f"{label}.path", max_length=1024)
        synthetic = as_bool(entry["synthetic"], f"{label}.synthetic")
        if is_cell:
            cell_start = as_int(entry["cell_start_char"], f"{label}.cell_start_char")
            cell_end = as_int(entry["cell_end_char"], f"{label}.cell_end_char", minimum=cell_start + 1)
            if cell_end - cell_start != end - start:
                raise ManifestError(f"{label}: cell span must be the same length as the text span")
            segments.append(SourceSegment(
                start, end, path, synthetic, cell_id=as_id(entry["cell_id"], f"{label}.cell_id"),
                cell_start_char=cell_start, cell_end_char=cell_end,
            ))
        else:
            first = as_int(entry["start_line"], f"{label}.start_line", minimum=1)
            last = as_int(entry["end_line"], f"{label}.end_line", minimum=first)
            segments.append(SourceSegment(start, end, path, synthetic, start_line=first, end_line=last))
    return SourceMap(doc_id, revision, tuple(segments))
