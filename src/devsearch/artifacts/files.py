"""Shared file primitives: strict JSONL reading, canonical JSONL writing, atomic writes.

Every persisted artifact goes through these so that format rules (UTF-8, LF only, no
blank lines, no duplicate keys, no NaN) and crash behavior (temporary file on the
destination filesystem, flush + fsync, then an atomic rename) are defined once.
"""

import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import canonical_json, loads_strict, read_utf8
from devsearch.artifacts.schema import as_object
from devsearch.errors import ManifestError


def read_jsonl(path: Path, where: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a JSONL file into (label, object) pairs; labels name the file and line."""
    text = read_utf8(path, where=where)
    if "\r" in text:
        raise ManifestError(f"{where}: carriage returns are not allowed; use LF line endings")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    records = []
    for number, line in enumerate(lines, start=1):
        label = f"{where} line {number}"
        if not line.strip():
            raise ManifestError(f"{label}: blank lines are not allowed")
        records.append((label, dict(as_object(loads_strict(line, where=label), label))))
    return records


def jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Canonical JSONL: one compact sorted-key object per line, LF terminated."""
    return "".join(canonical_json(record) + "\n" for record in records).encode("utf-8")


def write_atomic(path: Path, data: bytes, *, exclusive: bool = False) -> None:
    """Write via a same-directory temporary file, fsync, then rename into place.

    `exclusive=True` never replaces an existing file (hard-link publication fails with
    FileExistsError instead); otherwise an existing file is atomically replaced. The
    parent directory must already exist: nothing here creates directories.
    """
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
