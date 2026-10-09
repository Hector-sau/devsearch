"""Shared file primitives: strict JSONL reading, canonical JSONL writing, atomic writes.

Every persisted artifact goes through these so that format rules (UTF-8, LF only, no
blank lines, no duplicate keys, no NaN) and crash behavior (temporary file on the
destination filesystem, flush + fsync, then an atomic rename) are defined once.
"""

import os
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
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


def publish_staged(staging: Path, directory: Path, *, manifests: Sequence[str]) -> bool:
    """Move every entry of `staging` into `directory`; the named manifests are renamed last, in order.

    Until the first manifest lands, readers see no artifact, so a crash leaves only unpublished remnants.
    Remnants of an earlier interrupted publish are moved aside (never deleted) so a retry cannot collide.
    If every manifest is already present with identical bytes the artifact is already published and
    nothing changes (returns False); a manifest with different bytes is refused, because artifacts are
    immutable. Returns True when something was published. `staging` is removed afterwards.
    """
    if not staging.is_dir() or not directory.is_dir():
        raise ManifestError(f"staging and destination directories must both exist ({staging}, {directory})")
    entries = sorted(entry.name for entry in staging.iterdir())
    if not set(manifests) <= set(entries) or not manifests:
        raise ManifestError("every named manifest must be staged")
    present = [name for name in manifests if (directory / name).exists()]
    if present:
        same = len(present) == len(manifests) and all(
            (directory / name).read_bytes() == (staging / name).read_bytes() for name in manifests
        )
        if not same:
            raise ManifestError(f"{directory} already holds a different {', '.join(present)}; artifacts are immutable")
        shutil.rmtree(staging)
        return False
    data = [name for name in entries if name not in manifests]
    leftovers = [name for name in entries if os.path.lexists(directory / name)]
    if leftovers:
        abandoned = directory / f".abandoned-{uuid.uuid4().hex[:12]}"
        abandoned.mkdir()
        for name in leftovers:
            os.rename(directory / name, abandoned / name)
    for name in data:
        os.rename(staging / name, directory / name)
    for name in manifests:
        os.replace(staging / name, directory / name)
    shutil.rmtree(staging)
    return True


def new_staging(directory: Path) -> Path:
    """A private staging directory beside the destination (same filesystem, so renames stay atomic)."""
    staging = directory / f".staging-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    return staging
