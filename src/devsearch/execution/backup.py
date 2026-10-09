"""Backup stage: copy the evidence and the release to a fixed destination and prove the copy by checksum (docs/10 §6).

The backup is a separate, explicit step with one configured destination; it is not a reason to open arbitrary
network access. It copies into a staging directory beside the destination, re-reads every copied file and compares
SHA-256 against the source, writes a sealed `backup-receipt.json`, and only then renames the staging directory into
place. A crash or a full disk therefore leaves no backup that looks complete, and an existing backup is never
overwritten (an identical receipt is a no-op; a different one is refused). The receipt records relative names, sizes
and hashes only; it never contains host paths or credentials.
"""

import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import pretty_json, seal_manifest, sha256_file
from devsearch.errors import ManifestError
from devsearch.execution.records import SCHEMA_VERSION, read_sealed

RECEIPT_NAME = "backup-receipt.json"
RECEIPT_FIELDS = ("plan_id", "files", "sources", "totals")
_CHUNK = 1024 * 1024


def _copy_verified(source: Path, target: Path) -> dict[str, Any]:
    digest_source = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        while chunk := reader.read(_CHUNK):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    if sha256_file(target) != digest_source:
        raise ManifestError(f"the copy of {source.name} does not match its source checksum")
    return {"sha256": digest_source, "bytes": target.stat().st_size}


def _expand(sources: Mapping[str, Path]) -> list[tuple[str, Path]]:
    expanded = []
    for name, source in sorted(sources.items()):
        if name.startswith("/") or ".." in Path(name).parts or not name:
            raise ManifestError(f"backup name {name!r} must be a plain relative path")
        source = Path(source)
        if source.is_symlink():
            raise ManifestError(f"{source}: symbolic links are not backed up")
        if source.is_file():
            expanded.append((name, source))
        elif source.is_dir():
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    raise ManifestError(f"{path}: symbolic links are not backed up")
                if path.is_file() and not path.name.startswith(".staging-"):
                    expanded.append((f"{name}/{path.relative_to(source).as_posix()}", path))
        else:
            raise ManifestError(f"backup source {source} does not exist")
    names = [n for n, _ in expanded]
    if len(set(names)) != len(names):
        raise ManifestError("two backup sources map to the same name")
    return expanded


def backup(sources: Mapping[str, Path], destination: Path, *, plan_id: str) -> dict[str, Any]:
    """Copy `sources` (name -> file or directory) to `destination` (must not exist unless identical). Returns the receipt."""
    destination = Path(destination)
    if not destination.parent.is_dir():
        raise ManifestError(f"{destination.parent} does not exist; the backup never creates its parent (a missing volume must fail)")
    expanded = _expand(sources)
    if not expanded:
        raise ManifestError("nothing to back up")
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    staging.mkdir()
    try:
        files = {name: _copy_verified(source, staging / name) for name, source in expanded}
        receipt = seal_manifest({
            "schema_version": SCHEMA_VERSION, "kind": "backup-receipt", "plan_id": plan_id, "files": files,
            "sources": sorted(sources), "totals": {"files": len(files), "bytes": sum(f["bytes"] for f in files.values())},
        })
        (staging / RECEIPT_NAME).write_text(pretty_json(receipt), encoding="utf-8")
        verify_backup(staging)
        if destination.exists():
            existing = verify_backup(destination)
            if existing["manifest_sha256"] != receipt["manifest_sha256"]:
                raise ManifestError(f"{destination} already holds a different backup; backups are never overwritten")
            return existing
        os.rename(staging, destination)
        return receipt
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def verify_backup(directory: Path) -> dict[str, Any]:
    """Re-read every file and compare it with the receipt; also refuses unlisted files."""
    directory = Path(directory)
    receipt = read_sealed(directory / RECEIPT_NAME, "backup-receipt", RECEIPT_FIELDS, str(directory / RECEIPT_NAME))
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() and p.name != RECEIPT_NAME}
    if present != set(receipt["files"]):
        raise ManifestError(f"{directory}: files differ from the receipt (extra {sorted(present - set(receipt['files']))[:3]}, "
                            f"missing {sorted(set(receipt['files']) - present)[:3]})")
    for name, ref in receipt["files"].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != ref["bytes"] or sha256_file(path) != ref["sha256"]:
            raise ManifestError(f"{directory}: {name} does not match the receipt checksum")
    return receipt
