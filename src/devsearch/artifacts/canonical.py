"""Canonical JSON, SHA-256 and manifest hashing (docs/03 §3, docs/06 §2).

A manifest's own hash covers its canonical JSON without the hash field itself.
File bytes carry a separate SHA-256. Loaders reject duplicate keys and NaN/Infinity.
"""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devsearch.errors import ManifestError

MANIFEST_HASH_FIELD = "manifest_sha256"


def canonical_json(value: Any) -> str:
    """Stable key order, compact separators, UTF-8 characters, no NaN/Infinity."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_of(value: Any) -> str:
    """SHA-256 of a value's canonical JSON."""
    return sha256_text(canonical_json(value))


def pretty_json(value: Any) -> str:
    """Human-diffable serialization for files; hashing always uses canonical_json."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not allowed")


def loads_strict(text: str, *, where: str) -> Any:
    """Parse JSON, rejecting duplicate object keys and NaN/Infinity."""
    try:
        return json.loads(
            text, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant
        )
    except ValueError as error:  # includes JSONDecodeError
        raise ManifestError(f"{where}: invalid JSON: {error}") from error


def read_utf8(path: Path, *, where: str) -> str:
    """Read exact UTF-8 text with no newline translation."""
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManifestError(f"{where}: not valid UTF-8: {error}") from error
    except OSError as error:
        raise ManifestError(f"{where}: cannot read file: {error}") from error


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of the payload with its manifest hash added."""
    body = {key: value for key, value in payload.items() if key != MANIFEST_HASH_FIELD}
    return {**body, MANIFEST_HASH_FIELD: digest_of(body)}


def verify_manifest_hash(manifest: Mapping[str, Any], *, where: str) -> None:
    claimed = manifest.get(MANIFEST_HASH_FIELD)
    body = {key: value for key, value in manifest.items() if key != MANIFEST_HASH_FIELD}
    if claimed != digest_of(body):
        raise ManifestError(f"{where}: {MANIFEST_HASH_FIELD} does not match manifest content")
