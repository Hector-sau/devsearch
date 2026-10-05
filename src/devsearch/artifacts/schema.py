"""Strict field validators for persisted records (unknown fields, wrong types, bad IDs).

Each validator takes the raw JSON value plus a `where` label for error messages and
returns the validated value or raises ManifestError.
"""

import re
from collections.abc import Collection, Mapping
from typing import Any
from urllib.parse import urlsplit

from devsearch.errors import ManifestError

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def as_object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where}: expected a JSON object")
    return value


def exact_keys(record: Mapping[str, Any], keys: Collection[str], where: str) -> None:
    """Require exactly these keys: missing and unrecognized fields both fail."""
    missing = sorted(set(keys) - record.keys())
    unknown = sorted(record.keys() - set(keys))
    if missing:
        raise ManifestError(f"{where}: missing required field(s): {', '.join(missing)}")
    if unknown:
        raise ManifestError(f"{where}: unrecognized field(s): {', '.join(unknown)}")


def as_str(
    value: Any, where: str, *, max_length: int = 2000, allow_empty: bool = False,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{where}: expected a string")
    if not value and not allow_empty:
        raise ManifestError(f"{where}: must not be empty")
    if len(value) > max_length:
        raise ManifestError(f"{where}: longer than {max_length} characters")
    if pattern is not None and not pattern.fullmatch(value):
        raise ManifestError(f"{where}: unexpected format: {value!r}")
    return value


def as_id(value: Any, where: str) -> str:
    return as_str(value, where, pattern=_ID)


def as_sha256(value: Any, where: str) -> str:
    return as_str(value, where, pattern=_SHA256)


def as_label(value: Any, where: str, *, max_length: int = 200) -> str:
    """Non-empty descriptive text without control characters (titles, versions, licenses)."""
    text = as_str(value, where, max_length=max_length)
    if not text.isprintable() or text != text.strip():
        raise ManifestError(f"{where}: contains control/line-break characters or surrounding whitespace")
    return text


def as_int(value: Any, where: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{where}: expected an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ManifestError(f"{where}: out of range: {value}")
    return value


def as_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{where}: expected true or false")
    return value


def as_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{where}: expected a list")
    return value


def as_https_url(value: Any, where: str) -> str:
    """HTTPS only, with a host and no credentials. Rejects script/data/file schemes."""
    text = as_str(value, where, max_length=2048)
    if _FORBIDDEN_TEXT.search(text) or any(ch.isspace() for ch in text):
        raise ManifestError(f"{where}: URL contains whitespace or control characters")
    try:
        parts = urlsplit(text)
        host = parts.hostname
    except ValueError as error:
        raise ManifestError(f"{where}: malformed URL: {error}") from error
    if parts.scheme != "https" or not host or "@" in parts.netloc:
        raise ManifestError(f"{where}: must be an https URL with a host and no credentials")
    return text
