"""Strict TOML reading shared by every configuration loader.

Unknown or missing keys, wrong types (a bool is not an int), non-finite numbers and out-of-range values
raise `ConfigError` with the file and key named. Values are never defaulted silently: what a run used is
exactly what the file said, so the resolved settings can be recorded and hashed.
"""

import math
import tomllib
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from devsearch.errors import ConfigError

CONFIG_SCHEMA_VERSION = 1


def load_toml(path: Path) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{path}: invalid TOML: {error}") from error


def check_keys(table: Mapping[str, Any], required: Collection[str], optional: Collection[str], where: str) -> None:
    missing = sorted(set(required) - table.keys())
    unknown = sorted(table.keys() - set(required) - set(optional))
    if missing:
        raise ConfigError(f"{where}: missing required key(s): {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(unknown)}")


def as_int(value: Any, where: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: expected an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f"{where}: out of range: {value}")
    return value


def as_number(value: Any, where: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ConfigError(f"{where}: must be finite and between {minimum} and {maximum}")
    return number


def as_choice(value: Any, where: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ConfigError(f"{where}: must be one of {', '.join(choices)}; got {value!r}")
    return value


def as_table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a table")
    return value


def as_string(value: Any, where: str, *, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length or not value.isprintable():
        raise ConfigError(f"{where}: expected a non-empty printable string of at most {max_length} characters")
    return value


def as_boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where}: expected true or false")
    return value


def as_string_list(value: Any, where: str, *, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{where}: expected a list of non-empty strings")
    if unique and len(set(value)) != len(value):
        raise ConfigError(f"{where}: contains duplicates")
    return tuple(value)


def as_int_list(value: Any, where: str, minimum: int, maximum: int | None = None, *, unique: bool = True) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where}: expected a non-empty list of integers")
    items = tuple(as_int(item, f"{where}[{index}]", minimum, maximum) for index, item in enumerate(value))
    if unique and len(set(items)) != len(items):
        raise ConfigError(f"{where}: contains duplicates")
    return items


def check_version(table: Mapping[str, Any], where: str) -> None:
    if table["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"{where}: unsupported schema_version {table['schema_version']!r}")
