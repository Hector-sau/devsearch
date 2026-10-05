"""Strict TOML configuration for the baseline `index` and `retrieve` commands.

Unknown or missing keys, wrong types (a bool is not an int), non-finite numbers and
out-of-range values raise ConfigError. Examples live in `configs/`. Values are
explicit, not defaulted, so the resolved settings are always recorded.
"""

import math
import tomllib
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devsearch.contracts import SearchMode, SearchRequest
from devsearch.errors import ConfigError

CONFIG_SCHEMA_VERSION = 1
TOKENIZER_NAMES = ("lexical-v1",)
IDF_VARIANTS = ("lucene", "robertson")


@dataclass(frozen=True)
class IndexConfig:
    tokenizer: str
    idf: str
    k1: float
    b: float


@dataclass(frozen=True)
class SearchConfig:
    candidate_k: int
    top_k: int
    max_candidate_k: int
    max_query_chars: int
    sources: tuple[str, ...]
    versions: tuple[tuple[str, str], ...]

    def to_request(self, query: str, mode: SearchMode = "baseline") -> SearchRequest:
        return SearchRequest(
            query=query, sources=self.sources, versions=self.versions,
            candidate_k=self.candidate_k, top_k=self.top_k, mode=mode,
        )


def _load(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read config {path}: {error}") from error
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{path}: invalid TOML: {error}") from error


def _keys(table: Mapping[str, Any], required: Collection[str], optional: Collection[str], where: str) -> None:
    missing = sorted(set(required) - table.keys())
    unknown = sorted(table.keys() - set(required) - set(optional))
    if missing:
        raise ConfigError(f"{where}: missing required key(s): {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(unknown)}")


def _int(value: Any, where: str, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}: expected an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f"{where}: out of range: {value}")
    return value


def _number(value: Any, where: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: expected a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ConfigError(f"{where}: must be finite and between {minimum} and {maximum}")
    return number


def _choice(value: Any, where: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise ConfigError(f"{where}: must be one of {', '.join(choices)}; got {value!r}")
    return value


def _table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a table")
    return value


def _version(table: Mapping[str, Any], where: str) -> None:
    if table["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"{where}: unsupported schema_version {table['schema_version']!r}")


def load_index_config(path: Path) -> IndexConfig:
    where = str(path)
    table = _load(path)
    _keys(table, ("schema_version", "tokenizer", "bm25"), (), where)
    _version(table, where)
    bm25 = _table(table["bm25"], f"{where}: [bm25]")
    _keys(bm25, ("idf", "k1", "b"), (), f"{where}: [bm25]")
    return IndexConfig(
        tokenizer=_choice(table["tokenizer"], f"{where}: tokenizer", TOKENIZER_NAMES),
        idf=_choice(bm25["idf"], f"{where}: bm25.idf", IDF_VARIANTS),
        k1=_number(bm25["k1"], f"{where}: bm25.k1", 0.0, 10.0),
        b=_number(bm25["b"], f"{where}: bm25.b", 0.0, 1.0),
    )


def load_search_config(path: Path) -> SearchConfig:
    where = str(path)
    table = _load(path)
    _keys(table, ("schema_version", "candidate_k", "top_k", "max_candidate_k", "max_query_chars"), ("filters",), where)
    _version(table, where)
    max_candidate_k = _int(table["max_candidate_k"], f"{where}: max_candidate_k", 1)
    candidate_k = _int(table["candidate_k"], f"{where}: candidate_k", 1, max_candidate_k)
    top_k = _int(table["top_k"], f"{where}: top_k", 1, candidate_k)
    max_query_chars = _int(table["max_query_chars"], f"{where}: max_query_chars", 1, 100_000)

    filters = _table(table.get("filters", {}), f"{where}: [filters]")
    _keys(filters, (), ("sources", "versions"), f"{where}: [filters]")
    sources = filters.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(s, str) and s for s in sources):
        raise ConfigError(f"{where}: filters.sources must be a list of non-empty strings")
    if len(set(sources)) != len(sources):
        raise ConfigError(f"{where}: filters.sources contains duplicates")
    versions = _table(filters.get("versions", {}), f"{where}: [filters.versions]")
    if not all(isinstance(r, str) and r for r in versions.values()):
        raise ConfigError(f"{where}: filters.versions values must be non-empty strings")
    return SearchConfig(
        candidate_k=candidate_k, top_k=top_k, max_candidate_k=max_candidate_k,
        max_query_chars=max_query_chars, sources=tuple(sources),
        versions=tuple(sorted(versions.items())),
    )
