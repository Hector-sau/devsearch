"""Strict TOML configuration for the baseline `index` and `retrieve` commands.

Unknown or missing keys, wrong types (a bool is not an int), non-finite numbers and
out-of-range values raise ConfigError. Examples live in `configs/`. Values are
explicit, not defaulted, so the resolved settings are always recorded.
"""

from dataclasses import dataclass
from pathlib import Path

from devsearch.artifacts.tomlconfig import (
    as_choice, as_int, as_number, as_table, check_keys, check_version, load_toml,
)
from devsearch.contracts import SearchMode, SearchRequest
from devsearch.errors import ConfigError

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


def load_index_config(path: Path) -> IndexConfig:
    where = str(path)
    table = load_toml(path)
    check_keys(table, ("schema_version", "tokenizer", "bm25"), (), where)
    check_version(table, where)
    bm25 = as_table(table["bm25"], f"{where}: [bm25]")
    check_keys(bm25, ("idf", "k1", "b"), (), f"{where}: [bm25]")
    return IndexConfig(
        tokenizer=as_choice(table["tokenizer"], f"{where}: tokenizer", TOKENIZER_NAMES),
        idf=as_choice(bm25["idf"], f"{where}: bm25.idf", IDF_VARIANTS),
        k1=as_number(bm25["k1"], f"{where}: bm25.k1", 0.0, 10.0),
        b=as_number(bm25["b"], f"{where}: bm25.b", 0.0, 1.0),
    )


def load_search_config(path: Path) -> SearchConfig:
    where = str(path)
    table = load_toml(path)
    check_keys(table, ("schema_version", "candidate_k", "top_k", "max_candidate_k", "max_query_chars"), ("filters",), where)
    check_version(table, where)
    max_candidate_k = as_int(table["max_candidate_k"], f"{where}: max_candidate_k", 1)
    candidate_k = as_int(table["candidate_k"], f"{where}: candidate_k", 1, max_candidate_k)
    top_k = as_int(table["top_k"], f"{where}: top_k", 1, candidate_k)
    max_query_chars = as_int(table["max_query_chars"], f"{where}: max_query_chars", 1, 100_000)

    filters = as_table(table.get("filters", {}), f"{where}: [filters]")
    check_keys(filters, (), ("sources", "versions"), f"{where}: [filters]")
    sources = filters.get("sources", [])
    if not isinstance(sources, list) or not all(isinstance(s, str) and s for s in sources):
        raise ConfigError(f"{where}: filters.sources must be a list of non-empty strings")
    if len(set(sources)) != len(sources):
        raise ConfigError(f"{where}: filters.sources contains duplicates")
    versions = as_table(filters.get("versions", {}), f"{where}: [filters.versions]")
    if not all(isinstance(r, str) and r for r in versions.values()):
        raise ConfigError(f"{where}: filters.versions values must be non-empty strings")
    return SearchConfig(
        candidate_k=candidate_k, top_k=top_k, max_candidate_k=max_candidate_k,
        max_query_chars=max_query_chars, sources=tuple(sources),
        versions=tuple(sorted(versions.items())),
    )
