"""Strict configuration for `prepare` (docs/02 §3, docs/04): pinned sources, extraction policy and the token budget.

    schema_version = 1
    parser_version  = "static-extract-v1"     recorded on every document; a changed reader is a new parser version
    chunker_version = "chunker-v1"           part of every chunk id; a changed chunker is a new chunker version

    [policy]    max_pair_tokens = 256  max_query_tokens = 64  min_chunk_tokens = 8  overlong_code = "split"
                max_file_bytes = 2000000  max_files = 20000
    [tokenizer] assets = "model-assets"      the PINNED tokenizer that decides what fits; its fingerprint is reported

    [[source]]
    source_id    = "pytorch-tutorials"       the same id appears in search filters
    revision     = "<40-hex commit>"         exact; a branch name is not a revision
    snapshot     = "snapshots/pytorch-tutorials"          existing checkout (offline) or the place to fetch into
    git_url      = "https://github.com/pytorch/tutorials" optional, for acquisition only
    url_template = "https://github.com/pytorch/tutorials/blob/{revision}/{path}"   source links shown with results
    license      = "BSD-3-Clause"            declared SPDX identifier (recorded, not verified against the upstream text)
    license_file = "LICENSE"                 relative to the snapshot; its hash is recorded
    include      = ["**/*.rst", "**/*.py"]   glob patterns over snapshot-relative paths
    exclude      = ["**/_static/**"]
    snapshot_sha256 = "<64-hex>"             optional content pin for a snapshot that has no git metadata

Relative paths resolve against the config file. Unknown keys fail. The configuration never contains credentials.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from devsearch.artifacts.tomlconfig import (
    as_int, as_string, as_string_list, as_table, check_keys, check_version, load_toml,
)
from devsearch.errors import ConfigError

FORMATS = {".md": "markdown", ".mdx": "mdx", ".rst": "rst", ".ipynb": "notebook", ".py": "python-gallery"}
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PLACEHOLDERS = {"{revision}", "{path}"}


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    revision: str
    snapshot: Path
    git_url: str | None
    url_template: str | None
    license: str
    license_file: str | None
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    snapshot_sha256: str | None


@dataclass(frozen=True)
class PrepareConfig:
    parser_version: str
    chunker_version: str
    max_pair_tokens: int
    max_query_tokens: int
    min_chunk_tokens: int
    overlong_code: str
    max_file_bytes: int
    max_files: int
    tokenizer_assets: Path
    sources: tuple[SourceSpec, ...]


def load_prepare_config(path: Path) -> PrepareConfig:
    path = Path(path).resolve()
    where = str(path)
    table = load_toml(path)
    check_keys(table, ("schema_version", "parser_version", "chunker_version", "policy", "tokenizer", "source"), (), where)
    check_version(table, where)
    policy = as_table(table["policy"], f"{where}: [policy]")
    check_keys(policy, ("max_pair_tokens", "max_query_tokens", "min_chunk_tokens", "overlong_code", "max_file_bytes", "max_files"), (), f"{where}: [policy]")
    tokenizer = as_table(table["tokenizer"], f"{where}: [tokenizer]")
    check_keys(tokenizer, ("assets",), (), f"{where}: [tokenizer]")
    overlong = as_string(policy["overlong_code"], f"{where}: policy.overlong_code", max_length=16)
    if overlong not in ("split", "exclude"):
        raise ConfigError(f"{where}: policy.overlong_code must be 'split' or 'exclude'")
    raw_sources = table["source"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ConfigError(f"{where}: at least one [[source]] is required")
    sources, seen = [], set()
    for number, raw in enumerate(raw_sources, start=1):
        label = f"{where}: source #{number}"
        spec = as_table(raw, label)
        check_keys(spec, ("source_id", "revision", "snapshot", "license", "include"), ("git_url", "url_template", "license_file", "exclude", "snapshot_sha256"), label)
        source_id = as_string(spec["source_id"], f"{label}.source_id", max_length=64)
        if not _SOURCE_ID.match(source_id) or source_id in seen:
            raise ConfigError(f"{label}: source_id {source_id!r} must be a unique identifier of letters, digits, '.', '_' or '-'")
        seen.add(source_id)
        revision = as_string(spec["revision"], f"{label}.revision", max_length=40)
        if not _REVISION.match(revision):
            raise ConfigError(f"{label}: revision must be a full 40-hex commit id (a branch or tag is not a pinned revision)")
        template = spec.get("url_template")
        if template is not None:
            template = as_string(template, f"{label}.url_template", max_length=2048)
            if not template.startswith("https://") or {m for m in re.findall(r"\{[^{}]*\}", template)} - _PLACEHOLDERS or "@" in template.split("/")[2]:
                raise ConfigError(f"{label}: url_template must be a credential-free https URL using only {{revision}} and {{path}}")
        git_url = spec.get("git_url")
        if git_url is not None:
            git_url = as_string(git_url, f"{label}.git_url", max_length=2048)
            if not git_url.startswith(("https://", "file://")) or "@" in git_url.split("/")[2 if git_url.startswith("https://") else 0]:
                raise ConfigError(f"{label}: git_url must be a credential-free https:// (or file://) URL")
        include = as_string_list(spec["include"], f"{label}.include", unique=True)
        if not include:
            raise ConfigError(f"{label}: include needs at least one pattern")
        digest = spec.get("snapshot_sha256")
        if digest is not None and not re.match(r"^[0-9a-f]{64}$", as_string(digest, f"{label}.snapshot_sha256", max_length=64)):
            raise ConfigError(f"{label}: snapshot_sha256 must be 64 hex characters")
        license_file = spec.get("license_file")
        sources.append(SourceSpec(
            source_id, revision, (path.parent / as_string(spec["snapshot"], f"{label}.snapshot")).resolve(), git_url, template,
            as_string(spec["license"], f"{label}.license", max_length=200), None if license_file is None else as_string(license_file, f"{label}.license_file"),
            include, as_string_list(spec.get("exclude", []), f"{label}.exclude", unique=True), digest))
    return PrepareConfig(
        as_string(table["parser_version"], f"{where}: parser_version", max_length=64), as_string(table["chunker_version"], f"{where}: chunker_version", max_length=64),
        as_int(policy["max_pair_tokens"], f"{where}: policy.max_pair_tokens", 16, 4096), as_int(policy["max_query_tokens"], f"{where}: policy.max_query_tokens", 8, 4096),
        as_int(policy["min_chunk_tokens"], f"{where}: policy.min_chunk_tokens", 1, 1000), overlong,
        as_int(policy["max_file_bytes"], f"{where}: policy.max_file_bytes", 1024, 50_000_000), as_int(policy["max_files"], f"{where}: policy.max_files", 1, 1_000_000),
        (path.parent / as_string(tokenizer["assets"], f"{where}: tokenizer.assets")).resolve(), tuple(sources))
