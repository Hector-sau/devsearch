"""Versioned BM25 index artifact: build, load and verify (docs/03 §3, docs/04).

`index-manifest.json` (written by `devsearch.index`) records the corpus ID and the
corpus manifest's file hash, tokenizer rules fingerprint, BM25 variant and
parameters, statistics, the postings file and its hash, and the code snapshot that
built it. `index_id` covers corpus, tokenizer, parameters and postings bytes, not
the code snapshot, so rebuilding with identical inputs is idempotent. References
are relative to the manifest and must stay inside the runtime root.
"""

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import (
    canonical_json, digest_of, loads_strict, pretty_json, read_utf8, seal_manifest,
    sha256_file, sha256_hex, verify_manifest_hash,
)
from devsearch.artifacts.identity import code_identity
from devsearch.artifacts.paths import ensure_inside, relative_reference, resolve_reference
from devsearch.artifacts.schema import as_object, as_sha256, as_str, exact_keys
from devsearch.data.corpus import Corpus, load_corpus
from devsearch.errors import ManifestError
from devsearch.retrieval import tokenizer
from devsearch.retrieval.bm25 import (
    SCORE_DECIMALS, Bm25Index, Bm25Retriever, Postings, build_postings, parse_postings,
)
from devsearch.retrieval.config import IDF_VARIANTS, TOKENIZER_NAMES, IndexConfig

INDEX_SCHEMA_VERSION = 1
INDEX_KIND = "index"
_TIE_BREAK = "chunk_id_ascending"

_MANIFEST_FIELDS = (
    "schema_version", "kind", "index_id", "corpus", "tokenizer", "bm25", "stats",
    "postings", "built_with", "manifest_sha256",
)


@dataclass(frozen=True)
class LoadedIndex:
    manifest: dict[str, Any]
    corpus: Corpus
    index: Bm25Index

    @property
    def index_id(self) -> str:
        return self.manifest["index_id"]

    @cached_property
    def _retriever(self) -> Bm25Retriever:
        return Bm25Retriever(self.corpus, self.index)

    def retriever(self) -> Bm25Retriever:
        return self._retriever


def _bm25_record(config: IndexConfig) -> dict[str, Any]:
    return {
        "variant": "okapi", "idf": config.idf, "k1": config.k1, "b": config.b,
        "query_term_weighting": "binary", "score_decimals": SCORE_DECIMALS, "tie_break": _TIE_BREAK,
    }


def _stats_record(index: Bm25Index) -> dict[str, Any]:
    return {
        "chunk_count": index.chunk_count, "total_tokens": index.total_tokens,
        "average_length": index.average_length, "vocabulary_size": index.vocabulary_size,
    }


def _index_id(corpus_id: str, tokenizer_record: dict[str, Any], bm25: dict[str, Any], postings_sha: str) -> str:
    return "index-" + digest_of({
        "corpus_id": corpus_id, "tokenizer": tokenizer_record, "bm25": bm25,
        "postings_sha256": postings_sha,
    })[:32]


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_index(
    config: IndexConfig, corpus_manifest: Path, output_manifest: Path, root: Path
) -> tuple[dict[str, Any], bool]:
    """Build and write an index. Returns (manifest, created); created is False when an
    identical index already exists. Never overwrites a different artifact and never
    creates the output directory."""
    root = root.resolve(strict=True)
    corpus_path = ensure_inside(corpus_manifest, root, what="corpus manifest")
    output = ensure_inside(output_manifest, root, what="index output")
    if not output.parent.is_dir():
        raise ManifestError(f"index output directory does not exist: {output.parent}")
    corpus = load_corpus(corpus_path)
    if not corpus.chunks:
        raise ManifestError("corpus has no chunks; refusing to build an empty index")

    postings = build_postings(corpus.chunks)
    postings_bytes = (canonical_json(postings.to_record()) + "\n").encode("utf-8")
    postings_path = output.with_name(f"{output.stem}.postings.json")
    index = Bm25Index(postings, idf=config.idf, k1=config.k1, b=config.b)
    tokenizer_record = {"name": config.tokenizer, "fingerprint": tokenizer.fingerprint()}
    bm25 = _bm25_record(config)
    postings_sha = sha256_hex(postings_bytes)
    manifest = seal_manifest({
        "schema_version": INDEX_SCHEMA_VERSION, "kind": INDEX_KIND,
        "index_id": _index_id(corpus.corpus_id, tokenizer_record, bm25, postings_sha),
        "corpus": {
            "corpus_id": corpus.corpus_id,
            "manifest_path": relative_reference(output.parent, corpus_path),
            "manifest_file_sha256": sha256_file(corpus_path),
        },
        "tokenizer": tokenizer_record, "bm25": bm25, "stats": _stats_record(index),
        "postings": {"path": postings_path.name, "sha256": postings_sha},
        "built_with": code_identity(),
    })

    if output.exists():
        existing = as_object(loads_strict(read_utf8(output, where="existing index manifest"), where="existing index manifest"), "existing index manifest")
        if existing.get("index_id") != manifest["index_id"]:
            raise ManifestError(
                f"{output.name} already exists with a different index_id "
                f"({existing.get('index_id')}); choose a new output path. Artifacts are immutable"
            )
        load_index(output, root)  # the identical index must also still be intact
        return existing, False
    if postings_path.exists() and sha256_file(postings_path) != postings_sha:
        raise ManifestError(f"{postings_path.name} already exists with different content; not overwriting")
    _write_atomic(postings_path, postings_bytes)
    _write_atomic(output, pretty_json(manifest).encode("utf-8"))
    return manifest, True


def load_index(manifest_path: Path, root: Path) -> LoadedIndex:
    """Load an index with full reference, hash, tokenizer and corpus checks."""
    where = "index manifest"
    root = root.resolve(strict=True)
    path = ensure_inside(manifest_path, root, what=where)
    base = path.parent
    manifest = dict(as_object(loads_strict(read_utf8(path, where=where), where=where), where))
    exact_keys(manifest, _MANIFEST_FIELDS, where)
    if manifest["schema_version"] != INDEX_SCHEMA_VERSION or manifest["kind"] != INDEX_KIND:
        raise ManifestError(f"{where}: expected kind={INDEX_KIND!r}, schema_version={INDEX_SCHEMA_VERSION}")
    verify_manifest_hash(manifest, where=where)

    corpus_ref = as_object(manifest["corpus"], f"{where}.corpus")
    exact_keys(corpus_ref, ("corpus_id", "manifest_path", "manifest_file_sha256"), f"{where}.corpus")
    tokenizer_record = as_object(manifest["tokenizer"], f"{where}.tokenizer")
    exact_keys(tokenizer_record, ("name", "fingerprint"), f"{where}.tokenizer")
    if tokenizer_record["name"] not in TOKENIZER_NAMES:
        raise ManifestError(f"{where}.tokenizer: unsupported tokenizer {tokenizer_record['name']!r}")
    if tokenizer_record["fingerprint"] != tokenizer.fingerprint():
        raise ManifestError(
            f"{where}.tokenizer: built with different tokenizer rules than this code; rebuild the index"
        )
    bm25 = as_object(manifest["bm25"], f"{where}.bm25")
    exact_keys(bm25, ("variant", "idf", "k1", "b", "query_term_weighting", "score_decimals", "tie_break"), f"{where}.bm25")
    idf, k1, b = bm25["idf"], bm25["k1"], bm25["b"]
    numbers_ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (k1, b))
    if (
        bm25["variant"] != "okapi" or idf not in IDF_VARIANTS or not numbers_ok
        or bm25["query_term_weighting"] != "binary" or bm25["score_decimals"] != SCORE_DECIMALS
        or bm25["tie_break"] != _TIE_BREAK
    ):
        raise ManifestError(f"{where}.bm25: unsupported or malformed BM25 settings")

    corpus_path = resolve_reference(base, corpus_ref["manifest_path"], root, what=f"{where}.corpus.manifest_path")
    if sha256_file(corpus_path) != as_sha256(corpus_ref["manifest_file_sha256"], f"{where}.corpus.manifest_file_sha256"):
        raise ManifestError(f"{where}: the corpus manifest changed since this index was built")
    corpus = load_corpus(corpus_path)
    if corpus.corpus_id != corpus_ref["corpus_id"]:
        raise ManifestError(f"{where}: corpus_id does not match the referenced corpus")

    postings_ref = as_object(manifest["postings"], f"{where}.postings")
    exact_keys(postings_ref, ("path", "sha256"), f"{where}.postings")
    postings_path = resolve_reference(base, postings_ref["path"], root, what=f"{where}.postings.path")
    postings_sha = as_sha256(postings_ref["sha256"], f"{where}.postings.sha256")
    if sha256_file(postings_path) != postings_sha:
        raise ManifestError(f"{where}: postings file does not match its recorded SHA-256")
    postings = parse_postings(
        loads_strict(read_utf8(postings_path, where="postings"), where="postings"),
        expected_chunk_ids=[chunk.chunk_id for chunk in corpus.chunks], where="postings",
    )
    index = Bm25Index(postings, idf=idf, k1=float(k1), b=float(b))
    if manifest["stats"] != _stats_record(index):
        raise ManifestError(f"{where}.stats do not match the postings")
    if manifest["index_id"] != _index_id(corpus.corpus_id, dict(tokenizer_record), dict(bm25), postings_sha):
        raise ManifestError(f"{where}.index_id does not match the index contents")
    as_str(as_object(manifest["built_with"], f"{where}.built_with").get("source_snapshot_sha256"), f"{where}.built_with")
    return LoadedIndex(manifest=manifest, corpus=corpus, index=index)


def verify_index_deep(loaded: LoadedIndex) -> None:
    """Rebuild postings from the corpus text and require them to equal the stored ones."""
    fresh: Postings = build_postings(loaded.corpus.chunks)
    if fresh != loaded.index.postings:
        raise ManifestError("stored postings differ from a fresh build over the corpus text")
