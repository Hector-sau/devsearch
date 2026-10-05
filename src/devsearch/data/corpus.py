"""Versioned corpus artifact: documents, chunks, normalized text and source maps.

Layout, with every reference relative to and contained in the manifest directory:

    corpus-manifest.json   schema_version, kind, corpus_id, documents, chunks, files, manifest_sha256
    documents.jsonl        doc records (docs/02 §2), sorted by (doc_id, revision)
    chunks.jsonl           chunk records, sorted by (doc_id, revision, start_char, chunk_id)
    text/NNNN.txt          normalized UTF-8 text, never rewritten in place
    maps/NNNN.json         one source map per document (see devsearch.data.sourcemap)

Loading is strict: unknown or missing fields, duplicate IDs, dangling references,
hash mismatches, escaping paths, chunk text that differs from its normalized-text
slice and chunks with no source trace all raise ManifestError. Chunk IDs are
re-derived and must match. Text, offsets and IDs never depend on a model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import (
    canonical_json, digest_of, loads_strict, pretty_json, read_utf8, seal_manifest,
    sha256_file, sha256_text, verify_manifest_hash,
)
from devsearch.artifacts.paths import resolve_reference
from devsearch.artifacts.schema import (
    as_bool, as_https_url, as_id, as_int, as_label, as_list, as_object, as_sha256, as_str,
    exact_keys,
)
from devsearch.contracts import Chunk, Doc
from devsearch.data.sourcemap import (
    SOURCE_MAP_SCHEMA_VERSION, SourceMap, SourceSegment, parse_source_map,
)
from devsearch.errors import ManifestError

CORPUS_SCHEMA_VERSION = 1
CORPUS_KIND = "corpus"
MANIFEST_NAME = "corpus-manifest.json"
SPLITS = ("development", "train", "validation", "test")

_MANIFEST_FIELDS = (
    "schema_version", "kind", "corpus_id", "documents", "chunks", "files", "manifest_sha256",
)
_FILE_REF_FIELDS = ("path", "sha256", "count")
_INVENTORY_FIELDS = ("path", "sha256")
_DOC_FIELDS = (
    "doc_id", "source_id", "title", "revision", "source_url", "license", "content_sha256",
    "normalized_text_path", "source_map_path", "parser_version", "parse_status",
)
_CHUNK_FIELDS = (
    "chunk_id", "doc_id", "revision", "start_char", "end_char", "heading_path", "text",
    "chunker_version", "evidence_group_id", "owner_split", "code_fragment",
)
DocKey = tuple[str, str]


def derive_chunk_id(
    doc_id: str, revision: str, start_char: int, end_char: int, text: str, chunker_version: str
) -> str:
    """Deterministic ID from document identity, offsets, text hash and chunker version."""
    return "ch_" + digest_of({
        "doc_id": doc_id, "revision": revision, "start_char": start_char,
        "end_char": end_char, "text_sha256": sha256_text(text),
        "chunker_version": chunker_version,
    })[:32]


@dataclass(frozen=True)
class Corpus:
    corpus_id: str
    manifest_sha256: str
    docs: Mapping[DocKey, Doc]
    chunks: tuple[Chunk, ...]
    source_maps: Mapping[DocKey, SourceMap]

    def doc_of(self, chunk: Chunk) -> Doc:
        return self.docs[(chunk.doc_id, chunk.revision)]

    def revisions_by_source(self) -> dict[str, tuple[str, ...]]:
        found: dict[str, set[str]] = {}
        for doc in self.docs.values():
            found.setdefault(doc.source_id, set()).add(doc.revision)
        return {source: tuple(sorted(revisions)) for source, revisions in sorted(found.items())}

    def locate(self, chunk: Chunk) -> list[dict[str, Any]]:
        return self.source_maps[(chunk.doc_id, chunk.revision)].locate(chunk.start_char, chunk.end_char)


def _read_jsonl(path: Path, where: str) -> list[tuple[str, dict[str, Any]]]:
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


def _file_ref(value: Any, where: str, base: Path) -> tuple[Path, str, int]:
    ref = as_object(value, where)
    exact_keys(ref, _FILE_REF_FIELDS, where)
    path = resolve_reference(base, ref["path"], base, what=f"{where}.path")
    digest = as_sha256(ref["sha256"], f"{where}.sha256")
    if sha256_file(path) != digest:
        raise ManifestError(f"{where}: {ref['path']} does not match its recorded SHA-256")
    return path, digest, as_int(ref["count"], f"{where}.count")


def _parse_doc(record: dict[str, Any], where: str) -> dict[str, Any]:
    exact_keys(record, _DOC_FIELDS, where)
    if record["parse_status"] != "accepted":
        raise ManifestError(f"{where}.parse_status: only 'accepted' documents belong in a corpus")
    return {
        "doc_id": as_id(record["doc_id"], f"{where}.doc_id"),
        "source_id": as_id(record["source_id"], f"{where}.source_id"),
        "title": as_label(record["title"], f"{where}.title", max_length=500),
        "revision": as_label(record["revision"], f"{where}.revision", max_length=128),
        "source_url": None if record["source_url"] is None
        else as_https_url(record["source_url"], f"{where}.source_url"),
        "license": as_label(record["license"], f"{where}.license", max_length=200),
        "content_sha256": as_sha256(record["content_sha256"], f"{where}.content_sha256"),
        "normalized_text_path": record["normalized_text_path"],
        "source_map_path": record["source_map_path"],
        "parser_version": as_label(record["parser_version"], f"{where}.parser_version", max_length=64),
    }


def _parse_chunk(record: dict[str, Any], where: str) -> Chunk:
    exact_keys(record, _CHUNK_FIELDS, where)
    split = record["owner_split"]
    if split is not None and split not in SPLITS:
        raise ManifestError(f"{where}.owner_split: must be null or one of {', '.join(SPLITS)}")
    start = as_int(record["start_char"], f"{where}.start_char")
    headings = tuple(
        as_label(item, f"{where}.heading_path[{index}]", max_length=500)
        for index, item in enumerate(as_list(record["heading_path"], f"{where}.heading_path"))
    )
    return Chunk(
        chunk_id=as_id(record["chunk_id"], f"{where}.chunk_id"),
        doc_id=as_id(record["doc_id"], f"{where}.doc_id"),
        revision=as_label(record["revision"], f"{where}.revision", max_length=128),
        text=as_str(record["text"], f"{where}.text", max_length=1_000_000),
        start_char=start,
        end_char=as_int(record["end_char"], f"{where}.end_char", minimum=start + 1),
        heading_path=headings,
        chunker_version=as_label(record["chunker_version"], f"{where}.chunker_version", max_length=64),
        evidence_group_id=as_id(record["evidence_group_id"], f"{where}.evidence_group_id"),
        owner_split=split,
        code_fragment=as_bool(record["code_fragment"], f"{where}.code_fragment"),
    )


def load_corpus(manifest_path: Path) -> Corpus:
    """Load and fully validate a corpus; every referenced file stays inside the manifest directory."""
    where = "corpus manifest"
    try:
        base = manifest_path.resolve(strict=True).parent
    except OSError as error:
        raise ManifestError(f"{where}: cannot open {manifest_path}: {error.strerror or error}") from error
    manifest = dict(as_object(
        loads_strict(read_utf8(manifest_path, where=where), where=where), where
    ))
    exact_keys(manifest, _MANIFEST_FIELDS, where)
    if manifest["schema_version"] != CORPUS_SCHEMA_VERSION or manifest["kind"] != CORPUS_KIND:
        raise ManifestError(f"{where}: expected kind={CORPUS_KIND!r}, schema_version={CORPUS_SCHEMA_VERSION}")
    verify_manifest_hash(manifest, where=where)

    documents_path, documents_sha, document_count = _file_ref(manifest["documents"], f"{where}.documents", base)
    chunks_path, chunks_sha, chunk_count = _file_ref(manifest["chunks"], f"{where}.chunks", base)

    inventory: dict[str, str] = {}
    for index, item in enumerate(as_list(manifest["files"], f"{where}.files")):
        label = f"{where}.files[{index}]"
        entry = as_object(item, label)
        exact_keys(entry, _INVENTORY_FIELDS, label)
        path = resolve_reference(base, entry["path"], base, what=f"{label}.path")
        digest = as_sha256(entry["sha256"], f"{label}.sha256")
        if entry["path"] in inventory:
            raise ManifestError(f"{label}: duplicate path {entry['path']!r}")
        if sha256_file(path) != digest:
            raise ManifestError(f"{label}: {entry['path']} does not match its recorded SHA-256")
        inventory[entry["path"]] = digest
    if list(inventory) != sorted(inventory):
        raise ManifestError(f"{where}.files: entries must be sorted by path")
    expected_id = "corpus-" + digest_of({
        "schema_version": CORPUS_SCHEMA_VERSION, "documents_sha256": documents_sha,
        "chunks_sha256": chunks_sha,
        "files": [{"path": path, "sha256": digest} for path, digest in inventory.items()],
    })[:32]
    if manifest["corpus_id"] != expected_id:
        raise ManifestError(f"{where}.corpus_id does not match the corpus contents")

    docs: dict[DocKey, Doc] = {}
    source_maps: dict[DocKey, SourceMap] = {}
    used_files: set[str] = set()
    previous: DocKey | None = None
    document_records = _read_jsonl(documents_path, "documents.jsonl")
    for label, record in document_records:
        fields = _parse_doc(record, label)
        key = (fields["doc_id"], fields["revision"])
        if key in docs:
            raise ManifestError(f"{label}: duplicate document {key}")
        if previous is not None and key < previous:
            raise ManifestError(f"{label}: documents must be sorted by (doc_id, revision)")
        previous = key
        resolved: dict[str, Path] = {}
        for field in ("normalized_text_path", "source_map_path"):
            path = fields[field]
            resolved[field] = resolve_reference(base, path, base, what=f"{label}.{field}")
            if path not in inventory:
                raise ManifestError(f"{label}.{field}: {path!r} is not listed in the manifest files")
            if path in used_files:
                raise ManifestError(f"{label}.{field}: {path!r} is already used by another document")
            used_files.add(path)
        text = read_utf8(resolved["normalized_text_path"], where=f"{label} normalized text")
        if sha256_text(text) != fields["content_sha256"]:
            raise ManifestError(f"{label}: content_sha256 does not match the normalized text")
        map_text = read_utf8(resolved["source_map_path"], where=f"{label} source map")
        source_maps[key] = parse_source_map(
            loads_strict(map_text, where=f"{label} source map"), doc_id=key[0], revision=key[1],
            text_length=len(text), where=f"{label} source map",
        )
        docs[key] = Doc(
            doc_id=key[0], revision=key[1], source_id=fields["source_id"], title=fields["title"],
            text=text, source_url=fields["source_url"], content_sha256=fields["content_sha256"],
            parser_version=fields["parser_version"],
        )
    if set(inventory) != used_files:
        raise ManifestError(f"{where}.files: unreferenced file(s): {', '.join(sorted(set(inventory) - used_files))}")
    if len(docs) != document_count:
        raise ManifestError(f"{where}.documents.count does not match documents.jsonl")

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    previous_order: tuple[str, str, int, str] | None = None
    for label, record in _read_jsonl(chunks_path, "chunks.jsonl"):
        chunk = _parse_chunk(record, label)
        doc = docs.get((chunk.doc_id, chunk.revision))
        if doc is None:
            raise ManifestError(f"{label}: references unknown document {(chunk.doc_id, chunk.revision)}")
        if chunk.end_char > len(doc.text):
            raise ManifestError(f"{label}: end_char {chunk.end_char} is beyond the {len(doc.text)}-character text")
        if chunk.text != doc.text[chunk.start_char:chunk.end_char]:
            raise ManifestError(f"{label}: text differs from the normalized-text slice [{chunk.start_char}, {chunk.end_char})")
        if not chunk.text.strip():
            raise ManifestError(f"{label}: chunk text is empty or whitespace only")
        expected = derive_chunk_id(
            chunk.doc_id, chunk.revision, chunk.start_char, chunk.end_char, chunk.text, chunk.chunker_version
        )
        if chunk.chunk_id != expected:
            raise ManifestError(f"{label}: chunk_id does not match its derived ID")
        if chunk.chunk_id in seen_ids:
            raise ManifestError(f"{label}: duplicate chunk_id {chunk.chunk_id}")
        seen_ids.add(chunk.chunk_id)
        order = (chunk.doc_id, chunk.revision, chunk.start_char, chunk.chunk_id)
        if previous_order is not None and order <= previous_order:
            raise ManifestError(f"{label}: chunks must be sorted by (doc_id, revision, start_char, chunk_id)")
        previous_order = order
        if not source_maps[(chunk.doc_id, chunk.revision)].locate(chunk.start_char, chunk.end_char):
            raise ManifestError(f"{label}: chunk has no non-synthetic source location")
        chunks.append(chunk)
    if len(chunks) != chunk_count:
        raise ManifestError(f"{where}.chunks.count does not match chunks.jsonl")

    return Corpus(
        corpus_id=manifest["corpus_id"], manifest_sha256=manifest["manifest_sha256"],
        docs=docs, chunks=tuple(chunks), source_maps=source_maps,
    )


@dataclass(frozen=True)
class ChunkSpan:
    start_char: int
    end_char: int
    evidence_group_id: str
    heading_path: tuple[str, ...] = ()
    owner_split: str | None = None
    code_fragment: bool = False


@dataclass(frozen=True)
class DocInput:
    doc_id: str
    source_id: str
    title: str
    revision: str
    source_url: str | None
    license: str
    parser_version: str
    text: str
    segments: tuple[SourceSegment, ...]
    chunks: tuple[ChunkSpan, ...]


def write_corpus(directory: Path, documents: Sequence[DocInput], *, chunker_version: str) -> dict[str, Any]:
    """Write a corpus into an existing empty-of-manifest directory, then reload it to validate.

    Used by tests and the later `prepare` command. It never creates the target
    directory itself, so a missing mount fails rather than being replaced.
    """
    if not directory.is_dir():
        raise ManifestError(f"corpus directory does not exist: {directory}")
    if (directory / MANIFEST_NAME).exists():
        raise ManifestError(f"{MANIFEST_NAME} already exists in {directory}; corpora are immutable")
    ordered = sorted(documents, key=lambda doc: (doc.doc_id, doc.revision))
    keys = [(doc.doc_id, doc.revision) for doc in ordered]
    if len(set(keys)) != len(keys):
        raise ManifestError("duplicate (doc_id, revision) in corpus input")
    (directory / "text").mkdir(exist_ok=True)
    (directory / "maps").mkdir(exist_ok=True)

    inventory: list[dict[str, str]] = []
    document_lines: list[str] = []
    chunk_records: list[dict[str, Any]] = []
    for index, doc in enumerate(ordered):
        text_path, map_path = f"text/{index:04d}.txt", f"maps/{index:04d}.json"
        text_bytes = doc.text.encode("utf-8")
        (directory / text_path).write_bytes(text_bytes)
        map_body = pretty_json({
            "schema_version": SOURCE_MAP_SCHEMA_VERSION, "doc_id": doc.doc_id,
            "revision": doc.revision, "entries": [seg.to_record() for seg in doc.segments],
        })
        (directory / map_path).write_bytes(map_body.encode("utf-8"))
        content_sha = sha256_text(doc.text)
        inventory.append({"path": text_path, "sha256": content_sha})
        inventory.append({"path": map_path, "sha256": sha256_text(map_body)})
        document_lines.append(canonical_json({
            "doc_id": doc.doc_id, "source_id": doc.source_id, "title": doc.title,
            "revision": doc.revision, "source_url": doc.source_url, "license": doc.license,
            "content_sha256": content_sha, "normalized_text_path": text_path,
            "source_map_path": map_path, "parser_version": doc.parser_version,
            "parse_status": "accepted",
        }))
        for span in doc.chunks:
            text = doc.text[span.start_char:span.end_char]
            chunk_records.append({
                "chunk_id": derive_chunk_id(doc.doc_id, doc.revision, span.start_char, span.end_char, text, chunker_version),
                "doc_id": doc.doc_id, "revision": doc.revision, "start_char": span.start_char,
                "end_char": span.end_char, "heading_path": list(span.heading_path), "text": text,
                "chunker_version": chunker_version, "evidence_group_id": span.evidence_group_id,
                "owner_split": span.owner_split, "code_fragment": span.code_fragment,
            })
    chunk_records.sort(key=lambda r: (r["doc_id"], r["revision"], r["start_char"], r["chunk_id"]))
    documents_body = "".join(line + "\n" for line in document_lines)
    chunks_body = "".join(canonical_json(record) + "\n" for record in chunk_records)
    (directory / "documents.jsonl").write_bytes(documents_body.encode("utf-8"))
    (directory / "chunks.jsonl").write_bytes(chunks_body.encode("utf-8"))
    inventory.sort(key=lambda item: item["path"])
    documents_sha, chunks_sha = sha256_text(documents_body), sha256_text(chunks_body)
    manifest = seal_manifest({
        "schema_version": CORPUS_SCHEMA_VERSION, "kind": CORPUS_KIND,
        "corpus_id": "corpus-" + digest_of({
            "schema_version": CORPUS_SCHEMA_VERSION, "documents_sha256": documents_sha,
            "chunks_sha256": chunks_sha, "files": inventory,
        })[:32],
        "documents": {"path": "documents.jsonl", "sha256": documents_sha, "count": len(ordered)},
        "chunks": {"path": "chunks.jsonl", "sha256": chunks_sha, "count": len(chunk_records)},
        "files": inventory,
    })
    (directory / MANIFEST_NAME).write_bytes(pretty_json(manifest).encode("utf-8"))
    load_corpus(directory / MANIFEST_NAME)
    return manifest
