"""From pinned snapshots to a corpus artifact and an honest preparation report (docs/02 §3, T03).

Per source: acquire (only when asked and only if missing), verify the pinned revision, inventory the files by the
include/exclude patterns, extract each file statically, chunk it under the pinned tokenizer's budget, and publish the passages
as an immutable corpus artifact. The report records what was *not* indexed as faithfully as what was: quarantined files with
reasons and source links, dropped constructs by kind, excluded passages by reason, duplicates, and the measured token
statistics. Nothing is truncated; no document is executed. The report holds no absolute paths or timestamps, so the same
inputs reproduce it byte for byte.
"""

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from devsearch.artifacts.canonical import seal_manifest, sha256_file, sha256_text
from devsearch.data.corpus import ChunkSpan, DocInput, write_corpus
from devsearch.errors import ManifestError
from devsearch.ingestion.chunker import ChunkingPolicy, TokenCounter, chunk_document
from devsearch.ingestion.config import PrepareConfig, SourceSpec
from devsearch.ingestion.gallery import parse_gallery
from devsearch.ingestion.markdown import parse_markdown
from devsearch.ingestion.model import Extraction, ParseFailure, clean_title
from devsearch.ingestion.notebook import parse_notebook
from devsearch.ingestion.rst import parse_rst
from devsearch.ingestion.snapshot import acquire, content_hash, inventory, supported, verify_revision

REPORT_KIND = "prepare-report"
REPORT_SCHEMA_VERSION = 1
_DOC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,255}$")
QUARANTINE_LISTED = 200
READERS = {
    "markdown": lambda text: parse_markdown(text, mdx=False), "mdx": lambda text: parse_markdown(text, mdx=True),
    "rst": parse_rst, "notebook": parse_notebook, "python-gallery": parse_gallery,
}


@dataclass(frozen=True)
class Prepared:
    manifest: dict[str, Any]          # the corpus manifest
    report: dict[str, Any]


def _title(extraction: Extraction, relative: str) -> str:
    return clean_title(extraction.title or Path(relative).stem.replace("_", " ").replace("-", " ")) or "untitled"


def _read(path: Path, max_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        raise ParseFailure(f"the file is {size} bytes, more than the {max_bytes}-byte limit")
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise ParseFailure(f"not valid UTF-8: {error.reason} at byte {error.start}") from error


def _url(spec: SourceSpec, relative: str) -> str | None:
    if spec.url_template is None:
        return None
    return spec.url_template.format(revision=spec.revision, path=quote(relative, safe="/"))


def _license(spec: SourceSpec) -> dict[str, Any]:
    record: dict[str, Any] = {"id": spec.license, "file": spec.license_file, "file_sha256": None,
                              "review": "the identifier is declared in the configuration; it is recorded, not verified against the upstream licence text"}
    if spec.license_file is not None:
        path = spec.snapshot / spec.license_file
        if not path.is_file() or path.is_symlink():
            raise ManifestError(f"{spec.source_id}: license_file {spec.license_file!r} is not a regular file in the snapshot")
        record["file_sha256"] = sha256_file(path)
    return record


def prepare_corpus(config: PrepareConfig, counter: TokenCounter, passage_budget: int, output: Path, *, allow_acquire: bool,
                   tokenizer_info: dict[str, Any], config_sha256: str) -> Prepared:
    policy = ChunkingPolicy(passage_budget, config.min_chunk_tokens, config.overlong_code)
    documents: list[DocInput] = []
    source_reports: list[dict[str, Any]] = []
    seen_text: dict[str, tuple[str, int]] = {}
    totals: Counter = Counter()
    excluded_total: Counter = Counter()
    token_counts: list[int] = []
    for spec in sorted(config.sources, key=lambda s: s.source_id):
        if not spec.snapshot.exists():
            if not allow_acquire:
                raise ManifestError(f"{spec.source_id}: the snapshot {spec.snapshot} does not exist and --offline forbids acquiring it")
            acquire(spec)
        state = verify_revision(spec)
        found = inventory(spec, config.max_files)
        quarantine: list[dict[str, str]] = []
        dropped: Counter = Counter()
        notes: Counter = Counter()
        files = Counter(seen=found.seen, included=len(found.included), excluded_by_pattern=found.excluded, symlinks_skipped=len(found.symlinks))
        source_chunks, source_excluded, source_fragments = 0, Counter(), 0
        for relative in found.included:
            fmt = supported(relative)
            doc_id = f"{spec.source_id}:{relative}"
            link = _url(spec, relative)
            try:
                if fmt is None:
                    raise ParseFailure(f"no static reader for the extension {Path(relative).suffix or '(none)'!r}")
                if not _DOC_ID.match(doc_id):
                    raise ParseFailure("the path has characters that a document id cannot carry")
                extraction = READERS[fmt](_read(spec.snapshot / relative, config.max_file_bytes))
            except ParseFailure as error:
                quarantine.append({"path": relative, "reason": str(error), "source_url": link or ""})
                files["quarantined"] += 1
                continue
            dropped.update(extraction.dropped)
            notes.update(extraction.notes)
            chunked = chunk_document(extraction, doc_id=doc_id, revision=spec.revision, path=relative, counter=counter, policy=policy)
            source_excluded.update(chunked.excluded)
            kept: list[ChunkSpan] = []
            for span, tokens in zip(chunked.chunks, chunked.tokens):
                digest = sha256_text(chunked.text[span.start_char:span.end_char])
                if digest in seen_text:
                    source_excluded["duplicate-passage"] += 1
                    continue
                seen_text[digest] = (doc_id, span.start_char)
                kept.append(span)
                token_counts.append(tokens)
                source_fragments += span.code_fragment
            if not kept:
                files["no_passages"] += 1
                continue
            files["accepted"] += 1
            source_chunks += len(kept)
            documents.append(DocInput(doc_id, spec.source_id, _title(extraction, relative), spec.revision, link, spec.license,
                                      f"{config.parser_version}/{extraction.fmt}", chunked.text, chunked.segments, tuple(kept)))
        totals.update(files)
        excluded_total.update(source_excluded)
        source_reports.append({
            "source_id": spec.source_id, "revision": spec.revision, "revision_verification": state,
            "snapshot_content_sha256": content_hash(spec.snapshot) if state != "verified-content-hash" else spec.snapshot_sha256,
            "license": _license(spec), "files": dict(sorted(files.items())), "by_extension": found.by_extension,
            "dropped_lines": dict(sorted(dropped.items())), "notes": dict(sorted(notes.items())),
            "passages": {"accepted": source_chunks, "code_fragments": source_fragments, "excluded": dict(sorted(source_excluded.items()))},
            "quarantine": {"count": len(quarantine), "listed": quarantine[:QUARANTINE_LISTED]},
            "symlinks_skipped": list(found.symlinks[:QUARANTINE_LISTED]),
        })
    if not documents:
        raise ManifestError("no document produced any passage; nothing to publish (see the quarantine reasons by running with a source that parses)")
    manifest = write_corpus(output, documents, chunker_version=config.chunker_version)
    ordered = sorted(token_counts)
    stats = {"passages": len(ordered), "min": ordered[0], "median": statistics.median(ordered), "mean": round(statistics.fmean(ordered), 2),
             "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], "max": ordered[-1], "budget": passage_budget, "truncated": 0}
    body = {
        "schema_version": REPORT_SCHEMA_VERSION, "kind": REPORT_KIND, "config_sha256": config_sha256, "parser_version": config.parser_version,
        "chunker_version": config.chunker_version,
        "policy": {"max_pair_tokens": config.max_pair_tokens, "max_query_tokens": config.max_query_tokens, "passage_budget": passage_budget,
                   "min_chunk_tokens": config.min_chunk_tokens, "overlong_code": config.overlong_code},
        "tokenizer": tokenizer_info, "sources": source_reports,
        "totals": {"documents": len(documents), "passages": stats["passages"], "files": dict(sorted(totals.items())),
                   "passages_excluded": dict(sorted(excluded_total.items())), "passage_tokens": stats},
        "corpus": {"corpus_id": manifest["corpus_id"], "manifest_sha256": manifest["manifest_sha256"]},
        "limitations": [
            "Source revisions marked declared-unverified were not checked against git history or a content pin.",
            "Licence identifiers are declared by the configuration; they were not verified against upstream licence text.",
            "Token counts use the tokenizer recorded above; a diagnostic tokenizer proves the mechanics only, and the real pinned tokenizer must "
            "re-run this step before any length claim.",
            "Dropped constructs and quarantined files are not indexed; questions that need them are outside this corpus.",
        ],
    }
    return Prepared(manifest, seal_manifest(body))
