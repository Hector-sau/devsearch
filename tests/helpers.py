"""Shared test helpers: runtime-root copies of the fixture corpus and raw corpus writers."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MINI_CORPUS = FIXTURES / "mini_corpus"
CONFIGS = REPO / "configs"
SRC = REPO / "src"
sys.path.insert(0, str(FIXTURES))

from devsearch.artifacts.canonical import canonical_json, digest_of, seal_manifest, sha256_hex  # noqa: E402
from devsearch.contracts import SearchRequest  # noqa: E402
from devsearch.data.corpus import CORPUS_KIND, CORPUS_SCHEMA_VERSION, MANIFEST_NAME  # noqa: E402
from devsearch.retrieval.config import IndexConfig, SearchConfig  # noqa: E402


def index_config(idf="lucene", k1=1.2, b=0.75) -> IndexConfig:
    return IndexConfig(tokenizer="lexical-v1", idf=idf, k1=k1, b=b)


def search_limits(**overrides) -> SearchConfig:
    values = dict(candidate_k=50, top_k=5, max_candidate_k=100, max_query_chars=1000, sources=(), versions=())
    values.update(overrides)
    return SearchConfig(**values)


def request(query, *, sources=(), versions=(), candidate_k=50, top_k=5) -> SearchRequest:
    return SearchRequest(
        query=query, sources=tuple(sources), versions=tuple(versions),
        candidate_k=candidate_k, top_k=top_k, mode="baseline",
    )


class RootTestCase(unittest.TestCase):
    """Gives each test a private runtime root holding `corpus/` (a fixture copy) and `index/`."""

    def make_root(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="devsearch-test-")).resolve()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        shutil.copytree(MINI_CORPUS, root / "corpus")
        (root / "index").mkdir()
        return root

    def setUp(self):
        self.root = self.make_root()
        self.corpus_manifest = self.root / "corpus" / MANIFEST_NAME
        self.index_manifest = self.root / "index" / "index-manifest.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def write_raw_corpus(
    directory: Path, documents: list[dict], chunks: list[dict], files: dict[str, bytes],
    *, chunks_bytes: bytes | None = None, documents_bytes: bytes | None = None,
) -> Path:
    """Write a corpus from raw records with *consistent* hashes, so a test can plant one defect.

    `chunks_bytes`/`documents_bytes` replace the serialized JSONL to plant file-format defects.
    """
    for relative, data in files.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    inventory = [{"path": p, "sha256": sha256_hex(files[p])} for p in sorted(files)]
    documents_body = documents_bytes if documents_bytes is not None else "".join(canonical_json(r) + "\n" for r in documents).encode("utf-8")
    chunks_body = chunks_bytes if chunks_bytes is not None else "".join(canonical_json(r) + "\n" for r in chunks).encode("utf-8")
    (directory / "documents.jsonl").write_bytes(documents_body)
    (directory / "chunks.jsonl").write_bytes(chunks_body)
    documents_sha, chunks_sha = sha256_hex(documents_body), sha256_hex(chunks_body)
    manifest = seal_manifest({
        "schema_version": CORPUS_SCHEMA_VERSION, "kind": CORPUS_KIND,
        "corpus_id": "corpus-" + digest_of({
            "schema_version": CORPUS_SCHEMA_VERSION, "documents_sha256": documents_sha,
            "chunks_sha256": chunks_sha, "files": inventory,
        })[:32],
        "documents": {"path": "documents.jsonl", "sha256": documents_sha, "count": len(documents)},
        "chunks": {"path": "chunks.jsonl", "sha256": chunks_sha, "count": len(chunks)},
        "files": inventory,
    })
    manifest_path = directory / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def fixture_parts() -> tuple[list[dict], list[dict], dict[str, bytes]]:
    """Records and file bytes of the checked-in fixture corpus, ready to mutate and rewrite."""
    files = {
        path.relative_to(MINI_CORPUS).as_posix(): path.read_bytes()
        for folder in ("text", "maps") for path in sorted((MINI_CORPUS / folder).iterdir())
    }
    return read_jsonl(MINI_CORPUS / "documents.jsonl"), read_jsonl(MINI_CORPUS / "chunks.jsonl"), files


def run_module(module: str, *args: str, cwd: Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run `python -m devsearch.<module>` outside the checkout, with only src on the path."""
    environment = {k: v for k, v in os.environ.items() if k != "DEVSEARCH_RUNTIME_ROOT"}
    environment.update(PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
    environment.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-B", "-m", f"devsearch.{module}", *args], cwd=cwd, env=environment,
        capture_output=True, text=True, timeout=60, check=False,
    )


def synthetic_corpus(testcase, items, *, source="s", revision="r1"):
    """Corpus with one document and one chunk per (evidence_group_id, text[, owner]) item.

    Documents are named d:NNNN in input order, so tests control chunk order and identity.
    """
    from devsearch.data.corpus import ChunkSpan, DocInput, MANIFEST_NAME, load_corpus, write_corpus
    from devsearch.data.sourcemap import SourceSegment
    directory = Path(tempfile.mkdtemp(prefix="devsearch-corpus-")).resolve()
    testcase.addCleanup(shutil.rmtree, directory, ignore_errors=True)
    docs = []
    for index, item in enumerate(items):
        group, text = item[0], item[1]
        owner = item[2] if len(item) > 2 else None
        docs.append(DocInput(
            f"d:{index:04d}", source, f"Title {index}", revision, None, "lic", "p1", text + "\n",
            (SourceSegment(0, len(text), "f.md", False, start_line=1, end_line=1),),
            (ChunkSpan(0, len(text), group, (), owner),),
        ))
    write_corpus(directory, docs, chunker_version="c1")
    return load_corpus(directory / MANIFEST_NAME)
