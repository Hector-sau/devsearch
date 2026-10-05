"""Builds data drafts and a `data.toml` for the preparation stages from the fixture corpus.

The corpus is unassigned (no owners); `prepare_groups` assigns it. Drafts are generated *from the actual
assignment* — one synthetic question per evidence group of the chosen splits — so they are consistent
by construction. Question wording is test data (templated), not a quality signal.
"""

import json
import shutil
import tempfile
from pathlib import Path

from devsearch.artifacts.files import jsonl_bytes
from devsearch.data.corpus import load_corpus
from devsearch.data.config import load_data_config
from devsearch.data.prepare import SPLIT_MANIFEST, load_split, prepare_groups
from devsearch.data.queries import EvidenceSpan, Qrel, Review, qrel_record, query_record

from helpers import CONFIGS, MINI_CORPUS
from test_queries import make_query

ACCEPTED = (Review("agent", "fixture-reviewer", "accepted", "2026-10-05T12:00:00Z", "wf-1"),)
DATA_TOML = """schema_version = 1
protocol = "{protocol}"
[inputs]
{links}
[inputs.queries]
train = "queries-train.jsonl"
validation = "queries-validation.jsonl"
test = "queries-test.jsonl"
[inputs.qrels]
train = "qrels-train.jsonl"
validation = "qrels-validation.jsonl"
test = "qrels-test.jsonl"
[groups]
near_duplicate_threshold = 0.9
shingle_size = 5
seed = {seed}
[groups.targets]
train = {train}
validation = {validation}
test = {test}
[mining]
hard_negatives = 2
random_negatives = 2
seed = 42
hard_search_depth = 100
[candidates]
depth = 50
[index]
config = "{index}"
[review]
salt = "salt-for-the-test-suite"
"""


def write_config(directory: Path, *, protocol="prejudged_fixed_candidates_v1", seed=20261005, shares=(0.5, 0.25, 0.25), links=None) -> Path:
    path = directory / "data.toml"
    link_line = "" if links is None else f'links = "{links}"'
    path.write_text(DATA_TOML.format(protocol=protocol, seed=seed, train=shares[0], validation=shares[1], test=shares[2],
                                     links=link_line, index=CONFIGS / "index.toml"), encoding="utf-8")
    return path


def make_root(testcase) -> Path:
    root = Path(tempfile.mkdtemp(prefix="devsearch-prep-")).resolve()
    testcase.addCleanup(shutil.rmtree, root, ignore_errors=True)
    shutil.copytree(MINI_CORPUS, root / "corpus-in")
    (root / "prepared").mkdir()
    (root / "evaluation").mkdir()
    (root / "drafts").mkdir()
    return root


def draft_for(corpus, assignment_doc, *, per_split=None, split_of_group=None):
    """Queries and qrels: one per evidence group (or up to `per_split`) of every split, consistent with ownership."""
    owner_of_group = split_of_group or {}
    queries, qrels = [], []
    seen = {"train": 0, "validation": 0, "test": 0}
    by_group = {}
    for chunk in corpus.chunks:
        by_group.setdefault(chunk.evidence_group_id, []).append(chunk)
    counter = 0
    for group in sorted(by_group):
        chunks = by_group[group]
        split = chunks[0].owner_split or owner_of_group.get(group)
        if split not in seen or (per_split is not None and seen[split] >= per_split):
            continue
        seen[split] += 1
        counter += 1
        chunk = chunks[0]
        doc = corpus.doc_of(chunk)
        query_id = f"q-{split[:2]}-{counter}"
        text = f"How does {' '.join(chunk.heading_path).lower() or doc.title.lower()} work number {counter}?"
        queries.append(make_query(query_id, text, [group], split))
        span = EvidenceSpan(doc.doc_id, doc.revision, doc.content_sha256, chunk.start_char + 1, min(chunk.end_char, chunk.start_char + 30))
        qrels.append(Qrel(query_id, chunk.chunk_id, 1, (span,), "synthetic", "accepted", ACCEPTED))
    return queries, qrels


def write_drafts(directory: Path, queries, qrels) -> None:
    """One file per split, sorted by key, as the loaders require; duplicates stay duplicates (and are rejected on load)."""
    split_of = {q.query_id: q.split for q in queries}
    for split in ("train", "validation", "test"):
        members = sorted((q for q in queries if q.split == split), key=lambda q: q.query_id)
        judged = sorted((r for r in qrels if split_of.get(r.query_id) == split), key=lambda r: (r.query_id, r.chunk_id))
        (directory / f"queries-{split}.jsonl").write_bytes(jsonl_bytes(query_record(q) for q in members))
        (directory / f"qrels-{split}.jsonl").write_bytes(jsonl_bytes(qrel_record(r) for r in judged))


def stage_groups(root: Path, config_path: Path):
    config = load_data_config(config_path)
    manifest = prepare_groups(config, root / "corpus-in" / "corpus-manifest.json", root / "prepared", root)
    split = load_split(root / "prepared" / SPLIT_MANIFEST)
    return config, manifest, split


def drafts_from_split(root: Path, split) -> tuple[list, list]:
    corpus = load_corpus(split["corpus_path"])
    queries, qrels = draft_for(corpus, split)
    write_drafts(root / "drafts", queries, qrels)
    return queries, qrels


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
