"""A complete DIAGNOSTIC rehearsal bundle: engineering evidence that the pipeline runs end to end, nothing more.

Everything it makes is synthetic and says so: invented pseudo-technical documents (original text under the
reserved `.invalid` domain), templated questions, a scripted oracle standing in for a reviewer, and a tiny random
encoder with a local tokenizer. The questions name a topic's terms, which occur in all three of its sections, so BM25
cannot tell the sections apart; the answer section (the only one saying "canonical") carries a PLANTED cue a
reranker can learn. That lets the rehearsal exercise the "reranker qualifies and ships" path as well as the
"BM25 stays the default" path. It is a plumbing device, not a model of real relevance. It exercises preparation, the campaign, export, reload and backup without any
network, any downloaded model or any real data. Its numbers describe the plumbing, never ranking quality, and
its reports are stamped `diagnostic`; it can never satisfy a backend or data-readiness gate for a real campaign.

    python -m devsearch.rehearsal --output <existing empty directory> [--topics 36] [--seed 7]

The output directory receives `bundle/` (prepared data, diagnostic assets, protocol, bundle manifest),
`evaluation/` (the separate test side) and a `host-example.toml` for the `cpu-diagnostic` profile.
"""

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import pretty_json
from devsearch.artifacts.files import jsonl_bytes
from devsearch.data.config import load_data_config
from devsearch.data.corpus import ChunkSpan, DocInput, load_corpus, write_corpus
from devsearch.data.prepare import (
    EVALUATION_MANIFEST, PREPARED_MANIFEST, SPLIT_MANIFEST, load_evaluation, load_prepared, load_split, prepare_bank, prepare_groups,
    prepare_materialize,
)
from devsearch.data.queries import EvidenceSpan, Qrel, Review, qrel_record, query_record
from devsearch.data.review import build_review_pool, prejudged_universe
from devsearch.data.sourcemap import SourceSegment
from devsearch.errors import ManifestError
from devsearch.execution.assets import build_diagnostic_assets
from devsearch.execution.bundle import assemble_bundle

LICENSE = "rehearsal-original-synthetic"
STAMP = "2026-10-05T12:00:00Z"
REVIEWER = {"type": "agent", "id": "rehearsal-oracle-script", "revision": "script-1"}
SYLLABLES = ("ba", "ko", "ri", "tu", "ve", "mi", "so", "la", "ne", "du", "pa", "xi", "zo", "qua", "fre", "gli", "tho", "wyn", "jar", "cel")
CUE_WORDS = {"canonical", "this", "entry", "description", "component", "keeps", "register", "routine", "each", "call", "then", "flush", "handle", "returns", "count", "share", "between", "instances", "reset", "while", "open", "does", "work"}
FILLER = ("the", "value", "is", "stored", "when", "a", "request", "arrives", "and", "each", "stage", "keeps", "its", "own", "state", "for", "later",
          "use", "so", "that", "retries", "stay", "safe", "during", "a", "restart", "of", "the", "service")

DATA_TOML = """schema_version = 1
protocol = "prejudged_fixed_candidates_v1"
[inputs]
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
train = 0.5
validation = 0.25
test = 0.25
[mining]
hard_negatives = 2
random_negatives = 2
seed = 42
hard_search_depth = 100
[candidates]
depth = 50
[index]
config = "index.toml"
[review]
salt = "rehearsal-salt-{seed}-not-a-secret"
"""

INDEX_TOML = """schema_version = 1
tokenizer = "lexical-v1"
[bm25]
idf = "lucene"
k1 = 1.2
b = 0.75
"""

PROTOCOL_TOML = """# DIAGNOSTIC rehearsal protocol: the real structure with small counts so a CPU run finishes quickly.
schema_version = 1
[length]
max_pair_tokens = 128
max_query_tokens = 32
[matrix]
seeds = [42]
deployment_seed = 42
epochs = 3
head_seed = 1234
[[recipe]]
id = "frozen"
mode = "frozen"
learning_rate = 1e-3
[[recipe]]
id = "full"
mode = "full"
learning_rate = 3e-3
[training]
dropout = 0.1
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
max_grad_norm = 1.0
warmup_fraction = 0.10
effective_batch = 16
pos_weight = "auto"
validation_batch = 8
[selection]
min_group_mrr_gain = 0.02
max_p95_ms = 2000
[latency]
warmup = 10
passes = 3
queries = 50
seed = 20261005
[bootstrap]
confidence = 0.95
repetitions = 1000
seed = 20261005
[inference]
batch_candidates = [4, 2, 1, 8, 16]
"""

HOST_EXAMPLE = """# Example host binding for the cpu-diagnostic rehearsal profile. A real accelerator host needs its own file with a verified
# mount point (require_mount_point = true), an independent backup device and measured memory caps.
schema_version = 1
[device]
profile = "cpu-diagnostic"
[paths]
runtime_root = "{base}/runtime"
bundle = "{base}/bundle"
run_root = "{base}/runtime/run"
evaluation_root = "{base}/evaluation"
backup_dir = "{base}/backup"
[mount]
path = "{base}"
marker = ".devsearch-volume"
expected_id = "rehearsal-volume"
require_mount_point = false
[deadline]
at = "2099-01-01T00:00:00Z"
reserve_seconds = 120
[resources]
min_free_bytes = 100000000
host_memory_cap_bytes = 8589934592
device_memory_cap_bytes = 0
[limits]
process_timeout_seconds = 300
doctor_budget_seconds = 1800
cleanup_grace_seconds = 30
io_retries = 2
[backup]
require_independent_device = false
"""


def _words(rng: random.Random, count: int) -> list[str]:
    seen: set[str] = set()
    while len(seen) < count:
        seen.add("".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3)))))
    return sorted(seen)


def synthetic_documents(seed: int, topics: int) -> list[DocInput]:
    """One document per topic with three sections (overview, code-like, notes); every section is its own evidence group."""
    rng = random.Random(seed)
    names = _words(rng, topics * 4)
    documents = []
    for index in range(topics):
        a, b, c, d = names[index * 4:index * 4 + 4]
        sections = [
            ("Overview", False, f"The {a} component keeps a {b} register and a {c} routine for each {d}. This entry is the canonical description. "
                                f"{' '.join(rng.sample(FILLER, 8))}."),
            ("Usage", True, f"call {a}_{b}(handle, {c}) then {d}_flush(handle) {' '.join(rng.sample(FILLER, 6))} returns the {a} {d} count"),
            ("Notes", False, f"Do not share a {c} between two {a} instances or reset the {b} register while a {d} is open. {' '.join(rng.sample(FILLER, 9))}."),
        ]
        text, segments, spans = "", [], []
        title = f"Rehearsal topic {a}"
        for number, (heading, is_code, body) in enumerate(sections):
            if text:
                text += "\n\n"
            start = len(text)
            text += body
            segments.append(SourceSegment(start, len(text), f"rehearsal/{a}.md", False, start_line=number * 3 + 1, end_line=number * 3 + 2))
            spans.append(ChunkSpan(start, len(text), f"eg-{a}-{heading.lower()}", (title, heading), None, is_code))
        documents.append(DocInput(
            doc_id=f"doc-{a}", source_id="rehearsal-docs", title=title, revision="rehearsal-r1", source_url=f"https://docs.rehearsal.invalid/{a}",
            license=LICENSE, parser_version="rehearsal-parser-1", text=text + "\n", segments=tuple(segments), chunks=tuple(spans)))
    return documents


def _question(rng: random.Random, chunk_text: str, counter: int) -> str:
    """Names two or three topic terms (shared by every section of the topic) but never the planted cue."""
    rare = sorted({w for w in re.findall(r"[a-z]{4,}", chunk_text.lower()) if w not in FILLER and w not in CUE_WORDS})
    picks = rng.sample(rare, min(3, len(rare)))
    return f"How does the {picks[0]} work with {' and '.join(picks[1:]) or picks[0]} in case {counter}?"


def draft_queries(split, directory: Path, seed: int) -> dict[str, int]:
    """Templated synthetic questions: one per evidence group of every split, each with its own positive chunk."""
    from devsearch.data.queries import Query, parse_query
    corpus = split["corpus"]
    rng = random.Random(seed + 1)
    per_split: dict[str, list[Query]] = {"train": [], "validation": [], "test": []}
    qrels: dict[str, list[Qrel]] = {"train": [], "validation": [], "test": []}
    by_group: dict[str, list] = {}
    for chunk in corpus.chunks:
        by_group.setdefault(chunk.evidence_group_id, []).append(chunk)
    for counter, group in enumerate(sorted(by_group), start=1):
        chunk = by_group[group][0]
        if chunk.heading_path[-1:] != ("Overview",):                 # only the answer sections get questions
            continue
        owner = chunk.owner_split
        if owner not in per_split:
            continue
        query_id = f"rq-{owner[:2]}-{counter:04d}"
        text = _question(rng, chunk.text, counter)
        record = {
            "query_id": query_id, "text": text, "language": "en", "group_id": f"pg-{query_id}", "evidence_group_ids": [group], "parent_query_id": None,
            "task_family": "rehearsal", "split": owner, "filters": {"sources": [], "versions": {}}, "origin": "synthetic",
            "generator": {"model": "rehearsal-template-script", "prompt_sha256": "0" * 64, "seed": seed},
            "review_status": "accepted", "reviews": [{"reviewer_type": "agent", "reviewer_id": REVIEWER["id"], "status": "accepted", "reviewed_at": STAMP, "revision": REVIEWER["revision"]}],
        }
        per_split[owner].append(parse_query(record, f"rehearsal query {query_id}"))
        doc = corpus.doc_of(chunk)
        span = EvidenceSpan(doc.doc_id, doc.revision, doc.content_sha256, chunk.start_char, min(chunk.end_char, chunk.start_char + 40))
        qrels[owner].append(Qrel(query_id, chunk.chunk_id, 1, (span,), "synthetic", "accepted",
                                 (Review("agent", REVIEWER["id"], "accepted", STAMP, REVIEWER["revision"]),)))
    counts = {}
    for split_name in ("train", "validation", "test"):
        members = sorted(per_split[split_name], key=lambda q: q.query_id)
        (directory / f"queries-{split_name}.jsonl").write_bytes(jsonl_bytes(query_record(q) for q in members))
        (directory / f"qrels-{split_name}.jsonl").write_bytes(jsonl_bytes(qrel_record(r) for r in sorted(qrels[split_name], key=lambda r: (r.query_id, r.chunk_id))))
        counts[split_name] = len(members)
    return counts


def oracle_reviews(config, prepared_manifest: Path, evaluation_manifest: Path, root: Path, evaluation_root: Path, path: Path) -> int:
    """A scripted stand-in for a reviewer: it knows the initial positives. Labelled as a script, never as real review."""
    from devsearch.data.candidates import load_candidates
    prepared = load_prepared(prepared_manifest, root)
    loaded = prepared["loaded"]
    evaluation = load_evaluation(evaluation_manifest, evaluation_root, loaded)
    _, candidates = load_candidates(evaluation["directory"] / evaluation["manifest"]["candidates"]["path"], loaded)
    pool = build_review_pool(config.protocol, config.salt, prejudged_universe(loaded, evaluation["queries"], candidates, evaluation["qrels"]))
    positives = {(q.query_id, q.chunk_id) for q in evaluation["qrels"] if q.confirmed_positive}
    chunks = {c.chunk_id: c for c in loaded.corpus.chunks}
    lines = []
    for item in pool.items:
        value = 1 if (item.query_id, item.chunk_id) in positives else 0
        length = len(chunks[item.chunk_id].text)
        lines.append({"item_id": item.item_id, "role": "primary", "relevance": value, "status": "resolved",
                      "evidence_spans": [{"start": 0, "end": min(40, length)}] if value else [], "reviewer": dict(REVIEWER), "reviewed_at": STAMP,
                      "notes": "scripted rehearsal oracle; not a real review"})
    path.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lines), encoding="utf-8")
    return len(lines)


def vocabulary(corpus, directory: Path) -> list[str]:
    words: set[str] = set()
    for chunk in corpus.chunks:
        words.update(re.findall(r"[A-Za-z0-9]+", chunk.text.lower()))
    for path in directory.glob("queries-*.jsonl"):
        for line in path.read_text().splitlines():
            words.update(re.findall(r"[A-Za-z0-9]+", json.loads(line)["text"].lower()))
    return sorted(words)


def build_rehearsal(output: Path, *, topics: int = 60, seed: int = 7) -> dict[str, Any]:
    """Build the whole diagnostic tree under an existing empty directory and return where everything is."""
    output = Path(output).resolve(strict=True)
    if any(output.iterdir()):
        raise ManifestError(f"{output} is not empty; the rehearsal builds into a fresh directory")
    if topics < 18:
        raise ManifestError("need at least 18 topics (54 evidence groups) for three non-trivial splits")
    for name in ("corpus-in", "drafts", "bundle", "evaluation", "backup", "runtime"):
        (output / name).mkdir()
    (output / ".devsearch-volume").write_text("rehearsal-volume\n")
    bundle, prepared_dir = output / "bundle", output / "bundle" / "prepared"
    prepared_dir.mkdir()
    write_corpus(output / "corpus-in", synthetic_documents(seed, topics), chunker_version="rehearsal-chunker-1")
    drafts = output / "drafts"
    (drafts / "data.toml").write_text(DATA_TOML.format(seed=seed), encoding="utf-8")
    (drafts / "index.toml").write_text(INDEX_TOML, encoding="utf-8")
    config = load_data_config(drafts / "data.toml")
    prepare_groups(config, output / "corpus-in" / "corpus-manifest.json", prepared_dir, output)
    split = load_split(prepared_dir / SPLIT_MANIFEST)
    counts = draft_queries(split, drafts, seed)
    prepare_materialize(config, prepared_dir / SPLIT_MANIFEST, output / "evaluation", output)
    reviews = drafts / "reviews.jsonl"
    reviewed = oracle_reviews(config, prepared_dir / PREPARED_MANIFEST, output / "evaluation" / EVALUATION_MANIFEST, output, output, reviews)
    prepare_bank(config, prepared_dir / PREPARED_MANIFEST, output / "evaluation" / EVALUATION_MANIFEST, reviews, output)
    (bundle / "protocol.toml").write_text(PROTOCOL_TOML, encoding="utf-8")
    assets = bundle / "model-assets"
    assets.mkdir()
    corpus = load_corpus(prepared_dir / "corpus" / "corpus-manifest.json")
    build_diagnostic_assets(assets, vocabulary(corpus, drafts), seed=seed)
    manifest = assemble_bundle(bundle)
    (output / "host-example.toml").write_text(HOST_EXAMPLE.format(base=output), encoding="utf-8")
    shutil.copyfile(drafts / "index.toml", output / "index.toml")
    summary = {"bundle": str(bundle), "evaluation_root": str(output / "evaluation"), "host_example": str(output / "host-example.toml"),
               "bundle_manifest_sha256": manifest["manifest_sha256"], "queries": counts, "review_items": reviewed, "diagnostic": True,
               "note": "synthetic corpus, templated questions, scripted oracle, random tiny encoder: plumbing evidence only"}
    (output / "rehearsal-summary.json").write_text(pretty_json(summary), encoding="utf-8")
    return summary
