"""A small, fully consistent 'world' over the fixture corpus for data-layer tests.

Ownership is explicit (not randomly assigned) so tests can reason about exactly which chunks
may or may not appear: train / validation / test evidence groups, accepted synthetic queries
with confirmed positives, a built BM25 index and the merged-group structure.
"""

import shutil
import tempfile
from pathlib import Path

from devsearch.data.groups import merge_evidence_groups
from devsearch.data.queries import EvidenceSpan, Qrel, Review
from devsearch.retrieval.index_store import build_index, load_index

from helpers import MINI_CORPUS, index_config
from test_queries import make_query

TRAIN_GROUPS = ("eg-ckpt-save", "eg-ckpt-load", "eg-ckpt-resume", "eg-mode-switch", "eg-mode-grad", "eg-loop-intro",
                "eg-loop-code-1", "eg-tok-load", "eg-model-load")
VALIDATION_GROUPS = ("eg-ckpt-code", "eg-mode-accum", "eg-tok-padding", "eg-model-head")
TEST_GROUPS = ("eg-note-torch", "eg-note-hf", "eg-loop-code-2", "eg-loop-code-3", "eg-tok-unknown", "eg-tok-load-v2")
ACCEPTED = (Review("agent", "fixture-reviewer", "accepted", "2026-10-05T12:00:00Z", "wf-1"),)

# (query_id, text, evidence group, positive chunk heading path, revision)
TRAIN_QUERIES = [
    ("q-tr-1", "How do I save the state_dict of a model?", "eg-ckpt-save", ("Checkpoints", "Saving"), "fixture-r1"),
    ("q-tr-2", "How do I load_state_dict from a checkpoint?", "eg-ckpt-load", ("Checkpoints", "Loading"), "fixture-r1"),
    ("q-tr-3", "How do I resume training after an interruption?", "eg-ckpt-resume", ("Checkpoints", "Resuming"), "fixture-r1"),
    ("q-tr-4", "Why call model.eval() before validation?", "eg-mode-switch", ("Modes", "Switching"), "fixture-r1"),
    ("q-tr-5", "How does AutoTokenizer.from_pretrained load a tokenizer?", "eg-tok-load", ("Tokenizers", "Loading"), "fixture-v1"),
]
VALIDATION_QUERIES = [
    ("q-va-1", "How does gradient accumulation work?", "eg-mode-accum", ("Modes", "Accumulation"), "fixture-r1"),
    ("q-va-2", "What does padding do in a tokenizer batch?", "eg-tok-padding", ("Tokenizers", "Padding"), "fixture-v1"),
]
TEST_QUERIES = [
    ("q-te-1", "What is the unknown token <unk> in a tokenizer?", "eg-tok-unknown", ("Tokenizers", "Unknown"), "fixture-v1"),
    ("q-te-2", "How is a long training loop split into code fragments?", "eg-loop-code-2", ("Training loop", "Code"), "fixture-r1"),
]


class World:
    """Everything a data-layer test needs; create with `World.build(testcase)`."""

    @classmethod
    def build(cls, testcase) -> "World":
        world = cls()
        world.root = Path(tempfile.mkdtemp(prefix="devsearch-world-")).resolve()
        testcase.addCleanup(shutil.rmtree, world.root, ignore_errors=True)
        shutil.copytree(MINI_CORPUS, world.root / "corpus")
        (world.root / "index").mkdir()
        (world.root / "data").mkdir()
        world.corpus_manifest = world.root / "corpus" / "corpus-manifest.json"
        world.index_manifest = world.root / "index" / "index-manifest.json"
        build_index(index_config(), world.corpus_manifest, world.index_manifest, world.root)
        world.loaded = load_index(world.index_manifest, world.root)
        world.corpus = world.loaded.corpus
        world.merge = merge_evidence_groups(world.corpus)
        mg = world.merge.merged_of
        world.assignment = {}
        for split, groups in (("train", TRAIN_GROUPS), ("validation", VALIDATION_GROUPS), ("test", TEST_GROUPS)):
            for group in groups:
                world.assignment[mg[group]] = split
        world.queries = {"train": [], "validation": [], "test": []}
        world.qrels = {"train": [], "validation": [], "test": []}
        for split, specs in (("train", TRAIN_QUERIES), ("validation", VALIDATION_QUERIES), ("test", TEST_QUERIES)):
            for query_id, text, group, heading, revision in specs:
                world.queries[split].append(make_query(query_id, text, [group], split))
                world.qrels[split].append(world.positive(query_id, heading, revision))
        return world

    def chunk(self, heading, revision):
        return next(c for c in self.corpus.chunks if c.heading_path == heading and c.revision == revision)

    def positive(self, query_id, heading, revision, *, reviews=ACCEPTED, status="accepted", label_source="synthetic"):
        chunk = self.chunk(heading, revision)
        doc = self.corpus.doc_of(chunk)
        span = EvidenceSpan(doc.doc_id, doc.revision, doc.content_sha256, chunk.start_char + 1, min(chunk.end_char, chunk.start_char + 30))
        return Qrel(query_id, chunk.chunk_id, 1, (span,), label_source, status, tuple(reviews) if status != "unreviewed" else ())

    def chunk_ids_of(self, split):
        from devsearch.data.groups import owner_of_chunks
        owners = owner_of_chunks(self.corpus, self.merge, self.assignment)
        return {c for c, o in owners.items() if o == split}
