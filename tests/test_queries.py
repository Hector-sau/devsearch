"""Query and qrel records: strict schemas, review provenance and corpus-aware validation."""

import json
import unittest

from devsearch.data.corpus import load_corpus
from devsearch.data.queries import (
    Query, Review, derive_review_status, load_qrels, load_queries, parse_qrel, parse_query, qrel_record,
    query_record, validate_qrels, validate_queries,
)
from devsearch.errors import ManifestError

from helpers import RootTestCase

T = "2026-10-05T12:00:00Z"
GEN = {"model": "fixture-generator-1", "prompt_sha256": "a" * 64, "seed": 1}


def make_query(query_id, text, groups, split, *, group=None, parent=None, origin="synthetic", reviews=None,
               sources=(), versions=None, family="checkpoint-recovery") -> Query:
    reviews = reviews if reviews is not None else [{"reviewer_type": "agent", "reviewer_id": "fixture-reviewer", "status": "accepted", "reviewed_at": T, "revision": "wf-1"}]
    status = derive_review_status(tuple(Review(r["reviewer_type"], r["reviewer_id"], r["status"], r["reviewed_at"], r["revision"]) for r in reviews))
    return parse_query({
        "query_id": query_id, "text": text, "language": "en", "group_id": group or f"pg-{query_id}",
        "evidence_group_ids": list(groups), "parent_query_id": parent, "task_family": family, "split": split,
        "filters": {"sources": list(sources), "versions": dict(versions or {})}, "origin": origin,
        "generator": GEN if origin == "synthetic" else None, "review_status": status, "reviews": reviews,
    }, f"query {query_id}")


def query_dict(**overrides):
    base = query_record(make_query("q001", "How do I resume training?", ["eg-ckpt-resume"], "train"))
    base.update(overrides)
    return base


def qrel_dict(**overrides):
    base = {"query_id": "q001", "chunk_id": "c", "relevance": 0, "evidence_spans": [], "label_source": "synthetic",
            "review_status": "unreviewed", "reviews": []}
    base.update(overrides)
    return base


class QueryParsingTests(unittest.TestCase):
    def reject(self, fragment, **overrides):
        with self.assertRaisesRegex(ManifestError, fragment):
            parse_query(query_dict(**overrides), "q")

    def test_round_trip_and_derived_status(self):
        record = query_dict()
        self.assertEqual(query_record(parse_query(record, "q")), record)
        self.assertTrue(parse_query(record, "q").accepted)
        self.assertEqual(parse_query(query_dict(reviews=[], review_status="unreviewed"), "q").review_status, "unreviewed")

    def test_field_defects(self):
        self.reject("unrecognized field", extra=1)
        self.reject("split", split="holdout")
        self.reject("English", language="de")
        self.reject("at least one evidence group", evidence_group_ids=[])
        self.reject("duplicates", evidence_group_ids=["a", "a"])
        self.reject("trimmed", text=" padded ")
        self.reject("single-line", text="two\nlines")
        self.reject("must not be empty", text="")
        self.reject("origin", origin="llm")
        self.reject("generator", generator=None)                               # synthetic needs provenance
        self.reject("only synthetic", origin="manual")                          # manual must not carry a generator
        self.reject("unrecognized field", filters={"sources": [], "versions": {}, "x": 1})
        self.reject("expected a string", text=5)

    def test_review_status_cannot_claim_what_reviews_do_not_say(self):
        self.reject("does not match its reviews", review_status="unreviewed")      # has an accepted review
        self.reject("does not match its reviews", reviews=[], review_status="accepted")
        agent = {"reviewer_type": "agent", "reviewer_id": "r1", "status": "accepted", "reviewed_at": T, "revision": "wf-1"}
        human = {"reviewer_type": "human", "reviewer_id": "h1", "status": "rejected", "reviewed_at": T, "revision": None}
        self.assertEqual(parse_query(query_dict(reviews=[agent, human], review_status="ambiguous"), "q").review_status, "ambiguous")
        self.reject("does not match its reviews", reviews=[agent, human], review_status="accepted")

    def test_agent_and_human_reviews_are_kept_distinct(self):
        base = {"reviewer_id": "r1", "status": "accepted", "reviewed_at": T}
        self.reject("revision", reviews=[{**base, "reviewer_type": "agent", "revision": None}])       # agent needs a revision
        self.reject("human review carries no agent revision", reviews=[{**base, "reviewer_type": "human", "revision": "wf-1"}])
        self.reject("agent' or 'human", reviews=[{**base, "reviewer_type": "robot", "revision": None}])
        self.reject("not a real timestamp", reviews=[{**base, "reviewer_type": "human", "revision": None, "reviewed_at": "2026-13-45T25:61:61Z"}])
        self.reject("unexpected format", reviews=[{**base, "reviewer_type": "human", "revision": None, "reviewed_at": "yesterday"}])

    def test_derive_review_status(self):
        r = lambda s: Review("human", "h", s, T, None)
        self.assertEqual(derive_review_status(()), "unreviewed")
        self.assertEqual(derive_review_status((r("accepted"), r("accepted"))), "accepted")
        self.assertEqual(derive_review_status((r("accepted"), r("rejected"))), "ambiguous")


class QrelParsingTests(unittest.TestCase):
    SPAN = {"doc_id": "d", "revision": "r", "content_sha256": "b" * 64, "start_char": 1, "end_char": 5}

    def test_positive_needs_evidence_and_negative_cites_none(self):
        parse_qrel(qrel_dict(relevance=1, evidence_spans=[self.SPAN]), "w")
        parse_qrel(qrel_dict(relevance=0), "w")
        with self.assertRaisesRegex(ManifestError, "at least one evidence span"):
            parse_qrel(qrel_dict(relevance=1), "w")
        with self.assertRaisesRegex(ManifestError, "negative judgment cites no evidence"):
            parse_qrel(qrel_dict(relevance=0, evidence_spans=[self.SPAN]), "w")

    def test_binary_relevance_only_and_strict_fields(self):
        for bad in (2, -1, True, 0.5, "1", None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ManifestError, "binary"):
                parse_qrel(qrel_dict(relevance=bad), "w")
        with self.assertRaisesRegex(ManifestError, "label_source"):
            parse_qrel(qrel_dict(label_source="oracle"), "w")
        with self.assertRaisesRegex(ManifestError, "unrecognized"):
            parse_qrel(qrel_dict(extra=1), "w")

    def test_round_trip_and_confirmation(self):
        accepted = [{"reviewer_type": "human", "reviewer_id": "h1", "status": "accepted", "reviewed_at": T, "revision": None}]
        record = qrel_dict(relevance=1, evidence_spans=[self.SPAN], review_status="accepted", reviews=accepted)
        qrel = parse_qrel(record, "w")
        self.assertEqual(qrel_record(qrel), record)
        self.assertTrue(qrel.confirmed_positive)
        self.assertFalse(parse_qrel(qrel_dict(relevance=1, evidence_spans=[self.SPAN]), "w").confirmed_positive)   # unreviewed != confirmed


class CorpusAwareValidationTests(RootTestCase):
    def setUp(self):
        super().setUp()
        self.corpus = load_corpus(self.corpus_manifest)
        self.saving = next(c for c in self.corpus.chunks if c.heading_path == ("Checkpoints", "Saving"))
        self.doc = self.corpus.doc_of(self.saving)

    def span(self, **over):
        base = {"doc_id": self.doc.doc_id, "revision": self.doc.revision, "content_sha256": self.doc.content_sha256,
                "start_char": self.saving.start_char + 5, "end_char": self.saving.start_char + 40}
        base.update(over)
        return base

    def qrel(self, **over):
        return parse_qrel(qrel_dict(chunk_id=self.saving.chunk_id, relevance=1, evidence_spans=[self.span()], **over), "w")

    def test_valid_queries_and_qrels_pass(self):
        query = make_query("q001", "How are checkpoints saved?", ["eg-ckpt-save"], "train", sources=["fixture-torch"], versions={"fixture-torch": "fixture-r1"})
        validate_queries([query], self.corpus)
        validate_qrels([self.qrel()], {"q001": query}, self.corpus)

    def test_query_filters_and_groups_must_exist(self):
        cases = {
            "unknown source": make_query("q1", "x y", ["eg-ckpt-save"], "train", sources=["nope"]),
            "unknown version": make_query("q1", "x y", ["eg-ckpt-save"], "train", versions={"fixture-torch": "r9"}),
            "version for excluded source": make_query("q1", "x y", ["eg-ckpt-save"], "train", sources=["fixture-torch"], versions={"fixture-transformers": "fixture-v1"}),
            "unknown evidence group": make_query("q1", "x y", ["eg-ghost"], "train"),
        }
        for name, query in cases.items():
            with self.subTest(name), self.assertRaises(ManifestError):
                validate_queries([query], self.corpus)

    def test_paraphrases_share_group_and_split_with_their_parent(self):
        parent = make_query("q1", "How do I save a model?", ["eg-ckpt-save"], "train", group="pg-1")
        ok = make_query("q2", "What is the way to save a model?", ["eg-ckpt-save"], "train", group="pg-1", parent="q1")
        validate_queries([parent, ok], self.corpus)
        for bad in (make_query("q3", "Saving?", ["eg-ckpt-save"], "train", group="pg-other", parent="q1"),
                    make_query("q3", "Saving?", ["eg-ckpt-save"], "test", group="pg-1", parent="q1"),
                    make_query("q3", "Saving?", ["eg-ckpt-save"], "train", group="pg-1", parent="missing")):
            with self.subTest(bad=bad.query_id), self.assertRaises(ManifestError):
                validate_queries([parent, bad], self.corpus)

    def test_qrel_defects(self):
        query = make_query("q001", "How are checkpoints saved?", ["eg-ckpt-save"], "train")
        other = next(c for c in self.corpus.chunks if c.heading_path == ("Checkpoints", "Loading"))
        cases = {
            "unknown query": parse_qrel(qrel_dict(query_id="q404", chunk_id=self.saving.chunk_id, relevance=1, evidence_spans=[self.span()]), "w"),
            "unknown chunk": parse_qrel(qrel_dict(chunk_id="ch_" + "0" * 32, relevance=1, evidence_spans=[self.span()]), "w"),
            "wrong document hash": self.qrel() if False else parse_qrel(qrel_dict(chunk_id=self.saving.chunk_id, relevance=1, evidence_spans=[self.span(content_sha256="c" * 64)]), "w"),
            "span before the chunk": parse_qrel(qrel_dict(chunk_id=self.saving.chunk_id, relevance=1, evidence_spans=[self.span(start_char=0, end_char=10)]), "w"),
            "span past the chunk": parse_qrel(qrel_dict(chunk_id=self.saving.chunk_id, relevance=1, evidence_spans=[self.span(end_char=self.saving.end_char + 5)]), "w"),
            "positive outside the query groups": parse_qrel(qrel_dict(chunk_id=other.chunk_id, relevance=1, evidence_spans=[
                self.span(start_char=other.start_char + 1, end_char=other.start_char + 20)]), "w"),
        }
        for name, qrel in cases.items():
            with self.subTest(name), self.assertRaises(ManifestError):
                validate_qrels([qrel], {"q001": query}, self.corpus)
        # pooled-review positives may come from other groups (they are found by reviewers, not generated)
        pooled = parse_qrel(qrel_dict(chunk_id=other.chunk_id, relevance=1, label_source="pooled-review", evidence_spans=[
            self.span(start_char=other.start_char + 1, end_char=other.start_char + 20)]), "w")
        validate_qrels([pooled], {"q001": query}, self.corpus)


class FileLoadingTests(RootTestCase):
    def write(self, name, records):
        path = self.root / name
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
        return path

    def test_queries_must_be_sorted_unique_and_of_one_split(self):
        a, b = query_dict(query_id="q1"), query_dict(query_id="q2")
        self.assertEqual([q.query_id for q in load_queries(self.write("ok.jsonl", [a, b]))], ["q1", "q2"])
        with self.assertRaisesRegex(ManifestError, "sorted"):
            load_queries(self.write("unsorted.jsonl", [b, a]))
        with self.assertRaisesRegex(ManifestError, "sorted"):
            load_queries(self.write("dup.jsonl", [a, a]))
        with self.assertRaisesRegex(ManifestError, "one split only"):
            load_queries(self.write("mixed.jsonl", [a, query_dict(query_id="q2", split="test")]))

    def test_qrels_must_be_sorted_and_unique(self):
        a, b = qrel_dict(chunk_id="c1"), qrel_dict(chunk_id="c2")
        self.assertEqual(len(load_qrels(self.write("q.jsonl", [a, b]))), 2)
        with self.assertRaisesRegex(ManifestError, "sorted"):
            load_qrels(self.write("q2.jsonl", [b, a]))


if __name__ == "__main__":
    unittest.main()
