"""Evidence-group merging, deterministic split assignment and the independent cross-split audit."""

import dataclasses
import unittest

from devsearch.data.corpus import ChunkSpan, DocInput, MANIFEST_NAME, load_corpus, write_corpus
from devsearch.data.groups import (
    assign_groups, audit_assignment, jaccard, merge_evidence_groups, normalize_text, owner_of_chunks, shingles,
)
from devsearch.data.sourcemap import SourceSegment
from devsearch.errors import ManifestError

from helpers import RootTestCase, synthetic_corpus
from test_queries import make_query


class MergeTests(RootTestCase):
    def test_fixture_merges_exact_duplicate_notes_and_nothing_else(self):
        from devsearch.data.corpus import load_corpus
        corpus = load_corpus(self.corpus_manifest)
        merge = merge_evidence_groups(corpus)
        self.assertEqual(merge.merged_of["eg-note-torch"], merge.merged_of["eg-note-hf"])
        self.assertEqual(merge.merges_by_rule.get("exact_duplicate"), 1)
        singles = [m for m in merge.members.values() if len(m) == 1]
        self.assertEqual(len(merge.members), len({c.evidence_group_id for c in corpus.chunks}) - 1)
        self.assertEqual(len(singles), len(merge.members) - 1)

    def test_every_rule_merges_independently(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau"
        near = text.replace("tau", "upsilon")                       # 1 of 19 words changed: Jaccard well above 0.9? check below
        self.assertGreaterEqual(jaccard(shingles(text, 2), shingles(near, 2)), 0.85)
        corpus = synthetic_corpus(self, [
            ("g-exact-1", "the quick brown fox"), ("g-exact-2", "The  quick   BROWN fox"),
            ("g-near-1", text), ("g-near-2", near),
            ("g-solo", "something unrelated entirely different words here"),
        ])
        merge = merge_evidence_groups(corpus, near_duplicate_threshold=0.8, shingle_size=2)
        self.assertEqual(merge.merged_of["g-exact-1"], merge.merged_of["g-exact-2"])
        self.assertEqual(merge.merged_of["g-near-1"], merge.merged_of["g-near-2"])
        self.assertNotEqual(merge.merged_of["g-near-1"], merge.merged_of["g-exact-1"])
        self.assertEqual(len(merge.members[merge.merged_of["g-solo"]]), 1)
        self.assertEqual(merge.merges_by_rule, {"exact_duplicate": 1, "near_duplicate": 1})

    def test_overlapping_spans_of_one_document_merge_transitively(self):
        directory = self.root / "overlap"
        directory.mkdir()
        text = "".join(f"{i:03d}," for i in range(40))            # 160 chars, no repeated substrings
        doc = DocInput("d:one", "s", "T", "r1", None, "l", "p", text + "\n",
                       (SourceSegment(0, 160, "a.md", False, start_line=1, end_line=1),),
                       (ChunkSpan(0, 60, "g-a"), ChunkSpan(40, 100, "g-b"), ChunkSpan(80, 140, "g-c"), ChunkSpan(130, 160, "g-d")))
        write_corpus(directory, [doc], chunker_version="c")
        merge = merge_evidence_groups(load_corpus(directory / MANIFEST_NAME), near_duplicate_threshold=1.0)
        self.assertEqual(len(set(merge.merged_of.values())), 1)           # a-b, b-c, c-d chain into one group
        self.assertEqual(merge.merges_by_rule, {"overlap": 3})              # nothing else may have merged them

    def test_explicit_links_merge_and_unknown_links_fail(self):
        corpus = synthetic_corpus(self, [("g-1", "aaa bbb ccc"), ("g-2", "ddd eee fff"), ("g-3", "ggg hhh iii")])
        merge = merge_evidence_groups(corpus, extra_links=[("g-1", "g-3")])
        self.assertEqual(merge.merged_of["g-1"], merge.merged_of["g-3"])
        self.assertNotEqual(merge.merged_of["g-1"], merge.merged_of["g-2"])
        with self.assertRaisesRegex(ManifestError, "unknown evidence group"):
            merge_evidence_groups(corpus, extra_links=[("g-1", "nope")])
        with self.assertRaisesRegex(ManifestError, "between 0.5 and 1.0"):
            merge_evidence_groups(corpus, near_duplicate_threshold=0.2)

    def test_merged_ids_are_stable_content_identities(self):
        items = [("g-1", "aaa bbb ccc"), ("g-2", "aaa bbb ccc"), ("g-3", "xxx yyy zzz")]
        a = merge_evidence_groups(synthetic_corpus(self, items))
        b = merge_evidence_groups(synthetic_corpus(self, items))
        self.assertEqual(a.merged_of, b.merged_of)
        self.assertRegex(a.merged_of["g-1"], r"^mg-[0-9a-f]{16}$")

    def test_helpers(self):
        self.assertEqual(normalize_text("  A\tb\n C "), "a b c")
        self.assertEqual(jaccard(frozenset(), frozenset()), 1.0)
        self.assertEqual(jaccard(frozenset({1, 2}), frozenset({2, 3})), 1 / 3)


class AssignmentTests(RootTestCase):
    def corpus(self, n=60):
        return synthetic_corpus(self, [(f"g-{i:03d}", f"unique passage number {i} " + " ".join(f"w{i}x{j}" for j in range(6))) for i in range(n)])

    def test_assignment_is_reproducible_seeded_and_input_order_independent(self):
        corpus = self.corpus()
        merge = merge_evidence_groups(corpus)
        targets = {"train": 0.6, "validation": 0.2, "test": 0.2}
        first = assign_groups(merge, targets=targets, seed=7)
        self.assertEqual(first, assign_groups(merge, targets=targets, seed=7))
        self.assertNotEqual(first, assign_groups(merge, targets=targets, seed=8))
        reordered = dataclasses.replace(merge, members=dict(reversed(list(merge.members.items()))))
        self.assertEqual(first, assign_groups(reordered, targets=targets, seed=7))

    def test_shares_are_met_closely_and_every_group_is_assigned(self):
        merge = merge_evidence_groups(self.corpus(100))
        assignment = assign_groups(merge, targets={"train": 0.6, "validation": 0.2, "test": 0.2}, seed=1)
        self.assertEqual(set(assignment), set(merge.members))
        counts = {s: sum(1 for v in assignment.values() if v == s) for s in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 60, "validation": 20, "test": 20})
        self.assertNotIn("development", set(assignment.values()))

    def test_invalid_targets_are_rejected(self):
        merge = merge_evidence_groups(self.corpus(5))
        for bad in ({}, {"train": 0}, {"holdout": 1}, {"train": -1}):
            with self.subTest(bad=bad), self.assertRaises(ManifestError):
                assign_groups(merge, targets=bad, seed=1)

    def test_duplicate_passages_always_land_in_the_same_split(self):
        items = []
        for i in range(30):
            items += [(f"g-{i:02d}-a", f"same words repeated block {i} alpha beta"), (f"g-{i:02d}-b", f"same words repeated block {i} alpha beta")]
        corpus = synthetic_corpus(self, items)
        merge = merge_evidence_groups(corpus)
        assignment = assign_groups(merge, targets={"train": 0.5, "test": 0.5}, seed=3)
        owner = owner_of_chunks(corpus, merge, assignment)
        by_text = {}
        for chunk in corpus.chunks:
            by_text.setdefault(chunk.text, set()).add(owner[chunk.chunk_id])
        self.assertTrue(all(len(owners) == 1 for owners in by_text.values()))
        self.assertEqual({"train", "test"}, {o for s in by_text.values() for o in s})


class AuditTests(RootTestCase):
    def setUp(self):
        super().setUp()
        self.corpus = synthetic_corpus(self, [
            ("g-a1", "alpha shared paragraph one two three four five six"), ("g-a2", "Alpha  shared paragraph one two three four five six"),
            ("g-b", "bravo something completely different here today"), ("g-c", "charlie yet another distinct passage about tokens"),
        ])
        self.merge = merge_evidence_groups(self.corpus)
        self.mg_a, self.mg_b, self.mg_c = (self.merge.merged_of[g] for g in ("g-a1", "g-b", "g-c"))

    def audit(self, assignment, queries=(), corpus=None):
        return audit_assignment(corpus or self.corpus, self.merge, assignment, queries)

    def test_a_proper_assignment_passes_and_counts(self):
        assignment = {self.mg_a: "train", self.mg_b: "validation", self.mg_c: "test"}
        report = self.audit(assignment)
        self.assertTrue(report.ok, report.violations)
        self.assertEqual((report.counts["groups_train"], report.counts["chunks_train"], report.counts["chunks_test"]), (1, 2, 1))

    def test_split_exact_duplicates_are_caught_even_if_the_merger_were_bypassed(self):
        # Pretend the merger failed: split the two duplicate groups apart.
        broken = dataclasses.replace(self.merge, merged_of={**self.merge.merged_of, "g-a2": "mg-forged"},
                                     members={**self.merge.members, "mg-forged": ("g-a2",)}, chunks_of={**self.merge.chunks_of, "mg-forged": ()})
        assignment = {self.mg_a: "train", "mg-forged": "test", self.mg_b: "train", self.mg_c: "train"}
        report = audit_assignment(self.corpus, broken, assignment)
        self.assertFalse(report.ok)
        self.assertTrue(any("exact-duplicate" in v for v in report.violations), report.violations)
        self.assertTrue(any("near-duplicate" in v for v in report.violations), report.violations)

    def test_incomplete_or_foreign_assignments_are_violations_not_crashes(self):
        report = self.audit({self.mg_a: "train", self.mg_b: "test"})
        self.assertFalse(report.ok)
        self.assertTrue(any("1 missing" in v for v in report.violations), report.violations)
        report = self.audit({self.mg_a: "train", self.mg_b: "test", self.mg_c: "test", "mg-stranger": "train"})
        self.assertTrue(any("1 unknown" in v for v in report.violations), report.violations)
        with self.assertRaisesRegex(ManifestError, "does not cover merged group"):
            owner_of_chunks(self.corpus, self.merge, {self.mg_a: "train"})

    def test_record_owner_must_agree_with_the_assignment(self):
        owned = synthetic_corpus(self, [("g-a1", "alpha shared paragraph one two", "train"), ("g-b", "bravo something different", "train")])
        merge = merge_evidence_groups(owned)
        mg = {g: merge.merged_of[g] for g in ("g-a1", "g-b")}
        ok = audit_assignment(owned, merge, {mg["g-a1"]: "train", mg["g-b"]: "train"})
        self.assertTrue(ok.ok, ok.violations)
        bad = audit_assignment(owned, merge, {mg["g-a1"]: "test", mg["g-b"]: "train"})
        self.assertTrue(any("record says 'train' but assignment says 'test'" in v for v in bad.violations), bad.violations)

    def test_query_split_must_match_the_owner_of_its_evidence(self):
        assignment = {self.mg_a: "train", self.mg_b: "validation", self.mg_c: "test"}
        good = make_query("q1", "how does bravo work", ["g-b"], "validation")
        wrong = make_query("q2", "what is charlie about", ["g-c"], "train")
        self.assertTrue(self.audit(assignment, [good]).ok)
        report = self.audit(assignment, [good, wrong])
        self.assertTrue(any("query q2" in v and "owned by 'test'" in v for v in report.violations), report.violations)

    def test_paraphrases_must_stay_together_and_identical_text_cannot_cross_splits(self):
        assignment = {self.mg_a: "train", self.mg_b: "validation", self.mg_c: "test"}
        q1 = make_query("q1", "how do tokens work", ["g-b"], "validation", group="pg-1")
        q2 = make_query("q2", "how do tokens work", ["g-c"], "test", group="pg-1")
        report = self.audit(assignment, [q1, q2])
        self.assertTrue(any("paraphrase group 'pg-1' spans splits" in v for v in report.violations), report.violations)
        self.assertTrue(any("identical question text" in v for v in report.violations), report.violations)


if __name__ == "__main__":
    unittest.main()
