"""Blinded review pool, judgment bank and the two test seals."""

import json
import math
import unittest

from devsearch.artifacts.canonical import sha256_file
from devsearch.data.candidates import build_candidates
from devsearch.data.review import (
    ReviewResponse, assert_blinded, build_review_pool, hidden_records, item_id_for, parse_response, postrank_universe,
    prejudged_universe, qrels_diff, request_records, resolve_bank,
)
from devsearch.data.seals import (
    build_final_seal, build_initial_seal, load_bank, verify_bank_covers, verify_permutation_output,
    write_bank, write_review_pool,
)
from devsearch.errors import ManifestError

from dataset import World
from helpers import RootTestCase
from test_queries import make_query

STAMP = "2026-10-05T12:00:00Z"


class ReviewCase(RootTestCase):
    def setUp(self):
        super().setUp()
        self.world = World.build(self)
        self.loaded = self.world.loaded
        self.queries = self.world.queries["test"]
        self.by_id = {q.query_id: q for q in self.queries}
        self.qrels_v1 = self.world.qrels["test"]
        self.candidates = build_candidates(self.loaded, self.queries, 50)
        self.universe = prejudged_universe(self.loaded, self.queries, self.candidates, self.qrels_v1)
        self.pool = build_review_pool("prejudged_fixed_candidates_v1", "salt-for-tests", self.universe)
        self.chunks = {c.chunk_id: c for c in self.loaded.corpus.chunks}
        self.positive_pairs = {(q.query_id, q.chunk_id) for q in self.qrels_v1}

    def oracle(self, item, *, role="primary", reviewer=("agent", "oracle-1", "wf-1"), relevance=None, resolved=True):
        """A simulated reviewer that knows the known positives; used only to exercise mechanics."""
        kind, ident, revision = reviewer
        is_positive = (item.query_id, item.chunk_id) in self.positive_pairs
        value = (1 if is_positive else 0) if relevance is None else relevance
        length = len(self.chunks[item.chunk_id].text)
        spans = [(0, min(20, length))] if resolved and value == 1 else []
        return ReviewResponse(item.item_id, role, value if resolved else None, resolved, tuple(spans), kind, ident,
                              revision if kind == "agent" else None, STAMP)

    def full_responses(self, **kw):
        return [self.oracle(item, **kw) for item in self.pool.items]


class UniverseAndBlindingTests(ReviewCase):
    def test_universe_is_candidates_plus_known_positives_without_inserting_into_retrieval(self):
        for record in self.candidates:
            ids = {c["chunk_id"] for c in record["candidates"]}
            for chunk_id in ids:
                self.assertIn((record["query_id"], chunk_id), self.universe)
        for query_id, chunk_id in self.positive_pairs:
            entry = self.universe[(query_id, chunk_id)]
            self.assertTrue(entry["known_positive"])
            record = next(r for r in self.candidates if r["query_id"] == query_id)
            in_candidates = chunk_id in {c["chunk_id"] for c in record["candidates"]}
            self.assertEqual(entry["candidate_rank"] is not None, in_candidates)
        extra = [k for k, v in self.universe.items() if v["candidate_rank"] is None]
        for query_id, chunk_id in extra:                                      # known positives that retrieval missed stay outside the lists
            record = next(r for r in self.candidates if r["query_id"] == query_id)
            self.assertNotIn(chunk_id, {c["chunk_id"] for c in record["candidates"]})

    def test_a_known_positive_outside_the_query_filters_is_an_error(self):
        odd = make_query("q-te-f", "What is the unknown token?", ["eg-tok-unknown"], "test", sources=["fixture-torch"])
        qrel = self.world.positive("q-te-f", ("Tokenizers", "Unknown"), "fixture-v1")        # a transformers passage
        records = build_candidates(self.loaded, [odd], 50)
        with self.assertRaisesRegex(ManifestError, "outside the query's filtered corpus"):
            prejudged_universe(self.loaded, [odd], records, [qrel])

    def test_requests_are_blinded_and_carry_source_evidence(self):
        records = request_records(self.pool, self.loaded, self.by_id)
        self.assertEqual(len(records), len(self.pool.items))
        assert_blinded(records)
        sample = records[0]
        self.assertEqual(set(sample), {"item_id", "query", "passage"})
        self.assertEqual(set(sample["passage"]), {"title", "section", "text", "code_fragment", "source_url", "source_location"})
        text = json.dumps(records)
        self.assertNotIn("bm25", text)
        for item in self.pool.items[:5]:
            self.assertNotIn(item.chunk_id, text)
            self.assertNotIn(item.query_id, text)

    def test_assert_blinded_catches_every_leaky_key(self):
        for key in ("rank", "bm25_score", "label", "known_positive", "systems", "candidate_rank", "chunk_id"):
            with self.subTest(key=key), self.assertRaisesRegex(ManifestError, "leaks"):
                assert_blinded([{"item_id": "x", "passage": {"nested": [{key: 1}]}}])

    def test_order_and_ids_do_not_reveal_rank(self):
        records = request_records(self.pool, self.loaded, self.by_id)
        provenance = {i.item_id: i.provenance for i in self.pool.items}
        ranks = [provenance[r["item_id"]]["candidate_rank"] or 999 for r in records]
        self.assertNotEqual(ranks, sorted(ranks), "presentation order must not follow the candidate ranking")
        other = build_review_pool("prejudged_fixed_candidates_v1", "a-different-salt", self.universe)
        self.assertNotEqual([i.item_id for i in self.pool.items], [i.item_id for i in other.items])
        self.assertEqual(self.pool.items[0].item_id, item_id_for("salt-for-tests", self.pool.items[0].query_id, self.pool.items[0].chunk_id))
        self.assertNotEqual(self.pool.pool_id, other.pool_id)

    def test_hidden_provenance_is_kept_apart_and_complete(self):
        hidden = hidden_records(self.pool)
        self.assertEqual(len(hidden), len(self.pool.items))
        self.assertTrue(all({"item_id", "query_id", "chunk_id", "provenance"} <= set(h) for h in hidden))
        self.assertTrue(any(h["provenance"]["known_positive"] for h in hidden))

    def test_pool_validation(self):
        with self.assertRaisesRegex(ManifestError, "unknown evaluation protocol"):
            build_review_pool("adhoc", "salt-long-enough", self.universe)
        with self.assertRaisesRegex(ManifestError, "at least 8"):
            build_review_pool("postrank_pool_v1", "short", self.universe)

    def test_postrank_universe_pools_top_k_of_every_system_plus_positives(self):
        ranked = {
            "bm25": [{"query_id": r["query_id"], "ranking": [{"chunk_id": c["chunk_id"], "rank": c["rank"], "score": c["bm25_score"]} for c in r["candidates"]]} for r in self.candidates],
            "model": [{"query_id": r["query_id"], "ranking": [{"chunk_id": c["chunk_id"], "rank": i + 1, "score": 1.0} for i, c in enumerate(reversed(r["candidates"]))]} for r in self.candidates],
        }
        universe = postrank_universe(ranked, self.qrels_v1, depth=3)
        for record in self.candidates:
            top = [c["chunk_id"] for c in record["candidates"][:3]]
            for chunk_id in top:
                self.assertEqual(universe[(record["query_id"], chunk_id)]["systems"]["bm25"]["rank"], top.index(chunk_id) + 1)
        for query_id, chunk_id in self.positive_pairs:
            self.assertTrue(universe[(query_id, chunk_id)]["known_positive"])


class ResponseParsingTests(ReviewCase):
    def record(self, **over):
        item = self.pool.items[0]
        base = {"item_id": item.item_id, "role": "primary", "relevance": 0, "status": "resolved", "evidence_spans": [],
                "reviewer": {"type": "agent", "id": "r1", "revision": "wf-1"}, "reviewed_at": STAMP, "notes": None}
        base.update(over)
        return base

    def parse(self, **over):
        lengths = {i.item_id: len(self.chunks[i.chunk_id].text) for i in self.pool.items}
        return parse_response(self.record(**over), "r", lengths)

    def test_valid_responses(self):
        self.assertEqual(self.parse().relevance, 0)
        self.assertEqual(self.parse(relevance=1, evidence_spans=[{"start": 0, "end": 5}]).spans, ((0, 5),))
        self.assertIsNone(self.parse(status="unresolved", relevance=None).relevance)

    def test_defects(self):
        cases = {
            "unknown item": dict(item_id="it_" + "0" * 24),
            "bad role": dict(role="boss"),
            "bad status": dict(status="maybe"),
            "resolved without relevance": dict(relevance=None),
            "positive without spans": dict(relevance=1),
            "negative with spans": dict(relevance=0, evidence_spans=[{"start": 0, "end": 3}]),
            "span outside the passage": dict(relevance=1, evidence_spans=[{"start": 0, "end": 10**6}]),
            "unresolved with a label": dict(status="unresolved", relevance=1),
            "human with an agent revision": dict(reviewer={"type": "human", "id": "h", "revision": "wf"}),
            "agent without revision": dict(reviewer={"type": "agent", "id": "a", "revision": None}),
            "unknown reviewer type": dict(reviewer={"type": "crowd", "id": "a", "revision": None}),
            "bool relevance": dict(relevance=True),
        }
        for name, over in cases.items():
            with self.subTest(name), self.assertRaises(ManifestError):
                self.parse(**over)


class BankTests(ReviewCase):
    def test_a_complete_agreeing_review_resolves_every_item(self):
        bank = resolve_bank(self.pool, self.full_responses(), self.loaded)
        self.assertTrue(bank.ready)
        self.assertEqual(len(bank.qrels), len(self.pool.items))
        self.assertEqual(sum(q.relevance for q in bank.qrels), len(self.positive_pairs))
        self.assertEqual(bank.reviewer_counts, {"agent": len(self.pool.items), "human": 0})
        for qrel in bank.qrels:
            self.assertEqual(qrel.label_source, "pooled-review")
            if qrel.relevance == 1:
                chunk = self.chunks[qrel.chunk_id]
                self.assertTrue(all(chunk.start_char <= s.start_char < s.end_char <= chunk.end_char for s in qrel.evidence_spans))

    def test_missing_unresolved_and_disagreeing_items_are_never_turned_into_label_zero(self):
        responses = self.full_responses()
        gone = responses.pop(0)
        responses[0] = self.oracle(self.pool.items[1], resolved=False)
        item2 = self.pool.items[2]
        responses[1] = self.oracle(item2, relevance=1 - (1 if (item2.query_id, item2.chunk_id) in self.positive_pairs else 0))
        responses.append(self.oracle(item2, reviewer=("human", "h1", None), relevance=1 if (item2.query_id, item2.chunk_id) in self.positive_pairs else 0))
        bank = resolve_bank(self.pool, responses, self.loaded)
        reasons = dict(bank.unresolved)
        self.assertEqual(reasons[gone.item_id], "missing")
        self.assertEqual(reasons[self.pool.items[1].item_id], "unresolved")
        self.assertEqual(reasons[item2.item_id], "disagreement without adjudication")
        self.assertFalse(bank.ready)
        self.assertEqual(len(bank.qrels), len(self.pool.items) - 3)

    def test_an_adjudicator_settles_a_disagreement_and_both_reviewer_types_are_counted(self):
        item = self.pool.items[0]
        truth = 1 if (item.query_id, item.chunk_id) in self.positive_pairs else 0
        responses = self.full_responses()[1:] + [
            self.oracle(item, relevance=1 - truth), self.oracle(item, reviewer=("human", "h1", None), relevance=truth),
            self.oracle(item, role="adjudicator", reviewer=("human", "h2", None), relevance=truth),
        ]
        bank = resolve_bank(self.pool, responses, self.loaded)
        self.assertTrue(bank.ready)
        judged = next(q for q in bank.qrels if (q.query_id, q.chunk_id) == (item.query_id, item.chunk_id))
        self.assertEqual(judged.relevance, truth)
        self.assertEqual([r.reviewer_type for r in judged.reviews], ["human"])        # only the adjudicator's review decides
        self.assertEqual(bank.reviewer_counts["human"], 2)
        self.assertGreater(bank.reviewer_counts["agent"], 0)

    def test_two_adjudicators_are_an_error(self):
        item = self.pool.items[0]
        responses = self.full_responses() + [self.oracle(item, role="adjudicator"), self.oracle(item, role="adjudicator", reviewer=("human", "h", None))]
        with self.assertRaisesRegex(ManifestError, "more than one adjudicator"):
            resolve_bank(self.pool, responses, self.loaded)

    def test_diff_records_changes_without_deleting_queries(self):
        flipped = (self.qrels_v1[0].query_id, self.qrels_v1[0].chunk_id)
        responses = []
        for item in self.pool.items:
            relevance = 0 if (item.query_id, item.chunk_id) == flipped else None
            responses.append(self.oracle(item, relevance=relevance))
        bank = resolve_bank(self.pool, responses, self.loaded)
        diff = qrels_diff(self.qrels_v1, bank.qrels, [q.query_id for q in self.queries])
        self.assertEqual(diff["v1_positives"], 2)
        self.assertEqual(diff["v1_positives_confirmed"], 1)
        self.assertEqual(diff["v1_positives_reversed"], [f"{flipped[0]}:{flipped[1]}"])
        self.assertEqual(diff["queries_losing_all_positives"], [flipped[0]])
        self.assertEqual(diff["queries_total"], 2)                                        # the query stays in the record

    def test_new_reviewer_found_positives_are_reported(self):
        extra = next(i for i in self.pool.items if (i.query_id, i.chunk_id) not in self.positive_pairs)
        responses = [self.oracle(i, relevance=1 if i is extra else None) for i in self.pool.items]
        diff = qrels_diff(self.qrels_v1, resolve_bank(self.pool, responses, self.loaded).qrels, [q.query_id for q in self.queries])
        self.assertEqual(diff["new_positives"], [f"{extra.query_id}:{extra.chunk_id}"])


class FilesAndSealsTests(ReviewCase):
    def setUp(self):
        super().setUp()
        self.review_dir = self.world.root / "data" / "review"
        self.review_dir.mkdir()
        self.bank = resolve_bank(self.pool, self.full_responses(), self.loaded)

    def test_review_pool_files_are_written_once_with_a_private_provenance_file(self):
        manifest = write_review_pool(self.review_dir, self.pool, self.loaded, self.by_id)
        self.assertEqual(manifest["counts"]["items"], len(self.pool.items))
        self.assertEqual(manifest["files"]["review-request.jsonl"]["sha256"], sha256_file(self.review_dir / "review-request.jsonl"))
        request_text = (self.review_dir / "review-request.jsonl").read_text()
        hidden_text = (self.review_dir / "hidden-provenance.jsonl").read_text()
        self.assertNotIn("candidate_rank", request_text)
        self.assertIn("candidate_rank", hidden_text)
        self.assertNotIn("salt-for-tests", request_text)
        with self.assertRaises(FileExistsError):
            write_review_pool(self.review_dir, self.pool, self.loaded, self.by_id)

    def test_unresolved_items_block_sealing_and_leave_no_bank_files(self):
        incomplete = resolve_bank(self.pool, self.full_responses()[1:], self.loaded)
        with self.assertRaisesRegex(ManifestError, "sealing is blocked"):
            write_bank(self.review_dir, self.pool, incomplete, self.qrels_v1, self.loaded, self.queries)
        self.assertEqual(list(self.review_dir.iterdir()), [])

    def seal_bank(self):
        manifest = write_bank(self.review_dir, self.pool, self.bank, self.qrels_v1, self.loaded, self.queries)
        return manifest, self.review_dir / "judgment-bank-manifest.json"

    def test_bank_round_trip_and_tamper_detection(self):
        manifest, path = self.seal_bank()
        loaded_manifest, qrels = load_bank(path, self.loaded, self.queries)
        self.assertEqual(loaded_manifest["counts"], {"items": len(self.pool.items), "positives": 2, "queries": 2})
        self.assertEqual(qrels, self.bank.qrels)
        qrels_path = self.review_dir / "qrels-v2.jsonl"
        qrels_path.write_bytes(qrels_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ManifestError, "SHA-256"):
            load_bank(path, self.loaded, self.queries)

    def test_bank_must_cover_every_candidate_and_positive(self):
        counts = verify_bank_covers(self.bank.qrels, self.candidates, self.qrels_v1)
        self.assertEqual(counts["required_pairs"], len(self.pool.items))
        with self.assertRaisesRegex(ManifestError, "does not cover 1 required pair"):
            verify_bank_covers(self.bank.qrels[1:], self.candidates, self.qrels_v1)

    def ranked(self, reverse=False):
        out = []
        for record in self.candidates:
            order = list(reversed(record["candidates"])) if reverse else record["candidates"]
            out.append({"query_id": record["query_id"], "ranking": [{"chunk_id": c["chunk_id"], "rank": i + 1, "score": float(len(order) - i)} for i, c in enumerate(order)]})
        return out

    def test_system_outputs_must_be_permutations_of_the_frozen_candidates(self):
        verify_permutation_output(self.ranked(), self.candidates, "bm25")
        verify_permutation_output(self.ranked(reverse=True), self.candidates, "model")
        def broken(mutate):
            records = self.ranked()
            mutate(records)
            return records
        cases = {
            "dropped candidate": broken(lambda r: r[0]["ranking"].pop()),
            "duplicate candidate": broken(lambda r: r[0]["ranking"].append(dict(r[0]["ranking"][0]))),
            "foreign passage": broken(lambda r: r[0]["ranking"][0].update(chunk_id=self.pool.items[-1].chunk_id if self.pool.items[-1].chunk_id not in {c["chunk_id"] for c in self.candidates[0]["candidates"]} else "ch_" + "0" * 32)),
            "rank gap": broken(lambda r: r[0]["ranking"][1].update(rank=9)),
            "NaN score": broken(lambda r: r[0]["ranking"][0].update(score=float("nan"))),
            "infinite score": broken(lambda r: r[0]["ranking"][0].update(score=math.inf)),
            "missing query": broken(lambda r: r.pop()),
        }
        for name, records in cases.items():
            with self.subTest(name), self.assertRaises(ManifestError):
                verify_permutation_output(records, self.candidates, "x")

    def initial_seal(self, bank_sha):
        return build_initial_seal(
            protocol="prejudged_fixed_candidates_v1", loaded=self.loaded, test_queries_sha256="a" * 64,
            test_group_ids=["eg-tok-unknown", "eg-loop-code-2"], candidates_manifest_sha256="b" * 64, qrels_v1_sha256="c" * 64,
            bank_manifest_sha256=bank_sha, eligibility_rules={"denominator": "queries with >=1 confirmed positive"})

    def test_the_prejudged_protocol_seals_the_bank_with_the_initial_inputs(self):
        with self.assertRaisesRegex(ManifestError, "seals the complete judgment bank"):
            self.initial_seal(None)
        manifest, _ = self.seal_bank()
        seal = self.initial_seal(manifest["manifest_sha256"])
        self.assertEqual((seal["stage"], seal["bank_manifest_sha256"]), (1, manifest["manifest_sha256"]))

    def test_final_seal_binds_locked_systems_to_the_unchanged_bank(self):
        manifest, _ = self.seal_bank()
        initial = self.initial_seal(manifest["manifest_sha256"])
        outputs = {"bm25": ("1" * 64, self.ranked()), "full": ("2" * 64, self.ranked(reverse=True))}
        final = build_final_seal(initial_seal=initial, selection_id="selection-1", outputs=outputs, candidate_records=self.candidates,
                                 bank_manifest=manifest, bank_qrels=self.bank.qrels, qrels_v1=self.qrels_v1)
        self.assertEqual((final["stage"], final["selection_id"], sorted(final["systems"])), (2, "selection-1", ["bm25", "full"]))
        self.assertEqual(final["initial_seal_sha256"], initial["manifest_sha256"])
        self.assertEqual(final["qrels_v2_sha256"], manifest["files"]["qrels-v2.jsonl"]["sha256"])
        self.assertEqual(final["coverage"]["required_pairs"], len(self.pool.items))

    def test_final_seal_refuses_a_changed_bank_missing_coverage_or_bad_outputs(self):
        manifest, _ = self.seal_bank()
        initial = self.initial_seal(manifest["manifest_sha256"])
        good = {"bm25": ("1" * 64, self.ranked())}
        kwargs = dict(initial_seal=initial, selection_id="s", candidate_records=self.candidates, bank_manifest=manifest, bank_qrels=self.bank.qrels, qrels_v1=self.qrels_v1)
        with self.assertRaisesRegex(ManifestError, "bank changed after the initial seal"):
            build_final_seal(**{**kwargs, "outputs": good, "initial_seal": self.initial_seal("f" * 64)})
        with self.assertRaisesRegex(ManifestError, "does not cover"):
            build_final_seal(**{**kwargs, "outputs": good, "bank_qrels": self.bank.qrels[1:]})
        with self.assertRaisesRegex(ManifestError, "permutation"):
            bad = self.ranked(); bad[0]["ranking"].pop()
            build_final_seal(**{**kwargs, "outputs": {"bm25": ("1" * 64, bad)}})
        with self.assertRaisesRegex(ManifestError, "no locked system outputs"):
            build_final_seal(**{**kwargs, "outputs": {}})


if __name__ == "__main__":
    unittest.main()
