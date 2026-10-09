"""Candidate ordering and system scoring shared by validation, evaluation and serving."""

import math
import unittest

from devsearch.evaluation.ranking import (
    bm25_ranked_records, order_by_score, positives_by_query, ranked_record, score_ranked,
)

from dataset import World
from helpers import RootTestCase
from devsearch.data.candidates import build_candidates


class OrderingTests(unittest.TestCase):
    def test_descending_score_then_ascending_chunk_id(self):
        ordering = order_by_score(["c", "a", "b", "d"], [0.5, 0.5, 1.0, -1.0])
        self.assertEqual(ordering, [("b", 1, 1.0), ("a", 2, 0.5), ("c", 3, 0.5), ("d", 4, -1.0)])

    def test_negative_and_zero_scores_are_ordinary_scores(self):
        self.assertEqual([c for c, _, _ in order_by_score(["x", "y", "z"], [-2.0, 0.0, -1.0])], ["y", "z", "x"])

    def test_non_finite_or_mismatched_scores_are_rejected(self):
        for scores in ([0.0, math.nan], [0.0, math.inf], [0.0, -math.inf], [0.0, True], [0.0, None]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                order_by_score(["a", "b"], scores)
        with self.assertRaises(ValueError):
            order_by_score(["a", "b"], [1.0])

    def test_ranked_record_shape(self):
        record = ranked_record("q1", "cands-1", order_by_score(["a", "b"], [1.0, 2.0]))
        self.assertEqual(record, {"query_id": "q1", "candidate_set_id": "cands-1",
                                  "ranking": [{"chunk_id": "b", "rank": 1, "score": 2.0}, {"chunk_id": "a", "rank": 2, "score": 1.0}]})


class ScoringTests(RootTestCase):
    def setUp(self):
        super().setUp()
        self.world = World.build(self)
        self.queries = self.world.queries["validation"]
        self.records = build_candidates(self.world.loaded, self.queries, 50)
        self.group_of = {q.query_id: q.group_id for q in self.queries}

    def test_the_bm25_baseline_scores_over_queries_with_a_confirmed_positive(self):
        ranked = bm25_ranked_records(self.records)
        result = score_ranked(ranked, self.world.qrels["validation"], self.group_of)
        self.assertEqual((result["queries_scored"], result["queries_without_positive"]), (2, []))
        for query_id, metrics in result["per_query"].items():
            positive = next(iter(positives_by_query(self.world.qrels["validation"])[query_id]))
            ids = [c["chunk_id"] for r in self.records if r["query_id"] == query_id for c in r["candidates"]]
            self.assertEqual(metrics["mrr@10"], 1 / (ids.index(positive) + 1) if ids.index(positive) < 10 else 0.0)

    def test_queries_without_a_positive_are_reported_not_scored(self):
        qrels = [q for q in self.world.qrels["validation"] if q.query_id == "q-va-1"]
        result = score_ranked(bm25_ranked_records(self.records), qrels, self.group_of)
        self.assertEqual((result["queries_scored"], result["queries_without_positive"]), (1, ["q-va-2"]))
        with self.assertRaisesRegex(ValueError, "no query with a confirmed positive"):
            score_ranked(bm25_ranked_records(self.records), [], self.group_of)

    def test_an_unreviewed_positive_is_not_a_confirmed_positive(self):
        import dataclasses
        weak = [dataclasses.replace(q, review_status="unreviewed", reviews=()) for q in self.world.qrels["validation"]]
        self.assertEqual(positives_by_query(weak), {})

    def test_a_retrieval_miss_scores_zero_and_stays_in_the_denominator(self):
        record = self.records[0]
        positive = next(iter(positives_by_query(self.world.qrels["validation"])[record["query_id"]]))
        trimmed = dict(record, candidates=[c for c in record["candidates"] if c["chunk_id"] != positive])
        ranked = bm25_ranked_records([trimmed] + self.records[1:])
        result = score_ranked(ranked, self.world.qrels["validation"], self.group_of)
        self.assertEqual(result["per_query"][record["query_id"]], {"mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0})
        self.assertEqual(result["queries_scored"], 2)


if __name__ == "__main__":
    unittest.main()
