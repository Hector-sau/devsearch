"""Ranking metrics against hand-computed values, group aggregation and the paired group bootstrap."""

import math
import random
import unittest

from devsearch.evaluation.metrics import (
    aggregate, mrr_at, ndcg_at, paired_group_bootstrap, per_group, query_metrics, recall_at,
)


class MetricTests(unittest.TestCase):
    def test_mrr_is_the_reciprocal_rank_of_the_first_positive_within_k(self):
        self.assertEqual(mrr_at(["a", "b", "c"], {"b"}), 0.5)
        self.assertEqual(mrr_at(["a", "b", "c"], {"c", "b"}), 0.5)                  # first of several positives
        self.assertEqual(mrr_at(["a", "b", "c"], {"z"}), 0.0)                       # a retrieval miss scores zero
        ranked = [f"x{i}" for i in range(10)] + ["hit"]
        self.assertEqual(mrr_at(ranked, {"hit"}), 0.0)                              # rank 11 is outside the top 10
        self.assertEqual(mrr_at(ranked, {"hit"}, k=11), 1 / 11)

    def test_ndcg_matches_hand_computation(self):
        # One positive at rank 3: DCG = 1/log2(4) = 0.5, IDCG = 1.
        self.assertAlmostEqual(ndcg_at(["a", "b", "p"], {"p"}), 0.5)
        # Two positives at ranks 1 and 3: DCG = 1 + 0.5; IDCG = 1 + 1/log2(3).
        self.assertAlmostEqual(ndcg_at(["p1", "a", "p2"], {"p1", "p2"}), 1.5 / (1 + 1 / math.log2(3)))
        # A known positive the retriever missed still counts in the ideal: DCG 1, IDCG 1 + 1/log2(3).
        self.assertAlmostEqual(ndcg_at(["p1", "a", "b"], {"p1", "missed"}), 1 / (1 + 1 / math.log2(3)))
        self.assertEqual(ndcg_at(["a", "b"], {"p"}), 0.0)
        self.assertAlmostEqual(ndcg_at(["p"] + ["a"] * 20, {"p"}), 1.0)

    def test_ndcg_ideal_is_capped_at_k(self):
        positives = {f"p{i}" for i in range(15)}
        ranked = [f"p{i}" for i in range(15)]
        self.assertAlmostEqual(ndcg_at(ranked, positives), 1.0)                     # 15 positives, but only 10 slots

    def test_recall_is_the_fraction_of_known_positives_in_the_top_k(self):
        self.assertEqual(recall_at(["a", "b", "c"], {"b", "z"}), 0.5)
        self.assertEqual(recall_at(["a"] * 3, {"z"}), 0.0)
        self.assertEqual(recall_at([f"x{i}" for i in range(60)] + ["late"], {"late"}), 0.0)
        self.assertEqual(recall_at([f"x{i}" for i in range(60)] + ["late"], {"late"}, k=61), 1.0)

    def test_metrics_reject_queries_without_positives_instead_of_dividing_by_zero(self):
        for fn in (ndcg_at, recall_at):
            with self.assertRaises(ValueError):
                fn(["a"], set())
        self.assertEqual(mrr_at(["a"], set()), 0.0)

    def test_query_metrics_bundle(self):
        self.assertEqual(query_metrics(["a", "p"], {"p"}), {"mrr@10": 0.5, "ndcg@10": 1 / math.log2(3), "recall@50": 1.0})


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.per_query = {
            "q1": {"mrr@10": 1.0, "ndcg@10": 1.0, "recall@50": 1.0},
            "q2": {"mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0},
            "q3": {"mrr@10": 0.5, "ndcg@10": 0.5, "recall@50": 1.0},
            "q4": {"mrr@10": 0.5, "ndcg@10": 0.5, "recall@50": 1.0},
        }
        self.group_of = {"q1": "g1", "q2": "g1", "q3": "g2", "q4": "g3"}

    def test_query_macro_and_group_macro_differ_when_groups_are_unbalanced(self):
        result = aggregate(self.per_query, self.group_of, excluded_without_positive=["q9"])
        self.assertEqual(result["query_macro"]["mrr@10"], 0.5)                       # (1 + 0 + .5 + .5) / 4
        self.assertAlmostEqual(result["group_macro"]["mrr@10"], (0.5 + 0.5 + 0.5) / 3)   # g1 = .5, g2 = .5, g3 = .5
        self.assertEqual(result["group_macro"]["recall@50"], (0.5 + 1.0 + 1.0) / 3)
        self.assertEqual((result["queries_scored"], result["groups_scored"], result["queries_without_positive"]), (4, 3, ["q9"]))

    def test_groups_are_weighted_equally_whatever_their_size(self):
        per_query = {f"a{i}": {"mrr@10": 1.0, "ndcg@10": 1.0, "recall@50": 1.0} for i in range(9)}
        per_query["b"] = {"mrr@10": 0.0, "ndcg@10": 0.0, "recall@50": 0.0}
        group_of = {**{f"a{i}": "big" for i in range(9)}, "b": "small"}
        result = aggregate(per_query, group_of)
        self.assertEqual(result["query_macro"]["mrr@10"], 0.9)
        self.assertEqual(result["group_macro"]["mrr@10"], 0.5)

    def test_empty_input_is_an_error(self):
        with self.assertRaises(ValueError):
            aggregate({}, {})

    def test_per_group_means(self):
        self.assertEqual(per_group(self.per_query, self.group_of, "mrr@10"), {"g1": 0.5, "g2": 0.5, "g3": 0.5})


class BootstrapTests(unittest.TestCase):
    def test_a_constant_positive_difference_gives_a_degenerate_positive_interval(self):
        a = {f"g{i}": 0.7 for i in range(30)}
        b = {f"g{i}": 0.5 for i in range(30)}
        result = paired_group_bootstrap(a, b, repetitions=500, seed=1)
        self.assertAlmostEqual(result["observed_difference"], 0.2)
        self.assertAlmostEqual(result["interval"][0], 0.2)
        self.assertAlmostEqual(result["interval"][1], 0.2)
        self.assertTrue(result["interval_excludes_zero"])

    def test_noise_around_zero_spans_zero(self):
        rng = random.Random(5)
        a = {f"g{i}": rng.random() for i in range(40)}
        b = {f"g{i}": rng.random() for i in range(40)}
        result = paired_group_bootstrap(a, b, repetitions=2000, seed=7)
        low, high = result["interval"]
        self.assertLessEqual(low, result["observed_difference"] + 1e-12)
        self.assertGreaterEqual(high, result["observed_difference"] - 1e-12)
        self.assertEqual(result["groups"], 40)

    def test_same_seed_same_interval_different_seed_may_differ(self):
        rng = random.Random(3)
        a = {f"g{i}": rng.random() for i in range(25)}
        b = {f"g{i}": rng.random() for i in range(25)}
        first = paired_group_bootstrap(a, b, repetitions=1000, seed=11)
        self.assertEqual(first, paired_group_bootstrap(a, b, repetitions=1000, seed=11))
        self.assertNotEqual(first["interval"], paired_group_bootstrap(a, b, repetitions=1000, seed=12)["interval"])

    def test_both_systems_share_every_resample(self):
        # If the systems were resampled independently the interval for identical scores would not be exactly zero.
        scores = {f"g{i}": i / 10 for i in range(10)}
        result = paired_group_bootstrap(scores, dict(scores), repetitions=300, seed=2)
        self.assertEqual((result["observed_difference"], result["interval"]), (0.0, [0.0, 0.0]))
        self.assertFalse(result["interval_excludes_zero"])

    def test_mismatched_groups_and_bad_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_group_bootstrap({"a": 1.0}, {"b": 1.0})
        with self.assertRaises(ValueError):
            paired_group_bootstrap({}, {})
        with self.assertRaises(ValueError):
            paired_group_bootstrap({"a": 1.0}, {"a": 0.0}, repetitions=10)
        with self.assertRaises(ValueError):
            paired_group_bootstrap({"a": 1.0}, {"a": 0.0}, confidence=1.5)

    def test_the_documented_default_protocol_runs(self):
        a = {f"g{i}": 0.6 for i in range(20)}
        b = {f"g{i}": 0.55 + (i % 3) * 0.01 for i in range(20)}
        result = paired_group_bootstrap(a, b)
        self.assertEqual((result["repetitions"], result["seed"], result["confidence"]), (10_000, 20261005, 0.95))


if __name__ == "__main__":
    unittest.main()
