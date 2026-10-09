"""Meaningful ordering, provenance, fallback and fusion smoke regressions."""

import math
import unittest

from devsearch.contracts import SearchRequest
from devsearch.errors import ConfigError, ManifestError
from devsearch.evaluation.fusion import RankingPolicy
from devsearch.serving.pipeline import SearchPipeline
from devsearch.serving.release import check_smoke, record_smoke
from devsearch.serving import ui
from test_pipeline import FakeReranker, PipelineCase


class FusionOrderingTests(unittest.TestCase):
    def test_formula_uses_ranks_not_raw_score_magnitudes(self):
        p = RankingPolicy("rank_fusion")
        a = p.order(["a", "b", "c"], [0, 10, 5], [1, 2, 3])
        b = p.order(["a", "b", "c"], [-100, 99, 1], [1, 2, 3])
        self.assertEqual(a, b)
        self.assertEqual(a[0], ("b", 1, .5 / 62 + .5 / 61))
        self.assertEqual([x[0] for x in a], ["b", "a", "c"])

    def test_controls_and_fused_ties_are_deterministic(self):
        ids, logits, ranks = ["b", "a", "c"], [1, 2, 0], [1, 2, 3]
        self.assertEqual([c for c, _, _ in RankingPolicy("rank_fusion", 0).order(ids, logits, ranks)], ids)
        self.assertEqual([c for c, _, _ in RankingPolicy("rank_fusion", 1).order(ids, logits, ranks)], ["a", "b", "c"])
        self.assertEqual([c for c, _, _ in RankingPolicy("rank_fusion").order(ids, logits, ranks)], ["a", "b", "c"])

    def test_malformed_candidates_and_scores_cannot_be_fused(self):
        p = RankingPolicy("rank_fusion", 0)
        for ids, logits, ranks in [(["a", "a"], [1, 0], [1, 2]), (["a", "b"], [1, 0], [0, 1]),
                                   (["a", "b"], [1, 0], [1, 1]), (["a", "b"], [1, 0], [True, 2]),
                                   (["a", "b"], [1, math.nan], [1, 2]), (["a"], [], [1])]:
            with self.subTest(ids=ids, logits=logits, ranks=ranks), self.assertRaises(ValueError):
                p.order(ids, logits, ranks)

    def test_configs_are_strict_and_legacy_default_is_logit(self):
        self.assertEqual(RankingPolicy.from_record(), RankingPolicy())
        for bad in ({}, {"method": "rank_fusion"}, {"method": "logit", "alpha": .5}, {"method": "unknown"},
                    {"method": "rank_fusion", "alpha": float("nan"), "rank_constant": 60},
                    {"method": "rank_fusion", "alpha": True, "rank_constant": 60},
                    {"method": "rank_fusion", "alpha": .5, "rank_constant": 0}, [], "rank_fusion"):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                RankingPolicy.from_record(bad)
        self.assertEqual(RankingPolicy.from_record(RankingPolicy("rank_fusion").to_record()), RankingPolicy("rank_fusion"))


class FusionPipelineTests(PipelineCase):
    def fused(self, reranker=None):
        return SearchPipeline(self.loaded, self.limits, reranker, ranking=RankingPolicy("rank_fusion"))

    def test_online_order_equals_offline_order_and_preserves_logits(self):
        fake = FakeReranker(lambda q, p, stop: [float(i) for i in range(len(p))])
        pipeline = self.fused(fake)
        req = self.req("compare", top_k=10)
        candidates = self.loaded.retriever().retrieve(req)
        logits = [float(i) for i in range(len(candidates))]
        offline = pipeline.ranking.order([c.chunk.chunk_id for c in candidates], logits, [c.baseline_rank for c in candidates])
        response = pipeline.search(req)
        self.assertEqual([(r["chunk_id"], r["reranked_rank"], r["fusion_score"]) for r in response["reranked_results"]], offline)
        raw = {c.chunk.chunk_id: logit for c, logit in zip(candidates, logits)}
        self.assertEqual([r["rerank_logit"] for r in response["reranked_results"]], [raw[c] for c, _, _ in offline])
        self.assertEqual(response["ranking_policy"], pipeline.ranking.to_record())
        self.assertEqual(len(fake.calls[0][1]), response["candidate_count"])

    def test_baseline_filters_and_failure_fallback_are_preserved(self):
        plain = self.pipeline().search(self.req())
        self.assertEqual(self.fused().search(self.req())["baseline_results"], plain["baseline_results"])
        bad = self.fused(FakeReranker(RuntimeError("injected"))).search(self.req("compare"))
        self.assertEqual(bad["status"], "degraded")
        self.assertEqual(bad["reranked_results"], [])
        self.assertEqual(bad["baseline_results"], plain["baseline_results"])
        req = self.req("compare", sources=(plain["baseline_results"][0]["source_id"],))
        response = self.fused(FakeReranker()).search(req)
        self.assertTrue(all(r["source_id"] in req.sources for r in response["reranked_results"]))

    def test_fusion_smoke_refuses_policy_or_rank_change_even_with_tied_logits(self):
        pipeline = self.fused(FakeReranker(lambda q, p, stop: [1.] * len(p)))
        queries = [{"query": self.req().query}]
        smoke = record_smoke(pipeline, queries, self.limits)
        self.assertTrue(check_smoke(pipeline, smoke, self.limits)["passed"])
        smoke["queries"][0]["rerank"]["ids"].reverse()
        self.assertFalse(check_smoke(pipeline, smoke, self.limits)["passed"])
        smoke = record_smoke(pipeline, queries, self.limits)
        smoke["ranking_policy"]["alpha"] = .75
        self.assertFalse(check_smoke(pipeline, smoke, self.limits)["passed"])

    def test_ui_identifies_fusion_and_never_calls_fusion_score_a_logit(self):
        response = self.fused(FakeReranker()).search(self.req("compare"))
        view = ui.describe_response(response)
        self.assertIn("rank fusion", view["reranked_title"])
        self.assertIn("Rank-fusion score:", ui.row_summary(view["reranked"][0], "reranked"))

    def test_fusion_smoke_requires_real_success_not_degraded_or_all_empty_results(self):
        questions = [{"query": self.req().query}]
        for pipeline in (self.fused(), self.fused(FakeReranker(RuntimeError("failure")))):
            with self.subTest(pipeline=pipeline), self.assertRaises(ManifestError):
                record_smoke(pipeline, questions, self.limits)
        with self.assertRaises(ManifestError):
            record_smoke(self.fused(FakeReranker()), [{"query": "zzzzqqqq"}], self.limits)
        pipeline = self.fused(FakeReranker())
        smoke = record_smoke(pipeline, questions, self.limits)
        smoke["queries"][0]["rerank"] = None
        self.assertFalse(check_smoke(pipeline, smoke, self.limits)["passed"])


if __name__ == "__main__":
    unittest.main()
