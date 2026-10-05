"""SearchPipeline: baseline parity, reranking, all-or-nothing degradation, readiness and deadlines.

Fake scorers here test *failure transitions only*; they are test data, never evidence of ranking
quality. One test uses a real tiny model to prove online scoring equals the offline path.
"""

import math
import unittest

from devsearch.contracts import SearchRequest
from devsearch.errors import DeadlineExceeded, DeadlineExceededError, InvalidRequestError, NonFiniteScoreError
from devsearch.modeling.policy import LengthError
from devsearch.retrieval.index_store import build_index, load_index
from devsearch.retrieval.search import run_baseline
from devsearch.serving.admission import Admission
from devsearch.serving.pipeline import SearchPipeline
from mltest import HAVE_ML

from helpers import RootTestCase, index_config, search_limits


class FakeReranker:
    """Scripted behavior. `scores` is a callable (query, passages) -> scores, or an exception to raise."""

    model_id = "model-fake"

    def __init__(self, behavior=None, check=True):
        self.behavior, self.check, self.calls, self.checks = behavior, check, [], 0

    def score(self, query, passages, *, should_stop=None):
        self.calls.append((query, list(passages)))
        if isinstance(self.behavior, Exception):
            raise self.behavior
        if callable(self.behavior):
            return self.behavior(query, passages, should_stop)
        return [float(len(passages) - i) for i in range(len(passages))]    # reverses nothing: keeps the BM25 order

    def self_check(self):
        self.checks += 1
        if isinstance(self.check, Exception):
            raise self.check
        return {"passed": self.check}


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class PipelineCase(RootTestCase):
    def setUp(self):
        super().setUp()
        build_index(index_config(), self.corpus_manifest, self.index_manifest, self.root)
        self.loaded = load_index(self.index_manifest, self.root)
        self.limits = search_limits(candidate_k=10, top_k=3)
        self.clock = Clock()

    def pipeline(self, reranker=None):
        return SearchPipeline(self.loaded, self.limits, reranker, release_id="rel-test", clock=self.clock)

    def req(self, mode="baseline", query="How do I save and load a checkpoint state_dict?", **kw):
        return SearchRequest(query=query, mode=mode, candidate_k=kw.get("candidate_k", 10), top_k=kw.get("top_k", 3),
                             sources=kw.get("sources", ()), versions=kw.get("versions", ()))


class BaselineTests(PipelineCase):
    def test_baseline_results_equal_the_model_free_baseline(self):
        direct = run_baseline(self.loaded, self.req(), self.limits)
        document = self.pipeline().search(self.req(), request_id="r-1")
        self.assertEqual(document["baseline_results"], direct["baseline_results"])
        self.assertEqual(document["candidate_set_id"], direct["candidate_set_id"])
        self.assertEqual((document["status"], document["mode_effective"], document["model_id"]), ("ok", "baseline", None))
        self.assertEqual((document["request_id"], document["release_id"]), ("r-1", "rel-test"))
        self.assertEqual(document["reranked_results"], [])
        self.assertIsNone(document["timings_ms"]["rerank"])

    def test_a_baseline_pipeline_needs_no_reranker_and_reports_it_unready(self):
        pipeline = self.pipeline()
        self.assertTrue(pipeline.baseline_ready)
        self.assertFalse(pipeline.rerank_ready)
        self.assertFalse(pipeline.self_check())

    def test_invalid_requests_are_rejected_not_reinterpreted(self):
        with self.assertRaises(InvalidRequestError):
            self.pipeline().search(self.req(query="  "))
        with self.assertRaises(InvalidRequestError):
            self.pipeline().search(SearchRequest(query="q", mode="turbo"))
        with self.assertRaises(InvalidRequestError):
            self.pipeline().search(self.req(sources=("no-such-source",)))

    def test_no_lexical_match_is_a_valid_empty_result_and_never_calls_the_reranker(self):
        fake = FakeReranker()
        document = self.pipeline(fake).search(self.req("rerank", query="zzzzqqqq"))
        self.assertEqual(document["status"], "empty")
        self.assertEqual((document["baseline_results"], document["reranked_results"], document["warnings"]), ([], [], []))
        self.assertEqual(fake.calls, [])


class RerankTests(PipelineCase):
    def test_a_successful_rerank_is_a_permutation_of_the_same_candidates_with_ranks_and_logits(self):
        baseline = self.pipeline().search(self.req(top_k=10))
        ids = [r["chunk_id"] for r in baseline["baseline_results"]]
        self.assertGreater(len(ids), 3)
        fake = FakeReranker(lambda q, passages, stop: [float(i) for i in range(len(passages))])    # reverses the BM25 order
        document = self.pipeline(fake).search(self.req("compare", top_k=10))
        self.assertEqual(document["status"], "ok")
        self.assertEqual((document["mode_requested"], document["mode_effective"], document["model_id"]), ("compare", "compare", "model-fake"))
        reranked = document["reranked_results"]
        self.assertEqual(sorted(r["chunk_id"] for r in reranked), sorted(ids))
        self.assertEqual([r["chunk_id"] for r in reranked], ids[::-1])
        self.assertEqual([r["reranked_rank"] for r in reranked], list(range(1, len(ids) + 1)))
        self.assertTrue(all(r["rerank_logit"] is not None and r["baseline_rank"] >= 1 for r in reranked))
        self.assertEqual([r["chunk_id"] for r in document["baseline_results"]], ids)         # compare: both lists, same candidates
        self.assertTrue(all(r["reranked_rank"] is None and r["rerank_logit"] is None for r in document["baseline_results"]))
        self.assertEqual(document["candidate_set_id"], baseline["candidate_set_id"])
        self.assertIsNotNone(document["timings_ms"]["rerank"])

    def test_the_scored_text_is_exactly_the_candidate_passage_and_top_k_truncates_after_ordering(self):
        fake = FakeReranker()
        document = self.pipeline(fake).search(self.req("rerank", top_k=2))
        _, passages = fake.calls[0]
        self.assertEqual(len(passages), document["candidate_count"])
        self.assertEqual(len(document["reranked_results"]), 2)
        shown = {r["chunk_id"]: r["text"] for r in document["reranked_results"]}
        by_id = {c.chunk.chunk_id: c.chunk.text for c in self.loaded.retriever().retrieve(self.req("rerank"))}
        for chunk_id, text in shown.items():
            self.assertEqual(text, by_id[chunk_id])
            self.assertIn(text, passages)
        self.assertEqual(document["baseline_results"], [])                                     # rerank mode: only the reranked list

    def test_equal_logits_break_ties_by_ascending_chunk_id(self):
        fake = FakeReranker(lambda q, passages, stop: [1.0] * len(passages))
        document = self.pipeline(fake).search(self.req("rerank", top_k=10))
        ids = [r["chunk_id"] for r in document["reranked_results"]]
        self.assertEqual(ids, sorted(ids))

    def test_one_request_is_deterministic(self):
        fake = FakeReranker()
        pipeline = self.pipeline(fake)
        first, second = pipeline.search(self.req("rerank")), pipeline.search(self.req("rerank"))
        first["timings_ms"] = second["timings_ms"] = None
        self.assertEqual(first, second)


class DegradationTests(PipelineCase):
    def assertDegraded(self, document, code, *, model="model-fake"):
        self.assertEqual((document["status"], document["mode_effective"]), ("degraded", "baseline"))
        self.assertEqual(document["reranked_results"], [])
        self.assertTrue(all(r["reranked_rank"] is None and r["rerank_logit"] is None for r in document["baseline_results"]))
        self.assertGreater(len(document["baseline_results"]), 0)
        self.assertEqual([w["code"] for w in document["warnings"]], [code])
        self.assertEqual(document["model_id"], model)               # the attempted model stays visible for diagnosis

    def test_no_reranker_in_the_release_degrades_explicitly(self):
        document = self.pipeline().search(self.req("rerank"))
        self.assertDegraded(document, "rerank_unavailable", model=None)

    def test_a_timeout_discards_all_partial_scores(self):
        def slow(query, passages, should_stop):
            partial = [0.5]
            self.clock.now += 10
            if should_stop():
                raise DeadlineExceeded("stopped")
            return partial + [0.1] * (len(passages) - 1)
        fake = FakeReranker(slow)
        pipeline = self.pipeline(fake)
        document = pipeline.search(self.req("rerank"), deadline_at=self.clock.now + 1)
        self.assertDegraded(document, "rerank_timeout")
        self.assertTrue(pipeline.rerank_ready, "a timeout alone does not make the model unhealthy")

    def test_non_finite_scores_degrade_and_make_reranking_unready_until_a_self_check_passes(self):
        for bad in (math.nan, math.inf):
            fake = FakeReranker(lambda q, passages, stop, bad=bad: [bad] * len(passages))
            pipeline = self.pipeline(fake)
            self.assertDegraded(pipeline.search(self.req("rerank")), "rerank_nonfinite")
            self.assertFalse(pipeline.rerank_ready)
            before = len(fake.calls)
            follow_up = pipeline.search(self.req("rerank"))
            self.assertDegraded(follow_up, "rerank_not_ready")
            self.assertEqual(len(fake.calls), before, "an unready model is not invoked again, and is never reloaded")
            fake.behavior = None
            self.assertTrue(pipeline.self_check())
            self.assertTrue(pipeline.rerank_ready)
            self.assertEqual(pipeline.search(self.req("rerank"))["status"], "ok")

    def test_the_scorer_raising_a_nonfinite_error_is_handled_like_nan(self):
        fake = FakeReranker(NonFiniteScoreError("nan logit"))
        pipeline = self.pipeline(fake)
        self.assertDegraded(pipeline.search(self.req("rerank")), "rerank_nonfinite")
        self.assertFalse(pipeline.rerank_ready)

    def test_a_model_exception_degrades_without_a_stack_trace_and_marks_unready(self):
        fake = FakeReranker(RuntimeError("CUDA out of memory"))
        pipeline = self.pipeline(fake)
        document = pipeline.search(self.req("rerank"))
        self.assertDegraded(document, "rerank_failed")
        self.assertNotIn("CUDA", document["warnings"][0]["message"])
        self.assertFalse(pipeline.rerank_ready)

    def test_a_failing_self_check_keeps_reranking_unready(self):
        fake = FakeReranker(RuntimeError("boom"), check=False)
        pipeline = self.pipeline(fake)
        pipeline.search(self.req("rerank"))
        self.assertFalse(pipeline.self_check())
        fake.check = RuntimeError("device lost")
        self.assertFalse(pipeline.self_check())
        self.assertFalse(pipeline.rerank_ready)

    def test_a_wrong_number_of_scores_is_a_failure_not_a_partial_ranking(self):
        pipeline = self.pipeline(FakeReranker(lambda q, passages, stop: [1.0]))
        self.assertDegraded(pipeline.search(self.req("rerank")), "rerank_failed")
        self.assertFalse(pipeline.rerank_ready)
        self.assertEqual(pipeline.rerank_failure, "wrong_score_count")
        bad = self.pipeline(FakeReranker(lambda q, passages, stop: [None] * len(passages)))
        self.assertDegraded(bad.search(self.req("rerank")), "rerank_nonfinite")

    def test_the_pipeline_exposes_the_clock_deadlines_are_measured_on(self):
        self.assertIs(self.pipeline().clock, self.clock)

    def test_an_overlong_candidate_degrades_without_marking_the_model_unhealthy(self):
        pipeline = self.pipeline(FakeReranker(LengthError("pair has 300 tokens")))
        self.assertDegraded(pipeline.search(self.req("rerank")), "rerank_input_too_long")
        self.assertTrue(pipeline.rerank_ready)

    def test_a_deadline_already_expired_before_retrieval_is_an_error_not_a_fabricated_result(self):
        with self.assertRaises(DeadlineExceededError):
            self.pipeline().search(self.req(), deadline_at=self.clock.now - 1)

    def test_a_deadline_that_expires_during_retrieval_still_returns_the_complete_baseline_for_baseline_mode(self):
        document = self.pipeline().search(self.req(), deadline_at=self.clock.now + 1e-9)
        self.assertEqual(document["status"], "ok")


class AdmissionTests(unittest.TestCase):
    def test_one_slot_no_queue_and_release_by_the_owner_only_after_work(self):
        slot = Admission()
        self.assertFalse(slot.busy)
        self.assertTrue(slot.try_acquire())
        self.assertTrue(slot.busy)
        self.assertFalse(slot.try_acquire())
        slot.release()
        self.assertFalse(slot.busy)
        self.assertTrue(slot.try_acquire())


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class RealModelTests(PipelineCase):
    def test_online_ranking_equals_offline_scoring_and_ordering(self):
        import tempfile
        from pathlib import Path
        import shutil
        import torch
        from devsearch.evaluation.ranking import order_by_score
        from devsearch.modeling.io import export_model, load_model_bundle
        from devsearch.modeling.rerank import Reranker
        from devsearch.modeling.scoring import score_pairs
        from mlworld import MLWorld

        world = MLWorld.build_ml(self)
        model = world.model(encoder_seed=2, head_seed=9)
        out = Path(tempfile.mkdtemp(prefix="devsearch-pipe-")).resolve()
        self.addCleanup(shutil.rmtree, out, ignore_errors=True)
        examples = [(p.query_text, p.passage_text) for p in world.train_pairs[:4]]
        export_model(model, out / "m", tokenizer=world.tokenizer, policy=world.policy, provenance={}, parity_examples=examples,
                     device=torch.device("cpu"))
        loaded_model = load_model_bundle(out / "m", device=torch.device("cpu"))
        request = self.req("rerank", query="How do I save the state_dict of a model?", top_k=10)
        pipeline = SearchPipeline(world.loaded, self.limits, Reranker(loaded_model, device=torch.device("cpu"), batch_size=8), clock=self.clock)
        document = pipeline.search(request)
        candidates = world.loaded.retriever().retrieve(request)
        scores = score_pairs(model, world.tokenizer, world.policy, [(request.query, c.chunk.text) for c in candidates],
                             device=torch.device("cpu"), batch_size=8)
        expected = order_by_score([c.chunk.chunk_id for c in candidates], scores)
        self.assertEqual([r["chunk_id"] for r in document["reranked_results"]], [c for c, _, _ in expected][:10])
        self.assertEqual([r["rerank_logit"] for r in document["reranked_results"]], [s for _, _, s in expected][:10])
        self.assertTrue(pipeline.self_check())

    def test_the_inference_batch_size_is_part_of_the_scoring_identity(self):
        """Padding changes float rounding by ~1e-8, which can reorder *near-equal* logits (here: two
        identical-score candidates). Online and offline must therefore share the batch size; a different one
        stays within a documented 1e-5 tolerance and may only reorder candidates closer than that."""
        import torch
        from devsearch.evaluation.ranking import order_by_score
        from devsearch.modeling.scoring import score_pairs
        from mlworld import MLWorld

        world = MLWorld.build_ml(self)
        model = world.model(encoder_seed=2, head_seed=9)
        request = self.req("baseline", query="How do I save the state_dict of a model?", top_k=10)
        candidates = world.loaded.retriever().retrieve(request)
        pairs = [(request.query, c.chunk.text) for c in candidates]
        ids = [c.chunk.chunk_id for c in candidates]
        reference = score_pairs(model, world.tokenizer, world.policy, pairs, device=torch.device("cpu"), batch_size=3)
        self.assertEqual(reference, score_pairs(model, world.tokenizer, world.policy, pairs, device=torch.device("cpu"), batch_size=3))
        other = score_pairs(model, world.tokenizer, world.policy, pairs, device=torch.device("cpu"), batch_size=8)
        self.assertLess(max(abs(a - b) for a, b in zip(reference, other)), 1e-5)
        ranked_a = [c for c, _, _ in order_by_score(ids, reference)]
        ranked_b = [c for c, _, _ in order_by_score(ids, other)]
        by_id = dict(zip(ids, reference))
        for position, (a, b) in enumerate(zip(ranked_a, ranked_b)):
            if a != b:
                self.assertLess(abs(by_id[a] - by_id[b]), 1e-5, f"rank {position + 1}: {a} vs {b} differ by more than the tolerance")


if __name__ == "__main__":
    unittest.main()
