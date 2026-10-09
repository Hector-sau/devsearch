"""Baseline response document: contract fields, source mapping, determinism, validation."""

import json
import unittest

from devsearch.contracts import SearchRequest
from devsearch.errors import InvalidRequestError
from devsearch.retrieval.index_store import build_index, load_index
from devsearch.retrieval.search import run_baseline, validate_request

from helpers import RootTestCase, index_config, request, search_limits

RESULT_FIELDS = {
    "chunk_id", "source_id", "source_url", "revision", "title", "section", "text", "code_fragment",
    "source_location", "baseline_rank", "reranked_rank", "bm25_score", "rerank_logit",
}


class ValidationTests(unittest.TestCase):
    def test_valid_requests_pass(self):
        validate_request(request("checkpoint"), search_limits())
        validate_request(request("x", candidate_k=100, top_k=100), search_limits())
        validate_request(request("x" * 1000), search_limits())

    def test_invalid_requests_are_rejected_with_readable_errors(self):
        limits = search_limits()
        cases = {
            "empty": (request(""), "must not be empty"),
            "whitespace": (request(" \n\t "), "must not be empty"),
            "overlong": (request("x" * 1001), "limit is 1000"),
            "top_k zero": (request("q", top_k=0), "top_k"),
            "top_k above candidate_k": (request("q", candidate_k=3, top_k=4), "top_k"),
            "candidate_k above max": (request("q", candidate_k=101), "candidate_k"),
            "bool top_k": (SearchRequest(query="q", top_k=True), "top_k"),
            "rerank mode": (SearchRequest(query="q", mode="rerank"), "not available"),
            "compare mode": (SearchRequest(query="q", mode="compare"), "not available"),
        }
        for name, (bad, fragment) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(InvalidRequestError, fragment):
                validate_request(bad, limits)

    def test_error_carries_the_documented_code(self):
        self.assertEqual(InvalidRequestError.code, "invalid_request")


class ResponseTests(RootTestCase):
    def setUp(self):
        super().setUp()
        build_index(index_config(), self.corpus_manifest, self.index_manifest, self.root)
        self.loaded = load_index(self.index_manifest, self.root)

    def run_search(self, query, **kwargs):
        return run_baseline(self.loaded, request(query, **kwargs), search_limits())

    def test_response_contract_for_a_successful_search(self):
        response = self.run_search("how do I load_state_dict from a checkpoint")
        self.assertEqual(response["status"], "ok")
        self.assertEqual((response["mode_requested"], response["mode_effective"]), ("baseline", "baseline"))
        self.assertEqual((response["corpus_id"], response["index_id"]), (self.loaded.corpus.corpus_id, self.loaded.index_id))
        self.assertIsNone(response["model_id"])
        self.assertEqual(response["reranked_results"], [])
        self.assertEqual(response["warnings"], [])
        self.assertEqual((response["candidate_k"], response["top_k"]), (50, 5))
        self.assertEqual(len(response["baseline_results"]), 5)
        self.assertGreater(response["candidate_count"], 5)                     # depth is recorded apart from display count
        self.assertGreater(response["timings_ms"]["retrieval"], 0)
        self.assertIsNone(response["timings_ms"]["rerank"])
        for rank, result in enumerate(response["baseline_results"], start=1):
            self.assertEqual(set(result), RESULT_FIELDS)
            self.assertEqual(result["baseline_rank"], rank)
            self.assertIsNone(result["reranked_rank"])                         # BM25 ranks are never copied into rerank fields
            self.assertIsNone(result["rerank_logit"])
        json.dumps(response, allow_nan=False)                                  # fully serializable

    def test_top_result_carries_source_attribution_and_location(self):
        top = self.run_search("load_state_dict")["baseline_results"][0]
        self.assertEqual(top["source_id"], "fixture-torch")
        self.assertEqual(top["revision"], "fixture-r1")
        self.assertEqual(top["title"], "Saving and resuming checkpoints")
        self.assertEqual(top["section"], "Checkpoints > Loading")
        self.assertEqual(top["source_url"], "https://docs.fixture.invalid/torch/checkpoints")
        self.assertEqual(top["source_location"], [{"path": "docs/checkpoints.md", "start_line": 8, "end_line": 11}])
        self.assertFalse(top["code_fragment"])
        self.assertIn("model.load_state_dict(torch.load(path))", top["text"])

    def test_code_fragments_are_flagged_and_keep_exact_text(self):
        results = self.run_search("scheduler.step", candidate_k=50, top_k=5)["baseline_results"]
        fragment = next(r for r in results if r["code_fragment"])
        self.assertTrue(fragment["text"].startswith("    scheduler.step()"))     # indentation preserved byte for byte
        self.assertEqual(fragment["source_location"], [{"path": "docs/long-loop.md", "start_line": 14, "end_line": 19}])

    def test_text_is_returned_unescaped_because_escaping_is_a_rendering_concern(self):
        top = self.run_search("<unk>")["baseline_results"][0]
        self.assertIn("<unk>", top["text"])

    def test_zero_match_is_a_valid_empty_result(self):
        response = self.run_search("kubernetes ingress controller")
        self.assertEqual(response["status"], "empty")
        self.assertEqual((response["baseline_results"], response["reranked_results"], response["candidate_count"]), ([], [], 0))

    def test_filters_are_echoed_and_applied(self):
        response = self.run_search("model", sources=("fixture-transformers",), versions=(("fixture-transformers", "fixture-v1"),))
        self.assertEqual(response["filters"], {"sources": ["fixture-transformers"], "versions": {"fixture-transformers": "fixture-v1"}})
        self.assertTrue(response["baseline_results"])
        self.assertEqual({r["revision"] for r in response["baseline_results"]}, {"fixture-v1"})

    def test_invalid_filters_surface_as_invalid_requests(self):
        with self.assertRaises(InvalidRequestError):
            self.run_search("model", sources=("not-a-source",))

    def test_candidate_set_id_is_deterministic_and_sensitive(self):
        base = self.run_search("checkpoint")["candidate_set_id"]
        self.assertEqual(base, self.run_search("checkpoint")["candidate_set_id"])
        different = {
            self.run_search("checkpoint resume")["candidate_set_id"],
            self.run_search("checkpoint", candidate_k=3, top_k=3)["candidate_set_id"],
            self.run_search("checkpoint", sources=("fixture-torch",))["candidate_set_id"],
        }
        self.assertNotIn(base, different)
        self.assertEqual(len(different), 3)
        self.assertRegex(base, r"^cands-[0-9a-f]{32}$")

    def test_top_k_only_limits_display_not_the_candidate_set(self):
        shown_2 = self.run_search("model", top_k=2)
        shown_5 = self.run_search("model", top_k=5)
        self.assertEqual(len(shown_2["baseline_results"]), 2)
        self.assertEqual(shown_2["candidate_set_id"], shown_5["candidate_set_id"])
        self.assertEqual(shown_2["baseline_results"], shown_5["baseline_results"][:2])


if __name__ == "__main__":
    unittest.main()
