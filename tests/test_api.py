"""HTTP layer: error shapes, admission, degradation, readiness and the thin UI helpers (docs/04 "Search API").

The service tests drive the real FastAPI app in-process through `httpx.ASGITransport` (no socket), over a real
tiny index with a scripted fake reranker. Fake rerankers test *failure transitions only*; they are test data,
never evidence of ranking quality. Threads are gated on events with bounded waits, and the slot/health checks
use those events rather than sleeps, so a failing test can never hang the suite.
"""

import asyncio
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from devsearch.contracts import SearchRequest
from devsearch.errors import BusyError, DeadlineExceeded, IndexUnavailableError, ManifestError
from devsearch.retrieval.index_store import build_index, load_index
from devsearch.retrieval.search import run_baseline
from devsearch.serving import ui
from devsearch.serving.pipeline import SearchPipeline
from mltest import HAVE_SERVE, requires_serve

from helpers import SRC, RootTestCase, index_config, search_limits

if HAVE_SERVE:
    import httpx
    from devsearch.serving import api as api_module
    from devsearch.serving.api import SearchService, create_app, serve_forever

QUERY = "How do I save and load a checkpoint state_dict?"


class FakeReranker:
    """Scripted behavior: a callable (query, passages, should_stop) -> scores, or an exception to raise."""

    model_id = "model-fake"

    def __init__(self, behavior=None, check=True):
        self.behavior, self.check, self.calls, self.threads = behavior, check, [], []

    def score(self, query, passages, *, should_stop=None):
        self.threads.append(threading.current_thread())
        self.calls.append((query, list(passages)))
        if isinstance(self.behavior, Exception):
            raise self.behavior
        if callable(self.behavior):
            return self.behavior(query, passages, should_stop)
        return [float(len(passages) - i) for i in range(len(passages))]       # keeps the BM25 order

    def self_check(self):
        self.threads.append(threading.current_thread())
        if isinstance(self.check, Exception):
            raise self.check
        return {"passed": self.check}


class Gate:
    """A reranker behavior that blocks its worker until the test opens the gate (bounded, so it cannot hang)."""

    def __init__(self, timeout=10.0):
        self.entered, self.go, self.timeout = threading.Event(), threading.Event(), timeout

    def __call__(self, query, passages, should_stop):
        self.entered.set()
        if not self.go.wait(self.timeout):
            raise AssertionError("the test never opened the gate")
        return [float(len(passages) - i) for i in range(len(passages))]


class StepClock:
    """A monotonic clock a test can move; with `step` it advances by that much on every reading."""

    def __init__(self, now=100.0, step=0.0):
        self.now, self.step = now, step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


class ScriptedPipeline(SearchPipeline):
    """The real pipeline, whose `search` a test can replace to inject what a fake reranker cannot (index errors, a stuck worker)."""

    script = None

    def search(self, request, *, deadline_at=None, request_id=None):
        if self.script is not None:
            return self.script(request, deadline_at, request_id)
        return super().search(request, deadline_at=deadline_at, request_id=request_id)


def reverse_scores(query, passages, should_stop):
    return [float(i) for i in range(len(passages))]


def raising(error):
    def script(*args):
        raise error
    return script


@requires_serve
class ApiCase(RootTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        build_index(index_config(), self.corpus_manifest, self.index_manifest, self.root)
        self.loaded = load_index(self.index_manifest, self.root)
        self.limits = search_limits(candidate_k=10, top_k=3)
        self.clock = StepClock()

    def make(self, reranker=None, *, limits=None, clock=None, deadline_s=5.0, http_grace_s=1.0, default_mode="baseline", **release):
        clock = clock or self.clock
        pipeline = ScriptedPipeline(self.loaded, limits or self.limits, reranker, release_id="rel-test", clock=clock)
        service = SearchService(
            pipeline, deadline_s=deadline_s, http_grace_s=http_grace_s, clock=clock,
            release={"status": "evaluated" if reranker else "baseline", "release_id": "rel-test", "default_mode": default_mode, **release})
        self.addCleanup(service.close)
        return service, create_app(service)

    def gated(self, **kwargs):
        gate = Gate()
        self.addCleanup(gate.go.set)                  # whatever happens, no worker stays blocked after the test
        fake = FakeReranker(gate)
        return gate, fake, *self.make(fake, **kwargs)

    async def call(self, app, method, url, **kwargs):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", trust_env=False) as client:
            return await client.request(method, url, **kwargs)

    async def search(self, app, **payload):
        return await self.call(app, "POST", "/search", json=payload)

    async def until(self, predicate, what, timeout=5.0):
        end = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > end:
                self.fail(f"timed out waiting for {what}")
            await asyncio.sleep(0.005)

    def assertError(self, response, status, code, retryable):
        self.assertEqual(response.status_code, status, response.text)
        body = response.json(parse_constant=self.fail)
        self.assertEqual(set(body), {"request_id", "error"}, "an error body is never shaped like a result")
        self.assertEqual(set(body["error"]), {"code", "message", "retryable"})
        self.assertEqual((body["error"]["code"], body["error"]["retryable"]), (code, retryable))
        self.assertTrue(isinstance(body["error"]["message"], str) and body["error"]["message"])
        self.assertEqual(body["request_id"], response.headers["x-request-id"])
        return body

    def assertNoPartialRerank(self, document):
        self.assertEqual(document["reranked_results"], [])
        self.assertTrue(all(r["reranked_rank"] is None and r["rerank_logit"] is None for r in document["baseline_results"]))

    def assertDegraded(self, response, code, *, model="model-fake"):
        self.assertEqual(response.status_code, 200, response.text)
        document = response.json(parse_constant=self.fail)
        self.assertEqual((document["status"], document["mode_effective"]), ("degraded", "baseline"))
        self.assertEqual([w["code"] for w in document["warnings"]], [code])
        self.assertGreater(len(document["baseline_results"]), 0)
        self.assertNoPartialRerank(document)
        self.assertEqual(document["model_id"], model)
        return document


class SuccessTests(ApiCase):
    async def test_baseline_search_returns_the_pipeline_document_with_a_request_id(self):
        service, app = self.make()
        response = await self.search(app, query=QUERY)
        self.assertEqual(response.status_code, 200)
        document = response.json(parse_constant=self.fail)
        direct = run_baseline(self.loaded, SearchRequest(query=QUERY, candidate_k=10, top_k=3), self.limits)
        self.assertEqual(document["baseline_results"], direct["baseline_results"])
        self.assertEqual(document["candidate_set_id"], direct["candidate_set_id"])
        self.assertEqual((document["status"], document["mode_requested"], document["mode_effective"], document["model_id"]),
                         ("ok", "baseline", "baseline", None))
        self.assertEqual(document["reranked_results"], [])
        self.assertEqual((document["release_id"], document["corpus_id"], document["index_id"]),
                         ("rel-test", self.loaded.corpus.corpus_id, self.loaded.index_id))
        self.assertRegex(document["request_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(document["request_id"], response.headers["x-request-id"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertFalse(service.busy)

    async def test_request_ids_are_generated_per_request_and_not_taken_from_the_client(self):
        _, app = self.make()
        first = await self.call(app, "POST", "/search", json={"query": QUERY}, headers={"X-Request-ID": "client-chosen"})
        second = await self.call(app, "POST", "/search", json={"query": QUERY})
        self.assertNotEqual(first.json()["request_id"], "client-chosen")
        self.assertNotEqual(first.json()["request_id"], second.json()["request_id"])

    async def test_rerank_mode_returns_only_the_reranked_list_with_ranks_and_logits(self):
        _, app = self.make(FakeReranker(reverse_scores))
        baseline = (await self.search(app, query=QUERY, candidate_k=10, top_k=10)).json()
        ids = [r["chunk_id"] for r in baseline["baseline_results"]]
        response = await self.search(app, query=QUERY, mode="rerank", candidate_k=10, top_k=10)
        self.assertEqual(response.status_code, 200)
        document = response.json(parse_constant=self.fail)
        self.assertEqual((document["status"], document["mode_requested"], document["mode_effective"], document["model_id"]),
                         ("ok", "rerank", "rerank", "model-fake"))
        self.assertEqual([r["chunk_id"] for r in document["reranked_results"]], ids[::-1])
        self.assertEqual([r["reranked_rank"] for r in document["reranked_results"]], list(range(1, len(ids) + 1)))
        self.assertTrue(all(isinstance(r["rerank_logit"], float) for r in document["reranked_results"]))
        self.assertEqual(document["baseline_results"], [])
        self.assertIsNotNone(document["timings_ms"]["rerank"])

    async def test_compare_mode_returns_both_lists_over_the_same_candidates(self):
        _, app = self.make(FakeReranker(reverse_scores))
        response = await self.search(app, query=QUERY, mode="compare", candidate_k=10, top_k=10)
        document = response.json(parse_constant=self.fail)
        self.assertEqual((response.status_code, document["status"], document["mode_effective"]), (200, "ok", "compare"))
        baseline_ids = [r["chunk_id"] for r in document["baseline_results"]]
        self.assertEqual(sorted(baseline_ids), sorted(r["chunk_id"] for r in document["reranked_results"]))
        self.assertEqual([r["chunk_id"] for r in document["reranked_results"]], baseline_ids[::-1])
        self.assertTrue(all(r["reranked_rank"] is None and r["rerank_logit"] is None for r in document["baseline_results"]))
        self.assertTrue(all(r["baseline_rank"] >= 1 and r["reranked_rank"] >= 1 for r in document["reranked_results"]))

    async def test_omitted_fields_take_the_configured_defaults(self):
        _, app = self.make()
        document = (await self.search(app, query=QUERY)).json()
        self.assertEqual((document["candidate_k"], document["top_k"], len(document["baseline_results"])), (10, 3, 3))
        self.assertLessEqual(document["candidate_count"], 10)
        explicit = (await self.search(app, query=QUERY, candidate_k=5, top_k=2)).json()
        self.assertEqual((explicit["candidate_k"], explicit["top_k"], len(explicit["baseline_results"])), (5, 2, 2))

    async def test_the_default_mode_comes_from_the_release(self):
        _, app = self.make(FakeReranker(), default_mode="compare")
        document = (await self.search(app, query=QUERY)).json()
        self.assertEqual((document["mode_requested"], document["mode_effective"]), ("compare", "compare"))
        self.assertTrue(document["baseline_results"] and document["reranked_results"])
        _, baseline_app = self.make()
        self.assertEqual((await self.search(baseline_app, query=QUERY)).json()["mode_requested"], "baseline")

    async def test_source_and_version_filters_apply_before_ranking_and_versions_are_sorted_by_source(self):
        _, app = self.make()
        document = (await self.search(app, query="tokenizer", sources=["fixture-transformers"],
                                      versions={"fixture-transformers": "fixture-v2"})).json()
        self.assertGreater(len(document["baseline_results"]), 0)
        self.assertEqual({(r["source_id"], r["revision"]) for r in document["baseline_results"]}, {("fixture-transformers", "fixture-v2")})
        self.assertEqual(document["filters"], {"sources": ["fixture-transformers"], "versions": {"fixture-transformers": "fixture-v2"}})
        one = (await self.search(app, query="model", sources=["fixture-torch", "fixture-transformers"],
                                 versions={"fixture-transformers": "fixture-v2", "fixture-torch": "fixture-r1"})).json()
        two = (await self.search(app, query="model", sources=["fixture-torch", "fixture-transformers"],
                                 versions={"fixture-torch": "fixture-r1", "fixture-transformers": "fixture-v2"})).json()
        self.assertEqual(list(one["filters"]["versions"]), ["fixture-torch", "fixture-transformers"])
        self.assertEqual(one["candidate_set_id"], two["candidate_set_id"])

    async def test_a_filter_with_no_lexical_match_is_a_valid_empty_result(self):
        _, app = self.make(FakeReranker())
        response = await self.search(app, query="zzzzqqqq", mode="compare")
        document = response.json()
        self.assertEqual((response.status_code, document["status"]), (200, "empty"))
        self.assertEqual((document["baseline_results"], document["reranked_results"], document["warnings"]), ([], [], []))

    async def test_configured_default_filters_apply_only_when_the_field_is_omitted(self):
        limits = search_limits(candidate_k=10, top_k=3, sources=("fixture-transformers",))
        _, app = self.make(limits=limits)
        defaulted = (await self.search(app, query="model")).json()
        self.assertEqual(defaulted["filters"]["sources"], ["fixture-transformers"])
        self.assertTrue(all(r["source_id"] == "fixture-transformers" for r in defaulted["baseline_results"]))
        self.assertEqual((await self.search(app, query="model", sources=[])).json()["filters"]["sources"], [])

    async def test_responses_are_not_cached_and_carry_no_cors_headers(self):
        _, app = self.make()
        response = await self.call(app, "POST", "/search", json={"query": QUERY}, headers={"Origin": "https://evil.invalid"})
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertNotIn("access-control-allow-origin", response.headers)
        preflight = await self.call(app, "OPTIONS", "/search", headers={"Origin": "https://evil.invalid", "Access-Control-Request-Method": "POST"})
        self.assertEqual(preflight.status_code, 405)
        self.assertNotIn("access-control-allow-origin", preflight.headers)
        for path in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual((await self.call(app, "GET", path)).status_code, 404, path)


class ValidationTests(ApiCase):
    JSON = {"Content-Type": "application/json"}

    async def assertInvalid(self, app, **kwargs):
        response = await self.call(app, "POST", "/search", **kwargs)
        body = self.assertError(response, 422, "invalid_request", False)
        self.assertNotIn("detail", body)                     # never pydantic's default 422 shape
        return body

    async def test_malformed_json_is_an_invalid_request_not_a_framework_error(self):
        service, app = self.make()
        bodies = {"not json": b"{nope", "empty": b"", "truncated": b'{"query": "x"', "bad utf-8": b'{"query": "\xff"}',
                  "array": b"[]", "string": b'"query"', "number": b"5", "null": b"null", "deep nesting": b"[" * 5000 + b"]" * 5000}
        for name, body in bodies.items():
            with self.subTest(name):
                await self.assertInvalid(app, content=body, headers=self.JSON)
        self.assertFalse(service.busy)

    async def test_wrong_types_unknown_fields_and_nulls_are_rejected_not_coerced(self):
        _, app = self.make()
        bad = {
            "missing query": {}, "query int": {"query": 5}, "query null": {"query": None}, "query list": {"query": ["a"]},
            "sources string": {"query": "q", "sources": "fixture-torch"}, "sources int item": {"query": "q", "sources": [1]},
            "sources null": {"query": "q", "sources": None}, "versions list": {"query": "q", "versions": ["a", "b"]},
            "versions int": {"query": "q", "versions": {"fixture-torch": 1}}, "top_k string": {"query": "q", "top_k": "3"},
            "top_k float": {"query": "q", "top_k": 3.0}, "top_k bool": {"query": "q", "top_k": True},
            "candidate_k null": {"query": "q", "candidate_k": None}, "candidate_k string": {"query": "q", "candidate_k": "10"},
            "mode int": {"query": "q", "mode": 5}, "mode null": {"query": "q", "mode": None},
            "unknown field": {"query": "q", "extra": 1}, "typo": {"query": "q", "topk": 3},
        }
        for name, payload in bad.items():
            with self.subTest(name):
                await self.assertInvalid(app, json=payload)

    async def test_values_are_validated_by_the_pipeline_and_never_clamped(self):
        service, app = self.make(FakeReranker())
        cases = {
            "empty query": {"query": ""}, "blank query": {"query": "   "}, "overlong query": {"query": "x" * 1001},
            "top_k zero": {"query": "q", "top_k": 0}, "top_k negative": {"query": "q", "top_k": -1},
            "candidate_k over the limit": {"query": "q", "candidate_k": 101, "top_k": 3},
            "top_k above candidate_k": {"query": "q", "candidate_k": 5, "top_k": 6},
            "unknown mode": {"query": "q", "mode": "turbo"}, "empty mode": {"query": "q", "mode": ""},
            "unknown source": {"query": "q", "sources": ["no-such-source"]},
            "unknown version": {"query": "q", "sources": ["fixture-torch"], "versions": {"fixture-torch": "nope"}},
            "repeated source": {"query": "q", "sources": ["fixture-torch", "fixture-torch"]},
            "version for an excluded source": {"query": "q", "sources": ["fixture-torch"], "versions": {"fixture-transformers": "fixture-v1"}},
        }
        for name, payload in cases.items():
            with self.subTest(name):
                body = await self.assertInvalid(app, json=payload)
                self.assertNotIn("model-fake", json.dumps(body))
        self.assertEqual(service.pipeline.reranker.calls, [], "an invalid request never reaches the model")
        self.assertFalse(service.busy, "a rejected request frees the slot")

    async def test_only_a_json_content_type_is_accepted_so_a_cross_site_form_post_cannot_start_a_search(self):
        service, app = self.make()
        raw = json.dumps({"query": QUERY}).encode()
        for headers in ({}, {"Content-Type": "text/plain"}, {"Content-Type": "application/x-www-form-urlencoded"}):
            with self.subTest(headers):
                body = await self.assertInvalid(app, content=raw, headers=headers)
                self.assertIn("application/json", body["error"]["message"])
        ok = await self.call(app, "POST", "/search", content=raw, headers={"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(ok.status_code, 200)

    async def test_an_oversized_body_is_rejected_before_it_is_parsed(self):
        _, app = self.make()
        body = await self.assertInvalid(app, json={"query": "x" * 200_000})
        self.assertIn("exceeds", body["error"]["message"])

        async def chunks():                                  # no Content-Length: the cap must hold for streamed bodies too
            for _ in range(200):
                yield b" " * 1024
        body = await self.assertInvalid(app, content=chunks(), headers=self.JSON)
        self.assertIn("exceeds", body["error"]["message"])


class FailureMappingTests(ApiCase):
    async def test_an_expired_deadline_before_a_baseline_exists_is_504(self):
        service, app = self.make(clock=StepClock(step=10.0), deadline_s=1.0)        # every clock reading is 10 s later
        response = await self.search(app, query=QUERY)
        self.assertError(response, 504, "deadline_exceeded", True)
        self.assertFalse(service.busy)

    async def test_the_deadline_is_computed_at_admission_from_the_service_clock(self):
        seen = []
        service, app = self.make(deadline_s=7.5)
        service.pipeline.script = lambda request, deadline_at, request_id: seen.append((deadline_at, request_id)) or {"status": "ok"}
        response = await self.search(app, query=QUERY)
        self.assertEqual(seen, [(107.5, response.json()["request_id"])])

    async def test_index_unavailable_is_503_without_detail(self):
        service, app = self.make()
        service.pipeline.script = raising(IndexUnavailableError("corrupt postings in /srv/secret/index.json"))
        with self.assertLogs("devsearch.serving", "ERROR") as logs:
            response = await self.search(app, query=QUERY)
        self.assertError(response, 503, "index_unavailable", False)
        self.assertNotIn("secret", response.text)
        self.assertIn("/srv/secret/index.json", "\n".join(logs.output), "the detail goes to the local log")
        self.assertFalse(service.busy)

    async def test_unexpected_errors_are_500_with_a_generic_message_and_no_leak(self):
        service, app = self.make()
        for error in (RuntimeError("boom: /home/u/secret.txt token=abc"), KeyError("secret-key"), ValueError("secret value"),
                      ManifestError("secret manifest path")):
            with self.subTest(type(error).__name__):
                service.pipeline.script = raising(error)
                with self.assertLogs("devsearch.serving", "ERROR") as logs:
                    response = await self.search(app, query=QUERY)
                self.assertError(response, 500, "internal_error", False)
                for leaked in ("secret", "boom", "token", "Traceback", "Error"):
                    self.assertNotIn(leaked, response.text)
                self.assertIn("secret", "\n".join(logs.output))
                self.assertFalse(service.busy, "a failed search frees the slot")
        service.pipeline.script = None
        self.assertEqual((await self.search(app, query=QUERY)).status_code, 200)

    async def test_a_document_that_cannot_be_serialized_strictly_is_500_not_invalid_json(self):
        service, app = self.make()
        service.pipeline.script = lambda *a: {"status": "ok", "baseline_results": [{"bm25_score": math.nan}]}
        with self.assertLogs("devsearch.serving", "ERROR"):
            response = await self.search(app, query=QUERY)
        self.assertError(response, 500, "internal_error", False)

    async def test_a_worker_that_raises_timeout_error_is_a_500_not_an_http_deadline(self):
        service, app = self.make()
        service.pipeline.script = raising(TimeoutError("socket timed out in a model call"))
        with self.assertLogs("devsearch.serving", "ERROR"):
            response = await self.search(app, query=QUERY)
        self.assertError(response, 500, "internal_error", False)


class DegradationTests(ApiCase):
    async def test_a_model_exception_degrades_explicitly_and_hides_the_exception(self):
        _, app = self.make(FakeReranker(RuntimeError("CUDA out of memory")))
        response = await self.search(app, query=QUERY, mode="rerank")
        document = self.assertDegraded(response, "rerank_failed")
        self.assertNotIn("CUDA", response.text)
        self.assertEqual(document["mode_requested"], "rerank")

    async def test_non_finite_scores_degrade_and_no_nan_reaches_the_wire(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                _, app = self.make(FakeReranker(lambda q, passages, stop, bad=bad: [bad] * len(passages)))
                response = await self.search(app, query=QUERY, mode="compare")
                json.loads(response.text, parse_constant=self.fail)
                self.assertDegraded(response, "rerank_nonfinite")

    async def test_a_scorer_timeout_discards_every_partial_score(self):
        def slow(query, passages, should_stop):
            partial = [0.9]
            self.clock.now += 60
            if should_stop():
                raise DeadlineExceeded("stopped between batches")
            return partial + [0.1] * (len(passages) - 1)
        service, app = self.make(FakeReranker(slow), deadline_s=5.0)
        self.assertDegraded(await self.search(app, query=QUERY, mode="rerank"), "rerank_timeout")
        self.assertTrue(service.pipeline.rerank_ready, "a timeout alone does not make the model unhealthy")

    async def test_a_failed_compare_leaves_only_the_baseline_list(self):
        _, app = self.make(FakeReranker(RuntimeError("device lost")))
        document = self.assertDegraded(await self.search(app, query=QUERY, mode="compare"), "rerank_failed")
        self.assertEqual(len(document["baseline_results"]), 3)

    async def test_a_baseline_release_degrades_a_rerank_request_explicitly(self):
        _, app = self.make()
        self.assertDegraded(await self.search(app, query=QUERY, mode="rerank"), "rerank_unavailable", model=None)

    async def test_baseline_mode_is_unaffected_by_a_failed_model(self):
        _, app = self.make(FakeReranker(RuntimeError("boom")))
        await self.search(app, query=QUERY, mode="rerank")
        response = await self.search(app, query=QUERY)
        self.assertEqual((response.status_code, response.json()["status"], response.json()["warnings"]), (200, "ok", []))


class ReadinessTests(ApiCase):
    async def test_rerank_readiness_stays_false_after_a_model_exception_until_a_self_check_succeeds(self):
        fake = FakeReranker(RuntimeError("CUDA out of memory"))
        service, app = self.make(fake, default_mode="rerank")
        self.assertEqual((await self.call(app, "GET", "/ready?mode=rerank")).status_code, 200)
        self.assertDegraded(await self.search(app, query=QUERY, mode="rerank"), "rerank_failed")

        health = (await self.call(app, "GET", "/health")).json()
        self.assertEqual((health["baseline_ready"], health["rerank_ready"]), (True, False))
        not_ready = await self.call(app, "GET", "/ready?mode=rerank")
        self.assertError(not_ready, 503, "rerank_not_ready", True)
        self.assertEqual((await self.call(app, "GET", "/ready?mode=baseline")).status_code, 200)
        self.assertEqual((await self.call(app, "GET", "/ready")).status_code, 503, "no mode means the release's default (rerank)")

        calls = len(fake.calls)
        self.assertDegraded(await self.search(app, query=QUERY, mode="rerank"), "rerank_not_ready")
        self.assertEqual(len(fake.calls), calls, "an unready model is not invoked again, and never reloaded by a request")
        self.assertEqual((await self.search(app, query=QUERY, mode="baseline")).json()["status"], "ok")

        fake.check = False                                    # a failing self-check keeps it unready, still HTTP 200
        failed = await self.call(app, "POST", "/self-check")
        self.assertEqual((failed.status_code, failed.json()), (200, {"rerank_ready": False}))
        self.assertEqual((await self.call(app, "GET", "/ready?mode=rerank")).status_code, 503)

        fake.behavior, fake.check = None, True
        recovered = await self.call(app, "POST", "/self-check")
        self.assertEqual((recovered.status_code, recovered.json()), (200, {"rerank_ready": True}))
        self.assertEqual((await self.call(app, "GET", "/ready?mode=rerank")).status_code, 200)
        self.assertEqual((await self.call(app, "GET", "/ready")).status_code, 200)
        self.assertEqual((await self.search(app, query=QUERY, mode="rerank")).json()["status"], "ok")
        self.assertFalse(service.busy)

    async def test_a_baseline_release_is_baseline_ready_and_rerank_unready(self):
        _, app = self.make()
        self.assertEqual((await self.call(app, "GET", "/ready?mode=baseline")).status_code, 200)
        self.assertEqual((await self.call(app, "GET", "/ready")).status_code, 200)
        self.assertError(await self.call(app, "GET", "/ready?mode=rerank"), 503, "rerank_not_ready", False)
        recheck = await self.call(app, "POST", "/self-check")
        self.assertEqual((recheck.status_code, recheck.json()), (200, {"rerank_ready": False}))

    async def test_an_unknown_or_repeated_ready_mode_is_an_invalid_request(self):
        _, app = self.make(FakeReranker())
        for query in ("?mode=turbo", "?mode=compare", "?mode=", "?mode=baseline&mode=rerank", "?mode=BASELINE"):
            with self.subTest(query):
                self.assertError(await self.call(app, "GET", "/ready" + query), 422, "invalid_request", False)

    async def test_health_reports_status_and_identities_without_secrets(self):
        service, app = self.make(FakeReranker(), default_mode="compare", api_token="s3cret", manifest_sha256="ab" * 32)
        response = await self.call(app, "GET", "/health")
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(body), {"alive", "baseline_ready", "rerank_ready", "busy", "release"})
        self.assertEqual((body["alive"], body["baseline_ready"], body["rerank_ready"], body["busy"]), (True, True, True, False))
        self.assertEqual(body["release"], {
            "status": "evaluated", "release_id": "rel-test", "default_mode": "compare", "corpus_id": self.loaded.corpus.corpus_id,
            "index_id": self.loaded.index_id, "model_id": "model-fake", "manifest_sha256": "ab" * 32})
        self.assertNotIn("s3cret", response.text)


class SlotTests(ApiCase):
    async def test_health_and_ready_answer_while_a_search_is_blocked_in_the_worker(self):
        gate, fake, service, app = self.gated()
        task = asyncio.create_task(self.search(app, query=QUERY, mode="rerank"))
        await self.until(gate.entered.is_set, "the worker to enter the model")
        health = await self.call(app, "GET", "/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual((health.json()["alive"], health.json()["busy"], health.json()["rerank_ready"]), (True, True, True))
        for mode in ("baseline", "rerank"):
            self.assertEqual((await self.call(app, "GET", f"/ready?mode={mode}")).status_code, 200, "busy is not not-ready")
        gate.go.set()
        response = await task
        self.assertEqual((response.status_code, response.json()["status"]), (200, "ok"))
        self.assertFalse((await self.call(app, "GET", "/health")).json()["busy"])

    async def test_a_second_concurrent_search_gets_429_and_is_never_queued(self):
        gate, fake, service, app = self.gated()
        first = asyncio.create_task(self.search(app, query=QUERY, mode="rerank"))
        await self.until(gate.entered.is_set, "the worker to enter the model")
        refused = await self.search(app, query=QUERY, mode="rerank")
        self.assertError(refused, 429, "busy", True)
        self.assertError(await self.call(app, "POST", "/self-check"), 429, "busy", True)
        self.assertEqual(len(fake.calls), 1, "the refused request never reached the model")
        gate.go.set()
        self.assertEqual((await first).status_code, 200)
        await self.until(lambda: not service.busy, "the slot to be released")
        await asyncio.sleep(0.1)                              # a queued request would have run by now
        self.assertEqual(len(fake.calls), 1, "no waiting queue: the refused request did not run later")
        self.assertEqual((await self.search(app, query=QUERY, mode="rerank")).status_code, 200)
        self.assertEqual(len(fake.calls), 2)

    async def test_the_slot_survives_an_http_timeout_until_the_worker_really_ends(self):
        gate, fake, service, app = self.gated(deadline_s=0.05, http_grace_s=0.05)      # the pipeline's clock never moves: a stuck kernel
        with self.assertLogs("devsearch.serving", "WARNING") as logs:
            response = await self.search(app, query=QUERY, mode="rerank")
            self.assertError(response, 504, "deadline_exceeded", True)
            await self.until(gate.entered.is_set, "the worker to enter the model")     # the 504 can beat a slow thread start
            self.assertTrue(service.busy, "a 504 does not free the slot")
            self.assertTrue((await self.call(app, "GET", "/health")).json()["busy"])
            self.assertError(await self.search(app, query=QUERY), 429, "busy", True)
            self.assertError(await self.call(app, "POST", "/self-check"), 429, "busy", True)
            gate.go.set()
            await self.until(lambda: not service.busy, "the worker to finish and release the slot")
            await self.until(lambda: any("after its HTTP response" in line for line in logs.output), "the orphaned outcome to be logged")
        again = await self.search(app, query=QUERY, mode="rerank")
        self.assertEqual((again.status_code, again.json()["status"]), (200, "ok"))

    async def test_a_cancelled_or_disconnected_request_does_not_free_the_slot(self):
        gate, fake, service, app = self.gated()
        with self.assertLogs("devsearch.serving", "WARNING") as logs:
            task = asyncio.create_task(self.search(app, query=QUERY, mode="rerank"))
            await self.until(gate.entered.is_set, "the worker to enter the model")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(service.busy, "the client is gone but the model call is still running")
            self.assertError(await self.search(app, query=QUERY), 429, "busy", True)
            gate.go.set()
            await self.until(lambda: not service.busy, "the worker to finish and release the slot")
            await self.until(lambda: any("after its HTTP response" in line for line in logs.output), "the orphaned outcome to be logged")
        self.assertEqual((await self.search(app, query=QUERY)).status_code, 200)

    async def test_an_orphaned_worker_that_fails_is_logged_and_frees_the_slot(self):
        gate = Gate()
        self.addCleanup(gate.go.set)
        service, app = self.make(deadline_s=0.05, http_grace_s=0.05)

        def stuck_then_failing(request, deadline_at, request_id):
            gate(None, [], None)
            raise RuntimeError("late failure")
        service.pipeline.script = stuck_then_failing
        with self.assertLogs("devsearch.serving", "WARNING") as logs:
            self.assertError(await self.search(app, query=QUERY), 504, "deadline_exceeded", True)
            gate.go.set()
            await self.until(lambda: not service.busy, "the slot to be released")
            await self.until(lambda: any("failed after its HTTP response" in line for line in logs.output), "the failure to be logged")

    async def test_all_model_work_runs_on_one_dedicated_thread_off_the_event_loop(self):
        fake = FakeReranker()
        service, app = self.make(fake)
        for _ in range(2):
            await self.search(app, query=QUERY, mode="rerank")
        await self.call(app, "POST", "/self-check")
        self.assertEqual(len(fake.threads), 3)
        (worker,) = set(fake.threads)
        self.assertIsNot(worker, threading.current_thread())
        self.assertTrue(worker.name.startswith("devsearch-search"))

    async def test_work_cancelled_before_it_started_cannot_leak_the_slot(self):
        executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown, wait=True)
        blocker = threading.Event()
        self.addCleanup(blocker.set)
        executor.submit(blocker.wait, 10)                     # occupies the only thread, so the next job stays queued
        pipeline = ScriptedPipeline(self.loaded, self.limits, None, clock=self.clock)
        service = SearchService(pipeline, deadline_s=5.0, release={"status": "baseline", "release_id": "r", "default_mode": "baseline"},
                                clock=self.clock, executor=executor)
        future = service.submit(lambda deadline_at: 1)
        self.assertTrue(service.busy)
        self.assertTrue(future.cancel())
        self.assertFalse(service.busy, "a job that never started is not holding the device")
        service.close()                                       # an executor passed in is the caller's to shut down
        blocker.set()
        self.assertEqual(executor.submit(lambda: 7).result(timeout=5), 7)

    async def test_submit_refuses_instead_of_queueing_and_a_failed_submit_returns_the_slot(self):
        service, _ = self.make()
        release = threading.Event()
        self.addCleanup(release.set)
        held = service.submit(lambda deadline_at: release.wait(5))
        with self.assertRaises(BusyError):
            service.submit(lambda deadline_at: 1)
        release.set()
        self.assertTrue(held.result(timeout=5))
        self.assertFalse(service.busy)
        service.close()                                       # its own executor is shut down: submitting now fails...
        with self.assertRaises(RuntimeError):
            service.submit(lambda deadline_at: 1)
        self.assertFalse(service.busy, "...and the slot taken for that attempt is given back")


class FactoryTests(ApiCase):
    async def test_every_app_is_independent_with_its_own_slot_and_state(self):
        self.assertFalse(hasattr(api_module, "app"), "no global app is created at import time")
        gate, fake, first_service, first_app = self.gated()
        second_service, second_app = self.make(FakeReranker())
        self.assertIsNot(first_app, second_app)
        self.assertIs(first_app.state.service, first_service)
        self.assertIs(second_app.state.service, second_service)
        blocked = asyncio.create_task(self.search(first_app, query=QUERY, mode="rerank"))
        await self.until(gate.entered.is_set, "the first worker to enter the model")
        self.assertTrue((await self.call(first_app, "GET", "/health")).json()["busy"])
        self.assertFalse((await self.call(second_app, "GET", "/health")).json()["busy"])
        self.assertEqual((await self.search(second_app, query=QUERY, mode="rerank")).status_code, 200)
        gate.go.set()
        self.assertEqual((await blocked).status_code, 200)


class ServiceConfigTests(ApiCase):
    def pipeline(self, **kwargs):
        return SearchPipeline(self.loaded, self.limits, None, **kwargs)

    def test_bad_timeouts_are_rejected(self):
        release = {"status": "baseline", "release_id": "r", "default_mode": "baseline"}
        for deadline in (0, -1, math.nan, math.inf, "5", None):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                SearchService(self.pipeline(), deadline_s=deadline, release=release)
        for grace in (-1, math.nan, math.inf, "1"):
            with self.subTest(grace=grace), self.assertRaises(ValueError):
                SearchService(self.pipeline(), deadline_s=1.0, http_grace_s=grace, release=release)

    def test_a_malformed_or_contradictory_release_is_rejected(self):
        good = {"status": "baseline", "release_id": "rel", "default_mode": "baseline"}
        bad = {
            "missing status": {k: v for k, v in good.items() if k != "status"}, "unknown status": {**good, "status": "bogus"},
            "unknown default mode": {**good, "default_mode": "turbo"}, "no release id": {**good, "release_id": ""},
            "contradicts the pipeline's release id": {**good, "release_id": "other"},
            "contradicts the corpus": {**good, "corpus_id": "corpus-other"}, "contradicts the index": {**good, "index_id": "index-other"},
            "non-scalar identity": {**good, "model_id": ["a"]},
        }
        for name, release in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                SearchService(self.pipeline(release_id="rel"), deadline_s=1.0, release=release).close()
        SearchService(self.pipeline(release_id="rel"), deadline_s=1.0, release=good).close()

    def test_the_body_cap_always_admits_a_query_within_the_limit(self):
        service = SearchService(self.pipeline(), deadline_s=1.0, release={"status": "baseline", "release_id": "r", "default_mode": "baseline"})
        self.addCleanup(service.close)
        worst = json.dumps({"query": "\U0001F600" * self.limits.max_query_chars}, ensure_ascii=True)
        self.assertLess(len(worst.encode()), service.max_body_bytes)


class ServeForeverTests(ApiCase):
    def test_only_loopback_hosts_are_served_unless_opted_in(self):
        service, _ = self.make()
        with mock.patch.object(api_module.uvicorn, "run") as run:
            for host in ("0.0.0.0", "::", "192.168.1.20", "example.com", "", "127.0.0.2"):
                with self.subTest(host=host), self.assertRaises(ValueError):
                    serve_forever(service, host, 8000)
            run.assert_not_called()
            self.assertFalse(service.busy)
            for host in ("127.0.0.1", "::1", "localhost"):
                with self.subTest(host=host):
                    serve_forever(self.make()[0], host, 8123)
                    self.assertEqual(run.call_args.kwargs, {"host": host, "port": 8123, "log_level": "warning"})
                    self.assertEqual(type(run.call_args.args[0]).__name__, "FastAPI")
            serve_forever(self.make()[0], "0.0.0.0", 8123, allow_non_loopback=True)
            self.assertEqual(run.call_args.kwargs["host"], "0.0.0.0")

    def test_a_bad_port_is_rejected_and_the_service_is_closed_when_the_server_stops(self):
        service, _ = self.make()
        with mock.patch.object(api_module.uvicorn, "run") as run:
            for port in (-1, 65536, "8000", None, True):
                with self.subTest(port=port), self.assertRaises(ValueError):
                    serve_forever(service, "127.0.0.1", port)
            run.assert_not_called()
            serve_forever(service, "127.0.0.1", 8000)
        with self.assertRaises(RuntimeError):
            service.submit(lambda deadline_at: 1)            # close() ran after uvicorn.run returned
        self.assertFalse(service.busy)


# -- UI helpers (pure: no Streamlit, no network) -------------------------------------------------------------
def doc_result(rank, **overrides):
    record = {
        "chunk_id": f"ch_{rank}", "source_id": "fixture-torch", "source_url": "https://pytorch.org/tutorials/checkpoints",
        "revision": "r1", "title": f"Title {rank}", "section": "A > B", "text": f"text {rank}", "code_fragment": False,
        "source_location": [{"path": "docs/a.md", "start_line": 3, "end_line": 6}], "baseline_rank": rank, "reranked_rank": None,
        "bm25_score": 10.0 - rank, "rerank_logit": None}
    record.update(overrides)
    return record


def doc(**overrides):
    document = {"request_id": "rid", "release_id": "rel", "status": "ok", "mode_requested": "baseline", "mode_effective": "baseline",
                "corpus_id": "c", "index_id": "i", "model_id": None, "candidate_set_id": "cs", "candidate_count": 2,
                "baseline_results": [doc_result(1), doc_result(2)], "reranked_results": [],
                "timings_ms": {"retrieval": 1.25, "rerank": None, "total": 1.5}, "warnings": []}
    document.update(overrides)
    return document


class UiHelperTests(unittest.TestCase):
    def test_importing_the_ui_module_imports_neither_streamlit_nor_httpx(self):
        script = "import sys; import devsearch.serving.ui; assert not {'streamlit', 'httpx', 'fastapi'} & set(sys.modules), sorted(sys.modules)"
        result = subprocess.run([sys.executable, "-B", "-c", script], env={**os.environ, "PYTHONPATH": str(SRC)},
                                capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_payload_omits_unset_options_so_the_api_applies_its_defaults(self):
        self.assertEqual(ui.build_payload("q", "compare"), {"query": "q", "mode": "compare"})
        self.assertEqual(ui.build_payload("q", "rerank", ["a", "b"], candidate_k=20, top_k=4, versions={"a": "r1"}),
                         {"query": "q", "mode": "rerank", "sources": ["a", "b"], "versions": {"a": "r1"}, "candidate_k": 20, "top_k": 4})
        self.assertEqual(ui.build_payload("  keep  spacing ")["query"], "  keep  spacing ", "the query is never rewritten")

    def test_parse_source_filter_splits_without_correcting(self):
        self.assertEqual(ui.parse_source_filter(""), [])
        self.assertEqual(ui.parse_source_filter(" a, b  c ,,d "), ["a", "b", "c", "d"])
        self.assertEqual(ui.parse_source_filter("a,a"), ["a", "a"], "the API decides what a repeated source means")

    def test_every_displayed_logit_is_labelled_as_uncalibrated(self):
        self.assertEqual(ui.LOGIT_LABEL, "uncalibrated relevance score (not a probability, not comparable to BM25)")
        for value in (2.5, -3, 0.0, 1e-9):
            self.assertIn(ui.LOGIT_LABEL, ui.format_logit(value))
        self.assertTrue(ui.format_logit(2.5).startswith("2.500"))
        for value in (None, math.nan, math.inf, "x", True):
            self.assertEqual(ui.format_logit(value), "not reranked")
        self.assertEqual(ui.format_bm25(10.7385688684), "10.739")
        self.assertEqual(ui.format_bm25(None), "n/a")

    def test_source_links_must_be_https_urls_on_a_configured_host(self):
        hosts = ("pytorch.org", "huggingface.co")
        good = ["https://pytorch.org/tutorials/x", "https://docs.pytorch.org/stable/x.html", "https://huggingface.co/docs/y#z"]
        bad = ["javascript:alert(1)", "data:text/html,<script>1</script>", "http://pytorch.org/x", "//pytorch.org/x", "https://evil.example/pytorch.org",
               "https://pytorch.org.evil.example/x", "https://notpytorch.org/x", "https://user:pw@pytorch.org/x", "https://pytorch.org/a b",
               "https://pytorch.org/\n", "https:///x", "", None, 5, "https://[::1/x", "https://" + "a" * 3000 + ".pytorch.org/"]
        for url in good:
            self.assertEqual(ui.safe_link(url, hosts), url, url)
        for url in bad:
            self.assertIsNone(ui.safe_link(url, hosts), url)
        self.assertIsNone(ui.safe_link(good[0], ()), "an empty host list allows nothing")
        self.assertEqual(ui.safe_link("https://docs.fixture.invalid/x", ("fixture.invalid",)), "https://docs.fixture.invalid/x")

    def test_hosts_and_api_url_come_from_the_environment(self):
        self.assertEqual(ui.source_hosts({}), ui.DEFAULT_SOURCE_HOSTS)
        self.assertEqual(ui.source_hosts({"DEVSEARCH_SOURCE_HOSTS": " A.example , b.example,"}), ("a.example", "b.example"))
        self.assertEqual(ui.source_hosts({"DEVSEARCH_SOURCE_HOSTS": ""}), (), "set but empty: no links")
        self.assertEqual(ui.api_base_url({}), "http://127.0.0.1:8000")
        self.assertEqual(ui.api_base_url({"DEVSEARCH_API_URL": "http://localhost:9000/"}), "http://localhost:9000")
        for bad in ("localhost:9000", "ftp://x", "file:///etc/passwd", "http://", "javascript:1"):
            with self.subTest(bad), self.assertRaises(ValueError):
                ui.api_base_url({"DEVSEARCH_API_URL": bad})

    def test_result_rows_carry_ranks_scores_locations_and_validated_links(self):
        notebook = doc_result(2, source_location=[{"path": "nb.ipynb", "cell_id": "c1", "start_char": 4, "end_char": 40}],
                              code_fragment=True, source_url="javascript:alert(1)", section=None, title=None, reranked_rank=1, rerank_logit=1.5)
        rows = ui.result_rows([doc_result(1, source_location=[{"path": "a.md", "start_line": 7, "end_line": 7}]), notebook],
                              allowed_hosts=("pytorch.org",))
        first, second = rows
        self.assertEqual((first["title"], first["section"], first["location"], first["kind"]), ("Title 1", "A > B", "a.md line 7", "passage"))
        self.assertEqual(first["link"], "https://pytorch.org/tutorials/checkpoints")
        self.assertEqual((second["title"], second["section"], second["kind"]), ("(untitled)", "", "code excerpt"))
        self.assertEqual(second["location"], "nb.ipynb cell c1 chars 4-40")
        self.assertIsNone(second["link"])
        self.assertEqual(second["source_url"], "javascript:alert(1)", "kept so the page can show it as inert text")
        self.assertEqual(ui.format_location([{"path": "a.md", "start_line": 3, "end_line": 6}, "junk"]), "a.md lines 3-6")
        for empty in (None, [], "text", 5):
            self.assertEqual(ui.format_location(empty), "location unavailable")
        self.assertEqual(ui.result_rows(None), [])
        summary = ui.row_summary(second, "reranked")
        self.assertIn("#1  (untitled)", summary)
        self.assertIn(ui.LOGIT_LABEL, summary)
        self.assertNotIn(ui.LOGIT_LABEL, ui.row_summary(first, "baseline"))

    def test_describe_response_for_baseline_compare_degraded_and_empty(self):
        baseline = ui.describe_response(doc())
        self.assertEqual((baseline["layout"], baseline["banner"]["level"], baseline["counts"]), ("baseline", None, {"candidates": 2, "baseline": 2, "reranked": 0}))
        self.assertEqual(baseline["timings"], "retrieval 1.2 ms | rerank not run | total 1.5 ms")
        self.assertIn("does not say that an answer is correct", baseline["note"])

        reranked = [doc_result(2, reranked_rank=1, rerank_logit=0.5), doc_result(1, reranked_rank=2, rerank_logit=-0.5)]
        compare = ui.describe_response(doc(mode_requested="compare", mode_effective="compare", reranked_results=reranked, model_id="m",
                                           timings_ms={"retrieval": 1.0, "rerank": 2.0, "total": 3.0}))
        self.assertEqual((compare["layout"], compare["counts"]["reranked"]), ("compare", 2))
        self.assertEqual(sorted(r["chunk_id"] for r in compare["baseline"]), sorted(r["chunk_id"] for r in compare["reranked"]))
        self.assertEqual(compare["timings"], "retrieval 1.0 ms | rerank 2.0 ms | total 3.0 ms")
        only = ui.describe_response(doc(mode_requested="rerank", mode_effective="rerank", baseline_results=[], reranked_results=reranked))
        self.assertEqual(only["layout"], "reranked")

        degraded = ui.describe_response(doc(status="degraded", mode_requested="compare", model_id="m",
                                            warnings=[{"code": "rerank_failed", "message": "Reranking failed; showing BM25 results."}]))
        self.assertEqual((degraded["layout"], degraded["banner"]["level"], degraded["counts"]["reranked"]), ("baseline", "warning", 0))
        self.assertIn("No partial reranking", degraded["banner"]["text"])
        self.assertEqual(degraded["warnings"], ["rerank_failed: Reranking failed; showing BM25 results."])
        self.assertIn("reranking not applied", degraded["baseline_title"])

        empty = ui.describe_response(doc(status="empty", candidate_count=0, baseline_results=[], reranked_results=[]))
        self.assertEqual((empty["layout"], empty["banner"]["level"]), ("none", "info"))
        self.assertIn("not that no answer exists", empty["banner"]["text"])

    def test_describe_error_turns_api_error_bodies_into_readable_messages(self):
        body = {"request_id": "rid", "error": {"code": "busy", "message": "another search is already running", "retryable": True}}
        busy = ui.describe_error(429, body)
        self.assertEqual((busy["code"], busy["retryable"], busy["request_id"], busy["detail"]), ("busy", True, "rid", "another search is already running"))
        self.assertIn("nothing is queued", busy["hint"])
        for status in (422, 503, 504, 500):
            self.assertTrue(ui.describe_error(status, body)["title"])
        foreign = ui.describe_error(502, "<html>bad gateway</html>")
        self.assertEqual((foreign["code"], foreign["title"]), (None, "Unexpected response from the API (HTTP 502)"))
        self.assertNotIn("html", foreign["detail"])

    @requires_serve
    def test_call_api_returns_documents_and_reports_every_failure_without_raising(self):
        seen = []

        def handler(request):
            seen.append((str(request.url), json.loads(request.content), request.headers.get("content-type")))
            return responses.pop(0)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        body = {"request_id": "r", "error": {"code": "deadline_exceeded", "message": "too slow", "retryable": True}}
        responses = [httpx.Response(200, json=doc()), httpx.Response(429, json={**body, "error": {**body["error"], "code": "busy"}}),
                     httpx.Response(504, json=body), httpx.Response(200, text="not json"), httpx.Response(500, text="oops")]
        ok = ui.call_api({"query": "q", "mode": "baseline"}, base_url="http://127.0.0.1:8000", client=client)
        self.assertEqual((ok["ok"], ok["document"]["status"]), (True, "ok"))
        self.assertEqual(seen[0], ("http://127.0.0.1:8000/search", {"query": "q", "mode": "baseline"}, "application/json"))
        results = [ui.call_api({"query": "q"}, base_url="http://x", client=client) for _ in range(4)]
        self.assertEqual([(r["ok"], r["status"], r["code"]) for r in results],
                         [(False, 429, "busy"), (False, 504, "deadline_exceeded"), (False, 200, None), (False, 500, None)])

        def refuse(request):
            raise httpx.ConnectError("connection refused", request=request)

        def hang(request):
            raise httpx.ReadTimeout("slow", request=request)
        down_client, slow_client = httpx.Client(transport=httpx.MockTransport(refuse)), httpx.Client(transport=httpx.MockTransport(hang))
        self.addCleanup(down_client.close)
        self.addCleanup(slow_client.close)
        down = ui.call_api({"query": "q"}, base_url="http://127.0.0.1:1", client=down_client)
        self.assertEqual((down["ok"], down["code"], down["retryable"]), (False, "unreachable", True))
        self.assertIn("DEVSEARCH_API_URL", down["hint"])
        slow = ui.call_api({"query": "q"}, base_url="http://127.0.0.1:1", timeout=3, client=slow_client)
        self.assertEqual((slow["ok"], slow["code"]), (False, "client_timeout"))
        self.assertIn("3 s", slow["detail"])


class UiApiContractTests(ApiCase):
    """The page's helpers accept the documents the real service produces."""

    async def test_real_responses_describe_cleanly(self):
        _, app = self.make(FakeReranker(reverse_scores))
        compare = ui.describe_response((await self.search(app, query=QUERY, mode="compare")).json(), allowed_hosts=("fixture.invalid",))
        self.assertEqual((compare["layout"], compare["counts"]["baseline"], compare["counts"]["reranked"]), ("compare", 3, 3))
        self.assertTrue(all(r["link"] and r["location"] != "location unavailable" for r in compare["baseline"] + compare["reranked"]))
        self.assertTrue(all(ui.LOGIT_LABEL in r["logit_text"] for r in compare["reranked"]))
        empty = ui.describe_response((await self.search(app, query="zzzzqqqq")).json())
        self.assertEqual((empty["layout"], empty["banner"]["level"]), ("none", "info"))
        _, broken = self.make(FakeReranker(RuntimeError("boom")))
        degraded = ui.describe_response((await self.search(broken, query=QUERY, mode="compare")).json())
        self.assertEqual((degraded["layout"], degraded["banner"]["level"], degraded["warnings"][0][:13]), ("baseline", "warning", "rerank_failed"))
        error = await self.search(app, query="")
        described = ui.describe_error(error.status_code, error.json())
        self.assertEqual((described["code"], described["detail"]), ("invalid_request", "query must not be empty"))


@requires_serve
class UiPageSmokeTest(unittest.TestCase):
    """One real render of `main()` with a canned API (offline, well under a second): hostile result text must reach the
    page only as plain text/code, and a link only when it is a validated https URL."""

    HOSTILE = "<script>alert(1)</script> [x](javascript:alert(1)) ![i](http://evil.invalid/p.png)"

    @staticmethod
    def leaves(node):
        for child in getattr(node, "children", {}).values():
            if hasattr(child, "children"):
                yield from UiPageSmokeTest.leaves(child)
            else:
                yield child

    def render(self, outcome, *, sources="a, b", mode="rerank"):
        from streamlit.testing.v1 import AppTest

        def page():
            from devsearch.serving import ui
            ui.main()
        for name in [n for n in logging.root.manager.loggerDict if n.startswith("streamlit")]:    # "missing ScriptRunContext" noise in bare mode
            noisy = logging.getLogger(name)
            self.addCleanup(noisy.setLevel, noisy.level)
            noisy.setLevel(logging.ERROR)
        with mock.patch.object(ui, "call_api", return_value=outcome) as api:
            at = AppTest.from_function(page, default_timeout=25).run()
            self.assertFalse(at.exception)
            self.assertEqual(api.call_count, 0, "nothing is searched before the form is submitted")
            at.text_input[0].set_value("how do I save a checkpoint")
            at.text_input[1].set_value(sources)
            at.radio[0].set_value(mode)
            at.button[0].click().run()
        self.assertFalse(at.exception)
        return at, api

    def test_the_page_shows_untrusted_results_as_plain_text_and_links_only_validated_urls(self):
        hostile = self.HOSTILE
        rows = [doc_result(1, title=hostile, text=hostile, section=hostile),
                doc_result(2, title=hostile, text=hostile, source_url="javascript:alert(1)", code_fragment=True)]
        reranked = [doc_result(2, reranked_rank=1, rerank_logit=0.5), doc_result(1, reranked_rank=2, rerank_logit=-0.5)]
        document = doc(mode_requested="compare", mode_effective="compare", model_id="m", baseline_results=rows, reranked_results=reranked)
        at, api = self.render({"ok": True, "document": document})
        self.assertEqual(api.call_args.args[0], {"query": "how do I save a checkpoint", "mode": "rerank", "sources": ["a", "b"]})
        kinds = {}
        for element in self.leaves(at.main):
            kinds.setdefault(element.type, []).append(element)
        self.assertEqual((len(kinds["subheader"]), len(kinds["link_button"]), len(kinds["code"])), (2, 3, 4))
        self.assertTrue(any(hostile in t.value for t in at.text), "result text is shown, as text")
        self.assertTrue(any(hostile in c.value for c in at.code))
        for kind, elements in kinds.items():
            if kind not in ("text", "code"):                              # markdown-capable elements only ever get static wording
                for element in elements:
                    self.assertNotIn("evil.invalid", str(element.proto), kind)
                    self.assertNotIn("<script>", str(element.proto), kind)
        self.assertEqual({e.proto.url for e in kinds["link_button"]}, {"https://pytorch.org/tutorials/checkpoints"})
        self.assertIn("Source URL (not linked): javascript:alert(1)", [t.value for t in at.text])
        self.assertTrue(any(ui.LOGIT_LABEL in t.value for t in at.text))

    def test_an_api_error_is_shown_as_a_readable_message_with_no_results(self):
        body = {"request_id": "r", "error": {"code": "busy", "message": "**busy** [x](http://evil.invalid)", "retryable": True}}
        at, _ = self.render({"ok": False, **ui.describe_error(429, body)})
        self.assertEqual([e.value for e in at.error], ["The search service is busy"])
        self.assertIn("**busy** [x](http://evil.invalid)", [t.value for t in at.text])
        self.assertEqual(len(at.get("link_button")), 0)


if __name__ == "__main__":
    unittest.main()
