"""The shared search pipeline: BM25 candidates, optional rerank, one response document (docs/04).

The CLI, the API and the offline checks all call `SearchPipeline.search`, so candidate selection,
pair encoding, scoring and ordering are defined once. Reranking is all-or-nothing: only when every
candidate has one finite logit is a reranked list published. A timeout, non-finite score, length
violation or model exception discards every partial score and, because a complete baseline already
exists, returns an explicit `degraded` result carrying a stable warning code. A model exception or
non-finite output additionally marks reranking not ready until an explicit successful `self_check`
(nothing reloads the model on its own).

The pipeline is deliberately free of torch: the reranker is any object with `score`, `model_id` and
`self_check`, so a baseline-only release needs no ML dependency. Concurrency (a single slot) is the
API layer's job; this class is not thread-safe and is driven by one worker at a time.
"""

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from devsearch.contracts import RetrievalCandidate, SearchRequest
from devsearch.errors import DeadlineExceeded, DeadlineExceededError, NonFiniteScoreError
from devsearch.evaluation.fusion import RankingPolicy
from devsearch.modeling.policy import LengthError
from devsearch.retrieval.config import SearchConfig
from devsearch.retrieval.index_store import LoadedIndex
from devsearch.retrieval.search import candidate_set_id, result_record, validate_request

RESPONSE_SCHEMA_VERSION = 2
MODES = ("baseline", "rerank", "compare")


class RerankerPort(Protocol):
    @property
    def model_id(self) -> str: ...

    def score(self, query: str, passages: Sequence[str], *, should_stop: Callable[[], bool] | None = None) -> list[float]: ...

    def self_check(self) -> dict[str, Any]: ...


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


class SearchPipeline:
    def __init__(
        self, loaded: LoadedIndex, limits: SearchConfig, reranker: RerankerPort | None = None, *,
        release_id: str | None = None, clock: Callable[[], float] = time.monotonic,
        ranking: RankingPolicy | None = None,
    ) -> None:
        self.loaded = loaded
        self.limits = limits
        self.reranker = reranker
        self.release_id = release_id
        self.ranking = ranking or RankingPolicy()
        self._clock = clock
        self._rerank_healthy = reranker is not None
        self.rerank_failure: str | None = None

    @property
    def clock(self) -> Callable[[], float]:
        """The monotonic clock deadlines are measured on; the API service must use this very clock."""
        return self._clock

    # -- readiness -----------------------------------------------------------------------------------
    @property
    def baseline_ready(self) -> bool:
        return True                                 # an unusable index never constructs a pipeline

    @property
    def rerank_ready(self) -> bool:
        return self.reranker is not None and self._rerank_healthy

    def self_check(self) -> bool:
        """Explicit recovery: re-run the reranker's recorded smoke test. A pass restores readiness."""
        if self.reranker is None:
            return False
        try:
            report = self.reranker.self_check()
        except Exception as error:                  # a failing self-check keeps rerank not ready, whatever the cause
            self._rerank_healthy, self.rerank_failure = False, f"self_check_failed: {type(error).__name__}"
            return False
        self._rerank_healthy = bool(report.get("passed"))
        self.rerank_failure = None if self._rerank_healthy else "self_check_mismatch"
        return self._rerank_healthy

    # -- search --------------------------------------------------------------------------------------
    def search(self, request: SearchRequest, *, deadline_at: float | None = None, request_id: str | None = None) -> dict[str, Any]:
        """One response document. `deadline_at` is an absolute `clock()` value, or None for no deadline."""
        validate_request(request, self.limits, modes=MODES)
        started = self._clock()
        if deadline_at is not None and started >= deadline_at:
            raise DeadlineExceededError("the deadline passed before retrieval started; no baseline result exists")
        candidates = self.loaded.retriever().retrieve(request)
        retrieved = self._clock()
        warnings: list[dict[str, str]] = []
        rerank_ms: float | None = None
        reranked: list[dict[str, Any]] = []
        attempted_model = self.reranker.model_id if self.reranker is not None else None
        mode_effective = "baseline"
        if request.mode != "baseline" and candidates:
            reranked, rerank_ms, failure = self._rerank(request, candidates, deadline_at)
            if failure is None:
                mode_effective = request.mode
            else:
                warnings.append(failure)
        elif request.mode != "baseline":
            mode_effective = request.mode           # an empty candidate list is a valid empty result, not a failure
        degraded = request.mode != "baseline" and bool(candidates) and mode_effective == "baseline"
        finished = self._clock()
        top = candidates[:request.top_k]
        show_baseline = request.mode in ("baseline", "compare") or degraded
        return {
            "schema_version": RESPONSE_SCHEMA_VERSION, "request_id": request_id, "release_id": self.release_id,
            "status": "empty" if not candidates else "degraded" if degraded else "ok",
            "mode_requested": request.mode, "mode_effective": mode_effective,
            "ranking_policy": self.ranking.to_record() if request.mode != "baseline" else None,
            "corpus_id": self.loaded.corpus.corpus_id, "index_id": self.loaded.index_id,
            "model_id": attempted_model if request.mode != "baseline" else None,
            "candidate_set_id": candidate_set_id(self.loaded, request, candidates),
            "query": request.query, "filters": {"sources": list(request.sources), "versions": dict(request.versions)},
            "candidate_k": request.candidate_k, "candidate_count": len(candidates), "top_k": request.top_k,
            "baseline_results": [result_record(self.loaded, c) for c in top] if show_baseline else [],
            "reranked_results": reranked,
            "timings_ms": {"retrieval": round((retrieved - started) * 1000.0, 3),
                           "rerank": None if rerank_ms is None else round(rerank_ms, 3),
                           "total": round((finished - started) * 1000.0, 3)},
            "warnings": warnings,
        }

    def _unhealthy(self, reason: str, code: str, message: str) -> tuple[list[dict[str, Any]], None, dict[str, str]]:
        self._rerank_healthy, self.rerank_failure = False, reason
        return [], None, _warning(code, message)

    def _rerank(
        self, request: SearchRequest, candidates: tuple[RetrievalCandidate, ...], deadline_at: float | None,
    ) -> tuple[list[dict[str, Any]], float | None, dict[str, str] | None]:
        """(results, elapsed_ms, failure warning). On any failure the results are empty: nothing partial."""
        if self.reranker is None:
            return [], None, _warning("rerank_unavailable", "This release has no reranker; showing BM25 results.")
        if not self._rerank_healthy:
            return [], None, _warning("rerank_not_ready", f"Reranking is not ready ({self.rerank_failure}); showing BM25 results.")
        started = self._clock()
        should_stop = None if deadline_at is None else (lambda: self._clock() >= deadline_at)
        try:
            scores = self.reranker.score(request.query, [c.chunk.text for c in candidates], should_stop=should_stop)
        except DeadlineExceeded:
            return [], None, _warning("rerank_timeout", "Reranking exceeded the deadline; showing BM25 results.")
        except LengthError as error:
            return [], None, _warning("rerank_input_too_long", f"A candidate does not fit the pair-length policy ({error}); showing BM25 results.")
        except NonFiniteScoreError as error:
            return self._unhealthy("nonfinite_scores", "rerank_nonfinite", f"The reranker produced a non-finite score ({error}); showing BM25 results.")
        except Exception as error:                  # any model/device failure: discard, degrade, mark not ready
            return self._unhealthy(f"exception: {type(error).__name__}", "rerank_failed", "Reranking failed; showing BM25 results.")
        if len(scores) != len(candidates):
            return self._unhealthy("wrong_score_count", "rerank_failed", "Reranking failed; showing BM25 results.")
        try:
            ordering = self.ranking.order([c.chunk.chunk_id for c in candidates], scores, [c.baseline_rank for c in candidates])
        except ValueError as error:                 # NaN/inf or a non-number slipped through the scorer
            return self._unhealthy("nonfinite_scores", "rerank_nonfinite", f"The reranker produced an invalid score ({error}); showing BM25 results.")
        by_id = {c.chunk.chunk_id: c for c in candidates}
        logit_of = {c.chunk.chunk_id: float(score) for c, score in zip(candidates, scores)}
        results = []
        for chunk_id, rank, score in ordering[:request.top_k]:
            record = result_record(self.loaded, by_id[chunk_id])
            record["reranked_rank"], record["rerank_logit"] = rank, logit_of[chunk_id]
            if self.ranking.method == "rank_fusion":
                record["fusion_score"] = score
            results.append(record)
        return results, (self._clock() - started) * 1000.0, None
