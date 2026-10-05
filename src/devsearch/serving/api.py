"""HTTP layer: the single-slot search service and the FastAPI app (docs/04 "Search API", "Failure behavior").

`SearchService` owns what makes the API safe on one accelerator: one admission slot, taken *before* work
is submitted, and one process-local worker thread that runs every retrieval and model call (CPU included),
so the event loop and `/health` stay responsive while a search runs. The slot is released only by the worker,
in a `finally`, when its work has really ended: an HTTP timeout, a client disconnect or a cancelled handler
never frees it, because a running device kernel cannot be interrupted. A second search is refused (429),
never queued. `create_app` adds the routes; it holds no global state, so every call is independent.

Request validation lives in `SearchPipeline.search`; this module only checks JSON types (it never clamps or
repairs a value) and turns every failure into one error shape. Error bodies carry no stack trace, exception
text or result array; detail goes to the `devsearch.serving` logger (never the query text). No CORS, no
auth, no cache, no telemetry and no OpenAPI/docs pages (they load scripts from a CDN); bind defaults live
with the caller and `serve_forever` refuses non-loopback hosts.
"""

import asyncio
import json
import logging
import math
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from devsearch.contracts import SearchRequest
from devsearch.errors import BusyError, DeadlineExceededError, IndexUnavailableError, InvalidRequestError
from devsearch.retrieval.config import SearchConfig
from devsearch.serving.admission import Admission
from devsearch.serving.pipeline import MODES, SearchPipeline

logger = logging.getLogger("devsearch.serving")
T = TypeVar("T")

LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
RELEASE_STATUSES = ("baseline", "candidate", "evaluated")
READY_MODES = ("baseline", "rerank")
SEARCH_FIELDS = ("query", "sources", "versions", "candidate_k", "top_k", "mode")
_RELEASE_KEYS = ("status", "release_id", "default_mode", "corpus_id", "index_id", "model_id")
_NO_STORE = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


def _release_view(release: Mapping[str, Any], pipeline: SearchPipeline) -> dict[str, Any]:
    """The non-secret identity block `/health` echoes: validated, completed from the pipeline, never contradicting it."""
    if release.get("status") not in RELEASE_STATUSES:
        raise ValueError(f"release status must be one of {RELEASE_STATUSES}, got {release.get('status')!r}")
    if release.get("default_mode") not in MODES:
        raise ValueError(f"release default_mode must be one of {MODES}, got {release.get('default_mode')!r}")
    if not isinstance(release.get("release_id"), str) or not release["release_id"]:
        raise ValueError("release needs a nonempty release_id")
    view = {k: v for k, v in release.items() if k in _RELEASE_KEYS or k.endswith(("_id", "_sha256"))}
    if not all(v is None or isinstance(v, (str, int, float, bool)) for v in view.values()):
        raise ValueError("release identity fields must be JSON scalars")
    view.setdefault("corpus_id", pipeline.loaded.corpus.corpus_id)
    view.setdefault("index_id", pipeline.loaded.index_id)
    view.setdefault("model_id", getattr(pipeline.reranker, "model_id", None))
    actual = {"release_id": pipeline.release_id, "corpus_id": pipeline.loaded.corpus.corpus_id, "index_id": pipeline.loaded.index_id}
    for key, value in actual.items():
        if value is not None and view[key] != value:
            raise ValueError(f"release {key} {view[key]!r} contradicts the pipeline's {value!r}")
    return view


class SearchService:
    """One pipeline behind one admission slot and one worker thread. `deadline_at` is an absolute reading of the
    pipeline's clock, so the service uses `pipeline.clock` unless a test passes the very same clock explicitly."""

    def __init__(
        self, pipeline: SearchPipeline, *, deadline_s: float, http_grace_s: float = 1.0, release: Mapping[str, Any],
        clock: Callable[[], float] | None = None, executor: ThreadPoolExecutor | None = None,
    ) -> None:
        if not (isinstance(deadline_s, (int, float)) and math.isfinite(deadline_s) and deadline_s > 0):
            raise ValueError("deadline_s must be a positive, finite number of seconds")
        if not (isinstance(http_grace_s, (int, float)) and math.isfinite(http_grace_s) and http_grace_s >= 0):
            raise ValueError("http_grace_s must be a finite, non-negative number of seconds")
        self.pipeline = pipeline
        self.deadline_s, self.http_grace_s, self._clock = float(deadline_s), float(http_grace_s), clock or pipeline.clock
        self.release = _release_view(release, pipeline)
        self.default_mode: str = release["default_mode"]
        # A request that fits the query limit always fits the body (a JSON-escaped astral character is 12 bytes).
        self.max_body_bytes = 12 * pipeline.limits.max_query_chars + 64 * 1024
        self._admission = Admission()
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="devsearch-search")

    # -- state ---------------------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._admission.busy

    def health(self) -> dict[str, Any]:
        """Liveness plus readiness; reads flags only, so it never waits for the worker."""
        return {"alive": True, "baseline_ready": bool(self.pipeline.baseline_ready), "rerank_ready": bool(self.pipeline.rerank_ready),
                "busy": self.busy, "release": dict(self.release)}

    def ready(self, mode: str) -> bool:
        """Capability readiness; occupancy is deliberately not part of it (busy is not a failed startup)."""
        return bool(self.pipeline.baseline_ready if mode == "baseline" else self.pipeline.rerank_ready)

    @property
    def default_ready_mode(self) -> str:
        return "baseline" if self.default_mode == "baseline" else "rerank"

    def parse_request(self, payload: object) -> SearchRequest:
        return parse_search_request(payload, limits=self.pipeline.limits, default_mode=self.default_mode)

    # -- admission and execution -----------------------------------------------------------------------
    def submit(self, job: Callable[[float], T]) -> Future[T]:
        """Take the slot, then hand `job(deadline_at)` to the worker. BusyError if the slot is taken: no queue."""
        if not self._admission.try_acquire():
            raise BusyError("another search is already running; retry shortly")
        try:
            deadline_at = self._clock() + self.deadline_s            # the budget starts at admission
            future = self._executor.submit(self._work, job, deadline_at)
        except BaseException:
            self._admission.release()
            raise
        future.add_done_callback(self._release_if_never_started)
        return future

    def _work(self, job: Callable[[float], T], deadline_at: float) -> T:
        try:
            return job(deadline_at)
        finally:
            self._admission.release()                                # only the worker frees the slot, after the work ended

    def _release_if_never_started(self, future: Future) -> None:
        """Safety net: a job cancelled before it started (executor shutdown) never reaches `_work`'s finally.
        A job that ran is never `cancelled()`, so this can never free the slot of running work."""
        if future.cancelled():
            self._admission.release()

    async def run(self, job: Callable[[float], T], *, request_id: str = "-") -> T:
        """Admit and run `job`, waiting at most `deadline_s + http_grace_s` of HTTP time.

        The wait is shielded: a timeout or a cancelled handler abandons the *wait*, not the work, so the slot
        stays occupied until the worker ends. The pipeline degrades at its own deadline; this wait is the backstop
        for work that ignores it (a running kernel), answered with 504.
        """
        inner = asyncio.wrap_future(self.submit(job))
        abandoned = False

        def note_outcome(done: asyncio.Future) -> None:               # also marks the exception retrieved, so asyncio never warns
            if done.cancelled():
                return
            error = done.exception()
            if abandoned:
                if error is None:
                    logger.warning("request %s: the worker finished after its HTTP response; the slot is free again", request_id)
                else:
                    logger.error("request %s: the worker failed after its HTTP response", request_id, exc_info=error)

        inner.add_done_callback(note_outcome)
        timeout = self.deadline_s + self.http_grace_s
        try:
            return await asyncio.wait_for(asyncio.shield(inner), timeout)
        except TimeoutError:
            if inner.done():                                          # finished in the same instant, or raised its own TimeoutError
                return inner.result()
            abandoned = True
            logger.warning("request %s: no answer within the %.2f s HTTP deadline; responding 504, the slot stays occupied", request_id, timeout)
            raise DeadlineExceededError(
                f"the search did not finish within {timeout:g} s; the service stays busy until it stops") from None
        except asyncio.CancelledError:
            abandoned = True                                          # client went away: the work (and the slot) carries on
            raise

    def close(self) -> None:
        """Stop accepting work. Running work is not interrupted; an executor passed in by the caller is left alone."""
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)


# -- request parsing -----------------------------------------------------------------------------------
def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_search_request(payload: object, *, limits: SearchConfig, default_mode: str) -> SearchRequest:
    """JSON object -> SearchRequest. Checks types only: values are never clamped, coerced or repaired, and
    `null` is not "absent" (omit the field to get the configured default). Semantics belong to the pipeline."""
    if not isinstance(payload, dict):
        raise InvalidRequestError("the request body must be a JSON object")
    unknown = sorted(set(payload) - set(SEARCH_FIELDS))
    if unknown:
        raise InvalidRequestError(f"unknown field(s): {', '.join(unknown)}; expected {', '.join(SEARCH_FIELDS)}")
    if "query" not in payload:
        raise InvalidRequestError("query is required")
    if not isinstance(payload["query"], str):
        raise InvalidRequestError("query must be a string")
    sources = payload.get("sources", list(limits.sources))
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise InvalidRequestError("sources must be a list of source id strings")
    versions = payload.get("versions", dict(limits.versions))
    if not isinstance(versions, dict) or not all(isinstance(v, str) for v in versions.values()):
        raise InvalidRequestError("versions must be an object mapping a source id to one revision string")
    values = {"candidate_k": limits.candidate_k, "top_k": limits.top_k}
    for name in values:
        values[name] = payload.get(name, values[name])
        if not _is_int(values[name]):
            raise InvalidRequestError(f"{name} must be an integer")
    mode = payload.get("mode", default_mode)
    if not isinstance(mode, str):
        raise InvalidRequestError(f"mode must be one of {', '.join(MODES)}")
    return SearchRequest(query=payload["query"], sources=tuple(sources), versions=tuple(sorted(versions.items())),
                         candidate_k=values["candidate_k"], top_k=values["top_k"], mode=mode)


async def _read_json(request: Request, limit: int) -> object:
    """The JSON body, read with a size cap. Requiring the JSON content type also stops a cross-site form POST,
    which a browser sends without a preflight, from starting a search on a local service that has no auth."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise InvalidRequestError("Content-Type must be application/json")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise InvalidRequestError(f"the request body exceeds {limit} bytes")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise InvalidRequestError(f"the request body exceeds {limit} bytes")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks).decode("utf-8"))
    except (ValueError, RecursionError):                              # includes UnicodeDecodeError and JSONDecodeError
        raise InvalidRequestError("the request body is not valid UTF-8 JSON") from None


# -- responses -------------------------------------------------------------------------------------------
def error_body(request_id: str, code: str, message: str, retryable: bool) -> dict[str, Any]:
    return {"request_id": request_id, "error": {"code": code, "message": message, "retryable": retryable}}


def classify_error(error: BaseException, request_id: str) -> tuple[int, dict[str, Any]]:
    """(HTTP status, error body) per docs/04. Expected failures keep their own readable message, except that an
    index failure and anything unexpected get a fixed generic one; detail is logged, never returned."""
    if isinstance(error, InvalidRequestError):
        return 422, error_body(request_id, error.code, str(error) or "the request is invalid", error.retryable)
    if isinstance(error, BusyError):
        return 429, error_body(request_id, error.code, str(error) or "another search is already running", error.retryable)
    if isinstance(error, DeadlineExceededError):
        return 504, error_body(request_id, error.code, str(error) or "the deadline passed before a complete result existed", error.retryable)
    if isinstance(error, IndexUnavailableError):
        logger.error("request %s: index unavailable", request_id, exc_info=error)
        return 503, error_body(request_id, error.code, "The search index is not available.", error.retryable)
    logger.error("request %s: unexpected error", request_id, exc_info=error)
    return 500, error_body(request_id, "internal_error", "Internal error; details are in the local server log.", False)


def _respond(status: int, body: dict[str, Any], request_id: str) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers={**_NO_STORE, "X-Request-ID": request_id})


def _failure(error: BaseException, request_id: str) -> JSONResponse:
    status, body = classify_error(error, request_id)
    return _respond(status, body, request_id)


def create_app(service: SearchService) -> FastAPI:
    """A fresh app bound to one service; no module-level state, so apps (and their slots) are independent."""
    app = FastAPI(title="DevSearch", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = service

    @app.post("/search")
    async def search(request: Request):
        request_id = uuid.uuid4().hex
        try:
            search_request = service.parse_request(await _read_json(request, service.max_body_bytes))
            document = await service.run(
                lambda deadline_at: service.pipeline.search(search_request, deadline_at=deadline_at, request_id=request_id),
                request_id=request_id)
            return _respond(200, {**document, "request_id": request_id}, request_id)
        except Exception as error:                                    # CancelledError is not an Exception: it propagates
            return _failure(error, request_id)

    @app.get("/health")
    async def health():
        return _respond(200, service.health(), uuid.uuid4().hex)

    @app.get("/ready")
    async def ready(request: Request):
        request_id = uuid.uuid4().hex
        modes = request.query_params.getlist("mode") or [service.default_ready_mode]    # no mode: the release's default
        mode = modes[0] if len(modes) == 1 else None
        if mode not in READY_MODES:
            return _failure(InvalidRequestError(f"mode must be exactly one of {', '.join(READY_MODES)}"), request_id)
        if service.ready(mode):
            return _respond(200, {"request_id": request_id, "ready": True, "mode": mode}, request_id)
        if mode == "baseline":
            return _failure(IndexUnavailableError("baseline is not ready"), request_id)
        reason = "This release has no reranker." if service.pipeline.reranker is None else \
            "Reranking is not ready; POST /self-check to retry it, or restart the service."
        return _respond(503, error_body(request_id, "rerank_not_ready", reason, service.pipeline.reranker is not None), request_id)

    @app.post("/self-check")
    async def self_check():
        """The explicit recovery path after a model failure: same slot, same worker, same HTTP wait as a search."""
        request_id = uuid.uuid4().hex
        try:
            passed = await service.run(lambda _deadline_at: service.pipeline.self_check(), request_id=request_id)
            return _respond(200, {"rerank_ready": bool(passed)}, request_id)
        except Exception as error:
            return _failure(error, request_id)

    return app


def serve_forever(service: SearchService, host: str, port: int, *, allow_non_loopback: bool = False) -> None:
    """Run the app with uvicorn until stopped. There is no auth, so only loopback is allowed unless the caller opts in."""
    if host not in LOOPBACK_HOSTS and not allow_non_loopback:
        raise ValueError(f"refusing to bind {host!r}: the API has no authentication, so it only listens on {', '.join(LOOPBACK_HOSTS)}")
    if not _is_int(port) or not 0 <= port <= 65535:
        raise ValueError(f"port must be an integer in 0..65535, got {port!r}")
    try:
        uvicorn.run(create_app(service), host=host, port=port, log_level="warning")
    finally:
        service.close()
