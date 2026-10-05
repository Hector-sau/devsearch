"""Starting a verified release for serving (docs/04 "Search API" and "Release behavior"): the logic behind `devsearch.serve`.

A release is served only after, in this order: its structure and every file hash verify; candidate status is explicitly
allowed; the requested device resolves (an unavailable accelerator is an error, never a CPU fallback); an accelerator profile
holds the exclusive accelerator lease (training and serving never share the accelerator); the pipeline loads; a model
release's reranker passes its recorded parity self-check; and the recorded smoke replays. Any failure refuses to start. A
release that ships a model is never quietly served as a baseline one.

Everything here is torch-free until a model release needs a device; the HTTP layer needs the `serve` extra.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devsearch.artifacts.paths import ensure_inside
from devsearch.artifacts.tomlconfig import as_number, as_string, as_table, check_keys
from devsearch.errors import ConfigError, ManifestError, PrerequisiteError
from devsearch.execution.exporter import candidate_directory
from devsearch.execution.lock import Lease
from devsearch.modeling.devices import load_profile
from devsearch.modeling.jobs import ModelJob, load_model_job
from devsearch.serving.release import check_smoke, load_release, read_smoke, verify_release


@dataclass(frozen=True)
class ServeSettings:
    job: ModelJob
    host: str
    port: int
    deadline_s: float
    http_grace_s: float
    lease_path: Path | None

    @classmethod
    def load(cls, path: Path) -> "ServeSettings":
        job = load_model_job(path, required_extra=("serve",))
        where = str(Path(path).resolve())
        section = as_table(job.table["serve"], f"{where}: [serve]")
        check_keys(section, ("host", "port", "deadline_seconds", "http_grace_seconds"), ("lease_path",), f"{where}: [serve]")
        port = section["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ConfigError(f"{where}: serve.port must be an integer in 1..65535")
        lease = section.get("lease_path")
        return cls(job, as_string(section["host"], f"{where}: serve.host", max_length=255), port,
                   as_number(section["deadline_seconds"], f"{where}: serve.deadline_seconds", 0.001, 3600.0),
                   as_number(section["http_grace_seconds"], f"{where}: serve.http_grace_seconds", 0.0, 3600.0),
                   None if lease is None else (job.base / as_string(lease, f"{where}: serve.lease_path")).resolve())


@dataclass
class StartedService:
    service: Any
    release: dict[str, Any]
    settings: ServeSettings
    startup: dict[str, Any]
    lease: Lease | None

    def stop(self) -> None:
        self.service.close()
        if self.lease is not None:
            self.lease.release()


def _reranker_factory(device: Any, batch_size: int) -> Callable[[Path], Any]:
    from devsearch.modeling.io import load_model_bundle
    from devsearch.modeling.rerank import Reranker

    def factory(path: Path) -> Any:
        return Reranker(load_model_bundle(path, device=device), device=device, batch_size=batch_size)
    return factory


def start_service(*, config: Path, manifest: Path, allow_candidate: bool, runtime_root: Path) -> StartedService:
    settings = ServeSettings.load(config)
    try:
        from devsearch.serving.api import LOOPBACK_HOSTS, SearchService
    except ImportError as error:
        raise PrerequisiteError(f"serving needs the `serve` extra ({error.name} is not installed)") from error
    if settings.host not in LOOPBACK_HOSTS:
        raise ConfigError(f"serve.host {settings.host!r} is refused: the API has no authentication, so it only listens on {', '.join(LOOPBACK_HOSTS)}")
    directory = ensure_inside(candidate_directory(manifest), runtime_root, what="release")
    record = verify_release(directory, allow_candidate=allow_candidate)       # raises IndexUnavailableError for an un-allowed candidate
    profile = load_profile(settings.job.profile)
    needs_device = record["model"] is not None
    lease: Lease | None = None
    if needs_device and profile.accelerator:
        if settings.lease_path is None:
            raise ConfigError("serving a model on an accelerator must share the campaign's accelerator lease: set [serve] lease_path "
                              "to <run_root>/accelerator.lease so training and serving cannot overlap")
        lease = Lease(ensure_inside(settings.lease_path, runtime_root, what="accelerator lease"), purpose="serve", plan_id=record["release_id"])
        lease.acquire()
    try:
        device, factory = None, None
        if needs_device:
            from devsearch.modeling.devices import resolve_device
            device = resolve_device(profile)                      # raises DeviceUnavailable: never a CPU fallback
            factory = _reranker_factory(device, settings.job.batch_size)
        loaded = load_release(directory, allow_candidate=allow_candidate, reranker_factory=factory)
        smoke = check_smoke(loaded.pipeline, read_smoke(directory), loaded.search_config)
        if not smoke["passed"]:
            raise ManifestError(f"the release does not reproduce its recorded smoke results, so it will not be served: {smoke['failures'][:3]}")
        checks: dict[str, Any] = {"smoke_replay": "passed", "queries": smoke["queries"], "rerank_checked": smoke["rerank_checked"]}
        if loaded.pipeline.reranker is not None:
            if not loaded.pipeline.self_check():
                raise ManifestError("the release's reranker failed its recorded parity self-check, so it will not be served (and is not downgraded to BM25)")
            checks["rerank_self_check"] = "passed"
        release = {"status": record["status"], "release_id": record["release_id"], "default_mode": record["default_mode"]}
        service = SearchService(loaded.pipeline, deadline_s=settings.deadline_s, http_grace_s=settings.http_grace_s, release=release)
    except BaseException:
        if lease is not None:
            lease.release()
        raise
    startup = {"release_id": record["release_id"], "status": record["status"], "default_mode": record["default_mode"],
               "model_id": None if record["model"] is None else record["model"]["model_id"], "device": None if device is None else str(device),
               "profile": profile.name, "host": settings.host, "port": settings.port, "deadline_seconds": settings.deadline_s,
               "accelerator_lease": None if lease is None else str(lease.path), "checks": checks}
    return StartedService(service, record, settings, startup, lease)
