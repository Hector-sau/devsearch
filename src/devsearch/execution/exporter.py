"""Building releases from the command line (docs/03 §4, docs/04 §6): the logic behind `devsearch.export`.

Two operations, both producing a new immutable bundle that is *verified by reloading it before it is published*:

* `export_candidate`  index [+ checkpoint] -> a `baseline` release (no model) or a `candidate` release (a model without a
  final evaluation; serving it needs `--allow-candidate`). Smoke queries are the ones the operator declares in the export
  config; the model's fixed parity examples are the top BM25 passage of each smoke query. Neither may be test material,
  and the exporter itself never reads any test file.
* `export_evaluated`  candidate release + a campaign's final report -> an `evaluated` release. The report must verify, name the
  same default mode, be measured on the very corpus and index being shipped and (for a reranking release) select the very
  checkpoint being shipped. A report is bound by hashes, not signed: the campaign's own `export_release` stage remains the
  authoritative path, and the provenance of an operator-supplied report says so.

The destination is staged beside itself and renamed into place after the reload checks pass; an identical rerun is a no-op
and a different bundle at the same path is refused. Nothing creates parent directories.
"""

import json
import os
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from devsearch.artifacts.canonical import sha256_file
from devsearch.artifacts.paths import ensure_inside
from devsearch.errors import ManifestError
from devsearch.modeling.devices import load_profile, resolve_device
from devsearch.modeling.io import check_parity, export_model, load_model_bundle, verify_bundle
from devsearch.modeling.jobs import ModelJob, load_model_job
from devsearch.modeling.rerank import Reranker
from devsearch.retrieval.config import SearchConfig, load_search_config
from devsearch.retrieval.index_store import LoadedIndex, load_index
from devsearch.serving.pipeline import SearchPipeline
from devsearch.serving.release import (
    check_smoke, export_release, load_release, read_smoke, record_smoke, release_directory, verify_release,
)
from devsearch.training.resolve import resolve_model

MAX_SMOKE_QUERIES = 8
SOURCE = "devsearch.export"


class ExportSettings:
    """The export config: the shared model-job schema plus `[search] path` and `[smoke] queries` (all strict, all required)."""

    def __init__(self, path: Path) -> None:
        from devsearch.artifacts.tomlconfig import as_string, as_string_list, as_table, check_keys
        self.job: ModelJob = load_model_job(path, required_extra=("search", "smoke"))
        where = str(Path(path).resolve())
        search = as_table(self.job.table["search"], f"{where}: [search]")
        smoke = as_table(self.job.table["smoke"], f"{where}: [smoke]")
        check_keys(search, ("path",), (), f"{where}: [search]")
        check_keys(smoke, ("queries",), (), f"{where}: [smoke]")
        self.search_path = (self.job.base / as_string(search["path"], f"{where}: search.path")).resolve()
        queries = as_string_list(smoke["queries"], f"{where}: smoke.queries", unique=True)
        if not 1 <= len(queries) <= MAX_SMOKE_QUERIES or not all(q.strip() and len(q) <= 1000 for q in queries):
            raise ManifestError(f"{where}: smoke.queries needs between 1 and {MAX_SMOKE_QUERIES} distinct, non-empty questions of at most 1000 characters")
        self.queries = list(queries)
        self.config_sha256 = sha256_file(Path(path))


def _reranker(model_dir: Path, device: torch.device, batch_size: int) -> Reranker:
    return Reranker(load_model_bundle(model_dir, device=device), device=device, batch_size=batch_size)


def _verify_reload(directory: Path, settings: ExportSettings, device: torch.device | None) -> dict[str, Any]:
    """What a consumer will do on first use: verify every hash, load, re-check model parity and replay the recorded smoke."""
    batch = settings.job.batch_size

    def factory(path: Path):
        if device is None:
            raise ManifestError("this release ships a model, which needs a resolved device to reload")
        return _reranker(path, device, batch)
    loaded = load_release(directory, allow_candidate=True, reranker_factory=factory)
    checks: dict[str, Any] = {"hashes_and_activation_rules": "passed", "release_id": loaded.release_id}
    if loaded.pipeline.reranker is not None:
        parity = check_parity(loaded.pipeline.reranker.loaded, device=device, batch_size=batch)
        checks["model_parity_after_reload"] = "passed" if parity["passed"] else "failed"
        if not parity["passed"]:
            raise ManifestError(f"the exported model does not reproduce its recorded logits after reload: {parity}")
    smoke = check_smoke(loaded.pipeline, read_smoke(directory), loaded.search_config)
    checks["smoke_replay"] = "passed" if smoke["passed"] else "failed"
    if not smoke["passed"]:
        raise ManifestError(f"the exported release does not reproduce its recorded smoke results: {smoke['failures'][:3]}")
    return checks


def _publish(staged: Path, destination: Path, manifest: Mapping[str, Any]) -> bool:
    """Rename the verified bundle into place. Returns False for an identical existing bundle (nothing changed)."""
    if destination.exists():
        existing = verify_release(destination, allow_candidate=True)
        if existing["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ManifestError(f"{destination} already holds a different release ({existing['release_id']}); releases are immutable")
        return False
    os.rename(staged, destination)
    return True


class _Work:
    """A private staging directory beside the destination, removed whatever happens."""

    def __init__(self, destination: Path) -> None:
        if not destination.parent.is_dir():
            raise ManifestError(f"{destination.parent} does not exist; the exporter never creates parent directories")
        self.path = destination.with_name(f".{destination.name}.export-{os.getpid()}-{uuid.uuid4().hex[:8]}")

    def __enter__(self) -> Path:
        self.path.mkdir()
        return self.path

    def __exit__(self, *_exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)


def _index_layout(index_manifest: Path, loaded: LoadedIndex) -> tuple[Path, Path]:
    index_dir = index_manifest.parent
    corpus_dir = (index_dir / loaded.manifest["corpus"]["manifest_path"]).resolve().parent
    return corpus_dir, index_dir


def export_candidate(*, config: Path, index_manifest: Path, checkpoint: Path | None, output: Path, runtime_root: Path) -> dict[str, Any]:
    settings = ExportSettings(config)
    output = ensure_inside(Path(output), runtime_root, what="release output")
    index_manifest = ensure_inside(Path(index_manifest), runtime_root, what="index manifest")
    search_path = ensure_inside(settings.search_path, runtime_root, what="search config")
    loaded = load_index(index_manifest, runtime_root)
    corpus_dir, index_dir = _index_layout(index_manifest, loaded)
    search_config: SearchConfig = load_search_config(search_path)
    smoke_queries = [{"query": q} for q in settings.queries]
    base_pipeline = SearchPipeline(loaded, search_config, None)
    device = None
    with _Work(output) as work:
        model_dir, provenance = None, {"source": SOURCE, "export_config_sha256": settings.config_sha256, "corpus_id": loaded.corpus.corpus_id,
                                       "index_id": loaded.index_id, "inference_batch": settings.job.batch_size,
                                       "smoke": "operator-declared questions; never test material"}
        if checkpoint is not None:
            device = resolve_device(load_profile(settings.job.profile))
            model_dir = _model_bundle(settings, Path(checkpoint), base_pipeline, search_config, device, work, provenance)
        pipeline = SearchPipeline(loaded, search_config, None if model_dir is None else _reranker(model_dir, device, settings.job.batch_size))
        smoke = record_smoke(pipeline, smoke_queries, search_config)
        status, mode = ("baseline", "baseline") if model_dir is None else ("candidate", "rerank")
        manifest = export_release(work / "release", status=status, default_mode=mode, corpus_dir=corpus_dir, index_dir=index_dir, model_dir=model_dir,
                                  search_config_path=search_path, report_dir=None, smoke=smoke, provenance=provenance)
        checks = _verify_reload(work / "release", settings, device)
        published = _publish(work / "release", output, manifest)
    return {"status": manifest["status"], "default_mode": manifest["default_mode"], "release_id": manifest["release_id"],
            "model_id": None if manifest["model"] is None else manifest["model"]["model_id"], "output": str(output), "published": published,
            "reload_checks": checks}


def _model_bundle(settings: ExportSettings, reference: Path, pipeline: SearchPipeline, search_config: SearchConfig, device: torch.device,
                  work: Path, provenance: dict[str, Any]) -> Path:
    """An inference bundle for `reference`: an existing bundle is verified and used; a training checkpoint is exported."""
    if reference.is_dir():
        manifest = verify_bundle(reference)
        provenance["model_source"] = {"kind": "inference-bundle", "model_id": manifest["model_id"]}
        return reference
    resolved = resolve_model(reference, device=device, assets=settings.job.assets, policy=settings.job.policy, dropout=settings.job.dropout)
    details = dict(resolved.details)
    if details.get("resume_kind") != "epoch_boundary" or int(details.get("completed_epoch", 0)) < 1:
        raise ManifestError("only a checkpoint saved at the end of a completed training epoch can be exported (not epoch zero and not a mid-epoch snapshot)")
    examples: list[tuple[str, str]] = []
    for query in settings.queries:
        top = pipeline.search(_baseline_request(query, search_config))["baseline_results"]
        if top:
            examples.append((query, top[0]["text"]))
    if not examples:
        raise ManifestError("none of the smoke questions retrieved a passage, so the model's parity examples cannot be formed; declare questions the corpus answers")
    destination = work / "model"
    manifest = export_model(resolved.model, destination, tokenizer=resolved.tokenizer, policy=resolved.policy, device=device,
                            batch_size=settings.job.batch_size, parity_examples=examples,
                            provenance={"source": SOURCE, "checkpoint_sha256": resolved.checkpoint_sha256, **details})
    parity = check_parity(load_model_bundle(destination, device=device), device=device, batch_size=settings.job.batch_size)
    if not parity["passed"]:
        raise ManifestError(f"the exported model does not reproduce its recorded logits after reload: {parity}")
    provenance["model_source"] = {"kind": "training-checkpoint", "model_id": manifest["model_id"], **details}
    return destination


def _baseline_request(query: str, search_config: SearchConfig):
    from devsearch.contracts import SearchRequest
    return SearchRequest(query=query, sources=(), versions=(), candidate_k=search_config.candidate_k, top_k=1, mode="baseline")


def export_evaluated(*, config: Path, candidate: Path, final_report: Path, output: Path, runtime_root: Path) -> dict[str, Any]:
    settings = ExportSettings(config)
    output = ensure_inside(Path(output), runtime_root, what="release output")
    source = ensure_inside(release_directory(candidate), runtime_root, what="candidate release")
    report_path = ensure_inside(Path(final_report), runtime_root, what="final report")
    base = verify_release(source, allow_candidate=True)
    if base["status"] == "evaluated":
        raise ManifestError("this release is already evaluated; evaluate a candidate or baseline release")
    if report_path.name != "final-report.json" or not report_path.is_file():
        raise ManifestError(f"{report_path}: expected the campaign's final-report.json (beside per-query.jsonl and report.md)")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        default_mode = report["default_mode"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ManifestError(f"{report_path}: not a final report: {error}") from error
    shipped_model = default_mode == "rerank"
    if shipped_model and base["model"] is None:
        raise ManifestError("the report chose reranking but the candidate ships no model")
    loaded = load_index(source / base["index"]["path"], source)
    search_config = load_search_config(source / base["search_config"]["path"])
    queries = [{k: v for k, v in item.items() if k in ("query", "sources", "versions")} for item in read_smoke(source)["queries"]]
    device = resolve_device(load_profile(settings.job.profile)) if shipped_model else None
    with _Work(output) as work:
        model_dir = source / "model" if shipped_model else None
        pipeline = SearchPipeline(loaded, search_config, None if model_dir is None else _reranker(model_dir, device, settings.job.batch_size))
        smoke = record_smoke(pipeline, queries, search_config)
        provenance = {"source": SOURCE, "export_config_sha256": settings.config_sha256, "candidate_release_id": base["release_id"],
                      "inference_batch": settings.job.batch_size,
                      "report_origin": "operator-supplied final report, verified by hash and bound to this corpus, index and checkpoint; "
                                       "the campaign's export_release stage is the authoritative path",
                      "smoke": "the candidate release's own questions; never test material"}
        manifest = export_release(work / "release", status="evaluated", default_mode=default_mode, corpus_dir=source / "corpus", index_dir=source / "index",
                                  model_dir=model_dir, search_config_path=source / base["search_config"]["path"], report_dir=report_path.parent, smoke=smoke,
                                  provenance=provenance)
        checks = _verify_reload(work / "release", settings, device)
        published = _publish(work / "release", output, manifest)
    return {"status": manifest["status"], "default_mode": manifest["default_mode"], "release_id": manifest["release_id"],
            "model_id": None if manifest["model"] is None else manifest["model"]["model_id"], "final_report_id": manifest["evaluation"]["final_report_id"],
            "output": str(output), "published": published, "reload_checks": checks}
