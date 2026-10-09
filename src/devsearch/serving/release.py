"""Release bundles: the immutable thing that is served, transferred and backed up (docs/03 §3, docs/04, docs/10 §6).

Layout (all relative; the bundle is its own runtime root):

    release-manifest.json          sealed; lists every other file with SHA-256 and size
    corpus/ index/                 the exact corpus artifact and BM25 index being served
    model/                         inference bundle (safetensors, tokenizer, parity examples) — absent for a baseline release
    config/search.toml             the search limits the pipeline runs under
    evaluation/                    final-report.json, per-query.jsonl, report.md — evaluated releases only
    smoke/smoke.json               fixed validation queries with the expected output, re-checked after every reload

Status decides what may be activated: `baseline` (no model), `candidate` (a model, no evaluation: serving needs
`--allow-candidate`) and `evaluated` (a complete final report bound to the very model it ships). A damaged required
file blocks activation; a damaged model release is never quietly downgraded to a baseline one. Creating a bundle
stages it beside the destination and renames it into place, so a crash leaves no half-written release, and an
identical re-export is a no-op while a different one at the same path is refused.

This module is torch-free: the model is loaded through a caller-supplied factory, so baseline releases need no ML
dependency and the same loader serves the CLI, the API and the reload check.
"""

import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import loads_strict, pretty_json, read_utf8, seal_manifest, sha256_file, verify_manifest_hash
from devsearch.artifacts.identity import code_identity
from devsearch.artifacts.schema import as_object, exact_keys
from devsearch.contracts import SearchRequest
from devsearch.errors import IndexUnavailableError, ManifestError
from devsearch.evaluation.report import verify_report
from devsearch.retrieval.config import SearchConfig, load_search_config
from devsearch.retrieval.index_store import LoadedIndex, load_index
from devsearch.serving.pipeline import RerankerPort, SearchPipeline

RELEASE_SCHEMA_VERSION = 1
MANIFEST_NAME = "release-manifest.json"
STATUSES = ("baseline", "candidate", "evaluated")
FIELDS = ("status", "default_mode", "corpus", "index", "model", "search_config", "evaluation", "smoke", "files", "provenance")
SMOKE_TOP_K = 5
LOGIT_ATOL = 1e-4


@dataclass(frozen=True)
class LoadedRelease:
    manifest: dict[str, Any]
    directory: Path
    pipeline: SearchPipeline
    search_config: SearchConfig

    @property
    def release_id(self) -> str:
        return self.manifest["release_id"]


def _inventory(directory: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ManifestError(f"{path}: symbolic links are not allowed inside a release")
        if path.is_file() and path.name != MANIFEST_NAME:
            files[path.relative_to(directory).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def _identity_of(manifest_path: Path, key: str) -> str:
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))[key]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ManifestError(f"{manifest_path}: cannot read {key}: {error}") from error
    return value


def smoke_request(item: Mapping[str, Any], search_config: SearchConfig, mode: str) -> SearchRequest:
    return SearchRequest(query=item["query"], sources=tuple(item.get("sources", ())), versions=tuple(tuple(v) for v in item.get("versions", ())),
                         candidate_k=search_config.candidate_k, top_k=SMOKE_TOP_K, mode=mode)  # type: ignore[arg-type]


def record_smoke(pipeline: SearchPipeline, queries: Sequence[Mapping[str, Any]], search_config: SearchConfig) -> dict[str, Any]:
    """Run each fixed query through the pipeline and keep what must reproduce after any transfer or reload."""
    entries = []
    for item in queries:
        baseline = pipeline.search(smoke_request(item, search_config, "baseline"))
        entry = {"query": item["query"], "sources": list(item.get("sources", ())), "versions": [list(v) for v in item.get("versions", ())],
                 "candidate_set_id": baseline["candidate_set_id"], "baseline_ids": [r["chunk_id"] for r in baseline["baseline_results"]],
                 "rerank": None}
        if pipeline.rerank_ready:
            ranked = pipeline.search(smoke_request(item, search_config, "rerank"))
            if ranked["status"] == "ok":
                entry["rerank"] = {"ids": [r["chunk_id"] for r in ranked["reranked_results"]], "logits": [r["rerank_logit"] for r in ranked["reranked_results"]]}
            elif pipeline.ranking.method == "rank_fusion" and baseline["candidate_count"]:
                raise ManifestError("a fused smoke query degraded; no fusion candidate may be published")
        if pipeline.ranking.method == "rank_fusion" and baseline["candidate_count"] and entry["rerank"] is None:
            raise ManifestError("a fused smoke query has no successful neural ranking")
        entries.append(entry)
    smoke = {"schema_version": RELEASE_SCHEMA_VERSION, "kind": "smoke", "top_k": SMOKE_TOP_K, "queries": entries,
             "logit_atol": LOGIT_ATOL, "note": "validation queries only; never test material"}
    if pipeline.ranking.method == "rank_fusion":
        if not any(e["rerank"] is not None for e in entries):
            raise ManifestError("fusion smoke needs at least one nonempty successful neural ranking")
        smoke["ranking_policy"] = pipeline.ranking.to_record()
    return smoke


def check_smoke(pipeline: SearchPipeline, smoke: Mapping[str, Any], search_config: SearchConfig, *, atol: float | None = None) -> dict[str, Any]:
    """Re-run the recorded queries. BM25 must match exactly; reranking must match ids and, within `atol`, logits."""
    tolerance = float(smoke["logit_atol"] if atol is None else atol)
    failures: list[str] = []
    from devsearch.evaluation.fusion import RankingPolicy
    if RankingPolicy.from_record(smoke.get("ranking_policy")) != pipeline.ranking:
        failures.append("the ranking policy changed")
    rerank_checked = 0
    for entry in smoke["queries"]:
        baseline = pipeline.search(smoke_request(entry, search_config, "baseline"))
        if baseline["candidate_set_id"] != entry["candidate_set_id"]:
            failures.append(f"{entry['query']!r}: the candidate set changed")
        if [r["chunk_id"] for r in baseline["baseline_results"]] != entry["baseline_ids"]:
            failures.append(f"{entry['query']!r}: the BM25 top results changed")
        if pipeline.ranking.method == "rank_fusion" and baseline["candidate_count"] and entry["rerank"] is None:
            failures.append(f"{entry['query']!r}: a fused smoke query has no recorded neural ranking")
        if entry["rerank"] is not None:
            if not pipeline.rerank_ready:
                failures.append(f"{entry['query']!r}: reranking was recorded but the reranker is not ready")
                continue
            ranked = pipeline.search(smoke_request(entry, search_config, "rerank"))
            rerank_checked += 1
            if ranked["status"] != "ok":
                failures.append(f"{entry['query']!r}: reranking degraded ({[w['code'] for w in ranked['warnings']]})")
                continue
            ids = [r["chunk_id"] for r in ranked["reranked_results"]]
            logits = [r["rerank_logit"] for r in ranked["reranked_results"]]
            gap = max((abs(a - b) for a, b in zip(logits, entry["rerank"]["logits"])), default=0.0)
            if gap > tolerance:
                failures.append(f"{entry['query']!r}: a logit moved by {gap:.3g}, more than {tolerance:g}")
            elif ids != entry["rerank"]["ids"] and (pipeline.ranking.method == "rank_fusion"
                                                   or not _near_ties(entry["rerank"]["logits"], tolerance)):
                failures.append(f"{entry['query']!r}: the reranked order changed")
    if pipeline.ranking.method == "rank_fusion" and rerank_checked == 0:
        failures.append("fusion smoke replay checked no neural ranking")
    return {"passed": not failures, "failures": failures, "queries": len(smoke["queries"]), "rerank_checked": rerank_checked, "atol": tolerance}


def _near_ties(logits: Sequence[float], tolerance: float) -> bool:
    return any(abs(a - b) <= tolerance for i, a in enumerate(logits) for b in logits[i + 1:])


def export_release(
    destination: Path, *, status: str, default_mode: str, corpus_dir: Path, index_dir: Path, model_dir: Path | None, search_config_path: Path,
    report_dir: Path | None, smoke: Mapping[str, Any], provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble, verify and publish a release at `destination` (which must not exist unless it is identical)."""
    destination = Path(destination)
    if status not in STATUSES or default_mode not in ("baseline", "rerank"):
        raise ManifestError(f"status must be one of {', '.join(STATUSES)} and default_mode baseline or rerank")
    if status == "baseline" and (model_dir is not None or default_mode != "baseline"):
        raise ManifestError("a baseline release has no model and defaults to BM25")
    if status in ("candidate", "evaluated") and default_mode == "rerank" and model_dir is None:
        raise ManifestError("a release that defaults to reranking must ship its model")
    if status == "candidate" and model_dir is None:
        raise ManifestError("a candidate release exists to inspect a model; it needs one")
    if status == "evaluated" and report_dir is None:
        raise ManifestError("an evaluated release needs its final report")
    if status != "evaluated" and report_dir is not None:
        raise ManifestError("only an evaluated release carries a final report")
    if not destination.parent.is_dir():
        raise ManifestError(f"{destination.parent} does not exist; the exporter never creates parent directories")
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    staging.mkdir()
    try:
        shutil.copytree(corpus_dir, staging / "corpus", symlinks=False)
        shutil.copytree(index_dir, staging / "index", symlinks=False)
        (staging / "config").mkdir()
        shutil.copyfile(search_config_path, staging / "config" / "search.toml")
        model_id = None
        if model_dir is not None:
            shutil.copytree(model_dir, staging / "model", symlinks=False)
            model_manifest = json.loads((staging / "model" / "model-manifest.json").read_text(encoding="utf-8"))
            model_id = model_manifest["model_id"]
        evaluation = None
        if report_dir is not None:
            (staging / "evaluation").mkdir()
            for name in ("final-report.json", "per-query.jsonl", "report.md"):
                shutil.copyfile(report_dir / name, staging / "evaluation" / name)
            report = json.loads((staging / "evaluation" / "final-report.json").read_text(encoding="utf-8"))
            verify_report(report, (staging / "evaluation" / "per-query.jsonl").read_bytes())
            if report["default_mode"] != default_mode:
                raise ManifestError(f"the release defaults to {default_mode!r} but its report chose {report['default_mode']!r} on validation")
            if default_mode == "rerank":
                if model_manifest["provenance"].get("checkpoint_sha256") != report["selected"]["checkpoint_sha256"]:
                    raise ManifestError("the shipped model is not the checkpoint the final report selected on validation")
            evaluation = {"report_path": "evaluation/final-report.json", "final_report_id": report["final_report_id"],
                          "per_query_sha256": report["per_query_sha256"], "selection_id": report["selection_id"]}
        (staging / "smoke").mkdir()
        (staging / "smoke" / "smoke.json").write_text(pretty_json(smoke), encoding="utf-8")
        manifest = seal_manifest({
            "schema_version": RELEASE_SCHEMA_VERSION, "kind": "release", "status": status, "default_mode": default_mode,
            "corpus": {"path": "corpus/corpus-manifest.json"}, "index": {"path": "index/index-manifest.json"},
            "model": None if model_id is None else {"path": "model/model-manifest.json", "model_id": model_id},
            "search_config": {"path": "config/search.toml"}, "evaluation": evaluation, "smoke": {"path": "smoke/smoke.json"},
            "files": _inventory(staging), "provenance": {**dict(provenance), "code": code_identity()},
        })
        manifest = {**manifest, "release_id": "release-" + manifest["manifest_sha256"][:32]}
        (staging / MANIFEST_NAME).write_text(pretty_json(manifest), encoding="utf-8")
        verify_release(staging, allow_candidate=True)
        if destination.exists():
            existing = verify_release(destination, allow_candidate=True)
            if existing["manifest_sha256"] != manifest["manifest_sha256"]:
                raise ManifestError(f"{destination} already holds a different release ({existing['release_id']}); releases are immutable")
            return existing
        os.rename(staging, destination)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def release_directory(reference: Path) -> Path:
    """A release is named by its directory or by its manifest file."""
    reference = Path(reference)
    directory = reference.parent if reference.is_file() and reference.name == MANIFEST_NAME else reference
    if not directory.is_dir() or not (directory / MANIFEST_NAME).is_file():
        raise ManifestError(f"{reference} is neither a release directory nor its {MANIFEST_NAME}")
    return directory


def read_manifest(directory: Path) -> dict[str, Any]:
    path = Path(directory) / MANIFEST_NAME
    manifest = dict(as_object(loads_strict(read_utf8(path, where=str(path)), where=str(path)), str(path)))
    exact_keys(manifest, ("schema_version", "kind", "release_id", *FIELDS, "manifest_sha256"), str(path))
    if manifest["schema_version"] != RELEASE_SCHEMA_VERSION or manifest["kind"] != "release" or manifest["status"] not in STATUSES:
        raise ManifestError(f"{path}: not a schema-{RELEASE_SCHEMA_VERSION} release manifest")
    claimed = manifest["release_id"]
    body = {k: v for k, v in manifest.items() if k != "release_id"}
    verify_manifest_hash(body, where=str(path))
    if claimed != "release-" + manifest["manifest_sha256"][:32]:
        raise ManifestError(f"{path}: release_id does not match the manifest hash")
    if manifest["default_mode"] not in ("baseline", "rerank"):
        raise ManifestError(f"{path}: unknown default_mode")
    return manifest


def verify_release(directory: Path, *, allow_candidate: bool) -> dict[str, Any]:
    """Structure, activation rules and every file hash. Reads no tensors and needs no ML library."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    from devsearch.evaluation.fusion import RankingPolicy
    ranking = RankingPolicy.from_record(manifest["provenance"].get("ranking_policy"))
    if ranking.method == "rank_fusion" and manifest["model"] is None:
        raise ManifestError("a rank-fusion release must carry its neural model")
    if ranking.method == "rank_fusion" and manifest["status"] == "evaluated":
        raise ManifestError("rank-fusion promotion needs a fusion-aware evaluation contract; export a candidate")
    if manifest["status"] == "candidate" and not allow_candidate:
        raise IndexUnavailableError("this is a candidate release (a model without a final evaluation); activate it only with --allow-candidate")
    actual = _inventory(directory)
    if actual != manifest["files"]:
        differing = sorted((set(actual) ^ set(manifest["files"])) | {n for n in actual if n in manifest["files"] and actual[n] != manifest["files"][n]})
        raise ManifestError(f"{directory}: files differ from the release manifest: {differing[:5]}")
    status, model = manifest["status"], manifest["model"]
    if status == "baseline" and (model is not None or manifest["default_mode"] != "baseline"):
        raise ManifestError("a baseline release must not carry a model and defaults to BM25")
    if status == "candidate" and model is None:
        raise ManifestError("a candidate release is missing the model it exists to show")
    if status in ("candidate", "evaluated") and manifest["default_mode"] == "rerank" and model is None:
        raise ManifestError("a release that defaults to reranking is missing its model")
    if status == "evaluated":
        evaluation = manifest["evaluation"]
        if evaluation is None:
            raise ManifestError("an evaluated release must carry its final report")
        report = json.loads((directory / evaluation["report_path"]).read_text(encoding="utf-8"))
        verify_report(report, (directory / "evaluation" / "per-query.jsonl").read_bytes())
        if report["final_report_id"] != evaluation["final_report_id"] or report["default_mode"] != manifest["default_mode"]:
            raise ManifestError("the release manifest and its final report disagree")
        for key, shipped in (("corpus_id", _identity_of(directory / manifest["corpus"]["path"], "corpus_id")),
                             ("index_id", _identity_of(directory / manifest["index"]["path"], "index_id"))):
            if report["provenance"].get(key) != shipped:
                raise ManifestError(f"the final report was measured on a different {key[:-3]} ({report['provenance'].get(key)!r}) than the one this release ships ({shipped!r})")
    elif manifest["evaluation"] is not None:
        raise ManifestError("only an evaluated release carries a final report")
    return manifest


def load_release(
    directory: Path, *, allow_candidate: bool, reranker_factory: Callable[[Path], RerankerPort] | None = None,
    clock: Callable[[], float] | None = None,
) -> LoadedRelease:
    """Verify, then build the shared pipeline. A model release needs `reranker_factory`; it is never silently dropped."""
    directory = Path(directory).resolve(strict=True)
    manifest = verify_release(directory, allow_candidate=allow_candidate)
    try:
        loaded: LoadedIndex = load_index(directory / manifest["index"]["path"], directory)
    except ManifestError as error:
        raise IndexUnavailableError(f"the release index is invalid: {error}") from error
    search_config = load_search_config(directory / manifest["search_config"]["path"])
    reranker = None
    if manifest["model"] is not None:
        if reranker_factory is None:
            raise ManifestError("this release ships a model but no reranker loader was supplied (install the `ml` extra)")
        reranker = reranker_factory(directory / "model")
        if reranker.model_id != manifest["model"]["model_id"]:
            raise ManifestError("the loaded reranker is not the model named by the release manifest")
    kwargs = {} if clock is None else {"clock": clock}
    from devsearch.evaluation.fusion import RankingPolicy
    pipeline = SearchPipeline(loaded, search_config, reranker, release_id=manifest["release_id"],
                              ranking=RankingPolicy.from_record(manifest["provenance"].get("ranking_policy")), **kwargs)
    return LoadedRelease(manifest, directory, pipeline, search_config)


def read_smoke(directory: Path) -> dict[str, Any]:
    return dict(as_object(json.loads((Path(directory) / "smoke" / "smoke.json").read_text(encoding="utf-8")), "smoke.json"))
