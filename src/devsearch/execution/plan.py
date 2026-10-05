"""The immutable resolved execution plan (docs/10 §5): everything fixed before the first training step.

A plan is created only after the doctor's gates allow it, and never changes afterwards: a formal run changes
run-state and artifacts, never the plan or the source code. It carries hashes and policies, not host paths
(those live in the separate host-binding records) and not future outputs: later artifacts are *logical* stage
outputs bound afterwards by stage manifests and the immutable `validation-selection.json`.

Refusals (nothing is written): a doctor report for another host/bundle/code, a failed or unhealthy doctor, an
uncalibrated profile, a formal campaign whose gates are not all `passed` (diagnostic bundles may only make a
rehearsal plan, flagged `rehearsal: true`), or a window too short for the estimated work. Time and storage figures
are estimates derived from measured rates and are labelled as such; ranking quality is never estimated.
"""

import math
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import pretty_json, seal_manifest, sha256_file, sha256_text, verify_manifest_hash
from devsearch.artifacts.files import write_atomic
from devsearch.artifacts.identity import code_identity
from devsearch.data.pairs import load_pairs
from devsearch.errors import ManifestError, PrerequisiteError
from devsearch.evaluation.protocol import Protocol
from devsearch.execution.acceptance import evaluate_gates
from devsearch.execution.bundle import verify_bundle
from devsearch.execution.hostfile import HostConfig, binding_for
from devsearch.execution.records import SCHEMA_VERSION, parse_utc, read_json, read_sealed, write_json
from devsearch.modeling.devices import DeviceProfile, load_profile
from devsearch.modeling.tokenization import WordPieceTokenizer

PLAN_FIELDS = ("rehearsal", "code", "bundle", "profile", "backend", "model_assets", "data", "initialization", "matrix", "training", "policies",
               "stages", "retries", "budget", "capacity", "host_binding", "gates", "doctor_report_sha256")
UNCERTAINTY_MARGIN = 1.25
PLAN_NAME = "execution-plan.json"


def stage_graph(protocol: Protocol, evaluation_protocol: str) -> list[dict[str, Any]]:
    """The fixed stage list and dependencies. The plan stores it; the runner builds its stages from the same function."""
    runs = protocol.run_ids()
    train = [f"train:{recipe}:{seed}" for recipe, seed in runs]
    export = [f"export:{recipe}:{seed}" for recipe, seed in runs]
    stages: list[dict[str, Any]] = [
        {"stage_id": "preflight", "needs": []}, {"stage_id": "data_check", "needs": ["preflight"]},
        {"stage_id": "baseline_validation", "needs": ["data_check"]},
    ]
    stages += [{"stage_id": t, "needs": ["data_check"]} for t in train]
    stages += [{"stage_id": e, "needs": [f"train:{e.split(':', 1)[1]}"]} for e in export]
    deployment = [f"export:{recipe.recipe_id}:{protocol.deployment_seed}" for recipe in protocol.recipes]
    stages.append({"stage_id": "latency", "needs": [*deployment, "baseline_validation"]})
    stages.append({"stage_id": "select", "needs": ["baseline_validation", "latency", *train]})
    stages.append({"stage_id": "rank_test", "needs": ["select", *export]})
    if evaluation_protocol == "prejudged_fixed_candidates_v1":
        stages.append({"stage_id": "seal_final", "needs": ["rank_test"]})
    else:
        stages += [{"stage_id": "pool", "needs": ["rank_test"]}, {"stage_id": "review_gate", "needs": ["pool"]}, {"stage_id": "seal_final", "needs": ["review_gate"]}]
    stages += [
        {"stage_id": "score", "needs": ["seal_final", "select", "rank_test"]},
        {"stage_id": "export_release", "needs": ["score", "select", *export]},
        {"stage_id": "reload_verify", "needs": ["export_release"]},
        {"stage_id": "backup", "needs": ["reload_verify", "score"]},
        {"stage_id": "summary", "needs": ["backup"]},
    ]
    return stages


def estimate_stages(
    protocol: Protocol, doctor: Mapping[str, Any], *, pairs: int, validation_candidates: int, test_candidate_bound: int, validation_queries: int,
    candidate_depth: int,
) -> dict[str, Any]:
    """Seconds per stage from measured rates (None = unverified, never zero) and the total with a declared margin."""
    estimates = doctor["estimates"]
    infer = estimates.get("inference_pairs_per_second")
    io = estimates.get("io") or {}
    per_stage: dict[str, float | None] = {}
    for recipe, seed in protocol.run_ids():
        rate = (estimates["train"].get(recipe) or {}).get("pairs_per_second")
        known = rate and infer and io
        per_stage[f"train:{recipe}:{seed}"] = (
            protocol.epochs * (pairs / rate + validation_candidates / infer + io["checkpoint_save_seconds"]) if known else None)
        per_stage[f"export:{recipe}:{seed}"] = (io["export_seconds"] + io["reload_seconds"]) if io else None
    search_seconds = (candidate_depth / infer) if infer else None
    queries = min(protocol.latency_queries, validation_queries)
    per_stage["latency"] = ((protocol.latency_warmup + protocol.latency_passes * queries) * search_seconds * len(protocol.recipes)) if search_seconds else None
    per_stage["rank_test"] = (len(protocol.run_ids()) * test_candidate_bound / infer) if infer else None
    known = [v for v in per_stage.values() if v is not None]
    total = sum(known) * UNCERTAINTY_MARGIN
    return {"label": "estimate", "uncertainty_margin": UNCERTAINTY_MARGIN, "stage_seconds": per_stage,
            "total_seconds": total, "unverified_stages": sorted(k for k, v in per_stage.items() if v is None),
            "note": "derived from measured rates on diagnostic-length synthetic pairs; stages without a measured rate are unverified, not zero"}


def profile_sha256(profile: DeviceProfile) -> str:
    return sha256_text(resources.files("devsearch.profiles").joinpath(f"{profile.name}.toml").read_text(encoding="utf-8"))


def build_plan(
    host: HostConfig, bundle_dir: Path, doctor_report: Mapping[str, Any], *, receipts_dir: Path | None = None, now: str | None = None,
    wheel_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (plan, initial host binding) or refuse. Pure: writes nothing."""
    verify_manifest_hash(doctor_report, where="doctor report")
    binding = binding_for(host)
    if doctor_report.get("kind") != "doctor-report" or doctor_report["profile"] != host.profile:
        raise ManifestError("the doctor report is not for this device profile")
    if doctor_report["host_binding_sha256"] != binding["manifest_sha256"]:
        raise ManifestError("the doctor report was made for a different host file; rerun the doctor on this host")
    verified = verify_bundle(bundle_dir)
    bundle = verified["manifest"]
    if doctor_report["bundle_manifest_sha256"] != bundle["manifest_sha256"]:
        raise ManifestError("the doctor report was made for a different bundle; rerun the doctor")
    running = code_identity()
    if doctor_report["code"]["source_snapshot_sha256"] != running["source_snapshot_sha256"]:
        raise ManifestError("the code changed since the doctor ran; the plan identifies exactly the code that will execute")
    failed = [c for c in doctor_report["checks"] if c["status"] == "failed"]
    if failed or doctor_report["unhealthy_host"]:
        raise PrerequisiteError("the doctor found failures: " + ("; ".join(f"{c['name']}: {c['detail']}" for c in failed[:5]) or "the host is unhealthy"))
    calibration = doctor_report["calibration"]
    if not calibration["microbatch"] or not calibration["inference_batch"] or calibration.get("stopped"):
        raise PrerequisiteError("the profile is not calibrated: " + str(calibration.get("stopped") or "no microbatch or inference batch was selected"))
    protocol: Protocol = verified["protocol"]
    profile = load_profile(host.profile)
    diagnostic = bool(bundle["diagnostic"])
    gates = evaluate_gates(doctor_report, receipts_dir, running)
    blocking = {name: gate for name, gate in gates.items() if gate["status"] not in ("passed", "not_applicable")}
    if not diagnostic and blocking:
        detail = "; ".join(f"{name} is {gate['status']} ({gate['detail']})" for name, gate in blocking.items())
        raise PrerequisiteError(f"a formal plan needs every gate passed: {detail}")
    prepared = verified["prepared"]
    loaded = prepared["loaded"]
    base = prepared["directory"]
    _, pairs, _ = load_pairs(base / "pairs" / "pairs-manifest.json", loaded.corpus)
    from devsearch.data.candidates import load_candidates
    validation_manifest, validation_records = load_candidates(base / "candidates" / "candidates-validation-manifest.json", loaded)
    commitment = prepared["manifest"]["evaluation_commitment"]
    depth = validation_manifest["candidate_k"] or 50
    estimates = estimate_stages(protocol, doctor_report, pairs=len(pairs), validation_candidates=validation_manifest["counts"]["candidates"],
                                test_candidate_bound=commitment["test_queries"] * depth, validation_queries=len(validation_records), candidate_depth=depth)
    deadline = parse_utc(binding["deadline"], "deadline")
    now_time = parse_utc(now, "now") if now else parse_utc(doctor_report["created_at"], "doctor time")
    window = (deadline - now_time).total_seconds() - binding["reserve_seconds"]
    if estimates["total_seconds"] > window:
        raise PrerequisiteError(f"insufficient time: about {estimates['total_seconds']:.0f}s of known work against a {window:.0f}s window after the reserve")
    training = dict(protocol.training)
    microbatch = calibration["microbatch"]
    tokenizer = WordPieceTokenizer(Path(bundle_dir) / "model-assets")
    seal = verified["seal"]
    plan = seal_manifest({
        "schema_version": SCHEMA_VERSION, "kind": "execution-plan", "rehearsal": diagnostic,
        "code": {"identity": running, "wheel_sha256": wheel_sha256, "lock_sha256": (bundle["lock"] or {}).get("sha256")},
        "bundle": {"manifest_sha256": bundle["manifest_sha256"], "protocol_id": protocol.protocol_id, "protocol_sha256": bundle["protocol"]["sha256"],
                   "diagnostic": diagnostic},
        "profile": {"name": profile.name, "sha256": profile_sha256(profile), "device": profile.device, "accelerator": profile.accelerator,
                    "declared_verification": profile.verification},
        "backend": {"fingerprint": calibration.get("backend"), "doctor_backend_gate": gates[f"backend_verified[{profile.name}]"]["status"]},
        "model_assets": {"manifest_sha256": bundle["model_assets"]["manifest_sha256"], "asset_kind": bundle["model_assets"]["asset_kind"],
                         "pinned": bundle["model_assets"]["pinned"], "tokenizer_fingerprint": tokenizer.fingerprint,
                         "pair_special_tokens": tokenizer.pair_special_tokens},
        "data": {"corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id, "prepared_manifest_sha256": prepared["manifest"]["manifest_sha256"],
                 "split_manifest_sha256": prepared["manifest"]["split_manifest_sha256"], "pairs_manifest_sha256": prepared["manifest"]["pairs"]["manifest_sha256"],
                 "sampling_manifest_id": prepared["manifest"]["pairs"]["sampling_manifest_id"],
                 "validation_candidates_manifest_sha256": prepared["manifest"]["validation_candidates"]["manifest_sha256"],
                 "initial_seal_sha256": seal["manifest_sha256"], "evaluation_commitment_sha256": commitment["manifest_sha256"],
                 "evaluation_protocol": prepared["manifest"]["protocol"], "counts": {**prepared["manifest"]["counts"], "pairs": len(pairs),
                                                                                       "validation_candidates": validation_manifest["counts"]["candidates"]}},
        "initialization": {"encoder": "re-loaded from the pinned assets for every run; diagnostic weights are discarded", "head_seed": protocol.head_seed,
                           "diagnostic_weights_discarded": True, "resume": "epoch_boundary checkpoints only"},
        "matrix": [{"stage_id": f"train:{recipe.recipe_id}:{seed}", "recipe_id": recipe.recipe_id, "mode": recipe.mode, "seed": seed,
                    "epochs": protocol.epochs, "learning_rate": recipe.learning_rate} for recipe in protocol.recipes for seed in protocol.seeds],
        "training": {"precision": profile.precision, "workers": profile.workers, "effective_batch": training["effective_batch"], "microbatch": microbatch,
                     "accumulation_steps": training["effective_batch"] // microbatch, "max_pair_tokens": protocol.max_pair_tokens,
                     "max_query_tokens": protocol.max_query_tokens,
                     "passage_budget": protocol.max_pair_tokens - protocol.max_query_tokens - tokenizer.pair_special_tokens,
                     "inference_batch": calibration["inference_batch"], "shared": dict(training)},
        "policies": {"protocol_id": protocol.protocol_id, "selection": protocol.to_record()["selection"], "latency": protocol.to_record()["latency"],
                     "bootstrap": protocol.to_record()["bootstrap"], "judging": {"protocol": prepared["manifest"]["protocol"], "no_test_based_selection": True},
                     "search": {"candidate_k": depth, "top_k": 5, "max_candidate_k": max(100, depth), "max_query_chars": 1000}},
        "stages": stage_graph(protocol, prepared["manifest"]["protocol"]),
        "retries": {"setup_io": host.io_retries, "training": 0, "note": "OOM, non-finite values and device loss stop with evidence; a changed recipe needs a new plan"},
        "budget": {"initial_deadline": binding["deadline"], "initial_reserve_seconds": binding["reserve_seconds"], "estimates": estimates,
                   "window_seconds_at_planning": window},
        "capacity": doctor_report["estimates"].get("capacity"),
        "host_binding": {"id": "binding-" + binding["manifest_sha256"][:32], "sha256": binding["manifest_sha256"]},
        "gates": {name: {"status": gate["status"], "label": gate.get("label"), "detail": gate.get("detail"), "evidence": gate.get("evidence", [])}
                  for name, gate in sorted(gates.items())},
        "doctor_report_sha256": doctor_report["manifest_sha256"],
    })
    return plan, binding


def plan_id(plan: Mapping[str, Any]) -> str:
    return "plan-" + str(plan["manifest_sha256"])[:32]


def write_plan(output: Path, plan: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    """Write the plan and its revision-0 host binding side by side. Neither is ever overwritten with different content."""
    output = Path(output)
    if not output.parent.is_dir():
        raise ManifestError(f"{output.parent} does not exist; the planner never creates directories")
    binding_path = output.parent / f"host-binding-{binding['revision']}.json"
    for path, value in ((binding_path, binding), (output, plan)):
        if path.exists():
            if read_sealed_any(path)["manifest_sha256"] != value["manifest_sha256"]:
                raise ManifestError(f"{path} already exists with different content; plans and bindings are immutable")
    if not binding_path.exists():
        write_json(binding_path, binding, exclusive=True)
    if not output.exists():
        write_atomic(output, pretty_json(plan).encode("utf-8"), exclusive=True)


def read_sealed_any(path: Path) -> dict[str, Any]:
    data = read_json(Path(path), str(path))
    verify_manifest_hash(data, where=str(path))
    return data


def load_plan(path: Path) -> tuple[dict[str, Any], str]:
    """Strict read: schema, unknown fields, hash. Returns the plan and its manifest hash."""
    plan = read_sealed(Path(path), "execution-plan", PLAN_FIELDS, str(path))
    if not isinstance(plan["stages"], list) or not plan["stages"]:
        raise ManifestError(f"{path}: the stage graph is empty")
    for entry in plan["stages"]:
        if set(entry) != {"stage_id", "needs"}:
            raise ManifestError(f"{path}: malformed stage entry {entry!r}")
    if not math.isfinite(plan["budget"]["estimates"]["total_seconds"]):
        raise ManifestError(f"{path}: the time estimate is not finite")
    return plan, plan["manifest_sha256"]


def plan_file_sha256(path: Path) -> str:
    return sha256_file(Path(path))
