"""`doctor`: preflight and calibration on the target host, before any plan exists (docs/10 §3–§4, docs/12 §6).

It checks the host binding, verifies the prepared bundle and the pinned assets, measures the real tokenizer's
pair special-token count and the corpus against the joint length budget, and — on the profile's explicitly
requested device — calibrates in fresh worker processes: each declared microbatch is probed once (three full
optimizer updates at the maximum length) and the first that satisfies the memory cap wins; then five warm-up plus
fifty measured updates per recipe mode, the inference batch chosen from its own finite probe, and
save/load/export/reload timings. Test queries, candidates and judgments are never opened.

Every result carries a label: `measured` (taken on this host with these assets), `fixture` (diagnostic assets or
data, which prove mechanics only) or `declared` (stated in a file, not observed). The report is evidence, not
approval: gates are `passed`, `failed`, `unverified` or `not_applicable`, and the backend gate passes only for
real pinned assets exercised on the real accelerator. Unknown throughput or free space is `unverified`, never zero.

Limits (host file, recorded): 300 s per worker, 1,800 s for the whole run, 30 s cleanup grace, at most two
transient setup-I/O retries inside the same budget. A worker that cannot be confirmed gone after termination marks
the host unhealthy and no further worker starts.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import seal_manifest
from devsearch.artifacts.identity import code_identity
from devsearch.errors import ManifestError
from devsearch.execution.bundle import verify_bundle
from devsearch.execution.hostfile import HostConfig, binding_for, run_checks
from devsearch.execution.records import SCHEMA_VERSION, utc_now
from devsearch.modeling.devices import DeviceProfile, load_profile
from devsearch.modeling.policy import PairEncodingPolicy, check_corpus_budget
from devsearch.modeling.tokenization import WordPieceTokenizer

DOCTOR_FIELDS = ("profile", "host_binding_sha256", "bundle_manifest_sha256", "diagnostic", "code", "checks", "calibration", "gates", "estimates",
                 "limits", "elapsed_seconds", "created_at", "unhealthy_host")
GATES = ("implementation_verified", "data_ready", "portable_bundle_verified", "backend_verified")


class Budget:
    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds, self._clock, self._start = float(seconds), clock, clock()

    def used(self) -> float:
        return self._clock() - self._start

    def remaining(self) -> float:
        return self.seconds - self.used()


PROBE_MODULE = "devsearch.execution.probe"


def run_worker(spec: Mapping[str, Any], *, timeout: float, grace: float, retries: int = 2, module: str = PROBE_MODULE) -> dict[str, Any]:
    """Run one probe in a fresh process. Returns the probe result or a failure record; never raises for probe problems."""
    last_error: OSError | None = None
    for _ in range(retries + 1):                               # transient setup I/O (temp files, spawn) only
        try:
            return _run_once(spec, timeout=timeout, grace=grace, module=module)
        except OSError as error:
            last_error = error
    return {"ok": False, "reason": "setup_io", "error": str(last_error)}


def _run_once(spec: Mapping[str, Any], *, timeout: float, grace: float, module: str = PROBE_MODULE) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="devsearch-probe-") as scratch:
        spec_path, result_path = Path(scratch) / "spec.json", Path(scratch) / "result.json"
        spec_path.write_text(json.dumps({**spec, "scratch": scratch}), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        process = subprocess.Popen([sys.executable, "-m", module, str(spec_path), str(result_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()                                # request termination; no promise of instant cancellation
            try:
                process.communicate(timeout=grace)
                gone = True
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.communicate(timeout=grace)
                    gone = True
                except subprocess.TimeoutExpired:
                    gone = False
            return {"ok": False, "reason": "timeout", "timeout_seconds": timeout, "worker_gone": gone, "unhealthy": not gone}
        if result_path.is_file():
            return json.loads(result_path.read_text(encoding="utf-8"))
        return {"ok": False, "reason": "crashed", "returncode": process.returncode, "stderr_tail": (stderr or "")[-2000:]}


def _check(name: str, status: str, detail: str, label: str = "measured") -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail, "label": label}


def _physical_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _training_spec(common: Mapping[str, Any], protocol, recipe, microbatch: int) -> dict[str, Any]:
    return {**common, "mode": recipe.mode, "learning_rate": recipe.learning_rate, "microbatch": microbatch}


def calibrate(
    host: HostConfig, profile: DeviceProfile, assets: Path, protocol, budget: Budget, *, worker: Callable[..., dict[str, Any]] = run_worker,
) -> dict[str, Any]:
    """Probe microbatches, then time training (per recipe mode), inference and I/O. All probes share one budget."""
    training = dict(protocol.training)
    common = {"profile": profile.name, "assets": str(assets), "max_pair_tokens": protocol.max_pair_tokens, "max_query_tokens": protocol.max_query_tokens,
              "seed": 7, "head_seed": protocol.head_seed, "effective_batch": training["effective_batch"], "dropout": training["dropout"]}
    result: dict[str, Any] = {"attempts": [], "microbatch": None, "inference_batch": None, "training": {}, "inference": None, "io": None,
                              "stopped": None, "backend": None}
    state = {"unhealthy": False}

    def launch(spec: dict[str, Any]) -> dict[str, Any] | None:
        if state["unhealthy"]:
            result["stopped"] = "the host was marked unhealthy after a worker could not be confirmed gone; no new worker starts"
            return None
        if budget.remaining() <= 0:
            result["stopped"] = f"the doctor budget of {budget.seconds:.0f}s is exhausted"
            return None
        outcome = worker(spec, timeout=min(host.process_timeout_seconds, max(budget.remaining(), 1)), grace=host.cleanup_grace_seconds, retries=host.io_retries)
        if outcome.get("unhealthy"):
            state["unhealthy"] = True
        if outcome.get("backend") and result["backend"] is None:
            result["backend"] = outcome["backend"]
        return outcome

    def over_cap(outcome: Mapping[str, Any]) -> str | None:
        memory = outcome.get("memory", {})
        if profile.accelerator and host.device_memory_cap_bytes and memory.get("device_peak_bytes", 0) > host.device_memory_cap_bytes:
            return f"device peak {memory['device_peak_bytes']} bytes exceeds the cap {host.device_memory_cap_bytes}"
        if host.host_memory_cap_bytes and memory.get("host_peak_rss_bytes", 0) > host.host_memory_cap_bytes:
            return f"host peak {memory['host_peak_rss_bytes']} bytes exceeds the cap {host.host_memory_cap_bytes}"
        return None

    for microbatch in profile.train_microbatches:
        outcome = launch({**common, "kind": "train_probe", "mode": "full", "learning_rate": protocol.recipes[-1].learning_rate, "microbatch": microbatch})
        if outcome is None:
            break
        cap_problem = over_cap(outcome) if outcome.get("ok") else None
        result["attempts"].append({"microbatch": microbatch, "ok": bool(outcome.get("ok")) and cap_problem is None,
                                   "reason": cap_problem or outcome.get("reason"), "memory": outcome.get("memory"), "error": outcome.get("error")})
        if outcome.get("device_unavailable"):
            result["stopped"] = f"device unavailable: {outcome.get('error')}"
            break
        if outcome.get("ok") and cap_problem is None:
            result["microbatch"] = microbatch
            break
    if result["microbatch"] is not None:
        for recipe in protocol.recipes:
            outcome = launch({**_training_spec(common, protocol, recipe, result["microbatch"]), "kind": "train_timing", "warmup": 5, "measured": 50})
            if outcome is None:
                break
            cap_problem = over_cap(outcome) if outcome.get("ok") else None
            result["training"][recipe.recipe_id] = {**outcome, "memory_cap_problem": cap_problem, "ok": bool(outcome.get("ok")) and cap_problem is None}
        for batch in protocol.inference_batches:
            outcome = launch({**common, "kind": "infer_probe", "batch": batch})
            if outcome is None:
                break
            cap_problem = over_cap(outcome) if outcome.get("ok") else None
            result.setdefault("inference_attempts", []).append({"batch": batch, "ok": bool(outcome.get("ok")) and cap_problem is None, "reason": cap_problem or outcome.get("reason")})
            if outcome.get("ok") and cap_problem is None:
                result["inference_batch"] = batch
                break
        if result["inference_batch"] is not None:
            result["inference"] = launch({**common, "kind": "infer_timing", "batch": result["inference_batch"], "warmup": 5, "measured": 50})
        result["io"] = launch({**common, "kind": "io"})
    result["unhealthy_host"] = state["unhealthy"]
    return result


def _estimates(calibration: Mapping[str, Any], bundle_info: Mapping[str, Any]) -> dict[str, Any]:
    """Planning inputs derived from measured rates. Labelled estimates; ranking metrics are never estimated."""
    rates = {}
    for recipe_id, entry in calibration["training"].items():
        if entry.get("ok") and entry.get("pairs_per_second"):
            rates[recipe_id] = {"pairs_per_second": entry["pairs_per_second"], "trainable_parameters": entry["trainable_parameters"],
                                "peak_memory": entry["memory"]}
    infer = calibration.get("inference") or {}
    io = calibration.get("io") or {}
    return {
        "label": "estimate", "train": rates, "inference_pairs_per_second": infer.get("pairs_per_second") if infer.get("ok") else None,
        "io": {k: io.get(k) for k in ("checkpoint_save_seconds", "checkpoint_load_seconds", "export_seconds", "reload_seconds", "scratch_bytes")} if io.get("ok") else None,
        "bundle_bytes": bundle_info["bytes"], "note": "rates were measured on diagnostic-length synthetic pairs at this host; epoch and campaign times derived from them are estimates, not results",
    }


def capacity_estimate(protocol, bundle_bytes: int, total_parameters: int | None, *, margin: float = 1.2) -> dict[str, Any]:
    """Disk bytes a campaign may need (docs/12 §5): an ESTIMATE, labelled as such, with a declared safety margin.

    Per (recipe, seed) run: epoch zero plus up to three resumable checkpoints (weights + two Adam tensors = 12 bytes per
    parameter, conservatively applied to frozen runs too) and one fp32 inference export (4 bytes per parameter). Plus
    the release bundle, an independent backup copy of it, and temporary space for the largest atomic write.
    """
    runs = len(protocol.run_ids())
    if not total_parameters:
        return {"label": "estimate", "status": "unverified", "bytes": None, "detail": "the parameter count is unknown, so the checkpoint footprint is unverified, not zero"}
    checkpoints = runs * 4 * 12 * total_parameters
    exports = runs * 4 * total_parameters
    release = bundle_bytes + 4 * total_parameters
    temporary = 2 * 12 * total_parameters
    total = checkpoints + exports + release * 2 + temporary
    return {"label": "estimate", "status": "estimated", "runs": runs, "checkpoints_bytes": checkpoints, "exports_bytes": exports,
            "release_bytes": release, "backup_bytes": release, "temporary_bytes": temporary, "margin": margin, "bytes": int(total * margin),
            "detail": "models only plus bundle copies; measured installed/cache sizes are added by the plan when known"}


def _bundle_bytes(bundle: Path) -> int:
    return sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file())


def run_doctor(host: HostConfig, bundle_dir: Path, *, worker: Callable[..., dict[str, Any]] = run_worker, now: Callable[[], str] = utc_now) -> dict[str, Any]:
    started = time.monotonic()
    budget = Budget(host.doctor_budget_seconds)
    checks: list[dict[str, str]] = []
    binding = binding_for(host)
    profile = load_profile(host.profile)
    calibration: dict[str, Any] = {"attempts": [], "microbatch": None, "inference_batch": None, "training": {}, "inference": None, "io": None,
                                   "stopped": "not run", "backend": None}
    diagnostic, bundle_info, bundle_ok, protocol_for_capacity = True, {"bytes": 0}, False, None
    manifest_sha = None
    try:
        verified = verify_bundle(bundle_dir)
        bundle_ok = True
        manifest_sha = verified["manifest"]["manifest_sha256"]
        diagnostic = bool(verified["manifest"]["diagnostic"])
        bundle_info = {"bytes": _bundle_bytes(Path(bundle_dir))}
        protocol_for_capacity = verified["protocol"]
        checks.append(_check("bundle", "passed", f"every part verifies; corpus {verified['prepared']['loaded'].corpus.corpus_id}", "measured"))
        pinned = verified["assets"]["pinned"]
        checks.append(_check("model_assets_pinned", "passed" if pinned else "unverified",
                             "a real pretrained model at a pinned 40-hex revision" if pinned else
                             ("diagnostic random weights (rehearsal only)" if diagnostic else "the pretrained revision is not a pinned commit"),
                             "fixture" if diagnostic else "declared"))
        code = code_identity()
        if verified["manifest"]["code"]["source_snapshot_sha256"] != code["source_snapshot_sha256"]:
            checks.append(_check("code_identity", "unverified", "the installed code differs from the code that assembled the bundle; the plan binds the running code", "measured"))
        else:
            checks.append(_check("code_identity", "passed", "installed sources match the bundle's recorded snapshot", "measured"))
        tokenizer = WordPieceTokenizer(Path(bundle_dir) / "model-assets")
        policy = PairEncodingPolicy(verified["protocol"].max_pair_tokens, verified["protocol"].max_query_tokens)
        budget_report = check_corpus_budget(verified["prepared"]["loaded"].corpus.chunks, tokenizer, policy)
        checks.append(_check("pair_special_tokens", "passed", f"the pinned tokenizer adds {tokenizer.pair_special_tokens} special tokens per pair (measured, not assumed)", "measured"))
        excluded = budget_report.get("excluded_count", len(budget_report.get("excluded", [])))
        checks.append(_check("corpus_length_budget", "passed" if not excluded else "failed",
                             f"{excluded} passage(s) exceed the {budget_report.get('passage_budget')}-token passage budget" if excluded else
                             f"every passage fits the {budget_report.get('passage_budget')}-token passage budget (longest {budget_report.get('longest_passage_tokens')})",
                             "fixture" if diagnostic else "measured"))
        if not host.host_memory_cap_bytes:
            checks.append(_check("host_memory_cap", "failed", "no host memory cap is configured; a cap with a reserve must be frozen after discovery", "declared"))
        else:
            physical = _physical_memory_bytes()
            ok = physical is None or host.host_memory_cap_bytes <= physical
            checks.append(_check("host_memory_cap", "passed" if ok and physical else "unverified" if ok else "failed",
                                 f"cap {host.host_memory_cap_bytes} bytes; observed physical memory {physical}", "measured"))
        if profile.accelerator and not host.device_memory_cap_bytes:
            checks.append(_check("device_memory_cap", "failed", "an accelerator profile needs a frozen byte-valued device memory cap", "declared"))
        calibration = calibrate(host, profile, Path(bundle_dir) / "model-assets", verified["protocol"], budget, worker=worker)
        backend = calibration.get("backend") or {}
        total = backend.get("total_memory_bytes") or backend.get("recommended_max_memory_bytes")
        if profile.accelerator and host.device_memory_cap_bytes and total:
            checks.append(_check("device_memory_cap_vs_capacity", "passed" if host.device_memory_cap_bytes <= total else "failed",
                                 f"cap {host.device_memory_cap_bytes} bytes against observed {total} bytes", "measured"))
    except ManifestError as error:
        checks.append(_check("bundle", "failed", str(error), "measured"))
    estimates = _estimates(calibration, bundle_info) if bundle_ok else {"label": "estimate", "train": {}, "inference_pairs_per_second": None, "io": None,
                                                                         "bundle_bytes": 0, "note": "no bundle"}
    total_parameters = max((e.get("trainable_parameters") or 0 for e in estimates["train"].values()), default=0)
    capacity = capacity_estimate(protocol_for_capacity, bundle_info["bytes"], total_parameters) if protocol_for_capacity is not None else \
        {"label": "estimate", "status": "unverified", "bytes": None, "detail": "no verified bundle"}
    estimates["capacity"] = capacity
    required_free = int(capacity["bytes"] or 0)
    checks.extend({**c, "label": "measured"} for c in run_checks(binding, required_free_bytes=required_free))
    checks.append(_check("calibration", "passed" if calibration["microbatch"] and calibration["inference_batch"] and not calibration["stopped"] else "failed",
                         calibration["stopped"] or ("microbatch %s, inference batch %s" % (calibration["microbatch"], calibration["inference_batch"])), "fixture" if diagnostic else "measured"))
    gates = _gates(profile, diagnostic, bundle_ok, checks, calibration)
    unhealthy = bool(calibration.get("unhealthy_host"))
    report = seal_manifest({
        "schema_version": SCHEMA_VERSION, "kind": "doctor-report", "profile": profile.name, "host_binding_sha256": binding["manifest_sha256"],
        "bundle_manifest_sha256": manifest_sha, "diagnostic": diagnostic, "code": code_identity(), "checks": checks, "calibration": calibration,
        "gates": gates, "estimates": estimates,
        "limits": {"process_timeout_seconds": host.process_timeout_seconds, "doctor_budget_seconds": host.doctor_budget_seconds,
                   "cleanup_grace_seconds": host.cleanup_grace_seconds, "io_retries": host.io_retries},
        "elapsed_seconds": round(time.monotonic() - started, 3), "created_at": now(), "unhealthy_host": unhealthy,
    })
    return report


def _gates(profile: DeviceProfile, diagnostic: bool, bundle_ok: bool, checks: list[dict[str, str]], calibration: Mapping[str, Any]) -> dict[str, Any]:
    by_name = {c["name"]: c for c in checks}
    bad = [c["name"] for c in checks if c["status"] == "failed" and c["name"] not in ("calibration",)]
    data_status = "failed" if not bundle_ok or "corpus_length_budget" in bad or "bundle" in bad else ("unverified" if diagnostic else "passed")
    data = {"status": data_status, "label": "fixture" if diagnostic else "measured",
            "detail": "the bundle verifies and every passage fits the budget" + (" — diagnostic data proves mechanics, never data readiness" if diagnostic else "")}
    calibrated = bool(calibration.get("microbatch") and calibration.get("inference_batch") and not calibration.get("stopped"))
    io_ok = bool((calibration.get("io") or {}).get("ok"))
    train_ok = bool(calibration.get("training")) and all(t.get("ok") for t in calibration["training"].values())
    if not profile.accelerator:
        backend = {"status": "not_applicable", "detail": "the CPU diagnostic profile is never an accelerator verification"}
    elif not calibration.get("backend"):
        backend = {"status": "failed", "detail": calibration.get("stopped") or "no worker reached the device; backend unavailable"}
    elif diagnostic or not by_name.get("model_assets_pinned", {}).get("status") == "passed":
        backend = {"status": "unverified", "detail": "diagnostic or unpinned assets exercised the device; only real pinned MiniLM assets can verify the backend"}
    elif not (calibrated and io_ok and train_ok):
        backend = {"status": "failed", "detail": calibration.get("stopped") or "training, inference or save/reload probes failed on the device"}
    else:
        backend = {"status": "passed", "detail": "real pinned assets trained, timed, saved and reloaded on the requested device"}
    backend["label"] = "fixture" if diagnostic else "measured"
    backend["fingerprint"] = calibration.get("backend")
    return {
        "implementation_verified": {"status": "unverified", "label": "declared", "detail": "evidence comes from the recorded test receipt for this exact code identity, not from the doctor"},
        "data_ready": data,
        "portable_bundle_verified": {"status": "unverified", "label": "declared", "detail": "needs the clean-install rehearsal receipt; the doctor only verifies the bundle files"},
        f"backend_verified[{profile.name}]": backend,
    }
