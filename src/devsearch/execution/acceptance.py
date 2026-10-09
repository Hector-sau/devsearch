"""Gate evidence and the acceptance report (docs/10 §1, docs/11 §2).

Four independent gates: `implementation_verified`, `data_ready`, `portable_bundle_verified` and
`backend_verified[profile]`. `accelerator_run_ready[profile]` is true only when all four pass for the exact code,
data and profile identity. Some evidence cannot come from the doctor — a full test run, a clean-install rehearsal —
so it arrives as *receipts*: sealed records of commands actually run, with exit status, output hash and the code
snapshot they ran against. A receipt for different code counts for nothing; a missing one leaves the gate
`unverified`; one with a failing command, a skipped required test or a mismatched snapshot is `failed`/`unverified`
and says why. Nothing here upgrades fixture evidence: diagnostic data and CPU runs never pass a real-data or
backend gate.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import seal_manifest
from devsearch.errors import ManifestError
from devsearch.execution.records import SCHEMA_VERSION, GATE_STATUSES, read_sealed

RECEIPT_FIELDS = ("gate", "code_source_snapshot_sha256", "commands", "summary", "environment", "created_at")
RECEIPT_GATES = ("implementation_verified", "portable_bundle_verified")
LOCK_PLATFORMS = {"apple-mps": ("Darwin", "arm64"), "nvidia-cuda": ("Linux", "x86_64")}      # the platform each profile's hashed lock was resolved for


def make_receipt(gate: str, code_snapshot: str, commands: list[Mapping[str, Any]], summary: Mapping[str, Any], environment: Mapping[str, Any], created_at: str) -> dict[str, Any]:
    if gate not in RECEIPT_GATES:
        raise ManifestError(f"receipts exist for {', '.join(RECEIPT_GATES)}, not {gate!r}")
    return seal_manifest({"schema_version": SCHEMA_VERSION, "kind": "receipt", "gate": gate, "code_source_snapshot_sha256": code_snapshot,
                          "commands": [dict(c) for c in commands], "summary": dict(summary), "environment": dict(environment), "created_at": created_at})


def read_receipts(directory: Path | None) -> list[dict[str, Any]]:
    if directory is None or not Path(directory).is_dir():
        return []
    receipts = []
    for path in sorted(Path(directory).glob("*.json")):
        receipt = read_sealed(path, "receipt", RECEIPT_FIELDS, str(path))
        if receipt["gate"] not in RECEIPT_GATES:
            raise ManifestError(f"{path}: unknown gate {receipt['gate']!r}")
        receipts.append(receipt)
    return receipts


def _commands_ok(receipt: Mapping[str, Any]) -> bool:
    return bool(receipt["commands"]) and all(c.get("exit_status") == 0 for c in receipt["commands"])


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _dependency_problem(summary: Mapping[str, Any], *, profile: str | None, lock_sha256: str | None, formal: bool) -> str | None:
    """Why a portable receipt does not prove an install from the locked offline package set, or None. A formal plan also binds it to the plan's lock and platform."""
    deps = summary.get("dependencies")
    if not isinstance(deps, Mapping):
        return "the receipt records no dependency evidence: the install must come from a hashed lock and an offline wheelhouse"
    if not _is_sha256(deps.get("lock_sha256")):
        return "the receipt names no lock hash"
    if deps.get("borrowed_dependencies") is not False or deps.get("foreign_sys_path_entries"):
        return "the environment borrowed dependencies from outside itself, so missing wheelhouse packages would go unnoticed"
    if deps.get("pip_check_exit") != 0:
        return "pip check did not pass in the installed environment"
    if deps.get("installed_matches_lock") is not True:
        return "the installed packages are not exactly the locked ones"
    if not deps.get("wheelhouse_wheels"):
        return "the receipt records no wheelhouse"
    if lock_sha256 is not None and deps["lock_sha256"] != lock_sha256:
        return "the receipt was recorded with a different lock than the plan's bundle names"
    if formal:
        if lock_sha256 is None:
            return "the bundle names no lock, so the receipt cannot be bound to the lock a formal run uses"
        host = deps.get("install_host") or {}
        if profile in LOCK_PLATFORMS and deps.get("lock_profile") != profile:
            return f"the receipt was recorded with the {deps.get('lock_profile')!r} lock, not the {profile!r} lock"
        if profile in LOCK_PLATFORMS and (host.get("system"), host.get("machine")) != LOCK_PLATFORMS[profile]:
            return f"the install was made on {host.get('system')}/{host.get('machine')}, not on the {profile} lock's platform {'/'.join(LOCK_PLATFORMS[profile])}"
    return None


def _judge(gate: str, receipts: list[dict[str, Any]], code: Mapping[str, Any], *, profile: str | None = None, lock_sha256: str | None = None, formal: bool = False) -> dict[str, Any]:
    mine = [r for r in receipts if r["gate"] == gate]
    if not mine:
        return {"status": "unverified", "label": "declared", "detail": "no receipt: run the recorded commands for this exact code identity", "evidence": []}
    current = [r for r in mine if r["code_source_snapshot_sha256"] == code["source_snapshot_sha256"]]
    if not current:
        return {"status": "unverified", "label": "declared", "detail": "every receipt is for different code; evidence for other code proves nothing about this code", "evidence": [r["manifest_sha256"] for r in mine]}
    receipt = max(current, key=lambda r: r["created_at"])
    evidence = [receipt["manifest_sha256"]]
    if not _commands_ok(receipt):
        failed = [c.get("argv") for c in receipt["commands"] if c.get("exit_status") != 0]
        detail = f"a recorded command failed: {failed[0]}" if failed else "the receipt records no command, so nothing was actually run"
        return {"status": "failed", "label": "measured", "detail": detail, "evidence": evidence}
    summary = receipt["summary"]
    if gate == "implementation_verified":
        problems = []
        if not summary.get("tests_run"):
            problems.append("no tests were run")
        if summary.get("failures") or summary.get("errors"):
            problems.append("tests failed")
        if summary.get("skipped"):
            problems.append(f"{summary['skipped']} test(s) were skipped; a skip is not a pass")
        if problems:
            return {"status": "failed", "label": "measured", "detail": "; ".join(problems), "evidence": evidence}
        return {"status": "passed", "label": "measured", "detail": f"{summary['tests_run']} tests passed with no skips for this code identity", "evidence": evidence}
    if not (summary.get("outside_checkout") and summary.get("rehearsal_complete") and summary.get("offline")):
        return {"status": "failed", "label": "measured", "detail": "the rehearsal must be an offline, complete run from an installed wheel outside the checkout", "evidence": evidence}
    problem = _dependency_problem(summary, profile=profile, lock_sha256=lock_sha256, formal=formal)
    if problem:
        return {"status": "failed", "label": "measured", "detail": problem, "evidence": evidence}
    return {"status": "passed", "label": "measured", "detail": "an install from the hashed lock and the offline wheelhouse, with no borrowed dependencies, completed the diagnostic campaign outside the checkout", "evidence": evidence}


def evaluate_gates(doctor: Mapping[str, Any], receipts_dir: Path | None, code: Mapping[str, Any], *, profile: str | None = None,
                   lock_sha256: str | None = None, formal: bool = False) -> dict[str, dict[str, Any]]:
    """The four gates for this doctor report and code identity. Receipts override the doctor's `declared` placeholders. A formal plan passes its
    profile and the lock its bundle names, and the portable receipt must have been recorded with that lock on that lock's platform."""
    receipts = read_receipts(receipts_dir)
    gates = {name: dict(gate) for name, gate in doctor["gates"].items()}
    for name in RECEIPT_GATES:
        gates[name] = _judge(name, receipts, code, profile=profile, lock_sha256=lock_sha256, formal=formal)
    for gate in gates.values():
        if gate["status"] not in GATE_STATUSES:
            raise ManifestError(f"unknown gate status {gate['status']!r}")
    return gates


def accelerator_run_ready(gates: Mapping[str, Mapping[str, Any]], profile: str, *, diagnostic: bool) -> bool:
    """True only when every applicable gate passed on non-diagnostic evidence."""
    if diagnostic:
        return False
    names = ("implementation_verified", "data_ready", "portable_bundle_verified", f"backend_verified[{profile}]")
    return all(gates[name]["status"] == "passed" for name in names)


def build_report(
    *, plan: Mapping[str, Any], gates: Mapping[str, Mapping[str, Any]], profile: str, diagnostic: bool, stages: Mapping[str, Any],
    commands: list[Mapping[str, Any]], artifacts: Mapping[str, str], created_at: str,
) -> dict[str, Any]:
    """`acceptance-report.json`: gate verdicts with evidence hashes, the commands behind them and measured-vs-fixture labels."""
    ready = accelerator_run_ready(gates, profile, diagnostic=diagnostic)
    return seal_manifest({
        "schema_version": SCHEMA_VERSION, "kind": "acceptance-report", "plan_id": "plan-" + plan["manifest_sha256"][:32], "profile": profile,
        "evidence_label": "fixture" if diagnostic else "measured",
        "gates": {name: dict(gate) for name, gate in sorted(gates.items())}, f"accelerator_run_ready[{profile}]": ready,
        "backend_fingerprint": plan["backend"]["fingerprint"], "stages": {k: dict(v) for k, v in sorted(stages.items())},
        "commands": [dict(c) for c in commands], "artifacts": dict(artifacts), "created_at": created_at,
        "statement": ("Diagnostic rehearsal: synthetic data, a random tiny encoder and a CPU profile. It proves the pipeline mechanics only; no gate for a real accelerator "
                      "campaign is claimed." if diagnostic else
                      "Gate verdicts apply to the exact code, data and profile identity recorded in the plan."),
    })
