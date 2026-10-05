"""Execution records: host bindings, run state, stage manifests and the append-only event log (docs/10 §5).

All JSON is canonical and strict: unknown fields are rejected, hashed manifests never include their own hash
(`seal_manifest` adds `manifest_sha256` outside the hashed body), and every write is atomic (temporary file on
the destination filesystem, fsync, rename). Identity fields (`plan_id`, `stage_id`, `attempt_id`, input and
output hashes) bind a record to exactly one plan and one attempt, so recovery can tell a committed attempt
from a leftover. A *succeeded* log line never substitutes for an artifact check.
"""

import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import (
    canonical_json, loads_strict, pretty_json, read_utf8, seal_manifest, sha256_file, verify_manifest_hash,
)
from devsearch.artifacts.files import write_atomic
from devsearch.artifacts.schema import as_object, as_sha256, exact_keys
from devsearch.errors import ManifestError

SCHEMA_VERSION = 1
STAGE_STATUSES = ("pending", "running", "succeeded", "failed", "awaiting_review", "stopped_budget")
CAMPAIGN_STATUSES = ("pending", "running", "succeeded", "failed", "awaiting_review", "stopped_budget")
GATE_STATUSES = ("passed", "failed", "unverified", "not_applicable")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(value: str, where: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise ManifestError(f"{where}: expected a UTC time like 2026-10-06T08:00:00Z, got {value!r}") from error
    return parsed.replace(tzinfo=timezone.utc)


def new_attempt_id() -> str:
    return "attempt-" + uuid.uuid4().hex[:16]


def write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    write_atomic(path, pretty_json(value).encode("utf-8"), exclusive=exclusive)


def read_json(path: Path, where: str) -> dict[str, Any]:
    return dict(as_object(loads_strict(read_utf8(path, where=where), where=where), where))


def read_sealed(path: Path, kind: str, fields: Sequence[str], where: str) -> dict[str, Any]:
    manifest = read_json(path, where)
    exact_keys(manifest, ("schema_version", "kind", *fields, "manifest_sha256"), where)
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["kind"] != kind:
        raise ManifestError(f"{where}: expected kind={kind!r}, schema_version={SCHEMA_VERSION}")
    verify_manifest_hash(manifest, where=where)
    return manifest


# -- host binding ----------------------------------------------------------------------------------------
BINDING_FIELDS = ("revision", "parent_sha256", "paths", "mount", "deadline", "reserve_seconds", "limits", "resources", "device",
                  "authorization")


def binding_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return seal_manifest({"schema_version": SCHEMA_VERSION, "kind": "host-binding", **payload})


def binding_id(manifest: Mapping[str, Any]) -> str:
    return "binding-" + str(manifest["manifest_sha256"])[:32]


def read_binding(path: Path) -> dict[str, Any]:
    return read_sealed(path, "host-binding", BINDING_FIELDS, str(path))


def binding_chain(directory: Path, active_sha256: str) -> list[dict[str, Any]]:
    """Walk parents from the active binding to revision 0, verifying every hash; returns oldest first."""
    by_hash: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("host-binding-*.json")):
        manifest = read_binding(path)
        by_hash[manifest["manifest_sha256"]] = manifest
    chain, current = [], active_sha256
    while current is not None:
        manifest = by_hash.get(current)
        if manifest is None:
            raise ManifestError(f"host binding {current[:12]} is missing from {directory}; the chain cannot be verified")
        chain.append(manifest)
        if len(chain) > 100:
            raise ManifestError("host binding chain is implausibly long")
        current = manifest["parent_sha256"]
    chain.reverse()
    if [m["revision"] for m in chain] != list(range(len(chain))):
        raise ManifestError("host binding revisions are not consecutive from 0")
    return chain


# -- run state -------------------------------------------------------------------------------------------
STATE_FIELDS = ("plan_id", "plan_sha256", "status", "stages", "active_binding", "lease", "next_action", "last_error", "updated_at")
STAGE_STATE_FIELDS = ("status", "attempt", "attempt_id", "inputs", "outputs", "started_at", "finished_at", "error")


def new_state(plan_id: str, plan_sha256: str, stage_ids: Sequence[str], binding: Mapping[str, Any], *, now: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "kind": "run-state", "plan_id": plan_id, "plan_sha256": plan_sha256, "status": "pending",
        "stages": {sid: {"status": "pending", "attempt": 0, "attempt_id": None, "inputs": {}, "outputs": {}, "started_at": None,
                         "finished_at": None, "error": None} for sid in stage_ids},
        "active_binding": dict(binding), "lease": None, "next_action": f"run stage {stage_ids[0]}" if stage_ids else "nothing to run",
        "last_error": None, "updated_at": now,
    }


def validate_state(state: Mapping[str, Any], where: str = "run-state.json") -> None:
    exact_keys(state, ("schema_version", "kind", *STATE_FIELDS), where)
    if state["schema_version"] != SCHEMA_VERSION or state["kind"] != "run-state":
        raise ManifestError(f"{where}: not a schema-{SCHEMA_VERSION} run-state")
    if state["status"] not in CAMPAIGN_STATUSES:
        raise ManifestError(f"{where}: unknown campaign status {state['status']!r}")
    for stage_id, stage in as_object(state["stages"], f"{where}.stages").items():
        exact_keys(stage, STAGE_STATE_FIELDS, f"{where}.stages.{stage_id}")
        if stage["status"] not in STAGE_STATUSES:
            raise ManifestError(f"{where}: stage {stage_id} has unknown status {stage['status']!r}")
    exact_keys(state["active_binding"], ("id", "revision", "sha256"), f"{where}.active_binding")


def read_state(path: Path) -> dict[str, Any]:
    state = read_json(path, str(path))
    validate_state(state, str(path))
    return state


def write_state(path: Path, state: dict[str, Any], *, now: str) -> None:
    state["updated_at"] = now
    validate_state(state, "run-state (about to be written)")
    write_json(path, state)


# -- stage manifests -------------------------------------------------------------------------------------
STAGE_MANIFEST_FIELDS = ("plan_id", "stage_id", "attempt_id", "status", "inputs", "outputs", "notes")


def stage_manifest(*, plan_id: str, stage_id: str, attempt_id: str, status: str, inputs: Mapping[str, str],
                   outputs: Mapping[str, Mapping[str, Any]], notes: Mapping[str, Any]) -> dict[str, Any]:
    return seal_manifest({
        "schema_version": SCHEMA_VERSION, "kind": "stage", "plan_id": plan_id, "stage_id": stage_id, "attempt_id": attempt_id, "status": status,
        "inputs": dict(inputs), "outputs": {k: dict(v) for k, v in sorted(outputs.items())}, "notes": dict(notes),
    })


def read_stage_manifest(path: Path) -> dict[str, Any]:
    manifest = read_sealed(path, "stage", STAGE_MANIFEST_FIELDS, str(path))
    if manifest["status"] not in ("succeeded", "awaiting_review"):
        raise ManifestError(f"{path}: a committed stage manifest is succeeded or awaiting_review, not {manifest['status']!r}")
    for name, ref in as_object(manifest["outputs"], f"{path}.outputs").items():
        exact_keys(ref, ("sha256", "bytes"), f"{path}.outputs.{name}")
        as_sha256(ref["sha256"], f"{path}.outputs.{name}.sha256")
    for name, digest in as_object(manifest["inputs"], f"{path}.inputs").items():
        as_sha256(digest, f"{path}.inputs.{name}")
    return manifest


def describe_outputs(base: Path, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Hash every named output file (paths relative to `base`, which must contain them)."""
    outputs = {}
    for name in names:
        path = base / name
        resolved = path.resolve(strict=True)
        if base.resolve() not in resolved.parents or path.is_symlink() or not resolved.is_file():
            raise ManifestError(f"stage output {name!r} must be a regular file inside {base}")
        outputs[name] = {"sha256": sha256_file(resolved), "bytes": resolved.stat().st_size}
    return outputs


def verify_outputs(base: Path, outputs: Mapping[str, Mapping[str, Any]]) -> None:
    for name, ref in outputs.items():
        path = base / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != ref["bytes"] or sha256_file(path) != ref["sha256"]:
            raise ManifestError(f"stage output {name!r} is missing or does not match its recorded SHA-256")


# -- event log -------------------------------------------------------------------------------------------
class EventLog:
    """Append-only JSONL with fsync per event; a partial final line from a crash is discarded on open."""

    def __init__(self, path: Path, *, clock=utc_now) -> None:
        self.path = Path(path)
        self._clock = clock
        self._repair()
        self._sequence = len(self.read())

    def _repair(self) -> None:
        """A complete event always ends with a newline, so text after the last newline is an incomplete write."""
        if not self.path.is_file():
            return
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            keep = data[: data.rfind(b"\n") + 1]
            write_atomic(self.path, keep)

    def append(self, event: str, **fields: Any) -> None:
        self._sequence += 1
        line = canonical_json({"seq": self._sequence, "ts": self._clock(), "event": event, **fields}) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        events = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").split("\n"), start=1):
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                raise ManifestError(f"{self.path}: corrupt event at line {number}") from None
        return events


def monotonic() -> float:
    return time.monotonic()
