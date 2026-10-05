"""Host binding: `host.toml` and the runtime checks behind preflight and every guard (docs/10 §3, §4, §5).

The host file binds an installation to one machine: where the runtime root, run root, evaluation root and
backup live, which volume must be mounted under them, the execution deadline, and the byte-valued memory and
disk caps. It never stores credentials: the strict schema has no field for them and rejects unknown keys.

Checks never "fix" anything. A missing volume fails with exit 3 instead of writing to the internal disk; the
volume is identified by a marker file whose content must equal `expected_id`, and (unless a rehearsal
explicitly says otherwise, which is recorded as an unverified mount) the mount path must be a real mount point.
A check that cannot be made is `unverified`, not passed.

Example:

    schema_version = 1
    [device]     profile = "apple-mps"
    [paths]      runtime_root = "/Volumes/X/devsearch"  run_root = "/Volumes/X/devsearch/runs/c1"
                 evaluation_root = "/Volumes/E/eval"    backup_dir = "/Volumes/B/devsearch-backup"
    [mount]      path = "/Volumes/X"  marker = ".devsearch-volume"  expected_id = "x-2026-10"  require_mount_point = true
    [deadline]   at = "2026-10-06T08:00:00Z"   reserve_seconds = 1800
    [resources]  min_free_bytes = 30000000000  host_memory_cap_bytes = 17179869184  device_memory_cap_bytes = 0
    [limits]     process_timeout_seconds = 300  doctor_budget_seconds = 1800  cleanup_grace_seconds = 30  io_retries = 2
    [backup]     require_independent_device = true
"""

import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devsearch.artifacts.tomlconfig import (
    as_boolean, as_int, as_string, as_table, check_keys, check_version, load_toml,
)
from devsearch.errors import ConfigError, ManifestError, PrerequisiteError
from devsearch.execution.records import binding_manifest, parse_utc


@dataclass(frozen=True)
class HostConfig:
    profile: str
    runtime_root: Path
    run_root: Path
    evaluation_root: Path
    backup_dir: Path
    mount_path: Path
    mount_marker: str
    mount_expected_id: str
    require_mount_point: bool
    deadline: str
    reserve_seconds: int
    min_free_bytes: int
    host_memory_cap_bytes: int
    device_memory_cap_bytes: int
    process_timeout_seconds: int
    doctor_budget_seconds: int
    cleanup_grace_seconds: int
    io_retries: int
    require_independent_backup: bool


def _absolute(value: Any, where: str) -> Path:
    path = Path(as_string(value, where))
    if not path.is_absolute():
        raise ConfigError(f"{where}: must be an absolute path, got {str(path)!r}")
    return path


def load_host(path: Path) -> HostConfig:
    where = str(path)
    table = load_toml(path)
    check_keys(table, ("schema_version", "device", "paths", "mount", "deadline", "resources", "limits", "backup"), (), where)
    check_version(table, where)
    device = as_table(table["device"], f"{where}: [device]")
    check_keys(device, ("profile",), (), f"{where}: [device]")
    paths = as_table(table["paths"], f"{where}: [paths]")
    check_keys(paths, ("runtime_root", "run_root", "evaluation_root", "backup_dir"), (), f"{where}: [paths]")
    mount = as_table(table["mount"], f"{where}: [mount]")
    check_keys(mount, ("path", "marker", "expected_id", "require_mount_point"), (), f"{where}: [mount]")
    deadline = as_table(table["deadline"], f"{where}: [deadline]")
    check_keys(deadline, ("at", "reserve_seconds"), (), f"{where}: [deadline]")
    resources = as_table(table["resources"], f"{where}: [resources]")
    check_keys(resources, ("min_free_bytes", "host_memory_cap_bytes", "device_memory_cap_bytes"), (), f"{where}: [resources]")
    limits = as_table(table["limits"], f"{where}: [limits]")
    check_keys(limits, ("process_timeout_seconds", "doctor_budget_seconds", "cleanup_grace_seconds", "io_retries"), (), f"{where}: [limits]")
    backup = as_table(table["backup"], f"{where}: [backup]")
    check_keys(backup, ("require_independent_device",), (), f"{where}: [backup]")
    marker = as_string(mount["marker"], f"{where}: mount.marker", max_length=200)
    if "/" in marker or marker in (".", ".."):
        raise ConfigError(f"{where}: mount.marker must be a plain file name")
    try:
        parse_utc(deadline["at"], f"{where}: deadline.at")
    except ManifestError as error:
        raise ConfigError(str(error)) from error
    host = HostConfig(
        profile=as_string(device["profile"], f"{where}: device.profile", max_length=64),
        runtime_root=_absolute(paths["runtime_root"], f"{where}: paths.runtime_root"), run_root=_absolute(paths["run_root"], f"{where}: paths.run_root"),
        evaluation_root=_absolute(paths["evaluation_root"], f"{where}: paths.evaluation_root"),
        backup_dir=_absolute(paths["backup_dir"], f"{where}: paths.backup_dir"), mount_path=_absolute(mount["path"], f"{where}: mount.path"),
        mount_marker=marker, mount_expected_id=as_string(mount["expected_id"], f"{where}: mount.expected_id", max_length=200),
        require_mount_point=as_boolean(mount["require_mount_point"], f"{where}: mount.require_mount_point"),
        deadline=deadline["at"], reserve_seconds=as_int(deadline["reserve_seconds"], f"{where}: deadline.reserve_seconds", 0, 7 * 86400),
        min_free_bytes=as_int(resources["min_free_bytes"], f"{where}: resources.min_free_bytes", 0),
        host_memory_cap_bytes=as_int(resources["host_memory_cap_bytes"], f"{where}: resources.host_memory_cap_bytes", 0),
        device_memory_cap_bytes=as_int(resources["device_memory_cap_bytes"], f"{where}: resources.device_memory_cap_bytes", 0),
        process_timeout_seconds=as_int(limits["process_timeout_seconds"], f"{where}: limits.process_timeout_seconds", 1, 86400),
        doctor_budget_seconds=as_int(limits["doctor_budget_seconds"], f"{where}: limits.doctor_budget_seconds", 1, 7 * 86400),
        cleanup_grace_seconds=as_int(limits["cleanup_grace_seconds"], f"{where}: limits.cleanup_grace_seconds", 0, 3600),
        io_retries=as_int(limits["io_retries"], f"{where}: limits.io_retries", 0, 2),
        require_independent_backup=as_boolean(backup["require_independent_device"], f"{where}: backup.require_independent_device"),
    )
    for name in ("runtime_root", "run_root"):
        if not _inside(getattr(host, name), host.mount_path):
            raise ConfigError(f"{where}: paths.{name} must lie inside mount.path so a missing volume cannot divert writes")
    return host


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def first_binding_payload(host: HostConfig, *, additional_expense_authorization: str | None = None) -> dict[str, Any]:
    """Payload of host-binding revision 0 (absolute paths live here and nowhere in the experimental plan)."""
    return {
        "revision": 0, "parent_sha256": None,
        "paths": {"runtime_root": str(host.runtime_root), "run_root": str(host.run_root), "evaluation_root": str(host.evaluation_root),
                  "backup_dir": str(host.backup_dir)},
        "mount": {"path": str(host.mount_path), "marker": host.mount_marker, "expected_id": host.mount_expected_id,
                  "require_mount_point": host.require_mount_point},
        "deadline": host.deadline, "reserve_seconds": host.reserve_seconds,
        "limits": {"process_timeout_seconds": host.process_timeout_seconds, "doctor_budget_seconds": host.doctor_budget_seconds,
                   "cleanup_grace_seconds": host.cleanup_grace_seconds, "io_retries": host.io_retries,
                   "require_independent_backup": host.require_independent_backup},
        "resources": {"min_free_bytes": host.min_free_bytes, "host_memory_cap_bytes": host.host_memory_cap_bytes,
                      "device_memory_cap_bytes": host.device_memory_cap_bytes},
        "device": {"profile": host.profile}, "authorization": additional_expense_authorization,
    }


def binding_for(host: HostConfig) -> dict[str, Any]:
    return binding_manifest(first_binding_payload(host))


def revise_binding(parent: Mapping[str, Any], host: HostConfig, *, authorization: str | None = None) -> dict[str, Any]:
    """A new revision from an edited host file. Only paths, deadline and reserve may change; the device profile never."""
    if host.profile != parent["device"]["profile"]:
        raise ConfigError("a host-binding revision cannot change the device profile")
    payload = first_binding_payload(host)
    payload.update(revision=parent["revision"] + 1, parent_sha256=parent["manifest_sha256"], authorization=authorization)
    return binding_manifest(payload)


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def run_checks(binding: Mapping[str, Any], *, now: datetime | None = None, required_free_bytes: int = 0) -> list[dict[str, str]]:
    """Every check as a record `{name, status: passed|failed|unverified, detail}`; nothing here mutates the host
    except a write probe that creates and removes one small file in the run root."""
    now = now or datetime.now(timezone.utc)
    paths, mount = binding["paths"], binding["mount"]
    checks: list[dict[str, str]] = []
    mount_path = Path(mount["path"])
    marker = mount_path / mount["marker"]
    if not mount_path.is_dir():
        checks.append(_check("mount", "failed", f"mount path {mount_path} does not exist"))
    elif mount["require_mount_point"] and not os.path.ismount(mount_path):
        checks.append(_check("mount", "failed", f"{mount_path} is not a mount point: the volume is not mounted, so nothing may be written below it"))
    elif not marker.is_file() or marker.read_text(encoding="utf-8").strip() != mount["expected_id"]:
        checks.append(_check("mount", "failed", f"volume marker {marker} is missing or does not carry the expected identity"))
    elif not mount["require_mount_point"]:
        checks.append(_check("mount", "unverified", f"{mount_path} carries the expected marker but is not required to be a mount point (rehearsal only)"))
    else:
        checks.append(_check("mount", "passed", f"{mount_path} is a mount point with the expected volume identity"))
    for name in ("runtime_root", "run_root", "evaluation_root", "backup_dir"):
        path = Path(paths[name])
        if not path.is_dir():
            checks.append(_check(f"path:{name}", "failed", f"{path} does not exist; it is never created implicitly"))
        elif path.is_symlink() or path.resolve() != path.resolve(strict=True):
            checks.append(_check(f"path:{name}", "failed", f"{path} is a symbolic link; use the real location"))
        else:
            checks.append(_check(f"path:{name}", "passed", str(path)))
    for name in ("runtime_root", "run_root"):
        path = Path(paths[name])
        resolved = path.resolve() if path.exists() else path
        if path.exists() and not _inside(resolved, mount_path.resolve() if mount_path.exists() else mount_path):
            checks.append(_check(f"containment:{name}", "failed", f"{resolved} resolves outside the mount path"))
    run_root = Path(paths["run_root"])
    if run_root.is_dir():
        probe = run_root / f".write-probe-{uuid.uuid4().hex[:8]}"
        try:
            with probe.open("wb") as handle:
                handle.write(b"x")
                handle.flush()
                os.fsync(handle.fileno())
            probe.unlink()
            checks.append(_check("writable:run_root", "passed", "a probe file was written, synced and removed"))
        except OSError as error:
            checks.append(_check("writable:run_root", "failed", f"cannot write to the run root: {error}"))
        try:
            free = shutil.disk_usage(run_root).free
            needed = max(int(binding["resources"]["min_free_bytes"]), required_free_bytes)
            checks.append(_check("free_space:run_root", "passed" if free >= needed else "failed",
                                 f"{free} bytes free; {needed} bytes required (configured floor and the planned estimate)"))
        except OSError as error:
            checks.append(_check("free_space:run_root", "unverified", f"cannot read disk usage: {error}"))
    backup = Path(paths["backup_dir"])
    if backup.is_dir() and run_root.is_dir():
        if binding["limits"]["require_independent_backup"]:
            same = os.stat(backup).st_dev == os.stat(run_root).st_dev
            checks.append(_check("backup_independent", "failed" if same else "passed",
                                 "the backup shares a device with the run root; it is not an independent copy" if same else "the backup is on a different device"))
        else:
            checks.append(_check("backup_independent", "unverified", "an independent backup device is not required by this host file (rehearsal only)"))
    try:
        remaining = (parse_utc(binding["deadline"], "deadline") - now).total_seconds() - binding["reserve_seconds"]
        checks.append(_check("deadline", "passed" if remaining > 0 else "failed", f"{remaining:.0f} seconds remain after the {binding['reserve_seconds']}s reserve"))
    except ManifestError as error:
        checks.append(_check("deadline", "failed", str(error)))
    return checks


def verify_runtime(binding: Mapping[str, Any]) -> None:
    """Raise PrerequisiteError naming every failed check. `unverified` checks do not block (doctor reports them), and an
    expired deadline is a budget stop handled by the controller (exit 6), not a missing prerequisite (exit 3)."""
    failed = [c for c in run_checks(binding) if c["status"] == "failed" and c["name"] != "deadline"]
    if failed:
        raise PrerequisiteError("; ".join(f"{c['name']}: {c['detail']}" for c in failed))
