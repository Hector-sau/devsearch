"""Atomic, hash-verified, epoch-boundary checkpoints (docs/03 §2, review register R07).

Layout under `<run_dir>/checkpoints/`: `epoch-NNNN.pt` + `epoch-NNNN.json` (manifest with SHA-256),
`latest.json` and `best.json` as *pointers* (never duplicate weight files), and
`diagnostic-*.pt` snapshots. Only `resume_kind: epoch_boundary` checkpoints can be resumed;
mid-epoch snapshots are `diagnostic_only` and `--resume` rejects them. Writing is
temp file -> flush/fsync -> atomic rename -> manifest -> pointer, so a crash leaves the previous
valid checkpoint intact. Files are loaded with `weights_only=True`: no pickled objects execute.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch

from devsearch.artifacts.canonical import canonical_json, sha256_file
from devsearch.artifacts.files import write_atomic
from devsearch.errors import ManifestError

RESUME_KINDS = ("epoch_boundary", "diagnostic_only")


def checkpoint_dir(run_dir: Path) -> Path:
    return run_dir / "checkpoints"


def checkpoint_id(epoch: int) -> str:
    return f"epoch-{epoch:04d}"


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with temporary.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(run_dir: Path, name: str, state: dict[str, Any], *, resume_kind: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Write `<name>.pt` then its manifest. `summary` carries the identity fields recorded for verification."""
    if resume_kind not in RESUME_KINDS:
        raise ValueError(f"resume_kind must be one of {RESUME_KINDS}")
    directory = checkpoint_dir(run_dir)
    directory.mkdir(exist_ok=True)                      # the run directory itself must already exist
    state = {**state, "resume_kind": resume_kind}
    path = directory / f"{name}.pt"
    _atomic_torch_save(state, path)
    manifest = {"checkpoint_id": name, "file": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                "resume_kind": resume_kind, **summary}
    write_atomic(directory / f"{name}.json", (canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def write_pointer(run_dir: Path, which: str, manifest: dict[str, Any]) -> None:
    if which not in ("latest", "best"):
        raise ValueError("pointer must be 'latest' or 'best'")
    write_atomic(checkpoint_dir(run_dir) / f"{which}.json", (canonical_json({"checkpoint_id": manifest["checkpoint_id"], "sha256": manifest["sha256"]}) + "\n").encode("utf-8"))


def read_pointer(run_dir: Path, which: str) -> dict[str, Any] | None:
    path = checkpoint_dir(run_dir) / f"{which}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(run_dir: Path, name: str) -> dict[str, Any]:
    path = checkpoint_dir(run_dir) / f"{name}.json"
    if not path.is_file():
        raise ManifestError(f"checkpoint manifest {name}.json is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(run_dir: Path, name: str, device: torch.device, *, require_resumable: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the file hash against its manifest, then load tensors/state only."""
    manifest = read_manifest(run_dir, name)
    path = checkpoint_dir(run_dir) / manifest["file"]
    if not path.is_file() or sha256_file(path) != manifest["sha256"]:
        raise ManifestError(f"checkpoint {name} is missing or does not match its recorded SHA-256")
    if require_resumable and manifest["resume_kind"] != "epoch_boundary":
        raise ManifestError(f"checkpoint {name} is {manifest['resume_kind']}; only epoch_boundary checkpoints can be resumed")
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("resume_kind") != manifest["resume_kind"]:
        raise ManifestError(f"checkpoint {name}: manifest and file disagree about resume_kind")
    return state, manifest
