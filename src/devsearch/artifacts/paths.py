"""Runtime-root resolution and artifact-path containment (docs/03 §3, docs/06 §2).

References inside manifests are POSIX-style relative paths, resolved against the
referencing manifest's directory and required to stay inside an explicit root.
Absolute paths, backslashes, drive letters, escapes and missing files are rejected.
Nothing here creates directories: an unmounted volume must fail, not be replaced.
"""

import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

from devsearch.errors import ConfigError, ManifestError

RUNTIME_ROOT_ENV = "DEVSEARCH_RUNTIME_ROOT"


def resolve_root(
    explicit: str | os.PathLike[str] | None,
    manifest_path: Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Containment root: --runtime-root, else DEVSEARCH_RUNTIME_ROOT, else the manifest's directory."""
    environment = os.environ if environ is None else environ
    if explicit is not None:
        origin, chosen = "--runtime-root", Path(explicit)
    elif environment.get(RUNTIME_ROOT_ENV):
        origin, chosen = RUNTIME_ROOT_ENV, Path(environment[RUNTIME_ROOT_ENV])
    else:
        chosen = manifest_path.parent
        origin = "the manifest directory"
    try:
        root = chosen.resolve(strict=True)
    except OSError as error:
        raise ConfigError(
            f"runtime root from {origin} does not exist ({chosen}); "
            "is the external volume mounted? No substitute directory is created"
        ) from error
    if not root.is_dir():
        raise ConfigError(f"runtime root from {origin} is not a directory: {chosen}")
    return root


def ensure_inside(path: Path, root: Path, *, what: str) -> Path:
    """Resolve symlinks and require the path to stay inside the root."""
    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise ManifestError(f"{what}: cannot resolve path: {error}") from error
    if not resolved.is_relative_to(root):
        raise ManifestError(f"{what}: path escapes the root {root}: {path}")
    return resolved


def resolve_reference(base_dir: Path, relative: object, root: Path, *, what: str) -> Path:
    """Resolve a manifest reference to an existing regular file inside the root."""
    if not isinstance(relative, str) or not relative:
        raise ManifestError(f"{what}: path must be a non-empty string")
    if "\x00" in relative or "\\" in relative:
        raise ManifestError(f"{what}: path contains a forbidden character: {relative!r}")
    if PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ManifestError(f"{what}: path must be relative, not absolute: {relative!r}")
    target = ensure_inside(base_dir / relative, root, what=what)
    if not target.is_file():
        raise ManifestError(f"{what}: required file is missing: {relative}")
    return target


def relative_reference(base_dir: Path, target: Path) -> str:
    """POSIX relative path from a manifest directory to a target file."""
    return Path(os.path.relpath(target, base_dir)).as_posix()
