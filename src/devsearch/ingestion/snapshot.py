"""Pinned source snapshots: acquisition, verification and the file inventory (docs/04 "Source acquisition").

A snapshot is a directory holding a source's files at one exact revision. Acquisition is a separate, explicit, network step
(`git fetch` of the exact commit into an empty directory, with hooks and LFS filters disabled and nothing executed); everything
else reads the directory offline. The revision is *verified* when the checkout's HEAD equals it (and no tracked file is
modified), or when the directory's content hash equals a pinned `snapshot_sha256`; otherwise it is only *declared* and the
report says so. Symbolic links are never followed or read.
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from devsearch.artifacts.canonical import digest_of, sha256_file
from devsearch.errors import ManifestError, PrerequisiteError
from devsearch.ingestion.config import FORMATS, SourceSpec

GIT_TIMEOUT_S = 900
_PASS_THROUGH = ("PATH", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR",
                 "GIT_SSL_CAINFO", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE")
_SAFE_GIT = ("-c", "core.hooksPath=" + os.devnull, "-c", "filter.lfs.required=false", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.clean=",
             "-c", "filter.lfs.process=", "-c", "submodule.recurse=false", "-c", "protocol.version=2")


def _git(*args: str, cwd: Path | None = None) -> str:
    environment = {key: os.environ[key] for key in _PASS_THROUGH if key in os.environ}
    environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "LC_ALL": "C"})
    try:
        result = subprocess.run(["git", *_SAFE_GIT, *args], cwd=cwd, env=environment, capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False)
    except FileNotFoundError as error:
        raise PrerequisiteError("git is not installed; it is needed to acquire or verify a pinned snapshot") from error
    except subprocess.TimeoutExpired as error:
        raise PrerequisiteError(f"git {args[0]} did not finish within {GIT_TIMEOUT_S} s") from error
    if result.returncode != 0:
        raise PrerequisiteError(f"git {args[0]} failed ({result.returncode}): {result.stderr.strip()[-600:]}")
    return result.stdout.strip()


def acquire(spec: SourceSpec) -> None:
    """Fetch exactly `spec.revision` into `spec.snapshot` (which must not exist yet). Needs network; runs nothing from the source."""
    if spec.git_url is None:
        raise PrerequisiteError(f"{spec.source_id}: the snapshot {spec.snapshot} does not exist and no git_url is configured to acquire it")
    if not spec.snapshot.parent.is_dir():
        raise ManifestError(f"{spec.snapshot.parent} does not exist; acquisition never creates parent directories")
    if spec.snapshot.exists():
        raise ManifestError(f"{spec.snapshot} already exists; it is verified, never overwritten")
    spec.snapshot.mkdir()
    try:
        _git("init", "-q", str(spec.snapshot))
        _git("-C", str(spec.snapshot), "fetch", "-q", "--depth", "1", "--no-tags", "--no-recurse-submodules", spec.git_url, spec.revision)
        _git("-C", str(spec.snapshot), "-c", "advice.detachedHead=false", "checkout", "-q", "--detach", "FETCH_HEAD")
        head = _git("-C", str(spec.snapshot), "rev-parse", "HEAD")
        if head != spec.revision:
            raise ManifestError(f"{spec.source_id}: fetched {head}, not the pinned {spec.revision}")
    except BaseException:
        shutil.rmtree(spec.snapshot, ignore_errors=True)         # only ever our own, freshly created directory
        raise


def content_hash(snapshot: Path) -> str:
    """A hash over every regular file (relative path and content hash), excluding git metadata; symbolic links are not followed or hashed."""
    entries = []
    for path in _walk(snapshot, record_links=None):
        entries.append([path.relative_to(snapshot).as_posix(), sha256_file(path)])
    return digest_of(sorted(entries))


def verify_revision(spec: SourceSpec) -> str:
    """`verified-git-head`, `verified-content-hash` or `declared-unverified`; raises when a verification contradicts the pin."""
    if not spec.snapshot.is_dir():
        raise ManifestError(f"{spec.source_id}: snapshot {spec.snapshot} is not a directory")
    if (spec.snapshot / ".git").exists():
        head = _git("-C", str(spec.snapshot), "rev-parse", "HEAD")
        if head != spec.revision:
            raise ManifestError(f"{spec.source_id}: the checkout is at {head}, not the pinned {spec.revision}")
        if _git("-C", str(spec.snapshot), "status", "--porcelain", "--untracked-files=no"):
            raise ManifestError(f"{spec.source_id}: tracked files in the checkout were modified after the pinned revision")
        return "verified-git-head"
    if spec.snapshot_sha256 is not None:
        actual = content_hash(spec.snapshot)
        if actual != spec.snapshot_sha256:
            raise ManifestError(f"{spec.source_id}: the snapshot content hash {actual} differs from the pinned {spec.snapshot_sha256}")
        return "verified-content-hash"
    return "declared-unverified"


def _walk(root: Path, *, record_links: list[str] | None):
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(name for name in names if name != ".git")
        base = Path(directory)
        for name in list(names):
            if (base / name).is_symlink():
                names.remove(name)
                if record_links is not None:
                    record_links.append((base / name).relative_to(root).as_posix())
        for name in sorted(files):
            path = base / name
            if path.is_symlink():
                if record_links is not None:
                    record_links.append(path.relative_to(root).as_posix())
                continue
            yield path


def glob_regex(pattern: str) -> re.Pattern[str]:
    """`**` crosses directories, `*` and `?` stay inside one path segment; matching is over the whole relative path."""
    out, index = [], 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index):
            out.append("(?:.*/)?")
            index += 3
        elif pattern.startswith("**", index):
            out.append(".*")
            index += 2
        elif char == "*":
            out.append("[^/]*")
            index += 1
        elif char == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(char))
            index += 1
    return re.compile("^" + "".join(out) + "$")


@dataclass(frozen=True)
class Inventory:
    included: tuple[str, ...]                 # snapshot-relative POSIX paths, sorted
    seen: int
    excluded: int
    symlinks: tuple[str, ...]
    by_extension: dict[str, int]


def inventory(spec: SourceSpec, max_files: int) -> Inventory:
    include = [glob_regex(p) for p in spec.include]
    exclude = [glob_regex(p) for p in spec.exclude]
    links: list[str] = []
    seen, excluded, kept, by_ext = 0, 0, [], {}
    for path in _walk(spec.snapshot, record_links=links):
        relative = path.relative_to(spec.snapshot).as_posix()
        seen += 1
        if not any(p.match(relative) for p in include) or any(p.match(relative) for p in exclude):
            excluded += 1
            continue
        kept.append(relative)
        suffix = path.suffix.lower()
        by_ext[suffix or "(none)"] = by_ext.get(suffix or "(none)", 0) + 1
        if len(kept) > max_files:
            raise ManifestError(f"{spec.source_id}: more than {max_files} files match; narrow the include patterns or raise policy.max_files deliberately")
    return Inventory(tuple(sorted(kept)), seen, excluded, tuple(sorted(links)), dict(sorted(by_ext.items())))


def supported(relative: str) -> str | None:
    return FORMATS.get(Path(relative).suffix.lower())
