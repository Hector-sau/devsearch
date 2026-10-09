"""Resolve a hashed, platform-specific dependency lock for one device profile (docs/10 §2, docs/11 G02).

    lock_profile.py --profile apple-mps|nvidia-cuda [--python-version 3.12] [--output locks/] [--uv PATH]

The runtime dependencies of the `ml` and `serve` extras (read from pyproject.toml) are resolved with `uv pip compile` for the
profile's TARGET platform (not this machine's), at the newest compatible versions on PyPI today, with every artifact hash
recorded. The result is a `pip install --require-hashes -r` file: `locks/<profile>-cp<ver>.lock`. Resolution works from any host
and needs network and the `uv` tool; installation (`bootstrap_env.py`) needs neither once a wheelhouse exists.

What a lock is NOT: it is not evidence that the pinned versions work on the target. Whether torch's MPS or CUDA build runs on that
exact OS, driver and hardware is decided by `doctor` and the backend gate (docs/11 G06); until then the profile stays `unverified`.
For nvidia-cuda the lock takes PyPI's default Linux torch wheel (it brings its own CUDA runtime packages); the host driver must
support that CUDA version, which only a real run can show. The project wheel is not in the lock: it is pinned separately by
`plan --wheel-sha256`.
"""

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGETS = {
    "apple-mps": {"uv_platform": "aarch64-apple-darwin", "description": "macOS arm64 (Apple silicon), PyTorch MPS build"},
    "nvidia-cuda": {"uv_platform": "x86_64-manylinux_2_28", "description": "Linux x86_64 (glibc 2.28+), PyTorch CUDA build from PyPI"},
}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def requirement_names(pyproject: Path) -> list[str]:
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    names: list[str] = []
    for group in ("ml", "serve"):
        for requirement in extras[group]:
            match = _NAME.match(requirement)
            if match is None or match.group(0) in names:
                raise SystemExit(f"cannot read requirement {requirement!r} of extra {group!r} (or it is listed twice)")
            names.append(match.group(0))
    return names


def build_command(uv: str, requirements: Path, platform: str, python_version: str, output: Path) -> list[str]:
    return [uv, "pip", "compile", str(requirements), "--python-platform", platform, "--python-version", python_version, "--generate-hashes",
            "--no-header", "--no-annotate", "--quiet", "--output-file", str(output)]


def header(profile: str, python_version: str, command: list[str], tool: str, names: list[str], created_at: str) -> str:
    target = TARGETS[profile]
    return "\n".join([
        f"# DevSearch dependency lock for the `{profile}` profile: {target['description']}, CPython {python_version}.",
        f"# Generated {created_at} by scripts/lock_profile.py with {tool}; index: PyPI.",
        f"# Requirements (extras `ml` + `serve`): {', '.join(names)}.",
        "# Install: pip install --no-index --find-links <wheelhouse> --require-hashes -r THIS_FILE   (scripts/bootstrap_env.py does it).",
        "# UNVERIFIED on the target host: a lock proves resolvability and fixes artifact hashes, not that torch runs on that OS, driver or device.",
        "# The backend gate stays `unverified` until `doctor` and a real MiniLM run on that machine record otherwise.",
        "", ""])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=sorted(TARGETS), required=True)
    parser.add_argument("--python-version", default="3.12", help="Target CPython minor version (default 3.12, the intended verification interpreter).")
    parser.add_argument("--output", type=Path, default=REPO / "locks", help="Existing directory that receives the lock (default: locks/ in the checkout).")
    parser.add_argument("--uv", default=shutil.which("uv") or "uv", help="Path of the uv tool (pip install uv).")
    args = parser.parse_args()
    if not re.fullmatch(r"3\.(1[1-9])", args.python_version):
        parser.error("--python-version must be 3.11 or newer (the project requires it)")
    if not args.output.is_dir():
        raise SystemExit(f"{args.output} does not exist; it is never created implicitly")
    names = requirement_names(REPO / "pyproject.toml")
    tag = args.python_version.replace(".", "")
    destination = args.output / f"{args.profile}-cp{tag}.lock"
    if destination.exists():
        raise SystemExit(f"{destination} already exists; locks are never overwritten (move it aside to regenerate)")
    scratch = args.output / f".{destination.name}.work"
    scratch.mkdir()
    try:
        requirements = scratch / "requirements.in"
        requirements.write_text("\n".join(names) + "\n", encoding="utf-8")
        body = scratch / "body.lock"
        command = build_command(args.uv, requirements, TARGETS[args.profile]["uv_platform"], args.python_version, body)
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            version = subprocess.run([args.uv, "--version"], capture_output=True, text=True, check=False).stdout.strip() or "uv"
        except FileNotFoundError:
            raise SystemExit(f"{args.uv} not found: install the resolver with `pip install uv`") from None
        if result.returncode != 0 or not body.is_file():
            raise SystemExit(f"resolution failed ({result.returncode}): {result.stderr.strip()[-1500:]}")
        text = header(args.profile, args.python_version, command, version, names, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")) + body.read_text(encoding="utf-8")
        if "--hash=sha256:" not in text:
            raise SystemExit("the resolver produced no hashes; refusing to write an unhashed lock")
        destination.write_text(text, encoding="utf-8")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    pins = sum(1 for line in text.splitlines() if re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*==", line))
    print(f"{destination}: {pins} pinned packages, sha256 {hashlib.sha256(destination.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
