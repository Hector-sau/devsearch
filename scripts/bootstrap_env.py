"""Build the offline wheelhouse for a lock and install the runtime from it (docs/10 §2, docs/11 G02).

    bootstrap_env.py download --lock locks/apple-mps-cp312.lock --dest wheelhouse/ [--python-version 3.12] [--platform TAG ...]
    bootstrap_env.py install  --lock LOCK --wheelhouse wheelhouse/ --wheel dist/devsearch-*.whl --venv /path/to/new/venv

`download` runs on a connected machine: every artifact named in the lock is fetched with `pip download --require-hashes`
for the lock's target platform and kept in the wheelhouse; a hash mismatch aborts. `install` needs no network: it creates a NEW
virtual environment (never reuses one), installs only from the wheelhouse with `--no-index --require-hashes`, then installs the
project wheel without dependencies, runs `pip check` and imports the package. Both print a JSON report that includes the lock's
SHA-256, which `bundle assemble --lock` records so the plan binds the exact lock.

Neither step says anything about whether the installed torch works on the machine: that is the doctor's and the backend gate's job.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import venv
from pathlib import Path

PLATFORM_TAGS = {
    "apple-mps": ["macosx_14_0_arm64", "macosx_13_0_arm64", "macosx_12_0_arm64", "macosx_11_0_arm64"],
    # pip does not expand an explicit manylinux tag to the older glibc tags that vendor wheels (the nvidia-* packages: 2_25, 2_27) use
    "nvidia-cuda": [f"manylinux_2_{minor}_x86_64" for minor in range(28, 16, -1)] + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_of(lock: Path) -> str | None:
    for profile in PLATFORM_TAGS:
        if lock.name.startswith(profile + "-"):
            return profile
    return None


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    result = subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr.strip()[-1500:]}")
    return result


def download(args) -> int:
    lock = args.lock.resolve(strict=True)
    if not args.dest.is_dir():
        raise SystemExit(f"{args.dest} does not exist; the wheelhouse directory is never created implicitly")
    platforms = args.platform or PLATFORM_TAGS.get(profile_of(lock) or "", [])
    if not platforms:
        raise SystemExit("cannot infer the target platform from the lock's name; pass --platform TAG (repeatable)")
    command = [sys.executable, "-m", "pip", "download", "--disable-pip-version-check", "--require-hashes", "--no-deps", "--only-binary=:all:", "-r", str(lock),
               "--dest", str(args.dest), "--python-version", args.python_version, "--implementation", "cp", "--abi", f"cp{args.python_version.replace('.', '')}"]
    for tag in platforms:
        command += ["--platform", tag]
    run(command)
    files = sorted(path.name for path in args.dest.iterdir() if path.suffix == ".whl")
    print(json.dumps({"lock": lock.name, "lock_sha256": sha256(lock), "wheelhouse": str(args.dest.resolve()), "wheels": len(files), "platforms": platforms}, indent=2))
    return 0


def lock_python(lock: Path) -> str | None:
    match = re.search(r"-cp(\d)(\d+)\.lock$", lock.name)
    return f"{match.group(1)}.{match.group(2)}" if match else None


def install(args) -> int:
    lock, wheel = args.lock.resolve(strict=True), args.wheel.resolve(strict=True)
    wanted = lock_python(lock)
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if wanted is not None and wanted != running:
        raise SystemExit(f"{lock.name} was resolved for CPython {wanted} but this script runs under {running}; run it with a python{wanted} interpreter (the new environment uses the interpreter that runs the script)")
    if not args.wheelhouse.is_dir():
        raise SystemExit(f"{args.wheelhouse} is not a directory")
    if args.venv.exists():
        raise SystemExit(f"{args.venv} already exists; the install always builds a new environment")
    builder = venv.EnvBuilder(with_pip=True, symlinks=True, clear=False)
    builder.create(args.venv)
    python = str(args.venv / "bin" / "python")
    pip = [python, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--no-index"]
    run([*pip, "--find-links", str(args.wheelhouse), "--require-hashes", "-r", str(lock)])
    run([*pip, "--no-deps", str(wheel)])
    check = subprocess.run([python, "-I", "-m", "pip", "check", "--disable-pip-version-check"], capture_output=True, text=True, check=False)
    imported = run([python, "-I", "-c", "import devsearch, sys; print(devsearch.__file__)"]).stdout.strip()
    print(json.dumps({"venv": str(args.venv.resolve()), "lock": lock.name, "lock_sha256": sha256(lock), "wheel_sha256": sha256(wheel), "pip_check_exit": check.returncode,
                      "pip_check": check.stdout.strip()[-400:], "devsearch_imported_from": imported}, indent=2))
    return 0 if check.returncode == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--lock", type=Path, required=True)
    d.add_argument("--dest", type=Path, required=True)
    d.add_argument("--python-version", default="3.12")
    d.add_argument("--platform", action="append")
    i = sub.add_parser("install")
    i.add_argument("--lock", type=Path, required=True)
    i.add_argument("--wheelhouse", type=Path, required=True)
    i.add_argument("--wheel", type=Path, required=True)
    i.add_argument("--venv", type=Path, required=True)
    args = parser.parse_args()
    return download(args) if args.command == "download" else install(args)


if __name__ == "__main__":
    sys.exit(main())
