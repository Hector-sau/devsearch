"""Record evidence receipts for the two gates the doctor cannot measure itself (docs/10 §1, docs/11 §2, docs/13).

    record_receipt.py implementation --output receipts/ [--python PY]
    record_receipt.py portable --wheel dist/devsearch-*.whl --lock locks/<profile>-cpNNN.lock --wheelhouse WHEELHOUSE --output receipts/ [--python PY]

`implementation` runs the whole unit-test suite with the given interpreter (it must have the `ml` and `serve` extras
installed, otherwise ML and API tests would skip, and a skip is not a pass) and seals the exact command, its exit status,
the output hash and the counts it printed, bound to the source-snapshot hash of this checkout.

`portable` creates a brand-new virtual environment OUTSIDE the checkout with the interpreter named by `--python` (its minor version must be
the lock's), installs EVERYTHING from the hashed lock and the offline wheelhouse (`pip install --no-index --find-links WHEELHOUSE
--require-hashes -r LOCK`, then the wheel with `--no-deps`), runs `pip check`, and inspects the result: the installed packages must be exactly the
locked ones, and nothing on `sys.path` or behind an imported third-party module may lie outside the environment, so a package missing from the
wheelhouse can never be masked by a borrowed one. Only then does it run the whole diagnostic campaign from that environment with the checkout
off the import path: rehearsal tree -> doctor -> plan -> run -> status. The install and the campaign both run in a network namespace
(`unshare -rn`); if that is unavailable the receipt says `offline: false` and the gate stays failed rather than claiming it. The receipt records
the lock's name, hash, profile and CPython, the host it was installed on (system, machine, python), the wheelhouse size, the `pip check` exit
status and the facts above; a formal plan accepts it only for the lock its bundle names and for that lock's platform. It is bound to the
source-snapshot hash of the INSTALLED code; the script exits 1 unless that equals this checkout's, so a wheel built from other sources can never
vouch for this tree. It proves that the lock and wheelhouse install cleanly and run the CPU rehearsal on this host; it says nothing about another
platform's lock (the `apple-mps` lock needs a Mac) or about any accelerator.

A receipt for different code counts for nothing, a failing command or a skipped test fails the gate, and neither receipt can
turn diagnostic evidence into a real-data or accelerator result: this records engineering facts only.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from devsearch.artifacts.canonical import pretty_json  # noqa: E402
from devsearch.artifacts.identity import code_identity  # noqa: E402
from devsearch.execution.acceptance import LOCK_PLATFORMS, make_receipt  # noqa: E402

THIRD_PARTY = ("torch", "numpy", "transformers", "tokenizers", "safetensors", "fastapi", "uvicorn", "httpx", "streamlit")
_RAN = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
_VERDICT = re.compile(r"^(OK|FAILED)(?: \(([^)]*)\))?\s*$", re.MULTILINE)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def environment(python: str) -> dict:
    probe = subprocess.run([python, "-c", "import json,platform,importlib.metadata as m\n"
                            "out={'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'versions':{}}\n"
                            f"for n in {list(THIRD_PARTY)!r}:\n"
                            "    try: out['versions'][n]=m.version(n)\n"
                            "    except m.PackageNotFoundError: out['versions'][n]=None\n"
                            "print(json.dumps(out))"], capture_output=True, text=True, check=True)
    return json.loads(probe.stdout)


def run_recorded(argv: list[str], *, cwd: Path, env: dict | None = None, timeout: int = 3600) -> tuple[dict, str, str]:
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    return ({"argv": argv, "cwd": "(checkout)" if cwd == REPO else "(outside the checkout)", "exit_status": result.returncode,
             "stdout_sha256": sha256(result.stdout.encode()), "stderr_sha256": sha256(result.stderr.encode())}, result.stdout, result.stderr)


def write_receipt(directory: Path, receipt: dict) -> Path:
    if not directory.is_dir():
        raise SystemExit(f"{directory} does not exist; it is never created implicitly")
    path = directory / f"{receipt['gate']}-{receipt['code_source_snapshot_sha256'][:12]}-{receipt['created_at'].replace(':', '')}.json"
    path.write_text(pretty_json(receipt), encoding="utf-8")
    return path


def implementation(args) -> int:
    snapshot = code_identity(REPO / "src" / "devsearch")["source_snapshot_sha256"]
    argv = [args.python, "-m", "unittest", "discover", "-s", "tests"]
    entry, out, err = run_recorded(argv, cwd=REPO, env={**os.environ, "PYTHONPATH": "src:tests", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    ran, verdict = _RAN.search(err), _VERDICT.search(err)
    counts = dict(part.split("=") for part in (verdict.group(2) or "").split(", ") if "=" in part) if verdict else {}
    summary = {"tests_run": int(ran.group(1)) if ran else 0, "failures": int(counts.get("failures", 0)), "errors": int(counts.get("errors", 0)),
               "skipped": int(counts.get("skipped", 0)), "verdict": verdict.group(1) if verdict else "unknown"}
    env = environment(args.python)
    receipt = make_receipt("implementation_verified", snapshot, [entry], summary, env, now())
    path = write_receipt(Path(args.output), receipt)
    print(json.dumps({"receipt": str(path), "summary": summary, "exit_status": entry["exit_status"]}, indent=2))
    if entry["exit_status"] != 0:
        failing = sorted(set(re.findall(r"^(?:ERROR|FAIL): (\S+ \([^)]*\))", err, re.MULTILINE)))
        print(f"--- {len(failing)} distinct failing tests (not part of the receipt) ---", file=sys.stderr)
        print("\n".join(failing), file=sys.stderr)
        print("--- the last 12000 characters of the unittest output ---", file=sys.stderr)
        print(err[-12000:], file=sys.stderr)
    return 0 if entry["exit_status"] == 0 and summary["tests_run"] and not (summary["failures"] or summary["errors"] or summary["skipped"]) else 1


def network_isolation_available() -> bool:
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=20, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


PIP_OFFLINE = ["-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--no-index"]
INSPECT = (
    "import importlib.metadata as m, importlib.util as u, json, sys, sysconfig\n"
    "names = " + repr(THIRD_PARTY) + "\n"
    "def origin(n):\n"
    "    spec = u.find_spec(n)\n"
    "    return spec.origin if spec else None\n"
    "paths = sysconfig.get_paths()\n"
    "print(json.dumps({'sys_path': sys.path, 'prefix': sys.prefix, 'stdlib': [paths['stdlib'], paths['platstdlib']],\n"
    "                  'installed': {d.metadata['Name']: d.version for d in m.distributions()}, 'origins': {n: origin(n) for n in names}}))\n"
)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(text: str) -> dict[str, str]:
    """`name==version` pins of a hashed lock file, names normalised."""
    return {normalize(name): version for name, version in re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", text, re.MULTILINE)}


def lock_target(lock: Path) -> dict:
    match = re.search(r"-cp(\d)(\d+)\.lock$", lock.name)
    profile = next((p for p in LOCK_PLATFORMS if lock.name.startswith(p + "-")), None)
    return {"profile": profile, "python": f"{match.group(1)}.{match.group(2)}" if match else None}


def dependency_facts(inspection: dict, pins: dict[str, str], environment_dir: Path) -> dict:
    """What the installed environment says about itself: exactly the locked packages, and nothing resolved from outside it."""
    allowed_roots = [environment_dir.resolve(), *(Path(p).resolve() for p in inspection["stdlib"])]

    def inside(entry: str) -> bool:
        path = Path(entry).resolve() if entry else None
        if path is None:
            return False
        if any(path == root or root in path.parents for root in allowed_roots):
            return True
        return path.suffix == ".zip" and path.parent == Path(inspection["stdlib"][0]).resolve().parent      # pythonXY.zip next to the standard library

    foreign = sorted({entry for entry in inspection["sys_path"] if not inside(entry)})
    outside = sorted(name for name, origin in inspection["origins"].items() if origin and not inside(origin))
    installed = {normalize(name): version for name, version in inspection["installed"].items()}
    extras = sorted(set(installed) - set(pins) - {"pip", "devsearch", "setuptools", "wheel"})
    missing = sorted(name for name, version in pins.items() if installed.get(name) != version)
    return {"foreign_sys_path_entries": foreign + [f"{name} imported from {inspection['origins'][name]}" for name in outside],
            "borrowed_dependencies": bool(foreign or outside), "installed_matches_lock": not missing and not extras,
            "locked_packages_missing_or_different": missing, "unlocked_packages_installed": extras, "locked_packages": len(pins)}


def portable(args) -> int:
    wheel = Path(args.wheel).resolve(strict=True)
    lock = Path(args.lock).resolve(strict=True)
    wheelhouse = Path(args.wheelhouse).resolve(strict=True)
    target = lock_target(lock)
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    creator = subprocess.run([args.python, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"], capture_output=True, text=True, check=False).stdout.strip()
    if target["python"] is None or creator != target["python"]:
        raise SystemExit(f"{lock.name} was resolved for CPython {target['python']} but --python is {creator or 'not runnable'}; the clean environment must use the lock's interpreter ({running} runs this script)")
    checkout_snapshot = code_identity(REPO / "src" / "devsearch")["source_snapshot_sha256"]
    snapshot = checkout_snapshot                                        # replaced by the INSTALLED code's hash once the wheel is installed
    isolated = network_isolation_available()
    prefix = ["unshare", "-rn"] if isolated else []
    commands: list[dict] = []
    summary = {"outside_checkout": False, "rehearsal_complete": False, "offline": isolated, "wheel_sha256": sha256(wheel.read_bytes()),
               "network_isolation": "unshare -rn (network namespace), for the install and for the campaign" if isolated else "unavailable: the run was NOT proven offline"}
    work = Path(tempfile.mkdtemp(prefix="devsearch-portable-")).resolve()
    dependencies: dict = {"source": "hashed lock + offline wheelhouse: pip install --no-index --find-links WHEELHOUSE --require-hashes -r LOCK, then the wheel with --no-deps",
                          "lock": lock.name, "lock_sha256": sha256(lock.read_bytes()), "lock_profile": target["profile"], "lock_python": target["python"],
                          "wheelhouse_wheels": len(list(wheelhouse.glob("*.whl"))), "pip_check_exit": None, "installed_matches_lock": False, "borrowed_dependencies": True,
                          "foreign_sys_path_entries": [], "install_host": {"system": platform.system(), "machine": platform.machine(), "python": creator}}
    summary["dependencies"] = dependencies
    python = None
    try:
        if work == REPO or REPO in work.parents:
            raise SystemExit("the work directory must be outside the checkout")
        environment_dir = work / "env"
        created = subprocess.run([args.python, "-m", "venv", str(environment_dir)], capture_output=True, text=True, check=False)
        if created.returncode != 0:
            raise SystemExit(f"cannot create the clean environment with {args.python}: {created.stderr.strip()[-400:]}")
        python = str(environment_dir / "bin" / "python")
        locked, _, err = run_recorded([*prefix, python, *PIP_OFFLINE, "--find-links", str(wheelhouse), "--require-hashes", "-r", str(lock)], cwd=work)
        commands.append(locked)
        if locked["exit_status"] == 0:
            entry, _, err = run_recorded([*prefix, python, *PIP_OFFLINE, "--no-deps", str(wheel)], cwd=work)
            commands.append(entry)
        else:
            summary["failure"] = {"step": "install from the lock", "stderr_tail": err[-600:]}
        if all(c["exit_status"] == 0 for c in commands):
            checked = subprocess.run([python, "-I", "-m", "pip", "check", "--disable-pip-version-check"], capture_output=True, text=True, check=False)
            commands.append({"argv": [python, "-I", "-m", "pip", "check"], "cwd": "(outside the checkout)", "exit_status": checked.returncode,
                             "stdout_sha256": sha256(checked.stdout.encode()), "stderr_sha256": sha256(checked.stderr.encode())})
            dependencies["pip_check_exit"] = checked.returncode
            inspection = json.loads(subprocess.run([python, "-I", "-c", INSPECT], capture_output=True, text=True, check=True, cwd=work).stdout)
            dependencies.update(dependency_facts(inspection, lock_pins(lock.read_text(encoding="utf-8")), environment_dir))
            where = subprocess.run([python, "-I", "-c", "import devsearch,sys;print(devsearch.__file__)"], capture_output=True, text=True, cwd=work, check=False)
            location = Path(where.stdout.strip()).resolve() if where.returncode == 0 else None
            installed = subprocess.run([python, "-I", "-c", "from devsearch.artifacts.identity import code_identity as c; print(c()['source_snapshot_sha256'])"],
                                       capture_output=True, text=True, cwd=work, check=False)
            if installed.returncode == 0:
                snapshot = installed.stdout.strip()                      # the receipt binds to the code that actually ran, never to the checkout by assumption
            summary["checkout_snapshot_sha256"] = checkout_snapshot
            summary["wheel_matches_checkout"] = installed.returncode == 0 and snapshot == checkout_snapshot
            summary["outside_checkout"] = bool(location and environment_dir in location.parents and REPO not in location.parents and not (REPO / "src") in location.parents)
            commands.append({"argv": [python, "-I", "-c", "import devsearch"], "cwd": "(outside the checkout)", "exit_status": where.returncode, "stdout_sha256": sha256(where.stdout.encode()),
                             "stderr_sha256": sha256(where.stderr.encode())})
            if summary["outside_checkout"] and dependencies["borrowed_dependencies"] is False and dependencies["installed_matches_lock"]:
                commands.extend(campaign(python, work, isolated, summary))
        summary["rehearsal_complete"] = summary["rehearsal_complete"] and all(c["exit_status"] == 0 for c in commands)
        env = environment(python) if python and (environment_dir / "bin" / "python").exists() else {}
        env["third_party"] = "installed from the hashed lock and the offline wheelhouse into the new environment; nothing is shared with any other environment"
        receipt = make_receipt("portable_bundle_verified", snapshot, commands, summary, env, now())
        path = write_receipt(Path(args.output), receipt)
        print(json.dumps({"receipt": str(path), "summary": summary, "commands": [(c["argv"][2:] if len(c["argv"]) > 2 else c["argv"], c["exit_status"]) for c in commands]}, indent=2))
        clean = dependencies["pip_check_exit"] == 0 and dependencies["installed_matches_lock"] and dependencies["borrowed_dependencies"] is False and not dependencies["foreign_sys_path_entries"]
        return 0 if summary["outside_checkout"] and summary["rehearsal_complete"] and summary["offline"] and summary.get("wheel_matches_checkout") and clean else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


def campaign(python: str, work: Path, isolated: bool, summary: dict) -> list[dict]:
    """rehearsal -> doctor -> plan -> run -> status, every process isolated (`-I`), network-less when possible."""
    tree, plan_dir = work / "tree", work / "plan"
    tree.mkdir()
    plan_dir.mkdir()
    env = {"PATH": os.environ.get("PATH", ""), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "LC_ALL": "C.UTF-8"}
    prefix = ["unshare", "-rn"] if isolated else []
    steps = [
        ["-m", "devsearch.rehearsal", "--output", str(tree), "--topics", "36"],
        ["-m", "devsearch.doctor", "--profile", "cpu-diagnostic", "--host", str(tree / "host-example.toml"), "--output", str(work / "doctor-report.json")],
        ["-m", "devsearch.plan", "--host", str(tree / "host-example.toml"), "--doctor-report", str(work / "doctor-report.json"), "--output", str(plan_dir / "execution-plan.json"),
         "--wheel-sha256", summary["wheel_sha256"]],
        ["-m", "devsearch.run", "--plan", str(plan_dir / "execution-plan.json")],
        ["-m", "devsearch.status", "--plan", str(plan_dir / "execution-plan.json")],
    ]
    recorded = []
    for step in steps:
        entry, out, err = run_recorded([*prefix, python, "-I", *step], cwd=work, env=env)
        recorded.append(entry)
        if entry["exit_status"] != 0:
            summary["failure"] = {"step": step[1], "exit_status": entry["exit_status"], "stdout_tail": out[-600:], "stderr_tail": err[-600:]}
            return recorded
    status = json.loads(out)
    summary["rehearsal_complete"] = bool(status.get("complete"))
    summary["status_report_sha256"] = sha256(out.encode())
    return recorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="gate", required=True)
    for name in ("implementation", "portable"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True, help="Existing directory that receives the sealed receipt.")
        p.add_argument("--python", default=sys.executable, help="implementation: interpreter with the ml and serve extras installed. portable: the interpreter that creates the clean environment; its minor version must be the lock's (default: this one).")
        if name == "portable":
            p.add_argument("--wheel", type=Path, required=True)
            p.add_argument("--lock", type=Path, required=True, help="The hashed lock for this platform (locks/<profile>-cpNNN.lock); everything is installed from it.")
            p.add_argument("--wheelhouse", type=Path, required=True, help="The offline wheelhouse built for that lock by scripts/bootstrap_env.py download.")
    args = parser.parse_args()
    return implementation(args) if args.gate == "implementation" else portable(args)


if __name__ == "__main__":
    raise SystemExit(main())
