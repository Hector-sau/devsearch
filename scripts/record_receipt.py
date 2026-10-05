"""Record evidence receipts for the two gates the doctor cannot measure itself (docs/10 §1, docs/11 §2, docs/13).

    record_receipt.py implementation --output receipts/ [--python PY]
    record_receipt.py portable --wheel dist/devsearch-*.whl --output receipts/ [--python PY]

`implementation` runs the whole unit-test suite with the given interpreter (it must have the `ml` and `serve` extras
installed, otherwise ML and API tests would skip, and a skip is not a pass) and seals the exact command, its exit status,
the output hash and the counts it printed, bound to the source-snapshot hash of this checkout.

`portable` installs the wheel with pip into a brand-new virtual environment OUTSIDE the checkout and runs the whole
diagnostic campaign from it with the checkout off the import path: rehearsal tree -> doctor -> plan -> run -> status. Network
is blocked for those processes with a network namespace (`unshare -rn`); if that is unavailable the receipt says
`offline: false` and the gate stays failed rather than claiming it. Third-party libraries (torch, transformers ...) are
shared read-only from the interpreter's own site-packages through a .pth file; the `devsearch` package itself comes only from the
wheel, which the script verifies. The receipt is bound to the source-snapshot hash of the INSTALLED code; the script exits 1 unless that
equals this checkout's, so a wheel built from other sources can never vouch for this tree.

A receipt for different code counts for nothing, a failing command or a skipped test fails the gate, and neither receipt can
turn diagnostic evidence into a real-data or accelerator result: this records engineering facts only.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from devsearch.artifacts.canonical import pretty_json  # noqa: E402
from devsearch.artifacts.identity import code_identity  # noqa: E402
from devsearch.execution.acceptance import make_receipt  # noqa: E402

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
    return 0 if entry["exit_status"] == 0 and summary["tests_run"] and not (summary["failures"] or summary["errors"] or summary["skipped"]) else 1


def network_isolation_available() -> bool:
    try:
        return subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=20, check=False).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def portable(args) -> int:
    wheel = Path(args.wheel).resolve(strict=True)
    checkout_snapshot = code_identity(REPO / "src" / "devsearch")["source_snapshot_sha256"]
    snapshot = checkout_snapshot                                        # replaced by the INSTALLED code's hash once the wheel is installed
    isolated = network_isolation_available()
    commands: list[dict] = []
    summary = {"outside_checkout": False, "rehearsal_complete": False, "offline": isolated, "wheel_sha256": sha256(wheel.read_bytes()),
               "network_isolation": "unshare -rn (network namespace)" if isolated else "unavailable: the run was NOT proven offline"}
    work = Path(tempfile.mkdtemp(prefix="devsearch-portable-")).resolve()
    try:
        if work == REPO or REPO in work.parents:
            raise SystemExit("the work directory must be outside the checkout")
        environment_dir = work / "env"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment_dir)
        python = str(environment_dir / "bin" / "python")
        entry, _, err = run_recorded([python, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--no-index", "--no-deps", str(wheel)], cwd=work)
        commands.append(entry)
        if entry["exit_status"] == 0:
            site = Path(subprocess.run([python, "-I", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True, check=True).stdout.strip())
            third_party = subprocess.run([args.python, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True, check=True).stdout.strip()
            (site / "third_party.pth").write_text(third_party + "\n")
            where = subprocess.run([python, "-I", "-c", "import devsearch,sys;print(devsearch.__file__)"], capture_output=True, text=True, cwd=work, check=False)
            location = Path(where.stdout.strip()).resolve() if where.returncode == 0 else None
            installed = subprocess.run([python, "-I", "-c", "from devsearch.artifacts.identity import code_identity as c; print(c()['source_snapshot_sha256'])"],
                                       capture_output=True, text=True, cwd=work, check=False)
            if installed.returncode == 0:
                snapshot = installed.stdout.strip()                      # the receipt binds to the code that actually ran, never to the checkout by assumption
            summary["checkout_snapshot_sha256"] = checkout_snapshot
            summary["wheel_matches_checkout"] = snapshot == checkout_snapshot
            summary["outside_checkout"] = bool(location and environment_dir in location.parents and REPO not in location.parents and not (REPO / "src") in location.parents)
            commands.append({"argv": [python, "-I", "-c", "import devsearch"], "cwd": "(outside the checkout)", "exit_status": where.returncode, "stdout_sha256": sha256(where.stdout.encode()),
                             "stderr_sha256": sha256(where.stderr.encode())})
            if summary["outside_checkout"]:
                commands.extend(campaign(python, work, isolated, summary))
        summary["rehearsal_complete"] = summary["rehearsal_complete"] and all(c["exit_status"] == 0 for c in commands)
        env = environment(args.python)
        env["third_party"] = f"shared read-only from {args.python}'s site-packages through a .pth file; devsearch itself comes only from the wheel"
        receipt = make_receipt("portable_bundle_verified", snapshot, commands, summary, env, now())
        path = write_receipt(Path(args.output), receipt)
        print(json.dumps({"receipt": str(path), "summary": summary, "commands": [(c["argv"][2:] if len(c["argv"]) > 2 else c["argv"], c["exit_status"]) for c in commands]}, indent=2))
        return 0 if summary["outside_checkout"] and summary["rehearsal_complete"] and summary["offline"] and summary.get("wheel_matches_checkout") else 1
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
        p.add_argument("--python", default=sys.executable, help="Interpreter with the ml and serve extras installed (default: this one).")
        if name == "portable":
            p.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    return implementation(args) if args.gate == "implementation" else portable(args)


if __name__ == "__main__":
    raise SystemExit(main())
