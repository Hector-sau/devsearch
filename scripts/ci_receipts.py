"""Export CI-tested wheels/receipts and fetch them through authenticated GitHub.

pack runs in the evidence job. fetch runs on the operator's machine with gh;
verify runs offline on the rental. The operator pins verified-origin.json's
SHA-256 in the host config. That pin is the trust boundary, not a signature.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent.parent
REPOSITORY = "Hector-sau/devsearch"
WORKFLOW = ".github/workflows/non-gpu-evidence.yml"
MANIFEST = "ci-export.json"
ORIGIN = "verified-origin.json"
CRITICAL = ("src", "scripts", "tests", "configs", "locks", ".github")
MAX_ARTIFACT = 64 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    from devsearch.artifacts.canonical import loads_strict
    return loads_strict(Path(path).read_text(), where=str(path))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def safe_file(root, name):
    p = PurePosixPath(name)
    require(name and not p.is_absolute() and ".." not in p.parts and "\\" not in name, "unsafe artifact path")
    path = root / name
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()) and path.is_file(), f"missing/unsafe file: {name}")
    return path


def inventory(root):
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    return {n: sha(safe_file(root, n)) for n in sorted(filter(None, names))}


def check_tree(root, files):
    require(isinstance(files, dict) and files, "missing deployment inventory")
    for name, digest in files.items():
        require(sha(safe_file(root, name)) == digest, f"deployment file changed: {name}")
    # Reject extra executable/config/test inputs; build output and bytecode are not source.
    actual = {p.relative_to(root).as_posix() for d in CRITICAL for p in (root / d).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and not any(x.endswith(".egg-info") for x in p.parts)}
    require(actual <= files.keys(), f"unrecorded deployment files: {sorted(actual - files.keys())[:4]}")


def receipts_ok(directory, manifest, root, lock, profile):
    from devsearch.artifacts.identity import code_identity
    from devsearch.execution.acceptance import RECEIPT_FIELDS, RECEIPT_GATES, _judge
    from devsearch.execution.records import read_sealed
    code = code_identity(root / "src/devsearch")
    require(code["source_snapshot_sha256"] == manifest["code_snapshot"], "CI package source mismatch")
    require(profile == "nvidia-cuda" and manifest["profile"] == profile, "CI import requires the nvidia-cuda profile")
    require(lock.name == "nvidia-cuda-cp312.lock" and sha(lock) == manifest["lock_sha256"], "CI lock mismatch")
    receipts = [read_sealed(safe_file(directory, n), "receipt", RECEIPT_FIELDS, n) for n in manifest["receipts"]]
    require(len(receipts) == 2 and {r["gate"] for r in receipts} == set(RECEIPT_GATES), "exactly two distinct readiness receipts required")
    for gate in RECEIPT_GATES:
        verdict = _judge(gate, receipts, code, profile=profile, lock_sha256=sha(lock), formal=True)
        require(verdict["status"] == "passed", f"CI {gate}: {verdict['detail']}")
    portable = next(r for r in receipts if r["gate"] == "portable_bundle_verified")
    s = portable["summary"]
    require(s.get("wheel_sha256") == manifest["files"][manifest["wheel"]], "CI tested a different project wheel")
    require(s.get("wheel_matches_checkout") is True, "CI wheel did not match checkout")
    require(s["dependencies"].get("lock_python") == "3.12" and s["dependencies"]["install_host"].get("python") == "3.12", "CI Python mismatch")
    probe = s.get("isolation_probe") or {}
    # These are Linux's wire-recorded errno values, not errno constants on the
    # operator's OS (macOS uses 51/61 instead of Linux 101/111).
    control, inside = probe.get("control") or {}, probe.get("isolated") or {}
    connection = inside.get("connect") or {}
    require((control.get("connect") or {}).get("result") == "connected" and inside.get("interfaces") == ["lo"]
            and connection.get("result") == "error" and connection.get("errno") in (101, 111),
            "CI isolation positive control is not proven")
    require(len(manifest["wheelhouse"]) == s["dependencies"]["wheelhouse_wheels"] and bool(manifest["wheelhouse"]), "CI wheelhouse inventory mismatch")
    require(all(PurePosixPath(n).name == n and n.endswith(".whl") and re.fullmatch(r"[0-9a-f]{64}", h)
                for n, h in manifest["wheelhouse"].items()), "invalid dependency wheel inventory")
    return receipts


def verify(directory, expected_sha, root=REPO, lock=None, profile="nvidia-cuda"):
    """No network. Trust only the operator-pinned export from fetch; recheck every file."""
    directory, root = Path(directory), Path(root)
    lock = lock or root / "locks/nvidia-cuda-cp312.lock"
    origin_file = safe_file(directory, ORIGIN)
    require(re.fullmatch(r"[0-9a-f]{64}", expected_sha or "") and sha(origin_file) == expected_sha, "CI origin pin mismatch")
    origin = read(origin_file)
    require(origin.get("schema_version") == 1 and origin.get("repository") == REPOSITORY and origin.get("workflow") == WORKFLOW, "untrusted CI origin")
    require(origin.get("conclusion") == "success" and re.fullmatch(r"[0-9a-f]{40}", origin.get("commit", "")), "unsuccessful/invalid CI origin")
    require(sha(safe_file(directory, MANIFEST)) == origin["manifest_sha256"], "CI export manifest changed")
    m = read(directory / MANIFEST)
    require(m.get("schema_version") == 1 and m.get("commit") == origin["commit"] and m.get("run_id") == origin["run_id"]
            and m.get("run_attempt") == origin["run_attempt"], "CI run identity mismatch")
    require(m.get("repository") == REPOSITORY and m.get("workflow") == WORKFLOW, "CI manifest provenance mismatch")
    for name, digest in m["files"].items():
        require(sha(safe_file(directory, name)) == digest, f"CI artifact changed: {name}")
    require(set(m["receipts"]) | {m["wheel"]} <= m["files"].keys(), "unhashed CI input")
    require(m["wheel"].endswith(".whl"), "CI wheel missing")
    check_tree(root, m["deployment"])
    receipts_ok(directory, m, root, lock, profile)
    return {"directory": str(directory.resolve()), "origin_sha256": expected_sha, "commit": origin["commit"],
            "wheel": str(safe_file(directory, m["wheel"])), "wheel_sha256": m["files"][m["wheel"]], "manifest": m}


def pack(args):
    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY, "pack must run in the project's GitHub Actions job")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    require(commit == os.environ.get("GITHUB_SHA"), "checkout differs from workflow commit")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    directory = args.receipts.resolve()
    wheel = args.wheel.resolve()
    shutil.copy2(wheel, directory / wheel.name)
    receipts = sorted(p.name for gate in ("implementation_verified", "portable_bundle_verified") for p in directory.glob(gate + "-*.json"))
    from devsearch.artifacts.identity import code_identity
    m = {"schema_version": 1, "repository": REPOSITORY, "workflow": WORKFLOW, "commit": commit,
         "run_id": int(os.environ["GITHUB_RUN_ID"]), "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
         "profile": "nvidia-cuda", "lock_sha256": sha(args.lock), "code_snapshot": code_identity()["source_snapshot_sha256"],
         "deployment": inventory(REPO), "wheel": wheel.name, "receipts": receipts,
         "wheelhouse": {p.name: sha(p) for p in sorted(args.wheelhouse.glob("*.whl"))},
         "files": {n: sha(safe_file(directory, n)) for n in [*receipts, wheel.name]}}
    receipts_ok(directory, m, REPO, args.lock, "nvidia-cuda")
    write(directory / MANIFEST, m)


def gh_api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], timeout=120))


def validate_run(run, jobs, artifact, commit):
    require(run.get("head_sha") == commit and run.get("status") == "completed" and run.get("conclusion") == "success", "CI run is not successful on the requested commit")
    require(run.get("repository", {}).get("full_name") == REPOSITORY and run.get("head_repository", {}).get("full_name") == REPOSITORY, "CI repository mismatch")
    require(run.get("path") == WORKFLOW, "wrong CI workflow")
    by_name = {j["name"]: j for j in jobs}
    require(all(by_name.get(n, {}).get("conclusion") == "success" and by_name[n].get("status") == "completed"
                for n in ("locked-env", "real-model")), "both evidence jobs must succeed")
    require(artifact.get("name") == "locked-env-receipts-" + commit and not artifact.get("expired"), "wrong or expired CI artifact")
    require(artifact.get("workflow_run", {}).get("head_sha") == commit, "artifact commit mismatch")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", artifact.get("digest") or ""), "GitHub artifact digest unavailable")


def unpack(archive, target):
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        require(sum(i.file_size for i in members) <= MAX_ARTIFACT, "CI artifact too large")
        seen = set()
        for i in members:
            name = PurePosixPath(i.filename)
            require(not name.is_absolute() and ".." not in name.parts and "\\" not in i.filename, "unsafe ZIP path")
            require(i.filename not in seen and not stat.S_ISLNK(i.external_attr >> 16), "duplicate or linked ZIP input")
            seen.add(i.filename)
            path = target / name
            require(path.resolve().is_relative_to(target.resolve()), "ZIP escaped target")
            if i.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(z.read(i))


def fetch(args):
    require(not args.output.exists(), "output already exists; use a fresh evidence directory")
    run = gh_api(f"repos/{REPOSITORY}/actions/runs/{args.run_id}")
    jobs = gh_api(f"repos/{REPOSITORY}/actions/runs/{args.run_id}/attempts/{run['run_attempt']}/jobs?per_page=100")["jobs"]
    artifacts = gh_api(f"repos/{REPOSITORY}/actions/runs/{args.run_id}/artifacts?per_page=100")["artifacts"]
    matches = [a for a in artifacts if a["name"] == "locked-env-receipts-" + args.commit]
    require(len(matches) == 1, "expected one locked-env artifact")
    artifact = matches[0]
    validate_run(run, jobs, artifact, args.commit)
    require(artifact["size_in_bytes"] <= MAX_ARTIFACT, "CI archive too large")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ci-fetch-", dir=args.output.parent) as temp:
        archive = Path(temp) / "artifact.zip"
        with archive.open("wb") as out:
            subprocess.run(["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"], stdout=out, check=True, timeout=300)
        require("sha256:" + sha(archive) == artifact["digest"], "GitHub ZIP digest mismatch")
        unpacked = Path(temp) / "unpacked"
        unpacked.mkdir()
        unpack(archive, unpacked)
        origin = {"schema_version": 1, "repository": REPOSITORY, "workflow": WORKFLOW, "commit": args.commit,
                  "run_id": args.run_id, "run_attempt": run["run_attempt"], "conclusion": "success",
                  "artifact_id": artifact["id"], "artifact_digest": artifact["digest"],
                  "manifest_sha256": sha(safe_file(unpacked, MANIFEST)), "run_url": run["html_url"]}
        write(unpacked / ORIGIN, origin)
        pin = sha(unpacked / ORIGIN)
        verify(unpacked, pin)
        os.replace(unpacked, args.output)
    print(json.dumps({"import_dir": str(args.output.resolve()), "import_sha256": pin, "commit": args.commit}, indent=2))


def check_environment(python, environment_dir, lock, expected_snapshot, cwd):
    """Inspect installed packages without importing the checkout; no isolation claim."""
    from record_receipt import INSPECT, dependency_facts, lock_pins
    inspection = json.loads(subprocess.check_output([python, "-I", "-c", INSPECT], cwd=cwd, text=True, timeout=120))
    facts = dependency_facts(inspection, lock_pins(lock.read_text()), environment_dir)
    require(facts["installed_matches_lock"] and not facts["borrowed_dependencies"] and not facts["foreign_sys_path_entries"], "target dependencies are not isolated/exact: " + str(facts))
    subprocess.run([python, "-I", "-m", "pip", "check"], cwd=cwd, check=True, capture_output=True, timeout=120)
    probe = "import json,devsearch; from devsearch.artifacts.identity import code_identity; print(json.dumps({'path':devsearch.__file__, 'snapshot':code_identity()['source_snapshot_sha256']}))"
    installed = json.loads(subprocess.check_output([python, "-I", "-c", probe], cwd=cwd, text=True, timeout=60))
    require(Path(installed["path"]).resolve().is_relative_to(environment_dir.resolve()) and installed["snapshot"] == expected_snapshot, "target installed wheel source mismatch")
    return {**facts, "pip_check_exit": 0, "installed_code_snapshot": installed["snapshot"], "os_network_isolation_claimed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    for name in ("wheel", "wheelhouse", "lock", "receipts"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--run-id", type=int, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        pack(args) if args.command == "pack" else fetch(args)
        return 0
    except (ValueError, KeyError, TypeError, ManifestError, OSError, subprocess.SubprocessError) as error:
        print(f"CI evidence refused: {error}", file=sys.stderr)
        return 1


sys.path.insert(0, str(REPO / "src"))
from devsearch.errors import ManifestError  # noqa: E402
if __name__ == "__main__":
    sys.exit(main())
