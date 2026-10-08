#!/usr/bin/env python3
"""Real pinned model on CPU: fetch, verify, register, check, then the `gpu.py diagnose` crash-and-resume rehearsal with it.

    check_real_model_resume.py --python ENV_PY --work EMPTY_DIR --output receipt.json [--revision COMMIT]

ENV_PY is an interpreter whose environment has devsearch (the installed wheel) and the ml extras; every devsearch command runs with
`-I` from a working directory outside the checkout, so the installed package is what runs. Steps, each recorded with its command and
exit status: `gpu.py fetch-model` of the pinned commit (every byte verified against the hub) and the known weight hash; conversion of
`pytorch_model.bin` to safetensors (the same code gpu.py uses); `bundle register-assets`; `check_model_cpu.py --require-pretrained`;
then exactly the steps of `gpu.py diagnose`: a rehearsal tree on these assets (synthetic data), doctor, plan, a run killed with the
fault `mid_stage:train:full:42` (exit 137, no cleanup, like SIGKILL), `run --resume`, and `status` (complete, and the killed stage shows a
second attempt). It proves that the real encoder trains, crashes and resumes through the real controller on CPU; synthetic data, so it
says nothing about ranking quality, CUDA, memory or speed. Exit 0 all passed, 1 a step failed, 2 usage.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_ID = "microsoft/MiniLM-L12-H384-uncased"
PINNED = "44acabbec0ef496f6dbc93adadea57f376b7c0ec"
WEIGHTS = {"pytorch_model.bin": (133483028, "bcc275fe3e183a68e629c55355be77ef51c2ee780990173ca8f8edc65152fae7")}


def load_gpu():
    spec = importlib.util.spec_from_file_location("devsearch_gpu_script", REPO / "scripts" / "gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", required=True, help="interpreter of the environment with the installed wheel and the ml extras")
    parser.add_argument("--work", type=Path, required=True, help="an empty or missing directory outside the checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", default=PINNED)
    args = parser.parse_args(argv)
    work = args.work.resolve()
    if work.exists() and any(work.iterdir()):
        print(f"{work} is not empty", file=sys.stderr)
        return 2
    if REPO in work.parents or work == REPO:
        print("--work must be outside the checkout", file=sys.stderr)
        return 2
    work.mkdir(parents=True, exist_ok=True)
    gpu = load_gpu()
    steps = []
    receipt = {"receipt": "real-model CPU crash and resume", "repo": REPO_ID, "revision": args.revision,
               "started_utc": datetime.now(timezone.utc).isoformat(), "host": platform.platform(), "python": args.python,
               "code_key_of_checkout": gpu.code_key(), "steps": steps}

    def run(name, command, *, ok=(0,), env_extra=None, cwd=work):
        env = {k: v for k, v in __import__("os").environ.items() if not k.startswith("PYTHON")}
        env.update(env_extra or {})
        start = time.monotonic()
        result = subprocess.run([str(c) for c in command], cwd=cwd, capture_output=True, text=True, env=env)
        (work / "logs").mkdir(exist_ok=True)
        (work / "logs" / f"{len(steps):02d}-{name}.log").write_text(f"$ {' '.join(map(str, command))}\n{result.stdout}\n--- stderr\n{result.stderr}\nexit {result.returncode}\n")
        steps.append({"step": name, "command": " ".join(map(str, command)), "env": env_extra or {}, "exit": result.returncode,
                      "expected": list(ok), "seconds": round(time.monotonic() - start, 1)})
        print(f"[{'ok' if result.returncode in ok else 'FAIL'}] {name}: exit {result.returncode}", flush=True)
        if result.returncode not in ok:
            raise RuntimeError(f"{name} exited {result.returncode}: {result.stderr[-2000:]}")
        return result

    py = args.python
    passed = False
    try:
        version = run("environment", [py, "-I", "-c", "import devsearch, torch, transformers, sys; "
                                                      "print(devsearch.__file__); print(torch.__version__); print(transformers.__version__); print(sys.version)"])
        location, torch_version, transformers_version, _ = version.stdout.splitlines()[:4]
        outside = REPO not in Path(location).resolve().parents
        receipt.update(devsearch_imported_from=location, torch=torch_version, transformers=transformers_version,
                       installed_package_outside_checkout=outside)
        if not outside:
            raise RuntimeError(f"devsearch was imported from the checkout ({location}), not from the installed package; nothing else is checked")
        fetched = work / "fetch"
        run("fetch-model", [sys.executable, REPO / "scripts" / "gpu.py", "fetch-model", "--revision", args.revision, "--output", fetched])
        for name, (size, digest) in WEIGHTS.items():
            actual = {"bytes": (fetched / name).stat().st_size, "sha256": sha256(fetched / name)}
            steps.append({"step": f"verify {name}", "expected": {"bytes": size, "sha256": digest}, "actual": actual, "exit": 0 if actual == {"bytes": size, "sha256": digest} else 1})
            if steps[-1]["exit"]:
                raise RuntimeError(f"{name} does not match the pinned hash")
        assets = work / "assets" / "model"
        assets.mkdir(parents=True)
        skip = {"pytorch_model.bin", "assets-manifest.json", gpu.FETCH_MANIFEST}
        for path in sorted(fetched.iterdir()):
            if path.is_file() and path.name not in skip and not path.name.startswith("."):
                shutil.copy2(path, assets / path.name)
        run("convert-weights", [py, "-I", "-c", gpu.CONVERT, fetched / "pytorch_model.bin", assets / "model.safetensors"])
        run("register-assets", [py, "-I", "-m", "devsearch.bundle", "register-assets", "--directory", assets, "--repo", REPO_ID,
                                "--revision", args.revision, "--license", "MIT", "--kind", "pretrained"])
        receipt["assets_manifest_sha256"] = sha256(assets / "assets-manifest.json")
        run("check_model_cpu", [py, REPO / "scripts" / "check_model_cpu.py", "--assets", assets, "--require-pretrained",
                                "--output", work / "model-cpu-check.json"])
        receipt["model_cpu_check"] = json.loads((work / "model-cpu-check.json").read_text())
        # Exactly gpu.py stage_diagnose, with these assets and the cpu-diagnostic profile.
        directory = work / "diagnose"
        directory.mkdir()
        run("diagnose-rehearsal", [py, "-I", "-m", "devsearch.rehearsal", "--output", directory, "--topics", "24", "--assets", assets])
        (directory / "plan").mkdir()
        deadline = (datetime.now(timezone.utc) + timedelta(hours=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        host = directory / "host.toml"
        host.write_text(gpu.host_toml(profile="cpu-diagnostic", runtime_root=directory / "runtime", bundle=directory / "bundle",
                                      run_root=directory / "runtime" / "run", evaluation_root=directory / "evaluation", backup_dir=directory / "backup",
                                      mount=directory, volume_id="rehearsal-volume", deadline=deadline, min_free_bytes=10**9, device_cap=0,
                                      independent_backup=False), encoding="utf-8")
        plan = directory / "plan" / "execution-plan.json"
        run("diagnose-doctor", [py, "-I", "-m", "devsearch.doctor", "--profile", "cpu-diagnostic", "--host", host, "--output", directory / "doctor-report.json"])
        run("diagnose-plan", [py, "-I", "-m", "devsearch.plan", "--host", host, "--doctor-report", directory / "doctor-report.json", "--output", plan])
        crashed = run("diagnose-run-crash", [py, "-I", "-m", "devsearch.run", "--plan", plan], ok=(137,),
                      env_extra={"DEVSEARCH_FAULT": gpu.DIAGNOSE_FAULT, "DEVSEARCH_FAULT_AFTER": "1"})
        run("diagnose-run-resume", [py, "-I", "-m", "devsearch.run", "--plan", plan, "--resume"])
        status = run("diagnose-status", [py, "-I", "-m", "devsearch.status", "--plan", plan], ok=(0, 1, 4, 5, 6))
        report = json.loads(status.stdout)
        stages = report.get("stages", {})
        train = stages.get("train:full:42", {}) if isinstance(stages, dict) else next((s for s in stages if s.get("stage_id") == "train:full:42"), {})
        attempts = int(train.get("attempt", 0) or 0)
        passed = (status.returncode == 0 and crashed.returncode == 137 and attempts >= 2 and receipt["model_cpu_check"].get("passed", False) is True
                  and receipt["installed_package_outside_checkout"] is True)
        receipt.update(fault=gpu.DIAGNOSE_FAULT, crash_exit=crashed.returncode, status_exit=status.returncode, train_full_attempts=attempts,
                       status_overall={k: report[k] for k in ("state", "complete", "next") if k in report})
    except Exception as error:  # recorded, then the receipt still gets written
        receipt["error"] = str(error)[-4000:]
    receipt.update(passed=passed, finished_utc=datetime.now(timezone.utc).isoformat(),
                   note="real pinned encoder on CPU with synthetic rehearsal data: training, crash (fault exit 137), resume, export, reload and backup "
                        "plumbing only; no ranking, CUDA, memory or speed result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": passed, "train_full_attempts": receipt.get("train_full_attempts"), "error": receipt.get("error")}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
