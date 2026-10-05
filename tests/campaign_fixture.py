"""A completed diagnostic campaign built once per process: the realistic source of a final report for export/serve tests.

It runs the genuine `devsearch.run` in a child process against a private copy of the rehearsal tree, so the report, the
selection, the checkpoints and the campaign's own evaluated release are all real artifacts of the real stage code (on
synthetic data). Callers treat the tree as read-only.
"""

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from devsearch.execution.doctor import run_doctor
from devsearch.execution.hostfile import load_host
from devsearch.execution.plan import build_plan, write_plan
from devsearch.execution.records import read_json, read_stage_manifest
from rehearsal_fixture import rehearsal
from test_doctor import healthy, scripted

SRC = Path(__file__).resolve().parent.parent / "src"
_CACHE: dict = {}


def completed_campaign(**options) -> SimpleNamespace:
    key = tuple(sorted(options.items()))
    if key in _CACHE:
        return _CACHE[key]
    base = rehearsal(**options)
    root = Path(tempfile.mkdtemp(prefix="devsearch-campaign-")).resolve()
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    for entry in base.iterdir():
        target = root / entry.name
        shutil.copytree(entry, target, symlinks=True) if entry.is_dir() else shutil.copyfile(entry, target)
    host_file = root / "host-example.toml"
    host_file.write_text(host_file.read_text().replace(str(base), str(root)))
    host = load_host(host_file)
    report = run_doctor(host, root / "bundle", worker=scripted(healthy), now=lambda: "2026-10-05T00:00:00Z")
    plan, binding = build_plan(host, root / "bundle", report, now="2026-10-05T00:00:00Z")
    (root / "plan").mkdir()
    plan_path = root / "plan" / "execution-plan.json"
    write_plan(plan_path, plan, binding)
    result = subprocess.run([sys.executable, "-B", "-m", "devsearch.run", "--plan", str(plan_path)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(SRC)}, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(f"the fixture campaign did not complete ({result.returncode}): {result.stderr[-2000:]}")
    run_root = root / "runtime" / "run"

    def output(stage_id: str, name: str) -> Path:
        manifest = read_stage_manifest(run_root / "stages" / stage_id / "stage-manifest.json")
        return run_root / "stages" / stage_id / manifest["attempt_id"] / name

    selection = read_json(output("select", "validation-selection.json"), "selection")
    decision = selection["decision"]
    checkpoint = None
    if decision["recipe_id"] is not None:
        stage = f"train-{decision['recipe_id']}-{decision['seed']}"
        checkpoint = run_root / "work" / stage / "checkpoints" / f"{decision['checkpoint_id']}.json"
    _CACHE[key] = SimpleNamespace(root=root, run_root=run_root, plan=plan, plan_path=plan_path, output=output, selection=selection, decision=decision,
                                  checkpoint=checkpoint, report_dir=output("score", "final-report.json").parent, index=root / "bundle" / "prepared" / "index" / "index-manifest.json",
                                  assets=root / "bundle" / "model-assets", release=output("export_release", "release"), summary=json.loads(output("summary", "campaign-summary.json").read_text()))
    return _CACHE[key]
