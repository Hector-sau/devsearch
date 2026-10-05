"""Check a wheel in a temporary, isolated environment; never train.

Installs only the supplied wheel with network indexes disabled (the package has no
runtime dependencies), then, from a working directory outside the checkout and in
isolated mode (`-I`: no PYTHON* variables, no user site, no cwd on the path), runs
the real baseline commands over the small fixture corpus: build an index, search it,
validate it, and confirm unimplemented commands still fail closed. The fixture and
example configs are passed as *data files*; the installed package must not need the
original checkout or any cache. Passing proves package portability and CPU baseline
behavior on tiny fixtures only: no model quality, MPS or real-document result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv

COMMANDS = ("index", "retrieve", "validate_manifest", "prepare", "prepare_data", "train", "evaluate", "predict", "export", "serve",
            "doctor", "plan", "run", "status", "bundle", "rehearsal")
REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mini_corpus"
CONFIGS = REPO / "configs"


def run(python: Path, cwd: Path, *args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([str(python), "-I", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != expect:
        raise SystemExit(
            f"FAIL: {' '.join(args)} exited {result.returncode}, expected {expect}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        parser.error("Expected one .whl artifact")
    with tempfile.TemporaryDirectory(prefix="devsearch-wheel-check-") as directory:
        root = Path(directory)
        environment = root / "env"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / "bin" / "python"
        if not python.exists():
            python = environment / "Scripts" / "python.exe"
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--no-index", "--no-deps", str(wheel)],
            cwd=root,
            check=True,
        )
        run(python, root, "-c", (
            "from pathlib import Path; import sys; import devsearch; "
            "assert Path(devsearch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
            "assert not ({'torch', 'transformers', 'fastapi', 'streamlit', 'numpy'} & set(sys.modules)); "
            "print('Installed package imports without ML or web dependencies.')"
        ))

        # Work in a runtime root that holds copies of the data files only.
        work = root / "runtime"
        shutil.copytree(FIXTURE, work / "corpus")
        (work / "index").mkdir()
        shutil.copy(CONFIGS / "index.toml", work / "index.toml")
        shutil.copy(CONFIGS / "search.toml", work / "search.toml")
        manifest = work / "index" / "index-manifest.json"
        runtime = ("--runtime-root", str(work))

        built = json.loads(run(
            python, root, "-m", "devsearch.index", "--config", str(work / "index.toml"),
            "--corpus", str(work / "corpus" / "corpus-manifest.json"), "--output", str(manifest), *runtime,
        ).stdout)
        assert built["status"] == "built" and built["stats"]["chunk_count"] == 20, built
        found = json.loads(run(
            python, root, "-m", "devsearch.retrieve", "--config", str(work / "search.toml"),
            "--mode", "baseline", "--index", str(manifest), "--query", "how do I load_state_dict", *runtime,
        ).stdout)
        top = found["baseline_results"][0]
        assert found["status"] == "ok" and top["section"] == "Checkpoints > Loading", found
        assert top["source_location"] == [{"path": "docs/checkpoints.md", "start_line": 8, "end_line": 11}], top
        empty = json.loads(run(
            python, root, "-m", "devsearch.retrieve", "--config", str(work / "search.toml"),
            "--mode", "baseline", "--index", str(manifest), "--query", "kubernetes ingress", *runtime,
        ).stdout)
        assert empty["status"] == "empty" and empty["baseline_results"] == [], empty
        valid = json.loads(run(python, root, "-m", "devsearch.validate_manifest", "--manifest", str(manifest), *runtime).stdout)
        assert valid["status"] == "valid" and valid["kind"] == "index", valid
        run(python, root, "-m", "devsearch.retrieve", "--config", str(work / "search.toml"),
            "--mode", "rerank", "--index", str(manifest), "--query", "x", *runtime, expect=2)
        for command in COMMANDS:
            helped = run(python, root, "-m", f"devsearch.{command}", "--help")
            assert f"python -m devsearch.{command}" in helped.stdout, command
            run(python, root, "-m", f"devsearch.{command}", expect=2)          # required options missing: a usage error, nothing touched
        assert not any(path.name.endswith(".prepare-report.json") for path in root.rglob("*")), "a bare invocation wrote something"
    print(
        "PASS: installed wheel built, searched and validated a fixture index outside the checkout, and every command prints its help and "
        "refuses a bare invocation. Fixture-level CPU check only: it trains nothing and says nothing about models, accelerators or real documents."
    )


if __name__ == "__main__":
    main()
