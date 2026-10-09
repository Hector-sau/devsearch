"""Check a wheel in a temporary, isolated environment; never train.

Installs only the supplied wheel with network indexes disabled (the package has no
runtime dependencies), then, from a working directory outside the checkout and in
isolated mode (`-I`: no PYTHON* variables, no user site, no cwd on the path), runs
the real baseline commands over the small fixture corpus: build an index, search it,
validate it, and confirm invalid command invocations still fail closed. Before
installation, verifies the wheel's name, version and Python requirement against
pyproject.toml; after installation, verifies distribution and import versions agree.
The fixture and
example configs are passed as *data files*; the installed package must not need the
original checkout or any cache. Passing proves package portability and CPU baseline
behavior on tiny fixtures only: no model quality, MPS or real-document result.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib
import venv
import zipfile

COMMANDS = ("index", "retrieve", "validate_manifest", "prepare", "prepare_data", "train", "evaluate", "predict", "export", "serve",
            "doctor", "plan", "run", "status", "bundle", "rehearsal", "review")
REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "mini_corpus"
CONFIGS = REPO / "configs"


def expected_metadata() -> dict[str, str]:
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return {"Name": project["name"], "Version": project["version"], "Requires-Python": project["requires-python"]}


def check_wheel_metadata(wheel: Path, expected: dict[str, str]) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise SystemExit(f"FAIL: wheel must contain exactly one distribution METADATA file, found {len(names)}")
        metadata = BytesParser().parsebytes(archive.read(names[0]))
    for field, value in expected.items():
        actual = metadata.get_all(field, [])
        if actual != [value]:
            raise SystemExit(f"FAIL: wheel {field} is {actual!r}, expected {[value]!r} from pyproject.toml")


def run(python: Path, cwd: Path, *args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run([str(python), "-I", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != expect:
        raise SystemExit(
            f"FAIL: {' '.join(args)} exited {result.returncode}, expected {expect}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def check_installed_identity(python: Path, cwd: Path, expected: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return run(python, cwd, "-c", (
        "from importlib import metadata; from pathlib import Path; import sys; import devsearch; "
        f"expected = {expected!r}; "
        "distribution = metadata.distribution(expected['Name']); "
        "assert distribution.metadata['Name'] == expected['Name'], 'installed distribution name mismatch'; "
        "assert distribution.version == expected['Version'], 'installed metadata version mismatch'; "
        "assert devsearch.__version__ == distribution.version, 'import version and metadata version mismatch'; "
        "assert distribution.metadata['Requires-Python'] == expected['Requires-Python'], 'installed Python requirement mismatch'; "
        "assert Path(devsearch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), 'package imported from outside environment'; "
        "assert not ({'torch', 'transformers', 'fastapi', 'streamlit', 'numpy'} & set(sys.modules)); "
        "print(f'Installed {distribution.metadata[\"Name\"]} {distribution.version}: metadata and import versions agree; no ML/web imports.')"
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        parser.error("Expected one .whl artifact")
    expected = expected_metadata()
    check_wheel_metadata(wheel, expected)
    print(f"Wheel metadata matches pyproject.toml: {expected['Name']} {expected['Version']} ({expected['Requires-Python']}).")
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
        print(check_installed_identity(python, root, expected).stdout.strip())

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
        f"PASS: installed {expected['Name']} {expected['Version']} matches wheel metadata, built, searched and validated a fixture index outside the checkout, and every command prints its help and "
        "refuses a bare invocation. Fixture-level CPU check only: it trains nothing and says nothing about models, accelerators or real documents."
    )


if __name__ == "__main__":
    main()
