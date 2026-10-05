"""Check a scaffold wheel in a temporary, isolated environment; never train."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import venv


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
        subprocess.run(
            [str(python), "-I", "-c", (
                "from pathlib import Path; import sys; import devsearch; "
                "assert Path(devsearch.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
                "assert not ({'torch', 'transformers', 'fastapi', 'streamlit'} & set(sys.modules)); "
                "print('Installed package imports without ML or web dependencies.')"
            )],
            cwd=root,
            check=True,
        )
        subprocess.run([str(python), "-I", "-m", "devsearch.retrieve", "--help"], cwd=root, check=True)
    print("PASS: installed-wheel import/help only; application and training remain unimplemented.")


if __name__ == "__main__":
    main()
