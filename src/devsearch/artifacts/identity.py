"""Code snapshot identity for manifests (docs/03 §3, docs/06 §2).

A clean Git commit cannot be assumed (installed wheels have none, and a dirty tree
needs more than a commit ID), so identity is a hash over the installed package's
own Python sources. It is reproducible from the files alone and never invented.
"""

from pathlib import Path
from typing import Any

import devsearch
from devsearch.artifacts.canonical import digest_of, sha256_file


def source_inventory(package_dir: Path | None = None) -> list[list[str]]:
    """Sorted [relative POSIX path, sha256] pairs for the package's .py files."""
    root = package_dir or Path(devsearch.__file__).resolve().parent
    return [
        [path.relative_to(root).as_posix(), sha256_file(path)]
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def code_identity(package_dir: Path | None = None) -> dict[str, Any]:
    inventory = source_inventory(package_dir)
    return {
        "package": "devsearch",
        "version": devsearch.__version__,
        "source_snapshot_sha256": digest_of(inventory),
        "source_file_count": len(inventory),
    }
