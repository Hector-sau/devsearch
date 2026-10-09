"""A shared diagnostic rehearsal tree for execution tests: built once per process, removed at exit."""

import atexit
import shutil
import tempfile
from pathlib import Path

_CACHE: dict = {}


def rehearsal(**options) -> Path:
    """A built diagnostic tree for the given generator options (built once per distinct option set)."""
    key = tuple(sorted(options.items()))
    if key not in _CACHE:
        from devsearch.execution.rehearsal import build_rehearsal
        base = Path(tempfile.mkdtemp(prefix="devsearch-reh-")).resolve()
        build_rehearsal(base, **options)
        (base / "runtime" / "run").mkdir(exist_ok=True)
        _CACHE[key] = base
        atexit.register(shutil.rmtree, base, ignore_errors=True)
    return _CACHE[key]


def copy_rehearsal(testcase, **options) -> Path:
    """A private writable copy (the bundle, evaluation side, backup and runtime dirs) for tests that run campaigns."""
    base = rehearsal(**options)
    copy = Path(tempfile.mkdtemp(prefix="devsearch-reh-copy-")).resolve()
    testcase.addCleanup(shutil.rmtree, copy, ignore_errors=True)
    for entry in base.iterdir():
        target = copy / entry.name
        shutil.copytree(entry, target, symlinks=True) if entry.is_dir() else shutil.copyfile(entry, target)
    host = (copy / "host-example.toml")
    host.write_text(host.read_text().replace(str(base), str(copy)))
    return copy
