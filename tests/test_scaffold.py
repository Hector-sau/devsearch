"""Temporary scaffold safety tests, not retrieval/training acceptance tests.

Replace a command's fail-closed expectation with its real acceptance tests when
that command is implemented. No formal metrics, model quality or MPS are tested.
Run: PYTHONPATH=src python -m unittest discover -s tests -v
"""

import ast
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from devsearch._cli import COMMANDS
from devsearch.contracts import SearchRequest


class ScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.find_spec("devsearch")
        cls.package = Path(spec.origin).parent
        cls.environment = dict(os.environ)
        # Works with either the source tree or an installed wheel, outside cwd.
        cls.environment["PYTHONPATH"] = str(cls.package.parent)
        cls.environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def run_python(self, args, cwd):
        return subprocess.run(
            [sys.executable, "-B", *args], cwd=cwd,
            env=self.environment, capture_output=True, text=True,
            timeout=15, check=False,
        )

    def test_all_help_entrypoints_work_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            for command in COMMANDS:
                with self.subTest(command=command):
                    result = self.run_python(
                        ["-m", f"devsearch.{command}", "--help"], directory
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("SCAFFOLD ONLY", result.stdout)
                    self.assertIn("Only --help is implemented", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_operations_fail_closed_and_never_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for command, (_, options) in COMMANDS.items():
                with self.subTest(command=command):
                    args = ["-m", f"devsearch.{command}"]
                    # Valid proposed options still cannot execute anything.
                    for option in ("config", "output", "run-dir", "manifest"):
                        if option in options:
                            args.extend([f"--{option}", str(Path(directory) / option)])
                    result = self.run_python(args, directory)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("operation not implemented", result.stderr)
                    self.assertEqual(list(Path(directory).iterdir()), [])

    def test_imports_and_help_do_not_touch_network_or_write(self):
        modules = []
        for path in sorted(self.package.rglob("*.py")):
            parts = list(path.relative_to(self.package).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            modules.append(".".join(["devsearch", *parts]))
        script = f'''
import contextlib, importlib, io, os, sys
modules = {json.dumps(modules)}
def audit(event, args):
    if event.startswith("socket.") or event.startswith("subprocess."):
        raise AssertionError("unexpected external operation: " + event)
    if event in {{"os.mkdir", "os.remove", "os.rename", "os.rmdir"}}:
        raise AssertionError("unexpected filesystem mutation: " + event)
    if event == "open":
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
        ):
            raise AssertionError("unexpected write: " + str(args[0]))
sys.addaudithook(audit)
for name in modules:
    importlib.import_module(name)
from devsearch._cli import COMMANDS, main
for command in COMMANDS:
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            main(command, ["--help"])
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("help did not exit")
for blocked in ("torch", "transformers", "datasets", "numpy", "fastapi", "streamlit"):
    assert blocked not in sys.modules, blocked
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_python(["-c", script], directory)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_current_scaffold_import_boundaries(self):
        # Temporary stdlib-only rule: update deliberately when real adapters land.
        for path in self.package.rglob("*.py"):
            with self.subTest(file=str(path)):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        self.assertEqual(node.level, 0, "Use explicit package imports")
                        names = [node.module or ""]
                    else:
                        continue
                    for name in names:
                        if name.startswith("devsearch."):
                            expected = "devsearch.contracts" if path.parent != self.package else "devsearch._cli"
                            self.assertEqual(name, expected)
                        else:
                            self.assertIn(name.split(".")[0], sys.stdlib_module_names)

    def test_request_boundary_is_frozen_and_uses_immutable_filters(self):
        request = SearchRequest(
            query="How do checkpoints work?", sources=("pytorch-tutorials",),
            versions=(("pytorch-tutorials", "example-revision"),),
        )
        self.assertEqual(request.mode, "baseline")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            request.query = "changed"
        self.assertIsInstance(request.sources, tuple)
        self.assertIsInstance(request.versions, tuple)


if __name__ == "__main__":
    unittest.main()
