"""Scaffold safety tests for commands that are still stubs, plus package-boundary rules.

`index`, `retrieve` and `validate_manifest` are implemented and covered by
test_cli_baseline.py; their old fail-closed assertions were retired with them.
Replace a remaining command's fail-closed expectation with its real acceptance
tests when that command is implemented. No formal metrics, model quality or MPS
are tested here.
Run: PYTHONPATH=src python -m unittest discover -s tests -v
"""

import ast
import dataclasses
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from devsearch._cli import ALL_COMMANDS, IMPLEMENTED, STUB_COMMANDS
from devsearch.contracts import SearchRequest

DOCUMENTED_COMMANDS = {
    "prepare", "prepare_data", "index", "validate_manifest", "train", "evaluate",
    "predict", "retrieve", "serve", "export",
}

# Which first-party packages each package may import. Everything else must be stdlib.
CORE = {"devsearch.errors", "devsearch.contracts"}
ALLOWED_IMPORTS = {
    "devsearch.errors": set(),
    "devsearch.contracts": set(),
    "devsearch._cli": set(),
    "devsearch.artifacts": CORE | {"devsearch.artifacts", "devsearch"},
    "devsearch.data": CORE | {"devsearch.artifacts", "devsearch.data"},
    "devsearch.retrieval": CORE | {"devsearch.artifacts", "devsearch.data", "devsearch.retrieval"},
    "devsearch.modeling": {"devsearch.contracts"},
    "devsearch.training": {"devsearch.contracts"},
    "devsearch.evaluation": {"devsearch.contracts"},
    "devsearch.serving": {"devsearch.contracts"},
}
# Thin CLI entry modules (devsearch/<command>.py) may use any lower layer and the CLI helpers.
CLI_ALLOWED = CORE | {"devsearch._cli", "devsearch.artifacts", "devsearch.data", "devsearch.retrieval"}


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

    def test_command_inventory_matches_the_documented_contract(self):
        self.assertEqual(set(ALL_COMMANDS), DOCUMENTED_COMMANDS)
        self.assertEqual(len(ALL_COMMANDS), len(DOCUMENTED_COMMANDS))
        self.assertTrue(set(IMPLEMENTED).isdisjoint(STUB_COMMANDS))
        for command in ALL_COMMANDS:
            module = importlib.import_module(f"devsearch.{command}")
            self.assertTrue(callable(module.main), command)

    def test_stub_help_entrypoints_work_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            for command in STUB_COMMANDS:
                with self.subTest(command=command):
                    result = self.run_python(["-m", f"devsearch.{command}", "--help"], directory)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("SCAFFOLD ONLY", result.stdout)
                    self.assertIn("Only --help is implemented", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_stub_operations_fail_closed_and_never_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for command, (_, options) in STUB_COMMANDS.items():
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
from devsearch import _cli
for command in _cli.ALL_COMMANDS:
    entry = _cli.main if command in _cli.STUB_COMMANDS else importlib.import_module("devsearch." + command).main
    arguments = (command, ["--help"]) if command in _cli.STUB_COMMANDS else (["--help"],)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            entry(*arguments)
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("help did not exit")
for blocked in ("torch", "transformers", "datasets", "numpy", "fastapi", "streamlit", "yaml"):
    assert blocked not in sys.modules, blocked
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_python(["-c", script], directory)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_package_import_boundaries(self):
        # Layering: contracts/errors <- artifacts <- data <- retrieval <- CLI entry modules.
        # Core, retrieval and artifacts never import modeling/training/serving/evaluation,
        # CLI modules or UI code, and everything third-party stays out until deliberately added.
        for path in self.package.rglob("*.py"):
            relative = path.relative_to(self.package).with_suffix("")
            parts = ["devsearch", *relative.parts]
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join(parts)
            if len(parts) == 2 and parts[1] not in {"errors", "contracts", "_cli"}:
                allowed = CLI_ALLOWED                       # thin top-level command module
            elif len(parts) == 1:
                allowed = set()                             # the package root itself
            else:
                owner = next(key for key in sorted(ALLOWED_IMPORTS, key=len, reverse=True) if module == key or module.startswith(key + "."))
                allowed = ALLOWED_IMPORTS[owner]
            with self.subTest(module=module):
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
                        if name == "devsearch" or name.startswith("devsearch."):
                            self.assertTrue(
                                any(name == ok or name.startswith(ok + ".") for ok in allowed) or name == module,
                                f"{module} must not import {name}",
                            )
                        else:
                            self.assertIn(name.split(".")[0], sys.stdlib_module_names, f"{module} imports third-party {name}")

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
