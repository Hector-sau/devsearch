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
from mltest import requires_ml

DOCUMENTED_COMMANDS = {
    "prepare", "prepare_data", "index", "validate_manifest", "train", "evaluate",
    "predict", "retrieve", "serve", "export",
    # execution tooling (docs/10 §5, docs/13)
    "doctor", "plan", "run", "status", "bundle", "rehearsal",
}

# Which first-party packages each package may import. Everything else must be stdlib, except the
# optional third-party roots declared per package below (the `ml` / `serve` extras in pyproject.toml).
CORE = {"devsearch.errors", "devsearch.contracts"}
LOW = CORE | {"devsearch.artifacts"}
# Retrieval needs only the corpus model; it must never reach dataset builders (cycle) or any higher layer.
RETRIEVAL_DATA = {"devsearch.data.corpus", "devsearch.data.sourcemap"}
# Dataset builders that run a retriever (candidates, mining, review pools, seals) sit above retrieval.
BUILDER_ALLOWED = LOW | {"devsearch.data", "devsearch.retrieval"}
ALLOWED_IMPORTS = {
    "devsearch.errors": set(),
    "devsearch.contracts": set(),
    "devsearch._cli": set(),
    "devsearch.artifacts": CORE | {"devsearch.artifacts", "devsearch"},
    "devsearch.data": LOW | {"devsearch.data"},
    "devsearch.data.candidates": BUILDER_ALLOWED,
    "devsearch.data.config": BUILDER_ALLOWED,
    "devsearch.data.prepare": BUILDER_ALLOWED,
    "devsearch.data.pairs": BUILDER_ALLOWED,
    "devsearch.data.review": BUILDER_ALLOWED,
    "devsearch.data.seals": BUILDER_ALLOWED,
    "devsearch.retrieval": LOW | RETRIEVAL_DATA | {"devsearch.retrieval"},
    "devsearch.evaluation": LOW | {"devsearch.data", "devsearch.evaluation"},
    "devsearch.modeling": LOW | {"devsearch.modeling"},
    "devsearch.training": LOW | {"devsearch.data", "devsearch.evaluation", "devsearch.modeling", "devsearch.training"},
    "devsearch.serving": LOW | {"devsearch.data", "devsearch.retrieval", "devsearch.evaluation", "devsearch.modeling", "devsearch.serving"},
    # the orchestration layer sits on top of everything and is imported only by the thin command modules
    "devsearch.execution": LOW | {"devsearch.data", "devsearch.retrieval", "devsearch.evaluation", "devsearch.modeling", "devsearch.training",
                                  "devsearch.serving", "devsearch.execution"},
}
OPTIONAL_THIRD_PARTY = {
    "devsearch.modeling": {"torch", "tokenizers", "transformers", "safetensors"},
    "devsearch.training": {"torch"},
    "devsearch.serving": {"fastapi", "uvicorn", "starlette", "pydantic", "streamlit", "httpx"},
    "devsearch.execution": {"torch", "tokenizers", "transformers", "safetensors", "fastapi", "httpx"},
}
# Thin CLI entry modules (devsearch/<command>.py) may use any lower layer and the CLI helpers.
CLI_ALLOWED = LOW | {
    "devsearch._cli", "devsearch.data", "devsearch.retrieval", "devsearch.evaluation",
    "devsearch.modeling", "devsearch.training", "devsearch.serving", "devsearch.execution",
}


def _owner(module):
    return next(key for key in sorted(ALLOWED_IMPORTS, key=len, reverse=True) if module == key or module.startswith(key + "."))


def _third_party_roots(path):
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or ""]
        else:
            continue
        roots.update(n.split(".")[0] for n in names if n.split(".")[0] not in sys.stdlib_module_names and n.split(".")[0] != "devsearch")
    return roots


def _module_of(package_dir, path):
    parts = list(path.relative_to(package_dir).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(["devsearch", *parts])


def _transitive_third_party(package_dir):
    """module -> third-party roots it needs directly or through any first-party module it imports (parent packages included)."""
    modules = {_module_of(package_dir, path): path for path in package_dir.rglob("*.py")}
    direct = {name: _third_party_roots(path) for name, path in modules.items()}
    edges = {}
    for name, path in modules.items():
        found = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                targets = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for target in targets:
                parts = target.split(".")
                if parts[0] != "devsearch":
                    continue
                for end in range(1, len(parts) + 1):
                    candidate = ".".join(parts[:end])
                    if candidate in modules:
                        found.add(candidate)
        edges[name] = found - {name}
    needs = {name: set(roots) for name, roots in direct.items()}
    changed = True
    while changed:
        changed = False
        for name, deps in edges.items():
            for dep in deps:
                if not needs[dep] <= needs[name]:
                    needs[name] |= needs[dep]
                    changed = True
    return needs


ML_ROOTS = {"torch", "tokenizers", "transformers", "safetensors"}


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
        needs = _transitive_third_party(self.package)
        # a module that needs an optional library, directly or through anything it imports, is covered by the extras import tests
        modules = sorted(name for name, roots in needs.items() if not roots)
        self.assertIn("devsearch.retrieve", modules)
        self.assertNotIn("devsearch.execution.doctor", modules)      # imports the device layer, which needs torch
        self.assertIn("devsearch.execution.controller", modules)     # the state machine itself is dependency-free
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
for blocked in ("torch", "transformers", "tokenizers", "safetensors", "datasets", "numpy", "fastapi", "streamlit", "yaml"):
    assert blocked not in sys.modules, blocked
'''
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_python(["-c", script], directory)
            self.assertEqual(result.returncode, 0, result.stderr)

    @requires_ml
    def test_optional_extra_modules_import_without_network(self):
        needs = _transitive_third_party(self.package)
        modules = sorted(name for name, roots in needs.items() if roots and roots <= ML_ROOTS)
        self.assertTrue(modules)
        script = f'''
import importlib, sys
def audit(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.bind"):   # torch itself probes ldconfig; only the network is off limits
        raise AssertionError("unexpected network operation: " + event)
sys.addaudithook(audit)
for name in {json.dumps(modules)}:
    importlib.import_module(name)
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=directory, env=self.environment,
                                    capture_output=True, text=True, timeout=120, check=False)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_package_import_boundaries(self):
        # Layering: contracts/errors <- artifacts <- data <- retrieval <- dataset builders <- evaluation/
        # modeling <- training <- serving <- CLI entry modules. A lower layer never imports a higher one,
        # and third-party libraries are confined to the packages that declare them (the optional extras).
        for path in self.package.rglob("*.py"):
            relative = path.relative_to(self.package).with_suffix("")
            parts = ["devsearch", *relative.parts]
            if parts[-1] == "__init__":
                parts.pop()
            module = ".".join(parts)
            third_party = set()
            if len(parts) == 2 and parts[1] not in {"errors", "contracts", "_cli"}:
                allowed = CLI_ALLOWED                       # thin top-level command module
            elif len(parts) == 1:
                allowed = set()                             # the package root itself
            else:
                owner = _owner(module)
                allowed = ALLOWED_IMPORTS[owner]
                third_party = OPTIONAL_THIRD_PARTY.get(owner, set())
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
                        root = name.split(".")[0]
                        if name == "devsearch" or name.startswith("devsearch."):
                            self.assertTrue(
                                any(name == ok or name.startswith(ok + ".") for ok in allowed) or name == module,
                                f"{module} must not import {name}",
                            )
                        else:
                            self.assertTrue(
                                root in sys.stdlib_module_names or root in third_party,
                                f"{module} imports third-party {name}",
                            )

    def test_third_party_roots_are_declared_as_optional_extras(self):
        import tomllib
        project = tomllib.loads((self.package.parent.parent / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertEqual(project.get("dependencies", []), [], "the baseline has no runtime dependencies")
        declared = {
            requirement.split(";")[0].split("[")[0].strip().lower().replace("-", "_").split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("~")[0].strip()
            for extra in project["optional-dependencies"].values() for requirement in extra
        }
        used = set()
        for path in self.package.rglob("*.py"):
            used |= _third_party_roots(path)
        self.assertLessEqual(used, declared, f"undeclared third-party imports: {sorted(used - declared)}")

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
