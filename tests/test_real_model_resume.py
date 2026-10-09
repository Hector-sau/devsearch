"""scripts/check_real_model_resume.py without network or torch: argument refusals, the failure receipt, and the refusal of a package
imported from the checkout. A stand-in "interpreter" (a shell script) answers the environment probe; nothing reaches fetch-model."""

import importlib.util
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from helpers import REPO


def load():
    spec = importlib.util.spec_from_file_location("check_real_model_resume", REPO / "scripts" / "check_real_model_resume.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(os.name == "nt", "the stand-in interpreter is a POSIX shell script")
class RealModelResumeScriptTests(unittest.TestCase):
    def setUp(self):
        self.script = load()
        self.base = Path(tempfile.mkdtemp(prefix="devsearch-realmodel-"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)

    def interpreter(self, body: str) -> Path:
        path = self.base / "fake-python"
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def run_main(self, *argv):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = self.script.main([str(a) for a in argv])
        return code, out.getvalue(), err.getvalue()

    def test_a_failing_environment_still_writes_a_failed_receipt(self):
        python = self.interpreter("echo 'No module named torch' >&2\nexit 1\n")
        receipt = self.base / "receipt.json"
        code, out, _ = self.run_main("--python", python, "--work", self.base / "work", "--output", receipt)
        self.assertEqual(code, 1)
        report = json.loads(receipt.read_text())
        self.assertFalse(report["passed"])
        self.assertIn("environment exited 1", report["error"])
        self.assertEqual([s["step"] for s in report["steps"]], ["environment"])
        self.assertEqual(report["revision"], self.script.PINNED)
        self.assertEqual(json.loads(out.strip().splitlines()[-1])["passed"], False)

    def test_a_package_imported_from_the_checkout_fails_before_anything_else(self):
        inside = REPO / "src" / "devsearch" / "__init__.py"
        python = self.interpreter(f"printf '%s\\n' '{inside}' '2.14.1+cpu' '5.18.0' '3.12.0'\n")
        receipt = self.base / "receipt.json"
        code, _, _ = self.run_main("--python", python, "--work", self.base / "work", "--output", receipt)
        self.assertEqual(code, 1)
        report = json.loads(receipt.read_text())
        self.assertFalse(report["passed"])
        self.assertIs(report["installed_package_outside_checkout"], False)
        self.assertIn("imported from the checkout", report["error"])
        self.assertEqual([s["step"] for s in report["steps"]], ["environment"])     # no fetch, no network

    def test_a_work_directory_inside_the_checkout_or_not_empty_is_refused(self):
        python = self.interpreter("exit 0\n")
        code, _, err = self.run_main("--python", python, "--work", REPO / "build-realmodel-test", "--output", self.base / "r.json")
        self.assertEqual(code, 2)
        self.assertIn("outside the checkout", err)
        self.assertFalse((REPO / "build-realmodel-test").exists())
        busy = self.base / "busy"
        busy.mkdir()
        (busy / "x").write_text("x")
        code, _, err = self.run_main("--python", python, "--work", busy, "--output", self.base / "r.json")
        self.assertEqual(code, 2)
        self.assertIn("not empty", err)
        self.assertFalse((self.base / "r.json").exists())


if __name__ == "__main__":
    unittest.main()
