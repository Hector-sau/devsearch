"""The thin campaign-tool commands (`rehearsal`, `bundle`, `doctor`, `plan`) through their real `main`, with the real doctor workers.

Everything is the diagnostic CPU rehearsal: it proves the commands, their exit codes and refusals, not any real backend.
"""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mltest import HAVE_ML, requires_ml

if HAVE_ML:
    from devsearch import bundle, doctor, plan, rehearsal
    from rehearsal_fixture import copy_rehearsal


def call(module, *argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = module.main([str(a) for a in argv])
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


@requires_ml
class RehearsalCommandTests(unittest.TestCase):
    def test_a_tree_is_built_in_an_empty_directory_and_is_complete(self):
        target = Path(tempfile.mkdtemp(prefix="devsearch-reh-cli-")).resolve()
        self.addCleanup(shutil.rmtree, target, ignore_errors=True)
        code, out, err = call(rehearsal, "--output", target, "--topics", 18)
        self.assertEqual((code, err), (0, ""), err)
        summary = json.loads(out)
        self.assertTrue(summary["diagnostic"])
        for name in ("bundle", "evaluation", "backup", "runtime/run", "host-example.toml", "rehearsal-summary.json"):
            self.assertTrue((target / name).exists(), name)
        code, _, err = call(rehearsal, "--output", target, "--topics", 18)
        self.assertEqual(code, 1, "a non-empty directory is refused")
        empty = Path(tempfile.mkdtemp(prefix="devsearch-reh-cli-")).resolve()
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        self.assertEqual(call(rehearsal, "--output", empty, "--topics", 3)[0], 1)
        self.assertEqual(list(empty.iterdir()), [])


@requires_ml
class BundleCommandTests(unittest.TestCase):
    def test_assemble_is_idempotent_and_verify_detects_a_changed_protocol(self):
        root = copy_rehearsal(self, topics=18)
        directory = root / "bundle"
        before = json.loads((directory / "bundle-manifest.json").read_text())["manifest_sha256"]
        code, out, err = call(bundle, "assemble", "--bundle", directory)
        self.assertEqual((code, err), (0, ""), err)
        report = json.loads(out)
        self.assertEqual((report["manifest_sha256"], report["diagnostic"]), (before, True))
        self.assertEqual(call(bundle, "verify", "--bundle", directory)[0], 0)
        protocol = directory / "protocol.toml"
        protocol.write_text(protocol.read_text().replace("min_group_mrr_gain = 0.02", "min_group_mrr_gain = 0.0"))
        code, out, err = call(bundle, "verify", "--bundle", directory)
        self.assertEqual((code, out), (1, ""))
        self.assertIn("protocol.toml changed", err)
        self.assertEqual(call(bundle, "verify")[0], 2)


@requires_ml
class DoctorAndPlanCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.holder = unittest.TestCase()
        cls.root = copy_rehearsal(cls.holder, topics=18)
        cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        cls.host = cls.root / "host-example.toml"
        cls.report = cls.root / "doctor-report.json"
        cls.doctor = call(doctor, "--profile", "cpu-diagnostic", "--host", cls.host, "--output", cls.report)

    def test_the_real_doctor_calibrates_on_cpu_writes_its_report_and_never_overwrites_it(self):
        code, out, err = self.doctor
        self.assertEqual((code, err), (0, ""), err)
        summary = json.loads(out)
        self.assertTrue(summary["diagnostic"])
        self.assertEqual(summary["gates"]["backend_verified[cpu-diagnostic]"], "not_applicable")
        self.assertEqual(summary["failed_checks"], [])
        before = self.report.read_bytes()
        code, _, err = call(doctor, "--profile", "cpu-diagnostic", "--host", self.host, "--output", self.report)
        self.assertEqual(code, 1)
        self.assertIn("never overwritten", err)
        self.assertEqual(self.report.read_bytes(), before)
        self.assertEqual(call(doctor, "--profile", "apple-mps", "--host", self.host, "--output", self.root / "other.json")[0], 2)

    def test_plan_creates_an_immutable_plan_then_revises_only_the_binding(self):
        directory = self.root / "plan"
        directory.mkdir(exist_ok=True)
        path = directory / "execution-plan.json"
        if not path.exists():
            code, out, err = call(plan, "--host", self.host, "--doctor-report", self.report, "--output", path)
            self.assertEqual((code, err), (0, ""), err)
            created = json.loads(out)
            self.assertTrue(created["rehearsal"])
            self.assertEqual(created["estimate_label"], "estimate")
        self.assertTrue((directory / "host-binding-0.json").is_file())
        plan_before = path.read_bytes()
        code, out, err = call(plan, "--host", self.host, "--doctor-report", self.report, "--output", path)
        self.assertEqual((code, err), (0, ""), "the identical plan is a no-op")
        self.assertEqual(path.read_bytes(), plan_before)
        code, _, err = call(plan, "--host", self.host, "--doctor-report", self.report, "--output", path, "--wheel-sha256", "ab" * 32)
        self.assertEqual(code, 1, "a different plan at the same path is refused")
        self.assertIn("immutable", err)
        self.assertEqual(path.read_bytes(), plan_before)
        later = self.root / "host-later.toml"
        later.write_text(self.host.read_text().replace("2099-01-01T00:00:00Z", "2099-06-01T00:00:00Z"))
        code, out, err = call(plan, "--revise-binding", "--plan", path, "--host", later, "--authorization", "extra window approved for the test")
        self.assertEqual((code, err), (0, ""), err)
        self.assertEqual(json.loads(out)["revision"], 1)
        self.assertTrue((directory / "host-binding-1.json").is_file())
        other_profile = self.root / "host-other.toml"
        other_profile.write_text(self.host.read_text().replace('profile = "cpu-diagnostic"', 'profile = "apple-mps"'))
        code, _, err = call(plan, "--revise-binding", "--plan", path, "--host", other_profile)
        self.assertNotEqual(code, 0, "the device profile can never change in a revision")
        self.assertEqual(call(plan, "--revise-binding", "--host", self.host)[0], 2)
        self.assertEqual(call(plan, "--host", self.host)[0], 2)


if __name__ == "__main__":
    unittest.main()
