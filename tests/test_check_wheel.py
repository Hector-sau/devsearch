"""Reject stale or inconsistent release wheels without network or ML packages."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from helpers import REPO


spec = importlib.util.spec_from_file_location("release_check_wheel", REPO / "scripts" / "check_wheel.py")
check_wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_wheel)


class WheelIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="devsearch-wheel-identity-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.expected = check_wheel.expected_metadata()

    def wheel(self, *, metadata=None, imported_version=None, duplicate=False):
        fields = self.expected if metadata is None else metadata
        version = self.expected["Version"]
        path = self.root / f"devsearch-{version}-py3-none-any.whl"
        info = f"devsearch-{version}.dist-info"
        body = "Metadata-Version: 2.1\n" + "".join(f"{key}: {value}\n" for key, value in fields.items())
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("devsearch/__init__.py", f"__version__ = {(imported_version or version)!r}\n")
            archive.writestr(f"{info}/METADATA", body)
            archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            archive.writestr(f"{info}/RECORD", "")
            if duplicate:
                archive.writestr("other-1.0.0.dist-info/METADATA", body)
        return path

    def test_metadata_matching_the_current_release_is_accepted(self):
        check_wheel.check_wheel_metadata(self.wheel(), self.expected)

    def test_a_stale_release_version_is_rejected_before_installation(self):
        with self.assertRaisesRegex(SystemExit, "wheel Version"):
            check_wheel.check_wheel_metadata(self.wheel(metadata={**self.expected, "Version": "0.0.0"}), self.expected)

    def test_an_unrelated_distribution_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "wheel Name"):
            check_wheel.check_wheel_metadata(self.wheel(metadata={**self.expected, "Name": "other"}), self.expected)

    def test_a_missing_python_requirement_is_rejected(self):
        fields = {key: value for key, value in self.expected.items() if key != "Requires-Python"}
        with self.assertRaisesRegex(SystemExit, "wheel Requires-Python"):
            check_wheel.check_wheel_metadata(self.wheel(metadata=fields), self.expected)

    def test_a_changed_python_requirement_is_rejected(self):
        with self.assertRaisesRegex(SystemExit, "wheel Requires-Python"):
            check_wheel.check_wheel_metadata(self.wheel(metadata={**self.expected, "Requires-Python": ">=3.12"}), self.expected)

    def test_multiple_distributions_cannot_hide_the_checked_metadata(self):
        with self.assertRaisesRegex(SystemExit, "exactly one distribution"):
            check_wheel.check_wheel_metadata(self.wheel(duplicate=True), self.expected)

    def test_actual_offline_install_rejects_a_stale_imported_version(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "check_wheel.py"), str(self.wheel(imported_version="0.0.0"))],
            cwd=self.root, capture_output=True, text=True, timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("import version and metadata version mismatch", result.stderr)
        self.assertIn("Successfully installed devsearch", result.stdout)
        self.assertNotIn("PASS:", result.stdout)


if __name__ == "__main__":
    unittest.main()
