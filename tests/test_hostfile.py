"""Host binding: strict file, mount identity, containment, write probe, free space, independence and deadline."""

import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from devsearch.errors import ConfigError, PrerequisiteError
from devsearch.execution import hostfile
from devsearch.execution.hostfile import binding_for, load_host, revise_binding, run_checks, verify_runtime
from devsearch.execution.records import binding_manifest

TOML = """schema_version = 1
[device]
profile = "cpu-diagnostic"
[paths]
runtime_root = "{mount}/runtime"
bundle = "{mount}/bundle"
run_root = "{mount}/runtime/run"
evaluation_root = "{evaluation}"
backup_dir = "{backup}"
[mount]
path = "{mount}"
marker = ".devsearch-volume"
expected_id = "vol-1"
require_mount_point = false
[deadline]
at = "2099-01-01T00:00:00Z"
reserve_seconds = 60
[resources]
min_free_bytes = 1000
host_memory_cap_bytes = 1000000
device_memory_cap_bytes = 0
[limits]
process_timeout_seconds = 300
doctor_budget_seconds = 1800
cleanup_grace_seconds = 30
io_retries = 2
[backup]
require_independent_device = false
"""


class HostCase(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="devsearch-host-")).resolve()
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.mount = self.base / "vol"
        for sub in ("runtime/run", "bundle", "eval", "backup"):
            (self.mount / sub if sub.startswith(("runtime", "bundle")) else self.base / sub).mkdir(parents=True)
        (self.mount / ".devsearch-volume").write_text("vol-1\n")
        self.path = self.base / "host.toml"
        self.write()

    def write(self, text=None):
        self.path.write_text(text or TOML.format(mount=self.mount, evaluation=self.base / "eval", backup=self.base / "backup"))

    def checks(self, **kw):
        return {c["name"]: c for c in run_checks(binding_for(load_host(self.path)), **kw)}


class LoadTests(HostCase):
    def test_a_valid_file_produces_a_sealed_revision_zero_binding_with_absolute_paths(self):
        binding = binding_for(load_host(self.path))
        self.assertEqual((binding["revision"], binding["parent_sha256"]), (0, None))
        self.assertEqual(binding["paths"]["run_root"], str(self.mount / "runtime" / "run"))
        self.assertEqual(binding["paths"]["bundle"], str(self.mount / "bundle"))
        self.assertEqual(binding["device"], {"profile": "cpu-diagnostic"})
        self.assertIsNone(binding["authorization"])
        self.assertEqual(binding, binding_for(load_host(self.path)))

    def test_invalid_files_are_rejected(self):
        good = self.path.read_text()
        cases = {
            "unknown key (no credential field exists)": good.replace("[backup]", "[backup]\npassword = \"hunter2\""),
            "relative path": good.replace(f'run_root = "{self.mount}/runtime/run"', 'run_root = "runs/c1"'),
            "run root outside the mount": good.replace(f'run_root = "{self.mount}/runtime/run"', f'run_root = "{self.base}/elsewhere"'),
            "marker with a slash": good.replace('marker = ".devsearch-volume"', 'marker = "a/b"'),
            "bad deadline": good.replace("2099-01-01T00:00:00Z", "tomorrow"),
            "too many io retries": good.replace("io_retries = 2", "io_retries = 3"),
            "negative reserve": good.replace("reserve_seconds = 60", "reserve_seconds = -1"),
            "missing section": good.replace("[backup]\nrequire_independent_device = false\n", ""),
            "string for a boolean": good.replace("require_mount_point = false", 'require_mount_point = "no"'),
        }
        for name, text in cases.items():
            self.write(text)
            with self.subTest(name), self.assertRaises(ConfigError):
                load_host(self.path)


class CheckTests(HostCase):
    def test_a_healthy_rehearsal_host_passes_with_the_unverified_items_called_out(self):
        checks = self.checks()
        self.assertFalse([c for c in checks.values() if c["status"] == "failed"], checks)
        self.assertEqual(checks["mount"]["status"], "unverified")          # a plain directory is not a verified mount
        self.assertEqual(checks["backup_independent"]["status"], "unverified")
        self.assertEqual(checks["writable:run_root"]["status"], "passed")
        self.assertEqual(checks["deadline"]["status"], "passed")
        verify_runtime(binding_for(load_host(self.path)))

    def test_a_required_mount_point_that_is_only_a_directory_fails(self):
        self.write(self.path.read_text().replace("require_mount_point = false", "require_mount_point = true"))
        self.assertEqual(self.checks()["mount"]["status"], "failed")
        with self.assertRaisesRegex(PrerequisiteError, "not a mount point"):
            verify_runtime(binding_for(load_host(self.path)))
        with mock.patch.object(hostfile.os.path, "ismount", return_value=True):
            self.assertEqual(self.checks()["mount"]["status"], "passed")

    def test_a_missing_or_wrong_marker_means_the_volume_is_not_the_expected_one(self):
        (self.mount / ".devsearch-volume").write_text("some-other-volume\n")
        self.assertEqual(self.checks()["mount"]["status"], "failed")
        (self.mount / ".devsearch-volume").unlink()
        self.assertIn("marker", self.checks()["mount"]["detail"])
        with self.assertRaises(PrerequisiteError):
            verify_runtime(binding_for(load_host(self.path)))

    def test_an_unmounted_volume_leaves_no_directories_to_write_into(self):
        shutil.rmtree(self.mount)
        checks = self.checks()
        self.assertEqual(checks["mount"]["status"], "failed")
        self.assertEqual(checks["path:run_root"]["status"], "failed")
        self.assertNotIn("writable:run_root", checks)
        self.assertFalse(self.mount.exists(), "a check must never create the missing directory")

    def test_symbolic_links_are_refused_for_every_root(self):
        shutil.rmtree(self.base / "eval")
        (self.base / "eval-real").mkdir()
        (self.base / "eval").symlink_to(self.base / "eval-real")
        self.assertEqual(self.checks()["path:evaluation_root"]["status"], "failed")

    def test_a_run_root_that_resolves_outside_the_mount_fails_containment(self):
        outside = self.base / "outside"
        outside.mkdir()
        shutil.rmtree(self.mount / "runtime" / "run")
        (self.mount / "runtime" / "run").symlink_to(outside)
        checks = self.checks()
        self.assertEqual(checks["path:run_root"]["status"], "failed")

    def test_a_failing_write_probe_and_a_full_disk_are_failed_checks(self):
        real_open = Path.open
        def broken(self_, *a, **k):
            if ".write-probe-" in self_.name:
                raise OSError(28, "No space left on device")
            return real_open(self_, *a, **k)
        with mock.patch.object(Path, "open", broken):
            self.assertEqual(self.checks()["writable:run_root"]["status"], "failed")
        self.assertEqual(self.checks(required_free_bytes=10**18)["free_space:run_root"]["status"], "failed")
        self.assertEqual(self.checks(required_free_bytes=1)["free_space:run_root"]["status"], "passed")
        self.assertEqual([p.name for p in (self.mount / "runtime" / "run").iterdir() if p.name.startswith(".write-probe")], [])

    def test_a_backup_on_the_same_device_is_not_independent_when_independence_is_required(self):
        self.write(self.path.read_text().replace("require_independent_device = false", "require_independent_device = true"))
        self.assertEqual(self.checks()["backup_independent"]["status"], "failed")       # the temp tree is one filesystem

    def test_the_deadline_check_uses_the_reserve_and_an_expired_deadline_is_not_a_prerequisite_failure(self):
        late = datetime(2099, 1, 1, 0, 0, 30, tzinfo=timezone.utc)                              # 30 s after, reserve 60 s
        self.assertEqual(self.checks(now=datetime(2098, 12, 31, 23, 58, 0, tzinfo=timezone.utc))["deadline"]["status"], "passed")
        self.assertEqual(self.checks(now=datetime(2098, 12, 31, 23, 59, 30, tzinfo=timezone.utc))["deadline"]["status"], "failed")
        self.assertEqual(self.checks(now=late)["deadline"]["status"], "failed")
        expired = binding_manifest({**{k: v for k, v in binding_for(load_host(self.path)).items() if k not in ("manifest_sha256", "schema_version", "kind")},
                                    "deadline": "2020-01-01T00:00:00Z"})
        verify_runtime(expired)                                                                   # exit 3 is for missing prerequisites; expiry is a budget stop


class RevisionTests(HostCase):
    def test_a_revision_links_to_its_parent_and_may_change_paths_deadline_and_reserve_only(self):
        first = binding_for(load_host(self.path))
        self.write(self.path.read_text().replace("2099-01-01T00:00:00Z", "2099-06-01T00:00:00Z").replace("reserve_seconds = 60", "reserve_seconds = 90"))
        second = revise_binding(first, load_host(self.path), authorization="user approved the extra rental hours")
        self.assertEqual((second["revision"], second["parent_sha256"]), (1, first["manifest_sha256"]))
        self.assertEqual((second["deadline"], second["reserve_seconds"]), ("2099-06-01T00:00:00Z", 90))
        self.assertEqual(second["authorization"], "user approved the extra rental hours")
        self.write(self.path.read_text().replace('profile = "cpu-diagnostic"', 'profile = "apple-mps"'))
        with self.assertRaisesRegex(ConfigError, "device profile"):
            revise_binding(first, load_host(self.path))


if __name__ == "__main__":
    unittest.main()
