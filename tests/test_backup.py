"""Backup: verified copy, atomic publication, immutability, tamper detection, failure behavior."""

import errno
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch.errors import ManifestError
from devsearch.execution import backup as backup_module
from devsearch.execution.backup import RECEIPT_NAME, backup, verify_backup


class BackupCase(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="devsearch-bak-")).resolve()
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.src = self.base / "src"
        (self.src / "release" / "model").mkdir(parents=True)
        (self.src / "release" / "a.txt").write_text("alpha")
        (self.src / "release" / "model" / "w.bin").write_bytes(b"\x00\x01" * 1000)
        (self.src / "plan.json").write_text('{"plan": 1}')
        self.dest_root = self.base / "backup-volume"
        self.dest_root.mkdir()
        self.dest = self.dest_root / "plan-1"
        self.sources = {"release": self.src / "release", "plan.json": self.src / "plan.json"}

    def run_backup(self, **kw):
        return backup(kw.pop("sources", self.sources), kw.pop("destination", self.dest), plan_id="plan-1")


class CopyTests(BackupCase):
    def test_every_file_is_copied_verified_and_listed_by_relative_name(self):
        receipt = self.run_backup()
        self.assertEqual(set(receipt["files"]), {"release/a.txt", "release/model/w.bin", "plan.json"})
        self.assertEqual((receipt["totals"]["files"], receipt["totals"]["bytes"]), (3, 5 + 2000 + 11))
        self.assertEqual((self.dest / "release" / "model" / "w.bin").read_bytes(), b"\x00\x01" * 1000)
        self.assertEqual(verify_backup(self.dest)["manifest_sha256"], receipt["manifest_sha256"])
        self.assertNotIn(str(self.base), (self.dest / RECEIPT_NAME).read_text())

    def test_an_identical_rerun_is_a_noop_and_a_changed_source_is_refused_not_overwritten(self):
        first = self.run_backup()
        self.assertEqual(self.run_backup()["manifest_sha256"], first["manifest_sha256"])
        (self.src / "release" / "a.txt").write_text("changed")
        with self.assertRaisesRegex(ManifestError, "never overwritten"):
            self.run_backup()
        self.assertEqual((self.dest / "release" / "a.txt").read_text(), "alpha")
        self.assertEqual([p.name for p in self.dest_root.iterdir()], ["plan-1"])

    def test_a_missing_destination_volume_fails_without_creating_anything(self):
        with self.assertRaisesRegex(ManifestError, "never creates its parent"):
            self.run_backup(destination=self.base / "unmounted" / "plan-1")
        self.assertFalse((self.base / "unmounted").exists())

    def test_bad_sources_are_refused(self):
        with self.assertRaisesRegex(ManifestError, "does not exist"):
            self.run_backup(sources={"x": self.src / "nope"})
        with self.assertRaisesRegex(ManifestError, "nothing"):
            self.run_backup(sources={})
        (self.src / "link").symlink_to(self.src / "plan.json")
        with self.assertRaisesRegex(ManifestError, "symbolic"):
            self.run_backup(sources={"link": self.src / "link"})
        with self.assertRaisesRegex(ManifestError, "plain relative"):
            self.run_backup(sources={"../escape": self.src / "plan.json"})
        self.assertEqual(list(self.dest_root.iterdir()), [])


class FailureTests(BackupCase):
    def test_a_crash_before_the_final_rename_leaves_no_backup_that_looks_complete(self):
        with mock.patch.object(backup_module.os, "rename", side_effect=OSError("killed")):
            with self.assertRaises(OSError):
                self.run_backup()
        self.assertEqual(list(self.dest_root.iterdir()), [])
        self.run_backup()

    def test_a_full_disk_part_way_cleans_up_and_reports_the_error(self):
        real = backup_module._copy_verified
        calls = []

        def failing(source, target):
            calls.append(source.name)
            if len(calls) == 2:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real(source, target)
        with mock.patch.object(backup_module, "_copy_verified", failing):
            with self.assertRaises(OSError) as raised:
                self.run_backup()
        self.assertEqual(raised.exception.errno, errno.ENOSPC)
        self.assertEqual(list(self.dest_root.iterdir()), [])

    def test_a_copy_that_does_not_match_its_source_checksum_is_rejected(self):
        real_hash = backup_module.sha256_file
        seen = {"n": 0}

        def corrupt(path):
            seen["n"] += 1
            return "0" * 64 if seen["n"] == 2 else real_hash(path)
        with mock.patch.object(backup_module, "sha256_file", corrupt):
            with self.assertRaisesRegex(ManifestError, "checksum"):
                self.run_backup()
        self.assertEqual(list(self.dest_root.iterdir()), [])


class VerifyTests(BackupCase):
    def test_bit_rot_extra_and_missing_files_are_all_detected(self):
        self.run_backup()
        target = self.dest / "release" / "model" / "w.bin"
        original = target.read_bytes()
        target.write_bytes(original[:-1] + b"\xff")
        with self.assertRaisesRegex(ManifestError, "w.bin"):
            verify_backup(self.dest)
        target.write_bytes(original)
        (self.dest / "stray.txt").write_text("x")
        with self.assertRaisesRegex(ManifestError, "stray.txt"):
            verify_backup(self.dest)
        (self.dest / "stray.txt").unlink()
        (self.dest / "plan.json").unlink()
        with self.assertRaisesRegex(ManifestError, "plan.json"):
            verify_backup(self.dest)

    def test_an_edited_receipt_is_rejected(self):
        self.run_backup()
        path = self.dest / RECEIPT_NAME
        path.write_text(path.read_text().replace('"plan-1"', '"plan-2"'))
        with self.assertRaises(ManifestError):
            verify_backup(self.dest)


if __name__ == "__main__":
    unittest.main()
