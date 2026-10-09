"""Execution records: strict schemas, sealed manifests, binding chains, output hashing and the event log."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from devsearch.errors import ManifestError
from devsearch.execution.records import (
    EventLog, binding_chain, binding_id, binding_manifest, describe_outputs, new_state, parse_utc, read_binding, read_stage_manifest,
    read_state, stage_manifest, verify_outputs, write_json, write_state,
)


def binding(revision=0, parent=None, deadline="2026-10-06T08:00:00Z"):
    return binding_manifest({
        "revision": revision, "parent_sha256": parent, "paths": {"runtime_root": "/r", "run_root": "/r/runs", "evaluation_root": "/e", "backup_dir": "/b"},
        "mount": {"verified": True}, "deadline": deadline, "reserve_seconds": 1800, "limits": {}, "resources": {}, "device": {"profile": "cpu-diagnostic"},
        "authorization": None})


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-rec-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class StateTests(Base):
    def make(self):
        return new_state("plan-1", "a" * 64, ["preflight", "train"], {"id": "binding-x", "revision": 0, "sha256": "b" * 64}, now="2026-10-05T00:00:00Z")

    def test_round_trip_and_defaults(self):
        state = self.make()
        write_state(self.dir / "run-state.json", state, now="2026-10-05T00:00:01Z")
        loaded = read_state(self.dir / "run-state.json")
        self.assertEqual(loaded["status"], "pending")
        self.assertEqual(list(loaded["stages"]), ["preflight", "train"])
        self.assertEqual(loaded["updated_at"], "2026-10-05T00:00:01Z")
        self.assertEqual(loaded["stages"]["train"]["status"], "pending")

    def test_unknown_fields_and_statuses_are_rejected(self):
        state = self.make()
        for mutate in (lambda s: s.update(extra=1), lambda s: s.update(status="done"), lambda s: s["stages"]["train"].update(status="great"),
                       lambda s: s["stages"]["train"].update(unexpected=True), lambda s: s["active_binding"].pop("id")):
            broken = json.loads(json.dumps(state))
            mutate(broken)
            with self.assertRaises(ManifestError):
                write_state(self.dir / "bad.json", broken, now="2026-10-05T00:00:00Z")
        self.assertFalse((self.dir / "bad.json").exists())

    def test_utc_parsing_is_strict(self):
        self.assertEqual(parse_utc("2026-10-06T08:00:00Z", "t").year, 2026)
        for bad in ("2026-10-06", "2026-10-06T08:00:00", "yesterday", None, "2026-10-06T08:00:00+00:00"):
            with self.assertRaises(ManifestError):
                parse_utc(bad, "t")


class StageManifestTests(Base):
    def test_outputs_are_hashed_and_the_manifest_binds_plan_stage_attempt_inputs_and_outputs(self):
        (self.dir / "a.json").write_text('{"x": 1}')
        outputs = describe_outputs(self.dir, ["a.json"])
        manifest = stage_manifest(plan_id="plan-1", stage_id="s", attempt_id="attempt-1", status="succeeded", inputs={"dep": "c" * 64}, outputs=outputs, notes={"n": 1})
        write_json(self.dir / "stage-manifest.json", manifest)
        loaded = read_stage_manifest(self.dir / "stage-manifest.json")
        self.assertEqual((loaded["plan_id"], loaded["stage_id"], loaded["attempt_id"]), ("plan-1", "s", "attempt-1"))
        verify_outputs(self.dir, loaded["outputs"])
        (self.dir / "a.json").write_text('{"x": 2}')
        with self.assertRaisesRegex(ManifestError, "a.json"):
            verify_outputs(self.dir, loaded["outputs"])

    def test_edited_manifests_and_non_committable_statuses_are_rejected(self):
        (self.dir / "a.json").write_text("{}")
        manifest = stage_manifest(plan_id="p", stage_id="s", attempt_id="a", status="succeeded", inputs={}, outputs=describe_outputs(self.dir, ["a.json"]), notes={})
        edited = dict(manifest, stage_id="other")
        write_json(self.dir / "m.json", edited)
        with self.assertRaisesRegex(ManifestError, "manifest_sha256"):
            read_stage_manifest(self.dir / "m.json")
        failed = stage_manifest(plan_id="p", stage_id="s", attempt_id="a", status="failed", inputs={}, outputs={}, notes={})
        write_json(self.dir / "f.json", failed)
        with self.assertRaisesRegex(ManifestError, "committed"):
            read_stage_manifest(self.dir / "f.json")

    def test_outputs_must_be_regular_files_inside_the_base(self):
        (self.dir / "sub").mkdir()
        outside = self.dir.parent / "devsearch-outside.txt"
        outside.write_text("x")
        self.addCleanup(outside.unlink, missing_ok=True)
        (self.dir / "link").symlink_to(outside)
        for name in ("link", "../devsearch-outside.txt", "sub", "missing"):
            with self.subTest(name), self.assertRaises((ManifestError, FileNotFoundError)):
                describe_outputs(self.dir, [name])


class BindingTests(Base):
    def test_a_revision_chain_verifies_from_the_active_binding_back_to_revision_zero(self):
        first = binding()
        second = binding(1, first["manifest_sha256"], "2026-10-07T08:00:00Z")
        third = binding(2, second["manifest_sha256"], "2026-10-08T08:00:00Z")
        for index, manifest in enumerate((first, second, third)):
            write_json(self.dir / f"host-binding-{index}.json", manifest)
        chain = binding_chain(self.dir, third["manifest_sha256"])
        self.assertEqual([m["revision"] for m in chain], [0, 1, 2])
        self.assertEqual(binding_id(first), "binding-" + first["manifest_sha256"][:32])
        self.assertEqual(read_binding(self.dir / "host-binding-1.json")["deadline"], "2026-10-07T08:00:00Z")

    def test_a_missing_parent_or_gapped_revisions_fail_the_chain(self):
        first = binding()
        third = binding(2, first["manifest_sha256"])
        write_json(self.dir / "host-binding-0.json", first)
        write_json(self.dir / "host-binding-2.json", third)
        with self.assertRaisesRegex(ManifestError, "consecutive"):
            binding_chain(self.dir, third["manifest_sha256"])
        orphan = binding(1, "f" * 64)
        write_json(self.dir / "host-binding-1.json", orphan)
        with self.assertRaisesRegex(ManifestError, "missing"):
            binding_chain(self.dir, orphan["manifest_sha256"])

    def test_a_tampered_binding_is_rejected(self):
        manifest = binding()
        manifest["deadline"] = "2030-01-01T00:00:00Z"
        write_json(self.dir / "host-binding-0.json", manifest)
        with self.assertRaises(ManifestError):
            read_binding(self.dir / "host-binding-0.json")


class EventLogTests(Base):
    def test_events_are_appended_in_order_with_sequence_numbers(self):
        log = EventLog(self.dir / "events.jsonl", clock=lambda: "t")
        log.append("stage_started", stage_id="a")
        log.append("stage_succeeded", stage_id="a", seconds=1.5)
        events = EventLog(self.dir / "events.jsonl").read()
        self.assertEqual([(e["seq"], e["event"]) for e in events], [(1, "stage_started"), (2, "stage_succeeded")])

    def test_a_partial_final_line_is_discarded_and_appending_continues_cleanly(self):
        path = self.dir / "events.jsonl"
        log = EventLog(path, clock=lambda: "t")
        log.append("one")
        with path.open("a") as handle:
            handle.write('{"seq": 2, "ts": "t", "event": "tor')                # a crash mid-write
        torn = path.read_bytes()
        reopened = EventLog(path, clock=lambda: "t")
        self.assertEqual([e["event"] for e in reopened.read()], ["one"])
        self.assertEqual(path.read_bytes(), torn, "opening and reading a log never rewrites it (status is read-only)")
        reopened.append("two")
        self.assertEqual([e["event"] for e in EventLog(path).read()], ["one", "two"])
        self.assertEqual([e["seq"] for e in EventLog(path).read()], [1, 2])

    def test_corruption_before_the_tail_is_an_error_not_silently_skipped(self):
        path = self.dir / "events.jsonl"
        path.write_text('{"seq": 1}\nnot json\n{"seq": 3}\n')
        with self.assertRaisesRegex(ManifestError, "corrupt event at line 2"):
            EventLog(path)


if __name__ == "__main__":
    unittest.main()
