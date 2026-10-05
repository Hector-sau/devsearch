"""Campaign controller: recovery windows, leases, failure classes, budget and exit codes — in real processes.

The fake campaign has no ML. Crashes use `os._exit(137)` at the documented points, so there is no cleanup,
exactly like a killed process; the next invocation must recover from what is on disk.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from devsearch.execution.lock import Lease
from devsearch.execution.records import read_state, read_stage_manifest, write_json, binding_manifest

DRIVER = Path(__file__).resolve().parent / "controller_driver.py"
SRC = Path(__file__).resolve().parent.parent / "src"


class ControllerCase(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="devsearch-ctl-")).resolve()
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)
        (self.work / "control").mkdir()

    def drive(self, *args, fault=None, expect=None):
        env = {k: v for k, v in os.environ.items() if k != "DEVSEARCH_FAULT"}
        env["PYTHONPATH"] = str(SRC)
        if fault:
            env["DEVSEARCH_FAULT"] = fault
        result = subprocess.run([sys.executable, "-B", str(DRIVER), str(self.work), *args], capture_output=True, text=True, env=env, timeout=60)
        if expect is not None:
            self.assertEqual(result.returncode, expect, result.stderr)
        return result

    def ran(self):
        log = self.work / "ran.log"
        return log.read_text().split() if log.exists() else []

    def control(self, name):
        (self.work / "control" / name).write_text("")

    def clear(self, name):
        (self.work / "control" / name).unlink()

    def state(self):
        return read_state(self.work / "run" / "run-state.json")

    def stage_dir(self, stage):
        return self.work / "run" / "stages" / stage

    def events(self):
        return [json.loads(line)["event"] for line in (self.work / "run" / "events.jsonl").read_text().splitlines()]


class HappyPathTests(ControllerCase):
    def test_a_clean_run_commits_every_stage_and_reports_complete(self):
        self.drive(expect=0)
        self.assertEqual(self.ran(), ["a", "b", "c", "d"])
        state = self.state()
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual([s["status"] for s in state["stages"].values()], ["succeeded"] * 4)
        manifest = read_stage_manifest(self.stage_dir("b") / "stage-manifest.json")
        self.assertEqual(manifest["inputs"], {"a": read_stage_manifest(self.stage_dir("a") / "stage-manifest.json")["manifest_sha256"]})
        self.assertTrue((self.stage_dir("b") / manifest["attempt_id"] / "out.txt").is_file())
        self.assertFalse((self.work / "run" / "campaign.lock").exists(), "the lease is released on completion")
        self.assertIn("campaign_succeeded", self.events())

    def test_resume_of_a_completed_campaign_reuses_every_hash_valid_stage(self):
        self.drive(expect=0)
        self.drive("--resume", expect=0)
        self.assertEqual(self.ran(), ["a", "b", "c", "d"], "nothing runs twice")

    def test_usage_errors_are_exit_2_and_change_nothing(self):
        self.drive("--resume", expect=2)
        self.assertFalse((self.work / "run" / "run-state.json").exists())
        self.drive(expect=0)
        result = self.drive(expect=2)
        self.assertIn("--resume", result.stderr)
        self.assertEqual(self.ran(), ["a", "b", "c", "d"])


class CrashWindowTests(ControllerCase):
    def test_a_crash_after_outputs_but_before_the_manifest_ignores_the_leftover_and_reruns_only_that_stage(self):
        self.drive(fault="stage_outputs_written:b", expect=137)
        self.assertEqual(self.ran(), ["a", "b"])
        self.assertFalse((self.stage_dir("b") / "stage-manifest.json").exists())
        self.assertTrue((self.work / "run" / "campaign.lock").exists(), "a killed process leaves its lease behind")
        self.assertEqual(self.state()["stages"]["b"]["status"], "running")
        self.drive("--resume", expect=0)
        self.assertEqual(self.ran(), ["a", "b", "b", "c", "d"], "a is reused from its committed manifest; b runs again")
        self.assertTrue(any(p.name.startswith(".abandoned-") for p in self.stage_dir("b").iterdir()), "the leftover attempt is kept aside as evidence")
        self.assertEqual(self.state()["stages"]["b"]["attempt"], 2)
        self.assertIn("stage_interrupted_found", self.events())

    def test_a_crash_after_the_manifest_commit_is_reconstructed_without_rerunning_or_overwriting(self):
        self.drive(fault="stage_manifest_committed:b", expect=137)
        committed = read_stage_manifest(self.stage_dir("b") / "stage-manifest.json")
        self.assertEqual(self.state()["stages"]["b"]["status"], "running", "the state update never happened")
        self.drive("--resume", expect=0)
        self.assertEqual(self.ran(), ["a", "b", "c", "d"], "b is not run again")
        self.assertEqual(read_stage_manifest(self.stage_dir("b") / "stage-manifest.json")["manifest_sha256"], committed["manifest_sha256"])
        self.assertEqual(self.state()["stages"]["b"]["attempt_id"], committed["attempt_id"])
        self.assertIn("stage_reconstructed", self.events())

    def test_a_crash_after_the_state_update_changes_nothing_on_resume(self):
        self.drive(fault="state_updated:b", expect=137)
        self.drive("--resume", expect=0)
        self.assertEqual(self.ran(), ["a", "b", "c", "d"])

    def test_a_crash_in_the_middle_of_a_stage_reruns_that_stage_from_the_start(self):
        self.drive(fault="mid_stage:c", expect=137)
        self.drive("--resume", expect=0)
        self.assertEqual(self.ran(), ["a", "b", "c", "c", "d"])

    def test_a_reconstructed_stage_with_a_tampered_output_fails_instead_of_being_trusted(self):
        self.drive(fault="stage_manifest_committed:b", expect=137)
        manifest = read_stage_manifest(self.stage_dir("b") / "stage-manifest.json")
        (self.stage_dir("b") / manifest["attempt_id"] / "out.txt").write_text("tampered")
        result = self.drive("--resume", expect=1)
        self.assertIn("out.txt", result.stderr)
        self.assertEqual(self.ran(), ["a", "b"], "nothing was regenerated over the committed artifact")
        self.assertEqual((self.stage_dir("b") / manifest["attempt_id"] / "out.txt").read_text(), "tampered")

    def test_a_committed_manifest_from_an_unexpected_attempt_is_refused_not_overwritten(self):
        self.drive(fault="stage_manifest_committed:b", expect=137)
        state = self.state()
        state["stages"]["b"]["attempt_id"] = "attempt-someone-else"
        write_json(self.work / "run" / "run-state.json", state)
        result = self.drive("--resume", expect=1)
        self.assertIn("refusing to overwrite", result.stderr)


class IntegrityTests(ControllerCase):
    def test_a_corrupted_committed_output_is_a_failed_gate_not_a_silent_regeneration(self):
        self.drive(expect=0)
        manifest = read_stage_manifest(self.stage_dir("a") / "stage-manifest.json")
        target = self.stage_dir("a") / manifest["attempt_id"] / "out.txt"
        target.write_text("bit rot")
        result = self.drive("--resume", expect=1)
        self.assertIn("out.txt", result.stderr)
        self.assertEqual(self.ran(), ["a", "b", "c", "d"])
        self.assertEqual(target.read_text(), "bit rot")

    def test_a_dependency_that_changed_makes_downstream_outputs_stale_and_blocks_reuse(self):
        self.drive(expect=0)
        manifest_path = self.stage_dir("a") / "stage-manifest.json"
        manifest = read_stage_manifest(manifest_path)
        from devsearch.execution.records import stage_manifest
        forged = stage_manifest(plan_id=manifest["plan_id"], stage_id="a", attempt_id=manifest["attempt_id"], status="succeeded",
                                inputs=manifest["inputs"], outputs=manifest["outputs"], notes={"edited": True})
        write_json(manifest_path, forged)
        result = self.drive("--resume", expect=1)
        self.assertIn("no longer match", result.stderr)

    def test_a_missing_committed_manifest_for_a_succeeded_stage_fails(self):
        self.drive(expect=0)
        (self.stage_dir("c") / "stage-manifest.json").unlink()
        self.drive("--resume", expect=1)

    def test_a_changed_plan_cannot_resume_an_old_run(self):
        self.drive(fault="state_updated:a", expect=137)
        result = self.drive("--resume", "--plan-marker", "plan-v2", expect=1)
        self.assertIn("different plan", result.stderr)
        self.assertEqual(self.ran(), ["a"])

    def test_the_status_report_verifies_artifacts_not_just_the_state_file(self):
        self.drive(expect=0)
        from controller_driver import build
        campaign, _ = build(self.work)
        report = campaign.status_report()
        self.assertTrue(report["complete"])
        self.assertEqual({s["artifact_check"] for s in report["stages"].values()}, {"verified"})
        manifest = read_stage_manifest(self.stage_dir("d") / "stage-manifest.json")
        (self.stage_dir("d") / manifest["attempt_id"] / "out.txt").write_text("changed")
        report = campaign.status_report()
        self.assertFalse(report["complete"])
        self.assertTrue(report["stages"]["d"]["artifact_check"].startswith("FAILED"))
        self.assertEqual(self.state()["status"], "succeeded", "the state file alone would have claimed success")


class OwnershipTests(ControllerCase):
    def test_a_second_owner_is_refused_while_the_first_holds_the_lease(self):
        (self.work / "run").mkdir()
        with Lease(self.work / "run" / "campaign.lock", purpose="other-campaign", plan_id="p"):
            result = self.drive(expect=3)
            self.assertIn("second owner is refused", result.stderr)
        self.assertEqual(self.ran(), [])
        self.assertFalse((self.work / "run" / "run-state.json").exists())

    def test_a_killed_owners_stale_lease_is_recovered_after_its_identity_is_checked(self):
        self.drive(fault="stage_outputs_written:a", expect=137)
        lock = json.loads((self.work / "run" / "campaign.lock").read_text())
        self.assertNotEqual(lock["pid"], os.getpid())
        self.drive("--resume", expect=0)
        self.assertTrue(any(".stale-" in p.name for p in (self.work / "run").iterdir()))

    def test_a_lease_from_another_host_is_never_broken(self):
        (self.work / "run").mkdir()
        (self.work / "run" / "campaign.lock").write_text(json.dumps({"host": "other-machine", "pid": 1, "start": None, "token": "t" * 32, "purpose": "campaign", "plan_id": "p"}))
        result = self.drive(expect=3)
        self.assertIn("cannot verify", result.stderr)


class FailureClassTests(ControllerCase):
    def test_a_fatal_failure_stops_retains_evidence_and_is_not_retried_by_resume(self):
        self.control("fail-c")
        result = self.drive(expect=4)
        self.assertIn("out of memory", result.stderr)
        state = self.state()
        self.assertEqual((state["status"], state["stages"]["c"]["status"], state["stages"]["c"]["error"]["kind"]), ("failed", "failed", "fatal"))
        self.assertEqual(state["stages"]["b"]["status"], "succeeded")
        self.clear("fail-c")
        result = self.drive("--resume", expect=4)
        self.assertIn("new plan", result.stderr)
        self.assertEqual(self.ran(), ["a", "b", "c"], "the failed stage is not silently retried, even after the cause disappears")

    def test_an_unexpected_exception_is_an_execution_failure_not_a_traceback_or_a_success(self):
        self.control("bug-b")
        result = self.drive(expect=4)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.state()["stages"]["b"]["error"]["kind"], "fatal")

    def test_an_interrupt_is_resumable_and_completes_after_a_clean_restart(self):
        self.control("interrupt-c")
        self.drive(expect=4)
        self.assertEqual(self.state()["stages"]["c"]["error"]["kind"], "interrupted")
        self.clear("interrupt-c")
        self.drive("--resume", expect=0)
        self.assertEqual(self.ran(), ["a", "b", "c", "c", "d"])

    def test_disk_full_is_a_prerequisite_failure_that_keeps_prior_artifacts_and_resumes_once_space_returns(self):
        self.control("disk-c")
        result = self.drive(expect=3)
        self.assertIn("No space left", result.stderr)
        self.assertEqual(self.state()["stages"]["b"]["status"], "succeeded")
        self.assertTrue((self.stage_dir("b") / "stage-manifest.json").is_file())
        self.clear("disk-c")
        self.drive("--resume", expect=0)

    def test_a_missing_mount_stops_before_any_stage_and_never_relocates_work(self):
        self.control("unmount")
        result = self.drive(expect=3)
        self.assertIn("not mounted", result.stderr)
        self.assertEqual(self.ran(), [])
        self.clear("unmount")
        self.assertEqual(self.state()["status"], "failed", "a failed start is recorded, not left 'running'")
        self.drive(expect=2)                                          # the run exists now; continuing it is an explicit --resume
        self.drive("--resume", expect=0)
        self.assertEqual(self.ran(), ["a", "b", "c", "d"])

    def test_a_mount_lost_after_some_stages_keeps_their_artifacts(self):
        self.drive(fault="state_updated:b", expect=137)
        self.control("unmount")
        self.drive("--resume", expect=3)
        self.assertEqual(self.ran(), ["a", "b"])
        self.assertTrue((self.stage_dir("a") / "stage-manifest.json").is_file())


class ReviewAndBudgetTests(ControllerCase):
    def test_awaiting_review_pauses_with_exit_5_and_never_reaches_later_stages(self):
        self.control("review-c")
        self.drive(expect=5)
        state = self.state()
        self.assertEqual((state["status"], state["stages"]["c"]["status"], state["stages"]["d"]["status"]), ("awaiting_review", "awaiting_review", "pending"))
        self.assertEqual(self.ran(), ["a", "b", "c"])
        self.drive("--resume", expect=5)
        self.drive("--resume", "--reviews", expect=0)
        self.assertEqual(self.ran(), ["a", "b", "c", "c", "c", "d"])
        self.assertEqual(self.state()["status"], "succeeded")

    def test_a_stage_that_cannot_fit_the_remaining_window_stops_with_exit_6_before_starting(self):
        result = self.drive("--deadline", "2099-01-01T00:00:00Z", "--estimate", "c=10000000000000", expect=6)
        self.assertIn("budget stop", result.stderr)
        self.assertEqual(self.ran(), ["a", "b"])
        state = self.state()
        self.assertEqual((state["status"], state["stages"]["c"]["status"]), ("stopped_budget", "pending"))
        self.assertIn("host-binding revision", state["next_action"])

    def test_a_deadline_that_has_already_passed_starts_nothing(self):
        self.drive("--deadline", "2020-01-01T00:00:00Z", expect=6)
        self.assertEqual(self.ran(), [])

    def test_extending_the_deadline_through_a_binding_revision_allows_resume(self):
        self.drive("--estimate", "c=10000000000000", expect=6)
        plan_dir = self.work / "plan"
        first = json.loads((plan_dir / "host-binding-0.json").read_text())
        second = binding_manifest({
            "revision": 1, "parent_sha256": first["manifest_sha256"], "paths": first["paths"], "mount": first["mount"], "deadline": "2099-06-01T00:00:00Z",
            "reserve_seconds": 60, "limits": {}, "resources": {}, "device": first["device"], "authorization": None})
        write_json(plan_dir / "host-binding-1.json", second)
        from controller_driver import build
        campaign, initial = build(self.work)
        self.assertEqual(campaign.run(resume=True, initial_binding_sha256=initial, active_binding_sha256=second["manifest_sha256"]), 0)
        self.assertEqual(self.state()["active_binding"]["revision"], 1)

    def test_a_binding_chain_that_does_not_start_at_the_plans_initial_binding_is_refused(self):
        self.drive(fault="state_updated:a", expect=137)
        from controller_driver import build
        campaign, _ = build(self.work)
        self.assertEqual(campaign.run(resume=True, initial_binding_sha256="0" * 64), 1)


class GraphValidationTests(unittest.TestCase):
    def test_the_stage_graph_must_be_ordered_and_unique(self):
        from devsearch.errors import ConfigError
        from devsearch.execution.controller import Campaign, Stage
        noop = lambda ctx: None
        work = Path(tempfile.mkdtemp(prefix="devsearch-graph-"))
        self.addCleanup(shutil.rmtree, work, ignore_errors=True)
        for stages in ([Stage("a", (), noop), Stage("a", (), noop)], [Stage("a", ("b",), noop), Stage("b", (), noop)], [Stage("a", ("zzz",), noop)]):
            with self.assertRaises(ConfigError):
                Campaign(plan={}, plan_sha256="0" * 64, plan_dir=work, run_root=work, stages=stages, verify_runtime=lambda b: None, accelerator=False)


if __name__ == "__main__":
    unittest.main()
