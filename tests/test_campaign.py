"""The real campaign on a diagnostic bundle: the documented failure-injection rehearsal (docs/11 §4) and honest outcomes.

Everything here runs the genuine stage code (training, export, ranking, sealing, scoring, release, reload, backup) on
CPU with synthetic data. Crashes use `os._exit` in a real child process; patches (out-of-memory, NaN, clocks) run the
campaign in-process. These tests prove mechanics and recovery, never ranking quality.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from devsearch.errors import ManifestError
from mltest import HAVE_ML

if HAVE_ML:
    import torch
    from devsearch.execution import faults
    from devsearch.execution.controller import (
        EXIT_AWAITING_REVIEW, EXIT_BUDGET, EXIT_COMPLETE, EXIT_FAILURE, EXIT_INVALID, EXIT_PREREQUISITE,
    )
    from devsearch.execution.doctor import run_doctor
    from devsearch.execution.hostfile import load_host, revise_binding
    from devsearch.execution.lock import Lease
    from devsearch.execution.plan import build_plan, write_plan
    from devsearch.execution.records import read_json, read_stage_manifest, read_state, write_json
    from devsearch.execution.runner import open_campaign
    from devsearch.execution.stages import SELECTION_FILE
    from devsearch.modeling.cross_encoder import CrossEncoder
    from devsearch.run import OFFLINE
    from rehearsal_fixture import copy_rehearsal
    from test_doctor import healthy, scripted

SRC = Path(__file__).resolve().parent.parent / "src"
TOPICS = 36


class CampaignCase(unittest.TestCase):
    options: dict = {"topics": TOPICS}

    def planned_copy(self):
        """A private copy of the diagnostic tree with a (fake-doctor) plan written beside it: (root, host, plan_path, plan)."""
        root = copy_rehearsal(self, **self.options)
        host = load_host(root / "host-example.toml")
        report = run_doctor(host, root / "bundle", worker=scripted(healthy), now=lambda: "2026-10-05T00:00:00Z")
        plan, binding = build_plan(host, root / "bundle", report, now="2026-10-05T00:00:00Z")
        (root / "plan").mkdir()
        write_plan(root / "plan" / "execution-plan.json", plan, binding)
        (root / "plan" / "doctor-report.json").write_text(json.dumps(report))
        return root, host, root / "plan" / "execution-plan.json", plan

    def setUp(self):
        if not HAVE_ML:
            self.skipTest("torch/transformers/tokenizers/safetensors are not installed")
        self.root, self.host, self.plan_path, self.plan = self.planned_copy()
        self.run_root = self.root / "runtime" / "run"

    # -- drivers ----------------------------------------------------------------------------------------
    def cli(self, *extra, fault=None, fault_after=None, env=None):
        child = {k: v for k, v in os.environ.items() if k not in (faults.ENV, faults.ENV_AFTER)}
        child["PYTHONPATH"] = str(SRC)
        child.update(env or {})
        if fault:
            child[faults.ENV] = fault
        if fault_after is not None:
            child[faults.ENV_AFTER] = str(fault_after)
        return subprocess.run([sys.executable, "-B", "-m", "devsearch.run", "--plan", str(self.plan_path), *extra], capture_output=True, text=True,
                              env=child, timeout=900)

    def in_process(self, *, resume=False, reviews=None, clock=None):
        with mock.patch.dict(os.environ, OFFLINE):
            campaign, initial, active, _, _ = open_campaign(self.plan_path, reviews=reviews)
            if clock is not None:
                campaign._clock = clock
            return campaign, campaign.run(resume=resume, initial_binding_sha256=initial, active_binding_sha256=active)

    def state(self):
        return read_state(self.run_root / "run-state.json")

    def stage_dir(self, stage_id):
        return self.run_root / "stages" / stage_id

    def output(self, stage_id, name):
        manifest = read_stage_manifest(self.stage_dir(stage_id) / "stage-manifest.json")
        return self.stage_dir(stage_id) / manifest["attempt_id"] / name

    def events(self):
        return [json.loads(line) for line in (self.run_root / "events.jsonl").read_text().splitlines()]

    def statuses(self):
        return {k: v["status"] for k, v in self.state()["stages"].items()}

    def selection(self):
        return read_json(self.output("select", SELECTION_FILE), "selection")


class HappyPathTests(CampaignCase):
    def test_a_complete_campaign_produces_every_artifact_and_every_stage_verifies(self):
        result = self.cli()
        self.assertEqual(result.returncode, EXIT_COMPLETE, result.stderr[-3000:])
        self.assertEqual(set(self.statuses().values()), {"succeeded"})
        self.assertEqual(self.state()["status"], "succeeded")
        report = read_json(self.output("score", "final-report.json"), "report")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["selection_id"], self.selection()["selection_id"])
        self.assertTrue(any("DIAGNOSTIC REHEARSAL" in note for note in report["limitations"]))
        summary = read_json(self.output("summary", "campaign-summary.json"), "summary")
        self.assertEqual((summary["status"], summary["recipes_missing"], summary["backup"]["status"]), ("complete", [], "verified"))
        acceptance = read_json(self.output("summary", "acceptance-report.json"), "acceptance")
        self.assertEqual(acceptance["evidence_label"], "fixture")
        self.assertIs(acceptance["accelerator_run_ready[cpu-diagnostic]"], False)
        for estimate in summary["estimates_vs_observed_seconds"].values():
            self.assertIn("estimate", estimate)
        backup = self.root / "backup" / ("plan-" + self.plan["manifest_sha256"][:32])
        self.assertTrue((backup / "backup-receipt.json").is_file())
        self.assertTrue(any(p.name.endswith(".pt") for p in (backup / "best-checkpoints").iterdir()))

    def test_status_verifies_artifacts_and_resume_of_a_finished_campaign_reuses_everything(self):
        self.assertEqual(self.cli().returncode, 0)
        result = subprocess.run([sys.executable, "-B", "-m", "devsearch.status", "--plan", str(self.plan_path)], capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(SRC)}, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["complete"])
        before = (self.run_root / "events.jsonl").read_text().count("stage_started")
        self.assertEqual(self.cli("--resume").returncode, 0)
        self.assertEqual((self.run_root / "events.jsonl").read_text().count("stage_started"), before, "no stage ran twice")

    def test_the_default_mode_comes_from_validation_alone_and_the_test_ranking_never_chose_it(self):
        self.assertEqual(self.cli().returncode, 0)
        selection = self.selection()
        report = read_json(self.output("score", "final-report.json"), "report")
        self.assertEqual(selection["decision"]["default_mode"], report["default_mode"])
        self.assertEqual(selection["decision"]["default_mode"], "rerank", "the planted cue is learnable, so a recipe qualifies")
        release = read_json(self.output("export_release", "release/release-manifest.json"), "release")
        self.assertEqual((release["status"], release["default_mode"]), ("evaluated", "rerank"))
        self.assertEqual(release["model"]["model_id"], read_json(self.output(f"export:{selection['decision']['recipe_id']}:42", "export.json"), "e")["model_id"])


class NoTestAccessBeforeSelectionTests(CampaignCase):
    def test_no_file_of_the_evaluation_root_is_opened_before_the_selection_commits(self):
        evaluation = str(self.root / "evaluation")
        select_manifest = str(self.run_root / "stages" / "select" / "stage-manifest.json")
        log: list[tuple[str, str]] = []
        armed = {"on": False}

        def hook(event, args):
            if not armed["on"]:
                return
            if event == "open" and isinstance(args[0], str) and args[0].startswith(evaluation):
                log.append(("open", args[0]))
            elif event == "os.rename" and str(args[1]) == select_manifest:
                log.append(("select_committed", select_manifest))
        sys.addaudithook(hook)
        armed["on"] = True
        try:
            _, code = self.in_process()
        finally:
            armed["on"] = False
        self.assertEqual(code, 0)
        kinds = [k for k, _ in log]
        self.assertIn("select_committed", kinds)
        index = kinds.index("select_committed")
        self.assertEqual([k for k in kinds[:index] if k == "open"], [], "a test-side file was opened before `select` committed")
        self.assertTrue(any(k == "open" for k in kinds[index:]), "the test side is read after the lock")

    def test_world_refuses_the_evaluation_side_until_select_has_committed(self):
        with mock.patch.dict(os.environ, OFFLINE):
            campaign, *_ = open_campaign(self.plan_path)
        from devsearch.execution.controller import StageContext
        campaign.state = {"stages": {"select": {"status": "pending"}}}
        stages = {s.stage_id: s for s in campaign.stages}
        context = StageContext(campaign, stages["rank_test"], "attempt-x", self.root)
        world = next(stage.run.__self__.world for stage in campaign.stages if stage.stage_id == "rank_test")
        with self.assertRaisesRegex(ManifestError, "has not succeeded"):
            world.evaluation(context)


class CrashRecoveryTests(CampaignCase):
    def test_a_crash_in_the_middle_of_training_resumes_from_the_last_epoch_and_matches_an_uninterrupted_run(self):
        pairs = self.plan["data"]["counts"]["pairs"]
        per_epoch = math.ceil(pairs / self.plan["training"]["microbatch"])
        crashed = self.cli(fault="mid_stage:train:full:42", fault_after=per_epoch + 3)
        self.assertEqual(crashed.returncode, 137)
        run_dir = self.run_root / "work" / "train-full-42" / "checkpoints"
        completed = sorted(p.name for p in run_dir.glob("epoch-*.pt"))
        self.assertEqual(completed, ["epoch-0000.pt", "epoch-0001.pt"], "epoch zero and the first complete epoch exist; epoch 2 was interrupted mid-way")
        self.assertEqual(self.state()["stages"]["train:full:42"]["status"], "running")
        resumed = self.cli("--resume")
        self.assertEqual(resumed.returncode, 0, resumed.stderr[-3000:])
        interrupted_summary = read_json(self.output("train:full:42", "run-summary.json"), "run-summary")
        reference_root, _, reference_plan, _ = self.planned_copy()                    # an uninterrupted campaign over identical inputs
        env = {**os.environ, "PYTHONPATH": str(SRC)}
        env.pop(faults.ENV, None)
        run = subprocess.run([sys.executable, "-B", "-m", "devsearch.run", "--plan", str(reference_plan)], capture_output=True, text=True, env=env, timeout=900)
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        reference_dir = reference_root / "runtime" / "run" / "stages" / "train:full:42"
        manifest = read_stage_manifest(reference_dir / "stage-manifest.json")
        reference_summary = read_json(reference_dir / manifest["attempt_id"] / "run-summary.json", "ref")
        self.assertEqual(interrupted_summary["global_step"], reference_summary["global_step"])
        self.assertEqual(interrupted_summary["lr_trace_sha256"], reference_summary["lr_trace_sha256"], "the learning-rate sequence is identical")
        self.assertEqual(interrupted_summary["best_epoch"], reference_summary["best_epoch"])
        for resumed_epoch, reference_epoch in zip(interrupted_summary["epochs"], reference_summary["epochs"]):
            self.assertAlmostEqual(resumed_epoch["group_mrr"], reference_epoch["group_mrr"], places=6)
            self.assertAlmostEqual(resumed_epoch["group_ndcg"], reference_epoch["group_ndcg"], places=6)
            self.assertAlmostEqual(resumed_epoch["train_loss"], reference_epoch["train_loss"], places=5)

    def test_a_crash_after_a_real_stages_outputs_but_before_its_manifest_reruns_only_that_stage(self):
        self.assertEqual(self.cli(fault="stage_outputs_written:export:full:42").returncode, 137)
        self.assertFalse((self.stage_dir("export:full:42") / "stage-manifest.json").exists())
        before = {s: read_stage_manifest(self.stage_dir(s) / "stage-manifest.json")["manifest_sha256"] for s in ("train:full:42", "train:frozen:42")}
        self.assertEqual(self.cli("--resume").returncode, 0)
        after = {s: read_stage_manifest(self.stage_dir(s) / "stage-manifest.json")["manifest_sha256"] for s in before}
        self.assertEqual(before, after, "committed training stages were reused, not retrained")
        self.assertTrue(any(p.name.startswith(".abandoned-") for p in self.stage_dir("export:full:42").iterdir()))

    def test_a_crash_after_the_selection_manifest_commit_reconstructs_the_lock_without_reselecting(self):
        self.assertEqual(self.cli(fault="stage_manifest_committed:select").returncode, 137)
        committed = read_stage_manifest(self.stage_dir("select") / "stage-manifest.json")
        self.assertEqual(self.state()["stages"]["select"]["status"], "running")
        self.assertEqual(self.cli("--resume").returncode, 0)
        self.assertEqual(read_stage_manifest(self.stage_dir("select") / "stage-manifest.json")["manifest_sha256"], committed["manifest_sha256"])
        self.assertIn("stage_reconstructed", [e["event"] for e in self.events()])

    def test_a_second_owner_is_refused_and_a_dead_owners_lease_is_recovered(self):
        self.run_root.mkdir(exist_ok=True)
        with Lease(self.run_root / "campaign.lock", purpose="someone-else", plan_id="p"):
            result = self.cli()
            self.assertEqual(result.returncode, EXIT_PREREQUISITE)
            self.assertIn("second owner is refused", result.stderr)
        self.assertFalse((self.run_root / "run-state.json").exists())
        self.assertEqual(self.cli(fault="stage_outputs_written:preflight").returncode, 137)
        self.assertTrue((self.run_root / "campaign.lock").exists())
        self.assertEqual(self.cli("--resume").returncode, 0)


class FailureTests(CampaignCase):
    def test_out_of_memory_stops_with_evidence_and_is_never_retried_or_silently_adapted(self):
        calls = {"n": 0}
        real = __import__("devsearch.training.loop", fromlist=["collate"]).collate

        def oom(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 20:
                raise RuntimeError("MPS backend out of memory (simulated)")
            return real(*args, **kwargs)
        with mock.patch("devsearch.training.loop.collate", oom):
            _, code = self.in_process()
        self.assertEqual(code, EXIT_FAILURE)
        state = self.state()
        failed = [k for k, v in state["stages"].items() if v["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(state["stages"][failed[0]]["error"]["kind"], "fatal")
        self.assertIn("out of memory", state["stages"][failed[0]]["error"]["message"])
        run_dir = self.run_root / "work" / failed[0].replace(":", "-") / "checkpoints"
        self.assertTrue((run_dir / "epoch-0000.pt").is_file(), "valid checkpoints are retained as evidence")
        self.assertEqual(state["stages"]["select"]["status"], "pending")
        _, code = self.in_process(resume=True)
        self.assertEqual(code, EXIT_FAILURE, "resume does not retry a fatal failure")
        self.assertEqual(self.statuses()["select"], "pending")
        self.assertFalse((self.stage_dir("rank_test") / "stage-manifest.json").exists(), "no result exists for a campaign that did not finish")

    def test_a_non_finite_loss_stops_the_run_and_publishes_nothing(self):
        real_forward = CrossEncoder.forward
        calls = {"n": 0}

        def poisoned(self_, *args, **kwargs):
            calls["n"] += 1
            output = real_forward(self_, *args, **kwargs)
            return output * math.nan if calls["n"] > 30 else output
        with mock.patch.object(CrossEncoder, "forward", poisoned):
            _, code = self.in_process()
        self.assertEqual(code, EXIT_FAILURE)
        failed = next(v for v in self.state()["stages"].values() if v["status"] == "failed")
        self.assertIn("non-finite", failed["error"]["message"])
        self.assertEqual(self.statuses()["export_release"], "pending")

    def test_an_unavailable_requested_backend_is_a_failed_prerequisite_never_a_cpu_run(self):
        host_text = (self.root / "host-example.toml").read_text().replace('profile = "cpu-diagnostic"', 'profile = "nvidia-cuda"').replace(
            "device_memory_cap_bytes = 0", "device_memory_cap_bytes = 1000000000")
        (self.root / "host-cuda.toml").write_text(host_text)
        if torch.cuda.is_available():
            self.skipTest("a CUDA device exists on this machine")
        result = subprocess.run(
            [sys.executable, "-B", "-m", "devsearch.doctor", "--profile", "nvidia-cuda", "--host", str(self.root / "host-cuda.toml"),
             "--output", str(self.root / "plan" / "doctor-cuda.json")], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC)}, timeout=600)
        self.assertEqual(result.returncode, EXIT_PREREQUISITE, result.stderr)
        report = read_json(self.root / "plan" / "doctor-cuda.json", "doctor")
        gate = report["gates"]["backend_verified[nvidia-cuda]"]
        self.assertEqual(gate["status"], "failed")
        self.assertIn("CUDA", gate["detail"])
        self.assertIsNone(report["calibration"]["microbatch"])
        plan_try = subprocess.run([sys.executable, "-B", "-m", "devsearch.plan", "--host", str(self.root / "host-cuda.toml"), "--doctor-report",
                                   str(self.root / "plan" / "doctor-cuda.json"), "--output", str(self.root / "plan" / "plan-cuda.json")],
                                  capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC)}, timeout=300)
        self.assertNotEqual(plan_try.returncode, 0)
        self.assertFalse((self.root / "plan" / "plan-cuda.json").exists())

    def test_a_missing_model_asset_fails_before_any_work_and_is_never_reported_as_success(self):
        (self.root / "bundle" / "model-assets" / "model.safetensors").unlink()
        result = self.cli()
        self.assertEqual(result.returncode, EXIT_INVALID, result.stderr[-2000:])
        self.assertIn("model.safetensors", result.stderr)
        self.assertFalse((self.run_root / "run-state.json").exists(), "nothing was started, and nothing claims success")
        self.assertEqual(list((self.run_root / "stages").iterdir()) if (self.run_root / "stages").exists() else [], [])

    def test_a_changed_protocol_or_bundle_after_planning_is_refused_by_preflight(self):
        protocol = self.root / "bundle" / "protocol.toml"
        protocol.write_text(protocol.read_text().replace("min_group_mrr_gain = 0.02", "min_group_mrr_gain = 0.0"))
        result = self.cli()
        self.assertEqual(result.returncode, EXIT_INVALID, result.stderr[-2000:])
        self.assertIn("protocol", result.stderr)
        self.assertFalse((self.run_root / "run-state.json").exists())
        self.assertFalse(any((self.run_root / "work").iterdir()) if (self.run_root / "work").exists() else False, "no training started")

    def test_changing_the_source_after_planning_is_refused_by_preflight_before_any_work(self):
        """The plan identifies the exact code (a source-snapshot hash); edited code is a different program."""
        edited = Path(tempfile.mkdtemp(prefix="devsearch-src-")).resolve()
        self.addCleanup(shutil.rmtree, edited, ignore_errors=True)
        shutil.copytree(SRC / "devsearch", edited / "devsearch", ignore=shutil.ignore_patterns("__pycache__"))
        target = edited / "devsearch" / "errors.py"
        target.write_text(target.read_text() + "\n# an edit made after the plan was frozen\n")
        result = self.cli(env={"PYTHONPATH": str(edited)})
        self.assertEqual(result.returncode, EXIT_INVALID, result.stderr[-2000:])
        self.assertIn("not the code this plan identifies", result.stderr)
        self.assertEqual(self.statuses()["preflight"], "failed")
        self.assertEqual(self.statuses()["select"], "pending")
        self.assertFalse(any((self.run_root / "work").iterdir()) if (self.run_root / "work").exists() else False, "no training started")

    def test_tampered_test_candidates_stop_the_campaign_before_any_final_metric_exists(self):
        target = self.root / "evaluation" / "candidates" / "candidates-test.jsonl"
        target.write_bytes(target.read_bytes().replace(b'"rank":1', b'"rank":2', 1))
        result = self.cli()
        self.assertEqual(result.returncode, EXIT_INVALID, result.stderr[-2000:])
        statuses = self.statuses()
        self.assertEqual(statuses["select"], "succeeded", f"validation work was fine; the lock was committed before the test side was touched: {statuses}\n{result.stderr[-1500:]}")
        self.assertEqual(statuses["rank_test"], "failed")
        for later in ("score", "export_release", "backup", "summary"):
            self.assertEqual(statuses[later], "pending")
        self.assertEqual(list(self.stage_dir("score").glob("attempt-*/final-report.json")) if self.stage_dir("score").exists() else [], [])

    def test_a_missing_mount_stops_before_work_and_prior_artifacts_survive(self):
        self.assertEqual(self.cli(fault="state_updated:select").returncode, 137)
        marker = self.root / ".devsearch-volume"
        marker.rename(self.root / ".devsearch-volume.away")
        result = self.cli("--resume")
        self.assertEqual(result.returncode, EXIT_PREREQUISITE, result.stderr[-1500:])
        self.assertIn("marker", result.stderr)
        self.assertTrue((self.stage_dir("select") / "stage-manifest.json").is_file())
        (self.root / ".devsearch-volume.away").rename(marker)
        self.assertEqual(self.cli("--resume").returncode, 0)

    def test_a_corrupted_committed_output_is_a_failed_gate_not_a_silent_retrain(self):
        self.assertEqual(self.cli().returncode, 0)
        target = self.output("train:full:42", "run-summary.json")
        target.write_text(target.read_text().replace('"best_epoch"', '"best_epoc"'))
        result = self.cli("--resume")
        self.assertEqual(result.returncode, EXIT_INVALID)
        self.assertIn("run-summary.json", result.stderr)
        self.assertIn('"best_epoc"', target.read_text(), "the corrupted file was not regenerated over")

    def test_formal_compute_refuses_to_start_without_the_offline_environment(self):
        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            campaign, initial, active, _, _ = open_campaign(self.plan_path)
            code = campaign.run(resume=False, initial_binding_sha256=initial, active_binding_sha256=active)
        self.assertEqual(code, EXIT_PREREQUISITE)
        self.assertEqual(self.statuses()["preflight"], "failed")

    def test_a_plan_cannot_be_resumed_with_changed_content(self):
        self.assertEqual(self.cli(fault="state_updated:preflight").returncode, 137)
        data = json.loads(self.plan_path.read_text())
        data["training"]["microbatch"] = 2
        self.plan_path.write_text(json.dumps(data))
        result = self.cli("--resume")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest_sha256", result.stderr)
        self.assertEqual(self.statuses()["data_check"], "pending", "nothing ran under the altered plan")


class BudgetTests(CampaignCase):
    def test_a_window_that_closes_mid_campaign_stops_safely_and_a_binding_revision_lets_it_finish(self):
        committed_select = {"seen": False}
        base_clock = lambda: datetime.now(timezone.utc)

        def clock():
            path = self.run_root / "stages" / "select" / "stage-manifest.json"
            if path.exists():
                committed_select["seen"] = True
            return base_clock() + (timedelta(days=36500 * 2) if committed_select["seen"] else timedelta(0))
        campaign, code = self.in_process(clock=clock)
        self.assertEqual(code, EXIT_BUDGET)
        state = self.state()
        self.assertEqual(state["status"], "stopped_budget")
        self.assertEqual(state["stages"]["select"]["status"], "succeeded")
        self.assertEqual(state["stages"]["rank_test"]["status"], "pending")
        self.assertFalse((self.stage_dir("rank_test") / "stage-manifest.json").exists())
        later = self.root / "host-later.toml"
        later.write_text((self.root / "host-example.toml").read_text().replace("2099-01-01T00:00:00Z", "2199-01-01T00:00:00Z"))
        binding = json.loads((self.root / "plan" / "host-binding-0.json").read_text())
        revision = revise_binding(binding, load_host(later), authorization="the user approved the extra window")
        write_json(self.root / "plan" / "host-binding-1.json", revision)
        result = self.cli("--resume")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(self.state()["active_binding"]["revision"], 1)
        self.assertEqual(self.state()["status"], "succeeded")

    def test_a_training_run_stopped_by_the_epoch_gate_resumes_without_losing_epochs(self):
        from devsearch.execution import stages as stages_module
        seen = {"n": 0}
        real_remaining = stages_module.StageContext.remaining_seconds

        def tight(self_):
            if self_.stage_id == "train:full:42":
                seen["n"] += 1
                if seen["n"] in (3, 4):                    # the gate at epoch 2 sees too little time
                    return 0.001
            return real_remaining(self_)
        with mock.patch.object(stages_module.StageContext, "remaining_seconds", tight):
            _, code = self.in_process()
        self.assertEqual(code, EXIT_BUDGET)
        self.assertEqual(self.state()["stages"]["train:full:42"]["status"], "stopped_budget")
        run_dir = self.run_root / "work" / "train-full-42" / "checkpoints"
        kept = sorted(p.name for p in run_dir.glob("epoch-*.pt"))
        self.assertGreaterEqual(len(kept), 2)
        self.assertFalse(any(p.name.startswith("diagnostic-") for p in run_dir.iterdir()), "a gate stop at an epoch boundary writes no mid-epoch snapshot")
        _, code = self.in_process(resume=True)
        self.assertEqual(code, EXIT_COMPLETE)


class NoSignalTests(CampaignCase):
    options = {"topics": TOPICS, "signal": "none"}

    def test_when_no_reranker_qualifies_bm25_stays_the_default_and_the_release_ships_no_model(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        selection = self.selection()
        self.assertEqual(selection["decision"]["default_mode"], "baseline")
        self.assertIn("BM25", selection["decision"]["reason"])
        release = read_json(self.output("export_release", "release/release-manifest.json"), "release")
        self.assertEqual((release["status"], release["default_mode"], release["model"]), ("evaluated", "baseline", None))
        report = read_json(self.output("score", "final-report.json"), "report")
        self.assertEqual(report["default_mode"], "baseline")
        self.assertEqual({c["primary"] for c in report["comparisons"]}, {False}, "no comparison is primary when BM25 is the default")
        self.assertEqual(len(report["systems"]), 3, "every locked system's final result is retained, whatever the outcome")
        reload = read_json(self.output("reload_verify", "reload-report.json"), "reload")
        self.assertEqual(reload["checks"]["model_parity_after_reload"]["status"], "not_applicable")


class PostrankTests(CampaignCase):
    options = {"topics": TOPICS, "protocol": "postrank_pool_v1"}

    def oracle(self):
        """A scripted stand-in for the reviewer: it reads the pool's private provenance and the initial qrels."""
        pool_dir = self.output("pool", "review")
        hidden = [json.loads(line) for line in (pool_dir / "hidden-provenance.jsonl").read_text().splitlines()][1:]
        qrels = [json.loads(line) for line in (self.root / "evaluation" / "qrels-test.jsonl").read_text().splitlines()]
        positives = {(q["query_id"], q["chunk_id"]) for q in qrels if q["relevance"] == 1}
        lengths = {}
        for line in (self.root / "bundle" / "prepared" / "corpus" / "chunks.jsonl").read_text().splitlines():
            record = json.loads(line)
            lengths[record["chunk_id"]] = len(record["text"])
        rows = []
        for item in hidden:
            value = 1 if (item["query_id"], item["chunk_id"]) in positives else 0
            rows.append({"item_id": item["item_id"], "role": "primary", "relevance": value, "status": "resolved",
                         "evidence_spans": [{"start": 0, "end": min(30, lengths[item["chunk_id"]])}] if value else [],
                         "reviewer": {"type": "agent", "id": "rehearsal-oracle-script", "revision": "script-1"}, "reviewed_at": "2026-10-05T12:00:00Z", "notes": None})
        return rows

    def test_the_protocol_waits_for_review_then_finishes_only_with_a_complete_reviewed_pool(self):
        first = self.cli()
        self.assertEqual(first.returncode, EXIT_AWAITING_REVIEW, first.stderr[-2000:])
        state = self.state()
        self.assertEqual((state["status"], state["stages"]["review_gate"]["status"], state["stages"]["seal_final"]["status"]),
                         ("awaiting_review", "awaiting_review", "pending"))
        request = read_json(self.output("review_gate", "review-request.json"), "request")
        self.assertEqual(request["protocol"], "postrank_pool_v1")
        self.assertGreater(request["missing_reviewed_records"], 0)
        self.assertEqual(list(self.stage_dir("score").glob("attempt-*")) if self.stage_dir("score").exists() else [], [], "no final report while waiting")
        self.assertEqual(self.cli("--resume").returncode, EXIT_AWAITING_REVIEW)
        rows = self.oracle()
        partial = self.root / "reviews-partial.jsonl"
        partial.write_text("".join(json.dumps(r) + "\n" for r in rows[:-3]))
        incomplete = self.cli("--resume", "--reviews", str(partial))
        self.assertNotEqual(incomplete.returncode, 0)
        self.assertIn("unresolved or missing", incomplete.stderr)
        self.assertNotEqual(self.statuses()["seal_final"], "succeeded")
        full = self.root / "reviews.jsonl"
        full.write_text("".join(json.dumps(r) + "\n" for r in rows))
        done = self.cli("--resume", "--reviews", str(full))
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        report = read_json(self.output("score", "final-report.json"), "report")
        self.assertEqual(report["provenance"]["evaluation_protocol"], "postrank_pool_v1")
        seal = read_json(self.output("seal_final", "test-seal-final.json"), "seal")
        self.assertEqual(seal["coverage"]["pool_depth"], 10)


if __name__ == "__main__":
    unittest.main()
