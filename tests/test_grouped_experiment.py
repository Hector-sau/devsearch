"""Offline campaign safety tests. Synthetic files/processes are never GPU evidence."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from devsearch.errors import ManifestError
from devsearch.training.job import load_train_job

SPEC = importlib.util.spec_from_file_location("grouped_experiment", Path(__file__).resolve().parents[1] / "scripts/grouped_experiment.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class GroupedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.campaign = self.root / "campaign-new"
        self.paths = {}
        for name in runner.INPUT_NAMES:
            path = self.root / name
            path.write_text("{}")
            self.paths[name] = str(path)
        self.args = argparse.Namespace(runtime_root=self.root, campaign_dir=self.campaign, defer_host_checkpoints=True, defer_host_assets=True,
                                       **{n: Path(p) for n, p in self.paths.items()})

    def plan(self):
        with mock.patch.object(runner, "validate_inputs", return_value=({"fixture": True}, None)):
            runner.make_plan(self.args)
        return self.campaign / "plan.json"

    def test_six_frozen_configs_match_every_field_except_seed_recipe_and_objective(self):
        plan = runner.load_plan(self.plan())
        configs = [load_train_job(self.campaign / "configs" / (j["name"] + ".toml")) for j in plan["jobs"]]
        self.assertEqual([(x.config.seed, x.config.loss_kind) for x in configs], list(runner.ORDER))
        reference = configs[0].config.to_dict()
        for config in configs:
            self.assertEqual(config.profile, "nvidia-cuda")
            self.assertEqual((config.max_pair_tokens, config.max_query_tokens), (256, 64))
            self.assertEqual(config.pairs_manifest_sha256, runner.R6_SHA)
            value = config.config.to_dict()
            for variable in ("seed", "recipe_id", "loss_kind"):
                value.pop(variable)
            expected = {k: v for k, v in reference.items() if k not in ("seed", "recipe_id", "loss_kind")}
            self.assertEqual(value, expected)
        self.assertEqual(reference["head_seed"], 1234)
        self.assertEqual(reference["microbatch"], 5)
        self.assertEqual(reference["effective_batch"], 20)
        self.assertEqual(reference["epochs"], 3)
        self.assertEqual(reference["learning_rate"], 5e-5)
        self.assertEqual(plan["policy"]["selection"], "fixed-final-epoch-3-v1")
        self.assertFalse(plan["policy"]["test_access"])
        self.assertFalse(plan["policy"]["promotion"])

    def test_plan_is_non_overwriting_and_does_not_resolve_cuda(self):
        with mock.patch.object(runner, "worker", side_effect=AssertionError("worker called")):
            self.plan()
            with self.assertRaisesRegex(ManifestError, "new"):
                runner.make_plan(self.args)
        self.assertFalse((self.campaign / "state.json").exists())

    def test_plan_integrity_and_config_drift_rejected(self):
        path = self.plan()
        plan = runner.read(path)
        first = self.campaign / "configs" / (plan["jobs"][0]["name"] + ".toml")
        first.write_text(first.read_text().replace("0.00005", "0.0001"))
        with self.assertRaisesRegex(ManifestError, "configuration drift"):
            runner.load_plan(path)
        first.write_text(plan["jobs"][0]["config"])
        plan["jobs"].reverse()
        runner.write(path, plan)
        with self.assertRaisesRegex(ManifestError, "integrity"):
            runner.load_plan(path)

    def test_resealed_plan_cannot_expand_grid_or_deadline(self):
        path = self.plan()
        for change in ("grid", "deadline"):
            plan = runner.read(path)
            if change == "grid":
                plan["jobs"].append(plan["jobs"][0])
            else:
                plan["policy"]["active_seconds"] += 1
            plan.pop("plan_sha256")
            plan["plan_sha256"] = runner.digest_of(plan)
            with mock.patch.object(runner, "read", return_value=plan), self.assertRaises(ManifestError):
                runner.load_plan(path)

    def test_config_symlink_refused(self):
        path = self.plan()
        plan = runner.read(path)
        config = self.campaign / "configs" / (plan["jobs"][0]["name"] + ".toml")
        other = self.root / "other.toml"
        config.rename(other)
        config.symlink_to(other)
        with self.assertRaisesRegex(ManifestError, "symlink"):
            runner.load_plan(path)

    def test_source_drift_fails_before_loading_inputs(self):
        path = self.plan()
        with mock.patch.object(runner, "source_identity", return_value={"changed": True}), mock.patch.object(runner, "validate_inputs") as loader:
            with self.assertRaisesRegex(ManifestError, "source/tests/spec/lock drift"):
                runner.check_plan(path)
            loader.assert_not_called()

    def test_actual_manifest_pin_checked_before_nested_loader(self):
        with mock.patch("devsearch.training.inputs.load_training_inputs") as loader:
            with self.assertRaisesRegex(ManifestError, "original prepared"):
                runner.validate_inputs(self.paths, self.root, self.campaign)
            loader.assert_not_called()

    def test_missing_authorization_refuses_before_any_input_or_device_access(self):
        with mock.patch.object(runner, "check_plan") as check:
            with self.assertRaisesRegex(ManifestError, "authorize-cuda"):
                runner.run_campaign(self.root / "missing.json")
            check.assert_not_called()

    def test_wrong_runtime_environment_fails_without_cuda(self):
        with mock.patch.object(runner.platform, "system", return_value="Darwin"):
            with self.assertRaisesRegex(ManifestError, "existing Linux"):
                runner.installed_lock()

    def history(self):
        rows = []
        for i in range(60):
            path = self.root / f"old-{i}.pt"
            path.write_bytes(b"historical " + str(i).encode())
            rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": runner.sha256_file(path), "verified": True})
        path = self.root / "historical.json"
        runner.write(path, {"count": 60, "checkpoints": rows})
        return path, rows

    def test_all_sixty_history_files_hash_verified_and_untouched(self):
        path, rows = self.history()
        runner.retained_checkpoints(path, self.root, self.campaign)
        self.assertEqual([runner.sha256_file(Path(r["path"])) for r in rows], [r["sha256"] for r in rows])
        Path(rows[-1]["path"]).write_bytes(b"x" * rows[-1]["bytes"])
        with self.assertRaisesRegex(ManifestError, "hash differs"):
            runner.retained_checkpoints(path, self.root, self.campaign)

    def test_missing_duplicate_and_output_overlapping_history_refused(self):
        path, rows = self.history()
        for changed in (rows[:-1], [rows[0]] * 60, [{**r, "path": str(self.campaign / f"{i}.pt")} for i, r in enumerate(rows)]):
            runner.write(path, {"count": 60, "checkpoints": changed})
            with self.assertRaises(ManifestError):
                runner.retained_checkpoints(path, self.root, self.campaign)

    def test_deferred_host_inventory_never_counts_as_actual_byte_verification(self):
        path, rows = self.history()
        Path(rows[0]["path"]).unlink()
        runner.retained_checkpoints(path, self.root, self.campaign, verify_bytes=False)
        with self.assertRaises(FileNotFoundError):
            runner.retained_checkpoints(path, self.root, self.campaign)

    def test_persistent_budget_and_clock_rollback(self):
        state = {"started_unix": 1000, "deadline_unix": 3700}
        self.assertEqual(runner.remaining_seconds(state, now=1100), 2600)
        self.assertEqual(runner.remaining_seconds(state, now=3800), 0)
        with self.assertRaisesRegex(ManifestError, "backward"):
            runner.remaining_seconds(state, now=999)
        with self.assertRaisesRegex(ManifestError, "altered"):
            runner.remaining_seconds({**state, "deadline_unix": 3701}, now=1000)

    def test_smoke_must_fit_both_arms_and_all_six_runs(self):
        arms = self.smoke_reports()
        self.assertAlmostEqual(runner.fit_estimate(arms, 2700), 2 * (.1 / 24 * 2322 + .1 * 18 + .1 * 24 + .1 * 6) + 60)
        with self.assertRaisesRegex(ManifestError, "both"):
            runner.fit_estimate(arms[:1], 2700)
        with self.assertRaisesRegex(ManifestError, "cannot fit"):
            runner.fit_estimate([{**x, "train_seconds": 20} for x in arms], 100)
        with self.assertRaisesRegex(ManifestError, "recovery"):
            runner.fit_estimate([{**x, "recovery_passed": False} for x in arms], 2700)
        with self.assertRaisesRegex(ManifestError, "recovery"):
            runner.fit_estimate([{**x, "updates": 8} for x in arms], 2700)

    def smoke_reports(self):
        return [{"arm": arm, "updates": 24, "train_seconds": .1, "validation_seconds": .1, "save_seconds": .1, "load_seconds": .1,
                 "recovery_passed": True, "recovery_protocol": runner.POLICY["smoke_recovery"]["protocol"]} for arm in runner.ARMS]

    def test_timeout_terminates_real_local_child_and_preserves_log(self):
        log = self.root / "child.log"
        code = runner.supervise([sys.executable, "-c", "import time; print('local timeout fixture', flush=True); time.sleep(30)"], log, 0.15, cleanup_seconds=2)
        self.assertEqual(code, 6)
        self.assertIn("local timeout fixture", log.read_text())

    def test_term_ignoring_child_is_force_killed_within_cleanup_bound(self):
        started = time.monotonic()
        code = runner.supervise([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"],
                                self.root / "ignore.log", 0.15, cleanup_seconds=2)
        self.assertEqual(code, 6)
        self.assertLess(time.monotonic() - started, 3)

    def test_term_ignoring_descendant_is_killed_after_leader_exits(self):
        pid_file = self.root / "descendant.pid"
        descendant = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM, signal.SIG_IGN); Path(" + repr(str(pid_file)) + ").write_text(str(os.getpid())); time.sleep(30)"
        leader = "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', " + repr(descendant) + "]); time.sleep(30)"
        pid = None
        try:
            result = runner.supervise([sys.executable, "-c", leader], self.root / "tree.log", .5, cleanup_seconds=2)
            self.assertEqual(result, 6)
            self.assertTrue(pid_file.exists(), "descendant fixture failed to start")
            pid = int(pid_file.read_text())
            state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
            self.assertTrue(not state or state.startswith("Z"), f"live descendant survived cleanup: {pid} {state}")
        finally:
            if pid is None and pid_file.exists():
                pid = int(pid_file.read_text())
            if pid is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)

    def test_cleanup_failure_is_explicit_when_group_termination_cannot_be_confirmed(self):
        child = SimpleNamespace(pid=123456789, wait=lambda **_: 0, poll=lambda: 0)
        with mock.patch.object(runner.os, "killpg"), mock.patch.object(runner, "live_group_members", return_value=[111]):
            with self.assertRaisesRegex(ManifestError, "not confirmed"):
                runner.terminate_group(child, .01)

    def test_profile_bytes_and_resolved_profile_are_frozen(self):
        original = runner.source_identity()
        profile_path = "src/devsearch/profiles/nvidia-cuda.toml"
        self.assertIn(profile_path, original["runner_tests_spec_lock"])
        self.assertFalse(original["resolved_cuda_profile"]["tf32"])
        original_hash = runner.sha256_file
        with mock.patch.object(runner, "sha256_file", side_effect=lambda path: "f" * 64 if Path(path) == runner.REPO / profile_path else original_hash(path)):
            self.assertNotEqual(runner.source_identity(), original)
        from dataclasses import replace
        from devsearch.modeling.devices import load_profile
        changed = replace(load_profile("nvidia-cuda"), tf32=True)
        with mock.patch("devsearch.modeling.devices.load_profile", return_value=changed):
            with self.assertRaisesRegex(ManifestError, "TF32 disabled"):
                runner.source_identity()

    def test_actual_backend_flags_must_match_fp32_tf32_off(self):
        from devsearch.modeling.devices import load_profile
        fake = SimpleNamespace(float32="float32", get_default_dtype=lambda: "float32",
                               backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                                        cudnn=SimpleNamespace(allow_tf32=False)))
        runner.validate_cuda_policy(load_profile("nvidia-cuda"), fake)
        for backend in (fake.backends.cuda.matmul, fake.backends.cudnn):
            backend.allow_tf32 = True
            with self.assertRaisesRegex(ManifestError, "actual PyTorch"):
                runner.validate_cuda_policy(load_profile("nvidia-cuda"), fake)
            backend.allow_tf32 = False
        fake.get_default_dtype = lambda: "float64"
        with self.assertRaisesRegex(ManifestError, "actual PyTorch"):
            runner.validate_cuda_policy(load_profile("nvidia-cuda"), fake)

    def test_parent_deadline_also_interrupts_non_child_work(self):
        with self.assertRaises(KeyboardInterrupt):
            with runner.deadline_guard(0.03):
                time.sleep(1)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_failed_child_exit_is_not_silently_retried(self):
        code = runner.supervise([sys.executable, "-c", "raise SystemExit(7)"], self.root / "failure.log", 1)
        self.assertEqual(code, 7)

    def controller(self, *, fail_at=None, existing=None):
        path = self.plan()
        plan = runner.read(path)
        (self.root / "state").mkdir()
        completed = set(existing or ())
        calls = []
        if existing:
            now = time.time() - 50
            runner.write(self.campaign / "state.json", {"plan_sha256": plan["plan_sha256"], "started_unix": now,
                         "deadline_unix": now + 2700, "status": "interrupted", "completed": list(completed)})
        def child(command, log, remaining):
            task = command[-1]
            calls.append((task, remaining))
            if task == fail_at:
                return 7
            if task == "smoke":
                arms = self.smoke_reports()
                runner.write(self.campaign / "smoke/completed.json", {"plan_sha256": plan["plan_sha256"], "arms": arms, "estimate_seconds": 100})
            else:
                completed.add(task)
            return 0
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "check_plan", return_value=(plan, None)))
            stack.enter_context(mock.patch.object(runner, "installed_lock", return_value={"test": "fixture"}))
            stack.enter_context(mock.patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 2**30)))
            stack.enter_context(mock.patch.object(runner, "retained_checkpoints"))
            stack.enter_context(mock.patch.object(runner, "completed_job", side_effect=lambda p, j: j["name"] in completed))
            stack.enter_context(mock.patch.object(runner, "supervise", side_effect=child))
            result = runner.run_campaign(path, authorized=True)
        return result, calls, plan

    def test_controller_executes_one_smoke_then_exact_serial_six_no_promotion(self):
        result, calls, plan = self.controller()
        self.assertEqual([x[0] for x in calls], ["smoke"] + [j["name"] for j in plan["jobs"]])
        self.assertTrue(all(0 < x[1] <= 2700 for x in calls))
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["test_accessed"])
        self.assertFalse(result["promoted"])

    def test_controller_stops_on_first_implementation_failure(self):
        failed = f"{runner.ARMS[0]}-seed42"
        result, calls, _ = self.controller(fail_at=failed)
        self.assertEqual(result["status"], "failed")
        self.assertEqual([x[0] for x in calls], ["smoke", failed])

    def test_resume_skips_completed_run_and_preserves_original_deadline(self):
        done = f"{runner.ARMS[0]}-seed42"
        result, calls, _ = self.controller(existing=[done])
        self.assertNotIn(done, [x[0] for x in calls])
        self.assertTrue(all(x[1] < 2651 for x in calls))
        self.assertEqual(result["deadline_unix"] - result["started_unix"], 2700)

    def test_internal_worker_cannot_be_launched_from_stale_state(self):
        path = self.plan()
        plan = runner.read(path)
        runner.write(self.campaign / "state.json", {"supervisor_pid": -1})
        with mock.patch.object(runner, "check_plan", return_value=(plan, None)), mock.patch.object(runner.subprocess, "run") as probe:
            with self.assertRaisesRegex(ManifestError, "live supervisor"):
                runner.worker(path, "smoke")
            probe.assert_not_called()

    def test_cli_without_authorization_exits_nonzero(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(runner.main(["run", "--plan", str(self.root / "absent")]), 1)
        self.assertIn("authorize-cuda", stderr.getvalue())

    def checkpoint_fixture(self, epoch):
        path = self.plan()
        plan = runner.read(path)
        plan["inputs"] = {"validation": "v", "tokenizer": "t", "model": "m", "grouped_data": "g"}
        job = plan["jobs"][0]
        directory = self.campaign / "runs" / job["name"]
        cp = directory / "checkpoints"
        cp.mkdir()
        identity = runner.checkpoint_identity(plan, job)
        for completed in range(epoch + 1):
            name = f"epoch-{completed:04d}"
            payload = cp / (name + ".pt")
            payload.write_bytes(b"synthetic checkpoint fixture " + name.encode())
            manifest = {"checkpoint_id": name, "file": name + ".pt", "resume_kind": "epoch_boundary", "completed_epoch": completed,
                        "next_epoch": completed, "global_step": completed * 129, "identity_sha256": runner.digest_of(identity),
                        "bytes": payload.stat().st_size, "sha256": runner.sha256_file(payload)}
            runner.write(cp / (name + ".json"), manifest)
        pointer = {"checkpoint_id": manifest["checkpoint_id"], "sha256": manifest["sha256"]}
        runner.write(cp / "latest.json", pointer)
        if epoch == 3:
            runner.write(cp / "best.json", pointer)
        return plan, job, directory, cp

    def test_resume_uses_latest_complete_epoch_and_keeps_diagnostic_snapshot(self):
        plan, job, directory, cp = self.checkpoint_fixture(2)
        (cp / "diagnostic-epoch0002-step000280.pt").write_bytes(b"partial never resumed")
        self.assertEqual(runner.checkpoint_progress(plan, job), 2)
        self.assertTrue((cp / "diagnostic-epoch0002-step000280.pt").exists())

    def test_checkpoint_source_identity_mismatch_and_early_best_fail_closed(self):
        plan, job, directory, cp = self.checkpoint_fixture(1)
        manifest_path = cp / "epoch-0001.json"
        manifest = runner.read(manifest_path)
        runner.write(manifest_path, {**manifest, "identity_sha256": "f" * 64})
        with self.assertRaisesRegex(ManifestError, "identity differs"):
            runner.checkpoint_progress(plan, job)
        runner.write(manifest_path, manifest)
        runner.write(cp / "best.json", runner.read(cp / "latest.json"))
        with self.assertRaisesRegex(ManifestError, "before epoch3"):
            runner.checkpoint_progress(plan, job)

    def test_epoch3_without_completion_receipt_is_recoverable_but_not_skipped(self):
        plan, job, directory, cp = self.checkpoint_fixture(3)
        self.assertEqual(runner.checkpoint_progress(plan, job), 3)
        self.assertFalse(runner.completed_job(plan, job))
        outputs = {}
        for name in ("training-summary.json", "validation-epoch1.json", "validation-epoch2.json", "validation-epoch3.json"):
            runner.write(directory / name, {"fixture": True})
            outputs[name] = runner.sha256_file(directory / name)
        runner.write(directory / "completed.json", {"plan_sha256": plan["plan_sha256"],
                     "checkpoint_sha256": runner.read(cp / "latest.json")["sha256"], "outputs": outputs})
        self.assertTrue(runner.completed_job(plan, job))
        (directory / "validation-epoch3.json").write_text("changed")
        with self.assertRaisesRegex(ManifestError, "output drift"):
            runner.completed_job(plan, job)

    def test_runtime_exposure_mutation_rejected(self):
        ids = [SimpleNamespace(record=SimpleNamespace(pair_id=f"p{i}")) for i in range(2575)]
        inputs = SimpleNamespace(grouped_data=SimpleNamespace(groups=[SimpleNamespace(records=ids)]))
        plan = {"inputs": {"schedules": {"42": [{"exposure_sha256": runner.digest_of([f"p{i}" for i in range(2575)])}]}}}
        result = SimpleNamespace(orders={0: list(range(2575))})
        runner.verify_run_orders(result, inputs, plan, {"seed": 42})
        result.orders[0][:2] = [1, 0]
        with self.assertRaisesRegex(ManifestError, "exposure hash"):
            runner.verify_run_orders(result, inputs, plan, {"seed": 42})
        result.orders[0] = [0] * 2575
        with self.assertRaisesRegex(ManifestError, "count/order"):
            runner.verify_run_orders(result, inputs, plan, {"seed": 42})

    def test_validation_scores_keep_every_candidate_and_fixed_fusion_without_labels(self):
        from devsearch.training.data import ValidationQuery, ValidationSet
        q = ValidationQuery("q1", "query", "g1", ("a", "b", "c"), ("A", "B", "C"), frozenset({"b"}))
        inputs = SimpleNamespace(validation=ValidationSet((q,), ()), validation_candidates=[
            {"query_id": "q1", "candidate_set_id": "c1", "candidates": [{"chunk_id": cid, "rank": i + 1} for i, cid in enumerate(q.candidate_ids)]}])
        path = self.root / "scores.json"
        with mock.patch("devsearch.modeling.scoring.score_pairs", return_value=[0.1, 0.4, 0.2]):
            metrics = runner.score_validation(None, None, None, inputs, None, path, lambda: False)
        data = runner.read(path)
        self.assertEqual([r["chunk_id"] for r in data["raw"][0]["ranking"]], ["b", "c", "a"])
        self.assertEqual([r["chunk_id"] for r in data["fused"][0]["ranking"]], ["b", "a", "c"])
        self.assertEqual(data["alpha"], 0.5)
        self.assertEqual(data["rank_constant"], 60)
        self.assertNotIn("positives", path.read_text())
        self.assertNotIn("qrels", path.read_text())
        self.assertEqual(metrics["group_macro"]["mrr@10"], 1)
        failed_path = self.root / "partial.json"
        with mock.patch("devsearch.modeling.scoring.score_pairs", side_effect=RuntimeError("scoring failed")), self.assertRaises(RuntimeError):
            runner.score_validation(None, None, None, inputs, None, failed_path, lambda: False)
        self.assertFalse(failed_path.exists())

    def recovery_fixture(self):
        from test_training_grouped import GroupedTrainingTests
        case = GroupedTrainingTests(methodName="test_training_group_counts_and_exposures")
        self.addCleanup(case.doCleanups)
        case.setUp()
        return case

    def run_recovery_fixture(self, case, directory, arm):
        import torch
        return runner.recovery_smoke(config=case.config(arm), data=case.data, validation=case.w.validation,
                                     tokenizer=case.w.tokenizer, policy=case.w.policy, device=torch.device("cpu"),
                                     fresh=lambda seed: case.model(), directory=self.root / directory,
                                     identities=case.w.identities(), stopping=lambda: False, deadline=time.monotonic() + 30)

    def test_both_recovery_smokes_use_production_save_reload_and_parity(self):
        from devsearch.training import loop
        case = self.recovery_fixture()
        for index, arm in enumerate(runner.ARMS):
            with mock.patch.object(loop, "load_checkpoint", wraps=loop.load_checkpoint) as loader:
                report, _ = self.run_recovery_fixture(case, f"recovery-{index}", arm)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(loader.call_args.args[1], "epoch-0001")
            self.assertTrue(loader.call_args.kwargs["require_resumable"])
            self.assertTrue(report["recovery_passed"])
            self.assertFalse(report["formal"])
            self.assertEqual((report["resumed_from_epoch"], report["resumed_from_step"], report["final_step"]), (1, 2, 6))
            self.assertEqual(report["parity"], ["model", "optimizer", "scheduler", "rng", "steps", "learning_rate_trace"])
            for trajectory in ("continuous", "resumed"):
                for epoch in range(4):
                    self.assertTrue((self.root / f"recovery-{index}" / trajectory / "checkpoints" / f"epoch-{epoch:04d}.pt").is_file())

    def test_recovery_smoke_refuses_missing_rng_restore(self):
        case = self.recovery_fixture()
        with mock.patch("devsearch.training.loop._restore_rng", return_value=None):
            with self.assertRaisesRegex(ManifestError, "recovery.*mismatch"):
                self.run_recovery_fixture(case, "missing-rng", runner.ARMS[0])
        self.assertFalse((self.root / "missing-rng/recovery.json").exists())

    def test_recovery_smoke_refuses_missing_optimizer_restore(self):
        import torch
        case = self.recovery_fixture()
        with mock.patch.object(torch.optim.AdamW, "load_state_dict", return_value=None):
            with self.assertRaisesRegex(ManifestError, "recovery.*(mismatch|differ)"):
                self.run_recovery_fixture(case, "missing-optimizer", runner.ARMS[1])
        self.assertFalse((self.root / "missing-optimizer/recovery.json").exists())


if __name__ == "__main__":
    unittest.main()
