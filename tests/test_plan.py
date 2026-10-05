"""Execution plan: refusals, immutability, stage graph, estimates; acceptance gates and receipts."""

import copy
import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch.artifacts.canonical import seal_manifest
from devsearch.errors import ManifestError, PrerequisiteError
from devsearch.evaluation.protocol import load_protocol
from devsearch.execution.records import read_json, write_json
from helpers import CONFIGS
from mltest import HAVE_ML

if HAVE_ML:
    from devsearch.artifacts.identity import code_identity
    from devsearch.execution import acceptance, plan as plan_module
    from devsearch.execution.acceptance import accelerator_run_ready, evaluate_gates, make_receipt
    from devsearch.execution.doctor import run_doctor
    from devsearch.execution.hostfile import binding_for, load_host
    from devsearch.execution.plan import build_plan, load_plan, stage_graph, write_plan
    from rehearsal_fixture import rehearsal
    from test_doctor import healthy, scripted


def doctor_report(base, **host_changes):
    host = load_host(base / "host-example.toml")
    return host, run_doctor(host, base / "bundle", worker=scripted(healthy), now=lambda: "2026-10-05T00:00:00Z")


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class PlanCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = rehearsal()
        cls.host, cls.report = doctor_report(cls.base)

    def build(self, report=None, **kw):
        return build_plan(self.host, self.base / "bundle", report or self.report, now="2026-10-05T00:00:00Z", **kw)


class BuildTests(PlanCase):
    def test_a_rehearsal_plan_binds_hashes_policies_and_calibrated_choices_but_no_host_paths(self):
        plan, binding = self.build()
        self.assertEqual((plan["kind"], plan["rehearsal"], plan["profile"]["name"]), ("execution-plan", True, "cpu-diagnostic"))
        self.assertEqual(plan["training"]["microbatch"], 4)
        self.assertEqual((plan["training"]["effective_batch"], plan["training"]["accumulation_steps"], plan["training"]["precision"]), (16, 4, "fp32"))
        self.assertEqual(plan["training"]["inference_batch"], 4)
        self.assertEqual(plan["training"]["passage_budget"], 128 - 32 - 3)
        self.assertEqual([m["stage_id"] for m in plan["matrix"]], ["train:frozen:42", "train:full:42"])
        self.assertEqual(plan["code"]["identity"], code_identity())
        self.assertEqual(plan["host_binding"]["sha256"], binding["manifest_sha256"])
        self.assertEqual(plan["bundle"]["protocol_id"], load_protocol(self.base / "bundle" / "protocol.toml").protocol_id)
        self.assertNotIn(str(self.base), json.dumps(plan), "a plan identifies code and data independently of host paths")
        self.assertEqual(plan["budget"]["estimates"]["label"], "estimate")
        self.assertEqual(plan["data"]["evaluation_protocol"], "prejudged_fixed_candidates_v1")
        self.assertGreater(plan["data"]["counts"]["train"]["queries"], plan["data"]["counts"]["test"]["queries"])

    def test_the_plan_contains_no_future_checkpoint_selection_or_report_hashes(self):
        plan, _ = self.build()
        text = json.dumps(plan)
        for forbidden in ("checkpoint_sha256", "final_report_id", "selection_id", "qrels_v2", "bank_manifest"):
            self.assertNotIn(forbidden, text)

    def test_the_plan_is_deterministic_for_identical_inputs(self):
        self.assertEqual(self.build()[0]["manifest_sha256"], self.build()[0]["manifest_sha256"])

    def test_every_estimate_is_labelled_and_unmeasured_stages_are_unverified_not_zero(self):
        plan, _ = self.build()
        estimates = plan["budget"]["estimates"]
        self.assertEqual(estimates["label"], "estimate")
        self.assertGreater(estimates["total_seconds"], 0)
        self.assertGreater(estimates["stage_seconds"]["train:full:42"], estimates["stage_seconds"]["export:full:42"])
        broken = copy.deepcopy(dict(self.report))
        broken["estimates"]["inference_pairs_per_second"] = None
        broken = seal_manifest({k: v for k, v in broken.items() if k != "manifest_sha256"})
        plan2, _ = self.build(report=broken)
        self.assertIsNone(plan2["budget"]["estimates"]["stage_seconds"]["latency"])
        self.assertIn("latency", plan2["budget"]["estimates"]["unverified_stages"])


class RefusalTests(PlanCase):
    def reseal(self, **changes):
        report = {k: v for k, v in copy.deepcopy(dict(self.report)).items() if k != "manifest_sha256"}
        for key, value in changes.items():
            report[key] = value
        return seal_manifest(report)

    def test_an_unsealed_or_edited_doctor_report_is_refused(self):
        edited = dict(self.report, profile="apple-mps")
        with self.assertRaises(ManifestError):
            self.build(report=edited)

    def test_a_report_for_another_profile_host_bundle_or_code_is_refused(self):
        with self.assertRaisesRegex(ManifestError, "device profile"):
            self.build(report=self.reseal(profile="nvidia-cuda"))
        with self.assertRaisesRegex(ManifestError, "different host file"):
            self.build(report=self.reseal(host_binding_sha256="0" * 64))
        with self.assertRaisesRegex(ManifestError, "different bundle"):
            self.build(report=self.reseal(bundle_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ManifestError, "code changed"):
            self.build(report=self.reseal(code={**self.report["code"], "source_snapshot_sha256": "0" * 64}))

    def test_failed_checks_an_unhealthy_host_and_missing_calibration_are_refused(self):
        failing = [dict(c) for c in self.report["checks"]]
        failing[0]["status"] = "failed"
        failing[0]["detail"] = "the volume is not mounted"
        with self.assertRaisesRegex(PrerequisiteError, f"{failing[0]['name']}: the volume is not mounted"):
            self.build(report=self.reseal(checks=failing))
        with self.assertRaisesRegex(PrerequisiteError, "the host is unhealthy"):
            self.build(report=self.reseal(unhealthy_host=True))
        uncalibrated = dict(self.report["calibration"], microbatch=None)
        with self.assertRaisesRegex(PrerequisiteError, "not calibrated"):
            self.build(report=self.reseal(calibration=uncalibrated))

    def test_a_window_too_short_for_the_known_work_is_refused(self):
        with self.assertRaisesRegex(PrerequisiteError, "insufficient time"):
            build_plan(self.host, self.base / "bundle", self.report, now="2099-01-01T00:01:00Z")

    def test_a_formal_plan_needs_every_gate_passed_and_diagnostic_bundles_can_never_make_one(self):
        verified = plan_module.verify_bundle(self.base / "bundle")
        manifest = dict(verified["manifest"], diagnostic=False)
        with mock.patch.object(plan_module, "verify_bundle", return_value={**verified, "manifest": manifest}):
            with self.assertRaisesRegex(PrerequisiteError, "every gate passed"):
                self.build()
        # and with the real (diagnostic) bundle the plan is explicitly a rehearsal
        self.assertTrue(self.build()[0]["rehearsal"])


class WriteTests(PlanCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp(prefix="devsearch-plan-")).resolve()
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def test_write_load_round_trip_with_the_binding_beside_it(self):
        plan, binding = self.build()
        write_plan(self.out / "execution-plan.json", plan, binding)
        loaded, sha = load_plan(self.out / "execution-plan.json")
        self.assertEqual(sha, plan["manifest_sha256"])
        self.assertEqual(read_json(self.out / "host-binding-0.json", "b")["manifest_sha256"], binding["manifest_sha256"])
        self.assertEqual(loaded["stages"], plan["stages"])
        write_plan(self.out / "execution-plan.json", plan, binding)                    # identical rewrite is a no-op

    def test_a_different_plan_cannot_replace_an_existing_one_and_a_missing_directory_is_never_created(self):
        plan, binding = self.build()
        write_plan(self.out / "execution-plan.json", plan, binding)
        other = seal_manifest({k: v for k, v in dict(plan, rehearsal=False).items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ManifestError, "immutable"):
            write_plan(self.out / "execution-plan.json", other, binding)
        with self.assertRaisesRegex(ManifestError, "never creates"):
            write_plan(self.out / "missing" / "execution-plan.json", plan, binding)

    def test_strict_loading_rejects_unknown_fields_and_edits(self):
        plan, binding = self.build()
        write_plan(self.out / "execution-plan.json", plan, binding)
        data = json.loads((self.out / "execution-plan.json").read_text())
        for mutate in (lambda d: d.update(extra=1), lambda d: d["training"].update(microbatch=8), lambda d: d.update(stages=[])):
            broken = json.loads(json.dumps(data))
            mutate(broken)
            path = self.out / "broken.json"
            path.write_text(json.dumps(broken))
            with self.assertRaises(ManifestError):
                load_plan(path)


class StageGraphTests(unittest.TestCase):
    def protocol(self, **kw):
        return dataclasses.replace(load_protocol(CONFIGS / "protocol.toml"), **kw)

    @unittest.skipUnless(HAVE_ML, "ml libraries not installed")
    def test_the_graph_is_ordered_and_covers_every_documented_stage(self):
        graph = stage_graph(self.protocol(), "prejudged_fixed_candidates_v1")
        ids = [s["stage_id"] for s in graph]
        self.assertEqual(len(ids), len(set(ids)))
        for index, stage in enumerate(graph):
            self.assertTrue(set(stage["needs"]) <= set(ids[:index]), stage)
        self.assertEqual(ids[:3], ["preflight", "data_check", "baseline_validation"])
        self.assertEqual(ids[-5:], ["score", "export_release", "reload_verify", "backup", "summary"])
        by_id = {s["stage_id"]: s["needs"] for s in graph}
        self.assertEqual(by_id["select"], ["baseline_validation", "latency", "train:frozen:42", "train:full:42"])
        self.assertEqual(by_id["latency"], ["export:frozen:42", "export:full:42", "baseline_validation"])
        self.assertEqual(by_id["rank_test"], ["select", "export:frozen:42", "export:full:42"])
        self.assertNotIn("review_gate", ids)

    @unittest.skipUnless(HAVE_ML, "ml libraries not installed")
    def test_the_postrank_protocol_adds_pool_and_a_review_gate_between_ranking_and_sealing(self):
        ids = [s["stage_id"] for s in stage_graph(self.protocol(), "postrank_pool_v1")]
        self.assertLess(ids.index("rank_test"), ids.index("pool"))
        self.assertLess(ids.index("pool"), ids.index("review_gate"))
        self.assertLess(ids.index("review_gate"), ids.index("seal_final"))
        self.assertLess(ids.index("seal_final"), ids.index("score"))

    @unittest.skipUnless(HAVE_ML, "ml libraries not installed")
    def test_a_seed_extension_expands_the_matrix_but_latency_uses_only_the_deployment_seed(self):
        graph = stage_graph(self.protocol(seeds=(42, 43, 44)), "prejudged_fixed_candidates_v1")
        ids = [s["stage_id"] for s in graph]
        self.assertEqual(len([i for i in ids if i.startswith("train:")]), 6)
        self.assertEqual({n for s in graph if s["stage_id"] == "latency" for n in s["needs"]}, {"export:frozen:42", "export:full:42", "baseline_validation"})


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class GateTests(PlanCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-rcpt-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.code = code_identity()

    def receipt(self, gate, *, snapshot=None, exit_status=0, summary=None, name="r"):
        receipt = make_receipt(gate, snapshot or self.code["source_snapshot_sha256"], [{"argv": ["python", "-m", "unittest"], "exit_status": exit_status, "stdout_sha256": "0" * 64}],
                               summary or {"tests_run": 500, "failures": 0, "errors": 0, "skipped": 0}, {"python": "3.11"}, "2026-10-05T00:00:00Z")
        write_json(self.dir / f"{name}.json", receipt)
        return receipt

    def gates(self):
        return evaluate_gates(self.report, self.dir, self.code)

    def test_without_receipts_both_evidence_gates_stay_unverified_not_passed(self):
        gates = self.gates()
        self.assertEqual(gates["implementation_verified"]["status"], "unverified")
        self.assertEqual(gates["portable_bundle_verified"]["status"], "unverified")

    def test_a_matching_clean_receipt_passes_and_records_its_evidence_hash(self):
        receipt = self.receipt("implementation_verified")
        gate = self.gates()["implementation_verified"]
        self.assertEqual((gate["status"], gate["evidence"]), ("passed", [receipt["manifest_sha256"]]))

    def test_receipts_for_other_code_failing_commands_failures_and_skips_do_not_pass(self):
        self.receipt("implementation_verified", snapshot="0" * 64)
        self.assertEqual(self.gates()["implementation_verified"]["status"], "unverified")
        self.receipt("implementation_verified", exit_status=1, name="b")
        self.assertEqual(self.gates()["implementation_verified"]["status"], "failed")
        (self.dir / "b.json").unlink()
        self.receipt("implementation_verified", summary={"tests_run": 500, "failures": 1, "errors": 0, "skipped": 0}, name="c")
        self.assertEqual(self.gates()["implementation_verified"]["status"], "failed")
        (self.dir / "c.json").unlink()
        self.receipt("implementation_verified", summary={"tests_run": 500, "failures": 0, "errors": 0, "skipped": 7}, name="d")
        gate = self.gates()["implementation_verified"]
        self.assertEqual(gate["status"], "failed")
        self.assertIn("a skip is not a pass", gate["detail"])

    def test_a_receipt_that_records_no_command_fails_the_gate_instead_of_crashing(self):
        receipt = make_receipt("implementation_verified", self.code["source_snapshot_sha256"], [], {"tests_run": 500, "failures": 0, "errors": 0, "skipped": 0},
                               {"python": "3.11"}, "2026-10-05T00:00:00Z")
        write_json(self.dir / "empty.json", receipt)
        gate = self.gates()["implementation_verified"]
        self.assertEqual(gate["status"], "failed")
        self.assertIn("records no command", gate["detail"])

    def test_the_portable_gate_needs_an_offline_complete_run_outside_the_checkout(self):
        self.receipt("portable_bundle_verified", summary={"outside_checkout": True, "rehearsal_complete": True, "offline": False})
        self.assertEqual(self.gates()["portable_bundle_verified"]["status"], "failed")
        self.receipt("portable_bundle_verified", summary={"outside_checkout": True, "rehearsal_complete": True, "offline": True}, name="ok")
        self.assertEqual(self.gates()["portable_bundle_verified"]["status"], "passed")

    def test_an_edited_receipt_is_rejected(self):
        self.receipt("implementation_verified")
        path = self.dir / "r.json"
        path.write_text(path.read_text().replace('"tests_run": 500', '"tests_run": 5000'))
        with self.assertRaises(ManifestError):
            self.gates()

    def test_a_diagnostic_run_can_never_be_accelerator_ready_even_if_every_gate_passes(self):
        gates = {n: {"status": "passed"} for n in ("implementation_verified", "data_ready", "portable_bundle_verified", "backend_verified[apple-mps]")}
        self.assertTrue(accelerator_run_ready(gates, "apple-mps", diagnostic=False))
        self.assertFalse(accelerator_run_ready(gates, "apple-mps", diagnostic=True))
        gates["backend_verified[apple-mps]"] = {"status": "unverified"}
        self.assertFalse(accelerator_run_ready(gates, "apple-mps", diagnostic=False))

    def test_the_acceptance_report_labels_a_rehearsal_as_fixture_and_never_claims_readiness(self):
        plan, _ = self.build()
        report = acceptance.build_report(plan=plan, gates=self.gates(), profile="cpu-diagnostic", diagnostic=True, stages={}, commands=[], artifacts={}, created_at="t")
        self.assertEqual(report["evidence_label"], "fixture")
        self.assertIs(report["accelerator_run_ready[cpu-diagnostic]"], False)
        self.assertIn("pipeline mechanics only", report["statement"])
        self.assertEqual(binding_for(self.host)["manifest_sha256"], plan["host_binding"]["sha256"])


if __name__ == "__main__":
    unittest.main()
