"""Doctor: preflight, calibration choices, resource caps, gates, worker isolation — with fake workers for failure paths."""

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from devsearch.artifacts.canonical import verify_manifest_hash
from mltest import HAVE_ML

if HAVE_ML:
    from devsearch.execution import doctor
    from devsearch.execution.doctor import capacity_estimate, run_doctor, run_worker
    from devsearch.execution.hostfile import load_host
    from devsearch.execution.rehearsal import build_rehearsal
    from devsearch.modeling.devices import load_profile
    from devsearch.data.prepare import EVALUATION_MANIFEST

TOKEN = {"backend": {"device": "cpu", "profile": "cpu-diagnostic"}}
_STATE = {}
_AUDIT = {"active": False, "prefix": "", "opened": []}


def _hook(event, args):                           # audit hooks cannot be removed, so this one is inert unless a test arms it
    if _AUDIT["active"] and event == "open" and isinstance(args[0], str) and args[0].startswith(_AUDIT["prefix"]):
        _AUDIT["opened"].append(args[0])


sys.addaudithook(_hook)


def setUpModule():
    if not HAVE_ML:
        return
    base = Path(tempfile.mkdtemp(prefix="devsearch-doctor-")).resolve()
    _STATE["base"] = base
    build_rehearsal(base)
    (base / "runtime" / "run").mkdir(exist_ok=True)


def tearDownModule():
    if _STATE:
        shutil.rmtree(_STATE["base"], ignore_errors=True)


def ok_result(kind, **extra):
    base = {"ok": True, "kind": kind, "device": "cpu", **TOKEN, "memory": {"device_peak_bytes": 0, "host_peak_rss_bytes": 1000}}
    base.update(extra)
    return base


def scripted(table):
    """A fake worker: `table(spec)` returns a result dict; every spec is recorded."""
    calls = []

    def worker(spec, **kwargs):
        calls.append(spec)
        return table(spec)
    worker.calls = calls
    return worker


def healthy(spec):
    kind = spec["kind"]
    if kind == "train_probe":
        return ok_result(kind, trainable_parameters=1000, microbatch=spec["microbatch"])
    if kind == "train_timing":
        return ok_result(kind, pairs_per_second=100.0 if spec["mode"] == "full" else 400.0, trainable_parameters=1000 if spec["mode"] == "full" else 10,
                         finite=True, measured=50)
    if kind == "infer_probe":
        return ok_result(kind)
    if kind == "infer_timing":
        return ok_result(kind, pairs_per_second=800.0, finite=True, measured=50)
    return ok_result(kind, checkpoint_save_seconds=0.1, checkpoint_load_seconds=0.1, export_seconds=0.2, reload_seconds=0.2, scratch_bytes=10,
                     finite=True, reload_parity={"passed": True})


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class DoctorCase(unittest.TestCase):
    def setUp(self):
        self.base = _STATE["base"]
        self.host_path = self.base / "host-example.toml"

    def host(self, **replace):
        text = self.host_path.read_text()
        for old, new in replace.items():
            self.assertIn(old, text)
            text = text.replace(old, new)
        path = self.base / "host-variant.toml"
        path.write_text(text)
        return load_host(path)

    def run_doctor(self, worker, host=None, bundle=None):
        return run_doctor(host or load_host(self.host_path), bundle or self.base / "bundle", worker=worker, now=lambda: "2026-10-05T00:00:00Z")

    def check(self, report, name):
        return next(c for c in report["checks"] if c["name"] == name)


class ReportTests(DoctorCase):
    def test_a_healthy_diagnostic_run_is_sealed_labelled_and_never_claims_a_backend(self):
        worker = scripted(healthy)
        report = self.run_doctor(worker)
        verify_manifest_hash(report, where="doctor report")
        self.assertEqual((report["kind"], report["profile"], report["diagnostic"]), ("doctor-report", "cpu-diagnostic", True))
        self.assertEqual(report["calibration"]["microbatch"], 4)
        self.assertEqual(report["calibration"]["inference_batch"], 4)
        gates = report["gates"]
        self.assertEqual(gates["data_ready"]["status"], "unverified")
        self.assertEqual(gates["data_ready"]["label"], "fixture")
        self.assertEqual(gates["backend_verified[cpu-diagnostic]"]["status"], "not_applicable")
        self.assertEqual(gates["implementation_verified"]["status"], "unverified")
        self.assertEqual(gates["portable_bundle_verified"]["status"], "unverified")
        self.assertEqual(self.check(report, "calibration")["status"], "passed")
        self.assertEqual(self.check(report, "model_assets_pinned")["status"], "unverified")
        self.assertEqual(report["estimates"]["label"], "estimate")
        self.assertEqual(report["estimates"]["train"]["full"]["pairs_per_second"], 100.0)
        self.assertEqual(report["estimates"]["capacity"]["label"], "estimate")
        self.assertGreater(report["estimates"]["capacity"]["bytes"], 0)

    def test_every_probe_uses_the_protocol_lengths_asset_directory_and_the_declared_order(self):
        worker = scripted(healthy)
        self.run_doctor(worker)
        kinds = [c["kind"] for c in worker.calls]
        self.assertEqual(kinds, ["train_probe", "train_timing", "train_timing", "infer_probe", "infer_timing", "io"])
        first = worker.calls[0]
        self.assertEqual((first["max_pair_tokens"], first["max_query_tokens"], first["effective_batch"], first["mode"]), (128, 32, 16, "full"))
        self.assertTrue(first["assets"].endswith("bundle/model-assets"))
        self.assertEqual([c["mode"] for c in worker.calls if c["kind"] == "train_timing"], ["frozen", "full"])
        self.assertTrue(all(c["warmup"] == 5 and c["measured"] == 50 for c in worker.calls if c["kind"] == "train_timing"))

    def test_the_doctor_never_opens_a_test_file(self):
        evaluation = str(self.base / "evaluation")
        _AUDIT.update(active=True, prefix=evaluation, opened=[])
        try:
            (self.base / "evaluation" / EVALUATION_MANIFEST).read_bytes()                  # the hook itself must see an open
            self.assertEqual(len(_AUDIT["opened"]), 1)
            _AUDIT["opened"].clear()
            self.run_doctor(scripted(healthy))
            self.assertEqual(_AUDIT["opened"], [], "the doctor must not open the evaluation root at all")
        finally:
            _AUDIT["active"] = False


class CalibrationTests(DoctorCase):
    def test_the_first_microbatch_that_fits_wins_and_failures_are_recorded(self):
        def table(spec):
            if spec["kind"] == "train_probe" and spec["microbatch"] == 4:
                return {"ok": False, "kind": "train_probe", "reason": "out_of_memory", "error": "MPS backend out of memory"}
            return healthy(spec)
        worker = scripted(table)
        report = self.run_doctor(worker)
        attempts = report["calibration"]["attempts"]
        self.assertEqual([(a["microbatch"], a["ok"]) for a in attempts], [(4, False), (2, True)])
        self.assertEqual(attempts[0]["reason"], "out_of_memory")
        self.assertEqual(report["calibration"]["microbatch"], 2)
        self.assertEqual({c["microbatch"] for c in worker.calls if c["kind"] == "train_timing"}, {2}, "timing runs use the selected microbatch")

    def test_a_probe_that_exceeds_the_memory_cap_is_rejected_even_though_it_ran(self):
        def table(spec):
            result = healthy(spec)
            if spec["kind"] == "train_probe" and spec["microbatch"] == 4:
                result["memory"]["host_peak_rss_bytes"] = 10**12
            return result
        host = self.host(**{"host_memory_cap_bytes = 4294967296": "host_memory_cap_bytes = 5000000000"})
        report = self.run_doctor(scripted(table), host=host)
        attempt = report["calibration"]["attempts"][0]
        self.assertFalse(attempt["ok"])
        self.assertIn("exceeds the cap", attempt["reason"])
        self.assertEqual(report["calibration"]["microbatch"], 2)

    def test_when_no_candidate_fits_calibration_fails_and_nothing_further_runs(self):
        worker = scripted(lambda spec: {"ok": False, "kind": spec["kind"], "reason": "out_of_memory", "error": "oom"})
        report = self.run_doctor(worker)
        self.assertIsNone(report["calibration"]["microbatch"])
        self.assertEqual(self.check(report, "calibration")["status"], "failed")
        self.assertEqual({c["kind"] for c in worker.calls}, {"train_probe"})
        self.assertEqual(len(worker.calls), 4, "exactly the four declared candidates, once each")

    def test_the_inference_batch_has_its_own_finite_probe_and_does_not_change_training(self):
        def table(spec):
            if spec["kind"] == "infer_probe" and spec["batch"] in (4, 2):
                return {"ok": False, "kind": "infer_probe", "reason": "out_of_memory", "error": "oom"}
            return healthy(spec)
        report = self.run_doctor(scripted(table))
        self.assertEqual(report["calibration"]["inference_batch"], 1)
        self.assertEqual(report["calibration"]["microbatch"], 4)
        self.assertEqual([a["batch"] for a in report["calibration"]["inference_attempts"]], [4, 2, 1])

    def test_a_timeout_that_cannot_be_cleaned_up_marks_the_host_unhealthy_and_starts_no_new_worker(self):
        worker = scripted(lambda spec: {"ok": False, "reason": "timeout", "worker_gone": False, "unhealthy": True})
        report = self.run_doctor(worker)
        self.assertTrue(report["unhealthy_host"])
        self.assertEqual(len(worker.calls), 1)
        self.assertIn("unhealthy", report["calibration"]["stopped"])
        self.assertEqual(self.check(report, "calibration")["status"], "failed")

    def test_an_exhausted_budget_stops_calibration_and_fails_it(self):
        host = self.host(**{"doctor_budget_seconds = 1800": "doctor_budget_seconds = 1"})

        def slow(spec):
            time.sleep(1.2)
            return healthy(spec)
        worker = scripted(slow)
        report = self.run_doctor(worker, host=host)
        self.assertEqual(len(worker.calls), 1)
        self.assertIn("budget", report["calibration"]["stopped"])
        self.assertEqual(self.check(report, "calibration")["status"], "failed")


class AcceleratorGateTests(DoctorCase):
    def accelerator_host(self, profile="apple-mps", cap="2000000000"):
        return self.host(**{'profile = "cpu-diagnostic"': f'profile = "{profile}"', "device_memory_cap_bytes = 0": f"device_memory_cap_bytes = {cap}"})

    def test_an_unavailable_device_fails_the_backend_gate_with_the_reason_and_runs_nothing_else(self):
        worker = scripted(lambda spec: {"ok": False, "kind": spec["kind"], "reason": "DeviceUnavailable", "error": "MPS is built but not available on this host",
                                        "device_unavailable": True})
        report = self.run_doctor(worker, host=self.accelerator_host())
        gate = report["gates"]["backend_verified[apple-mps]"]
        self.assertEqual(gate["status"], "failed")
        self.assertIn("not available", gate["detail"])
        self.assertEqual(len(worker.calls), 1)

    def test_diagnostic_assets_on_a_real_accelerator_stay_unverified(self):
        def table(spec):
            result = healthy(spec)
            result["backend"] = {"device": "mps", "profile": "apple-mps", "recommended_max_memory_bytes": 10**10}
            return result
        report = self.run_doctor(scripted(table), host=self.accelerator_host())
        gate = report["gates"]["backend_verified[apple-mps]"]
        self.assertEqual((gate["status"], gate["label"]), ("unverified", "fixture"))
        self.assertEqual(self.check(report, "device_memory_cap_vs_capacity")["status"], "passed")

    def test_a_missing_or_oversized_device_cap_is_rejected(self):
        missing = self.run_doctor(scripted(healthy), host=self.accelerator_host(cap="0"))
        self.assertEqual(self.check(missing, "device_memory_cap")["status"], "failed")

        def table(spec):
            result = healthy(spec)
            result["backend"] = {"device": "cuda:0", "profile": "nvidia-cuda", "total_memory_bytes": 10**9}
            return result
        oversized = self.run_doctor(scripted(table), host=self.accelerator_host("nvidia-cuda", cap=str(2 * 10**9)))
        self.assertEqual(self.check(oversized, "device_memory_cap_vs_capacity")["status"], "failed")

    def test_the_host_memory_cap_must_exist_and_not_exceed_physical_memory(self):
        report = self.run_doctor(scripted(healthy), host=self.host(**{"host_memory_cap_bytes = 4294967296": "host_memory_cap_bytes = 0"}))
        self.assertEqual(self.check(report, "host_memory_cap")["status"], "failed")
        report = self.run_doctor(scripted(healthy), host=self.host(**{"host_memory_cap_bytes = 4294967296": "host_memory_cap_bytes = 99999999999999999"}))
        self.assertEqual(self.check(report, "host_memory_cap")["status"], "failed")


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class BackendGateDecisionTests(unittest.TestCase):
    """The backend gate decision itself, with pinned real assets (which the fixture bundles cannot be): every measurement it relies on must be
    successful and finite, including the inference timing."""

    PINNED = [{"name": "model_assets_pinned", "status": "passed", "detail": "", "label": "measured"}]

    @staticmethod
    def calibration(**override):
        timing = {"ok": True, "finite": True, "pairs_per_second": 120.0, "measured": 50}
        io = {"ok": True, "finite": True, "reload_parity": {"passed": True}, "checkpoint_save_seconds": 0.1, "checkpoint_load_seconds": 0.1,
              "export_seconds": 0.2, "reload_seconds": 0.2}
        base = {"microbatch": 8, "inference_batch": 32, "stopped": None, "backend": {"device": "mps"}, "training": {"frozen": dict(timing), "full": dict(timing)},
                "inference": dict(timing), "io": io}
        base.update(override)
        return base

    def gate(self, calibration, profile="apple-mps"):
        from devsearch.execution.doctor import _gates
        return _gates(load_profile(profile), False, True, list(self.PINNED), calibration)[f"backend_verified[{profile}]"]

    def test_complete_successful_finite_measurements_pass(self):
        self.assertEqual(self.gate(self.calibration())["status"], "passed")
        self.assertEqual(self.gate(self.calibration(), "nvidia-cuda")["status"], "passed")

    def test_a_failed_or_missing_inference_timing_never_passes(self):
        for inference in ({"ok": False, "finite": True, "pairs_per_second": 120.0, "measured": 50}, None, {}):
            gate = self.gate(self.calibration(inference=inference))
            self.assertEqual(gate["status"], "failed", inference)
            self.assertIn("inference timing", gate["detail"])

    def test_non_finite_or_empty_measurements_never_pass(self):
        good = {"ok": True, "finite": True, "pairs_per_second": 120.0, "measured": 50}
        for bad in ({**good, "pairs_per_second": float("nan")}, {**good, "pairs_per_second": float("inf")}, {**good, "pairs_per_second": 0.0},
                    {**good, "pairs_per_second": None}, {**good, "finite": False}, {**good, "measured": 0}, {k: v for k, v in good.items() if k != "finite"},
                    {**good, "pairs_per_second": True}):
            self.assertEqual(self.gate(self.calibration(inference=bad))["status"], "failed", bad)
            self.assertEqual(self.gate(self.calibration(training={"full": bad}))["status"], "failed", bad)

    def test_a_failed_or_unfinite_save_reload_probe_never_passes(self):
        io = self.calibration()["io"]
        for bad in (None, {**io, "ok": False}, {**io, "finite": False}, {**io, "reload_parity": {"passed": False}}, {**io, "reload_parity": None},
                    {**io, "export_seconds": float("nan")}, {**io, "reload_seconds": -1.0}, {k: v for k, v in io.items() if k != "checkpoint_save_seconds"}):
            gate = self.gate(self.calibration(io=bad))
            self.assertEqual(gate["status"], "failed", bad)
            self.assertIn("save/reload", gate["detail"])

    def test_one_bad_training_recipe_among_good_ones_blocks_the_gate(self):
        calibration = self.calibration()
        calibration["training"]["full"] = {"ok": False, "finite": True, "pairs_per_second": 100.0, "measured": 50}
        gate = self.gate(calibration)
        self.assertEqual(gate["status"], "failed")
        self.assertIn("training timing", gate["detail"])
        self.assertEqual(self.gate(self.calibration(training={}))["status"], "failed", "no training measurement at all")


class DiagnosticDataTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_ML, "torch is not installed")
    def test_rehearsal_data_stays_diagnostic_even_when_the_assets_look_pretrained(self):
        """Fixture data with real-looking pinned assets must never be mistaken for a real bundle."""
        from devsearch.execution.bundle import BUNDLE_MANIFEST, assemble_bundle
        from devsearch.modeling.assets import ASSETS_MANIFEST, register_assets
        copy = Path(tempfile.mkdtemp(prefix="devsearch-diag-")).resolve()
        self.addCleanup(shutil.rmtree, copy, ignore_errors=True)
        shutil.copytree(_STATE["base"] / "bundle", copy / "bundle")
        assets = copy / "bundle" / "model-assets"
        (assets / ASSETS_MANIFEST).unlink()
        (copy / "bundle" / BUNDLE_MANIFEST).unlink()
        register_assets(assets, asset_kind="pretrained", source={"repo": "example/model", "revision": "0123456789abcdef0123456789abcdef01234567"})
        manifest = assemble_bundle(copy / "bundle")
        self.assertEqual((manifest["model_assets"]["asset_kind"], manifest["model_assets"]["pinned"]), ("pretrained", True))
        self.assertTrue(manifest["diagnostic"], "the corpus was written by the rehearsal generator")


class BundleFailureTests(DoctorCase):
    def test_a_tampered_bundle_fails_the_check_skips_calibration_and_fails_data_readiness(self):
        copy = self.base.parent / (self.base.name + "-tampered")
        shutil.copytree(self.base / "bundle", copy / "bundle", symlinks=True)
        self.addCleanup(shutil.rmtree, copy, ignore_errors=True)
        target = copy / "bundle" / "prepared" / "data" / "queries-train.jsonl"
        target.write_bytes(target.read_bytes() + b"\n")
        worker = scripted(healthy)
        report = self.run_doctor(worker, bundle=copy / "bundle")
        self.assertEqual(self.check(report, "bundle")["status"], "failed")
        self.assertEqual(worker.calls, [])
        self.assertEqual(report["gates"]["data_ready"]["status"], "failed")
        self.assertIsNone(report["bundle_manifest_sha256"])


class CapacityTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_ML, "ml libraries not installed")
    def test_the_capacity_estimate_follows_docs_12_arithmetic_and_flags_unknowns(self):
        from devsearch.evaluation.protocol import load_protocol
        from helpers import CONFIGS
        protocol = load_protocol(CONFIGS / "protocol.toml")                       # two runs
        estimate = capacity_estimate(protocol, 1000, 33_000_000, margin=1.0)
        self.assertEqual(estimate["checkpoints_bytes"], 2 * 4 * 12 * 33_000_000)  # docs/12 §5: 4 checkpoints x 12N per run
        self.assertEqual(estimate["exports_bytes"], 2 * 4 * 33_000_000)
        self.assertEqual(estimate["label"], "estimate")
        unknown = capacity_estimate(protocol, 1000, None)
        self.assertEqual((unknown["status"], unknown["bytes"]), ("unverified", None))


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class WorkerIsolationTests(unittest.TestCase):
    def run_stub(self, mode, timeout=30, grace=1):
        return run_worker({"mode": mode, "value": 7}, timeout=timeout, grace=grace, retries=0, module="probe_sleeper")

    def test_a_worker_result_is_returned(self):
        self.assertEqual(self.run_stub("ok"), {"ok": True, "echo": 7})

    def test_a_crashed_worker_is_reported_with_its_stderr_not_raised(self):
        result = self.run_stub("crash")
        self.assertEqual((result["ok"], result["reason"], result["returncode"]), (False, "crashed", 3))
        self.assertIn("simulated driver crash", result["stderr_tail"])

    def test_a_hung_worker_is_terminated_within_timeout_plus_grace(self):
        started = time.monotonic()
        result = self.run_stub("sleep", timeout=1, grace=2)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual((result["ok"], result["reason"], result["worker_gone"], result["unhealthy"]), (False, "timeout", True, False))

    def test_a_worker_that_ignores_termination_is_killed(self):
        result = self.run_stub("ignore_term", timeout=1, grace=1)
        self.assertEqual((result["reason"], result["worker_gone"]), ("timeout", True))

    def test_a_worker_that_cannot_be_confirmed_gone_marks_the_host_unhealthy(self):
        class Stuck:
            returncode = None

            def communicate(self, timeout=None):
                raise doctor.subprocess.TimeoutExpired("x", timeout)

            def terminate(self):
                pass

            def kill(self):
                pass
        with mock.patch.object(doctor.subprocess, "Popen", return_value=Stuck()):
            result = run_worker({"mode": "ok"}, timeout=0.1, grace=0.1, retries=0)
        self.assertEqual((result["reason"], result["worker_gone"], result["unhealthy"]), ("timeout", False, True))

    def test_transient_setup_io_is_retried_at_most_twice_then_reported(self):
        calls = []

        def flaky(spec, **kw):
            calls.append(1)
            raise OSError("temporary file system hiccup")
        with mock.patch.object(doctor, "_run_once", side_effect=flaky):
            result = run_worker({"mode": "ok"}, timeout=1, grace=1, retries=2)
        self.assertEqual((len(calls), result["ok"], result["reason"]), (3, False, "setup_io"))

    @unittest.skipUnless(HAVE_ML, "ml libraries not installed")
    def test_the_real_probe_runs_on_cpu_in_a_fresh_process(self):
        spec = {"kind": "infer_probe", "profile": "cpu-diagnostic", "assets": str(_STATE["base"] / "bundle" / "model-assets"), "max_pair_tokens": 64,
                "max_query_tokens": 16, "seed": 1, "head_seed": 2, "effective_batch": 16, "batch": 2}
        result = run_worker(spec, timeout=120, grace=5, retries=0)
        self.assertTrue(result["ok"], result)
        self.assertEqual((result["device"], result["finite"]), ("cpu", True))

    def test_an_unavailable_requested_accelerator_is_an_error_not_a_cpu_fallback(self):
        spec = {"kind": "infer_probe", "profile": "nvidia-cuda", "assets": str(_STATE["base"] / "bundle" / "model-assets"), "max_pair_tokens": 64,
                "max_query_tokens": 16, "seed": 1, "head_seed": 2, "effective_batch": 16, "batch": 2}
        # A fresh worker interpreter with no visible CUDA devices exercises the
        # actual unavailable-device error on GPU hosts too, without skipping.
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            result = run_worker(spec, timeout=120, grace=5, retries=0)
        self.assertFalse(result["ok"])
        self.assertTrue(result["device_unavailable"])
        self.assertIn("CUDA", result["error"])


if __name__ == "__main__":
    unittest.main()
