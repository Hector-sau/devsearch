"""`devsearch.serve`: startup refusals, the lease, and a real loopback server process (docs/04 "Search API").

The server tests start the genuine command in a child process and talk to it over a real loopback socket. Everything
uses synthetic data on CPU; nothing here says anything about accelerators or ranking quality.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
from pathlib import Path

from mltest import HAVE_ML, requires_ml, requires_serve

if HAVE_ML:
    from campaign_fixture import completed_campaign
    from devsearch import serve
    from devsearch.execution.lock import Lease
    from opsworld import call

SRC = Path(__file__).resolve().parent.parent / "src"
SERVE_TOML = """schema_version = 1

[device]
profile = "{profile}"

[model]
batch_size = {batch}

[serve]
host = "{host}"
port = {port}
deadline_seconds = 5.0
http_grace_seconds = 1.0
{lease}
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def http(method, port, path, body=None, timeout=20):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


@requires_ml
@requires_serve
class ServeCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign = completed_campaign(topics=36)
        cls.holder = Path(tempfile.mkdtemp(prefix="devsearch-serve-", dir=cls.campaign.root / "runtime")).resolve()
        cls.addClassCleanup(shutil.rmtree, cls.holder, ignore_errors=True)

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(dir=self.holder))
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)

    def config(self, *, profile="cpu-diagnostic", host="127.0.0.1", port=None, lease=None, name="serve.toml", batch=None) -> Path:
        path = self.out / name
        batch = batch or self.campaign.plan["training"]["inference_batch"]            # the value the campaign exported and verified the model with
        path.write_text(SERVE_TOML.format(profile=profile, host=host, port=port or free_port(), batch=batch, lease="" if lease is None else f'lease_path = "{lease}"'))
        return path

    def check(self, config, release=None, *extra):
        return call(serve, "--config", config, "--manifest", release or self.campaign.release, "--runtime-root", self.campaign.root, "--check-only", *extra)

    def startup(self, out):
        return json.loads(out.strip().splitlines()[0])


class StartupCheckTests(ServeCase):
    def test_an_evaluated_release_passes_every_startup_check_without_binding_a_port(self):
        port = free_port()
        code, out, err = self.check(self.config(port=port))
        self.assertEqual((code, err), (0, ""), err)
        report = self.startup(out)
        self.assertEqual((report["event"], report["status"], report["default_mode"], report["device"]), ("startup_checks_passed", "evaluated", "rerank", "cpu"))
        self.assertEqual(report["checks"]["smoke_replay"], "passed")
        self.assertEqual(report["checks"]["rerank_self_check"], "passed")
        self.assertIsNone(report["accelerator_lease"])
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))          # --check-only left the port untouched

    def test_the_manifest_file_names_the_release_too(self):
        code, _, err = self.check(self.config(), self.campaign.release / "release-manifest.json")
        self.assertEqual((code, err), (0, ""), err)

    def test_a_baseline_release_is_checked_without_importing_the_ml_stack(self):
        from devsearch import export
        (self.out / "search.toml").write_bytes((self.campaign.release / "config" / "search.toml").read_bytes())
        export_config = self.out / "export.toml"
        export_config.write_text('schema_version = 1\n[device]\nprofile = "cpu-diagnostic"\n[model]\nbatch_size = 4\n[search]\npath = "search.toml"\n'
                                 '[smoke]\nqueries = ["How is the register kept?"]\n')
        code, _, err = call(export, "--config", export_config, "--mode", "baseline", "--index", self.campaign.index, "--output", self.out / "baseline",
                            "--runtime-root", self.campaign.root)
        self.assertEqual((code, err), (0, ""), err)
        script = ("import sys\nfrom devsearch import serve\n"
                  f"code = serve.main(['--config', {str(self.config())!r}, '--manifest', {str(self.out / 'baseline')!r}, '--runtime-root', {str(self.campaign.root)!r}, '--check-only'])\n"
                  "assert 'torch' not in sys.modules, 'a baseline release must not import torch'\n"
                  "assert 'transformers' not in sys.modules\n"
                  "sys.exit(code)\n")
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC)}, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        self.assertEqual(json.loads(result.stdout.splitlines()[0])["device"], None)

    def test_the_batch_size_must_match_the_one_the_release_was_verified_with(self):
        recorded = self.campaign.plan["training"]["inference_batch"]
        self.assertEqual(json.loads((self.campaign.release / "release-manifest.json").read_text())["provenance"]["inference_batch"], recorded)
        code, out, err = self.check(self.config(batch=recorded + 1))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("scoring identity", err)
        self.assertEqual(self.check(self.config(batch=recorded))[0], 0)

    def test_an_unknown_profile_is_refused_even_for_a_baseline_release(self):
        code, out, err = self.check(self.config(profile="quantum"))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("unknown device profile", err)

    def test_a_candidate_needs_allow_candidate(self):
        from devsearch import export
        index = self.campaign.index
        export_config = self.out / "export.toml"
        (self.out / "search.toml").write_bytes((self.campaign.release / "config" / "search.toml").read_bytes())
        export_config.write_text(
            f'schema_version = 1\n[device]\nprofile = "cpu-diagnostic"\n[model]\nbatch_size = 4\ndropout = {self.campaign.plan["training"]["shared"]["dropout"]}\n'
            f'[assets]\ndirectory = "{self.campaign.assets}"\n[policy]\nmax_pair_tokens = 128\nmax_query_tokens = 32\n[search]\npath = "search.toml"\n'
            f'[smoke]\nqueries = ["How is the register kept?"]\n')
        code, _, err = call(export, "--config", export_config, "--index", index, "--checkpoint", self.campaign.checkpoint, "--output", self.out / "candidate",
                            "--runtime-root", self.campaign.root)
        self.assertEqual((code, err), (0, ""), err)
        code, out, err = self.check(self.config(batch=4), self.out / "candidate")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("--allow-candidate", err)
        code, out, err = self.check(self.config(batch=4), self.out / "candidate", "--allow-candidate")
        self.assertEqual((code, err), (0, ""), err)
        self.assertEqual(self.startup(out)["status"], "candidate")

    def test_a_damaged_model_file_blocks_startup_and_is_never_served_as_a_baseline(self):
        damaged = self.out / "release"
        shutil.copytree(self.campaign.release, damaged)
        weights = next((damaged / "model").glob("*.safetensors"))
        weights.write_bytes(weights.read_bytes()[:-4] + b"\x00\x00\x00\x00")
        code, out, err = self.check(self.config(), damaged)
        self.assertEqual((code, out), (1, ""))
        self.assertIn("differ from the release manifest", err)
        self.assertNotIn("Traceback", err)

    def test_an_edited_final_report_a_missing_file_and_a_non_release_are_refused(self):
        edited = self.out / "edited"
        shutil.copytree(self.campaign.release, edited)
        report = edited / "evaluation" / "final-report.json"
        report.write_text(report.read_text().replace('"status": "complete"', '"status": "complete" '))
        self.assertEqual(self.check(self.config(), edited)[0], 1)
        (edited / "model").rename(self.out / "moved-model")
        code, _, err = self.check(self.config(), edited)
        self.assertEqual(code, 1)
        code, _, err = self.check(self.config(), self.out)
        self.assertEqual(code, 1)
        self.assertIn("neither a release directory", err)

    def test_a_release_outside_the_runtime_root_is_refused(self):
        outside = Path(tempfile.mkdtemp(prefix="devsearch-outside-"))
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        shutil.copytree(self.campaign.release, outside / "release")
        code, _, err = self.check(self.config(), outside / "release")
        self.assertEqual(code, 1)
        self.assertIn("release", err)

    def test_a_non_loopback_host_is_refused_before_anything_loads(self):
        code, out, err = self.check(self.config(host="0.0.0.0"))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("no authentication", err)

    def test_strict_configuration_failures_name_the_problem(self):
        text = self.config().read_text()
        cases = {
            "no serve section": text.replace("[serve]", "[other]"),
            "missing deadline": text.replace("deadline_seconds = 5.0\n", ""),
            "bad port": text.replace("port = ", "port = 70000 #"),
            "zero deadline": text.replace("deadline_seconds = 5.0", "deadline_seconds = 0"),
            "unknown key": text + "workers = 4\n",
        }
        for label, body in cases.items():
            with self.subTest(label):
                path = self.out / "bad.toml"
                path.write_text(body)
                code, out, err = self.check(path)
                self.assertEqual((code, out), (1, ""), err)
                self.assertIn("devsearch.serve: error", err)
                self.assertNotIn("Traceback", err)


class AcceleratorTests(ServeCase):
    """Prove refusal and lease discipline with simulated unavailability on any host."""

    def test_an_accelerator_profile_must_share_the_campaigns_lease(self):
        code, out, err = self.check(self.config(profile="apple-mps"))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("accelerator lease", err)

    def test_an_unavailable_accelerator_is_an_error_never_a_cpu_fallback_and_the_lease_is_released(self):
        lease = self.out / "accelerator.lease"
        with patch("torch.backends.mps.is_available", return_value=False):
            code, out, err = self.check(self.config(profile="apple-mps", lease=lease))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("mps", err.lower())
        self.assertFalse(lease.exists(), "a refused start must not leave the accelerator lease behind")

    def test_a_held_lease_refuses_a_second_owner(self):
        lease_path = self.out / "accelerator.lease"
        held = Lease(lease_path, purpose="campaign", plan_id="plan-x")
        held.acquire()
        self.addCleanup(held.release)
        code, out, err = self.check(self.config(profile="apple-mps", lease=lease_path))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("second owner is refused", err)
        self.assertTrue(lease_path.exists(), "the first owner keeps its lease")


class LiveServerTests(ServeCase):
    def start(self, config, release=None):
        child = {**os.environ, "PYTHONPATH": str(SRC)}
        process = subprocess.Popen([sys.executable, "-B", "-m", "devsearch.serve", "--config", str(config), "--manifest", str(release or self.campaign.release),
                                    "--runtime-root", str(self.campaign.root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=child)
        self.addCleanup(self.stop, process)
        return process

    def stop(self, process):
        if process.poll() is None:
            process.kill()
        process.communicate()

    def wait_ready(self, process, port, timeout=90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail(f"the server exited early ({process.returncode}): {process.stderr.read()[-1500:]}")
            try:
                if http("GET", port, "/health", timeout=2)[0] == 200:
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.2)
        self.fail("the server never answered /health")

    def test_a_real_process_serves_baseline_rerank_and_compare_over_loopback_and_stops_cleanly(self):
        port = free_port()
        process = self.start(self.config(port=port))
        self.wait_ready(process, port)
        status, health = http("GET", port, "/health")
        self.assertEqual((status, health["alive"], health["baseline_ready"], health["rerank_ready"]), (200, True, True, True))
        self.assertEqual(health["release"]["status"], "evaluated")
        self.assertEqual(http("GET", port, "/ready?mode=rerank")[0], 200)
        query = json.loads((self.campaign.release / "smoke" / "smoke.json").read_text())["queries"][0]["query"]
        status, baseline = http("POST", port, "/search", {"query": query, "mode": "baseline"})
        self.assertEqual((status, baseline["status"]), (200, "ok"))
        self.assertTrue(baseline["baseline_results"])
        status, rerank = http("POST", port, "/search", {"query": query, "mode": "rerank"})
        self.assertEqual((status, rerank["status"]), (200, "ok"))
        self.assertTrue(rerank["reranked_results"])
        self.assertTrue(all(isinstance(item["rerank_logit"], float) for item in rerank["reranked_results"]))
        status, compare = http("POST", port, "/search", {"query": query, "mode": "compare"})
        self.assertEqual((status, compare["status"]), (200, "ok"))
        status, invalid = http("POST", port, "/search", {"query": "", "mode": "baseline"})
        self.assertEqual((status, invalid["error"]["code"] if "error" in invalid else invalid.get("code")), (422, "invalid_request"))
        self.assertNotIn("Traceback", json.dumps(invalid))
        process.send_signal(signal.SIGTERM)
        out, err = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0, err[-1500:])
        events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        self.assertEqual([e["event"] for e in events], ["startup_checks_passed", "listening"])
        self.assertIn('"event": "stopped"', err)
        with self.assertRaises((urllib.error.URLError, ConnectionError, OSError)):
            http("GET", port, "/health", timeout=2)

    def test_a_port_already_in_use_fails_the_start_and_cleans_up(self):
        port = free_port()
        first = self.start(self.config(port=port))
        self.wait_ready(first, port)
        second = self.start(self.config(port=port, name="second.toml"))
        out, err = second.communicate(timeout=120)
        self.assertNotEqual(second.returncode, 0, out)
        self.assertEqual(http("GET", port, "/health")[0], 200, "the first server is unaffected")


if __name__ == "__main__":
    unittest.main()
