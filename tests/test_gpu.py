"""`scripts/gpu.py`, the rented-GPU entry point, exercised on CPU with small original fixtures.

The workflow test drives check -> prepare -> (review import) -> prepare -> diagnose -> run -> status on a local git source, a
WordPiece fixture model and a dataset package built by `gpu.py package`, under the `cpu-diagnostic` profile: every output is
diagnostic and says so. The other tests cover what must stop the script: missing prerequisites, incomplete review coverage, a
tokenizer that encodes differently, a changed frozen identity, a formal run without pretrained assets, receipts or a passed
diagnosis, and model downloads whose bytes do not match the hub's hashes.
"""

import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from helpers import REPO
from mltest import requires_ml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_fixture  # noqa: E402

GPU = gpu_fixture.load_gpu()


def gpu(*args):
    result = subprocess.run([sys.executable, REPO / "scripts" / "gpu.py", *map(str, args)], capture_output=True, text=True, timeout=3600,
                            env={**{k: v for k, v in os.environ.items() if not k.startswith("DEVSEARCH")}, "PYTHONPATH": str(REPO / "src")})
    return result.returncode, result.stdout, result.stderr


def review_import(root: Path, judgments: Path, reviewer: str):
    result = subprocess.run([sys.executable, "-m", "devsearch.review", "import", "--request", root / "evaluation" / "review" / "review-request.jsonl",
                             "--judgments", judgments, "--reviewer-id", reviewer, "--reviewer-type", "agent", "--revision", "fixture-oracle-v1",
                             "--role", "primary", "--output", root / "review" / "responses" / f"{reviewer}.jsonl"],
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=600)
    assert result.returncode == 0, result.stderr


@requires_ml
class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Path(tempfile.mkdtemp(prefix="devsearch-gpu-")).resolve()
        cls.world = gpu_fixture.World(cls.base)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, ignore_errors=True)

    def judge(self, root: Path, name: str) -> Path:
        path = root.parent / f"{root.name}-judgments-{name}.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in gpu_fixture.oracle_judgments(root, root / "review" / "batches" / name)))
        return path

    def test_the_whole_cpu_workflow_stops_at_each_prerequisite_and_then_completes(self):
        root = self.world.runtime()
        config = self.world.config(root)
        code, out, err = gpu("check", "--config", config)
        self.assertEqual(code, 0, out + err)
        code, _, err = gpu("run", "--config", config)
        self.assertEqual(code, 3)
        self.assertIn("PREREQUISITE prepare", err)

        code, out, err = gpu("prepare", "--config", config)
        self.assertEqual(code, 3, err)
        self.assertIn("PREREQUISITE review-bank", err)
        self.assertTrue(any((root / "review" / "batches" / "A").iterdir()) and any((root / "review" / "batches" / "B").iterdir()))
        self.assertFalse((root / "bundle" / "prepared" / "initial-seal.json").exists(), "nothing is sealed without the review bank")
        audit = json.loads((root / "data" / "membership-audit.json").read_text())
        self.assertTrue(audit["identical"], audit["differences"])
        frozen = json.loads((root / "state" / "frozen.json").read_text())
        self.assertEqual((frozen["profile"], frozen["model_commit"], frozen["dataset_sha256"]), ("cpu-diagnostic", "1" * 40, self.world.package_sha256))

        code, out, err = gpu("cmd", "--config", config, "--", "review", "progress", "--request", root / "evaluation" / "review" / "review-request.jsonl",
                             "--batches-dir", root / "review" / "batches" / "A", "--responses-dir", root / "review" / "responses",
                             "--reviewer-id", "fixture-oracle-a", "--pass", "A", "--limit", "2")
        self.assertEqual(code, 0, err)
        self.assertEqual(len(json.loads(out)["next"]["next"]), 2, "`gpu.py cmd` runs devsearch commands with this configuration's interpreter and root")
        review_import(root, self.judge(root, "A"), "fixture-oracle-a")
        code, _, err = gpu("prepare", "--config", config)
        self.assertEqual(code, 3, "one primary pass of two")
        self.assertIn("review coverage incomplete", err)
        self.assertFalse((root / "bundle" / "prepared" / "initial-seal.json").exists())
        review_import(root, self.judge(root, "B"), "fixture-oracle-b")
        code, out, err = gpu("prepare", "--config", config)
        self.assertEqual(code, 0, err)
        self.assertTrue((root / "bundle" / "bundle-manifest.json").is_file())
        self.assertTrue(json.loads((root / "state" / "bundle.json").read_text())["diagnostic"], "diagnostic fixture assets make a diagnostic bundle")

        code, _, err = gpu("run", "--config", config)
        self.assertEqual(code, 3)
        self.assertIn("PREREQUISITE diagnose", err)
        code, out, err = gpu("diagnose", "--config", config)
        self.assertEqual(code, 0, out + err)
        diagnose = json.loads((root / "state" / "diagnose.json").read_text())
        self.assertTrue(diagnose["passed"])
        self.assertEqual(diagnose["crash_exit"], 137)
        self.assertGreaterEqual(diagnose["train_full_attempts"], 2, "the injected crash was recovered by a resume")
        self.assertEqual(gpu("diagnose", "--config", config)[0], 0, "a passed diagnosis for the same identity is not repeated")

        code, out, err = gpu("run", "--config", config)
        self.assertEqual(code, 0, out + err)
        self.assertIn("diagnostic run", err)
        reports = list((root / "campaign" / "run").rglob("final-report.json"))
        self.assertTrue(reports)
        for path in reports:
            report = json.loads(path.read_text())
            self.assertTrue(report["provenance"]["diagnostic_rehearsal"], "a cpu-diagnostic campaign is stamped diagnostic")
            self.assertTrue(any("DIAGNOSTIC REHEARSAL" in note for note in report["limitations"]))
        self.assertTrue(any((root / "backup").iterdir()), "the backup was written")
        code, out, _ = gpu("status", "--config", config)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["next"].startswith("done"))
        self.assertEqual(gpu("run", "--config", config)[0], 0, "a finished campaign resumes to completion without redoing work")
        code, out, err = gpu("run", "--config", config, "--extend-hours", "1")
        self.assertEqual(code, 0, out + err)
        self.assertTrue((root / "campaign" / "plan" / "host-binding-1.json").is_file(), "the extension is a new host binding")

        for change in ({"sha256": "0" * 64}, {"kind": "pretrained"}):
            code, _, err = gpu("diagnose", "--config", self.world.config(root, **change))
            self.assertEqual(code, 1, change)
            self.assertIn("frozen", err)

    def test_a_tokenizer_that_encodes_differently_stops_before_any_data_is_built(self):
        root = self.world.runtime()
        model = self.base / "cased-model"
        shutil.copytree(self.world.model, model)
        (model / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": False}))
        config = self.world.config(root)
        config.write_text(config.read_text().replace(str(self.world.model), str(model)))
        code, _, err = gpu("prepare", "--config", config)
        self.assertEqual(code, 3, err)
        self.assertIn("PREREQUISITE protocol-revision", err)
        self.assertFalse((root / "bundle" / "prepared" / "split-manifest.json").exists())

    def test_a_package_that_does_not_match_its_pin_is_not_used(self):
        root = self.world.runtime()
        package = root / "inbox" / "dataset.tar.gz"
        package.write_bytes(package.read_bytes() + b"\0")
        code, out, err = gpu("prepare", "--config", self.world.config(root))
        self.assertEqual(code, 3)
        self.assertIn("PREREQUISITE dataset-package", out + err)
        self.assertFalse((root / "dataset").exists())


class GateCase(unittest.TestCase):
    """A runtime root configured for the nvidia-cuda profile, with only the record of a finished `prepare`."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="devsearch-gpu-gate-")).resolve()
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        root = self.base / "root"
        root.mkdir()
        config = self.base / "gpu.toml"
        config.write_text(f"""schema_version = 1
[runtime]
root = "{root}"
time_limit_hours = 1
[backend]
profile = "nvidia-cuda"
[environment]
mode = "current"
[model]
repo = "x/y"
revision = "{'2' * 40}"
license = "MIT"
source = "local"
local_dir = "{self.base}"
[dataset]
package = "p.tar.gz"
sha256 = "{'3' * 64}"
""")
        self.rt = GPU.Runtime(GPU.Config(config))
        self.rt.state.mkdir()
        self.rt.record("prepare")


class FormalRunGateTests(GateCase):
    """A formal (accelerator) run needs pretrained assets, this code's receipts and a passed diagnosis; each stops with its name."""

    def stop_name(self):
        with self.assertRaises(GPU.Stop) as caught:
            GPU.stage_run(self.rt)
        self.assertEqual(caught.exception.code, 3)
        return caught.exception.prerequisite

    def test_each_missing_piece_is_named(self):
        self.rt.record("bundle", diagnostic=True)
        self.assertEqual(self.stop_name(), "pretrained-assets")
        self.rt.record("bundle", diagnostic=False)
        self.assertEqual(self.stop_name(), "receipts")
        self.rt.record("receipts", code_key=self.rt.code_key(), directory=str(self.base))
        self.assertEqual(self.stop_name(), "diagnose")
        self.rt.record("diagnose", passed=True, identity={**GPU.identity(self.rt), "code_key": "older-code"})
        self.assertEqual(self.stop_name(), "diagnose", "a diagnosis of other code does not count")


class RunnerTests(GateCase):
    def test_a_step_over_its_time_limit_is_stopped_with_its_child_processes(self):
        spawn = ("import subprocess, sys, time\n"
                 "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
                 "print(child.pid, flush=True)\ntime.sleep(120)\n")
        with self.assertRaises(GPU.Stop) as caught:
            self.rt.run("sleeper", [sys.executable, "-c", spawn], timeout=3)
        self.assertEqual(caught.exception.code, 4)
        self.assertIn("time limit", str(caught.exception))
        log = next(self.rt.logs.glob("*-sleeper.log")).read_text()
        grandchild = int([line for line in log.splitlines() if line.strip().isdigit()][-1])
        for _ in range(50):
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            threading.Event().wait(0.1)
        else:
            self.fail("the grandchild process survived the time limit")


class FakeHub(http.server.BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):
        body = self.routes.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        if isinstance(body, int):
            self.send_response(body)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class DownloadTests(unittest.TestCase):
    """The pinned-model download against a local fake hub: commit resolution, LFS and git-blob verification, refusal on mismatch."""

    COMMIT = "ab" * 20
    REPO_ID = "fixture/model"

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="devsearch-gpu-dl-")).resolve()
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        patcher = mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.files = {"config.json": json.dumps({"model_type": "bert", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1,
                                                 "vocab_size": 7, "max_position_embeddings": 16}).encode(),
                      "vocab.txt": b"[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n", "model.safetensors": b"not-really-tensors" * 10}

    def serve(self, files, *, lie_about=None, status=None):
        siblings = []
        for name, data in files.items():
            entry = {"rfilename": name, "size": len(data), "blobId": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
            if name.endswith((".safetensors", ".bin")):
                entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            siblings.append(entry)
        api = f"/api/models/{self.REPO_ID}/revision"
        FakeHub.routes = {f"{api}/main": json.dumps({"sha": self.COMMIT}).encode(),
                          f"{api}/{self.COMMIT}?blobs=true": json.dumps({"sha": self.COMMIT, "siblings": siblings}).encode()}
        for name, data in files.items():
            FakeHub.routes[f"/{self.REPO_ID}/resolve/{self.COMMIT}/{name}"] = (data[:-1] + b"X") if name == lie_about else data
        if status:
            FakeHub.routes[f"{api}/main"] = status

    def runtime(self, revision=GPU.RESOLVE_ONCE):
        root = self.base / "root"
        root.mkdir(exist_ok=True)
        config = self.base / f"gpu-{revision[:6]}.toml"
        config.write_text(f"""schema_version = 1
[runtime]
root = "{root}"
time_limit_hours = 1
[backend]
profile = "cpu-diagnostic"
[environment]
mode = "current"
[model]
repo = "{self.REPO_ID}"
revision = "{revision}"
license = "MIT"
source = "huggingface"
endpoint = "http://127.0.0.1:{self.server.server_address[1]}"
[dataset]
package = "p.tar.gz"
sha256 = "{'3' * 64}"
""")
        rt = GPU.Runtime(GPU.Config(config))
        rt.state.mkdir(exist_ok=True)
        rt.logs.mkdir(exist_ok=True)
        return rt

    def test_a_verified_download_is_registered_with_its_provenance_and_the_commit_is_frozen(self):
        self.serve(self.files)
        rt = self.runtime()
        record = GPU.step_model(rt)
        self.assertEqual((record["commit"], record["pinned"], record["asset_kind"]), (self.COMMIT, True, "pretrained"))
        provenance = json.loads((rt.root / "assets" / "model-provenance.json").read_text())
        self.assertEqual(provenance["commit"], self.COMMIT)
        self.assertEqual(provenance["files"]["model.safetensors"]["sha256"], hashlib.sha256(self.files["model.safetensors"]).hexdigest())
        self.assertEqual(json.loads((rt.state / "frozen.json").read_text())["model_commit"], self.COMMIT)
        with self.assertRaises(GPU.Stop) as caught:
            GPU.check_frozen(self.runtime(revision="cd" * 20))
        self.assertEqual(caught.exception.code, 1, "another commit is never picked up silently")

    def test_bytes_that_do_not_match_the_hub_hash_are_refused_and_nothing_is_registered(self):
        for name in ("model.safetensors", "vocab.txt"):
            with self.subTest(name=name):
                shutil.rmtree(self.base / "root", ignore_errors=True)
                self.serve(self.files, lie_about=name)
                rt = self.runtime()
                with self.assertRaises(GPU.Stop) as caught:
                    GPU.step_model(rt)
                self.assertEqual(caught.exception.code, 1)
                self.assertIn("integrity check failed", str(caught.exception))
                self.assertFalse((rt.root / "assets" / "model").exists())
                self.assertEqual(list((rt.root / "assets" / "download").rglob("*.part")), [])

    def test_a_refused_credential_is_a_named_prerequisite(self):
        self.serve(self.files, status=401)
        with self.assertRaises(GPU.Stop) as caught:
            GPU.step_model(self.runtime())
        self.assertEqual((caught.exception.code, caught.exception.prerequisite), (3, "model-credentials"))

    @requires_ml
    def test_pickled_weights_are_converted_once_and_both_hashes_recorded(self):
        import io

        import torch
        buffer = io.BytesIO()
        torch.save({"embeddings.word_embeddings.weight": torch.ones(7, 8)}, buffer)
        files = {k: v for k, v in self.files.items() if k != "model.safetensors"}
        files["pytorch_model.bin"] = buffer.getvalue()
        self.serve(files)
        rt = self.runtime()
        GPU.step_model(rt)
        conversion = json.loads((rt.root / "assets" / "model-provenance.json").read_text())["conversion"]
        self.assertEqual(conversion["from_sha256"], hashlib.sha256(files["pytorch_model.bin"]).hexdigest())
        self.assertTrue((rt.root / "assets" / "model" / "model.safetensors").is_file())
        self.assertFalse((rt.root / "assets" / "model" / "pytorch_model.bin").exists(), "a pickled file is never part of the asset set")


@requires_ml
class FetchThenLocalTests(DownloadTests):
    """`gpu.py fetch-model` on a connected machine, then `source = "local"` on the server: hashes recorded by the fetch are re-checked."""

    def fetch(self, files, **overrides):
        import io
        import types
        self.serve(files)
        out = self.base / "fetched"
        shutil.rmtree(out, ignore_errors=True)
        args = types.SimpleNamespace(repo=self.REPO_ID, revision=self.COMMIT, output=out, endpoint=f"http://127.0.0.1:{self.server.server_address[1]}", token_env="HF_TOKEN")
        args.__dict__.update(overrides)
        with mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(GPU.stage_fetch_model(args), 0)
        return out

    def local_runtime(self, directory, commit):
        root = self.base / "server-root"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir()
        config = self.base / "gpu-local.toml"
        config.write_text(f"""schema_version = 1
[runtime]
root = "{root}"
time_limit_hours = 1
[backend]
profile = "cpu-diagnostic"
[environment]
mode = "current"
[model]
repo = "{self.REPO_ID}"
revision = "{commit}"
license = "MIT"
source = "local"
local_dir = "{directory}"
[dataset]
package = "p.tar.gz"
sha256 = "{'3' * 64}"
""")
        rt = GPU.Runtime(GPU.Config(config))
        rt.state.mkdir()
        rt.logs.mkdir()
        return rt

    def pickled(self):
        import io

        import torch
        buffer = io.BytesIO()
        torch.save({"embeddings.word_embeddings.weight": torch.ones(7, 8)}, buffer)
        files = {k: v for k, v in self.files.items() if k != "model.safetensors"}
        files["pytorch_model.bin"] = buffer.getvalue()
        return files

    def test_a_fetched_directory_is_verified_converted_and_registered_on_the_server(self):
        files = self.pickled()
        out = self.fetch(files)
        manifest = json.loads((out / "fetch-manifest.json").read_text())
        self.assertEqual((manifest["commit"], manifest["weights"]), (self.COMMIT, "pytorch_model.bin"))
        self.assertEqual(manifest["files"]["pytorch_model.bin"]["sha256_actual"], hashlib.sha256(files["pytorch_model.bin"]).hexdigest())
        rt = self.local_runtime(out, self.COMMIT)
        record = GPU.step_model(rt)
        self.assertEqual((record["pinned"], record["asset_kind"]), (True, "pretrained"))
        provenance = json.loads((rt.root / "assets" / "model-provenance.json").read_text())
        self.assertTrue(provenance["hub_hashes_verified_by_fetch_model"])
        self.assertEqual(provenance["conversion"]["from_sha256"], hashlib.sha256(files["pytorch_model.bin"]).hexdigest())
        self.assertFalse((rt.root / "assets" / "model" / "pytorch_model.bin").exists())

    def test_a_file_changed_after_the_fetch_or_a_wrong_commit_is_refused(self):
        out = self.fetch(self.pickled())
        (out / "vocab.txt").write_bytes(b"tampered\n")
        with self.assertRaises(GPU.Stop) as caught:
            GPU.step_model(self.local_runtime(out, self.COMMIT))
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("hash recorded", str(caught.exception))
        out = self.fetch(self.pickled())
        with self.assertRaises(GPU.Stop) as caught:
            GPU.step_model(self.local_runtime(out, "cd" * 20))
        self.assertEqual(caught.exception.code, 1)

    def test_fetch_refuses_a_branch_name_other_than_main_and_corrupt_downloads(self):
        self.serve(self.files, lie_about="vocab.txt")
        import types
        args = types.SimpleNamespace(repo=self.REPO_ID, revision="v1.0", output=self.base / "x", endpoint=f"http://127.0.0.1:{self.server.server_address[1]}", token_env="HF_TOKEN")
        with self.assertRaises(GPU.Stop) as caught:
            GPU.stage_fetch_model(args)
        self.assertEqual(caught.exception.code, 1)
        args.revision, args.output = self.COMMIT, self.base / "y"
        with self.assertRaises(GPU.Stop) as caught:
            GPU.stage_fetch_model(args)
        self.assertIn("integrity check failed", str(caught.exception))


class ConfigTests(unittest.TestCase):
    def test_the_shipped_configuration_is_valid_and_formal(self):
        config = GPU.Config(REPO / "configs" / "gpu.toml")
        self.assertEqual((config.profile, config.env_mode, config.model_kind, config.min_primary_reviews), ("nvidia-cuda", "lock", "pretrained", 2))
        self.assertTrue(config.lock.is_file())
        self.assertTrue(config.record_receipts)

    def test_unknown_keys_and_branch_revisions_are_refused(self):
        text = (REPO / "configs" / "gpu.toml").read_text()
        base = Path(tempfile.mkdtemp(prefix="devsearch-gpu-cfg-"))
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)
        for bad in (text.replace('revision = "resolve-main-once"', 'revision = "main"'), text + "\n[extra]\nx = 1\n",
                    text.replace("[receipts]", "[receipts]\nskip_tests = true")):
            path = base / "bad.toml"
            path.write_text(bad)
            with self.assertRaises(GPU.Stop) as caught:
                GPU.Config(path)
            self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
