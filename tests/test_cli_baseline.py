"""`index`, `retrieve` and `validate_manifest` as real commands, run outside the checkout."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch import retrieve as retrieve_module

from helpers import CONFIGS, SRC, RootTestCase, run_module

INDEX_CONFIG = CONFIGS / "index.toml"
SEARCH_CONFIG = CONFIGS / "search.toml"


class CliCase(RootTestCase):
    def setUp(self):
        super().setUp()
        self.cwd = Path(tempfile.mkdtemp(prefix="devsearch-cwd-")).resolve()
        self.addCleanup(shutil.rmtree, self.cwd, ignore_errors=True)
        self.root_arg = ("--runtime-root", str(self.root))

    def index(self, *extra, config=INDEX_CONFIG, corpus=None, output=None):
        return run_module(
            "index", "--config", str(config), "--corpus", str(corpus or self.corpus_manifest),
            "--output", str(output or self.index_manifest), *extra, cwd=self.cwd,
        )

    def retrieve(self, query, *extra, config=SEARCH_CONFIG, mode="baseline", index=None):
        return run_module(
            "retrieve", "--config", str(config), "--mode", mode, "--index", str(index or self.index_manifest),
            "--query", query, *extra, cwd=self.cwd,
        )

    def built(self):
        result = self.index(*self.root_arg)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def search_config(self, extra=""):
        path = self.cwd / "search.toml"
        path.write_text(SEARCH_CONFIG.read_text(encoding="utf-8").replace("[filters]\nsources = []\n", "") + extra, encoding="utf-8")
        return path


class EndToEndTests(CliCase):
    def test_index_retrieve_and_validate_work_outside_the_checkout(self):
        summary = self.built()
        self.assertEqual(summary["status"], "built")
        self.assertEqual(summary["stats"]["chunk_count"], 20)
        self.assertTrue(self.index_manifest.is_file())

        found = self.retrieve("how do I load_state_dict from a checkpoint", *self.root_arg)
        self.assertEqual(found.returncode, 0, found.stderr)
        response = json.loads(found.stdout)
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["index_id"], summary["index_id"])
        self.assertEqual(response["baseline_results"][0]["section"], "Checkpoints > Loading")
        self.assertEqual(len(response["config_sha256"]), 64)

        verified = run_module("validate_manifest", "--manifest", str(self.index_manifest), *self.root_arg, cwd=self.cwd)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        self.assertEqual(json.loads(verified.stdout)["status"], "valid")
        corpus = run_module("validate_manifest", "--manifest", str(self.corpus_manifest), cwd=self.cwd)
        self.assertEqual(json.loads(corpus.stdout), {"status": "valid", "kind": "corpus", "id": json.loads(corpus.stdout)["id"], "documents": 6, "chunks": 20})
        self.assertEqual(list(self.cwd.iterdir()), [])                       # nothing written to the working directory

    def test_zero_match_query_succeeds_with_an_empty_status(self):
        self.built()
        result = self.retrieve("kubernetes ingress controller", *self.root_arg)
        self.assertEqual(result.returncode, 0, result.stderr)
        response = json.loads(result.stdout)
        self.assertEqual((response["status"], response["baseline_results"], response["candidate_count"]), ("empty", [], 0))

    def test_rebuilding_is_idempotent(self):
        first = self.built()
        again = self.built()
        self.assertEqual((first["status"], again["status"]), ("built", "unchanged"))
        self.assertEqual(first["index_id"], again["index_id"])

    def test_filters_come_from_the_search_config(self):
        self.built()
        config = self.search_config('[filters]\nsources = ["fixture-transformers"]\n[filters.versions]\nfixture-transformers = "fixture-v2"\n')
        response = json.loads(self.retrieve("from_pretrained", *self.root_arg, config=config).stdout)
        self.assertEqual({r["revision"] for r in response["baseline_results"]}, {"fixture-v2"})
        self.assertEqual(response["filters"]["sources"], ["fixture-transformers"])

    def test_result_file_is_written_once_and_never_overwritten(self):
        self.built()
        target = self.cwd / "results.json"
        first = self.retrieve("checkpoint", "--output", str(target), *self.root_arg)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("status=ok", first.stdout)
        content = target.read_bytes()
        self.assertEqual(json.loads(content)["status"], "ok")
        second = self.retrieve("optimizer", "--output", str(target), *self.root_arg)
        self.assertEqual(second.returncode, 1)
        self.assertIn("refusing to overwrite", second.stderr)
        self.assertEqual(target.read_bytes(), content)

    def test_a_failed_write_leaves_no_partial_result_file(self):
        target = self.cwd / "partial.json"
        with mock.patch("devsearch.retrieve.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                retrieve_module._write_new(target, '{"status": "ok"}')
        self.assertFalse(target.exists(), "a truncated result would block every rerun")
        retrieve_module._write_new(target, "{}")                               # and a rerun now succeeds
        self.assertEqual(target.read_text(encoding="utf-8"), "{}")

    def test_runtime_root_can_come_from_the_environment(self):
        result = run_module(
            "index", "--config", str(INDEX_CONFIG), "--corpus", str(self.corpus_manifest),
            "--output", str(self.index_manifest), cwd=self.cwd, env_extra={"DEVSEARCH_RUNTIME_ROOT": str(self.root)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_without_a_root_artifacts_must_be_colocated(self):
        result = self.index()                                                  # corpus lives in ../corpus
        self.assertEqual(result.returncode, 1)
        self.assertIn("escapes", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(self.index_manifest.exists())


class FailClosedTests(CliCase):
    def test_reranking_is_not_implemented_and_fails_closed(self):
        self.built()
        target = self.cwd / "r.json"
        for mode in ("rerank", "compare"):
            with self.subTest(mode=mode):
                result = self.retrieve("checkpoint", "--output", str(target), *self.root_arg, mode=mode)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn("not implemented", result.stderr)
        with_checkpoint = self.retrieve("checkpoint", "--checkpoint", str(self.cwd / "x.pt"), "--output", str(target), *self.root_arg)
        self.assertEqual(with_checkpoint.returncode, 2)
        self.assertFalse(target.exists())

    def test_invalid_inputs_exit_one_with_a_readable_error_and_no_stdout(self):
        self.built()
        bad_config = self.cwd / "bad.toml"
        bad_config.write_text("schema_version = 1\nsurprise = true\n", encoding="utf-8")
        unknown_source = self.search_config('[filters]\nsources = ["nope"]\n')
        cases = {
            "unknown config key": self.retrieve("q", *self.root_arg, config=bad_config),
            "empty query": self.retrieve("   ", *self.root_arg),
            "unknown source": self.retrieve("checkpoint", *self.root_arg, config=unknown_source),
            "missing index": self.retrieve("q", *self.root_arg, index=self.root / "index" / "none.json"),
            "missing config": self.retrieve("q", *self.root_arg, config=self.cwd / "none.toml"),
            "overlong query": self.retrieve("x" * 1001, *self.root_arg),
        }
        for name, result in cases.items():
            with self.subTest(name):
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr.startswith("devsearch.retrieve: error:"), result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_index_failures_leave_no_artifacts(self):
        bad_config = self.cwd / "bad.toml"
        bad_config.write_text("tokenizer = 5\n", encoding="utf-8")
        missing_root = self.root / "unmounted-volume"
        cases = {
            "bad config": self.index(*self.root_arg, config=bad_config),
            "missing corpus": self.index(*self.root_arg, corpus=self.root / "corpus" / "nope.json"),
            "missing output dir": self.index(*self.root_arg, output=missing_root / "index-manifest.json"),
            "missing runtime root": self.index("--runtime-root", str(missing_root)),
        }
        for name, result in cases.items():
            with self.subTest(name):
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertTrue(result.stderr.startswith("devsearch.index: error:"), result.stderr)
        self.assertFalse(missing_root.exists(), "an unavailable volume must not be recreated")
        self.assertEqual(list((self.root / "index").iterdir()), [])

    def test_a_tampered_index_is_refused_by_retrieve_and_validate(self):
        self.built()
        postings = self.root / "index" / "index-manifest.postings.json"
        postings.write_text(postings.read_text(encoding="utf-8").replace("alpha", "alphb"), encoding="utf-8")
        postings.write_bytes(postings.read_bytes() + b" ")
        for result in (
            self.retrieve("checkpoint", *self.root_arg),
            run_module("validate_manifest", "--manifest", str(self.index_manifest), *self.root_arg, cwd=self.cwd),
        ):
            self.assertEqual(result.returncode, 1)
            self.assertIn("SHA-256", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_validate_manifest_rejects_unknown_kinds_and_bad_json(self):
        unknown = self.cwd / "m.json"
        unknown.write_text('{"kind": "release"}', encoding="utf-8")
        result = run_module("validate_manifest", "--manifest", str(unknown), cwd=self.cwd)
        self.assertEqual(result.returncode, 1)
        self.assertIn("only 'corpus' and 'index' are implemented", result.stderr)
        unknown.write_text("{", encoding="utf-8")
        for bad in (unknown, self.cwd / "none.json"):
            result = run_module("validate_manifest", "--manifest", str(bad), cwd=self.cwd)
            self.assertEqual(result.returncode, 1)
            self.assertTrue(result.stderr.startswith("devsearch.validate_manifest: error:"), result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_usage_errors_exit_two(self):
        for module in ("index", "retrieve", "validate_manifest"):
            with self.subTest(module=module):
                result = run_module(module, cwd=self.cwd)
                self.assertEqual(result.returncode, 2)
                self.assertIn("required", result.stderr)
        self.assertEqual(run_module("retrieve", "--mode", "magic", "--config", "c", "--index", "i", "--query", "q", cwd=self.cwd).returncode, 2)

    def test_help_describes_real_behavior_not_the_scaffold(self):
        for module in ("index", "retrieve", "validate_manifest"):
            with self.subTest(module=module):
                result = run_module(module, "--help", cwd=self.cwd)
                self.assertEqual(result.returncode, 0)
                self.assertNotIn("SCAFFOLD", result.stdout)
                self.assertIn("--runtime-root", result.stdout)


class OfflineTests(CliCase):
    def test_baseline_commands_never_open_a_socket_or_spawn_a_process(self):
        script = """
import contextlib, io, json, sys
def audit(event, args):
    if event.startswith(("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork")):
        raise AssertionError("forbidden operation: " + event)
sys.addaudithook(audit)
from devsearch import index, retrieve, validate_manifest
modules = {"index": index, "retrieve": retrieve, "validate_manifest": validate_manifest}
for name, argv in json.loads(sys.argv[1]):
    with contextlib.redirect_stdout(io.StringIO()):
        assert modules[name].main(argv) == 0, name
for blocked in ("torch", "transformers", "numpy", "fastapi", "streamlit", "yaml", "requests"):
    assert blocked not in sys.modules, blocked
print("offline-ok")
"""
        root = ["--runtime-root", str(self.root)]
        steps = [
            ["index", ["--config", str(INDEX_CONFIG), "--corpus", str(self.corpus_manifest), "--output", str(self.index_manifest), *root]],
            ["retrieve", ["--config", str(SEARCH_CONFIG), "--mode", "baseline", "--index", str(self.index_manifest), "--query", "load_state_dict", *root]],
            ["validate_manifest", ["--manifest", str(self.index_manifest), *root]],
        ]
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, json.dumps(steps)], cwd=self.cwd, capture_output=True, text=True,
            env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}, timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "offline-ok")


if __name__ == "__main__":
    unittest.main()
