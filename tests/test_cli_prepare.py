"""`devsearch.prepare_data` command: stage routing, exit codes, output naming, no partial artifacts."""

import contextlib
import io
import json
import subprocess
import sys
import unittest

from devsearch import prepare_data
from devsearch.data.prepare import EVALUATION_MANIFEST, INITIAL_SEAL, PREPARED_MANIFEST, SPLIT_MANIFEST, load_split
from helpers import SRC

from prepworld import drafts_from_split, make_root, write_config


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = prepare_data.main(list(argv))
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


class PrepareDataCliTests(unittest.TestCase):
    def setUp(self):
        self.root = make_root(self)
        self.config = write_config(self.root / "drafts")
        self.corpus = self.root / "corpus-in" / "corpus-manifest.json"

    def groups(self, **kw):
        return run("--config", str(self.config), "--corpus", str(self.corpus), "--output", str(self.root / "prepared" / SPLIT_MANIFEST),
                   "--runtime-root", str(self.root), **kw)

    def test_the_documented_form_runs_the_groups_stage_by_default(self):
        code, out, err = self.groups()
        self.assertEqual((code, err), (0, ""))
        report = json.loads(out)
        self.assertEqual((report["stage"], report["kind"]), ("groups", "split"))
        self.assertTrue((self.root / "prepared" / SPLIT_MANIFEST).is_file())

    def test_all_stages_run_in_order_through_the_cli(self):
        self.assertEqual(self.groups()[0], 0)
        drafts_from_split(self.root, load_split(self.root / "prepared" / SPLIT_MANIFEST))
        code, out, err = run("--config", str(self.config), "--stage", "materialize", "--split-manifest", str(self.root / "prepared" / SPLIT_MANIFEST),
                             "--evaluation-dir", str(self.root / "evaluation"), "--output", str(self.root / "prepared" / PREPARED_MANIFEST),
                             "--runtime-root", str(self.root))
        self.assertEqual((code, err), (0, ""), err)
        self.assertTrue((self.root / "evaluation" / EVALUATION_MANIFEST).is_file())

    def test_a_wrong_output_name_a_missing_option_and_a_bad_input_fail_cleanly_without_artifacts(self):
        code, out, err = run("--config", str(self.config), "--corpus", str(self.corpus), "--output", str(self.root / "prepared" / "whatever.json"),
                             "--runtime-root", str(self.root))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("devsearch.prepare_data: error:", err)
        self.assertIn(SPLIT_MANIFEST, err)
        self.assertNotIn("Traceback", err)
        code, _, err = run("--config", str(self.config), "--stage", "materialize", "--output", str(self.root / "prepared" / PREPARED_MANIFEST),
                           "--runtime-root", str(self.root))
        self.assertEqual(code, 1)
        self.assertIn("--split-manifest", err)
        code, _, err = run("--config", str(self.root / "missing.toml"), "--corpus", str(self.corpus), "--output", str(self.root / "prepared" / SPLIT_MANIFEST),
                           "--runtime-root", str(self.root))
        self.assertEqual(code, 1)
        self.assertEqual(list((self.root / "prepared").iterdir()), [])

    def test_unknown_options_are_a_usage_error(self):
        self.assertEqual(run("--config", str(self.config), "--output", "x", "--stage", "nonsense")[0], 2)
        self.assertEqual(run("--bogus")[0], 2)

    def test_the_seal_stage_names_are_distinct_from_their_manifests(self):
        self.assertEqual(prepare_data.OUTPUT_NAMES["bank"], INITIAL_SEAL)

    def test_help_works_in_a_fresh_process_outside_the_checkout_and_writes_nothing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-m", "devsearch.prepare_data", "--help"], cwd=directory, capture_output=True, text=True,
                                    env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--stage", result.stdout)
            self.assertNotIn("SCAFFOLD ONLY", result.stdout)


if __name__ == "__main__":
    unittest.main()
