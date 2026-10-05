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
from mltest import HAVE_ML

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

    def test_a_seal_stage_must_name_the_file_it_really_writes(self):
        for stage, extra in (("bank", ["--reviews", str(self.root / "reviews.jsonl")]), ("seal-initial", [])):
            with self.subTest(stage):
                code, out, err = run("--config", str(self.config), "--stage", stage, "--prepared-manifest", str(self.root / "prepared" / PREPARED_MANIFEST),
                                     "--evaluation-manifest", str(self.root / "evaluation" / EVALUATION_MANIFEST), *extra,
                                     "--output", str(self.root / INITIAL_SEAL), "--runtime-root", str(self.root))
                self.assertEqual((code, out), (1, ""))
                self.assertIn("inside the prepared directory", err)

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


@unittest.skipUnless(HAVE_ML, "the tokenizers library is not installed")
class LengthCheckTests(unittest.TestCase):
    """`[lengths]`: every passage and question (the held-out split included) is measured before anything is written."""

    def setUp(self):
        self.root = make_root(self)
        self.config = write_config(self.root / "drafts")
        self.corpus = self.root / "corpus-in" / "corpus-manifest.json"
        self.split = self.root / "prepared" / SPLIT_MANIFEST
        self.assertEqual(run("--config", str(self.config), "--corpus", str(self.corpus), "--output", str(self.split), "--runtime-root", str(self.root))[0], 0)
        drafts_from_split(self.root, load_split(self.split))
        from devsearch.data.corpus import load_corpus
        from devsearch.modeling.tokenization import write_diagnostic_tokenizer
        words = {w for chunk in load_corpus(self.corpus).chunks for w in chunk.text.replace(".", " ").replace("`", " ").split()}
        (self.root / "tokenizer").mkdir()
        write_diagnostic_tokenizer(self.root / "tokenizer", sorted(words))

    def materialize(self, lengths: str):
        self.config.write_text(self.config.read_text() + lengths)
        return run("--config", str(self.config), "--stage", "materialize", "--split-manifest", str(self.split), "--evaluation-dir", str(self.root / "evaluation"),
                   "--output", str(self.root / "prepared" / PREPARED_MANIFEST), "--runtime-root", str(self.root))

    def test_a_policy_everything_fits_reports_that_lengths_were_checked(self):
        code, out, err = self.materialize('[lengths]\ntokenizer_assets = "../tokenizer"\nmax_pair_tokens = 4000\nmax_query_tokens = 64\n')
        self.assertEqual((code, err), (0, ""), err)
        self.assertTrue(json.loads(out)["lengths_checked"])

    def test_without_the_section_nothing_is_measured_and_the_output_says_so(self):
        code, out, err = self.materialize("")
        self.assertEqual((code, err), (0, ""), err)
        self.assertFalse(json.loads(out)["lengths_checked"])

    def test_an_over_long_question_or_passage_is_found_before_anything_is_written(self):
        code, out, err = self.materialize('[lengths]\ntokenizer_assets = "../tokenizer"\nmax_pair_tokens = 4000\nmax_query_tokens = 8\n')
        self.assertEqual((code, out), (1, ""))
        self.assertIn("nothing was written", err)
        self.assertIn("reserve 8", err)
        self.assertFalse((self.root / "prepared" / PREPARED_MANIFEST).exists())
        self.assertEqual(list((self.root / "evaluation").iterdir()), [], "no evaluation-side input was created")

    def test_a_duplicated_or_inconsistent_section_is_a_configuration_error(self):
        section = '[lengths]\ntokenizer_assets = "../tokenizer"\nmax_pair_tokens = 4000\nmax_query_tokens = 64\n'
        code, out, err = self.materialize(section + section)
        self.assertEqual((code, out), (1, ""))
        code, out, err = self.materialize('[lengths]\ntokenizer_assets = "../tokenizer"\nmax_pair_tokens = 64\nmax_query_tokens = 64\n')
        self.assertEqual((code, out), (1, ""))


class ExampleConfigTests(unittest.TestCase):
    def test_the_shipped_example_fails_validation_until_the_salt_is_replaced(self):
        from helpers import CONFIGS
        from devsearch.data.config import load_data_config
        from devsearch.errors import ConfigError
        with self.assertRaisesRegex(ConfigError, "review.salt"):
            load_data_config(CONFIGS / "data.toml")


if __name__ == "__main__":
    unittest.main()
