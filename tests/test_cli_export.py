"""`devsearch.export`: baseline, candidate and evaluated bundles on a diagnostic tree (docs/03 §4, docs/04 §6).

Real checkpoints, real model export and reload, and a real campaign's final report (all on synthetic data). They prove
the build/verify/refuse behavior and the bindings between report, checkpoint, corpus and index; never ranking quality.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import CONFIGS
from mltest import HAVE_ML, requires_ml
from opsworld import OperatorCase, call

if HAVE_ML:
    from campaign_fixture import completed_campaign
    from devsearch import export
    from devsearch.errors import IndexUnavailableError
    from devsearch.modeling.devices import load_profile, resolve_device
    from devsearch.modeling.io import ENCODER_FILE, HEAD_FILE, load_model_bundle
    from devsearch.modeling.rerank import Reranker
    from devsearch.serving.release import load_release, verify_release

EXPORT_TOML = """schema_version = 1

[device]
profile = "cpu-diagnostic"

[model]
batch_size = 4
dropout = {dropout}

[assets]
directory = "{assets}"

[policy]
max_pair_tokens = 128
max_query_tokens = 32

[search]
path = "search.toml"

[smoke]
queries = {queries}
"""


def run(*argv):
    return call(export, *argv)


def cpu_factory(path):
    device = resolve_device(load_profile("cpu-diagnostic"))
    return Reranker(load_model_bundle(path, device=device), device=device, batch_size=4)


@requires_ml
class ExportCase(OperatorCase):
    """A trained run (shared) plus a private output directory and an export config per test."""

    def write_export_config(self, directory: Path, *, assets=None, queries=None, dropout=0.0, name="export.toml") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "search.toml").write_text((CONFIGS / "search.toml").read_text())
        queries = queries or self.smoke_questions()
        config = directory / name
        config.write_text(EXPORT_TOML.format(assets=assets or (self.root / "bundle" / "model-assets"), dropout=dropout, queries=json.dumps(queries)))
        return config

    def smoke_questions(self, count=3):
        from devsearch.training.inputs import load_training_inputs
        inputs = load_training_inputs(self.prepared / "prepared-data-manifest.json", self.root)
        return [q.text for q in inputs.validation_queries[:count]]


class CandidateExportTests(ExportCase):
    def setUp(self):
        super().setUp()
        self.config = self.write_export_config(self.out_dir)
        self.index = self.prepared / "index" / "index-manifest.json"

    def export(self, *extra, output="bundle"):
        return run("--config", self.config, "--index", self.index, "--output", self.out_dir / output, "--runtime-root", self.root, *extra)

    def staging_leftovers(self):
        return [p.name for p in self.out_dir.iterdir() if p.name.startswith(".")]

    def test_a_baseline_bundle_has_no_model_defaults_to_bm25_and_verifies_without_a_loader(self):
        code, out, err = self.export("--mode", "baseline")
        self.assertEqual((code, err), (0, ""), err)
        summary = json.loads(out)
        self.assertEqual((summary["status"], summary["default_mode"], summary["model_id"], summary["published"]), ("baseline", "baseline", None, True))
        manifest = verify_release(self.out_dir / "bundle", allow_candidate=False)
        self.assertIsNone(manifest["model"])
        self.assertEqual(summary["release_id"], manifest["release_id"])
        loaded = load_release(self.out_dir / "bundle", allow_candidate=False)
        self.assertFalse(loaded.pipeline.rerank_ready)
        self.assertEqual(self.staging_leftovers(), [])

    def test_a_candidate_ships_the_checkpoint_it_was_built_from_and_needs_allow_candidate(self):
        code, out, err = self.export("--checkpoint", self.checkpoint("best.json"))
        self.assertEqual((code, err), (0, ""), err)
        summary = json.loads(out)
        self.assertEqual((summary["status"], summary["default_mode"]), ("candidate", "rerank"))
        self.assertEqual(summary["reload_checks"]["model_parity_after_reload"], "passed")
        self.assertEqual(summary["reload_checks"]["smoke_replay"], "passed")
        bundle = self.out_dir / "bundle"
        with self.assertRaisesRegex(IndexUnavailableError, "allow-candidate"):
            load_release(bundle, allow_candidate=False, reranker_factory=cpu_factory)
        loaded = load_release(bundle, allow_candidate=True, reranker_factory=cpu_factory)
        self.assertTrue(loaded.pipeline.rerank_ready)
        model = json.loads((bundle / "model" / "model-manifest.json").read_text())
        checkpoint = json.loads(self.checkpoint("epoch-0001.json").read_text())
        self.assertEqual(model["provenance"]["checkpoint_sha256"], checkpoint["sha256"])
        self.assertEqual((model["provenance"]["source"], model["provenance"]["completed_epoch"]), ("devsearch.export", 1))
        self.assertIn("never test material", json.dumps(loaded.manifest["provenance"]))
        self.assertEqual(self.staging_leftovers(), [])

    def test_an_identical_rerun_changes_nothing_and_a_different_bundle_at_the_same_path_is_refused(self):
        self.assertEqual(self.export("--mode", "baseline")[0], 0)
        before = {p: p.read_bytes() for p in (self.out_dir / "bundle").rglob("*") if p.is_file()}
        code, out, err = self.export("--mode", "baseline")
        self.assertEqual((code, err), (0, ""), err)
        self.assertFalse(json.loads(out)["published"])
        self.assertEqual(before, {p: p.read_bytes() for p in (self.out_dir / "bundle").rglob("*") if p.is_file()})
        other = self.write_export_config(self.out_dir / "other", queries=self.smoke_questions(2))
        code, _, err = run("--config", other, "--mode", "baseline", "--index", self.index, "--output", self.out_dir / "bundle", "--runtime-root", self.root)
        self.assertEqual(code, 1)
        self.assertIn("different release", err)
        self.assertEqual(self.staging_leftovers(), [])

    def test_an_exported_bundle_is_reproducible_bit_for_bit_in_its_model_weights(self):
        self.assertEqual(self.export("--checkpoint", self.checkpoint("epoch-0001.json"), output="one")[0], 0)
        self.assertEqual(self.export("--checkpoint", self.checkpoint("best.json"), output="two")[0], 0)
        one = json.loads((self.out_dir / "one" / "model" / "model-manifest.json").read_text())
        two = json.loads((self.out_dir / "two" / "model" / "model-manifest.json").read_text())
        self.assertEqual(one["model_id"], two["model_id"])
        for name in (ENCODER_FILE, HEAD_FILE):
            self.assertEqual(one["files"][name], two["files"][name])

    def test_an_inference_bundle_can_be_the_model_source(self):
        self.assertEqual(self.export("--checkpoint", self.checkpoint("epoch-0001.json"), output="first")[0], 0)
        code, out, err = self.export("--checkpoint", self.out_dir / "first" / "model", output="second")
        self.assertEqual((code, err), (0, ""), err)
        self.assertEqual(json.loads(out)["model_id"], json.loads((self.out_dir / "first" / "model" / "model-manifest.json").read_text())["model_id"])

    def test_only_a_completed_epoch_checkpoint_can_become_a_candidate(self):
        code, out, err = self.export("--checkpoint", self.checkpoint("epoch-0000.json"))
        self.assertEqual((code, out), (1, ""))
        self.assertIn("completed training epoch", err)
        self.assertFalse((self.out_dir / "bundle").exists())
        self.assertEqual(self.staging_leftovers(), [])

    def test_a_failing_reload_check_publishes_nothing_and_cleans_up(self):
        failing = {"passed": False, "failures": ["forced"], "queries": 1, "rerank_checked": 0, "atol": 0.0}
        with mock.patch("devsearch.execution.exporter.check_smoke", return_value=failing):
            code, out, err = self.export("--mode", "baseline")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("does not reproduce its recorded smoke", err)
        self.assertFalse((self.out_dir / "bundle").exists())
        self.assertEqual(self.staging_leftovers(), [])

    def test_usage_errors_exit_2_before_any_work(self):
        cases = [
            ("neither a mode nor a checkpoint", ["--index", self.index]),
            ("both", ["--index", self.index, "--mode", "baseline", "--checkpoint", self.checkpoint()]),
            ("no index", ["--mode", "baseline"]),
            ("a candidate without a report", ["--candidate", self.out_dir]),
            ("a report without a candidate", ["--final-report", self.out_dir / "final-report.json"]),
            ("evaluated mixed with an index", ["--candidate", self.out_dir, "--final-report", self.out_dir / "x.json", "--index", self.index]),
        ]
        for label, extra in cases:
            with self.subTest(label):
                code, out, err = run("--config", self.config, "--output", self.out_dir / "bundle", "--runtime-root", self.root, *extra)
                self.assertEqual((code, out), (2, ""), err)
                self.assertFalse((self.out_dir / "bundle").exists())

    def test_the_output_must_be_inside_the_runtime_root_and_its_parent_must_exist(self):
        outside = Path(tempfile.mkdtemp(prefix="devsearch-outside-"))
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        code, _, err = run("--config", self.config, "--mode", "baseline", "--index", self.index, "--output", outside / "bundle", "--runtime-root", self.root)
        self.assertEqual(code, 1)
        self.assertIn("release output", err)
        self.assertEqual(list(outside.iterdir()), [])
        code, _, err = self.export("--mode", "baseline", output="nowhere/bundle")
        self.assertEqual(code, 1)
        self.assertIn("never creates parent directories", err)
        self.assertFalse((self.out_dir / "nowhere").exists())

    def test_strict_configuration_failures_name_the_problem(self):
        text = self.config.read_text()
        cases = {
            "no smoke section": text.replace("[smoke]\nqueries", "[unknown]\nqueries"),
            "no questions": text[:text.index("queries = ")] + "queries = []\n",
            "too many questions": text[:text.index("queries = ")] + "queries = " + json.dumps([f"question {i}?" for i in range(9)]) + "\n",
            "duplicate questions": text[:text.index("queries = ")] + 'queries = ["a b", "a b"]\n',
            "bad dropout": text.replace("dropout = 0.0", "dropout = 1.5"),
            "missing search file": text.replace('path = "search.toml"', 'path = "nowhere.toml"'),
        }
        for label, body in cases.items():
            with self.subTest(label):
                config = self.out_dir / "bad.toml"
                config.write_text(body)
                code, out, err = run("--config", config, "--mode", "baseline", "--index", self.index, "--output", self.out_dir / "bundle", "--runtime-root", self.root)
                self.assertEqual((code, out), (1, ""), err)
                self.assertIn("devsearch.export: error", err)
                self.assertNotIn("Traceback", err)
                self.assertFalse((self.out_dir / "bundle").exists())

    def test_exporting_needs_no_test_side_files_at_all(self):
        hidden = self.root / "evaluation.hidden"
        (self.root / "evaluation").rename(hidden)
        try:
            code, _, err = self.export("--mode", "baseline")
            code_model, _, err_model = self.export("--checkpoint", self.checkpoint("best.json"), output="model")
        finally:
            hidden.rename(self.root / "evaluation")
        self.assertEqual((code, err, code_model, err_model), (0, "", 0, ""))


@requires_ml
class EvaluatedExportTests(unittest.TestCase):
    """A candidate built from a real campaign's selected checkpoint, then promoted with that campaign's own final report."""

    @classmethod
    def setUpClass(cls):
        cls.campaign = completed_campaign(topics=36)
        cls.holder = Path(tempfile.mkdtemp(prefix="devsearch-eval-export-", dir=cls.campaign.root / "runtime")).resolve()
        cls.addClassCleanup(shutil.rmtree, cls.holder, ignore_errors=True)

    def setUp(self):
        self.assertEqual(self.campaign.decision["default_mode"], "rerank", "the planted rehearsal must ship the reranker")
        self.out = Path(tempfile.mkdtemp(dir=self.holder))
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        shared = self.campaign.plan["training"]["shared"]
        self.config = self.out / "export.toml"
        shutil.copyfile(self.campaign.release / "config" / "search.toml", self.out / "search.toml")      # the limits the campaign itself ran under
        questions = [item["query"] for item in json.loads((self.campaign.release / "smoke" / "smoke.json").read_text())["queries"]][:3]
        self.config.write_text(EXPORT_TOML.format(assets=self.campaign.assets, dropout=shared["dropout"], queries=json.dumps(questions)))

    def candidate(self, name="candidate", checkpoint=None):
        code, out, err = run("--config", self.config, "--index", self.campaign.index, "--checkpoint", checkpoint or self.campaign.checkpoint,
                             "--output", self.out / name, "--runtime-root", self.campaign.root)
        self.assertEqual((code, err), (0, ""), err)
        return self.out / name

    def promote(self, candidate, report, name="evaluated"):
        return run("--config", self.config, "--candidate", candidate, "--final-report", report, "--output", self.out / name, "--runtime-root", self.campaign.root)

    def report_copy(self, name="report"):
        copy = self.out / name
        shutil.copytree(self.campaign.report_dir, copy)
        return copy

    def test_the_campaign_report_promotes_the_candidate_built_from_its_selected_checkpoint(self):
        candidate = self.candidate()
        code, out, err = self.promote(candidate, self.campaign.report_dir / "final-report.json")
        self.assertEqual((code, err), (0, ""), err)
        summary = json.loads(out)
        self.assertEqual((summary["status"], summary["default_mode"]), ("evaluated", "rerank"))
        report = json.loads((self.campaign.report_dir / "final-report.json").read_text())
        self.assertEqual(summary["final_report_id"], report["final_report_id"])
        loaded = load_release(self.out / "evaluated", allow_candidate=False, reranker_factory=cpu_factory)
        self.assertTrue(loaded.pipeline.rerank_ready)
        provenance = loaded.manifest["provenance"]
        self.assertIn("authoritative path", provenance["report_origin"])
        # the same weights as the model the campaign itself shipped: both come from the one selected checkpoint
        ours = json.loads((self.out / "evaluated" / "model" / "model-manifest.json").read_text())["files"]
        theirs = json.loads((self.campaign.release / "model" / "model-manifest.json").read_text())["files"]
        for name in (ENCODER_FILE, HEAD_FILE):
            self.assertEqual(ours[name]["sha256"], theirs[name]["sha256"], name)
        # and the same corpus, index and report files as the campaign's own evaluated release
        campaign_files = json.loads((self.campaign.release / "release-manifest.json").read_text())["files"]
        own_files = loaded.manifest["files"]
        for name, record in campaign_files.items():
            if name.startswith(("corpus/", "index/", "evaluation/", "config/")):  # everything but the model and the smoke file
                self.assertEqual(own_files[name], record, name)

    def test_the_candidate_can_be_named_by_its_manifest_file(self):
        candidate = self.candidate()
        code, _, err = self.promote(candidate / "release-manifest.json", self.campaign.report_dir / "final-report.json")
        self.assertEqual((code, err), (0, ""), err)

    def test_a_model_other_than_the_selected_checkpoint_cannot_be_promoted(self):
        others = [p for p in self.campaign.checkpoint.parent.glob("epoch-000[1-3].json") if p != self.campaign.checkpoint]
        self.assertTrue(others, "the diagnostic recipe completes several epochs")
        candidate = self.candidate("other", checkpoint=others[0])
        code, out, err = self.promote(candidate, self.campaign.report_dir / "final-report.json", "wrong")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("not the checkpoint the final report selected", err)
        self.assertFalse((self.out / "wrong").exists())
        self.assertEqual([p.name for p in self.out.iterdir() if p.name.startswith(".")], [])

    def test_an_edited_incomplete_or_misnamed_report_is_refused(self):
        candidate = self.candidate()
        edited = self.report_copy()
        data = json.loads((edited / "final-report.json").read_text())
        data["systems"][0]["group_macro"]["mrr@10"] = 0.999
        (edited / "final-report.json").write_text(json.dumps(data))
        code, _, err = self.promote(candidate, edited / "final-report.json", "out-edited")
        self.assertEqual(code, 1)
        self.assertIn("final_report_id", err)
        incomplete = self.report_copy("incomplete")
        (incomplete / "per-query.jsonl").unlink()
        code, _, err = self.promote(candidate, incomplete / "final-report.json", "out-incomplete")
        self.assertEqual(code, 1)
        self.assertIn("per-query.jsonl", err)
        misnamed = self.out / "renamed.json"
        shutil.copyfile(self.campaign.report_dir / "final-report.json", misnamed)
        code, _, err = self.promote(candidate, misnamed, "out-misnamed")
        self.assertEqual(code, 1)
        self.assertIn("final-report.json", err)
        for name in ("out-edited", "out-incomplete", "out-misnamed"):
            self.assertFalse((self.out / name).exists())

    def test_an_already_evaluated_release_and_a_non_release_are_refused(self):
        code, _, err = self.promote(self.campaign.release, self.campaign.report_dir / "final-report.json", "again")
        self.assertEqual(code, 1)
        self.assertIn("already evaluated", err)
        code, _, err = self.promote(self.out, self.campaign.report_dir / "final-report.json", "nothing")
        self.assertEqual(code, 1)
        self.assertIn("neither a release directory", err)

    def test_a_rerun_is_idempotent_and_an_occupied_output_is_never_overwritten(self):
        candidate = self.candidate()
        report = self.campaign.report_dir / "final-report.json"
        self.assertEqual(self.promote(candidate, report)[0], 0)
        code, out, err = self.promote(candidate, report)
        self.assertEqual((code, err), (0, ""), err)
        self.assertFalse(json.loads(out)["published"])
        (self.out / "occupied").mkdir()
        (self.out / "occupied" / "note.txt").write_text("not a release")
        code, _, err = self.promote(candidate, report, "occupied")
        self.assertEqual(code, 1)
        self.assertEqual((self.out / "occupied" / "note.txt").read_text(), "not a release")


@requires_ml
class BaselineDefaultEvaluatedExportTests(unittest.TestCase):
    """When validation did not support the reranker the report says baseline, and the evaluated release honestly ships no model."""

    @classmethod
    def setUpClass(cls):
        cls.campaign = completed_campaign(topics=36, signal="none")
        cls.holder = Path(tempfile.mkdtemp(prefix="devsearch-eval-baseline-", dir=cls.campaign.root / "runtime")).resolve()
        cls.addClassCleanup(shutil.rmtree, cls.holder, ignore_errors=True)

    def test_a_baseline_default_report_promotes_a_baseline_bundle_and_drops_the_model(self):
        self.assertEqual(self.campaign.decision["default_mode"], "baseline")
        shutil.copyfile(self.campaign.release / "config" / "search.toml", self.holder / "search.toml")
        questions = [item["query"] for item in json.loads((self.campaign.release / "smoke" / "smoke.json").read_text())["queries"]][:2]
        config = self.holder / "export.toml"
        config.write_text(EXPORT_TOML.format(assets=self.campaign.assets, dropout=0.0, queries=json.dumps(questions)))
        base = ("--config", config, "--runtime-root", self.campaign.root)
        code, _, err = run(*base, "--mode", "baseline", "--index", self.campaign.index, "--output", self.holder / "baseline")
        self.assertEqual((code, err), (0, ""), err)
        code, out, err = run(*base, "--candidate", self.holder / "baseline", "--final-report", self.campaign.report_dir / "final-report.json",
                             "--output", self.holder / "evaluated")
        self.assertEqual((code, err), (0, ""), err)
        summary = json.loads(out)
        self.assertEqual((summary["status"], summary["default_mode"], summary["model_id"]), ("evaluated", "baseline", None))
        loaded = load_release(self.holder / "evaluated", allow_candidate=False)
        self.assertFalse(loaded.pipeline.rerank_ready)


if __name__ == "__main__":
    unittest.main()
