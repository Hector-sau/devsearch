"""The operator commands `train`, `predict` and `evaluate` on a diagnostic tree (docs/03 §4, docs/04 §6).

They run the genuine loop, model loading and ranking on CPU with synthetic data, so they prove behavior, refusals and
contracts, never ranking quality. The test side is deliberately unreachable from `evaluate`.
"""

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mltest import HAVE_ML
from opsworld import TRAIN_TOML, OperatorCase, call

if HAVE_ML:
    from devsearch import evaluate, predict, train
    from devsearch.modeling.devices import load_profile, resolve_device
    from devsearch.modeling.policy import PairEncodingPolicy
    from devsearch.modeling.scoring import score_pairs
    from devsearch.training.resolve import resolve_model


class TrainTests(OperatorCase):
    def test_training_completes_with_checkpoints_events_and_a_summary(self):
        code, out, err = self.trained
        self.assertEqual((code, err), (0, ""), err)
        summary = json.loads(out)
        self.assertEqual((summary["status"], summary["completed_epochs"], summary["device"]), ("completed", 1, "cpu"))
        self.assertEqual(summary["recipe_id"], "cli-frozen")
        names = {p.name for p in (self.run_dir / "checkpoints").iterdir()}
        self.assertTrue({"epoch-0000.json", "epoch-0001.json", "latest.json", "best.json"} <= names, names)
        events = [json.loads(line) for line in (self.run_dir / "events.jsonl").read_text().splitlines()]
        self.assertTrue(events)
        self.assertTrue(all(isinstance(e, dict) for e in events))

    def test_an_existing_run_is_never_silently_restarted_or_resumed_from_a_forked_id(self):
        code, _, err = call(train, "--config", self.config, "--run-dir", self.run_dir, "--runtime-root", self.root)
        self.assertEqual(code, 1)
        self.assertIn("--resume", err)
        self.assertNotIn("Traceback", err)
        code, _, err = call(train, "--config", self.config, "--run-dir", self.run_dir, "--runtime-root", self.root, "--resume", "epoch-0000")
        self.assertEqual(code, 1)
        self.assertIn("only the latest complete-epoch checkpoint", err)

    def test_resuming_a_finished_run_changes_nothing(self):
        before = {p.name: p.read_bytes() for p in (self.run_dir / "checkpoints").iterdir()}
        code, out, err = call(train, "--config", self.config, "--run-dir", self.run_dir, "--runtime-root", self.root, "--resume", "latest")
        self.assertEqual((code, err), (0, ""), err)
        self.assertEqual(json.loads(out)["completed_epochs"], 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.run_dir / "checkpoints").iterdir()})

    def test_a_changed_recipe_cannot_continue_an_existing_run(self):
        changed = self.root / "changed.toml"
        changed.write_text(TRAIN_TOML.format(epochs=1).replace("learning_rate = 0.01", "learning_rate = 0.02"))
        code, _, err = call(train, "--config", changed, "--run-dir", self.run_dir, "--runtime-root", self.root, "--resume", "latest")
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)

    def test_resume_with_no_checkpoint_and_a_missing_run_directory_fail_without_creating_anything(self):
        empty = self.out_dir / "empty"
        empty.mkdir()
        code, _, err = call(train, "--config", self.config, "--run-dir", empty, "--runtime-root", self.root, "--resume", "latest")
        self.assertEqual(code, 1)
        self.assertIn("nothing to resume", err)
        missing = self.out_dir / "missing"
        code, _, err = call(train, "--config", self.config, "--run-dir", missing, "--runtime-root", self.root)
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)
        self.assertFalse(missing.exists())

    def test_the_run_directory_and_inputs_must_lie_inside_the_runtime_root(self):
        elsewhere = Path(tempfile.mkdtemp(prefix="devsearch-outside-"))
        self.addCleanup(shutil.rmtree, elsewhere, ignore_errors=True)
        code, _, err = call(train, "--config", self.config, "--run-dir", elsewhere, "--runtime-root", self.root)
        self.assertEqual(code, 1)
        self.assertIn("run directory", err)
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_strict_configuration_failures_name_the_problem_and_leave_no_artifacts(self):
        cases = {
            "no-job-sections": re.sub(r"\n\[(data|assets|device|policy)\]\n[^\[]*", "\n", TRAIN_TOML.format(epochs=1)),
            "unknown-key": TRAIN_TOML.format(epochs=1).replace("[device]\n", "[device]\nextra = 1\n"),
            "too-many-epochs": TRAIN_TOML.format(epochs=9),
            "indivisible-batch": TRAIN_TOML.format(epochs=1).replace("microbatch = 4", "microbatch = 3"),
            "bad-profile": TRAIN_TOML.format(epochs=1).replace("cpu-diagnostic", "quantum"),
        }
        for name, text in cases.items():
            with self.subTest(name):
                config = self.out_dir / f"{name}.toml"
                config.write_text(text)
                run = self.out_dir / f"run-{name}"
                run.mkdir()
                code, out, err = call(train, "--config", config, "--run-dir", run, "--runtime-root", self.root)
                self.assertEqual((code, out), (1, ""), err)
                self.assertIn("devsearch.train: error", err)
                self.assertNotIn("Traceback", err)
                self.assertEqual(list(run.iterdir()), [])


class PredictTests(OperatorCase):
    def predict(self, pairs, output=None, checkpoint=None, config=None):
        output = output or self.out_dir / "scores.jsonl"
        return call(predict, "--config", config or self.model_config, "--checkpoint", checkpoint or self.checkpoint(), "--pairs", pairs,
                    "--output", output, "--runtime-root", self.root), output

    def test_scores_match_the_shared_scorer_in_input_order_and_carry_the_model_identity(self):
        pairs = self.pairs_file()
        (code, out, err), output = self.predict(pairs)
        self.assertEqual((code, err), (0, ""), err)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual([r["pair_id"] for r in rows], ["p0", "p1", "p2"])
        self.assertEqual(len({r["model_id"] for r in rows}), 1)
        self.assertEqual(json.loads(out)["model_id"], rows[0]["model_id"])
        device = resolve_device(load_profile("cpu-diagnostic"))
        resolved = resolve_model(self.checkpoint(), device=device, assets=self.root / "bundle" / "model-assets", policy=PairEncodingPolicy(128, 32))
        inputs = [json.loads(line) for line in pairs.read_text().splitlines()]
        direct = score_pairs(resolved.model, resolved.tokenizer, resolved.policy, [(i["query"], i["passage"]) for i in inputs], device=device, batch_size=4)
        self.assertEqual([r["logit"] for r in rows], direct)
        self.assertTrue(all(isinstance(r["logit"], float) for r in rows))

    def test_the_best_pointer_and_the_checkpoint_manifest_resolve_to_the_same_model(self):
        (a, _, _), first = self.predict(self.pairs_file(), self.out_dir / "a.jsonl", self.checkpoint("best.json"))
        (b, _, _), second = self.predict(self.pairs_file(), self.out_dir / "b.jsonl", self.checkpoint("epoch-0001.json"))
        self.assertEqual((a, b), (0, 0))
        self.assertEqual(first.read_text(), second.read_text())

    def test_an_existing_output_is_never_overwritten(self):
        pairs = self.pairs_file()
        (code, _, _), output = self.predict(pairs)
        self.assertEqual(code, 0)
        kept = output.read_bytes()
        (code, _, err), _ = self.predict(pairs, output)
        self.assertEqual(code, 1)
        self.assertIn("already exists", err)
        self.assertEqual(output.read_bytes(), kept)

    def test_scoring_is_all_or_nothing_and_failures_name_the_cause(self):
        long_text = "word " * 2000
        bad = {
            "a pair that does not fit the length policy": (self.pairs_file(2, {1: {"passage": long_text}}, "long.jsonl"), "token"),
            "a duplicate pair id": (self.pairs_file(2, {1: {"pair_id": "p0"}}, "dup.jsonl"), "duplicate"),
            "an unknown field": (None, "unrecognized field"),
        }
        for number, (label, (pairs, needle)) in enumerate(bad.items()):
            with self.subTest(label):
                if pairs is None:
                    pairs = self.out_dir / "extra.jsonl"
                    pairs.write_text(json.dumps({"pair_id": "x", "query": "q", "passage": "p", "extra": 1}) + "\n")
                (code, out, err), output = self.predict(pairs, self.out_dir / f"scores-{number}.jsonl")
                self.assertEqual((code, out), (1, ""), err)
                self.assertIn(needle, err.lower())
                self.assertNotIn("Traceback", err)
                self.assertFalse(output.exists())
                self.assertEqual([p for p in self.out_dir.iterdir() if p.name.startswith(".")], [])

    def test_empty_input_missing_checkpoint_and_tampered_checkpoint_fail_cleanly(self):
        empty = self.out_dir / "empty.jsonl"
        empty.write_text("")
        (code, _, err), output = self.predict(empty)
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
        (code, _, err), output = self.predict(self.pairs_file(), checkpoint=self.run_dir / "checkpoints" / "nope.json")
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
        copy = self.out_dir / "checkpoints"
        shutil.copytree(self.run_dir / "checkpoints", copy)
        weights = next(copy.glob("epoch-0001.pt"))
        weights.write_bytes(weights.read_bytes()[:-8] + b"tampered")
        (code, _, err), output = self.predict(self.pairs_file(), checkpoint=copy / "epoch-0001.json")
        self.assertEqual(code, 1)
        self.assertIn("SHA-256", err)
        self.assertFalse(output.exists())

    def test_a_manifest_naming_a_path_instead_of_a_file_is_refused(self):
        copy = self.out_dir / "checkpoints"
        shutil.copytree(self.run_dir / "checkpoints", copy)
        manifest = json.loads((copy / "epoch-0001.json").read_text())
        manifest["file"] = "../checkpoints/epoch-0001.pt"
        (copy / "epoch-0001.json").write_text(json.dumps(manifest))
        (code, _, err), output = self.predict(self.pairs_file(), checkpoint=copy / "epoch-0001.json")
        self.assertEqual(code, 1)
        self.assertIn("plain `file` name", err)
        self.assertFalse(output.exists())

    def test_the_output_must_be_inside_the_runtime_root_and_its_directory_must_exist(self):
        outside = Path(tempfile.mkdtemp(prefix="devsearch-outside-"))
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        (code, _, err), output = self.predict(self.pairs_file(), outside / "scores.jsonl")
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())
        (code, _, err), output = self.predict(self.pairs_file(), self.out_dir / "no-such-dir" / "scores.jsonl")
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)

    def test_a_checkpoint_needs_the_assets_and_policy_a_bundle_carries_itself(self):
        config = self.out_dir / "bare.toml"
        config.write_text("schema_version = 1\n\n[device]\nprofile = \"cpu-diagnostic\"\n\n[model]\nbatch_size = 4\n")
        (code, _, err), output = self.predict(self.pairs_file(), config=config)
        self.assertEqual(code, 1)
        self.assertIn("assets", err)
        self.assertFalse(output.exists())


class EvaluateTests(OperatorCase):
    def evaluate(self, *extra, mode="baseline", split="validation", name="metrics.json"):
        output = self.out_dir / name
        return call(evaluate, "--config", self.model_config, "--mode", mode, "--index", self.prepared / "index" / "index-manifest.json",
                    "--split", split, "--output", output, "--runtime-root", self.root, *extra), output

    def test_baseline_and_rerank_score_the_same_frozen_candidates_over_the_same_denominator(self):
        (code, out, err), base_path = self.evaluate(name="base.json")
        self.assertEqual((code, err), (0, ""), err)
        (code, out, err), rerank_path = self.evaluate("--checkpoint", self.checkpoint(), mode="rerank", name="rerank.json")
        self.assertEqual((code, err), (0, ""), err)
        base, rerank = json.loads(base_path.read_text()), json.loads(rerank_path.read_text())
        validation_queries = (self.prepared / "data" / "queries-validation.jsonl").read_text().splitlines()
        self.assertEqual(base["queries_scored"] + len(base["queries_without_positive"]), len(validation_queries))
        for key in ("queries_scored", "groups_scored", "queries_without_positive", "positives_missing_from_candidates", "candidate_source", "candidate_depth", "index_id"):
            self.assertEqual(base[key], rerank[key], key)
        self.assertIn("stored", base["candidate_source"])
        self.assertEqual((base["mode"], rerank["mode"]), ("baseline", "rerank"))
        self.assertIsNone(base["model_id"])
        self.assertTrue(rerank["model_id"])
        # a reranker only reorders the same lists, so the known-positive recall of the fixed candidate list cannot change
        self.assertEqual(base["group_macro"]["recall@50"], rerank["group_macro"]["recall@50"])
        self.assertEqual(base["query_macro"]["recall@50"], rerank["query_macro"]["recall@50"])
        self.assertIn("never inserted", base["recall_note"])
        self.assertEqual(base["label_provenance"]["query_origins"], rerank["label_provenance"]["query_origins"])
        self.assertTrue(set(base["per_query"]) == set(rerank["per_query"]))
        for metrics in list(base["per_query"].values()) + list(rerank["per_query"].values()):
            self.assertTrue(all(0.0 <= metrics[m] <= 1.0 for m in ("mrr@10", "ndcg@10", "recall@50")))

    def test_the_summary_line_reports_the_group_macro_scores(self):
        (code, out, _), path = self.evaluate()
        self.assertEqual(code, 0)
        line = json.loads(out)
        self.assertEqual(line["group_macro"], json.loads(path.read_text())["group_macro"])

    def test_metrics_are_never_overwritten_and_must_stay_inside_the_runtime_root(self):
        (code, _, _), path = self.evaluate()
        self.assertEqual(code, 0)
        kept = path.read_bytes()
        (code, _, err), _ = self.evaluate()
        self.assertEqual(code, 1)
        self.assertIn("already exists", err)
        self.assertEqual(path.read_bytes(), kept)
        outside = Path(tempfile.mkdtemp(prefix="devsearch-outside-"))
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        code, _, err = call(evaluate, "--config", self.model_config, "--mode", "baseline", "--index", self.prepared / "index" / "index-manifest.json",
                            "--output", outside / "m.json", "--runtime-root", self.root)
        self.assertEqual(code, 1)
        self.assertEqual(list(outside.iterdir()), [])

    def test_the_test_split_and_the_final_stages_are_refused_with_an_explanation_and_no_output(self):
        for extra in (("--split", "test"), ("--stage", "rank"), ("--stage", "pool"), ("--stage", "score")):
            with self.subTest(extra):
                output = self.out_dir / "t.json"
                code, out, err = call(evaluate, "--config", self.model_config, "--mode", "baseline", "--index", self.prepared / "index" / "index-manifest.json",
                                      "--output", output, "--runtime-root", self.root, *extra)
                self.assertEqual((code, out), (2, ""))
                self.assertIn("campaign", err)
                self.assertFalse(output.exists())

    def test_the_test_side_is_never_opened_by_a_standalone_evaluation(self):
        """Even with the evaluation directory present beside the prepared data, no test file is read."""
        opened: list[str] = []
        real_open = Path.open
        real_read_text = Path.read_text
        real_read_bytes = Path.read_bytes

        def watch(path):
            if "evaluation" in Path(path).parts and "bundle" not in Path(path).parts:
                opened.append(str(path))

        def spy_open(self, *a, **k):
            watch(self)
            return real_open(self, *a, **k)

        def spy_text(self, *a, **k):
            watch(self)
            return real_read_text(self, *a, **k)

        def spy_bytes(self, *a, **k):
            watch(self)
            return real_read_bytes(self, *a, **k)

        with mock.patch.object(Path, "open", spy_open), mock.patch.object(Path, "read_text", spy_text), mock.patch.object(Path, "read_bytes", spy_bytes):
            (code, _, err), _ = self.evaluate("--checkpoint", self.checkpoint(), mode="rerank")
        self.assertEqual(code, 0, err)
        self.assertEqual(opened, [])

    def test_other_training_visible_splits_use_regenerated_deterministic_candidates(self):
        (code, _, err), path = self.evaluate(split="train", name="train.json")
        self.assertEqual((code, err), (0, ""), err)
        report = json.loads(path.read_text())
        self.assertIn("regenerated", report["candidate_source"])
        (_, _, _), again = self.evaluate(split="train", name="train-again.json")
        self.assertEqual(report["per_query"], json.loads(again.read_text())["per_query"])

    def test_a_split_the_prepared_data_does_not_hold_and_a_misplaced_index_fail_cleanly(self):
        (code, _, err), path = self.evaluate(split="development")
        self.assertEqual(code, 1)
        self.assertIn("no development queries", err)
        self.assertFalse(path.exists())
        code, _, err = call(evaluate, "--config", self.model_config, "--mode", "baseline", "--index", self.root / "corpus-in" / "corpus-manifest.json",
                            "--output", self.out_dir / "x.json", "--runtime-root", self.root)
        self.assertEqual(code, 1)
        self.assertIn("--index must be", err)

    def test_rerank_without_a_checkpoint_fails_and_baseline_needs_no_encoder_assets(self):
        (code, _, err), path = self.evaluate(mode="rerank")
        self.assertEqual(code, 1)
        self.assertIn("--checkpoint", err)
        self.assertFalse(path.exists())
        bare = self.out_dir / "bare.toml"
        bare.write_text("schema_version = 1\n\n[device]\nprofile = \"cpu-diagnostic\"\n\n[model]\nbatch_size = 4\n")
        code, _, err = call(evaluate, "--config", bare, "--mode", "baseline", "--index", self.prepared / "index" / "index-manifest.json",
                            "--output", self.out_dir / "bare.json", "--runtime-root", self.root)
        self.assertEqual((code, err), (0, ""), err)


if __name__ == "__main__":
    unittest.main()
