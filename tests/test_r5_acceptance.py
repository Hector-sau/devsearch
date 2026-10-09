"""New holdout runner refuses altered preregistration/questions and accelerators."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from devsearch.errors import ManifestError
from devsearch.evaluation.fusion import RankingPolicy
from helpers import REPO
from mltest import requires_serve

spec = importlib.util.spec_from_file_location("r5_acceptance", REPO / "scripts/r5_cpu_acceptance.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FrozenInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {"kind": "r5-independent-evaluation-preregistration", "deployment_seed": 42,
                     "checkpoint": {"checkpoint_sha256": "74b7fd525d75a0aab2a126e49eb759704cd0cf51a2c955122db938d4cd9f2a3c"},
                     "ranking_policy": {"method": "rank_fusion", "alpha": .5, "rank_constant": 60}, "candidate_k": 50, "batch_size": 8,
                     "policy": {"max_pair_tokens": 256, "max_query_tokens": 64, "truncation": False}}
        self.questions = [{"query_id": "q-new", "group_id": "g-new", "review_status": "accepted", "reviewer_type": "agent"}]

    def write(self, name, obj):
        path = self.root / name
        path.write_text(json.dumps(obj))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def seal(self):
        plan_sha = self.write("frozen-evaluation-plan.json", self.plan)
        questions_sha = self.write("questions-sealed.json", self.questions)
        seal_sha = self.write("questions-seal.json", {"kind": "new_r5_holdout_questions_seal", "status": "sealed_before_ranking",
                                                     "frozen_evaluation_plan_sha256": plan_sha,
                                                     "questions_sha256": questions_sha, "queries": len(self.questions)})
        self.binding_sha = self.write("input-binding.json", {
            "schema_version": 1, "kind": "r5_holdout_input_binding", "frozen_evaluation_plan_sha256": plan_sha,
            "questions_seal_sha256": seal_sha, "questions_sha256": questions_sha,
            "policy": {"max_pair_tokens": 256, "max_query_tokens": 64},
            "corpus_id": "corpus-frozen", "index_id": "index-frozen", "tokenizer_fingerprint": "tokenizer-frozen"})
        return plan_sha

    def frozen(self, plan_sha):
        return runner.frozen_inputs(self.root, plan_sha, self.binding_sha)

    def test_valid_frozen_inputs_are_read_without_model_or_network(self):
        plan, seal, questions, binding = self.frozen(self.seal())
        self.assertEqual(plan, self.plan)
        self.assertEqual(questions, self.questions)

    def test_modified_plan_and_postseal_question_change_are_refused(self):
        sha = self.seal()
        self.write("frozen-evaluation-plan.json", {**self.plan, "deployment_seed": 44})
        with self.assertRaisesRegex(ManifestError, "plan changed"):
            self.frozen(sha)
        sha = self.seal()
        self.write("questions-sealed.json", [{**self.questions[0], "review_status": "unreviewed"}])
        with self.assertRaisesRegex(ManifestError, "seal does not match"):
            self.frozen(sha)

    def test_replacing_questions_and_their_seal_cannot_preserve_the_frozen_plan(self):
        sha = self.seal()
        altered_questions = [{**self.questions[0], "text": "A replacement chosen after seeing rankings"}]
        altered_seal = json.loads((self.root / "questions-seal.json").read_text())
        altered_seal["questions_sha256"] = self.write("questions-sealed.json", altered_questions)
        self.write("questions-seal.json", altered_seal)
        with self.assertRaisesRegex(ManifestError, "fixed input binding"):
            self.frozen(sha)
        binding = json.loads((self.root / "input-binding.json").read_text())
        binding["questions_sha256"] = altered_seal["questions_sha256"]
        binding["questions_seal_sha256"] = hashlib.sha256((self.root / "questions-seal.json").read_bytes()).hexdigest()
        self.write("input-binding.json", binding)
        with self.assertRaisesRegex(ManifestError, "input binding changed"):
            self.frozen(sha)

    def test_different_model_policy_and_incomplete_review_cannot_enter_acceptance(self):
        for name, value in (("deployment_seed", 44), ("candidate_k", 100), ("batch_size", 4),
                            ("ranking_policy", {"method": "logit"})):
            original = self.plan[name]
            self.plan[name] = value
            with self.subTest(name=name), self.assertRaises(ManifestError):
                self.frozen(self.seal())
            self.plan[name] = original
        self.questions[0]["review_status"] = "unreviewed"
        with self.assertRaisesRegex(ManifestError, "coverage is incomplete"):
            self.frozen(self.seal())

    def test_duplicate_questions_or_groups_are_refused_even_with_valid_hashes(self):
        self.questions *= 2
        with self.assertRaises(ManifestError):
            self.frozen(self.seal())
        self.questions[1] = {**self.questions[0], "query_id": "q-other"}
        with self.assertRaises(ManifestError):
            self.frozen(self.seal())

    @requires_serve
    def test_accelerator_profile_is_refused_before_service_start_or_output_creation(self):
        args = SimpleNamespace(config=self.root / "serve.toml", output=self.root / "output", runtime_root=self.root)
        with mock.patch("devsearch.execution.serving.ServeSettings.load", return_value=SimpleNamespace(job=SimpleNamespace(profile="nvidia-cuda"))), \
                mock.patch("devsearch.execution.serving.start_service") as start:
            with self.assertRaisesRegex(ManifestError, "refuses every accelerator"):
                asyncio.run(runner.exercise(args, {}, {}, [], {}))
            start.assert_not_called()
        self.assertFalse(args.output.exists())


class LoadedIdentityTests(unittest.TestCase):
    def setUp(self):
        checkpoint = "checkpoint-frozen"
        self.plan = {"checkpoint": {"checkpoint_sha256": checkpoint}, "ranking_policy": RankingPolicy("rank_fusion").to_record(),
                     "batch_size": 8}
        self.binding = {"corpus_id": "corpus-frozen", "index_id": "index-frozen", "tokenizer_fingerprint": "tokenizer-frozen",
                        "policy": {"max_pair_tokens": 256, "max_query_tokens": 64}}
        self.loaded_model = SimpleNamespace(policy=SimpleNamespace(max_pair_tokens=256, max_query_tokens=64),
                                            tokenizer=SimpleNamespace(fingerprint="tokenizer-frozen"),
                                            manifest={"provenance": {"checkpoint_sha256": checkpoint}})
        self.pipeline = SimpleNamespace(loaded=SimpleNamespace(corpus=SimpleNamespace(corpus_id="corpus-frozen"), index_id="index-frozen"),
                                        ranking=RankingPolicy("rank_fusion"),
                                        reranker=SimpleNamespace(loaded=self.loaded_model, device=SimpleNamespace(type="cpu"), batch_size=8))
        self.started = SimpleNamespace(service=SimpleNamespace(pipeline=self.pipeline), settings=SimpleNamespace(job=SimpleNamespace(batch_size=8)),
                                       release={"provenance": {"model_source": {"checkpoint_sha256": checkpoint}}})

    def test_actual_loaded_inputs_match_the_fixed_binding(self):
        result = runner.verify_loaded_identity(self.started, self.plan, self.binding)
        self.assertEqual(result["policy"], self.binding["policy"])
        self.assertEqual(result["index_id"], "index-frozen")

    def test_another_corpus_or_index_with_same_checkpoint_is_refused(self):
        for obj, field in ((self.pipeline.loaded.corpus, "corpus_id"), (self.pipeline.loaded, "index_id")):
            original = getattr(obj, field)
            setattr(obj, field, "different")
            with self.subTest(field=field), self.assertRaisesRegex(ManifestError, "corpus or index"):
                runner.verify_loaded_identity(self.started, self.plan, self.binding)
            setattr(obj, field, original)

    def test_changed_tokenizer_or_text_limits_with_same_checkpoint_is_refused(self):
        for obj, field, value in ((self.loaded_model.tokenizer, "fingerprint", "other"),
                                  (self.loaded_model.policy, "max_pair_tokens", 128),
                                  (self.loaded_model.policy, "max_query_tokens", 32)):
            original = getattr(obj, field)
            setattr(obj, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ManifestError, "encoding policy or tokenizer"):
                runner.verify_loaded_identity(self.started, self.plan, self.binding)
            setattr(obj, field, original)

    def test_inner_checkpoint_cannot_disagree_with_release_provenance(self):
        self.loaded_model.manifest["provenance"]["checkpoint_sha256"] = "another-checkpoint"
        with self.assertRaisesRegex(ManifestError, "inner model checkpoint"):
            runner.verify_loaded_identity(self.started, self.plan, self.binding)

    def test_actual_accelerator_batch_and_absent_model_are_refused(self):
        self.pipeline.reranker.device.type = "cuda"
        with self.assertRaisesRegex(ManifestError, "actual CPU model"):
            runner.verify_loaded_identity(self.started, self.plan, self.binding)
        self.pipeline.reranker.device.type = "cpu"
        self.pipeline.reranker.batch_size = 4
        with self.assertRaisesRegex(ManifestError, "inference batch changed"):
            runner.verify_loaded_identity(self.started, self.plan, self.binding)
        self.pipeline.reranker = None
        with self.assertRaisesRegex(ManifestError, "actual CPU model"):
            runner.verify_loaded_identity(self.started, self.plan, self.binding)


class HardwareReceiptTests(unittest.TestCase):
    def test_container_limits_remain_distinct_from_visible_host_resources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "cpuinfo").write_text("processor : 0\nmodel name : Test CPU\n")
            (root / "memory.max").write_text("2147483648\n")
            (root / "cpu.max").write_text("50000 100000\n")
            result = runner.cpu_environment(root / "cpuinfo", root)
            self.assertEqual(result["cpu_model_from_proc"], "Test CPU")
            self.assertEqual(result["cgroup_mount_root_memory_max"], "2147483648")
            self.assertEqual(result["cgroup_mount_root_cpu_max"], "50000 100000")
            self.assertIsNone(result["effective_container_limits"])
            self.assertIn("not the process/container quota", result["limits_note"])
            missing = runner.cpu_environment(root / "missing", root / "missing")
            self.assertIsNone(missing["cpu_model_from_proc"])
            self.assertIsNone(missing["cgroup_mount_root_memory_max"])


if __name__ == "__main__":
    unittest.main()
