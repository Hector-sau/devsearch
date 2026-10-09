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
                     "ranking_policy": {"method": "rank_fusion", "alpha": .5, "rank_constant": 60}, "candidate_k": 50, "batch_size": 8}
        self.questions = [{"query_id": "q-new", "group_id": "g-new", "review_status": "accepted", "reviewer_type": "agent"}]

    def write(self, name, obj):
        path = self.root / name
        path.write_text(json.dumps(obj))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def seal(self):
        plan_sha = self.write("frozen-evaluation-plan.json", self.plan)
        self.write("questions-seal.json", {"kind": "new_r5_holdout_questions_seal", "status": "sealed_before_ranking",
                                           "frozen_evaluation_plan_sha256": plan_sha,
                                           "questions_sha256": self.write("questions-sealed.json", self.questions), "queries": len(self.questions)})
        return plan_sha

    def test_valid_frozen_inputs_are_read_without_model_or_network(self):
        plan, seal, questions = runner.frozen_inputs(self.root, self.seal())
        self.assertEqual(plan, self.plan)
        self.assertEqual(questions, self.questions)

    def test_modified_plan_and_postseal_question_change_are_refused(self):
        sha = self.seal()
        self.write("frozen-evaluation-plan.json", {**self.plan, "deployment_seed": 44})
        with self.assertRaisesRegex(ManifestError, "plan changed"):
            runner.frozen_inputs(self.root, sha)
        sha = self.seal()
        self.write("questions-sealed.json", [{**self.questions[0], "review_status": "unreviewed"}])
        with self.assertRaisesRegex(ManifestError, "seal does not match"):
            runner.frozen_inputs(self.root, sha)

    def test_different_model_policy_and_incomplete_review_cannot_enter_acceptance(self):
        for name, value in (("deployment_seed", 44), ("candidate_k", 100), ("batch_size", 4),
                            ("ranking_policy", {"method": "logit"})):
            original = self.plan[name]
            self.plan[name] = value
            with self.subTest(name=name), self.assertRaises(ManifestError):
                runner.frozen_inputs(self.root, self.seal())
            self.plan[name] = original
        self.questions[0]["review_status"] = "unreviewed"
        with self.assertRaisesRegex(ManifestError, "coverage is incomplete"):
            runner.frozen_inputs(self.root, self.seal())

    def test_duplicate_questions_or_groups_are_refused_even_with_valid_hashes(self):
        self.questions *= 2
        with self.assertRaises(ManifestError):
            runner.frozen_inputs(self.root, self.seal())
        self.questions[1] = {**self.questions[0], "query_id": "q-other"}
        with self.assertRaises(ManifestError):
            runner.frozen_inputs(self.root, self.seal())

    @requires_serve
    def test_accelerator_profile_is_refused_before_service_start_or_output_creation(self):
        args = SimpleNamespace(config=self.root / "serve.toml", output=self.root / "output", runtime_root=self.root)
        with mock.patch("devsearch.execution.serving.ServeSettings.load", return_value=SimpleNamespace(job=SimpleNamespace(profile="nvidia-cuda"))), \
                mock.patch("devsearch.execution.serving.start_service") as start:
            with self.assertRaisesRegex(ManifestError, "refuses every accelerator"):
                asyncio.run(runner.exercise(args, {}, {}, []))
            start.assert_not_called()
        self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
