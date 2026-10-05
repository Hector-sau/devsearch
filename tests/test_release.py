"""Release bundles: assembly, activation rules, tamper detection, immutability, smoke parity, no silent downgrade."""

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch.contracts import SearchRequest
from devsearch.errors import IndexUnavailableError, ManifestError
from devsearch.evaluation.protocol import load_protocol
from devsearch.evaluation.ranking import order_by_score, ranked_record
from devsearch.evaluation.report import build_final_report
from devsearch.evaluation.selection import EpochResult, RunResult, select
from devsearch.retrieval.index_store import build_index, load_index
from devsearch.serving import release as release_module
from devsearch.serving.pipeline import SearchPipeline
from devsearch.serving.release import (
    MANIFEST_NAME, check_smoke, export_release, load_release, read_smoke, record_smoke, verify_release,
)
from mltest import HAVE_ML

from helpers import CONFIGS, RootTestCase, index_config, search_limits

if HAVE_ML:
    import torch
    from devsearch.modeling.io import export_model, load_model_bundle
    from devsearch.modeling.rerank import Reranker
    from mlworld import MLWorld

QUERIES = [{"query": "How do I save and load a checkpoint state_dict?"}, {"query": "What does gradient accumulation do?"},
           {"query": "tokenizer padding batch", "sources": []}]


class ReleaseCase(RootTestCase):
    def setUp(self):
        super().setUp()
        build_index(index_config(), self.corpus_manifest, self.index_manifest, self.root)
        self.loaded = load_index(self.index_manifest, self.root)
        self.limits = search_limits(candidate_k=10, top_k=5)
        self.search_toml = self.root / "search.toml"
        self.search_toml.write_text((CONFIGS / "search.toml").read_text().replace("candidate_k = 50", "candidate_k = 10"))
        self.out = Path(tempfile.mkdtemp(prefix="devsearch-rel-")).resolve()
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        from devsearch.retrieval.config import load_search_config
        self.config = load_search_config(self.search_toml)

    def smoke(self, reranker=None):
        pipeline = SearchPipeline(self.loaded, self.config, reranker, release_id="x")
        return record_smoke(pipeline, QUERIES, self.config)

    def export(self, name="rel", *, status="baseline", default_mode="baseline", model=None, report=None, smoke=None):
        return export_release(self.out / name, status=status, default_mode=default_mode, corpus_dir=self.root / "corpus", index_dir=self.root / "index",
                              model_dir=model, search_config_path=self.search_toml, report_dir=report, smoke=smoke or self.smoke(), provenance={"purpose": "test"})


class BaselineReleaseTests(ReleaseCase):
    def test_a_baseline_release_verifies_loads_and_reproduces_its_smoke_results_without_ml(self):
        manifest = self.export()
        self.assertEqual((manifest["status"], manifest["default_mode"], manifest["model"], manifest["evaluation"]), ("baseline", "baseline", None, None))
        self.assertTrue(manifest["release_id"].startswith("release-"))
        verify_release(self.out / "rel", allow_candidate=False)
        loaded = load_release(self.out / "rel", allow_candidate=False)
        self.assertEqual(loaded.release_id, manifest["release_id"])
        self.assertTrue(loaded.pipeline.baseline_ready)
        self.assertFalse(loaded.pipeline.rerank_ready)
        report = check_smoke(loaded.pipeline, read_smoke(self.out / "rel"), loaded.search_config)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["queries"], 3)
        document = loaded.pipeline.search(SearchRequest(query=QUERIES[0]["query"], mode="baseline", candidate_k=10, top_k=5))
        self.assertEqual(document["release_id"], manifest["release_id"])

    def test_the_manifest_lists_every_file_and_holds_no_absolute_paths(self):
        manifest = self.export()
        names = {p.relative_to(self.out / "rel").as_posix() for p in (self.out / "rel").rglob("*") if p.is_file()}
        self.assertEqual(names, set(manifest["files"]) | {MANIFEST_NAME})
        self.assertNotIn(str(self.out), (self.out / "rel" / MANIFEST_NAME).read_text())
        self.assertNotIn(str(self.root), json.dumps(manifest["provenance"]))

    def test_any_changed_added_or_removed_file_is_detected(self):
        self.export()
        release = self.out / "rel"
        for relative in ("index/index-manifest.json", "config/search.toml", "smoke/smoke.json"):
            path = release / relative
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(change=relative), self.assertRaisesRegex(ManifestError, "differ"):
                verify_release(release, allow_candidate=False)
            path.write_bytes(original)
        (release / "extra.txt").write_text("x")
        with self.assertRaisesRegex(ManifestError, "extra.txt"):
            verify_release(release, allow_candidate=False)
        (release / "extra.txt").unlink()
        (release / "smoke" / "smoke.json").unlink()
        with self.assertRaisesRegex(ManifestError, "smoke"):
            verify_release(release, allow_candidate=False)

    def test_symlinks_and_an_edited_manifest_are_rejected(self):
        self.export()
        release = self.out / "rel"
        outside = self.out / "outside.txt"
        outside.write_text("x")
        (release / "link").symlink_to(outside)
        with self.assertRaisesRegex(ManifestError, "symbolic"):
            verify_release(release, allow_candidate=False)
        (release / "link").unlink()
        manifest_path = release / MANIFEST_NAME
        data = json.loads(manifest_path.read_text())
        data["default_mode"] = "rerank"
        manifest_path.write_text(json.dumps(data))
        with self.assertRaises(ManifestError):
            verify_release(release, allow_candidate=False)

    def test_status_rules_are_enforced_when_exporting(self):
        with self.assertRaisesRegex(ManifestError, "no model"):
            self.export(default_mode="rerank")
        with self.assertRaisesRegex(ManifestError, "model"):
            self.export("c", status="candidate")
        with self.assertRaisesRegex(ManifestError, "final report"):
            self.export("e", status="evaluated")
        with self.assertRaisesRegex(ManifestError, "status must be"):
            self.export("x", status="released")
        self.assertEqual(list(self.out.iterdir()), [], "a refused export leaves nothing behind")

    def test_a_crash_before_the_final_rename_leaves_no_release_and_is_retryable(self):
        with mock.patch.object(release_module.os, "rename", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.export()
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(self.export()["status"], "baseline")

    def test_an_identical_reexport_is_idempotent_and_a_different_one_is_refused(self):
        first = self.export()
        self.assertEqual(self.export()["manifest_sha256"], first["manifest_sha256"])
        different = self.smoke()
        different["queries"][0]["baseline_ids"] = list(reversed(different["queries"][0]["baseline_ids"]))
        with self.assertRaisesRegex(ManifestError, "immutable"):
            self.export(smoke=different)

    def test_smoke_checks_catch_a_changed_corpus_ordering(self):
        self.export()
        loaded = load_release(self.out / "rel", allow_candidate=False)
        smoke = read_smoke(self.out / "rel")
        smoke["queries"][0]["baseline_ids"] = smoke["queries"][0]["baseline_ids"][::-1]
        report = check_smoke(loaded.pipeline, smoke, loaded.search_config)
        self.assertFalse(report["passed"])
        self.assertIn("BM25 top results changed", report["failures"][0])
        smoke["queries"][0]["candidate_set_id"] = "cands-other"
        self.assertIn("candidate set changed", " ".join(check_smoke(loaded.pipeline, smoke, loaded.search_config)["failures"]))

    def test_a_missing_parent_is_never_created(self):
        with self.assertRaisesRegex(ManifestError, "never creates parent"):
            export_release(self.out / "nowhere" / "rel", status="baseline", default_mode="baseline", corpus_dir=self.root / "corpus",
                           index_dir=self.root / "index", model_dir=None, search_config_path=self.search_toml, report_dir=None, smoke=self.smoke(), provenance={})


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class ModelReleaseTests(ReleaseCase):
    CHECKPOINT_SHA = "ab" * 32

    def setUp(self):
        super().setUp()
        self.w = MLWorld.build_ml(self)
        self.model = self.w.model(encoder_seed=2, head_seed=9)
        self.model_dir = self.out / "model-src"
        examples = [(p.query_text, p.passage_text) for p in self.w.train_pairs[:4]]
        self.model_manifest = export_model(self.model, self.model_dir, tokenizer=self.w.tokenizer, policy=self.w.policy,
                                           provenance={"checkpoint_sha256": self.CHECKPOINT_SHA, "purpose": "test"}, parity_examples=examples,
                                           device=torch.device("cpu"))
        # the ML world builds its own corpus/index; the release must be built from that same pair
        self.root, self.loaded = self.w.root, self.w.loaded
        self.search_toml = self.w.root / "search.toml"
        self.search_toml.write_text((CONFIGS / "search.toml").read_text().replace("candidate_k = 50", "candidate_k = 10"))
        from devsearch.retrieval.config import load_search_config
        self.config = load_search_config(self.search_toml)

    def factory(self, path):
        return Reranker(load_model_bundle(path, device=torch.device("cpu")), device=torch.device("cpu"), batch_size=4)

    def model_smoke(self):
        reranker = self.factory(self.model_dir)
        return self.smoke(reranker)

    def report_dir(self, selected_sha=None, default="rerank"):
        protocol = dataclasses.replace(load_protocol(CONFIGS / "protocol.toml"), bootstrap_repetitions=200)
        sha = selected_sha or self.CHECKPOINT_SHA
        weak = tuple(EpochResult(i, 0.2, 0.3, f"epoch-{i:04d}", f"{i:064x}") for i in (1, 2, 3))
        strong = tuple(EpochResult(i, 0.9, 0.9, f"epoch-{i:04d}", sha) for i in (1, 2, 3))
        runs = {("frozen", 42): RunResult("frozen", 42, weak), ("full", 42): RunResult("full", 42, strong)}
        if default == "baseline":
            runs[("full", 42)] = RunResult("full", 42, weak)
        selection = select(protocol, baseline={"group_mrr": 0.2, "group_ndcg": 0.3}, runs=runs, latency_p95_ms={"frozen": 1.0, "full": 1.0}, plan_id="plan-1")
        groups = {f"q{i}": f"g{(i - 1) // 2}" for i in range(1, 9)}
        from devsearch.data.queries import Qrel, Review
        qrels = [Qrel(q, "c1", 1, (), "pooled-review", "accepted", (Review("agent", "r", "accepted", "2026-10-05T12:00:00Z", "wf"),)) for q in sorted(groups)]

        def system(position):
            out = []
            for q in sorted(groups):
                ordered = ["c2"] * 0 + ["c3", "c4", "c5", "c6", "c7"]
                ordered = ordered[:position - 1] + ["c1"] + ordered[position - 1:]
                out.append(ranked_record(q, "cs", order_by_score(ordered, [float(len(ordered) - i) for i in range(len(ordered))])))
            return out
        built = build_final_report(
            plan_id="plan-1", protocol=protocol, selection=selection, systems={"bm25": system(3), "frozen-seed42": system(3), "full-seed42": system(1)},
            system_meta={"bm25": {"role": "baseline"}, "frozen-seed42": {"role": "locked", "recipe_id": "frozen", "seed": 42},
                         "full-seed42": {"role": "locked", "recipe_id": "full", "seed": 42}},
            qrels=qrels, group_of=groups, provenance={"query_origins": {"synthetic": 8}, "reviewers": {"agent": 8, "human": 0}}, seals={"initial": "a" * 64})
        directory = Path(tempfile.mkdtemp(prefix=f"report-{default}-", dir=self.out))
        (directory / "final-report.json").write_text(json.dumps(built.report))
        (directory / "per-query.jsonl").write_bytes(built.per_query)
        (directory / "report.md").write_text(built.markdown)
        return directory

    def export_model_release(self, name="rel", **kw):
        return export_release(self.out / name, corpus_dir=self.w.root / "corpus", index_dir=self.w.root / "index", search_config_path=self.search_toml,
                              provenance={"purpose": "test"}, smoke=kw.pop("smoke", None) or self.model_smoke(), **kw)

    def test_a_candidate_needs_allow_candidate_and_ships_a_verified_reranker(self):
        manifest = self.export_model_release(status="candidate", default_mode="rerank", model_dir=self.model_dir, report_dir=None)
        self.assertEqual(manifest["model"]["model_id"], self.model_manifest["model_id"])
        with self.assertRaisesRegex(IndexUnavailableError, "allow-candidate"):
            load_release(self.out / "rel", allow_candidate=False, reranker_factory=self.factory)
        loaded = load_release(self.out / "rel", allow_candidate=True, reranker_factory=self.factory)
        self.assertTrue(loaded.pipeline.rerank_ready)
        self.assertTrue(loaded.pipeline.self_check())
        report = check_smoke(loaded.pipeline, read_smoke(self.out / "rel"), loaded.search_config)
        self.assertTrue(report["passed"], report)
        self.assertGreater(report["rerank_checked"], 0)

    def test_a_model_release_without_a_loader_is_an_error_not_a_silent_baseline(self):
        self.export_model_release(status="candidate", default_mode="rerank", model_dir=self.model_dir, report_dir=None)
        with self.assertRaisesRegex(ManifestError, "ml"):
            load_release(self.out / "rel", allow_candidate=True)

    def test_a_damaged_model_blocks_activation_instead_of_falling_back_to_baseline(self):
        self.export_model_release(status="candidate", default_mode="rerank", model_dir=self.model_dir, report_dir=None)
        weights = self.out / "rel" / "model" / "encoder.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1] + b"\x00")
        with self.assertRaisesRegex(ManifestError, "encoder.safetensors"):
            load_release(self.out / "rel", allow_candidate=True, reranker_factory=self.factory)

    def test_the_loaded_reranker_must_be_the_model_the_manifest_names(self):
        self.export_model_release(status="candidate", default_mode="rerank", model_dir=self.model_dir, report_dir=None)
        wrong = mock.Mock()
        wrong.model_id = "model-someone-else"
        with self.assertRaisesRegex(ManifestError, "not the model named"):
            load_release(self.out / "rel", allow_candidate=True, reranker_factory=lambda path: wrong)

    def test_an_evaluated_release_binds_its_report_to_the_shipped_checkpoint(self):
        manifest = self.export_model_release(status="evaluated", default_mode="rerank", model_dir=self.model_dir, report_dir=self.report_dir())
        self.assertEqual(manifest["evaluation"]["selection_id"], json.loads((self.out / "rel" / "evaluation" / "final-report.json").read_text())["selection_id"])
        loaded = load_release(self.out / "rel", allow_candidate=False, reranker_factory=self.factory)
        self.assertEqual(loaded.manifest["status"], "evaluated")
        self.assertTrue(loaded.pipeline.rerank_ready)
        with self.assertRaisesRegex(ManifestError, "not the checkpoint the final report selected"):
            self.export_model_release("wrong", status="evaluated", default_mode="rerank", model_dir=self.model_dir, report_dir=self.report_dir("cd" * 32))

    def test_a_report_that_chose_baseline_cannot_ship_as_a_reranking_release(self):
        with self.assertRaisesRegex(ManifestError, "chose 'baseline'"):
            self.export_model_release(status="evaluated", default_mode="rerank", model_dir=self.model_dir, report_dir=self.report_dir(default="baseline"))
        manifest = self.export_model_release("honest", status="evaluated", default_mode="baseline", model_dir=None, report_dir=self.report_dir(default="baseline"),
                                             smoke=self.smoke())
        self.assertEqual((manifest["default_mode"], manifest["model"]), ("baseline", None))

    def test_an_edited_final_report_blocks_the_release(self):
        self.export_model_release(status="evaluated", default_mode="rerank", model_dir=self.model_dir, report_dir=self.report_dir())
        report_path = self.out / "rel" / "evaluation" / "final-report.json"
        data = json.loads(report_path.read_text())
        data["systems"][0]["group_macro"]["mrr@10"] = 0.999
        report_path.write_text(json.dumps(data))
        with self.assertRaises(ManifestError):
            verify_release(self.out / "rel", allow_candidate=True)


if __name__ == "__main__":
    unittest.main()
