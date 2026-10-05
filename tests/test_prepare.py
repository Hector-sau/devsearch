"""Preparation stages: group assignment/audit, materialization, anti-leak separation, bank and seal."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch.artifacts import files as artifact_files
from devsearch.artifacts.files import new_staging, publish_staged
from devsearch.data.candidates import load_candidates
from devsearch.data.config import load_data_config
from devsearch.data.corpus import load_corpus
from devsearch.data.prepare import (
    EVALUATION_MANIFEST, INITIAL_SEAL, PREPARED_MANIFEST, SPLIT_MANIFEST, load_evaluation, load_prepared,
    prepare_bank, prepare_groups, prepare_initial_seal_postrank, prepare_materialize,
)
from devsearch.data.review import build_review_pool, prejudged_universe
from devsearch.data.seals import load_seal
from devsearch.errors import ConfigError, ManifestError

import prepworld
from prepworld import draft_for, drafts_from_split, make_root, read_json, stage_groups, write_config, write_drafts


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-pub-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def stage(self, data=b"data", manifest=b'{"m": 1}'):
        staging = new_staging(self.dir)
        (staging / "a.bin").write_bytes(data)
        (staging / "m.json").write_bytes(manifest)
        return staging

    def test_a_publish_moves_data_first_and_the_manifest_last(self):
        order = []
        real = os.rename
        with mock.patch.object(artifact_files.os, "rename", side_effect=lambda a, b: (order.append(Path(b).name), real(a, b))[1]), \
             mock.patch.object(artifact_files.os, "replace", side_effect=lambda a, b: (order.append(Path(b).name), os.replace.__wrapped__(a, b) if hasattr(os.replace, "__wrapped__") else real(a, b))[1]):
            self.assertTrue(publish_staged(self.stage(), self.dir, manifests=["m.json"]))
        self.assertEqual(order[-1], "m.json")
        self.assertEqual({p.name for p in self.dir.iterdir()}, {"a.bin", "m.json"})

    def test_a_crash_before_the_manifest_leaves_nothing_visible_and_a_retry_succeeds(self):
        with mock.patch.object(artifact_files.os, "replace", side_effect=OSError("crash")):
            with self.assertRaises(OSError):
                publish_staged(self.stage(), self.dir, manifests=["m.json"])
        self.assertFalse((self.dir / "m.json").exists())
        self.assertTrue(publish_staged(self.stage(), self.dir, manifests=["m.json"]))
        self.assertEqual((self.dir / "a.bin").read_bytes(), b"data")
        self.assertTrue(any(p.name.startswith(".abandoned-") for p in self.dir.iterdir()), "the remnant was moved aside, not deleted")

    def test_an_identical_republish_is_a_noop_and_a_different_one_is_refused(self):
        publish_staged(self.stage(), self.dir, manifests=["m.json"])
        self.assertFalse(publish_staged(self.stage(), self.dir, manifests=["m.json"]))
        with self.assertRaisesRegex(ManifestError, "immutable"):
            publish_staged(self.stage(manifest=b'{"m": 2}'), self.dir, manifests=["m.json"])
        self.assertEqual((self.dir / "m.json").read_bytes(), b'{"m": 1}')

    def test_every_named_manifest_must_be_staged_and_the_destination_must_exist(self):
        staging = self.stage()
        with self.assertRaisesRegex(ManifestError, "staged"):
            publish_staged(staging, self.dir, manifests=["missing.json"])
        with self.assertRaisesRegex(ManifestError, "must both exist"):
            publish_staged(staging, self.dir / "nowhere", manifests=["m.json"])


class DataConfigTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-cfg-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_a_valid_config_resolves_paths_and_records_its_hash(self):
        config = load_data_config(write_config(self.dir))
        self.assertEqual(dict(config.queries)["test"], self.dir / "queries-test.jsonl")
        self.assertEqual(set(dict(config.qrels)), {"train", "validation", "test"})
        self.assertEqual(config.target_map, {"test": 0.25, "train": 0.5, "validation": 0.25})
        self.assertEqual(config.candidate_depth, 50)
        self.assertEqual(len(config.config_sha256), 64)

    def test_unknown_missing_and_bad_values_fail(self):
        good = write_config(self.dir).read_text()
        cases = {
            "unknown key": good.replace("[candidates]", "[candidates]\nextra = 1"),
            "missing section": good.replace("[review]", "[reviews]"),
            "short salt": good.replace("salt-for-the-test-suite", "short"),
            "unknown protocol": good.replace("prejudged_fixed_candidates_v1", "some_other_protocol"),
            "bad depth": good.replace("depth = 50", "depth = 500"),
            "missing split target": good.replace("test = 0.25\n", ""),
            "unknown split target": good.replace("test = 0.25", "holdout = 0.25"),
            "impossible mining": good.replace("hard_negatives = 2\nrandom_negatives = 2", "hard_negatives = 0\nrandom_negatives = 0"),
            "bool seed": good.replace("seed = 20261005", "seed = true"),
            "missing test file": good.replace('test = "queries-test.jsonl"\n', ""),
            "unknown split file": good.replace('test = "queries-test.jsonl"', 'test = "queries-test.jsonl"\nholdout = "x.jsonl"'),
        }
        for name, text in cases.items():
            path = self.dir / "bad.toml"
            path.write_text(text)
            with self.subTest(name), self.assertRaises(ConfigError):
                load_data_config(path)


class GroupStageTests(unittest.TestCase):
    def setUp(self):
        self.root = make_root(self)
        self.config_path = write_config(self.root / "drafts")

    def test_groups_are_merged_assigned_audited_and_published_with_an_assigned_corpus(self):
        config, manifest, split = stage_groups(self.root, self.config_path)
        original = load_corpus(self.root / "corpus-in" / "corpus-manifest.json")
        assigned = split["corpus"]
        self.assertEqual(manifest["owner_origin"], "assigned")
        self.assertNotEqual(assigned.corpus_id, original.corpus_id)
        self.assertEqual([c.chunk_id for c in assigned.chunks], [c.chunk_id for c in original.chunks])
        self.assertTrue(all(c.owner_split in ("train", "validation", "test") for c in assigned.chunks))
        self.assertEqual({c.owner_split for c in assigned.chunks}, {"train", "validation", "test"})
        self.assertEqual(sum(v["chunks"] for v in manifest["counts"].values()), len(original.chunks))
        audit = read_json(self.root / "prepared" / "audit.json")
        self.assertEqual((audit["ok"], audit["violations"]), (True, []))
        self.assertEqual(set(split["assignment"].values()), {"train", "validation", "test"})

    def test_a_rerun_is_idempotent_and_a_different_seed_cannot_overwrite(self):
        _, first, _ = stage_groups(self.root, self.config_path)
        _, again, _ = stage_groups(self.root, self.config_path)
        self.assertEqual(first["manifest_sha256"], again["manifest_sha256"])
        other = write_config(self.root / "drafts", seed=7)
        config = load_data_config(other)
        with self.assertRaisesRegex(ManifestError, "immutable"):
            prepare_groups(config, self.root / "corpus-in" / "corpus-manifest.json", self.root / "prepared", self.root)
        self.assertEqual(read_json(self.root / "prepared" / SPLIT_MANIFEST)["manifest_sha256"], first["manifest_sha256"])

    def test_the_assignment_is_deterministic_for_a_seed_and_depends_on_it(self):
        _, _, one = stage_groups(self.root, self.config_path)
        second_root = make_root(self)
        _, _, two = stage_groups(second_root, write_config(second_root / "drafts"))
        self.assertEqual(one["assignment"], two["assignment"])
        third_root = make_root(self)
        _, _, three = stage_groups(third_root, write_config(third_root / "drafts", seed=3))
        self.assertNotEqual(one["assignment"], three["assignment"])

    def test_an_already_assigned_corpus_is_adopted_when_consistent(self):
        _, first, split = stage_groups(self.root, self.config_path)
        again_root = make_root(self)
        shutil.rmtree(again_root / "corpus-in")
        shutil.copytree(split["corpus_path"].parent, again_root / "corpus-in")
        config, adopted, adopted_split = stage_groups(again_root, write_config(again_root / "drafts", seed=999))
        self.assertEqual(adopted["owner_origin"], "adopted")
        self.assertEqual(adopted["corpus"]["corpus_id"], first["corpus"]["corpus_id"])
        self.assertEqual(adopted_split["assignment"], split["assignment"])

    def test_a_partially_assigned_corpus_is_refused(self):
        _, _, split = stage_groups(self.root, self.config_path)
        corpus = split["corpus"]
        from devsearch.data.corpus import reowned_documents, write_corpus
        groups = sorted({c.evidence_group_id for c in corpus.chunks})
        assigned = set(groups[::2])                      # whole evidence groups, so the writer's one-owner-per-group rule is respected
        owners = {c.chunk_id: (c.owner_split if c.evidence_group_id in assigned else None) for c in corpus.chunks}
        partial = self.root / "partial"
        partial.mkdir()
        documents, version = reowned_documents(corpus, owners)
        write_corpus(partial, documents, chunker_version=version)
        fresh = self.root / "prepared2"
        fresh.mkdir()
        with self.assertRaisesRegex(ManifestError, "partially assigned"):
            prepare_groups(load_data_config(self.config_path), partial / "corpus-manifest.json", fresh, self.root)
        self.assertEqual(list(fresh.iterdir()), [])

    def test_a_merged_group_spanning_declared_owners_is_refused_with_its_name(self):
        _, _, split = stage_groups(self.root, self.config_path)
        corpus = split["corpus"]
        groups = {c.evidence_group_id: c for c in corpus.chunks}
        # force two different declared owners across two groups that an explicit link joins
        a, b = sorted(groups)[:2]
        owners = {c.chunk_id: ("train" if c.evidence_group_id == a else "test") for c in corpus.chunks}
        from devsearch.data.corpus import reowned_documents, write_corpus
        directory = self.root / "conflict"
        directory.mkdir()
        documents, version = reowned_documents(corpus, owners)
        write_corpus(directory, documents, chunker_version=version)
        links = self.root / "drafts" / "links.jsonl"
        links.write_text(json.dumps({"a": a, "b": b}) + "\n")
        config_path = write_config(self.root / "drafts", links="links.jsonl")
        fresh = self.root / "prepared3"
        fresh.mkdir()
        with self.assertRaisesRegex(ManifestError, "spans declared owners"):
            prepare_groups(load_data_config(config_path), directory / "corpus-manifest.json", fresh, self.root)

    def test_an_unknown_link_and_paths_outside_the_root_are_refused(self):
        links = self.root / "drafts" / "links.jsonl"
        links.write_text(json.dumps({"a": "eg-nope", "b": "eg-also-nope"}) + "\n")
        config = load_data_config(write_config(self.root / "drafts", links="links.jsonl"))
        with self.assertRaisesRegex(ManifestError, "unknown evidence group"):
            prepare_groups(config, self.root / "corpus-in" / "corpus-manifest.json", self.root / "prepared", self.root)
        outside = Path(tempfile.mkdtemp(prefix="devsearch-outside-")).resolve()
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        with self.assertRaises(ManifestError):
            prepare_groups(load_data_config(self.config_path), self.root / "corpus-in" / "corpus-manifest.json", outside, self.root)

    def test_a_missing_directory_is_never_created(self):
        with self.assertRaises(ManifestError):
            prepare_groups(load_data_config(self.config_path), self.root / "corpus-in" / "corpus-manifest.json", self.root / "no-such-dir", self.root)
        self.assertFalse((self.root / "no-such-dir").exists())


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.root = make_root(self)
        self.config_path = write_config(self.root / "drafts")
        self.config, _, self.split = stage_groups(self.root, self.config_path)
        self.queries, self.qrels = drafts_from_split(self.root, self.split)
        self.manifest = prepare_materialize(self.config, self.root / "prepared" / SPLIT_MANIFEST, self.root / "evaluation", self.root)

    def test_artifacts_are_written_and_verified(self):
        prepared = load_prepared(self.root / "prepared" / PREPARED_MANIFEST, self.root)
        self.assertEqual(prepared["manifest"]["manifest_sha256"], self.manifest["manifest_sha256"])
        names = {p.name for p in (self.root / "prepared" / "data").iterdir()}
        self.assertEqual(names, {"queries-train.jsonl", "qrels-train.jsonl", "queries-validation.jsonl", "qrels-validation.jsonl"})
        self.assertTrue((self.root / "prepared" / "pairs" / "pairs-manifest.json").is_file())
        self.assertTrue((self.root / "prepared" / "candidates" / "candidates-validation-manifest.json").is_file())
        self.assertTrue((self.root / "evaluation" / EVALUATION_MANIFEST).is_file())
        self.assertTrue((self.root / "evaluation" / "review" / "review-pool-manifest.json").is_file())

    def test_no_test_material_exists_anywhere_in_the_training_visible_directory(self):
        test_queries = [q for q in self.queries if q.split == "test"]
        test_qrels = [r for r in self.qrels if r.query_id in {q.query_id for q in test_queries}]
        self.assertTrue(test_queries and test_qrels)
        for path in (self.root / "prepared").rglob("*"):
            if not path.is_file() or path.relative_to(self.root / "prepared").parts[0] in ("corpus", "index"):
                continue            # the corpus and index are shared documents; they hold no question or judgment
            text = path.read_text(encoding="utf-8")
            for query in test_queries:
                self.assertNotIn(query.query_id, text, f"{path.name} leaks test query id {query.query_id}")
                self.assertNotIn(query.text, text, f"{path.name} leaks test question text")
        evaluation = (self.root / "evaluation" / "queries-test.jsonl").read_text()
        for query in test_queries:
            self.assertIn(query.query_id, evaluation)
        commitment = self.manifest["evaluation_commitment"]
        self.assertEqual(commitment["test_queries"], len(test_queries))

    def test_pairs_use_only_train_queries_and_a_train_only_pool_and_candidates_cover_validation(self):
        prepared = load_prepared(self.root / "prepared" / PREPARED_MANIFEST, self.root)
        loaded = prepared["loaded"]
        from devsearch.data.pairs import load_pairs
        _, pairs, pool = load_pairs(self.root / "prepared" / "pairs" / "pairs-manifest.json", loaded.corpus)
        train_ids = {q.query_id for q in self.queries if q.split == "train"}
        self.assertTrue(pairs)
        self.assertLessEqual({p.query_id for p in pairs}, train_ids)
        owners = {c.chunk_id: c.owner_split for c in loaded.corpus.chunks}
        self.assertEqual({owners[c] for c in pool.chunk_ids}, {"train"})
        _, records = load_candidates(self.root / "prepared" / "candidates" / "candidates-validation-manifest.json", loaded)
        self.assertEqual({r["query_id"] for r in records}, {q.query_id for q in self.queries if q.split == "validation"})

    def test_the_evaluation_inputs_verify_against_the_index(self):
        prepared = load_prepared(self.root / "prepared" / PREPARED_MANIFEST, self.root)
        evaluation = load_evaluation(self.root / "evaluation" / EVALUATION_MANIFEST, self.root, prepared["loaded"])
        self.assertEqual({q.query_id for q in evaluation["queries"]}, {q.query_id for q in self.queries if q.split == "test"})
        self.assertEqual(evaluation["manifest"]["test_groups_sha256"], prepared["manifest"]["evaluation_commitment"] and evaluation["manifest"]["test_groups_sha256"])
        self.assertEqual(prepared["manifest"]["evaluation_commitment"]["manifest_sha256"], evaluation["manifest"]["manifest_sha256"])

    def test_a_rerun_is_idempotent(self):
        again = prepare_materialize(self.config, self.root / "prepared" / SPLIT_MANIFEST, self.root / "evaluation", self.root)
        self.assertEqual(again["manifest_sha256"], self.manifest["manifest_sha256"])

    def test_tampering_with_a_prepared_file_is_detected(self):
        target = self.root / "prepared" / "data" / "qrels-train.jsonl"
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ManifestError, "qrels-train.jsonl"):
            load_prepared(self.root / "prepared" / PREPARED_MANIFEST, self.root)
        target.write_bytes(original)
        load_prepared(self.root / "prepared" / PREPARED_MANIFEST, self.root)
        pairs = self.root / "prepared" / "pairs" / "pairs.jsonl"
        data = pairs.read_bytes()
        pairs.write_bytes(data.replace(b'"label":1', b'"label":0', 1))
        with self.assertRaises(ManifestError):
            from devsearch.data.pairs import load_pairs
            load_pairs(self.root / "prepared" / "pairs" / "pairs-manifest.json", load_prepared(self.root / "prepared" / PREPARED_MANIFEST, self.root)["loaded"].corpus)


class MaterializeRejectionTests(unittest.TestCase):
    def setUp(self):
        self.root = make_root(self)
        self.config_path = write_config(self.root / "drafts")
        self.config, _, self.split = stage_groups(self.root, self.config_path)
        self.queries, self.qrels = draft_for(self.split["corpus"], self.split)

    def run_stage(self, queries=None, qrels=None, evaluation=None, **kwargs):
        write_drafts(self.root / "drafts", queries if queries is not None else self.queries, qrels if qrels is not None else self.qrels)
        return prepare_materialize(self.config, self.root / "prepared" / SPLIT_MANIFEST, evaluation or self.root / "evaluation", self.root, **kwargs)

    def test_the_evaluation_directory_must_be_separate_from_the_prepared_one(self):
        with self.assertRaisesRegex(ManifestError, "separate"):
            self.run_stage(evaluation=self.root / "prepared")
        nested = self.root / "prepared" / "inside"
        nested.mkdir()
        with self.assertRaisesRegex(ManifestError, "separate"):
            self.run_stage(evaluation=nested)

    def test_a_query_whose_split_disagrees_with_its_group_owner_is_refused(self):
        wrong = list(self.queries)
        original = wrong[0]
        other = "validation" if original.split != "validation" else "train"
        wrong[0] = prepworld.make_query(original.query_id, original.text, list(original.evidence_group_ids), other)
        with self.assertRaises(ManifestError):
            self.run_stage(queries=wrong)

    def test_a_split_without_queries_or_positives_is_refused(self):
        without_test = [q for q in self.queries if q.split != "test"]
        ids = {q.query_id for q in without_test}
        with self.assertRaisesRegex(ManifestError, "no test queries"):
            self.run_stage(queries=without_test, qrels=[r for r in self.qrels if r.query_id in ids])
        test_ids = {q.query_id for q in self.queries if q.split == "test"}
        with self.assertRaisesRegex(ManifestError, "no confirmed positive"):
            self.run_stage(qrels=[r for r in self.qrels if r.query_id not in test_ids])

    def test_duplicate_query_ids_and_dangling_qrels_are_refused(self):
        with self.assertRaises(ManifestError):
            self.run_stage(queries=self.queries + [self.queries[0]])
        with self.assertRaises(ManifestError):
            self.run_stage(qrels=self.qrels + [self.qrels[0]])

    def test_a_positive_owned_by_another_split_would_leak_and_is_refused(self):
        train_query = next(q for q in self.queries if q.split == "train")
        corpus = self.split["corpus"]
        foreign = next(c for c in corpus.chunks if c.owner_split == "test")
        doc = corpus.doc_of(foreign)
        from devsearch.data.queries import EvidenceSpan, Qrel
        leak = Qrel(train_query.query_id, foreign.chunk_id, 1, (EvidenceSpan(doc.doc_id, doc.revision, doc.content_sha256, foreign.start_char + 1, min(foreign.end_char, foreign.start_char + 20)),),
                    "synthetic", "accepted", prepworld.ACCEPTED)
        with self.assertRaises(ManifestError):
            self.run_stage(qrels=self.qrels + [leak])

    def test_a_separate_evaluation_root_is_supported_and_must_contain_its_directory(self):
        other = Path(tempfile.mkdtemp(prefix="devsearch-evalroot-")).resolve()
        self.addCleanup(shutil.rmtree, other, ignore_errors=True)
        (other / "eval").mkdir()
        manifest = self.run_stage(evaluation=other / "eval", evaluation_root=other)
        self.assertTrue((other / "eval" / EVALUATION_MANIFEST).is_file())
        self.assertFalse((self.root / "evaluation" / EVALUATION_MANIFEST).exists())
        self.assertEqual(manifest["kind"], "prepared-data")
        with self.assertRaises(ManifestError):
            self.run_stage(evaluation=self.root / "evaluation", evaluation_root=other)


class BankAndSealTests(unittest.TestCase):
    def setUp(self):
        self.root = make_root(self)
        self.config_path = write_config(self.root / "drafts")
        self.config, _, self.split = stage_groups(self.root, self.config_path)
        self.queries, self.qrels = drafts_from_split(self.root, self.split)
        prepare_materialize(self.config, self.root / "prepared" / SPLIT_MANIFEST, self.root / "evaluation", self.root)
        self.prepared = load_prepared(self.root / "prepared" / PREPARED_MANIFEST, self.root)
        self.loaded = self.prepared["loaded"]
        self.evaluation = load_evaluation(self.root / "evaluation" / EVALUATION_MANIFEST, self.root, self.loaded)
        _, self.candidates = load_candidates(self.root / "evaluation" / "candidates" / "candidates-test-manifest.json", self.loaded)
        universe = prejudged_universe(self.loaded, self.evaluation["queries"], self.candidates, self.evaluation["qrels"])
        self.pool = build_review_pool(self.config.protocol, self.config.salt, universe)
        self.positives = {(q.query_id, q.chunk_id) for q in self.evaluation["qrels"] if q.confirmed_positive}
        self.chunks = {c.chunk_id: c for c in self.loaded.corpus.chunks}

    def reviews(self, *, skip=0, flip=None):
        records = []
        for index, item in enumerate(self.pool.items):
            if index < skip:
                continue
            positive = (item.query_id, item.chunk_id) in self.positives
            value = 1 if positive else 0
            if flip == item.item_id:
                value = 1 - value
            records.append({
                "item_id": item.item_id, "role": "primary", "relevance": value, "status": "resolved",
                "evidence_spans": [{"start": 0, "end": min(20, len(self.chunks[item.chunk_id].text))}] if value else [],
                "reviewer": {"type": "agent", "id": "oracle-1", "revision": "wf-1"}, "reviewed_at": "2026-10-05T12:00:00Z", "notes": None,
            })
        path = self.root / "drafts" / "reviews.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
        return path

    def seal(self, reviews):
        return prepare_bank(self.config, self.root / "prepared" / PREPARED_MANIFEST, self.root / "evaluation" / EVALUATION_MANIFEST, reviews, self.root)

    def test_a_complete_bank_writes_the_bank_in_the_evaluation_side_and_the_seal_in_the_prepared_side(self):
        seal = self.seal(self.reviews())
        self.assertEqual((seal["stage"], seal["protocol"]), (1, "prejudged_fixed_candidates_v1"))
        self.assertIsNotNone(seal["bank_manifest_sha256"])
        self.assertTrue((self.root / "evaluation" / "bank" / "judgment-bank-manifest.json").is_file())
        self.assertEqual(load_seal(self.root / "prepared" / INITIAL_SEAL, 1)["manifest_sha256"], seal["manifest_sha256"])
        self.assertNotIn("bank", {p.name for p in (self.root / "prepared").iterdir()})
        self.assertEqual(self.seal(self.reviews())["manifest_sha256"], seal["manifest_sha256"], "an identical rerun is idempotent")

    def test_missing_reviews_block_sealing_and_write_nothing(self):
        with self.assertRaisesRegex(ManifestError, "unresolved or missing"):
            self.seal(self.reviews(skip=1))
        self.assertFalse((self.root / "prepared" / INITIAL_SEAL).exists())
        self.assertFalse((self.root / "evaluation" / "bank" / "judgment-bank-manifest.json").exists())
        self.seal(self.reviews())                                            # a retry after the failure succeeds

    def test_a_wrong_salt_cannot_rebuild_the_recorded_pool(self):
        reviews = self.reviews()
        config = load_data_config(write_config(self.root / "drafts"))
        object.__setattr__(config, "salt", "a-different-salt-value")
        with self.assertRaisesRegex(ManifestError, "differs from the recorded pool"):
            prepare_bank(config, self.root / "prepared" / PREPARED_MANIFEST, self.root / "evaluation" / EVALUATION_MANIFEST, reviews, self.root)

    def test_a_review_for_an_item_outside_the_pool_is_refused(self):
        path = self.reviews()
        record = json.loads(path.read_text().splitlines()[0])
        record["item_id"] = "it_" + "0" * 24
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ManifestError, "not in the review pool"):
            self.seal(path)

    def test_the_postrank_protocol_seals_without_a_bank(self):
        root = make_root(self)
        config_path = write_config(root / "drafts", protocol="postrank_pool_v1")
        config, _, split = stage_groups(root, config_path)
        drafts_from_split(root, split)
        prepare_materialize(config, root / "prepared" / SPLIT_MANIFEST, root / "evaluation", root)
        self.assertFalse((root / "evaluation" / "review").exists(), "the post-rank pool does not exist before ranking")
        seal = prepare_initial_seal_postrank(config, root / "prepared" / PREPARED_MANIFEST, root / "evaluation" / EVALUATION_MANIFEST, root)
        self.assertIsNone(seal["bank_manifest_sha256"])
        with self.assertRaisesRegex(ManifestError, "post-rank"):
            prepare_bank(config, root / "prepared" / PREPARED_MANIFEST, root / "evaluation" / EVALUATION_MANIFEST, self.reviews(), root)
        with self.assertRaisesRegex(ManifestError, "prejudged"):
            prepare_initial_seal_postrank(self.config, self.root / "prepared" / PREPARED_MANIFEST, self.root / "evaluation" / EVALUATION_MANIFEST, self.root)


if __name__ == "__main__":
    unittest.main()
