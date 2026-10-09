"""Train-only negative mining and unique BCE pair materialization."""

import dataclasses
import json
import unittest

from devsearch.artifacts.canonical import canonical_json, seal_manifest, sha256_file
from devsearch.contracts import SearchRequest
from devsearch.data.pairs import (
    MiningConfig, build_mining_pool, load_pairs, mine_pairs, sampling_identity, write_pair_artifacts,
)
from devsearch.errors import ManifestError

from dataset import World
from helpers import RootTestCase
from test_queries import make_query


class PairsCase(RootTestCase):
    def setUp(self):
        super().setUp()
        self.world = World.build(self)
        self.pool = build_mining_pool(self.world.corpus, self.world.merge, self.world.assignment)
        self.config = MiningConfig(hard_negatives=2, random_negatives=2, seed=42)
        self.sampling = sampling_identity(self.config, self.pool, corpus_id="c", index_id="i", queries_sha256="q", qrels_sha256="r")

    def mine(self, queries=None, qrels=None, config=None, pool=None):
        w = self.world
        return mine_pairs(corpus=w.corpus, retriever=w.loaded.retriever(), merge=w.merge, pool=pool or self.pool,
                          queries=queries if queries is not None else w.queries["train"],
                          qrels=qrels if qrels is not None else w.qrels["train"], config=config or self.config, sampling_id=self.sampling)

    def group_of(self, chunk_id):
        return self.world.merge.merged_of[next(c.evidence_group_id for c in self.world.corpus.chunks if c.chunk_id == chunk_id)]


class MiningPoolTests(PairsCase):
    def test_pool_contains_exactly_the_train_owned_chunks(self):
        self.assertEqual(set(self.pool.chunk_ids), self.world.chunk_ids_of("train"))
        for split in ("validation", "test"):
            self.assertFalse(set(self.pool.chunk_ids) & self.world.chunk_ids_of(split))
        self.assertEqual(list(self.pool.chunk_ids), sorted(self.pool.chunk_ids))
        self.assertRegex(self.pool.pool_id, r"^pool-[0-9a-f]{32}$")

    def test_budget_exclusions_shrink_the_pool_and_change_its_identity(self):
        dropped = self.pool.chunk_ids[0]
        smaller = build_mining_pool(self.world.corpus, self.world.merge, self.world.assignment, excluded_chunk_ids=[dropped])
        self.assertNotIn(dropped, smaller.chunk_ids)
        self.assertNotEqual(smaller.pool_id, self.pool.pool_id)


class MiningTests(PairsCase):
    def test_every_pair_comes_from_the_train_pool_and_the_counts_reconcile(self):
        result = self.mine()
        pool = set(self.pool.chunk_ids)
        self.assertTrue(all(p.chunk_id in pool for p in result.pairs))
        self.assertEqual(result.report["queries_used"], 5)
        self.assertEqual((result.report["positives"], result.report["negatives"], result.report["pairs_total"]), (5, 20, 25))
        self.assertEqual(result.report["negatives_bm25"], 10)
        self.assertEqual(result.report["negatives_random"], 10)
        self.assertEqual(result.report["pos_weight"], 4.0)
        self.assertEqual(len(result.pairs), len({(p.query_id, p.chunk_id) for p in result.pairs}))
        self.assertEqual(result.report["shortage_queries"], [])

    def test_negatives_never_come_from_a_group_holding_a_known_positive(self):
        result = self.mine()
        by_query = {}
        for p in result.pairs:
            by_query.setdefault(p.query_id, []).append(p)
        for query_id, pairs in by_query.items():
            positive_groups = {self.group_of(p.chunk_id) for p in pairs if p.label == 1}
            for p in pairs:
                if p.label == 0:
                    self.assertNotIn(self.group_of(p.chunk_id), positive_groups, (query_id, p))

    def test_hard_negatives_are_the_best_allowed_bm25_ranks_inside_the_pool(self):
        result = self.mine()
        w = self.world
        for query in w.queries["train"]:
            request = SearchRequest(query.text, query.sources, query.versions, candidate_k=100)
            ranked = [c.chunk.chunk_id for c in w.loaded.retriever().retrieve_within(request, set(self.pool.chunk_ids))]
            positive = next(p.chunk_id for p in result.pairs if p.query_id == query.query_id and p.label == 1)
            allowed = [c for c in ranked if self.group_of(c) != self.group_of(positive)]
            mined = [p.chunk_id for p in result.pairs if p.query_id == query.query_id and p.negative_method == "bm25"]
            self.assertEqual(sorted(mined), sorted(allowed[:2]), query.query_id)

    def test_mining_is_deterministic_and_the_seed_changes_random_negatives_only(self):
        first, again = self.mine(), self.mine()
        self.assertEqual(first.pairs, again.pairs)
        other = self.mine(config=dataclasses.replace(self.config, seed=43))
        key = lambda r, method: sorted((p.query_id, p.chunk_id) for p in r.pairs if p.negative_method == method)
        self.assertEqual(key(first, "bm25"), key(other, "bm25"))
        self.assertNotEqual(key(first, "random"), key(other, "random"))

    def test_assumed_negatives_keep_their_status_and_method(self):
        result = self.mine()
        for p in result.pairs:
            if p.label == 0:
                self.assertEqual((p.label_status, p.negative_method in ("bm25", "random")), ("assumed_negative", True))
            else:
                self.assertEqual((p.label_status, p.negative_method), ("reviewed_positive", None))
        self.assertEqual({r["label_status"] for r in result.sampling}, {"assumed_negative"})
        self.assertEqual(len(result.sampling), 20)

    def test_a_query_filter_confines_negatives_to_that_filter(self):
        w = self.world
        query = make_query("q-tr-f", "How do I load a model?", ["eg-model-load"], "train", sources=["fixture-torch"])
        qrel = w.positive("q-tr-f", ("Checkpoints", "Loading"), "fixture-r1")
        qrel = dataclasses.replace(qrel, chunk_id=qrel.chunk_id)
        # The positive belongs to eg-ckpt-load; the query names eg-model-load, so use a consistent pair instead.
        query = make_query("q-tr-f", "How do I load a model?", ["eg-ckpt-load"], "train", sources=["fixture-torch"])
        result = self.mine(queries=[query], qrels=[qrel])
        sources = {w.corpus.doc_of(next(c for c in w.corpus.chunks if c.chunk_id == p.chunk_id)).source_id for p in result.pairs}
        self.assertEqual(sources, {"fixture-torch"})

    def test_exclusions_are_counted_not_hidden(self):
        w = self.world
        unaccepted = make_query("q-tr-u", "How do I save things?", ["eg-ckpt-save"], "train", reviews=[])
        no_positive = make_query("q-tr-n", "Anything about padding?", ["eg-ckpt-save"], "train")
        outside = make_query("q-tr-o", "How is padding done?", ["eg-ckpt-save"], "train")
        outside_qrel = w.positive("q-tr-o", ("Tokenizers", "Padding"), "fixture-v1", label_source="pooled-review")   # validation-owned chunk
        unreviewed = dataclasses.replace(w.positive("q-tr-n", ("Checkpoints", "Saving"), "fixture-r1"), review_status="unreviewed", reviews=())
        result = self.mine(queries=[unaccepted, no_positive, outside], qrels=[outside_qrel, unreviewed])
        self.assertEqual(result.report["queries_used"], 0)
        self.assertEqual(result.report["exclusions"], {"no_usable_positive": 2, "positive_outside_pool": 1, "query_not_accepted": 1})
        self.assertEqual(result.pairs, ())

    def test_a_short_pool_records_shortages_and_never_borrows_from_other_splits(self):
        tiny = dataclasses.replace(self.pool, chunk_ids=tuple(sorted(self.pool.chunk_ids))[:4])
        w = self.world
        queries = [q for q in w.queries["train"] if q.query_id in ("q-tr-1", "q-tr-2")]
        qrels = [q for q in w.qrels["train"] if q.query_id in ("q-tr-1", "q-tr-2")]
        in_pool = [q for q in qrels if q.chunk_id in set(tiny.chunk_ids)]
        result = self.mine(queries=queries, qrels=qrels, pool=tiny)
        self.assertTrue(all(p.chunk_id in set(tiny.chunk_ids) for p in result.pairs))
        self.assertTrue(result.report["exclusions"] or result.report["shortage_queries"] or len(in_pool) == 2)
        if result.report["queries_used"]:
            self.assertTrue(result.report["shortage_queries"], "fewer than 4 negatives available must be recorded")

    def test_only_train_queries_and_only_their_judgments_are_accepted(self):
        w = self.world
        with self.assertRaisesRegex(ManifestError, "only train queries may be mined"):
            self.mine(queries=w.queries["train"] + w.queries["test"], qrels=w.qrels["train"])
        with self.assertRaisesRegex(ManifestError, "not in the supplied train queries"):
            self.mine(qrels=w.qrels["train"] + w.qrels["test"])

    def test_mining_config_is_validated(self):
        for kwargs in ({"hard_negatives": -1}, {"hard_negatives": 0, "random_negatives": 0}, {"hard_search_depth": 1}, {"hard_search_depth": 10**6}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ManifestError):
                MiningConfig(**kwargs)


class PairArtifactTests(PairsCase):
    def setUp(self):
        super().setUp()
        self.result = self.mine()
        self.directory = self.world.root / "data" / "pairs"
        self.directory.mkdir()
        self.manifest = write_pair_artifacts(self.directory, self.result, corpus_id=self.world.corpus.corpus_id, index_id=self.world.loaded.index_id,
                                             queries_sha256="q" * 64, qrels_sha256="r" * 64, config=self.config)

    def load(self):
        return load_pairs(self.directory / "pairs-manifest.json", self.world.corpus)

    def reseal(self, change):
        path = self.directory / "pairs-manifest.json"
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(seal_manifest(manifest)))

    def test_round_trip(self):
        manifest, pairs, pool = self.load()
        self.assertEqual(pairs, self.result.pairs)
        self.assertEqual(pool, self.pool)
        self.assertEqual(manifest["report"]["pos_weight"], 4.0)

    def test_files_are_written_once(self):
        with self.assertRaises(FileExistsError):
            write_pair_artifacts(self.directory, self.result, corpus_id=self.world.corpus.corpus_id, index_id="i", queries_sha256="q" * 64, qrels_sha256="r" * 64, config=self.config)

    def test_tampering_is_detected(self):
        pairs = self.directory / "pairs.jsonl"
        original = pairs.read_bytes()
        pairs.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ManifestError, "SHA-256"):
            self.load()
        pairs.write_bytes(original)
        self.load()

    def rewrite_pairs(self, change):
        pairs_path = self.directory / "pairs.jsonl"
        records = [json.loads(line) for line in pairs_path.read_text().splitlines()]
        change(records)
        data = "".join(canonical_json(r) + "\n" for r in records)
        pairs_path.write_text(data)
        self.reseal(lambda m: m["files"]["pairs.jsonl"].update(sha256=sha256_file(pairs_path)))

    def test_consistent_hashes_cannot_hide_a_bad_pair_set(self):
        other = next(c for c in self.world.chunk_ids_of("test"))
        cases = {
            "chunk outside the train pool": (lambda r: r[0].update(chunk_id=other), "outside the train-owned mining pool|pair_id"),
            "duplicate pair": (lambda r: r.append(dict(r[0])), "duplicate|pair_id"),
            "positive labeled as an assumed negative": (lambda r: next(x for x in r if x["label"] == 1).update(label_status="assumed_negative"), "inconsistent"),
            "forged pair id": (lambda r: r[0].update(pair_id="pr_" + "0" * 32), "pair_id"),
            "non binary label": (lambda r: r[0].update(label=2), "0 or 1"),
            "unknown negative method": (lambda r: [x.update(negative_method="oracle") for x in r if x["label"] == 0], "inconsistent"),
        }
        for name, (change, fragment) in cases.items():
            with self.subTest(name):
                self.directory2 = self.directory
                original = (self.directory / "pairs.jsonl").read_bytes()
                manifest_original = (self.directory / "pairs-manifest.json").read_bytes()
                self.rewrite_pairs(change)
                with self.assertRaisesRegex(ManifestError, fragment):
                    self.load()
                (self.directory / "pairs.jsonl").write_bytes(original)
                (self.directory / "pairs-manifest.json").write_bytes(manifest_original)
        self.load()

    def test_reported_counts_must_reconcile(self):
        self.reseal(lambda m: m["report"].update(positives=99))
        with self.assertRaisesRegex(ManifestError, "do not reconcile"):
            self.load()

    def test_a_pool_that_names_foreign_chunks_is_rejected(self):
        pool_path = self.directory / "mining-pool.json"
        doc = json.loads(pool_path.read_text())
        doc["chunk_ids"] = sorted(doc["chunk_ids"] + ["ch_" + "f" * 32])
        pool_path.write_text(json.dumps(doc))
        self.reseal(lambda m: m["files"]["mining-pool.json"].update(sha256=sha256_file(pool_path)))
        with self.assertRaisesRegex(ManifestError, "pool_id|not in the corpus"):
            self.load()


if __name__ == "__main__":
    unittest.main()
