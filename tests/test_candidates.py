"""Fixed candidate files: determinism, retained failures, tamper detection, no inserted passages."""

import json
import unittest

from devsearch.artifacts.canonical import seal_manifest, sha256_file, sha256_text
from devsearch.data.candidates import (
    build_candidates, candidate_record, load_candidates, verify_candidates_deep, write_candidate_artifacts,
)
from devsearch.errors import ManifestError
from devsearch.retrieval.search import run_baseline

from dataset import World
from helpers import RootTestCase, search_limits
from test_queries import make_query


class CandidateTests(RootTestCase):
    def setUp(self):
        super().setUp()
        self.world = World.build(self)
        self.loaded = self.world.loaded
        self.queries = self.world.queries["validation"] + [make_query("q-va-9", "kubernetes ingress controller", ["eg-mode-accum"], "validation")]
        self.queries.sort(key=lambda q: q.query_id)
        self.directory = self.world.root / "data" / "cands"
        self.directory.mkdir()

    def write(self, k=50):
        records = build_candidates(self.loaded, self.queries, k)
        return records, write_candidate_artifacts(self.directory, "validation", records, self.loaded, queries_sha256="q" * 64)

    def test_records_are_the_natural_ranked_list_with_hashes(self):
        records, _ = self.write()
        record = next(r for r in records if r["query_id"] == "q-va-1")
        self.assertEqual((record["requested"], record["actual"]), (50, len(record["candidates"])))
        self.assertEqual([c["rank"] for c in record["candidates"]], list(range(1, record["actual"] + 1)))
        chunks = {c.chunk_id: c for c in self.world.corpus.chunks}
        for item in record["candidates"]:
            self.assertEqual(item["text_sha256"], sha256_text(chunks[item["chunk_id"]].text))
        scores = [c["bm25_score"] for c in record["candidates"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_a_zero_match_query_keeps_an_empty_list_as_a_retrieval_failure(self):
        records, manifest = self.write()
        empty = next(r for r in records if r["query_id"] == "q-va-9")
        self.assertEqual((empty["actual"], empty["candidates"]), (0, []))
        self.assertEqual(manifest["counts"]["empty"], 1)
        self.assertEqual(manifest["counts"]["queries"], len(self.queries))

    def test_candidate_set_id_matches_the_live_retrieve_output(self):
        query = self.queries[0]
        record = candidate_record(self.loaded, query, 50)
        live = run_baseline(self.loaded, __import__("helpers").request(query.text, sources=query.sources, versions=query.versions, candidate_k=50, top_k=5), search_limits())
        self.assertEqual(record["candidate_set_id"], live["candidate_set_id"])
        self.assertEqual([c["chunk_id"] for c in record["candidates"][:5]], [r["chunk_id"] for r in live["baseline_results"]])

    def test_depth_is_recorded_separately_and_changes_the_config_hash(self):
        r50, r10 = candidate_record(self.loaded, self.queries[0], 50), candidate_record(self.loaded, self.queries[0], 10)
        self.assertNotEqual(r50["retrieval_config_sha256"], r10["retrieval_config_sha256"])
        self.assertEqual(r10["requested"], 10)
        self.assertLessEqual(r10["actual"], 10)

    def test_round_trip_and_deep_verification(self):
        records, manifest = self.write()
        loaded_manifest, loaded_records = load_candidates(self.directory / "candidates-validation-manifest.json", self.loaded)
        self.assertEqual(loaded_records, records)
        verify_candidates_deep(loaded_records, self.loaded, self.queries)

    def test_files_are_never_overwritten(self):
        self.write()
        with self.assertRaises(FileExistsError):
            self.write()

    def test_tampering_is_detected_at_load(self):
        self.write()
        path = self.directory / "candidates-validation.jsonl"
        original = path.read_bytes()
        path.write_bytes(original.replace(b'"rank":1', b'"rank":2', 1))
        with self.assertRaisesRegex(ManifestError, "SHA-256"):
            load_candidates(self.directory / "candidates-validation-manifest.json", self.loaded)

    def fresh_directory(self, name):
        directory = self.world.root / "data" / name
        directory.mkdir()
        records = build_candidates(self.loaded, self.queries, 50)
        write_candidate_artifacts(directory, "validation", records, self.loaded, queries_sha256="q" * 64)
        return directory

    def reseal_file(self, mutate, directory=None):
        """Rewrite the candidate file and re-seal its manifest so only the planted defect remains."""
        directory = directory or self.directory
        path = directory / "candidates-validation.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(records)
        path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records))
        manifest_path = directory / "candidates-validation-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["file"]["sha256"] = sha256_file(path)
        manifest_path.write_text(json.dumps(seal_manifest(manifest)))
        return manifest_path

    def test_structurally_invalid_lists_are_rejected_even_with_consistent_hashes(self):
        first_with_results = lambda records, minimum=1: next(r for r in records if r["actual"] >= minimum)
        cases = {
            "rank gap": (lambda r: first_with_results(r, 3)["candidates"][1].update(rank=7), "dense"),
            "unknown chunk": (lambda r: first_with_results(r)["candidates"][0].update(chunk_id="ch_" + "0" * 32), "unknown or its text hash"),
            "wrong text hash": (lambda r: first_with_results(r)["candidates"][0].update(text_sha256="0" * 64), "unknown or its text hash"),
            "count mismatch": (lambda r: r[0].update(actual=99), "counts are inconsistent"),
            "other index": (lambda r: r[0].update(index_id="index-" + "0" * 32), "different corpus/index"),
            "unsorted records": (lambda r: r.reverse(), "sorted"),
        }
        for number, (name, (mutate, fragment)) in enumerate(cases.items()):
            with self.subTest(name):
                directory = self.fresh_directory(f"defect-{number}")
                manifest_path = self.reseal_file(mutate, directory)
                with self.assertRaisesRegex(ManifestError, fragment):
                    load_candidates(manifest_path, self.loaded)

    def test_a_score_order_violation_is_rejected(self):
        self.write()
        def swap(records):
            record = next(r for r in records if r["actual"] > 2)
            a, b = record["candidates"][0], record["candidates"][1]
            record["candidates"][0], record["candidates"][1] = {**b, "rank": 1}, {**a, "rank": 2}
        manifest_path = self.reseal_file(swap)
        with self.assertRaisesRegex(ManifestError, "not ordered by descending score"):
            load_candidates(manifest_path, self.loaded)

    def test_an_inserted_gold_passage_passes_structure_checks_but_fails_the_deep_check(self):
        records, _ = self.write()
        gold = self.world.positive("q-va-2", ("Tokenizers", "Padding"), "fixture-v1").chunk_id
        victim = next(r for r in records if r["query_id"] == "q-va-1")
        chunk = next(c for c in self.world.corpus.chunks if c.chunk_id == gold)
        forged = {**victim, "candidates": victim["candidates"] + [{"chunk_id": gold, "text_sha256": sha256_text(chunk.text), "bm25_score": -1.0, "rank": victim["actual"] + 1}], "actual": victim["actual"] + 1}
        tampered = [forged if r["query_id"] == "q-va-1" else r for r in records]
        with self.assertRaisesRegex(ManifestError, "differ from a fresh retrieval"):
            verify_candidates_deep(tampered, self.loaded, self.queries)

    def test_deep_verification_requires_the_same_query_ids(self):
        records, _ = self.write()
        with self.assertRaisesRegex(ManifestError, "different query IDs"):
            verify_candidates_deep(records[:-1], self.loaded, self.queries)


if __name__ == "__main__":
    unittest.main()
