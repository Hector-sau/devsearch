"""Ranked-output artifacts: permutation enforcement, hash binding, atomic publication."""

import shutil
import tempfile
import unittest
from pathlib import Path

from devsearch.errors import ManifestError
from devsearch.evaluation.outputs import MANIFEST_NAME, load_ranked_outputs, write_ranked_outputs
from devsearch.evaluation.ranking import bm25_ranked_records, order_by_score, ranked_record

CANDIDATES = [
    {"query_id": "q1", "candidate_set_id": "cs1", "candidates": [{"chunk_id": "a", "rank": 1, "bm25_score": 2.0}, {"chunk_id": "b", "rank": 2, "bm25_score": 1.0}, {"chunk_id": "c", "rank": 3, "bm25_score": 0.5}]},
    {"query_id": "q2", "candidate_set_id": "cs2", "candidates": [{"chunk_id": "d", "rank": 1, "bm25_score": 3.0}, {"chunk_id": "e", "rank": 2, "bm25_score": 2.0}]},
]


def reranked(scores):
    return [ranked_record(r["query_id"], r["candidate_set_id"], order_by_score([c["chunk_id"] for c in r["candidates"]], scores[r["query_id"]])) for r in CANDIDATES]


class OutputsTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-out-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.systems = {"bm25": ({"role": "baseline"}, bm25_ranked_records(CANDIDATES)),
                        "full-seed42": ({"role": "locked", "recipe_id": "full", "seed": 42}, reranked({"q1": [0.1, 0.9, 0.5], "q2": [1.0, -1.0]}))}

    def write(self, systems=None, directory=None):
        return write_ranked_outputs(directory or self.dir, systems or self.systems, CANDIDATES, candidates_manifest_sha256="c" * 64, selection_id="selection-1", protocol_id="protocol-1")

    def test_round_trip_preserves_every_system_and_binds_the_selection(self):
        manifest = self.write()
        loaded_manifest, loaded = load_ranked_outputs(self.dir / MANIFEST_NAME, CANDIDATES)
        self.assertEqual(loaded_manifest["manifest_sha256"], manifest["manifest_sha256"])
        self.assertEqual((loaded_manifest["selection_id"], loaded_manifest["protocol_id"]), ("selection-1", "protocol-1"))
        self.assertEqual(loaded["bm25"], self.systems["bm25"][1])
        self.assertEqual([i["chunk_id"] for i in loaded["full-seed42"][0]["ranking"]], ["b", "c", "a"])

    def test_a_system_that_drops_adds_duplicates_or_renumbers_is_rejected_and_writes_nothing(self):
        for name, mutate in {
            "dropped query": lambda recs: recs[:-1],
            "dropped candidate": lambda recs: [{**recs[0], "ranking": recs[0]["ranking"][:-1]}, recs[1]],
            "foreign chunk": lambda recs: [{**recs[0], "ranking": recs[0]["ranking"][:-1] + [{"chunk_id": "z", "rank": 3, "score": 0.0}]}, recs[1]],
            "duplicate chunk": lambda recs: [{**recs[0], "ranking": [recs[0]["ranking"][0]] * 3}, recs[1]],
            "bad ranks": lambda recs: [{**recs[0], "ranking": [{**i, "rank": i["rank"] + 1} for i in recs[0]["ranking"]]}, recs[1]],
            "nan score": lambda recs: [{**recs[0], "ranking": [{**recs[0]["ranking"][0], "score": float("nan")}] + recs[0]["ranking"][1:]}, recs[1]],
            "boolean score": lambda recs: [{**recs[0], "ranking": [{**recs[0]["ranking"][0], "score": True}] + recs[0]["ranking"][1:]}, recs[1]],
        }.items():
            with self.subTest(name):
                systems = {"bad": ({}, mutate(bm25_ranked_records(CANDIDATES)))}
                with self.assertRaises(ManifestError):
                    self.write(systems)
                self.assertEqual(list(self.dir.iterdir()), [])

    def test_tampering_with_an_output_file_or_the_manifest_is_detected(self):
        self.write()
        target = self.dir / "ranked-bm25.jsonl"
        original = target.read_bytes()
        target.write_bytes(original.replace(b'"a"', b'"b"', 1))
        with self.assertRaisesRegex(ManifestError, "ranked-bm25.jsonl"):
            load_ranked_outputs(self.dir / MANIFEST_NAME, CANDIDATES)
        target.write_bytes(original)
        manifest = self.dir / MANIFEST_NAME
        manifest.write_text(manifest.read_text().replace("selection-1", "selection-2"))
        with self.assertRaises(ManifestError):
            load_ranked_outputs(manifest, CANDIDATES)

    def test_loading_enforces_permutation_even_if_the_hashes_were_regenerated(self):
        self.write()
        other = [{**CANDIDATES[0], "candidates": CANDIDATES[0]["candidates"][:2]}, CANDIDATES[1]]
        with self.assertRaisesRegex(ManifestError, "permutation"):
            load_ranked_outputs(self.dir / MANIFEST_NAME, other)

    def test_identical_rewrite_is_a_noop_and_a_changed_one_is_refused(self):
        first = self.write()
        self.assertEqual(self.write()["manifest_sha256"], first["manifest_sha256"])
        changed = dict(self.systems, **{"full-seed42": ({"role": "locked"}, reranked({"q1": [0.9, 0.1, 0.5], "q2": [1.0, -1.0]}))})
        with self.assertRaisesRegex(ManifestError, "immutable"):
            self.write(changed)

    def test_system_names_must_be_plain_identifiers(self):
        for name in ("../x", "a/b", "", "sp ace"):
            with self.subTest(name), self.assertRaises(ManifestError):
                self.write({name: ({}, bm25_ranked_records(CANDIDATES))})

    def test_missing_directory_and_empty_system_set_are_refused(self):
        with self.assertRaises(ManifestError):
            self.write(directory=self.dir / "nowhere")
        with self.assertRaises(ManifestError):
            write_ranked_outputs(self.dir, {}, CANDIDATES, candidates_manifest_sha256="c" * 64, selection_id="s", protocol_id="p")


if __name__ == "__main__":
    unittest.main()
