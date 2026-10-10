"""Frozen-input arithmetic replay: no private data, models, network or SDK."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("validation_replay_script", REPO / "scripts/replay_validation.py")
REPLAY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REPLAY)


def fixture():
    queries, orders, labels = [], {}, []
    for query_id, group_id in (("PRIVATE_QUERY_A", "PRIVATE_GROUP_A"), ("PRIVATE_QUERY_B", "PRIVATE_GROUP_B")):
        queries.append({"query_id": query_id, "group_id": group_id, "evidence_group_ids": [group_id + "_SOURCE"],
                        "text": "PRIVATE_QUESTION_TEXT", "split": "validation", "language": "en"})
        chunks = [f"PRIVATE_CHUNK_{query_id}_{position}" for position in range(50)]
        permutations = {name: list(range(50)) for name in REPLAY.SYSTEMS}
        for seed, position in ((42, 0), (43, 2), (44, 3)):
            order = list(range(50))
            order.remove(1)
            order.insert(position, 1)
            permutations[f"R5-seed{seed}-fused"] = order
        orders[query_id] = {"chunk_ids": chunks, "system_permutations": permutations}
        labels += [{"query_id": query_id, "chunk_id": chunk, "relevance": int(position == 1),
                    "evidence_quote": "PRIVATE_REVIEW_QUOTE"} for position, chunk in enumerate(chunks[:15])]
    rankings = {"ranking_orders": orders, "exact_order_equality_verified_on_server": True,
                "ranking_scores_omitted": True, "rank_source_sha256": {name: "a" * 64 for name in REPLAY.SYSTEMS}}
    release = {"schema_version": "devsearch-sampled-agent-human-assisted-release-v4", "status": "sealed",
               "query_count": 2, "items": 30}
    return queries, rankings, release, labels


class ValidationReplayTests(unittest.TestCase):
    def setUp(self):
        self.inputs = fixture()
        temporary = tempfile.TemporaryDirectory(prefix="devsearch-validation-replay-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def validated(self, values=None):
        return REPLAY.validate_inputs(*(values or self.inputs), expected_queries=2, expected_groups=2, expected_pairs=30)

    def write_inputs(self):
        queries, rankings, release, labels = copy.deepcopy(self.inputs)
        self.release_path = self.root / "release"
        self.release_path.mkdir()
        self.query_path, self.ranking_path = self.root / "queries.jsonl", self.root / "orders.json"
        self.query_path.write_text("".join(json.dumps(query) + "\n" for query in queries))
        self.ranking_path.write_text(json.dumps(rankings))
        self.label_path = self.release_path / "resolved-labels.json"
        self.label_path.write_text(json.dumps(labels))
        release["resolved_labels_sha256"] = hashlib.sha256(self.label_path.read_bytes()).hexdigest()
        self.manifest_path = self.release_path / "label-release.json"
        self.manifest_path.write_text(json.dumps(release))
        self.pins = {"queries": hashlib.sha256(self.query_path.read_bytes()).hexdigest(),
                     "rankings": hashlib.sha256(self.ranking_path.read_bytes()).hexdigest(),
                     "label_release": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()}
        self.output = self.root / "aggregate.json"

    def cli(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.multiple(REPLAY, PINNED_SHA256=self.pins, QUERY_COUNT=2, GROUP_COUNT=2, PAIR_COUNT=30), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = REPLAY.main(["--queries", str(self.query_path), "--rankings", str(self.ranking_path),
                                "--release", str(self.release_path), "--output", str(self.output)])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_hand_computed_metrics_and_all_query_denominators(self):
        report = REPLAY.summarize(self.validated(), {}, bootstrap_repetitions=100)
        self.assertEqual(report["counts"]["queries"], 2)
        self.assertEqual(report["counts"]["excluded_queries"], 0)
        self.assertEqual(report["systems"]["BM25"]["group_macro"]["mrr@10"], .5)
        expected = (1 + 1 / 3 + 1 / 4) / 3
        self.assertAlmostEqual(report["R5_fused_three_seed_summary"]["group_macro"]["mrr@10"]["mean"], expected)
        self.assertFalse(report["R5_fused_three_seed_summary"]["all_three_mrr_improved"])

    def test_seeds_are_averaged_inside_groups_before_bootstrap(self):
        received = []
        def capture(a, b, **kwargs):
            received.append((a, b, kwargs))
            return {"groups": len(a)}
        with mock.patch.object(REPLAY, "paired_group_bootstrap", side_effect=capture):
            REPLAY.summarize(self.validated(), {})
        a, b, settings = received[0]
        self.assertEqual(len(a), 2)  # Two groups, not six seed/group rows.
        self.assertEqual(set(a), set(b))
        self.assertTrue(all(abs(score - (1 + 1 / 3 + 1 / 4) / 3) < 1e-12 for score in a.values()))
        self.assertEqual(settings["seed"], 20261008)

    def test_known_positive_outside_retrieval_stays_in_metric_denominator(self):
        values = copy.deepcopy(self.inputs)
        values[3][14].update(chunk_id="PRIVATE_MISSED_SOURCE_POSITIVE", relevance=1)
        report = REPLAY.summarize(self.validated(values), {}, bootstrap_repetitions=100)
        self.assertEqual(report["systems"]["BM25"]["query_macro"]["known_positive_recall@50"], .75)
        self.assertEqual(report["counts"]["excluded_queries"], 0)
        self.assertEqual(report["systems"]["BM25"]["query_macro"]["mrr@10"], .5)

    def test_missing_top10_judgment_is_not_treated_as_a_negative(self):
        values = copy.deepcopy(self.inputs)
        values[3][0]["chunk_id"] = "PRIVATE_OUTSIDE_CANDIDATE"
        with self.assertRaisesRegex(ValueError, "incomplete labeled top-10"):
            self.validated(values)

    def test_duplicate_out_of_range_and_boolean_permutations_are_refused(self):
        for replacement in (0, 50, True):
            with self.subTest(replacement=replacement):
                values = copy.deepcopy(self.inputs)
                values[1]["ranking_orders"]["PRIVATE_QUERY_A"]["system_permutations"]["BM25"][1] = replacement
                with self.assertRaisesRegex(ValueError, "malformed system permutation"):
                    self.validated(values)

    def test_duplicate_pool_chunk_or_missing_system_cannot_change_candidate_pool(self):
        values = copy.deepcopy(self.inputs)
        row = values[1]["ranking_orders"]["PRIVATE_QUERY_A"]
        row["chunk_ids"][1] = row["chunk_ids"][0]
        with self.assertRaisesRegex(ValueError, "duplicate chunks"):
            self.validated(values)
        values = copy.deepcopy(self.inputs)
        values[1]["ranking_orders"]["PRIVATE_QUERY_A"]["system_permutations"].pop("R5-seed42-fused")
        with self.assertRaisesRegex(ValueError, "system inventory"):
            self.validated(values)

    def test_duplicate_query_or_group_membership_is_refused(self):
        for field in ("query_id", "group_id", "evidence_group_ids"):
            with self.subTest(field=field):
                values = copy.deepcopy(self.inputs)
                values[0][1][field] = values[0][0][field]
                with self.assertRaises(ValueError):
                    self.validated(values)

    def test_duplicate_or_nonbinary_labels_are_refused(self):
        values = copy.deepcopy(self.inputs)
        values[3][1] = copy.deepcopy(values[3][0])
        with self.assertRaisesRegex(ValueError, "duplicate resolved"):
            self.validated(values)
        for decision in (None, True, 2):
            values = copy.deepcopy(self.inputs)
            values[3][0]["relevance"] = decision
            with self.assertRaisesRegex(ValueError, "not binary"):
                self.validated(values)

    def test_missing_positive_or_unsealed_release_blocks_replay(self):
        values = copy.deepcopy(self.inputs)
        values[3][1]["relevance"] = 0
        with self.assertRaisesRegex(ValueError, "refuses to exclude"):
            self.validated(values)
        values = copy.deepcopy(self.inputs)
        values[2]["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "not sealed"):
            self.validated(values)

    def test_duplicate_json_members_are_refused_instead_of_silently_overwritten(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON member"):
            REPLAY.parse_json(b'{"ranking_orders": {}, "ranking_orders": {}}')

    def test_query_or_ranking_byte_tampering_refuses_without_output(self):
        self.write_inputs()
        self.query_path.write_bytes(self.query_path.read_bytes() + b"\n")
        code, _, error = self.cli()
        self.assertEqual(code, 2)
        self.assertIn("queries SHA-256 mismatch", error)
        self.assertFalse(self.output.exists())
        self.query_path.write_bytes(self.query_path.read_bytes()[:-1])
        self.ranking_path.write_bytes(self.ranking_path.read_bytes() + b" ")
        code, _, error = self.cli()
        self.assertEqual(code, 2)
        self.assertIn("rankings SHA-256 mismatch", error)
        self.assertFalse(self.output.exists())

    def test_modified_labels_and_updated_manifest_cannot_bypass_fixed_release_hash(self):
        self.write_inputs()
        self.label_path.write_bytes(self.label_path.read_bytes() + b" ")
        code, _, error = self.cli()
        self.assertEqual(code, 2)
        self.assertIn("resolved labels SHA-256 mismatch", error)
        manifest = json.loads(self.manifest_path.read_text())
        manifest["resolved_labels_sha256"] = hashlib.sha256(self.label_path.read_bytes()).hexdigest()
        self.manifest_path.write_text(json.dumps(manifest))
        code, _, error = self.cli()
        self.assertEqual(code, 2)
        self.assertIn("label release SHA-256 mismatch", error)
        self.assertFalse(self.output.exists())

    def test_cli_exports_only_aggregates_and_hashes(self):
        self.write_inputs()
        code, stdout, stderr = self.cli()
        self.assertEqual(code, 0, stderr)
        report = json.loads(self.output.read_text())
        combined = stdout + self.output.read_text()
        self.assertNotIn("PRIVATE_", combined)
        self.assertNotIn(str(self.root), combined)
        self.assertEqual(report["input_sha256"]["queries"], self.pins["queries"])
        self.assertFalse(report["scope"]["full_original_review_ledger_reaudited"])

    def test_output_is_never_overwritten(self):
        self.write_inputs()
        self.output.write_text("existing evidence\n")
        code, _, error = self.cli()
        self.assertEqual(code, 2)
        self.assertIn("refusing overwrite", error)
        self.assertEqual(self.output.read_text(), "existing evidence\n")


if __name__ == "__main__":
    unittest.main()
