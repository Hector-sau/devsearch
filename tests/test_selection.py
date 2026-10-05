"""Protocol loading and the validation-only selection rule (docs/10 §6)."""

import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path

from devsearch.errors import ConfigError
from devsearch.evaluation.protocol import load_protocol
from devsearch.evaluation.selection import EpochResult, RunResult, SelectionBlocked, best_epoch, select
from helpers import CONFIGS

BASELINE = {"group_mrr": 0.50, "group_ndcg": 0.55}


def epochs(*triples):
    return tuple(EpochResult(i, mrr, ndcg, f"epoch-{i:04d}", f"{i:064x}") for i, (mrr, ndcg) in enumerate(triples, start=1))


def run(recipe, seed, *triples):
    return RunResult(recipe, seed, epochs(*triples))


class ProtocolTests(unittest.TestCase):
    def test_the_shipped_protocol_loads_and_matches_the_documented_defaults(self):
        protocol = load_protocol(CONFIGS / "protocol.toml")
        self.assertEqual((protocol.max_pair_tokens, protocol.max_query_tokens, protocol.epochs), (256, 64, 3))
        self.assertEqual([(r.recipe_id, r.mode, r.learning_rate) for r in protocol.recipes], [("frozen", "frozen", 1e-3), ("full", "full", 2e-5)])
        self.assertEqual((protocol.seeds, protocol.deployment_seed), ((42,), 42))
        self.assertEqual((protocol.min_group_mrr_gain, protocol.max_p95_ms), (0.02, 2000.0))
        self.assertEqual((protocol.latency_warmup, protocol.latency_passes, protocol.latency_queries), (10, 3, 50))
        self.assertEqual((protocol.bootstrap_confidence, protocol.bootstrap_repetitions, protocol.bootstrap_seed), (0.95, 10000, 20261005))
        self.assertEqual(protocol.inference_batches, (4, 2, 1, 8, 16))
        self.assertEqual(dict(protocol.training)["effective_batch"], 16)
        self.assertEqual(protocol.run_ids(), [("frozen", 42), ("full", 42)])
        self.assertTrue(protocol.protocol_id.startswith("protocol-"))

    def test_the_protocol_id_changes_with_any_declared_value(self):
        protocol = load_protocol(CONFIGS / "protocol.toml")
        other = dataclasses.replace(protocol, min_group_mrr_gain=0.03)
        self.assertNotEqual(protocol.protocol_id, other.protocol_id)

    def test_invalid_protocols_fail(self):
        good = (CONFIGS / "protocol.toml").read_text()
        cases = {
            "deployment seed not declared": good.replace("deployment_seed = 42", "deployment_seed = 7"),
            "more than three epochs": good.replace("epochs = 3", "epochs = 4"),
            "duplicate recipe ids": good.replace('id = "full"', 'id = "frozen"'),
            "unknown mode": good.replace('mode = "full"', 'mode = "partial"'),
            "unknown key": good.replace("[selection]", "[selection]\nextra = 1"),
            "no recipes": good.split("[[recipe]]")[0] + "[training]" + good.split("[training]")[1],
            "query reserve above the pair limit": good.replace("max_query_tokens = 64", "max_query_tokens = 300"),
            "too few bootstrap resamples": good.replace("repetitions = 10000", "repetitions = 10"),
            "bad pos_weight": good.replace('pos_weight = "auto"', 'pos_weight = "heavy"'),
        }
        directory = Path(tempfile.mkdtemp(prefix="devsearch-proto-"))
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        for name, text in cases.items():
            path = directory / "p.toml"
            path.write_text(text)
            with self.subTest(name), self.assertRaises(ConfigError):
                load_protocol(path)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(CONFIGS / "protocol.toml")

    def decide(self, runs, latency=None, baseline=BASELINE, protocol=None):
        protocol = protocol or self.protocol
        return select(protocol, baseline=baseline, runs=runs, latency_p95_ms=latency if latency is not None else {"frozen": 100.0, "full": 100.0}, plan_id="plan-x")

    def good(self):
        return {("frozen", 42): run("frozen", 42, (0.52, 0.56), (0.55, 0.58), (0.54, 0.57)),
                ("full", 42): run("full", 42, (0.60, 0.62), (0.66, 0.68), (0.64, 0.66))}

    def test_best_epoch_prefers_mrr_then_ndcg_then_the_earlier_epoch(self):
        self.assertEqual(best_epoch(run("r", 1, (0.5, 0.5), (0.6, 0.5), (0.6, 0.4))).epoch, 2)
        self.assertEqual(best_epoch(run("r", 1, (0.6, 0.5), (0.6, 0.7), (0.6, 0.6))).epoch, 2)
        self.assertEqual(best_epoch(run("r", 1, (0.6, 0.5), (0.6, 0.5), (0.6, 0.5))).epoch, 1)

    def test_the_winner_is_the_qualifying_recipe_with_the_highest_mean_mrr_and_its_deployment_checkpoint(self):
        record = self.decide(self.good())
        decision = record["decision"]
        self.assertEqual((decision["default_mode"], decision["recipe_id"], decision["seed"], decision["checkpoint_id"]), ("rerank", "full", 42, "epoch-0002"))
        full = next(r for r in record["recipes"] if r["recipe_id"] == "full")
        self.assertTrue(full["qualifies"] and full["latency_ok"])
        self.assertAlmostEqual(full["mean_mrr_gain"], 0.16)
        self.assertEqual(record["plan_id"], "plan-x")
        self.assertEqual(record["protocol_id"], self.protocol.protocol_id)
        self.assertEqual(self.decide(self.good())["selection_id"], record["selection_id"], "the record is deterministic")

    def test_the_gain_threshold_is_inclusive_and_just_below_fails(self):
        weak_full = run("full", 42, (0.1, 0.1), (0.1, 0.1), (0.1, 0.1))
        exactly = {("frozen", 42): run("frozen", 42, (0.52, 0.56), (0.52, 0.56), (0.52, 0.56)), ("full", 42): weak_full}
        self.assertEqual(self.decide(exactly)["decision"]["recipe_id"], "frozen")                # +0.02 passes
        just_below = {("frozen", 42): run("frozen", 42, (0.5199, 0.56), (0.5199, 0.56), (0.5199, 0.56)), ("full", 42): weak_full}
        self.assertEqual(self.decide(just_below)["decision"]["default_mode"], "baseline")        # +0.0199 does not
        no_ndcg_gain = {("frozen", 42): run("frozen", 42, (0.52, 0.55), (0.52, 0.55), (0.52, 0.55)), ("full", 42): weak_full}
        self.assertEqual(self.decide(no_ndcg_gain)["decision"]["recipe_id"], "frozen")           # an unchanged nDCG is non-negative

    def test_a_group_ndcg_decline_disqualifies_even_with_a_large_mrr_gain(self):
        runs = self.good()
        runs[("full", 42)] = run("full", 42, (0.8, 0.54), (0.8, 0.54), (0.8, 0.54))
        record = self.decide(runs)
        self.assertEqual(record["decision"]["recipe_id"], "frozen")
        self.assertFalse(next(r for r in record["recipes"] if r["recipe_id"] == "full")["qualifies"])

    def test_latency_is_a_hard_gate_and_baseline_is_the_honest_default_when_nothing_qualifies(self):
        record = self.decide(self.good(), latency={"frozen": 3000.0, "full": 2000.5})
        self.assertEqual(record["decision"]["default_mode"], "baseline")
        self.assertIn("BM25", record["decision"]["reason"])
        self.assertEqual(self.decide(self.good(), latency={"frozen": 3000.0, "full": 2000.0})["decision"]["recipe_id"], "full")

    def test_missing_latency_for_a_quality_qualifier_blocks_the_selection(self):
        with self.assertRaisesRegex(SelectionBlocked, "no latency"):
            self.decide(self.good(), latency={"frozen": 100.0, "full": None})
        with self.assertRaisesRegex(SelectionBlocked, "no latency"):
            self.decide(self.good(), latency={"frozen": 100.0})
        runs = {("frozen", 42): run("frozen", 42, (0.1, 0.1), (0.1, 0.1), (0.1, 0.1)), ("full", 42): run("full", 42, (0.1, 0.1), (0.1, 0.1), (0.1, 0.1))}
        self.assertEqual(self.decide(runs, latency={})["decision"]["default_mode"], "baseline")       # nothing could qualify, latency is moot

    def test_a_missing_or_incomplete_required_run_blocks_instead_of_selecting_survivors(self):
        runs = self.good()
        del runs[("frozen", 42)]
        with self.assertRaisesRegex(SelectionBlocked, "missing"):
            self.decide(runs)
        runs[("frozen", 42)] = None
        with self.assertRaisesRegex(SelectionBlocked, "missing"):
            self.decide(runs)
        runs[("frozen", 42)] = run("frozen", 42, (0.6, 0.6), (0.6, 0.6))
        with self.assertRaisesRegex(SelectionBlocked, "completed epochs"):
            self.decide(runs)
        runs[("frozen", 42)] = run("frozen", 42, (0.6, 0.6), (float("nan"), 0.6), (0.6, 0.6))
        with self.assertRaisesRegex(SelectionBlocked, "non-finite"):
            self.decide(runs)
        runs[("frozen", 42)] = run("full", 42, (0.6, 0.6), (0.6, 0.6), (0.6, 0.6))
        with self.assertRaisesRegex(SelectionBlocked, "filed under"):
            self.decide(runs)

    def test_a_non_finite_baseline_blocks(self):
        with self.assertRaises(SelectionBlocked):
            self.decide(self.good(), baseline={"group_mrr": float("nan"), "group_ndcg": 0.5})
        with self.assertRaises(SelectionBlocked):
            self.decide(self.good(), baseline={"group_mrr": 0.5})

    def test_equal_recipes_break_ties_by_ndcg_then_latency_then_recipe_id(self):
        same = ((0.7, 0.7),) * 3
        runs = {("frozen", 42): run("frozen", 42, *same), ("full", 42): run("full", 42, *same)}
        self.assertEqual(self.decide(runs, latency={"frozen": 300.0, "full": 100.0})["decision"]["recipe_id"], "full")
        self.assertEqual(self.decide(runs, latency={"frozen": 100.0, "full": 100.0})["decision"]["recipe_id"], "frozen")
        better_ndcg = {("frozen", 42): run("frozen", 42, *same), ("full", 42): run("full", 42, *(((0.7, 0.8),) * 3))}
        self.assertEqual(self.decide(better_ndcg, latency={"frozen": 1.0, "full": 1999.0})["decision"]["recipe_id"], "full")

    def test_a_seed_extension_uses_the_declared_deployment_seed_not_the_best_seed(self):
        protocol = dataclasses.replace(self.protocol, seeds=(42, 43, 44))
        def seeds(recipe, a, b, c):
            return {(recipe, 42): run(recipe, 42, *((a, a),) * 3), (recipe, 43): run(recipe, 43, *((b, b),) * 3), (recipe, 44): run(recipe, 44, *((c, c),) * 3)}
        runs = {**seeds("frozen", 0.1, 0.1, 0.1), **seeds("full", 0.62, 0.9, 0.58)}
        record = self.decide(runs, protocol=protocol)
        self.assertEqual(record["decision"]["seed"], 42)
        self.assertEqual(record["decision"]["checkpoint_sha256"], f"{1:064x}")
        full = next(r for r in record["recipes"] if r["recipe_id"] == "full")
        self.assertAlmostEqual(full["mean_group_mrr"], (0.62 + 0.9 + 0.58) / 3)
        self.assertEqual([s["seed"] for s in full["seeds"]], [42, 43, 44])
        # a weak deployment seed vetoes the recipe even though the mean and the best seed look excellent
        weak = {**seeds("frozen", 0.1, 0.1, 0.1), **seeds("full", 0.45, 0.95, 0.95)}
        self.assertEqual(self.decide(weak, protocol=protocol)["decision"]["default_mode"], "baseline")
        # a missing extension seed blocks
        partial = dict(runs)
        del partial[("full", 44)]
        with self.assertRaisesRegex(SelectionBlocked, "missing"):
            self.decide(partial, protocol=protocol)


if __name__ == "__main__":
    unittest.main()
