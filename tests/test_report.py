"""Final report assembly: shared denominator, group-macro scoring, honest outcomes, provenance, tamper detection."""

import dataclasses
import json
import unittest

from devsearch.data.queries import Qrel, Review
from devsearch.errors import ManifestError
from devsearch.evaluation.protocol import load_protocol
from devsearch.evaluation.ranking import order_by_score, ranked_record
from devsearch.evaluation.report import BASELINE, build_final_report, render_markdown, verify_report
from devsearch.evaluation.selection import EpochResult, RunResult, select
from helpers import CONFIGS

REVIEW = (Review("agent", "r1", "accepted", "2026-10-05T12:00:00Z", "wf-1"),)
GROUPS = {f"q{i}": f"g{(i - 1) // 2}" for i in range(1, 13)}          # 6 groups of 2 queries
CANDIDATES = [f"c{n}" for n in range(1, 11)]                           # ten candidates per query


def qrels():
    return [Qrel(q, f"c{(int(q[1:]) % 4) + 1}", 1, (), "pooled-review", "accepted", REVIEW) for q in sorted(GROUPS)]


def system(position_of_positive):
    """A ranking that places each query's known positive at 1-based `position_of_positive(query)`."""
    records = []
    for q in sorted(GROUPS):
        positive = f"c{(int(q[1:]) % 4) + 1}"
        others = [c for c in CANDIDATES if c != positive]
        position = position_of_positive(q)
        ordered = others[:position - 1] + [positive] + others[position - 1:]
        scores = [float(len(ordered) - i) for i in range(len(ordered))]
        records.append(ranked_record(q, "cands-" + q, order_by_score(ordered, scores)))
    return records


class ReportCase(unittest.TestCase):
    def setUp(self):
        self.protocol = dataclasses.replace(load_protocol(CONFIGS / "protocol.toml"), bootstrap_repetitions=500)
        baseline_epochs = tuple(EpochResult(i, 0.2, 0.3, f"epoch-{i:04d}", f"{i:064x}") for i in (1, 2, 3))
        better = tuple(EpochResult(i, 0.9, 0.9, f"epoch-{i:04d}", f"{i:064x}") for i in (1, 2, 3))
        self.selection = select(self.protocol, baseline={"group_mrr": 0.2, "group_ndcg": 0.3},
                                runs={("frozen", 42): RunResult("frozen", 42, baseline_epochs), ("full", 42): RunResult("full", 42, better)},
                                latency_p95_ms={"frozen": 10.0, "full": 10.0}, plan_id="plan-1")
        self.provenance = {"query_origins": {"synthetic": 12}, "reviewers": {"agent": 100, "human": 0}}
        self.seals = {"initial": "a" * 64, "final": "b" * 64}

    def build(self, systems=None, meta=None, qrels_=None, selection=None):
        systems = systems or {BASELINE: system(lambda q: 3), "frozen-seed42": system(lambda q: 3), "full-seed42": system(lambda q: 1)}
        meta = meta or {BASELINE: {"role": "baseline"}, "frozen-seed42": {"role": "locked", "recipe_id": "frozen", "seed": 42},
                        "full-seed42": {"role": "locked", "recipe_id": "full", "seed": 42}}
        return build_final_report(plan_id="plan-1", protocol=self.protocol, selection=selection or self.selection, systems=systems, system_meta=meta,
                                  qrels=qrels_ or qrels(), group_of=GROUPS, provenance=self.provenance, seals=self.seals)


class ScoringTests(ReportCase):
    def test_known_answer_metrics_per_system(self):
        report = self.build().report
        by_name = {s["system"]: s for s in report["systems"]}
        self.assertAlmostEqual(by_name[BASELINE]["group_macro"]["mrr@10"], 1 / 3)
        self.assertAlmostEqual(by_name["full-seed42"]["group_macro"]["mrr@10"], 1.0)
        self.assertAlmostEqual(by_name["full-seed42"]["group_macro"]["ndcg@10"], 1.0)
        self.assertAlmostEqual(by_name[BASELINE]["group_macro"]["ndcg@10"], 1 / 2)          # 1 / log2(3 + 1) = 0.5
        self.assertAlmostEqual(by_name[BASELINE]["group_macro"]["recall@50"], 1.0)
        self.assertEqual((report["denominator"]["queries_scored"], report["denominator"]["groups_scored"]), (12, 6))

    def test_queries_without_a_confirmed_positive_are_listed_not_scored_or_divided(self):
        partial = [r for r in qrels() if r.query_id != "q1"]
        report = self.build(qrels_=partial).report
        self.assertEqual(report["denominator"]["queries_without_confirmed_positive"], ["q1"])
        self.assertEqual(report["denominator"]["queries_scored"], 11)
        self.assertEqual(report["denominator"]["groups_scored"], 6)

    def test_a_system_must_cover_the_same_queries_as_the_baseline(self):
        systems = {BASELINE: system(lambda q: 3), "frozen-seed42": system(lambda q: 3)[:-1]}
        meta = {BASELINE: {"role": "baseline"}, "frozen-seed42": {"role": "locked", "recipe_id": "frozen", "seed": 42}}
        with self.assertRaisesRegex(ManifestError, "same queries"):
            self.build(systems=systems, meta=meta)
        with self.assertRaisesRegex(ManifestError, "metadata"):
            self.build(meta={BASELINE: {"role": "baseline"}})
        with self.assertRaisesRegex(ManifestError, "baseline"):
            self.build(systems={"frozen-seed42": system(lambda q: 3)}, meta={"frozen-seed42": {"role": "locked"}})


class ComparisonTests(ReportCase):
    def test_the_selected_deployment_system_is_primary_and_outcomes_follow_the_interval(self):
        report = self.build().report
        comparisons = {c["system"]: c for c in report["comparisons"]}
        self.assertEqual(report["default_mode"], "rerank")
        self.assertTrue(comparisons["full-seed42"]["primary"])
        self.assertFalse(comparisons["frozen-seed42"]["primary"])
        full = comparisons["full-seed42"]["differences"]["mrr@10"]
        self.assertTrue(full["outcome"].startswith("improved"))
        self.assertAlmostEqual(full["observed_difference"], 2 / 3)
        same = comparisons["frozen-seed42"]["differences"]["mrr@10"]
        self.assertEqual(same["observed_difference"], 0.0)
        self.assertTrue(same["outcome"].startswith("inconclusive"))
        self.assertEqual((full["repetitions"], full["seed"]), (500, 20261005))

    def test_a_reranker_worse_than_bm25_is_reported_as_worse(self):
        systems = {BASELINE: system(lambda q: 2), "frozen-seed42": system(lambda q: 2), "full-seed42": system(lambda q: 6)}
        report = self.build(systems=systems).report
        worse = next(c for c in report["comparisons"] if c["system"] == "full-seed42")["differences"]["mrr@10"]
        self.assertTrue(worse["outcome"].startswith("worse"))
        self.assertLess(worse["observed_difference"], 0)

    def test_when_baseline_is_the_default_no_comparison_is_primary(self):
        none_selected = dict(self.selection, decision={**self.selection["decision"], "default_mode": "baseline", "recipe_id": None, "seed": None,
                                                       "checkpoint_id": None, "checkpoint_sha256": None})
        report = self.build(selection=none_selected).report
        self.assertEqual(report["default_mode"], "baseline")
        self.assertFalse(any(c["primary"] for c in report["comparisons"]))
        self.assertTrue(any(c["deployment_seed_comparison"] for c in report["comparisons"]))

    def test_a_seed_extension_adds_recipe_mean_intervals_conditional_on_the_seeds(self):
        protocol = dataclasses.replace(self.protocol, seeds=(42, 43))
        systems = {BASELINE: system(lambda q: 3)}
        meta = {BASELINE: {"role": "baseline"}}
        for recipe, position in (("frozen", 3), ("full", 1)):
            for seed in (42, 43):
                systems[f"{recipe}-seed{seed}"] = system(lambda q, p=position: p)
                meta[f"{recipe}-seed{seed}"] = {"role": "locked", "recipe_id": recipe, "seed": seed}
        epochs = tuple(EpochResult(i, 0.5, 0.5, f"epoch-{i:04d}", f"{i:064x}") for i in (1, 2, 3))
        selection = select(protocol, baseline={"group_mrr": 0.2, "group_ndcg": 0.3},
                           runs={(r, s): RunResult(r, s, epochs) for r in ("frozen", "full") for s in (42, 43)},
                           latency_p95_ms={"frozen": 1.0, "full": 1.0}, plan_id="plan-1")
        report = build_final_report(plan_id="plan-1", protocol=protocol, selection=selection, systems=systems, system_meta=meta, qrels=qrels(),
                                    group_of=GROUPS, provenance=self.provenance, seals=self.seals).report
        self.assertEqual({m["recipe_id"] for m in report["recipe_means"]}, {"frozen", "full"})
        self.assertIn("not a training-variability interval", report["recipe_means"][0]["note"])
        self.assertEqual(len(report["comparisons"]), 4, "every seed is reported separately")


class IntegrityTests(ReportCase):
    def test_the_id_is_stable_and_edits_are_detected(self):
        first, second = self.build(), self.build()
        self.assertEqual(first.report["final_report_id"], second.report["final_report_id"])
        verify_report(first.report, first.per_query)
        edited = json.loads(json.dumps(first.report))
        edited["systems"][0]["group_macro"]["mrr@10"] = 0.99
        with self.assertRaisesRegex(ManifestError, "final_report_id"):
            verify_report(edited, first.per_query)
        with self.assertRaisesRegex(ManifestError, "per-query rows"):
            verify_report(first.report, first.per_query + b"\n")
        incomplete = dict(first.report, status="partial")
        with self.assertRaises(ManifestError):
            verify_report(incomplete, first.per_query)

    def test_per_query_rows_cover_every_system_and_scored_query(self):
        built = self.build()
        rows = [json.loads(line) for line in built.per_query.decode().splitlines()]
        self.assertEqual(len(rows), 3 * 12)
        self.assertEqual({r["system"] for r in rows}, {BASELINE, "frozen-seed42", "full-seed42"})
        sample = next(r for r in rows if r["system"] == "full-seed42" and r["query_id"] == "q1")
        self.assertEqual((sample["mrr@10"], len(sample["top10"]), sample["group_id"]), (1.0, 10, "g0"))

    def test_provenance_is_labelled_synthetic_and_agent_in_the_report_and_the_summary(self):
        built = self.build()
        joined = " ".join(built.report["limitations"])
        self.assertIn("SYNTHETIC", joined)
        self.assertIn("AGENT review", joined)
        self.assertIn("not exhaustive relevance", joined)
        self.assertIn("SYNTHETIC", built.markdown)
        self.assertEqual(render_markdown(built.report), built.markdown)
        human = dict(self.provenance, reviewers={"agent": 10, "human": 5}, query_origins={"real": 3})
        self.provenance = human
        limits = " ".join(self.build().report["limitations"])
        self.assertNotIn("SYNTHETIC", limits)
        self.assertNotIn("All judgments come from AGENT", limits)
        self.assertIn("mix agent and human", limits)

    def test_the_summary_states_the_default_mode_and_every_interval(self):
        text = self.build().markdown
        self.assertIn("Default mode chosen on validation only: **rerank**", text)
        self.assertIn("(primary)", text)
        self.assertIn("improved", text)
        self.assertIn("inconclusive", text)


if __name__ == "__main__":
    unittest.main()
