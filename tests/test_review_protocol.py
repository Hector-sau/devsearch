"""Review protocol revision 5 (`primary_audit_v1`): one complete primary pass, a seeded stratified second review with escalation,
and independent adjudication; the audit, the bank and the config all apply the same declared rule.

The judgments here are simulated to exercise the rule; they are test data, not reviews.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from devsearch.data import review_workflow as wf
from devsearch.data.config import load_review_protocol
from devsearch.data.review import ReviewResponse, resolve_bank
from devsearch.data.seals import write_bank
from devsearch.data.review_protocol import ReviewProtocol, evaluate, protocol_from_table, rank_band
from devsearch.errors import ConfigError, ManifestError
from helpers import REPO
from test_review import STAMP, ReviewCase

PROTOCOL = ReviewProtocol("primary_audit_v1", 1, "judge-P", 20261007, 0.25, 0.08, 4, 0.10)


def response(item, reviewer, relevance, *, role="primary", resolved=True):
    return ReviewResponse(item, role, relevance if resolved else None, resolved, ((0, 5),) if resolved and relevance == 1 else (), "agent", reviewer, "rev-1", STAMP)


def synthetic_pool(questions=12, per_question=50, positives=(1, 2, 7, 30)):
    """{item: hidden} for `questions` questions with ranks 1..per_question, and the primary labels: positives at the given ranks."""
    items, labels = {}, {}
    for q in range(questions):
        for rank in range(1, per_question + 1):
            item = f"it_q{q:02d}r{rank:02d}"
            items[item] = {"query_id": f"q{q:02d}", "provenance": {"known_positive": False, "candidate_rank": rank, "bm25_score": 1.0}}
            labels[item] = 1 if rank in positives else 0
        outside = f"it_q{q:02d}kp"
        items[outside] = {"query_id": f"q{q:02d}", "provenance": {"known_positive": True, "candidate_rank": None, "bm25_score": None}}
        labels[outside] = 1
    return items, labels


class ProtocolRuleTests(unittest.TestCase):
    def setUp(self):
        self.items, self.labels = synthetic_pool()
        self.primary = [response(i, "judge-P", label) for i, label in self.labels.items()]

    def evaluate(self, responses, protocol=PROTOCOL):
        return evaluate(protocol, self.items, responses)

    def second(self, items, flip=()):
        return [response(i, "judge-S", 1 - self.labels[i] if i in flip else self.labels[i]) for i in items]

    def test_rank_bands_cover_the_candidate_depth_and_known_positives_outside_it(self):
        self.assertEqual([rank_band({"candidate_rank": r}) for r in (1, 5, 6, 10, 11, 20, 21, 50)], ["1-5", "1-5", "6-10", "6-10", "11-20", "11-20", "21+", "21+"])
        self.assertEqual(rank_band({"candidate_rank": None}), "outside-candidates")

    def test_nothing_is_sampled_before_the_primary_pass_is_complete(self):
        result = self.evaluate(self.primary[:-1])
        self.assertFalse(result["plan"]["primary_pass"]["complete"])
        self.assertNotIn("sample", result["plan"])
        self.assertEqual(set(result["states"].values()), {"awaiting the complete primary pass", "needs primary review"})

    def test_the_sample_is_seeded_stratified_and_covers_labels_ranks_and_questions(self):
        plan = self.evaluate(self.primary)["plan"]
        strata = plan["strata"]
        self.assertEqual(sorted(strata), sorted(f"label-{label}|rank-{band}" for label, band in
                                                [(1, "1-5"), (1, "21+"), (1, "6-10"), (1, "outside-candidates"), (0, "1-5"), (0, "6-10"), (0, "11-20"), (0, "21+")]))
        # label-0 rank 21+: 12 questions x 29 items = 348; ceil(0.08 * 348) = 28 >= min 4
        self.assertEqual(strata["label-0|rank-21+"], {**strata["label-0|rank-21+"], "items": 348, "sampled": 28})
        # label-1 rank 6-10: one positive (rank 7) per question = 12; ceil(0.25 * 12) = 3 < min 4
        self.assertEqual(strata["label-1|rank-6-10"]["sampled"], 4)
        self.assertTrue(all(s["sampled"] >= min(s["items"], 4) for s in strata.values()))
        self.assertEqual(plan["questions_in_sample"], plan["questions"], "every question has a sampled item")
        self.assertEqual(plan, self.evaluate(list(reversed(self.primary)))["plan"], "independent of response order")
        other = self.evaluate(self.primary, replace(PROTOCOL, audit_seed=1))["plan"]
        self.assertNotEqual(plan["sample"], other["sample"], "the seed decides the draw")
        self.assertEqual({k: v["sampled"] for k, v in plan["strata"].items()}, {k: v["sampled"] for k, v in other["strata"].items()})

    def test_a_question_without_a_sampled_item_gets_one_preferring_a_positive(self):
        protocol = replace(PROTOCOL, audit_rate_positive=0.01, audit_rate_negative=0.01, audit_min_per_stratum=1)
        plan = self.evaluate(self.primary, protocol)["plan"]
        self.assertTrue(plan["question_coverage_added"])
        self.assertTrue(all(self.labels[i] == 1 for i in plan["question_coverage_added"]))
        self.assertEqual(plan["questions_in_sample"], 12)

    def test_unsampled_items_wait_for_the_sample_then_become_final(self):
        result = self.evaluate(self.primary)
        sample = result["plan"]["sample"]
        self.assertEqual(sorted(result["plan"]["second_reviews_missing"]), sorted(sample))
        self.assertEqual(sum(1 for s in result["states"].values() if s == "needs second review"), len(sample))
        self.assertIn("awaiting the second-review sample", result["states"].values())
        done = self.evaluate(self.primary + self.second(sample))
        self.assertTrue(all(s.startswith("final") for s in done["states"].values()))
        self.assertEqual(done["plan"]["escalated_strata"], [])
        confirmed = [i for i, s in done["states"].items() if s == "final: primary confirmed by second review"]
        self.assertEqual(sorted(confirmed), sorted(sample))
        self.assertEqual([r.reviewer_id for r in done["basis"][sample[0]]], ["judge-P", "judge-S"])

    def test_a_stratum_over_the_threshold_is_escalated_to_a_full_second_review(self):
        sample = self.evaluate(self.primary)["plan"]["sample"]
        stratum = [i for i in sample if self.labels[i] == 0 and self.items[i]["provenance"]["candidate_rank"] and self.items[i]["provenance"]["candidate_rank"] > 20]
        flips = stratum[:3]                                  # 3 of 28 = 10.7% > 10%
        result = self.evaluate(self.primary + self.second(sample, flip=flips))
        plan = result["plan"]
        self.assertEqual(plan["escalated_strata"], ["label-0|rank-21+"])
        self.assertEqual(plan["required_second_reviews"], len(set(sample) | {i for i in self.items if self.labels[i] == 0 and (self.items[i]["provenance"]["candidate_rank"] or 0) > 20}))
        self.assertEqual(sum(1 for s in result["states"].values() if s == "needs adjudication: second review disagrees"), 3)
        self.assertEqual(sum(1 for s in result["states"].values() if s == "needs second review"), 348 - 28)
        two = self.evaluate(self.primary + self.second(sample, flip=flips[:2]))["plan"]   # 2 of 28 = 7.1%: not escalated
        self.assertEqual(two["escalated_strata"], [])

    def test_disputes_and_doubt_go_to_an_independent_adjudicator(self):
        sample = self.evaluate(self.primary)["plan"]["sample"]
        unsure = next(i for i in self.items if i not in sample)
        primary = [r if r.item_id != unsure else response(unsure, "judge-P", None, resolved=False) for r in self.primary]
        sample = self.evaluate(primary)["plan"]["sample"]   # an unresolved primary is in no stratum: it goes to adjudication instead
        self.assertNotIn(unsure, sample)
        disputed = next(i for i in sample if (self.items[i]["provenance"]["candidate_rank"] or 0) > 20 and self.labels[i] == 0)   # 1 of 28: no escalation
        responses = primary + self.second(sample, flip=[disputed])
        states = self.evaluate(responses)["states"]
        self.assertEqual(states[unsure], "needs adjudication: unresolved primary")
        self.assertEqual(states[disputed], "needs adjudication: second review disagrees")
        not_independent = self.evaluate(responses + [response(disputed, "judge-S", 1, role="adjudicator")])["states"]
        self.assertEqual(not_independent[disputed], "adjudicator is not independent")
        doubtful = self.evaluate(responses + [response(disputed, "judge-X", None, role="adjudicator", resolved=False)])["states"]
        self.assertEqual(doubtful[disputed], "adjudicator left it unresolved")
        settled = self.evaluate(responses + [response(disputed, "judge-X", None, role="adjudicator", resolved=False),
                                             response(disputed, "judge-Y", self.labels[disputed], role="adjudicator"),
                                             response(unsure, "judge-Y", 0, role="adjudicator")])
        self.assertEqual(settled["states"][disputed], "final: adjudicated")
        self.assertEqual(settled["states"][unsure], "final: adjudicated")
        self.assertTrue(all(s.startswith("final") for s in settled["states"].values()))
        self.assertEqual([r.reviewer_id for r in settled["basis"][disputed]], ["judge-Y"])

    def test_a_second_review_by_the_primary_reviewer_does_not_count(self):
        sample = self.evaluate(self.primary)["plan"]["sample"]
        again = [response(i, "judge-P", self.labels[i]) for i in sample]
        self.assertEqual(self.evaluate(self.primary + again)["plan"]["second_reviews_missing"], sorted(sample),
                         "the primary reviewer's own repeat is its first judgment, not a second review")


class ConfigTests(unittest.TestCase):
    table = {"salt": "s" * 8, "mode": "primary_audit_v1", "primary_reviewer": "judge-P", "audit_seed": 20261007, "audit_rate_positive": 0.25,
             "audit_rate_negative": 0.08, "audit_min_per_stratum": 30, "escalate_disagreement_rate": 0.10}

    def test_the_audit_mode_needs_every_parameter_declared(self):
        self.assertEqual(protocol_from_table(self.table, "t").describe()["audit_seed"], 20261007)
        for key in ("primary_reviewer", "audit_seed", "audit_rate_positive", "audit_rate_negative", "audit_min_per_stratum", "escalate_disagreement_rate"):
            with self.subTest(key=key), self.assertRaisesRegex(ConfigError, key):
                protocol_from_table({k: v for k, v in self.table.items() if k != key}, "t")
        for key, bad in (("audit_rate_negative", 0), ("escalate_disagreement_rate", 2), ("primary_reviewer", "pilot-1"), ("audit_min_per_stratum", 0)):
            with self.subTest(bad=key), self.assertRaises(ConfigError):
                protocol_from_table({**self.table, key: bad}, "t")
        with self.assertRaisesRegex(ConfigError, "min_primary_reviews"):
            protocol_from_table({**self.table, "min_primary_reviews": 2}, "t")
        with self.assertRaisesRegex(ConfigError, "audit_seed"):
            protocol_from_table({"salt": "s" * 8, "audit_seed": 1}, "t")
        self.assertEqual(protocol_from_table({"salt": "s" * 8}, "t"), ReviewProtocol())

    def test_the_shipped_data_configs_declare_their_protocol(self):
        self.assertEqual(load_review_protocol(REPO / "configs" / "data.toml").mode, "primary_audit_v1")


class AuditAndBankTests(ReviewCase):
    """The workflow audit and the bank stage apply the same rule to a real (fixture) pool."""

    def setUp(self):
        super().setUp()
        self.protocol = replace(PROTOCOL, primary_reviewer="oracle-1", audit_min_per_stratum=2)
        self.hidden = {i.item_id: {"query_id": i.query_id, "provenance": dict(i.provenance)} for i in self.pool.items}

    def test_the_bank_seals_only_with_the_sample_second_reviewed(self):
        primary = self.full_responses()
        bank = resolve_bank(self.pool, primary, self.loaded, protocol=self.protocol)
        self.assertFalse(bank.ready)
        sample = evaluate(self.protocol, self.hidden, primary)["plan"]["sample"]
        self.assertEqual(sorted(i for i, why in bank.unresolved if why == "needs second review"), sorted(sample))
        second = [self.oracle(item, reviewer=("human", "h1", None)) for item in self.pool.items if item.item_id in sample]
        bank = resolve_bank(self.pool, primary + second, self.loaded, protocol=self.protocol)
        self.assertTrue(bank.ready)
        self.assertEqual(len(bank.qrels), len(self.pool.items))
        self.assertEqual(sum(q.relevance for q in bank.qrels), len(self.positive_pairs))
        reviewed = [q for q in bank.qrels if len(q.reviews) == 2]
        self.assertEqual(len(reviewed), len(sample))
        self.assertEqual({r.reviewer_type for q in reviewed for r in q.reviews}, {"agent", "human"})
        self.assertEqual(resolve_bank(self.pool, primary, self.loaded, min_primaries=1).ready, True, "the double rule is unchanged")

    def test_the_workflow_audit_reports_the_plan_and_the_queues(self):
        lengths = {i.item_id: len(self.chunks[i.chunk_id].text) for i in self.pool.items}
        to_record = lambda r: {"item_id": r.item_id, "role": r.role, "relevance": r.relevance, "status": "resolved" if r.resolved else "unresolved",
                               "evidence_spans": [{"start": a, "end": b} for a, b in r.spans],
                               "reviewer": {"type": r.reviewer_type, "id": r.reviewer_id, "revision": r.revision}, "reviewed_at": r.reviewed_at, "notes": None}
        request = {item: {"passage": {"text": "x" * length}} for item, length in lengths.items()}
        report = wf.audit(request, [to_record(r) for r in self.full_responses()], protocol=self.protocol, hidden=self.hidden)
        self.assertFalse(report["complete"])
        self.assertEqual(report["second_review_queue"], report["second_review_plan"]["sample"])
        self.assertEqual(report["review_protocol"]["mode"], "primary_audit_v1")
        with self.assertRaisesRegex(ManifestError, "hidden provenance"):
            wf.audit(request, [], protocol=self.protocol)



class AdjudicationGateTests(ReviewCase):
    """An adjudicator settles a label but never a gate: neither the audit nor the bank seal accepts an adjudication in place of the
    complete primary pass, a required second review or the escalation decision."""

    ADJ = ("agent", "adjudicator-1", "adj-1")

    def setUp(self):
        super().setUp()
        self.protocol = replace(PROTOCOL, primary_reviewer="oracle-1", audit_min_per_stratum=2)
        self.hidden = {i.item_id: {"query_id": i.query_id, "provenance": dict(i.provenance)} for i in self.pool.items}
        self.items = {i.item_id: i for i in self.pool.items}
        self.review_dir = self.world.root / "data" / "gate-bank"
        self.review_dir.mkdir()

    def adjudicate(self, item_id, relevance=None):
        return self.oracle(self.items[item_id], role="adjudicator", reviewer=self.ADJ, relevance=relevance)

    def second(self, item_ids, flip=()):
        out = []
        for item_id in item_ids:
            item = self.items[item_id]
            truth = 1 if (item.query_id, item.chunk_id) in self.positive_pairs else 0
            out.append(self.oracle(item, reviewer=("human", "h1", None), relevance=1 - truth if item_id in flip else truth))
        return out

    def plan(self, responses):
        return evaluate(self.protocol, self.hidden, responses)["plan"]

    def assertBlocked(self, responses, item_id, state):
        """The audit leaves the item at `state` (incomplete), the bank lists it unresolved for that reason, and sealing refuses."""
        self.assertEqual(evaluate(self.protocol, self.hidden, responses)["states"][item_id], state)
        lengths = {i.item_id: len(self.chunks[i.chunk_id].text) for i in self.pool.items}
        request = {item: {"passage": {"text": "x" * length}} for item, length in lengths.items()}
        records = [{"item_id": r.item_id, "role": r.role, "relevance": r.relevance, "status": "resolved" if r.resolved else "unresolved",
                    "evidence_spans": [{"start": a, "end": b} for a, b in r.spans], "notes": None, "reviewed_at": r.reviewed_at,
                    "reviewer": {"type": r.reviewer_type, "id": r.reviewer_id, "revision": r.revision}} for r in responses]
        report = wf.audit(request, records, protocol=self.protocol, hidden=self.hidden)
        self.assertFalse(report["complete"])
        self.assertGreaterEqual(report["states"].get(state, 0), 1)
        self.assertNotIn(item_id, report["adjudication_queue"], "already adjudicated: it waits on the gate, not on another adjudicator")
        bank = resolve_bank(self.pool, responses, self.loaded, protocol=self.protocol)
        self.assertEqual(dict(bank.unresolved).get(item_id), state)
        with self.assertRaisesRegex(ManifestError, "sealing is blocked"):
            write_bank(self.review_dir, self.pool, bank, self.qrels_v1, self.loaded, self.queries)

    def assertSeals(self, responses):
        bank = resolve_bank(self.pool, responses, self.loaded, protocol=self.protocol)
        self.assertTrue(bank.ready, bank.unresolved[:3])
        write_bank(self.review_dir, self.pool, bank, self.qrels_v1, self.loaded, self.queries)
        return bank

    # case 1: the complete primary pass ------------------------------------------------------------------------------------------
    def test_an_adjudication_does_not_replace_a_missing_primary_judgment(self):
        missing = self.pool.items[0].item_id
        responses = [r for r in self.full_responses() if r.item_id != missing] + [self.adjudicate(missing)]
        self.assertBlocked(responses, missing, "needs primary review")
        self.assertFalse(self.plan(responses)["primary_pass"]["complete"], "the adjudication is not a primary judgment")

    def test_an_adjudication_does_not_bypass_an_incomplete_primary_pass(self):
        missing, doubtful = self.pool.items[0].item_id, self.pool.items[1].item_id
        responses = [r for r in self.full_responses() if r.item_id not in (missing, doubtful)]
        responses += [self.oracle(self.items[doubtful], resolved=False), self.adjudicate(doubtful)]
        self.assertBlocked(responses, doubtful, "awaiting the complete primary pass")

    # case 2: required second reviews and the escalation decision ----------------------------------------------------------------
    def test_an_adjudication_does_not_replace_a_required_second_review(self):
        primary = self.full_responses()
        sample = self.plan(primary)["sample"]
        target = sample[0]
        responses = primary + self.second(sample[1:]) + [self.adjudicate(target)]
        self.assertBlocked(responses, target, "needs second review")
        plan = self.plan(responses)
        self.assertFalse(plan["escalation_decided"], "an adjudication is not a second review")
        self.assertIn(target, plan["second_reviews_missing"])

    def test_an_adjudication_does_not_bypass_the_escalation_decision(self):
        doubtful = self.pool.items[0].item_id
        primary = [r if r.item_id != doubtful else self.oracle(self.items[doubtful], resolved=False) for r in self.full_responses()]
        sample = self.plan(primary)["sample"]
        self.assertNotIn(doubtful, sample)
        responses = primary + self.second(sample[1:]) + [self.adjudicate(doubtful)]   # one sampled item still unreviewed
        self.assertBlocked(responses, doubtful, "awaiting the second-review sample")
        self.assertSeals(responses + self.second(sample[:1]))                       # the decision made, the adjudication counts

    def test_an_adjudication_does_not_replace_the_second_review_an_escalation_requires(self):
        primary = self.full_responses()
        plan = self.plan(primary)
        stratum = next(name for name, row in plan["strata"].items() if row["items"] > row["sampled_with_question_coverage"])
        members = [i for i in self.hidden if f"label-{1 if (self.items[i].query_id, self.items[i].chunk_id) in self.positive_pairs else 0}|rank-"
                   f"{rank_band(self.hidden[i]['provenance'])}" == stratum]
        sampled = [i for i in members if i in plan["sample"]]
        rest = [i for i in members if i not in plan["sample"]]
        disputed, target = sampled[0], rest[0]
        responses = primary + self.second(plan["sample"], flip=[disputed]) + [self.adjudicate(disputed)]
        self.assertEqual(self.plan(responses)["escalated_strata"], [stratum])
        self.assertBlocked(responses + [self.adjudicate(target)], target, "needs second review")
        bank = self.assertSeals(responses + self.second(rest))
        self.assertEqual(len(bank.qrels), len(self.pool.items))

    def test_an_adjudication_of_an_undisputed_item_is_refused_not_used(self):
        primary = self.full_responses()
        sample = self.plan(primary)["sample"]
        undisputed = next(i for i in self.hidden if i not in sample)
        truth = 1 if (self.items[undisputed].query_id, self.items[undisputed].chunk_id) in self.positive_pairs else 0
        self.assertBlocked(primary + self.second(sample) + [self.adjudicate(undisputed, relevance=1 - truth)], undisputed,
                           "adjudicator on an undisputed item")

class CommandTests(unittest.TestCase):
    def test_audit_reads_the_protocol_from_the_data_config_and_cuts_second_review_batches(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)
        items, labels = synthetic_pool(questions=3, per_question=10)
        texts = {item: f"Passage {item} about zero_grad and backward." for item in items}
        (base / "request.jsonl").write_text("".join(json.dumps({"item_id": i, "query": {"text": f"Question {h['query_id']}?", "filters": {"sources": [], "versions": {}}},
                                                               "passage": {"title": "T", "section": "S", "text": texts[i], "code_fragment": False, "source_url": None,
                                                                           "source_location": []}}) + "\n" for i, h in items.items()))
        (base / "hidden-provenance.jsonl").write_text(json.dumps({"item_id": "_pool", "salt": "x", "protocol": "prejudged_fixed_candidates_v1"}) + "\n" +
                                                      "".join(json.dumps({"item_id": i, "query_id": h["query_id"], "chunk_id": "c", "provenance": h["provenance"]}) + "\n" for i, h in items.items()))
        (base / "data.toml").write_text('[review]\nsalt = "ssssssss"\nmode = "primary_audit_v1"\nprimary_reviewer = "judge-P"\naudit_seed = 7\n'
                                        'audit_rate_positive = 0.25\naudit_rate_negative = 0.08\naudit_min_per_stratum = 2\nescalate_disagreement_rate = 0.1\n')
        (base / "judged.jsonl").write_text("".join(json.dumps({"item_id": i, "relevance": labels[i], "evidence_quote": "zero_grad" if labels[i] else "", "note": ""}) + "\n" for i in items))
        (base / "responses").mkdir()

        def review(*args):
            return subprocess.run([sys.executable, "-m", "devsearch.review", *map(str, args)], capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src")})
        imported = review("import", "--request", base / "request.jsonl", "--judgments", base / "judged.jsonl", "--reviewer-id", "judge-P", "--reviewer-type", "agent",
                          "--revision", "r1", "--role", "primary", "--output", base / "responses" / "P.jsonl")
        self.assertEqual(imported.returncode, 0, imported.stderr)
        audit = review("audit", "--request", base / "request.jsonl", "--responses-dir", base / "responses", "--output", base / "audit.json", "--data-config", base / "data.toml")
        self.assertEqual(audit.returncode, 3, audit.stderr)
        report = json.loads((base / "audit.json").read_text())
        self.assertTrue(report["second_review_queue"])
        (base / "S").mkdir()
        cut = review("batches", "--request", base / "request.jsonl", "--output-dir", base / "S", "--pass", "S", "--items", base / "audit.json", "--queue", "second-review")
        self.assertEqual(cut.returncode, 0, cut.stderr)
        self.assertEqual(sorted(json.loads(line)["item_id"] for p in (base / "S").glob("*.jsonl") for line in p.read_text().splitlines()),
                         report["second_review_queue"])
        second = [json.loads(line) for p in (base / "S").glob("*.jsonl") for line in p.read_text().splitlines()]
        (base / "second.jsonl").write_text("".join(json.dumps({"item_id": r["item_id"], "relevance": labels[r["item_id"]],
                                                              "evidence_quote": "zero_grad" if labels[r["item_id"]] else "", "note": ""}) + "\n" for r in second))
        imported = review("import", "--request", base / "request.jsonl", "--judgments", base / "second.jsonl", "--reviewer-id", "judge-S", "--reviewer-type", "human",
                          "--role", "primary", "--output", base / "responses" / "S.jsonl")
        self.assertEqual(imported.returncode, 0, imported.stderr)
        audit = review("audit", "--request", base / "request.jsonl", "--responses-dir", base / "responses", "--output", base / "audit2.json", "--data-config", base / "data.toml",
                       "--merged", base / "all.jsonl")
        self.assertEqual(audit.returncode, 0, audit.stderr)
        self.assertTrue((base / "all.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
