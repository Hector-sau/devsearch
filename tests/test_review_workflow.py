"""The candidate-review workflow: blinded judge batches, judgment import, the coverage audit and the adjudication queue.

Missing or incomplete judgments are never filled in: they keep the audit incomplete (exit 3), and only a complete audit writes the
merged responses the bank stage reads.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from devsearch.data import review_workflow as wf
from devsearch.errors import ManifestError
from helpers import REPO

PASSAGES = {
    "it_aaaaaaaaaaaaaaaaaaaaaaaa": "Call optimizer.zero_grad() before\nbackward() so gradients do not accumulate across steps.",
    "it_bbbbbbbbbbbbbbbbbbbbbbbb": "The DataLoader shuffles the dataset at every epoch when shuffle is true.",
    "it_cccccccccccccccccccccccc": "Tensors can be moved to a device with the to() method.",
}


def request_records():
    return [{"item_id": item, "query": {"text": "How do I reset gradients between steps?", "filters": {"sources": [], "versions": {}}},
             "passage": {"title": "Optimizers", "section": "Training loop", "text": text, "code_fragment": False, "source_url": None,
                         "source_location": []}} for item, text in PASSAGES.items()]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.request = {r["item_id"]: r for r in request_records()}
        self.a, self.b, self.c = sorted(PASSAGES)

    def judge(self, reviewer, lines, role="primary", reviewer_type="agent"):
        responses, _ = wf.import_judgments(lines, self.request, reviewer_type=reviewer_type, reviewer_id=reviewer,
                                           revision="judge-v1" if reviewer_type == "agent" else None, role=role, reviewed_at="2026-10-06T00:00:00Z")
        return responses

    def all_zero_except_a(self, quote="zero_grad() before backward()"):
        return [{"item_id": self.a, "relevance": 1, "evidence_quote": quote, "note": ""},
                {"item_id": self.b, "relevance": 0, "evidence_quote": "", "note": ""},
                {"item_id": self.c, "relevance": 0, "evidence_quote": "", "note": ""}]

    def test_batches_are_blinded_compact_complete_and_ordered_per_pass(self):
        first = wf.judge_batches(self.request, pass_name="A", batch_size=2, seed=1)
        second = wf.judge_batches(self.request, pass_name="B", batch_size=2, seed=1)
        self.assertEqual([len(b) for b in first], [2, 1])
        self.assertEqual(sorted(i["item_id"] for b in first for i in b), sorted(PASSAGES))
        self.assertEqual(set(first[0][0]), {"item_id", "question", "title", "section", "text"})
        self.assertEqual(first, wf.judge_batches(self.request, pass_name="A", batch_size=2, seed=1), "deterministic")
        self.assertNotEqual([i["item_id"] for b in first for i in b], [i["item_id"] for b in second for i in b])

    def test_a_leaky_request_is_refused(self):
        path = Path(tempfile.mkdtemp()) / "request.jsonl"
        self.addCleanup(shutil.rmtree, path.parent, ignore_errors=True)
        records = request_records()
        records[0]["passage"]["rank"] = 3
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ManifestError, "leaks"):
            wf.load_request(path)

    def test_import_maps_quotes_to_spans_and_never_turns_doubt_into_zero(self):
        responses = self.judge("A", self.all_zero_except_a())
        span = responses[0]["evidence_spans"][0]
        self.assertEqual(PASSAGES[self.a][span["start"]:span["end"]], "zero_grad() before\nbackward()")
        doubtful = self.judge("A", [{"item_id": self.a, "relevance": "unsure", "evidence_quote": "", "note": ""},
                                    {"item_id": self.b, "relevance": 1, "evidence_quote": "not in the passage", "note": ""}])
        self.assertEqual([(r["status"], r["relevance"]) for r in doubtful], [("unresolved", None), ("unresolved", None)])
        with self.assertRaisesRegex(ManifestError, "not in the review request"):
            self.judge("A", [{"item_id": "it_dddddddddddddddddddddddd", "relevance": 0}])
        with self.assertRaisesRegex(ManifestError, "judged twice"):
            self.judge("A", [{"item_id": self.a, "relevance": 0}, {"item_id": self.a, "relevance": 0}])
        human = self.judge("person-1", self.all_zero_except_a(), reviewer_type="human")
        self.assertEqual(human[0]["reviewer"], {"type": "human", "id": "person-1", "revision": None})

    def test_coverage_needs_two_distinct_unanimous_primaries_or_an_adjudication(self):
        one = wf.audit(self.request, self.judge("A", self.all_zero_except_a()))
        self.assertFalse(one["complete"])
        self.assertEqual(one["states"], {"needs primary reviews (1 of 2)": 3})
        self.assertEqual(one["needs_primary"], sorted(PASSAGES))
        same_twice = wf.audit(self.request, self.judge("A", self.all_zero_except_a()) * 2)
        self.assertFalse(same_twice["complete"], "one reviewer twice is still one review")
        both = wf.audit(self.request, self.judge("A", self.all_zero_except_a()) + self.judge("B", self.all_zero_except_a()))
        self.assertEqual((both["complete"], both["final"], both["coverage"]), (True, 3, 1.0))
        b_lines = self.all_zero_except_a()
        b_lines[1] = {"item_id": self.b, "relevance": 1, "evidence_quote": "shuffles the dataset", "note": ""}
        b_lines[2] = {"item_id": self.c, "relevance": "unsure", "evidence_quote": "", "note": ""}
        split = self.judge("A", self.all_zero_except_a()) + self.judge("B", b_lines)
        report = wf.audit(self.request, split)
        self.assertFalse(report["complete"])
        self.assertEqual(report["adjudication_queue"], sorted([self.b, self.c]))
        self.assertEqual(report["states"]["needs adjudication: primaries disagree"], 1)
        self.assertEqual(report["states"]["needs adjudication: unresolved primary"], 1)
        settled = self.judge("adj-1", [{"item_id": self.b, "relevance": 0, "evidence_quote": "", "note": ""},
                                       {"item_id": self.c, "relevance": "unsure", "evidence_quote": "", "note": ""}], role="adjudicator")
        report = wf.audit(self.request, split + settled)
        self.assertFalse(report["complete"], "an adjudicator who is unsure settles nothing")
        self.assertEqual(report["adjudication_queue"], [self.c])
        final = self.judge("adj-2", [{"item_id": self.c, "relevance": 0, "evidence_quote": "", "note": ""}], role="adjudicator")
        report = wf.audit(self.request, split + settled[:1] + final)
        self.assertTrue(report["complete"])


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-review-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.request = self.dir / "review-request.jsonl"
        self.request.write_text("".join(json.dumps(r) + "\n" for r in request_records()))

    def run_review(self, *args):
        result = subprocess.run([sys.executable, "-m", "devsearch.review", *args], capture_output=True, text=True, timeout=120,
                                env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
        return result.returncode, result.stdout, result.stderr

    def judgments(self, name, quote="zero_grad() before backward()"):
        a, b, c = sorted(PASSAGES)
        path = self.dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(line) + "\n" for line in [
            {"item_id": a, "relevance": 1, "evidence_quote": quote, "note": ""},
            {"item_id": b, "relevance": 0, "evidence_quote": "", "note": ""}, {"item_id": c, "relevance": 0, "evidence_quote": "", "note": ""}]))
        return path

    def test_batches_import_and_audit_block_until_coverage_is_complete(self):
        (self.dir / "batches").mkdir()
        code, out, err = self.run_review("batches", "--request", str(self.request), "--output-dir", str(self.dir / "batches"), "--pass", "A", "--batch-size", "2")
        self.assertEqual(code, 0, err)
        self.assertEqual(sorted(p.name for p in (self.dir / "batches").iterdir()), ["A-0001.jsonl", "A-0002.jsonl"])
        responses = self.dir / "responses"
        code, _, err = self.run_review("import", "--request", str(self.request), "--judgments", str(self.judgments("a")), "--reviewer-id", "judge-a",
                                       "--reviewer-type", "agent", "--revision", "judge-v1", "--role", "primary", "--output", str(responses / "a.jsonl"))
        self.assertEqual(code, 0, err)
        merged = self.dir / "all-responses.jsonl"
        code, out, err = self.run_review("audit", "--request", str(self.request), "--responses-dir", str(responses), "--output", str(self.dir / "audit.json"),
                                         "--merged", str(merged))
        self.assertEqual(code, 3, "one pass of two: incomplete")
        self.assertIn("sealing and formal evaluation stay blocked", err)
        self.assertFalse(merged.exists(), "nothing for the bank while coverage is incomplete")
        code, _, err = self.run_review("import", "--request", str(self.request), "--judgments", str(self.judgments("b")), "--reviewer-id", "judge-b",
                                       "--reviewer-type", "agent", "--revision", "judge-v1", "--role", "primary", "--output", str(responses / "b.jsonl"))
        self.assertEqual(code, 0, err)
        code, out, err = self.run_review("import", "--request", str(self.request), "--judgments", str(self.judgments("b")), "--reviewer-id", "judge-b",
                                         "--reviewer-type", "agent", "--revision", "judge-v1", "--role", "primary", "--output", str(responses / "b.jsonl"))
        self.assertEqual(code, 1, "an import never overwrites")
        code, out, err = self.run_review("audit", "--request", str(self.request), "--responses-dir", str(responses), "--output", str(self.dir / "audit.json"),
                                         "--merged", str(merged))
        self.assertEqual(code, 0, err)
        self.assertEqual(len(merged.read_text().splitlines()), 6)
        self.assertTrue(json.loads((self.dir / "audit.json").read_text())["complete"])

    def test_an_agent_import_needs_a_revision_and_a_human_import_has_none(self):
        code, _, err = self.run_review("import", "--request", str(self.request), "--judgments", str(self.judgments("x")), "--reviewer-id", "j",
                                       "--reviewer-type", "agent", "--role", "primary", "--output", str(self.dir / "r.jsonl"))
        self.assertEqual(code, 1)
        self.assertIn("--revision", err)
        code, _, err = self.run_review("import", "--request", str(self.request), "--judgments", str(self.judgments("y")), "--reviewer-id", "p",
                                       "--reviewer-type", "human", "--revision", "x", "--role", "primary", "--output", str(self.dir / "r.jsonl"))
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
