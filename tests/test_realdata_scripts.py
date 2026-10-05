"""scripts/realdata: passage selection, mechanical checks of generated questions, draft building. Pure parts run everywhere; the
end-to-end flow on the synthetic rehearsal corpus needs the `tokenizers` library."""

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from helpers import REPO
from mltest import requires_ml

SCRIPTS = REPO / "scripts" / "realdata"


def load(name):
    spec = importlib.util.spec_from_file_location(f"realdata_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


select = load("select_passages")
validate = load("validate_generations")
carry_over = load("carry_generations")


def chunk(text, fragment=False):
    return SimpleNamespace(text=text, code_fragment=fragment)


PROSE = ("Gradient accumulation sums the gradients of several small batches before one optimizer step is taken, so that the effective batch "
         "size can be larger than what fits in memory at once. The optimizer then updates the parameters a single time.")


class SelectionTests(unittest.TestCase):
    def test_explanatory_prose_is_eligible_and_code_and_boilerplate_are_not(self):
        self.assertTrue(select.eligible(chunk(PROSE)))
        self.assertFalse(select.eligible(chunk(PROSE, fragment=True)), "a code fragment is never selected")
        self.assertFalse(select.eligible(chunk("short")), "too short")
        self.assertFalse(select.eligible(chunk("Copyright (c) the authors. " + PROSE)), "boilerplate")
        code = "\n".join(f"x{i} = torch.zeros({i})" for i in range(20))
        self.assertFalse(select.eligible(chunk(code)))
        trace = 'Traceback (most recent call last):\n  File "/a/b.py", line 20, in <module>\n    run()\n  File "/a/c.py", line 3, in run\n    raise ValueError("bad value given to the function here")\nValueError: bad value given to the function here, expected something else entirely'
        self.assertFalse(select.eligible(chunk(trace)), "a traceback is not explanatory prose")

    def test_the_ranking_is_deterministic_and_seed_dependent(self):
        self.assertEqual(select.rank_key(1, "eg-a"), select.rank_key(1, "eg-a"))
        self.assertNotEqual(select.rank_key(1, "eg-a"), select.rank_key(2, "eg-a"))


class CarryOverTests(unittest.TestCase):
    def test_a_question_follows_its_passage_only_while_chunk_id_and_text_are_unchanged(self):
        old_selection = [{"ref": "te-0001", "chunk_id": "ch_a"}, {"ref": "tr-0001", "chunk_id": "ch_b"}, {"ref": "tr-0002", "chunk_id": "ch_c"}, {"ref": "tr-0003", "chunk_id": "ch_d"}]
        old_passages = {"te-0001": {"text": "alpha"}, "tr-0001": {"text": "beta"}, "tr-0002": {"text": "gamma"}, "tr-0003": {"text": "delta"}}
        old_gen = {"te-0001": {"ref": "te-0001", "question": "Q a?"}, "tr-0001": {"ref": "tr-0001", "question": "Q b?"}, "tr-0002": {"ref": "tr-0002", "skip": "navigation links"}}
        # the rebuilt assignment moved ch_a to train, changed the text of ch_b, dropped ch_c's generation context and added a new chunk
        new_selection = [{"ref": "tr-0001", "chunk_id": "ch_a"}, {"ref": "tr-0002", "chunk_id": "ch_b"}, {"ref": "va-0001", "chunk_id": "ch_c"}, {"ref": "va-0002", "chunk_id": "ch_d"}, {"ref": "va-0003", "chunk_id": "ch_e"}]
        new_passages = {"tr-0001": {"text": "alpha"}, "tr-0002": {"text": "beta, reworded"}, "va-0001": {"text": "gamma"}, "va-0002": {"text": "delta"}, "va-0003": {"text": "epsilon"}}
        carried, todo = carry_over.carry(old_selection, old_passages, old_gen, new_selection, new_passages)
        self.assertEqual(carried, [{"ref": "tr-0001", "question": "Q a?"}, {"ref": "va-0001", "skip": "navigation links"}], "the line moves to the new ref, whichever split owns it now")
        self.assertEqual(todo, ["tr-0002", "va-0002", "va-0003"], "changed text, a passage without any generation line, and a new chunk are generated")
        self.assertEqual(old_gen["te-0001"]["ref"], "te-0001", "the old lines are not modified")


class MechanicalCheckTests(unittest.TestCase):
    def test_a_quote_is_located_through_wrapped_lines_and_must_exist(self):
        text = "The optimizer\nupdates the parameters\na single time."
        span = validate.locate("optimizer updates the parameters a single time", text)
        self.assertEqual(text[span[0]:span[1]], "optimizer\nupdates the parameters\na single time")
        self.assertIsNone(validate.locate("the optimizer never updates anything", text))
        self.assertIsNone(validate.locate("   ", text))

    def test_six_word_copies_are_found_and_paraphrases_are_not(self):
        passage = "Gradient accumulation sums the gradients of several small batches before one optimizer step"
        self.assertTrue(validate.grams("why does gradient accumulation sums the gradients of several work", 6) & validate.grams(passage))
        self.assertFalse(validate.grams("how can many tiny batches be combined before updating weights", 6) & validate.grams(passage))

    def test_hidden_context_phrases_are_recognised(self):
        for bad in ("What does the passage say about accumulation?", "How is this tutorial's loss computed?", "Why does the code above fail?", "In this section what is a tensor?"):
            self.assertTrue(validate.HIDDEN.search(bad), bad)
        for good in ("How do I accumulate gradients over micro batches?", "Why does autograd fail on in-place operations?"):
            self.assertFalse(validate.HIDDEN.search(good), good)

    def test_near_duplicate_detection_uses_word_overlap(self):
        a = {"how", "do", "i", "freeze", "layers", "in", "a", "model"}
        self.assertGreaterEqual(validate.jaccard(a, a | {"please"}), 0.85)
        self.assertLess(validate.jaccard(a, {"what", "is", "autograd"}), 0.2)


@requires_ml
class EndToEndTests(unittest.TestCase):
    """selection -> generation checks -> drafts on the synthetic rehearsal corpus (fabricated agent output, real validators)."""

    @classmethod
    def setUpClass(cls):
        from rehearsal_fixture import rehearsal
        cls.base = rehearsal(topics=36)

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="devsearch-realdata-")).resolve()
        self.addCleanup(shutil.rmtree, self.work, ignore_errors=True)

    def run_script(self, name, *args):
        return subprocess.run([sys.executable, str(SCRIPTS / name), *map(str, args)], capture_output=True, text=True, timeout=300)

    def test_the_flow_produces_valid_drafts_and_rejects_bad_generations(self):
        manifest = self.base / "bundle" / "prepared" / "split-manifest.json"
        result = self.run_script("select_passages.py", "--split-manifest", manifest, "--output-dir", self.work, "--train", "5", "--validation", "3", "--test", "3", "--batch-size", "4")
        self.assertEqual(result.returncode, 0, result.stderr)
        selection = [json.loads(line) for line in (self.work / "selection.jsonl").read_text().splitlines()]
        self.assertTrue(selection)
        self.assertEqual(len({r["evidence_group_id"] for r in selection}), len(selection), "one passage per evidence group")
        passages = {}
        for path in sorted((self.work / "batches").glob("*.jsonl")):
            for line in path.read_text().splitlines():
                passages[json.loads(line)["ref"]] = json.loads(line)
        self.assertTrue(all(set(p) == {"ref", "title", "section", "text"} for p in passages.values()), "agents see no chunk id and no split")
        (self.work / "gen").mkdir()
        (self.work / "rev").mkdir()
        good, lines = None, []
        for index, record in enumerate(selection):
            text = passages[record["ref"]]["text"]
            if index == 0:
                lines.append({"ref": record["ref"], "question": "How is this tutorial laid out?", "evidence_quote": text.split()[0], "task_family": "other"})      # hidden context
            elif index == 1:
                lines.append({"ref": record["ref"], "question": "Why would a developer need such a register routine here?", "evidence_quote": "this exact text is not in the passage", "task_family": "why"})
            else:
                words = text.split()
                lines.append({"ref": record["ref"], "question": f"Which routine releases the held {index} resources afterwards today?", "evidence_quote": " ".join(words[2:8]), "task_family": "api"})
                good = good or record["ref"]
        (self.work / "gen" / "all.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        tokenizer = self.work / "tok"
        tokenizer.mkdir()
        from devsearch.modeling.tokenization import write_diagnostic_tokenizer
        write_diagnostic_tokenizer(tokenizer, ["which", "routine", "releases", "the", "held", "resources", "afterwards", "today"])
        result = self.run_script("validate_generations.py", "--work-dir", self.work, "--tokenizer-assets", tokenizer)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        rejected = [json.loads(line) for line in (self.work / "rejected-gen.jsonl").read_text().splitlines()]
        reasons = " ".join(" ".join(r["reasons"]) for r in rejected)
        self.assertIn("hidden context", reasons)
        self.assertIn("evidence quote is not an exact", reasons)
        self.assertEqual(summary["accepted"], len(selection) - 2)
        for line in (self.work / "review-batches").glob("*.jsonl"):
            for row in map(json.loads, line.read_text().splitlines()):
                self.assertEqual(set(row), {"ref", "question", "passage"}, "reviewers never see the quote, the generator or a label")
        accepted = [json.loads(line) for line in (self.work / "accepted-gen.jsonl").read_text().splitlines()]
        reviews = [{"ref": a["ref"], "answerable": "yes" if i % 5 else "partial", "self_contained": True, "generic": False, "answer": "x"} for i, a in enumerate(accepted)]
        (self.work / "rev" / "all.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reviews))
        drafts = self.work / "drafts"
        drafts.mkdir()
        result = self.run_script("build_drafts.py", "--work-dir", self.work, "--split-manifest", manifest, "--output-dir", drafts)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertIn("no human reviewed", report["labels"])
        from devsearch.data.queries import load_qrels, load_queries
        total = 0
        for split in ("train", "validation", "test"):
            queries = list(load_queries(drafts / f"queries-{split}.jsonl"))
            qrels = list(load_qrels(drafts / f"qrels-{split}.jsonl"))
            self.assertEqual(len(queries), len(qrels))
            total += len(queries)
            for query in queries:
                self.assertEqual((query.origin, query.review_status, query.reviews[0].reviewer_type), ("synthetic", "accepted", "agent"))
                self.assertEqual(query.split, split)
        self.assertEqual(total, len(accepted) - len([r for i, r in enumerate(accepted) if i % 5 == 0]), "partial reviews are dropped")
        again = self.run_script("build_drafts.py", "--work-dir", self.work, "--split-manifest", manifest, "--output-dir", drafts)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("never overwritten", again.stderr)


if __name__ == "__main__":
    unittest.main()
