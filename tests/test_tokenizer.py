"""Tokenizer rules, including identifier handling and the pinned rules fingerprint."""

import unittest

from devsearch.retrieval import tokenizer
from devsearch.retrieval.tokenizer import fingerprint, tokenize, unique_query_terms


class TokenizerTests(unittest.TestCase):
    def test_golden_tokens(self):
        cases = {
            "load_state_dict": ["load_state_dict", "load", "state", "dict"],
            "torch.nn.Module.eval()": ["torch.nn.module.eval", "torch", "nn", "module", "eval"],
            "AutoTokenizer.from_pretrained('x')": [
                "autotokenizer.from_pretrained", "autotokenizer", "from_pretrained",
                "auto", "tokenizer", "pretrained",
            ],
            "HTTPServer float32 __init__": [
                "httpserver", "http", "server", "float32", "float", "32", "__init__", "init",
            ],
            "state dict": ["state", "dict"],
            "<unk> [CLS] no_grad": ["unk", "cls", "no_grad", "grad"],
            "ＮＦＫＣ ﬁle": ["nfkc", "file"],   # full-width and ligature -> NFKC
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(tokenize(text), expected)

    def test_identifier_query_matches_more_tokens_than_plain_words(self):
        identifier, plain = set(tokenize("load_state_dict")), set(tokenize("load state dict"))
        self.assertLess(plain, identifier)
        self.assertIn("load_state_dict", identifier - plain)

    def test_dropped_tokens(self):
        for text in ("the of and a", "x", "__", "?!", "", "   "):
            with self.subTest(text=text):
                self.assertEqual(tokenize(text), [])

    def test_trailing_punctuation_does_not_join_runs(self):
        self.assertEqual(tokenize("see torch.save. Then"), ["see", "torch.save", "torch", "save"])

    def test_repeats_are_kept_for_documents_but_not_for_queries(self):
        self.assertEqual(tokenize("beta beta"), ["beta", "beta"])
        self.assertEqual(unique_query_terms("beta alpha beta"), ["beta", "alpha"])

    def test_rules_fingerprint_is_pinned(self):
        # Changing any tokenizer rule changes this value. Update it deliberately,
        # bump tokenizer.NAME or ALGORITHM_REVISION, and rebuild every index.
        self.assertEqual(
            fingerprint(), "d63858e8997743fa45d9e9a5f29bbcd97885b042f100ae4e913dfbf830ee701e"
        )
        self.assertEqual(tokenizer.NAME, "lexical-v1")

    def test_fingerprint_reflects_behavior_not_only_declared_rules(self):
        original = tokenizer._split_segment
        try:
            tokenizer._split_segment = lambda segment: [segment]   # behavior change, rules text untouched
            self.assertNotEqual(fingerprint(), "d63858e8997743fa45d9e9a5f29bbcd97885b042f100ae4e913dfbf830ee701e")
        finally:
            tokenizer._split_segment = original


if __name__ == "__main__":
    unittest.main()
