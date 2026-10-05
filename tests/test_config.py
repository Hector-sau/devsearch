"""Strict configuration loading: the shipped examples are valid; every defect is rejected."""

import shutil
import tempfile
import unittest
from pathlib import Path

from devsearch.errors import ConfigError
from devsearch.retrieval.config import load_index_config, load_search_config

from helpers import CONFIGS

INDEX_OK = """schema_version = 1
tokenizer = "lexical-v1"
[bm25]
idf = "lucene"
k1 = 1.2
b = 0.75
"""
SEARCH_OK = """schema_version = 1
candidate_k = 50
top_k = 5
max_candidate_k = 100
max_query_chars = 1000
"""


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def write(self, text, name="config.toml") -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_shipped_example_configs_load(self):
        index = load_index_config(CONFIGS / "index.toml")
        self.assertEqual((index.tokenizer, index.idf, index.k1, index.b), ("lexical-v1", "lucene", 1.2, 0.75))
        search = load_search_config(CONFIGS / "search.toml")
        self.assertEqual((search.candidate_k, search.top_k, search.max_candidate_k), (50, 5, 100))
        self.assertEqual((search.sources, search.versions), ((), ()))

    def test_search_filters_and_request_conversion(self):
        text = SEARCH_OK + '[filters]\nsources = ["a", "b"]\n[filters.versions]\nb = "r2"\na = "r1"\n'
        config = load_search_config(self.write(text))
        self.assertEqual(config.sources, ("a", "b"))
        self.assertEqual(config.versions, (("a", "r1"), ("b", "r2")))   # sorted: deterministic
        request = config.to_request("q")
        self.assertEqual((request.query, request.mode, request.candidate_k, request.top_k), ("q", "baseline", 50, 5))
        self.assertEqual(request.versions, config.versions)

    def assert_index_rejected(self, text, fragment):
        with self.assertRaisesRegex(ConfigError, fragment):
            load_index_config(self.write(text))

    def assert_search_rejected(self, text, fragment):
        with self.assertRaisesRegex(ConfigError, fragment):
            load_search_config(self.write(text))

    def test_index_config_defects(self):
        self.assert_index_rejected(INDEX_OK + "extra = 1\n", "unknown key.*extra")
        self.assert_index_rejected(INDEX_OK.replace('tokenizer = "lexical-v1"\n', ""), "missing required key.*tokenizer")
        self.assert_index_rejected(INDEX_OK.replace("k1 = 1.2\n", ""), "missing required key.*k1")
        self.assert_index_rejected(INDEX_OK.replace('"lexical-v1"', '"other"'), "must be one of")
        self.assert_index_rejected(INDEX_OK.replace('"lucene"', '"classic"'), "bm25.idf")
        self.assert_index_rejected(INDEX_OK.replace("k1 = 1.2", 'k1 = "1.2"'), "expected a number")
        self.assert_index_rejected(INDEX_OK.replace("k1 = 1.2", "k1 = true"), "expected a number")
        self.assert_index_rejected(INDEX_OK.replace("k1 = 1.2", "k1 = nan"), "finite")
        self.assert_index_rejected(INDEX_OK.replace("k1 = 1.2", "k1 = inf"), "finite")
        self.assert_index_rejected(INDEX_OK.replace("k1 = 1.2", "k1 = -0.1"), "between")
        self.assert_index_rejected(INDEX_OK.replace("b = 0.75", "b = 1.5"), "between")
        self.assert_index_rejected(INDEX_OK.replace("schema_version = 1", "schema_version = 2"), "schema_version")
        self.assert_index_rejected(INDEX_OK + "[bm25]\n", "invalid TOML")   # duplicate table
        self.assert_index_rejected('schema_version = 1\ntokenizer = "lexical-v1"\nbm25 = 1\n', "expected a table")
        self.assert_index_rejected(INDEX_OK.replace("[bm25]\n", "[bm25]\nextra = 1\n"), "unknown key.*extra")
        self.assert_index_rejected("this is = not toml [", "invalid TOML")

    def test_search_config_defects(self):
        self.assert_search_rejected(SEARCH_OK + "extra = 1\n", "unknown key.*extra")
        self.assert_search_rejected(SEARCH_OK.replace("top_k = 5\n", ""), "missing required key.*top_k")
        self.assert_search_rejected(SEARCH_OK.replace("top_k = 5", "top_k = 60"), "top_k: out of range")
        self.assert_search_rejected(SEARCH_OK.replace("candidate_k = 50", "candidate_k = 101"), "candidate_k: out of range")
        self.assert_search_rejected(SEARCH_OK.replace("top_k = 5", "top_k = 0"), "out of range")
        self.assert_search_rejected(SEARCH_OK.replace("top_k = 5", "top_k = true"), "expected an integer")
        self.assert_search_rejected(SEARCH_OK.replace("top_k = 5", "top_k = 5.0"), "expected an integer")
        self.assert_search_rejected(SEARCH_OK.replace("max_query_chars = 1000", "max_query_chars = 0"), "out of range")
        self.assert_search_rejected(SEARCH_OK + "[filters]\nbogus = 1\n", "unknown key.*bogus")
        self.assert_search_rejected(SEARCH_OK + '[filters]\nsources = "a"\n', "list of non-empty strings")
        self.assert_search_rejected(SEARCH_OK + '[filters]\nsources = ["a", "a"]\n', "duplicates")
        self.assert_search_rejected(SEARCH_OK + '[filters]\nsources = [""]\n', "non-empty")
        self.assert_search_rejected(SEARCH_OK + "[filters.versions]\na = 1\n", "non-empty strings")
        self.assert_search_rejected(SEARCH_OK + "candidate_k = 7\n", "invalid TOML")   # duplicate key

    def test_unreadable_and_non_utf8_files(self):
        with self.assertRaisesRegex(ConfigError, "cannot read config"):
            load_index_config(self.directory / "missing.toml")
        bad = self.directory / "bad.toml"
        bad.write_bytes(b"\xff\xfe\x00")
        with self.assertRaisesRegex(ConfigError, "invalid TOML"):
            load_search_config(bad)


if __name__ == "__main__":
    unittest.main()
