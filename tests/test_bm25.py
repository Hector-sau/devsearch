"""BM25 behavior: hand-computed scores, filters, zero-match, ties and identifier queries."""

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from devsearch.contracts import SearchRequest
from devsearch.data.corpus import ChunkSpan, DocInput, MANIFEST_NAME, load_corpus, write_corpus
from devsearch.data.sourcemap import SourceSegment
from devsearch.errors import InvalidRequestError, ManifestError
from devsearch.retrieval.bm25 import Bm25Index, Bm25Retriever, build_postings, parse_postings
from devsearch.retrieval.tokenizer import tokenize, unique_query_terms

from helpers import RootTestCase, request

K1, B = 1.2, 0.75


def reference_score(tf, length, avg_length, df, n, *, idf_kind, k1=K1, b=B):
    """Independent restatement of the documented formula (does not call the implementation)."""
    ratio = (n - df + 0.5) / (df + 0.5)
    idf = math.log(1 + ratio) if idf_kind == "lucene" else math.log(ratio)
    return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_length))


class SyntheticCorpusCase(unittest.TestCase):
    def synthetic(self, texts, *, idf="lucene", k1=K1, b=B):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        docs = [
            DocInput(f"d:{i:02d}", "src", f"T{i}", "r1", None, "lic", "p1", text + "\n",
                     (SourceSegment(0, len(text), "f.md", False, start_line=1, end_line=1),),
                     (ChunkSpan(0, len(text), f"eg{i}"),))
            for i, text in enumerate(texts)
        ]
        write_corpus(directory, docs, chunker_version="c1")
        corpus = load_corpus(directory / MANIFEST_NAME)
        index = Bm25Index(build_postings(corpus.chunks), idf=idf, k1=k1, b=b)
        return corpus, index, Bm25Retriever(corpus, index)


class ScoringTests(SyntheticCorpusCase):
    TEXTS = ["alpha beta beta", "alpha gamma gamma gamma gamma", "delta"]

    def test_scores_match_the_hand_computed_formula(self):
        corpus, index, retriever = self.synthetic(self.TEXTS)
        self.assertEqual((index.chunk_count, index.total_tokens, index.vocabulary_size), (3, 9, 4))
        self.assertEqual(index.average_length, 3.0)
        results = retriever.retrieve(request("alpha beta"))
        expected_a = (reference_score(1, 3, 3.0, 2, 3, idf_kind="lucene")
                      + reference_score(2, 3, 3.0, 1, 3, idf_kind="lucene"))
        expected_b = reference_score(1, 5, 3.0, 2, 3, idf_kind="lucene")
        self.assertEqual([r.chunk.text for r in results], ["alpha beta beta", "alpha gamma gamma gamma gamma"])
        self.assertAlmostEqual(results[0].bm25_score, expected_a, places=9)
        self.assertAlmostEqual(results[1].bm25_score, expected_b, places=9)
        self.assertEqual([r.baseline_rank for r in results], [1, 2])
        self.assertGreater(expected_a, expected_b)

    def test_query_term_repeats_do_not_change_scores(self):
        _, _, retriever = self.synthetic(self.TEXTS)
        once = retriever.retrieve(request("alpha beta"))
        repeated = retriever.retrieve(request("alpha ALPHA beta beta alpha"))
        self.assertEqual([(r.chunk.chunk_id, r.bm25_score) for r in once], [(r.chunk.chunk_id, r.bm25_score) for r in repeated])

    def test_robertson_idf_can_be_zero_yet_the_match_is_still_returned(self):
        # N=2, df=1 -> ln((2-1+.5)/(1+.5)) = ln(1) = 0: a lexical match with score exactly 0.
        _, _, retriever = self.synthetic(["alpha beta", "gamma delta"], idf="robertson")
        results = retriever.retrieve(request("alpha"))
        self.assertEqual([(r.chunk.text, r.bm25_score) for r in results], [("alpha beta", 0.0)])
        self.assertEqual(retriever.retrieve(request("epsilon")), ())          # no overlap: nothing, not zero-padding
        _, _, lucene = self.synthetic(["alpha beta", "gamma delta"], idf="lucene")
        self.assertGreater(lucene.retrieve(request("alpha"))[0].bm25_score, 0)

    def test_negative_scores_still_rank_by_descending_score(self):
        texts = ["common one", "common two three four five", "common"]
        _, _, retriever = self.synthetic(texts, idf="robertson")
        results = retriever.retrieve(request("common"))
        self.assertEqual(len(results), 3)                                      # eligibility ignores the sign
        average = sum(len(tokenize(t)) for t in texts) / 3
        expected = sorted(
            ((round(reference_score(1, len(tokenize(t)), average, 3, 3, idf_kind="robertson"), 10), f"d:{i:02d}")
             for i, t in enumerate(texts)),
            key=lambda pair: (-pair[0], pair[1]),
        )
        self.assertTrue(all(r.bm25_score < 0 for r in results))
        self.assertEqual([r.chunk.doc_id for r in results], [doc for _, doc in expected])
        for result, (score, _) in zip(results, expected):
            self.assertAlmostEqual(result.bm25_score, score, places=9)

    def test_b_zero_ignores_length_and_k1_zero_ignores_frequency(self):
        _, _, no_length = self.synthetic(["alpha", "alpha one two three four"], b=0.0)
        scores = {r.chunk.text: r.bm25_score for r in no_length.retrieve(request("alpha"))}
        self.assertEqual(len(set(scores.values())), 1)
        _, _, binary = self.synthetic(["beta beta beta", "beta", "gamma"], k1=0.0)
        self.assertEqual(len({r.bm25_score for r in binary.retrieve(request("beta"))}), 1)

    def test_candidate_k_truncates_after_ordering_and_ranks_are_dense(self):
        texts = [f"alpha {'filler ' * i}" for i in range(6)] + ["omega"]
        _, _, retriever = self.synthetic(texts)
        everything = retriever.retrieve(request("alpha", candidate_k=50))
        top3 = retriever.retrieve(request("alpha", candidate_k=3))
        self.assertEqual(len(everything), 6)
        self.assertEqual([r.chunk.chunk_id for r in top3], [r.chunk.chunk_id for r in everything[:3]])
        self.assertEqual([r.baseline_rank for r in everything], [1, 2, 3, 4, 5, 6])
        self.assertEqual(retriever.retrieve(request("alpha", candidate_k=1))[0].chunk.chunk_id, everything[0].chunk.chunk_id)

    def test_degenerate_corpus_without_tokens_matches_nothing(self):
        _, _, retriever = self.synthetic(["the of and", "a an the"])
        self.assertEqual(retriever.retrieve(request("the")), ())


class FixtureRetrievalTests(RootTestCase):
    def setUp(self):
        super().setUp()
        self.corpus = load_corpus(self.corpus_manifest)
        self.postings = build_postings(self.corpus.chunks)

    def retriever(self, idf="lucene"):
        return Bm25Retriever(self.corpus, Bm25Index(self.postings, idf=idf, k1=K1, b=B))

    def search(self, query, **kwargs):
        return self.retriever().retrieve(request(query, **kwargs))

    def headings(self, results):
        return [r.chunk.heading_path for r in results]

    def test_identifier_queries_find_the_chunk_that_contains_the_identifier(self):
        cases = {
            "load_state_dict": ("Checkpoints", "Loading"),
            "AutoModelForSequenceClassification": ("Models", "Heads"),
            "<unk>": ("Tokenizers", "Unknown"),
            "tokenizer.convert_ids_to_tokens": ("Tokenizers", "Unknown"),
            "requires_grad": ("Modes", "Gradients"),
            "return_tensors": ("Tokenizers", "Loading"),
        }
        for query, heading in cases.items():
            with self.subTest(query=query):
                top = self.search(query)[0]
                self.assertEqual(top.chunk.heading_path, heading)
        self.assertIn("torch.no_grad()", self.search("torch.no_grad()")[0].chunk.text)

    def test_exact_identifier_outranks_the_same_words_inside_another_identifier(self):
        results = self.search("state_dict")
        order = self.headings(results)
        self.assertLess(order.index(("Checkpoints", "Saving")), order.index(("Checkpoints", "Loading")))

    def test_plain_words_still_find_identifier_parts(self):
        found = {r.chunk.heading_path for r in self.search("state dict")}
        self.assertIn(("Checkpoints", "Loading"), found)

    def test_zero_match_queries_return_nothing(self):
        for query in ("kubernetes ingress controller", "the of and a", "?!", "zzzz", "   "):
            with self.subTest(query=query):
                self.assertEqual(self.search(query), ())

    def test_nonpositive_scores_are_eligible_and_ordered_by_the_documented_rule(self):
        containing = [i for i, c in enumerate(self.corpus.chunks) if "model" in tokenize(c.text)]
        self.assertGreater(len(containing), len(self.corpus.chunks) / 2)   # precondition: idf(model) < 0
        results = self.retriever("robertson").retrieve(request("model"))
        self.assertEqual(len(results), len(containing))
        self.assertTrue(all(r.bm25_score < 0 for r in results))
        self.assertEqual(
            [r.chunk.chunk_id for r in results],
            [c for _, c in sorted(((-r.bm25_score, r.chunk.chunk_id) for r in results))],
        )
        self.assertTrue(all(r.bm25_score > 0 for r in self.retriever("lucene").retrieve(request("model"))))

    def test_exact_ties_are_broken_by_ascending_chunk_id(self):
        results = self.search("tiebreakmarker")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].bm25_score, results[1].bm25_score)
        self.assertNotEqual(results[0].chunk.doc_id, results[1].chunk.doc_id)
        self.assertLess(results[0].chunk.chunk_id, results[1].chunk.chunk_id)
        self.assertEqual(self.search("tiebreakmarker"), results)            # repeatable

    def test_filters_mask_before_ranking_and_keep_global_statistics(self):
        everything = {r.chunk.chunk_id: r.bm25_score for r in self.search("tokenizer attention")}
        scoped = self.search("tokenizer attention", sources=("fixture-transformers",))
        self.assertTrue(scoped)
        self.assertLess(len(scoped), len(everything) + 1)
        for result in scoped:
            self.assertEqual(self.corpus.doc_of(result.chunk).source_id, "fixture-transformers")
            # Equal scores prove N, df and average length were NOT recomputed over the subset.
            self.assertEqual(result.bm25_score, everything[result.chunk.chunk_id])

    def test_version_filter_selects_one_exact_snapshot(self):
        versions = (("fixture-transformers", "fixture-v2"),)
        results = self.search("from_pretrained", sources=("fixture-transformers",), versions=versions)
        self.assertEqual({r.chunk.revision for r in results}, {"fixture-v2"})
        self.assertEqual(len(results), 1)
        # Without `sources`, a version entry constrains only its own source: other sources stay visible.
        v1 = self.search("model tokenizer", versions=(("fixture-transformers", "fixture-v1"),))
        by_source = {}
        for r in v1:
            by_source.setdefault(self.corpus.doc_of(r.chunk).source_id, set()).add(r.chunk.revision)
        self.assertEqual(by_source, {"fixture-torch": {"fixture-r1"}, "fixture-transformers": {"fixture-v1"}})

    def test_valid_filter_without_a_lexical_match_is_empty_not_an_error(self):
        self.assertEqual(self.search("attention mask padded", sources=("fixture-torch",)), ())
        self.assertEqual(self.search("kubernetes", sources=("fixture-transformers",)), ())

    def test_invalid_filters_are_errors_and_never_widen_silently(self):
        cases = {
            "unknown source": dict(sources=("nope",)),
            "unknown version": dict(versions=(("fixture-torch", "fixture-r9"),)),
            "version for unknown source": dict(versions=(("nope", "x"),)),
            "duplicate sources": dict(sources=("fixture-torch", "fixture-torch")),
            "repeated version key": dict(versions=(("fixture-torch", "fixture-r1"), ("fixture-torch", "fixture-r1"))),
            "version for excluded source": dict(sources=("fixture-torch",), versions=(("fixture-transformers", "fixture-v1"),)),
        }
        for name, kwargs in cases.items():
            with self.subTest(name), self.assertRaises(InvalidRequestError):
                self.search("checkpoint", **kwargs)

    def test_candidate_k_must_be_positive(self):
        with self.assertRaises(InvalidRequestError):
            self.search("checkpoint", candidate_k=0)

    def test_retriever_rejects_an_index_for_different_chunks(self):
        other = build_postings(self.corpus.chunks[:-1])
        with self.assertRaisesRegex(ManifestError, "different chunks"):
            Bm25Retriever(self.corpus, Bm25Index(other, idf="lucene", k1=K1, b=B))

    def test_request_mode_does_not_affect_baseline_retrieval(self):
        base = SearchRequest(query="checkpoint", mode="baseline")
        self.assertEqual(self.retriever().retrieve(base), self.retriever().retrieve(SearchRequest(query="checkpoint", mode="compare")))


class PostingsParsingTests(SyntheticCorpusCase):
    def setUp(self):
        self.corpus, self.index, _ = self.synthetic(["alpha beta", "alpha gamma", "delta"])
        self.record = self.index.postings.to_record()
        self.ids = [c.chunk_id for c in self.corpus.chunks]

    def parse(self, record):
        return parse_postings(record, expected_chunk_ids=self.ids, where="p")

    def test_round_trip(self):
        self.assertEqual(self.parse(self.record), self.index.postings)

    def test_defects_are_rejected(self):
        def change(fn):
            record = {**self.record, "terms": {k: [list(p) for p in v] for k, v in self.record["terms"].items()}}
            fn(record)
            return record
        cases = {
            "unknown key": change(lambda r: r.update(extra=1)),
            "chunk order": change(lambda r: r.update(chunk_ids=list(reversed(r["chunk_ids"])))),
            "doc length": change(lambda r: r.update(doc_lengths=[9, 2, 1])),
            "short lengths": change(lambda r: r.update(doc_lengths=[2, 2])),
            "unsorted postings": change(lambda r: r["terms"].update(alpha=[[1, 1], [0, 1]])),
            "index out of range": change(lambda r: r["terms"].update(alpha=[[0, 1], [7, 1]])),
            "zero frequency": change(lambda r: r["terms"].update(alpha=[[0, 0], [1, 1]])),
            "empty list": change(lambda r: r["terms"].update(alpha=[])),
            "bad pair": change(lambda r: r["terms"].update(alpha=[[0, 1, 2]])),
            "future schema": change(lambda r: r.update(schema_version=2)),
        }
        for name, record in cases.items():
            with self.subTest(name), self.assertRaises(ManifestError):
                self.parse(record)


class QueryTermTests(unittest.TestCase):
    def test_query_terms_use_the_same_tokenizer_as_documents(self):
        self.assertEqual(unique_query_terms("State_Dict"), tokenize("state_dict"))


if __name__ == "__main__":
    unittest.main()
