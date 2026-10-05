"""Index artifact: deterministic build, idempotency, containment and tamper detection."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch.artifacts.canonical import seal_manifest
from devsearch.artifacts.identity import code_identity
from devsearch.data.corpus import ChunkSpan, DocInput, MANIFEST_NAME, write_corpus
from devsearch.data.sourcemap import SourceSegment
from devsearch.errors import ManifestError
from devsearch.retrieval.index_store import build_index, load_index, verify_index_deep

from helpers import MINI_CORPUS, RootTestCase, index_config, request


class IndexBuildTests(RootTestCase):
    def build(self, config=None, **overrides):
        arguments = dict(corpus_manifest=self.corpus_manifest, output_manifest=self.index_manifest, root=self.root)
        arguments.update(overrides)
        return build_index(config or index_config(), **arguments)

    def test_build_writes_a_verifiable_manifest_and_postings(self):
        manifest, created = self.build()
        self.assertTrue(created)
        self.assertEqual(sorted(p.name for p in (self.root / "index").iterdir()),
                         ["index-manifest.json", "index-manifest.postings.json"])   # no temp files left
        self.assertEqual(manifest["corpus"]["manifest_path"], "../corpus/corpus-manifest.json")
        self.assertEqual(manifest["postings"]["path"], "index-manifest.postings.json")
        self.assertEqual(manifest["stats"]["chunk_count"], 20)
        self.assertEqual(manifest["bm25"], {
            "variant": "okapi", "idf": "lucene", "k1": 1.2, "b": 0.75,
            "query_term_weighting": "binary", "score_decimals": 10, "tie_break": "chunk_id_ascending",
        })
        self.assertEqual(manifest["built_with"], code_identity())
        loaded = load_index(self.index_manifest, self.root)
        self.assertEqual(loaded.index_id, manifest["index_id"])
        verify_index_deep(loaded)
        self.assertTrue(loaded.retriever().retrieve(request("load_state_dict")))

    def test_identical_inputs_build_identical_bytes_anywhere(self):
        manifest, _ = self.build()
        other = self.make_root()
        again, _ = build_index(index_config(), other / "corpus" / MANIFEST_NAME, other / "index" / "index-manifest.json", other)
        self.assertEqual(manifest["index_id"], again["index_id"])
        self.assertEqual((self.root / "index" / "index-manifest.postings.json").read_bytes(),
                         (other / "index" / "index-manifest.postings.json").read_bytes())
        self.assertEqual(self.index_manifest.read_bytes(), (other / "index" / "index-manifest.json").read_bytes())

    def test_index_id_covers_parameters_and_corpus(self):
        base = self.build()[0]["index_id"]
        ids = {base}
        for index, config in enumerate([index_config(k1=1.5), index_config(b=0.5), index_config(idf="robertson")]):
            output = self.root / "index" / f"variant-{index}.json"
            ids.add(build_index(config, self.corpus_manifest, output, self.root)[0]["index_id"])
        self.assertEqual(len(ids), 4)
        smaller = self.root / "corpus2"
        smaller.mkdir()
        doc = DocInput("d:one", "s", "T", "r1", None, "l", "p", "alpha beta\n",
                       (SourceSegment(0, 10, "a.md", False, start_line=1, end_line=1),), (ChunkSpan(0, 10, "eg"),))
        write_corpus(smaller, [doc], chunker_version="c")
        different = build_index(index_config(), smaller / MANIFEST_NAME, self.root / "index" / "other.json", self.root)[0]
        self.assertNotIn(different["index_id"], ids)

    def test_rebuild_is_idempotent_and_leaves_files_untouched(self):
        self.build()
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.root / "index").iterdir()}
        manifest, created = self.build()
        self.assertFalse(created)
        self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.root / "index").iterdir()})
        self.assertEqual(manifest["index_id"], load_index(self.index_manifest, self.root).index_id)

    def test_an_unresolved_root_is_accepted_through_a_symlink(self):
        link = self.root.parent / (self.root.name + "-link")
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError:
            self.skipTest("symlinks unavailable")
        self.addCleanup(link.unlink)
        manifest, created = self.build(root=link)
        self.assertTrue(created)
        self.assertEqual(load_index(self.index_manifest, link).index_id, manifest["index_id"])

    def test_a_different_index_never_overwrites_an_existing_output(self):
        self.build()
        before = self.index_manifest.read_bytes()
        with self.assertRaisesRegex(ManifestError, "different index_id"):
            self.build(index_config(k1=2.0))
        self.assertEqual(self.index_manifest.read_bytes(), before)

    def test_existing_but_damaged_identical_index_is_reported_not_trusted(self):
        self.build()
        postings = self.root / "index" / "index-manifest.postings.json"
        postings.write_text(postings.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "postings file does not match"):
            self.build()

    def test_stray_postings_file_with_other_content_blocks_the_build(self):
        stray = self.root / "index" / "index-manifest.postings.json"
        stray.write_text("not the postings")
        with self.assertRaisesRegex(ManifestError, "different content"):
            self.build()
        self.assertEqual(stray.read_text(), "not the postings")
        self.assertFalse(self.index_manifest.exists())

    def test_missing_output_directory_fails_without_creating_it(self):
        missing = self.root / "unmounted-volume"
        with self.assertRaisesRegex(ManifestError, "does not exist"):
            self.build(output_manifest=missing / "index-manifest.json")
        self.assertFalse(missing.exists())

    def test_paths_outside_the_runtime_root_are_rejected(self):
        outside = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        shutil.copytree(MINI_CORPUS, outside / "corpus")
        with self.assertRaisesRegex(ManifestError, "escapes"):
            self.build(corpus_manifest=outside / "corpus" / MANIFEST_NAME)
        with self.assertRaisesRegex(ManifestError, "escapes"):
            self.build(output_manifest=outside / "index-manifest.json")
        self.assertEqual(list(outside.glob("*.json")), [])

    def test_an_empty_corpus_is_refused(self):
        empty = self.root / "empty"
        empty.mkdir()
        doc = DocInput("d:one", "s", "T", "r1", None, "l", "p", "alpha\n", (SourceSegment(0, 5, "a.md", False, start_line=1, end_line=1),), ())
        write_corpus(empty, [doc], chunker_version="c")
        with self.assertRaisesRegex(ManifestError, "no chunks"):
            self.build(corpus_manifest=empty / MANIFEST_NAME)


class IndexLoadDefectTests(RootTestCase):
    def setUp(self):
        super().setUp()
        build_index(index_config(), self.corpus_manifest, self.index_manifest, self.root)

    def edit(self, change, *, reseal=True):
        manifest = json.loads(self.index_manifest.read_text(encoding="utf-8"))
        change(manifest)
        if reseal:
            manifest = seal_manifest(manifest)
        self.index_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    def assert_load_fails(self, expected):
        with self.assertRaisesRegex(ManifestError, expected):
            load_index(self.index_manifest, self.root)

    def test_clean_index_loads(self):
        load_index(self.index_manifest, self.root)

    def test_manifest_edits(self):
        cases = {
            "unsealed stats edit": (lambda m: m["stats"].update(chunk_count=1), False, "manifest_sha256"),
            "resealed stats edit": (lambda m: m["stats"].update(chunk_count=1), True, "stats do not match"),
            "resealed index id": (lambda m: m.update(index_id="index-" + "0" * 32), True, "index_id does not match"),
            "other tokenizer rules": (lambda m: m["tokenizer"].update(fingerprint="0" * 64), True, "different tokenizer rules"),
            "unknown tokenizer": (lambda m: m["tokenizer"].update(name="porter-v9"), True, "unsupported tokenizer"),
            "unknown variant": (lambda m: m["bm25"].update(variant="bm25f"), True, "unsupported or malformed"),
            "tie break changed": (lambda m: m["bm25"].update(tie_break="score_only"), True, "unsupported or malformed"),
            "score decimals changed": (lambda m: m["bm25"].update(score_decimals=3), True, "unsupported or malformed"),
            "string k1": (lambda m: m["bm25"].update(k1="1.2"), True, "unsupported or malformed"),
            "unknown manifest key": (lambda m: m.update(extra=1), True, "unrecognized field"),
            "wrong kind": (lambda m: m.update(kind="corpus"), True, "expected kind"),
            "future schema": (lambda m: m.update(schema_version=2), True, "expected kind"),
            "absolute corpus path": (lambda m: m["corpus"].update(manifest_path=str(self.corpus_manifest)), True, "relative"),
            "corpus path escapes": (lambda m: m["corpus"].update(manifest_path="../../etc/passwd"), True, "escapes"),
            "wrong corpus id": (lambda m: m["corpus"].update(corpus_id="corpus-" + "0" * 32), True, "corpus_id does not match"),
            "wrong postings hash": (lambda m: m["postings"].update(sha256="0" * 64), True, "postings file does not match"),
            "postings path missing": (lambda m: m["postings"].update(path="nope.json"), True, "missing"),
        }
        for name, (change, reseal, expected) in cases.items():
            with self.subTest(name):
                shutil.rmtree(self.root / "index")
                (self.root / "index").mkdir()
                build_index(index_config(), self.corpus_manifest, self.index_manifest, self.root)
                self.edit(change, reseal=reseal)
                self.assert_load_fails(expected)

    def test_postings_tampering_is_detected(self):
        postings = self.root / "index" / "index-manifest.postings.json"
        postings.write_text(postings.read_text(encoding="utf-8").replace('"alpha"', '"alphb"', 1) + "\n", encoding="utf-8")
        self.assert_load_fails("postings file does not match")

    def test_corpus_changes_after_the_build_are_detected(self):
        self.corpus_manifest.write_text(self.corpus_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assert_load_fails("corpus manifest changed")

    def test_corpus_content_changes_are_detected(self):
        text = self.root / "corpus" / "text" / "0000.txt"
        text.write_text(text.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "SHA-256"):
            load_index(self.index_manifest, self.root)

    def test_manifest_outside_the_root_is_rejected(self):
        other_root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, other_root, ignore_errors=True)
        with self.assertRaisesRegex(ManifestError, "escapes"):
            load_index(self.index_manifest, other_root)

    def test_deep_verification_catches_postings_built_by_other_tokenizer_code(self):
        # The shallow fingerprint can pass if code changed without a version bump;
        # `verify_index_deep` rebuilds from the corpus text and must still object.
        shutil.rmtree(self.root / "index")
        (self.root / "index").mkdir()
        with mock.patch("devsearch.retrieval.bm25.tokenize", lambda text: text.lower().split()):
            build_index(index_config(), self.corpus_manifest, self.index_manifest, self.root)
        loaded = load_index(self.index_manifest, self.root)
        with self.assertRaisesRegex(ManifestError, "differ from a fresh build"):
            verify_index_deep(loaded)


if __name__ == "__main__":
    unittest.main()
