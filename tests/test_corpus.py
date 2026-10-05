"""Corpus loading: identities, source mapping, fixture reproducibility and strict rejection."""

import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch.artifacts.canonical import canonical_json, seal_manifest, sha256_text
from devsearch.data.corpus import ChunkSpan, DocInput, MANIFEST_NAME, derive_chunk_id, load_corpus, write_corpus
from devsearch.data.sourcemap import SourceSegment
from devsearch.errors import ManifestError

from helpers import MINI_CORPUS, RootTestCase, fixture_parts, write_raw_corpus
import build_mini_corpus  # noqa: E402  (helpers puts tests/fixtures on sys.path)

FIRST_CHUNK_ID = "ch_196fb375552d3925b5e6043611f08bcd"
FIXTURE_CORPUS_ID = "corpus-2d10d5b932df6c713cc4ad2b9ac732e1"


class FixtureCorpusTests(RootTestCase):
    def test_fixture_loads_with_expected_shape_and_identity(self):
        corpus = load_corpus(self.corpus_manifest)
        self.assertEqual((len(corpus.docs), len(corpus.chunks)), (6, 20))
        self.assertEqual(corpus.corpus_id, FIXTURE_CORPUS_ID)
        self.assertEqual(corpus.chunks[0].chunk_id, FIRST_CHUNK_ID)
        self.assertEqual(corpus.revisions_by_source(), {
            "fixture-torch": ("fixture-r1",), "fixture-transformers": ("fixture-v1", "fixture-v2"),
        })

    def test_document_identity_is_doc_id_plus_revision(self):
        corpus = load_corpus(self.corpus_manifest)
        doc_id = "fixture-transformers:docs/tokenizers.md"
        self.assertIn((doc_id, "fixture-v1"), corpus.docs)
        self.assertIn((doc_id, "fixture-v2"), corpus.docs)
        padding = [c for c in corpus.chunks if c.heading_path == ("Tokenizers", "Padding")]
        self.assertEqual(len(padding), 2)
        self.assertEqual(padding[0].text, padding[1].text)
        self.assertNotEqual(padding[0].chunk_id, padding[1].chunk_id)   # same text, different revision

    def test_chunk_ids_are_deterministic_and_depend_on_every_input(self):
        base = ("doc:a", "r1", 0, 5, "hello", "c1")
        reference = derive_chunk_id(*base)
        self.assertEqual(reference, derive_chunk_id(*base))
        self.assertRegex(reference, r"^ch_[0-9a-f]{32}$")
        variants = [
            ("doc:b", "r1", 0, 5, "hello", "c1"), ("doc:a", "r2", 0, 5, "hello", "c1"),
            ("doc:a", "r1", 1, 5, "hello", "c1"), ("doc:a", "r1", 0, 6, "hello", "c1"),
            ("doc:a", "r1", 0, 5, "hellp", "c1"), ("doc:a", "r1", 0, 5, "hello", "c2"),
        ]
        ids = {derive_chunk_id(*variant) for variant in variants}
        self.assertEqual(len(ids), len(variants))
        self.assertNotIn(reference, ids)

    def test_every_chunk_matches_its_normalized_text_slice(self):
        corpus = load_corpus(self.corpus_manifest)
        for chunk in corpus.chunks:
            doc = corpus.doc_of(chunk)
            self.assertEqual(chunk.text, doc.text[chunk.start_char:chunk.end_char])
            self.assertEqual(sha256_text(doc.text), doc.content_sha256)

    def test_source_locations_skip_generated_text(self):
        corpus = load_corpus(self.corpus_manifest)
        saving = corpus.chunks[0]
        self.assertEqual(saving.heading_path, ("Checkpoints", "Saving"))
        # The document starts with a synthetic generated title on line 1 that is never reported.
        self.assertEqual(corpus.locate(saving), [{"path": "docs/checkpoints.md", "start_line": 3, "end_line": 6}])
        self.assertTrue(corpus.source_maps[(saving.doc_id, saving.revision)].segments[0].synthetic)

    def test_code_fragments_keep_indentation_and_are_flagged(self):
        corpus = load_corpus(self.corpus_manifest)
        fragments = [c for c in corpus.chunks if c.code_fragment]
        self.assertEqual(len(fragments), 3)
        self.assertTrue(fragments[1].text.startswith("    scheduler.step()\n    model.eval()"))
        self.assertIn("\n        for batch in val_loader:\n            logits", fragments[1].text)
        self.assertEqual([c.start_char for c in fragments], sorted(c.start_char for c in fragments))
        self.assertFalse(any(c.code_fragment for c in corpus.chunks if c.heading_path != ("Training loop", "Code")))

    def test_checked_in_fixture_matches_its_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            build_mini_corpus.build(Path(directory))
            built = sorted(p.relative_to(directory).as_posix() for p in Path(directory).rglob("*") if p.is_file())
            shipped = sorted(p.relative_to(MINI_CORPUS).as_posix() for p in MINI_CORPUS.rglob("*") if p.is_file())
            self.assertEqual(built, shipped)
            for name in built:
                self.assertEqual((Path(directory) / name).read_bytes(), (MINI_CORPUS / name).read_bytes(), name)

    def test_writer_is_immutable_and_never_creates_the_target(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ManifestError, "does not exist"):
                write_corpus(Path(directory) / "unmounted", build_mini_corpus.documents(), chunker_version="c")
            self.assertFalse((Path(directory) / "unmounted").exists())
            build_mini_corpus.build(Path(directory))
            build_mini_corpus.build(Path(directory))                   # identical content: idempotent, not an error
            changed = [dataclasses.replace(build_mini_corpus.documents()[0], title="Another title")]
            with self.assertRaisesRegex(ManifestError, "immutable"):
                write_corpus(Path(directory), changed, chunker_version=build_mini_corpus.CHUNKER_VERSION)
            with self.assertRaisesRegex(ManifestError, "duplicate"):
                write_corpus(self.root / "index", build_mini_corpus.documents() * 2, chunker_version="c")


class NotebookAndLineMappingTests(unittest.TestCase):
    def build(self, segments, chunks, text="AAAABBBBCC\n"):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        doc = DocInput("d:one", "src", "T", "r1", None, "lic", "p1", text, tuple(segments), tuple(chunks))
        write_corpus(directory, [doc], chunker_version="c1")
        return load_corpus(directory / MANIFEST_NAME)

    def test_cell_mapping_is_exact_for_partial_overlap(self):
        cell = SourceSegment(0, 10, "nb.ipynb", False, cell_id="c1", cell_start_char=100, cell_end_char=110)
        corpus = self.build([cell], [ChunkSpan(2, 6, "eg1")])
        self.assertEqual(corpus.locate(corpus.chunks[0]), [
            {"path": "nb.ipynb", "cell_id": "c1", "start_char": 102, "end_char": 106},
        ])

    def test_line_ranges_merge_across_entries_and_other_paths_stay_separate(self):
        segments = [
            SourceSegment(0, 4, "a.md", False, start_line=3, end_line=3),
            SourceSegment(4, 8, "a.md", False, start_line=4, end_line=7),
            SourceSegment(8, 10, "b.md", False, start_line=1, end_line=2),
        ]
        corpus = self.build(segments, [ChunkSpan(0, 10, "eg1")])
        self.assertEqual(corpus.locate(corpus.chunks[0]), [
            {"path": "a.md", "start_line": 3, "end_line": 7},
            {"path": "b.md", "start_line": 1, "end_line": 2},
        ])

    def test_chunk_over_only_synthetic_text_is_rejected(self):
        segments = [SourceSegment(0, 10, "a.md", True, start_line=1, end_line=1)]
        with self.assertRaisesRegex(ManifestError, "no non-synthetic source location"):
            self.build(segments, [ChunkSpan(0, 10, "eg1")])


class CorpusRejectionTests(RootTestCase):
    def load_corrupted(self, mutate, **raw):
        documents, chunks, files = fixture_parts()
        mutate(documents, chunks, files)
        target = self.root / "bad"
        target.mkdir(exist_ok=True)
        return load_corpus(write_raw_corpus(target, documents, chunks, files, **raw))

    def assert_rejected(self, mutate, expected, **raw):
        with self.assertRaisesRegex(ManifestError, expected):
            self.load_corrupted(mutate, **raw)

    def edit_map(self, files, name, change):
        data = json.loads(files[name])
        change(data)
        files[name] = json.dumps(data).encode("utf-8")

    def test_raw_writer_produces_a_valid_corpus_when_nothing_is_corrupted(self):
        # Guards the rejection tests below against passing for the wrong reason.
        self.assertEqual(len(self.load_corrupted(lambda d, c, f: None).chunks), 20)

    def test_record_field_defects(self):
        cases = [
            (lambda d, c, f: c[0].update(extra=1), "unrecognized field.*extra"),
            (lambda d, c, f: c[0].pop("code_fragment"), "missing required field.*code_fragment"),
            (lambda d, c, f: d[0].update(extra=1), "unrecognized field.*extra"),
            (lambda d, c, f: d[0].pop("license"), "missing required field.*license"),
            (lambda d, c, f: c[0].update(owner_split="holdout"), "owner_split"),
            (lambda d, c, f: c[0].update(code_fragment="no"), "true or false"),
            (lambda d, c, f: c[0].update(heading_path="Checkpoints"), "expected a list"),
            (lambda d, c, f: c[0].update(start_char=True), "expected an integer"),
            (lambda d, c, f: c[0].update(end_char=c[0]["start_char"]), "out of range"),
            (lambda d, c, f: d[0].update(parse_status="quarantined"), "only 'accepted'"),
            (lambda d, c, f: d[0].update(content_sha256="ABC"), "unexpected format"),
            (lambda d, c, f: d[0].update(title="two\nlines"), "control"),
        ]
        for index, (mutate, expected) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_rejected(mutate, expected)

    def test_source_urls_must_be_safe_https(self):
        for url in ("javascript:alert(1)", "http://docs.fixture.invalid/x", "data:text/html,x", "https://u:p@h.invalid/"):
            with self.subTest(url=url):
                self.assert_rejected(lambda d, c, f: d[0].update(source_url=url), "https URL")
        # A null URL is allowed (the spec never invents links); it surfaces as null in results.
        self.assertIsNone(self.load_corrupted(lambda d, c, f: d[0].update(source_url=None)).docs[
            ("fixture-torch:docs/checkpoints.md", "fixture-r1")].source_url)

    def test_chunk_consistency_defects(self):
        cases = [
            (lambda d, c, f: c[0].update(text=c[0]["text"] + "x"), "text differs"),
            (lambda d, c, f: c[0].update(chunk_id="ch_" + "0" * 32), "derived ID"),
            (lambda d, c, f: c.insert(1, dict(c[0])), "duplicate chunk_id"),
            (lambda d, c, f: c[0].update(doc_id="fixture-torch:docs/missing.md"), "unknown document"),
            (lambda d, c, f: c[0].update(end_char=10**6), "beyond"),
            (lambda d, c, f: c.__setitem__(slice(0, 2), [c[1], c[0]]), "sorted"),
        ]
        for index, (mutate, expected) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_rejected(mutate, expected)

    def test_document_and_file_defects(self):
        cases = [
            (lambda d, c, f: d.__setitem__(slice(0, 2), [d[1], d[0]]), "sorted by"),
            (lambda d, c, f: d.append(dict(d[0])), "duplicate document"),
            (lambda d, c, f: d[0].update(content_sha256="0" * 64), "content_sha256"),
            (lambda d, c, f: d[0].update(normalized_text_path="../outside.txt"), "escapes"),
            (lambda d, c, f: d[0].update(normalized_text_path="/etc/hostname"), "relative"),
            (lambda d, c, f: d[0].update(normalized_text_path="text\\0000.txt"), "forbidden"),
            (lambda d, c, f: d[0].update(normalized_text_path="text/missing.txt"), "missing"),
            (lambda d, c, f: d[1].update(normalized_text_path=d[0]["normalized_text_path"]), "already used"),
            (lambda d, c, f: f.update({"text/orphan.txt": b"unreferenced"}), "unreferenced"),
            (lambda d, c, f: f.update({"text/0000.txt": b"\xff\xfe"}), "UTF-8"),
        ]
        for index, (mutate, expected) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_rejected(mutate, expected)

    def test_source_map_defects(self):
        def overlap(data):
            data["entries"][2]["start_char"] = data["entries"][1]["end_char"] - 1
        def beyond(data):
            data["entries"][-1]["end_char"] = 10**6
        def other_doc(data):
            data["doc_id"] = "someone:else"
        def synthetic_only(data):
            data["entries"][1]["synthetic"] = True      # the 'Saving' chunk loses its trace
        def unknown_key(data):
            data["entries"][0]["extra"] = 1
        def bad_cell(data):
            data["entries"][1].update(cell_id="c1", cell_start_char=0, cell_end_char=5)
        for change, expected in [
            (overlap, "must not overlap"), (beyond, "beyond"), (other_doc, "different"),
            (synthetic_only, "no non-synthetic source location"), (unknown_key, "unrecognized field"),
            (bad_cell, "unrecognized field"),
        ]:
            with self.subTest(change=change.__name__):
                self.assert_rejected(lambda d, c, f: self.edit_map(f, "maps/0000.json", change), expected)

    def test_jsonl_file_format_defects(self):
        documents, chunks, _ = fixture_parts()
        good = [canonical_json(record) for record in chunks]
        bodies = {
            "duplicate key": ('{"chunk_id": "a", "chunk_id": "b"}\n', "duplicate key"),
            "NaN": ('{"start_char": NaN}\n', "non-finite"),
            "blank line": (good[0] + "\n\n" + good[1] + "\n", "blank"),
            "CRLF": (good[0] + "\r\n", "carriage return"),
            "array line": ("[1, 2]\n", "expected a JSON object"),
            "truncated": (good[0][:40] + "\n", "invalid JSON"),
        }
        for name, (body, expected) in bodies.items():
            with self.subTest(name):
                self.assert_rejected(lambda d, c, f: None, expected, chunks_bytes=body.encode("utf-8"))
        with self.subTest("invalid utf-8"):
            self.assert_rejected(lambda d, c, f: None, "UTF-8", chunks_bytes=b"\xff\xfe\n")

    def edit_manifest(self, change, *, reseal=True):
        path = self.corpus_manifest
        manifest = json.loads(path.read_text(encoding="utf-8"))
        change(manifest)
        path.write_text(json.dumps(seal_manifest(manifest) if reseal else manifest), encoding="utf-8")

    def test_manifest_level_defects(self):
        cases = {
            "tampered without resealing": (lambda m: m["chunks"].update(count=19), False, "manifest_sha256"),
            "count mismatch": (lambda m: m["chunks"].update(count=19), True, "count does not match"),
            "document count mismatch": (lambda m: m["documents"].update(count=1), True, "count does not match"),
            "wrong corpus id": (lambda m: m.update(corpus_id="corpus-" + "0" * 32), True, "corpus_id does not match"),
            "wrong kind": (lambda m: m.update(kind="index"), True, "expected kind"),
            "future schema": (lambda m: m.update(schema_version=2), True, "expected kind"),
            "unknown key": (lambda m: m.update(extra=1), True, "unrecognized field"),
            "unsorted inventory": (lambda m: m["files"].reverse(), True, "sorted by path"),
            "duplicate inventory path": (lambda m: m["files"].append(dict(m["files"][0])), True, "sorted by path|duplicate"),
            "inventory hash mismatch": (lambda m: m["files"][0].update(sha256="0" * 64), True, "SHA-256"),
            "documents hash mismatch": (lambda m: m["documents"].update(sha256="0" * 64), True, "SHA-256"),
        }
        for name, (change, reseal, expected) in cases.items():
            with self.subTest(name):
                shutil.rmtree(self.root / "corpus")
                shutil.copytree(MINI_CORPUS, self.root / "corpus")
                self.edit_manifest(change, reseal=reseal)
                with self.assertRaisesRegex(ManifestError, expected):
                    load_corpus(self.corpus_manifest)

    def test_files_changed_or_removed_after_the_manifest_was_written(self):
        text = self.root / "corpus" / "text" / "0000.txt"
        text.write_text(text.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "SHA-256"):
            load_corpus(self.corpus_manifest)
        text.unlink()
        with self.assertRaisesRegex(ManifestError, "missing"):
            load_corpus(self.corpus_manifest)

    def test_symlink_that_leaves_the_corpus_directory_is_rejected(self):
        outside = self.root / "outside.txt"
        outside.write_text("secret")
        target = self.root / "corpus" / "text" / "0000.txt"
        target.unlink()
        try:
            target.symlink_to(outside)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ManifestError, "escapes"):
            load_corpus(self.corpus_manifest)


if __name__ == "__main__":
    unittest.main()


def _two_chunk_docs(owner_a, owner_b, group_a="eg-shared", group_b="eg-shared"):
    def doc(i, owner, group):
        return DocInput(f"d:{i}", "s", "T", "r1", None, "l", "p", "alpha beta\n",
                        (SourceSegment(0, 10, "a.md", False, start_line=1, end_line=1),),
                        (ChunkSpan(0, 10, group, (), owner),))
    return [doc(0, owner_a, group_a), doc(1, owner_b, group_b)]


class WriterAtomicityRegressionTests(unittest.TestCase):
    """F01: invalid or interrupted writes must never publish a manifest, and must be retryable."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.docs = build_mini_corpus.documents()

    def write(self, docs=None):
        return write_corpus(self.directory, docs or self.docs, chunker_version=build_mini_corpus.CHUNKER_VERSION)

    def visible(self):
        return sorted(p.name for p in self.directory.iterdir())

    def test_invalid_input_leaves_nothing_published_and_a_valid_retry_succeeds(self):
        bad = [dataclasses.replace(self.docs[0], source_url="http://insecure.invalid/x")] + self.docs[1:]
        with self.assertRaisesRegex(ManifestError, "https URL"):
            self.write(bad)
        self.assertEqual(self.visible(), [], "no manifest, no data files and no staging directory may remain")
        manifest = self.write()                                                   # the retry that F01 used to refuse
        self.assertEqual(load_corpus(self.directory / MANIFEST_NAME).corpus_id, manifest["corpus_id"])

    def test_rewriting_identical_content_is_idempotent_and_different_content_is_refused(self):
        first = self.write()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir() if p.is_file()}
        self.assertEqual(self.write()["corpus_id"], first["corpus_id"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir() if p.is_file()})
        with self.assertRaisesRegex(ManifestError, "different corpus_id"):
            self.write([dataclasses.replace(self.docs[0], title="Other")])
        self.assertEqual(load_corpus(self.directory / MANIFEST_NAME).corpus_id, first["corpus_id"])

    def test_a_crash_at_every_rename_before_the_manifest_publishes_nothing_and_is_retryable(self):
        real_rename, real_replace = os.rename, os.replace
        calls = {"n": 0}
        def count(fn):
            def wrapper(*args, **kwargs):
                calls["n"] += 1
                return fn(*args, **kwargs)
            return wrapper
        with mock.patch("devsearch.data.corpus.os.rename", count(real_rename)), \
             mock.patch("devsearch.data.corpus.os.replace", count(real_replace)):
            reference = self.write()
        total = calls["n"]
        self.assertGreaterEqual(total, 5)                                         # four entries, then the manifest
        for crash_at in range(1, total + 1):                                      # includes the manifest rename itself
            with self.subTest(crash_at=crash_at):
                shutil.rmtree(self.directory)
                self.directory.mkdir()
                calls["n"] = 0
                def boom(real):
                    def wrapper(*args, **kwargs):
                        calls["n"] += 1
                        if calls["n"] == crash_at:
                            raise OSError("simulated crash")
                        return real(*args, **kwargs)
                    return wrapper
                with mock.patch("devsearch.data.corpus.os.rename", boom(real_rename)), \
                     mock.patch("devsearch.data.corpus.os.replace", boom(real_replace)):
                    with self.assertRaisesRegex(OSError, "simulated crash"):
                        self.write()
                self.assertFalse((self.directory / MANIFEST_NAME).exists(), "a manifest must not exist before publication completes")
                self.assertEqual([p for p in self.directory.iterdir() if p.name.startswith(".staging-")], [])
                retried = self.write()                                            # retry is safe, whatever was left behind
                self.assertEqual(retried["corpus_id"], reference["corpus_id"])
                self.assertEqual(len(load_corpus(self.directory / MANIFEST_NAME).chunks), 20)

    def test_remnants_of_an_interrupted_publish_are_set_aside_never_deleted(self):
        (self.directory / "text").mkdir()
        (self.directory / "text" / "0000.txt").write_text("remnant from an earlier attempt")
        (self.directory / "chunks.jsonl").write_text("remnant\n")
        self.write()
        abandoned = [p for p in self.directory.iterdir() if p.name.startswith(".abandoned-")]
        self.assertEqual(len(abandoned), 1)
        self.assertEqual((abandoned[0] / "text" / "0000.txt").read_text(), "remnant from an earlier attempt")
        self.assertEqual(len(load_corpus(self.directory / MANIFEST_NAME).chunks), 20)


class EvidenceGroupOwnershipRegressionTests(unittest.TestCase):
    """F02: one owner per evidence group across all chunks, so training and test cannot share evidence."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def write(self, *owners, **groups):
        return write_corpus(self.directory, _two_chunk_docs(*owners, **groups), chunker_version="c")

    def test_same_group_with_train_and_test_owners_is_rejected(self):
        with self.assertRaisesRegex(ManifestError, "conflicting owners.*'train'.*'test'"):
            self.write("train", "test")
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_same_group_mixing_unassigned_and_assigned_is_rejected_in_either_order(self):
        for owners in ((None, "train"), ("train", None), (None, "test"), ("validation", None)):
            with self.subTest(owners=owners), self.assertRaisesRegex(ManifestError, "conflicting owners"):
                self.write(*owners)

    def test_unassigned_groups_and_consistently_assigned_groups_are_allowed(self):
        self.write(None, None)
        for owner in ("train", "validation", "test", "development"):
            shutil.rmtree(self.directory)
            self.directory.mkdir()
            self.write(owner, owner)
            self.assertEqual({c.owner_split for c in load_corpus(self.directory / MANIFEST_NAME).chunks}, {owner})

    def test_different_groups_may_have_different_owners(self):
        self.write("train", "test", group_a="eg-a", group_b="eg-b")
        self.assertEqual(sorted(c.owner_split for c in load_corpus(self.directory / MANIFEST_NAME).chunks), ["test", "train"])

    def test_a_conflict_in_a_hand_built_corpus_with_consistent_hashes_is_rejected_at_load(self):
        # The reviewer's probe: hashes are valid, only the ownership is inconsistent.
        documents, chunks, files = fixture_parts()
        shared = next(c for c in chunks if c["evidence_group_id"] == "eg-tok-padding")
        twin = next(c for c in chunks if c["evidence_group_id"] == "eg-tok-padding" and c is not shared)
        shared["owner_split"], twin["owner_split"] = "train", "test"
        target = self.directory / "bad"
        target.mkdir()
        with self.assertRaisesRegex(ManifestError, "conflicting owners"):
            load_corpus(write_raw_corpus(target, documents, chunks, files))
