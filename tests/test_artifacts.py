"""Canonical hashing, strict JSON, path containment, schema validators and code identity."""

import shutil
import tempfile
import unittest
from pathlib import Path

from devsearch.artifacts import canonical
from devsearch.artifacts.identity import code_identity, source_inventory
from devsearch.artifacts.paths import RUNTIME_ROOT_ENV, ensure_inside, relative_reference, resolve_reference, resolve_root
from devsearch.artifacts.schema import as_bool, as_https_url, as_id, as_int, as_label, exact_keys
from devsearch.errors import ConfigError, ManifestError


class CanonicalTests(unittest.TestCase):
    def test_canonical_json_is_stable_and_compact(self):
        self.assertEqual(canonical.canonical_json({"b": 1, "a": [True, None, "é"]}), '{"a":[true,null,"é"],"b":1}')
        self.assertEqual(canonical.canonical_json({"a": 1, "b": 2}), canonical.canonical_json({"b": 2, "a": 1}))

    def test_non_finite_numbers_are_rejected_both_ways(self):
        with self.assertRaises(ValueError):
            canonical.canonical_json({"x": float("nan")})
        for text in ('{"x": NaN}', '{"x": Infinity}'):
            with self.assertRaisesRegex(ManifestError, "non-finite"):
                canonical.loads_strict(text, where="t")

    def test_duplicate_keys_and_bad_json_are_rejected(self):
        with self.assertRaisesRegex(ManifestError, "duplicate key"):
            canonical.loads_strict('{"a": 1, "a": 2}', where="t")
        with self.assertRaisesRegex(ManifestError, "invalid JSON"):
            canonical.loads_strict("{", where="t")

    def test_manifest_hash_excludes_itself_and_detects_edits(self):
        sealed = canonical.seal_manifest({"kind": "x", "n": 1})
        canonical.verify_manifest_hash(sealed, where="t")
        self.assertEqual(canonical.seal_manifest(sealed), sealed)   # sealing is idempotent
        with self.assertRaisesRegex(ManifestError, "manifest_sha256"):
            canonical.verify_manifest_hash({**sealed, "n": 2}, where="t")
        with self.assertRaisesRegex(ManifestError, "manifest_sha256"):
            canonical.verify_manifest_hash({"kind": "x", "n": 1}, where="t")   # missing hash

    def test_known_sha256_vector(self):
        self.assertEqual(
            canonical.sha256_text("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_invalid_utf8_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(ManifestError, "UTF-8"):
                canonical.read_utf8(path, where="t")


class PathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "inside").mkdir()
        (self.root / "inside" / "a.json").write_text("{}")
        self.outside = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(shutil.rmtree, self.outside, ignore_errors=True)
        (self.outside / "secret.txt").write_text("x")

    def resolve(self, relative):
        return resolve_reference(self.root / "inside", relative, self.root, what="ref")

    def test_valid_reference_and_parent_hop_within_root(self):
        self.assertEqual(self.resolve("a.json"), self.root / "inside" / "a.json")
        (self.root / "b.json").write_text("{}")
        self.assertEqual(self.resolve("../b.json"), self.root / "b.json")

    def test_rejected_references(self):
        cases = {
            "absolute": str(self.root / "inside" / "a.json"),
            "backslash": "sub\\a.json",
            "drive": "C:/a.json",
            "escape": "../../" + self.outside.name + "/secret.txt",
            "missing": "nope.json",
            "directory": ".",
            "empty": "",
            "nul": "a\x00.json",
        }
        for name, relative in cases.items():
            with self.subTest(name):
                with self.assertRaises(ManifestError):
                    self.resolve(relative)
        with self.assertRaises(ManifestError):
            self.resolve(None)

    def test_symlink_escape_is_rejected(self):
        link = self.root / "inside" / "link.txt"
        try:
            link.symlink_to(self.outside / "secret.txt")
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(ManifestError, "escapes"):
            self.resolve("link.txt")

    def test_ensure_inside_and_relative_reference(self):
        self.assertEqual(ensure_inside(self.root / "inside" / "new.json", self.root, what="x"), self.root / "inside" / "new.json")
        with self.assertRaisesRegex(ManifestError, "escapes"):
            ensure_inside(self.outside / "secret.txt", self.root, what="x")
        self.assertEqual(relative_reference(self.root / "inside", self.root / "b.json"), "../b.json")

    def test_resolve_root_precedence_and_missing_root(self):
        manifest = self.root / "inside" / "m.json"
        self.assertEqual(resolve_root(None, manifest, {}), self.root / "inside")
        self.assertEqual(resolve_root(None, manifest, {RUNTIME_ROOT_ENV: str(self.root)}), self.root)
        self.assertEqual(resolve_root(self.outside, manifest, {RUNTIME_ROOT_ENV: str(self.root)}), self.outside)
        missing = self.root / "unmounted-volume"
        with self.assertRaisesRegex(ConfigError, "does not exist"):
            resolve_root(missing, manifest, {})
        self.assertFalse(missing.exists(), "a missing root must never be created")
        with self.assertRaisesRegex(ConfigError, "not a directory"):
            resolve_root(self.root / "inside" / "a.json", manifest, {})


class SchemaTests(unittest.TestCase):
    def test_exact_keys_reports_missing_and_unknown(self):
        exact_keys({"a": 1}, ("a",), "w")
        with self.assertRaisesRegex(ManifestError, "missing required field.*b"):
            exact_keys({"a": 1}, ("a", "b"), "w")
        with self.assertRaisesRegex(ManifestError, "unrecognized field.*z"):
            exact_keys({"a": 1, "z": 2}, ("a",), "w")

    def test_ints_do_not_accept_bools_or_floats(self):
        self.assertEqual(as_int(3, "w"), 3)
        for bad in (True, 1.0, "1", None, -1):
            with self.subTest(bad=bad), self.assertRaises(ManifestError):
                as_int(bad, "w")
        with self.assertRaises(ManifestError):
            as_bool(1, "w")

    def test_ids_and_labels(self):
        self.assertEqual(as_id("pytorch:tutorials/a_b.md", "w"), "pytorch:tutorials/a_b.md")
        for bad in ("", " x", "x y", "x\n", "-lead", "a" * 300):
            with self.subTest(bad=bad), self.assertRaises(ManifestError):
                as_id(bad, "w")
        for bad in ("  padded", "line\nbreak", "tab\there", ""):
            with self.subTest(bad=bad), self.assertRaises(ManifestError):
                as_label(bad, "w")

    def test_urls_must_be_credential_free_https(self):
        self.assertEqual(as_https_url("https://example.invalid/a#b", "w"), "https://example.invalid/a#b")
        for bad in (
            "javascript:alert(1)", "http://example.invalid/", "data:text/html,<b>", "file:///etc/passwd",
            "https://user:pw@example.invalid/", "https:///no-host", "https://exa mple.invalid/",
            "//example.invalid/", "", "https://example.invalid/\n",
        ):
            with self.subTest(bad=bad), self.assertRaises(ManifestError):
                as_https_url(bad, "w")


class IdentityTests(unittest.TestCase):
    def test_code_identity_is_reproducible_and_tracks_content(self):
        first, second = code_identity(), code_identity()
        self.assertEqual(first, second)
        self.assertRegex(first["source_snapshot_sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(first["source_file_count"], 10)
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "pkg"
            package.mkdir()
            (package / "a.py").write_text("x = 1\n")
            before = code_identity(package)["source_snapshot_sha256"]
            (package / "a.py").write_text("x = 2\n")
            self.assertNotEqual(before, code_identity(package)["source_snapshot_sha256"])
            self.assertEqual(source_inventory(package)[0][0], "a.py")

    def test_inventory_ignores_bytecode_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "__pycache__").mkdir()
            (package / "__pycache__" / "c.py").write_text("junk")
            (package / "m.py").write_text("x")
            self.assertEqual([name for name, _ in source_inventory(package)], ["m.py"])


if __name__ == "__main__":
    unittest.main()
