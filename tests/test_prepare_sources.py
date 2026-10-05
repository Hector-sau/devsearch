"""`prepare`: pinned snapshots -> corpus artifact + report (T03). Config strictness, acquisition and verification, quarantine,
determinism and the command. A whitespace counter stands in for the tokenizer except in the gated command tests."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from devsearch.artifacts.canonical import pretty_json
from devsearch.data.corpus import load_corpus
from devsearch.errors import ConfigError, ManifestError, PrerequisiteError
from devsearch.ingestion.config import load_prepare_config
from devsearch.ingestion.pipeline import prepare_corpus
from devsearch.ingestion.snapshot import acquire, content_hash, glob_regex, inventory, verify_revision
from mltest import requires_ml

REVISION = "a" * 40
GIT = shutil.which("git")

CONFIG = """schema_version = 1
parser_version = "static-extract-v1"
chunker_version = "chunker-v1"

[policy]
max_pair_tokens = 64
max_query_tokens = 16
min_chunk_tokens = 4
overlong_code = "split"
max_file_bytes = 4096
max_files = 100

[tokenizer]
assets = "tokenizer"

[[source]]
source_id = "demo-docs"
revision = "{revision}"
snapshot = "snapshot"
url_template = "https://example.org/demo/blob/{{revision}}/{{path}}"
license = "MIT"
license_file = "LICENSE"
include = ["docs/**", "tutorials/**", "notebooks/**"]
exclude = ["**/_static/**", "**/draft_*"]
"""

GUIDE = """# Gradients

Gradient accumulation sums gradients over several micro batches before one optimizer step is taken.

## Usage

Call backward on every micro batch and step once after the last of them.

```python
loss.backward()
optimizer.step()
```

## Legal

Copyright footer shared by every page of this documentation set today.
"""

NOTEBOOK = json.dumps({"nbformat": 4, "metadata": {}, "cells": [
    {"cell_type": "markdown", "id": "m1", "metadata": {}, "source": "# Notebook\n\nThis notebook explains how tensors hold numbers and shapes."},
    {"cell_type": "code", "id": "c1", "metadata": {}, "source": "x = torch.zeros(3)\nprint(x.shape)", "outputs": [{"output_type": "stream", "text": "OUTPUT"}]},
]})


def build_snapshot(root: Path) -> Path:
    snapshot = root / "snapshot"
    for relative, text in {
        "LICENSE": "MIT License\n",
        "docs/guide.md": GUIDE,
        "docs/api.mdx": "import X from 'y';\n\n# API\n\n<Tip>\n\nThe optimizer object exposes a step method for updates today.\n\n</Tip>\n",
        "docs/other.md": "# Other\n\nA different page about learning rates and their schedules in training.\n\n## Legal\n\nCopyright footer shared by every page of this documentation set today.\n",
        "docs/broken.md": "# Broken\n\n```python\nnever closed\n",
        "docs/notes.txt": "plain text is not a supported format\n",
        "docs/draft_x.md": "# Draft\n\nExcluded by pattern entirely.\n",
        "docs/_static/skip.md": "# Skip\n\nExcluded directory.\n",
        "tutorials/intro.rst": "Optimizers\n==========\n\nAn optimizer updates parameters using :func:`torch.optim.step` and gradients.\n\n.. toctree::\n\n   other\n",
        "tutorials/basics.py": '"""\nBasics\n======\n\nHow a tensor is created in practice here.\n"""\n\nimport torch\n\n# %%\n# Create one\n# ----------\n#\n# Making a tensor of zeros takes one call.\n\nx = torch.zeros(3)\n',
        "notebooks/demo.ipynb": NOTEBOOK,
    }.items():
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (snapshot / "docs" / "binary.md").write_bytes(b"abc\x00def\n")
    (snapshot / "docs" / "latin1.md").write_bytes("caf\xe9 is # not utf-8\n".encode("latin-1"))
    (snapshot / "docs" / "huge.md").write_text("# Huge\n\n" + "word " * 2000)
    os.symlink(snapshot / "docs" / "guide.md", snapshot / "docs" / "link.md")
    return snapshot


class Words:
    def count(self, text: str) -> int:
        return len(text.split())


class PrepareCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="devsearch-prep-")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.snapshot = build_snapshot(self.root)
        (self.root / "tokenizer").mkdir()
        self.config_path = self.root / "sources.toml"
        self.config_path.write_text(CONFIG.format(revision=REVISION))
        self.output = self.root / "corpus"
        self.output.mkdir()

    def config(self, text=None):
        if text is not None:
            self.config_path.write_text(text)
        return load_prepare_config(self.config_path)

    def prepare(self, config=None, output=None, **kw):
        config = config or self.config()
        return prepare_corpus(config, Words(), 40, output or self.output, allow_acquire=kw.pop("allow_acquire", False),
                              tokenizer_info={"fingerprint": "fake", "asset_kind": None}, config_sha256="0" * 64, **kw)


class BuildTests(PrepareCase):
    def test_every_format_is_read_and_the_corpus_loads_with_source_links(self):
        prepared = self.prepare()
        corpus = load_corpus(self.output / "corpus-manifest.json")
        by_doc = {}
        for chunk in corpus.chunks:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        self.assertEqual(sorted(by_doc), ["demo-docs:docs/api.mdx", "demo-docs:docs/guide.md", "demo-docs:docs/other.md", "demo-docs:notebooks/demo.ipynb",
                                          "demo-docs:tutorials/basics.py", "demo-docs:tutorials/intro.rst"])
        versions = {doc.doc_id: doc.parser_version for doc in corpus.docs.values()}
        self.assertEqual(versions["demo-docs:docs/guide.md"], "static-extract-v1/markdown")
        self.assertEqual(versions["demo-docs:docs/api.mdx"], "static-extract-v1/mdx")
        self.assertEqual(versions["demo-docs:tutorials/intro.rst"], "static-extract-v1/rst")
        self.assertEqual(versions["demo-docs:notebooks/demo.ipynb"], "static-extract-v1/notebook")
        self.assertEqual(versions["demo-docs:tutorials/basics.py"], "static-extract-v1/python-gallery")
        guide = next(doc for doc in corpus.docs.values() if doc.doc_id.endswith("guide.md"))
        self.assertEqual(guide.source_url, f"https://example.org/demo/blob/{REVISION}/docs/guide.md")
        self.assertEqual((guide.revision, guide.license, guide.title), (REVISION, "MIT", "Gradients"))
        for chunk in corpus.chunks:
            self.assertLessEqual(len(chunk.text.split()), 40)
            self.assertTrue(corpus.locate(chunk))
        notebook_locations = corpus.locate(by_doc["demo-docs:notebooks/demo.ipynb"][0])
        self.assertTrue(all("cell_id" in location for location in notebook_locations))
        self.assertNotIn("OUTPUT", " ".join(chunk.text for chunk in corpus.chunks), "notebook outputs are never indexed")
        self.assertNotIn("toctree", " ".join(chunk.text for chunk in corpus.chunks))
        self.assertEqual(prepared.manifest["corpus_id"], corpus.corpus_id)

    def test_the_report_accounts_for_everything_that_was_not_indexed(self):
        report = self.prepare().report
        source = report["sources"][0]
        self.assertEqual(source["revision_verification"], "declared-unverified")
        reasons = {item["path"]: item["reason"] for item in source["quarantine"]["listed"]}
        self.assertEqual(sorted(reasons), ["docs/binary.md", "docs/broken.md", "docs/huge.md", "docs/latin1.md", "docs/notes.txt"])
        self.assertIn("NUL", reasons["docs/binary.md"])
        self.assertIn("never closed", reasons["docs/broken.md"])
        self.assertIn("limit", reasons["docs/huge.md"])
        self.assertIn("UTF-8", reasons["docs/latin1.md"])
        self.assertIn("no static reader", reasons["docs/notes.txt"])
        link = next(item for item in source["quarantine"]["listed"] if item["path"] == "docs/broken.md")["source_url"]
        self.assertEqual(link, f"https://example.org/demo/blob/{REVISION}/docs/broken.md")
        self.assertEqual(source["files"]["quarantined"], 5)
        self.assertEqual(source["symlinks_skipped"], ["docs/link.md"])
        self.assertEqual(source["dropped_lines"]["mdx-statement"], 1)
        self.assertEqual(source["dropped_lines"]["rst-directive:toctree"], 3)
        self.assertEqual(source["passages"]["excluded"]["duplicate-passage"], 1, "the repeated footer is kept once")
        self.assertEqual(source["license"]["id"], "MIT")
        self.assertEqual(len(source["license"]["file_sha256"]), 64)
        self.assertIn("not verified", source["license"]["review"])
        totals = report["totals"]
        self.assertEqual(totals["passage_tokens"]["truncated"], 0)
        self.assertLessEqual(totals["passage_tokens"]["max"], 40)
        self.assertEqual(totals["passage_tokens"]["budget"], 40)
        self.assertTrue(any("tokenizer" in note for note in report["limitations"]))
        encoded = json.dumps(report)
        self.assertNotIn(str(self.root), encoded, "the report holds no absolute paths")

    def test_the_same_inputs_reproduce_the_corpus_and_the_report_byte_for_byte(self):
        first = self.prepare()
        other = self.root / "corpus2"
        other.mkdir()
        second = self.prepare(output=other)
        self.assertEqual(first.manifest["corpus_id"], second.manifest["corpus_id"])
        self.assertEqual(pretty_json(first.report), pretty_json(second.report))
        for name in ("documents.jsonl", "chunks.jsonl"):
            self.assertEqual((self.output / name).read_bytes(), (other / name).read_bytes())

    def test_a_published_corpus_is_immutable(self):
        self.prepare()
        again = self.prepare()                                        # identical: fine
        self.assertEqual(again.manifest["corpus_id"], load_corpus(self.output / "corpus-manifest.json").corpus_id)
        changed = self.config(CONFIG.format(revision=REVISION).replace("chunker-v1", "chunker-v2"))
        with self.assertRaisesRegex(ManifestError, "immutable"):
            prepare_corpus(changed, Words(), 40, self.output, allow_acquire=False, tokenizer_info={}, config_sha256="0" * 64)

    def test_nothing_publishable_is_an_error_not_an_empty_corpus(self):
        text = CONFIG.format(revision=REVISION).replace('include = ["docs/**", "tutorials/**", "notebooks/**"]', 'include = ["docs/notes.txt"]')
        with self.assertRaisesRegex(ManifestError, "no document produced any passage"):
            self.prepare(self.config(text))
        self.assertFalse((self.output / "corpus-manifest.json").exists())

    def test_offline_and_missing_acquisition_inputs_fail_closed(self):
        shutil.rmtree(self.snapshot)
        with self.assertRaisesRegex(ManifestError, "--offline forbids"):
            self.prepare(allow_acquire=False)
        with self.assertRaisesRegex(PrerequisiteError, "no git_url"):
            self.prepare(allow_acquire=True)


class GlobTests(unittest.TestCase):
    def test_double_star_crosses_directories_and_single_star_does_not(self):
        cases = [("**/*.md", "a.md", True), ("**/*.md", "a/b/c.md", True), ("*.md", "a/b.md", False), ("docs/**", "docs/x/y.rst", True),
                 ("docs/*.md", "docs/x/y.md", False), ("**/_static/**", "docs/_static/a.md", True), ("**/draft_*", "docs/draft_1.md", True),
                 ("a?c.md", "abc.md", True), ("a?c.md", "a/c.md", False), ("docs/**", "documents/x.md", False)]
        for pattern, path, expected in cases:
            with self.subTest(pattern=pattern, path=path):
                self.assertEqual(bool(glob_regex(pattern).match(path)), expected)


class ConfigTests(PrepareCase):
    def test_strict_validation_names_the_problem(self):
        good = CONFIG.format(revision=REVISION)
        cases = {
            "a branch is not a revision": good.replace(REVISION, "main"),
            "a short sha": good.replace(REVISION, "abc123"),
            "credentials in the template": good.replace("https://example.org", "https://user:token@example.org"),
            "an http template": good.replace("https://example.org", "http://example.org"),
            "unknown placeholder": good.replace("{path}", "{filename}"),
            "unknown key": good.replace('license = "MIT"', 'license = "MIT"\nsecret = "x"'),
            "empty include": good.replace('include = ["docs/**", "tutorials/**", "notebooks/**"]', "include = []"),
            "bad overlong_code": good.replace('overlong_code = "split"', 'overlong_code = "truncate"'),
            "missing tokenizer": good.replace('[tokenizer]\nassets = "tokenizer"\n', ""),
            "bad id": good.replace("demo-docs", "demo docs"),
            "credentialed git url": good.replace('license = "MIT"', 'license = "MIT"\ngit_url = "https://user:pw@github.com/x/y"'),
            "ssh git url": good.replace('license = "MIT"', 'license = "MIT"\ngit_url = "git@github.com:x/y.git"'),
            "bad pin": good.replace('license = "MIT"', 'license = "MIT"\nsnapshot_sha256 = "xyz"'),
        }
        for label, text in cases.items():
            with self.subTest(label):
                with self.assertRaises(ConfigError):
                    self.config(text)

    def test_duplicate_source_ids_are_refused(self):
        block = CONFIG.format(revision=REVISION).split("[[source]]")[1]
        with self.assertRaisesRegex(ConfigError, "unique"):
            self.config(CONFIG.format(revision=REVISION) + "\n[[source]]" + block)

    def test_relative_paths_resolve_against_the_config_file(self):
        config = self.config()
        self.assertEqual(config.sources[0].snapshot, self.snapshot)
        self.assertEqual(config.tokenizer_assets, self.root / "tokenizer")


class VerificationTests(PrepareCase):
    def run_git(self, *args, cwd=None):
        env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}
        return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()

    def spec(self, revision=REVISION, **changes):
        text = CONFIG.format(revision=revision)
        for old, new in changes.items():
            text = text.replace(old, new)
        return self.config(text).sources[0]

    def test_a_content_pin_verifies_and_a_mismatch_refuses(self):
        pin = content_hash(self.snapshot)
        self.assertEqual(verify_revision(self.spec(**{'license = "MIT"': f'license = "MIT"\nsnapshot_sha256 = "{pin}"'})), "verified-content-hash")
        (self.snapshot / "docs" / "guide.md").write_text(GUIDE + "\nchanged\n")
        with self.assertRaisesRegex(ManifestError, "differs from the pinned"):
            verify_revision(self.spec(**{'license = "MIT"': f'license = "MIT"\nsnapshot_sha256 = "{pin}"'}))

    @unittest.skipUnless(GIT, "git is not installed")
    def test_a_git_checkout_is_verified_against_its_head_and_a_dirty_tree_is_refused(self):
        self.run_git("init", "-q", str(self.snapshot))
        self.run_git("-C", str(self.snapshot), "add", "-A")
        self.run_git("-C", str(self.snapshot), "commit", "-q", "-m", "snapshot")
        head = self.run_git("-C", str(self.snapshot), "rev-parse", "HEAD")
        self.assertEqual(verify_revision(self.spec(head)), "verified-git-head")
        with self.assertRaisesRegex(ManifestError, "not the pinned"):
            verify_revision(self.spec("b" * 40))
        (self.snapshot / "docs" / "guide.md").write_text("tampered\n")
        with self.assertRaisesRegex(ManifestError, "modified"):
            verify_revision(self.spec(head))

    @unittest.skipUnless(GIT, "git is not installed")
    def test_acquisition_fetches_the_exact_commit_runs_no_hooks_and_leaves_nothing_on_failure(self):
        remote = self.root / "remote"
        shutil.copytree(self.snapshot, remote, symlinks=True)
        self.run_git("init", "-q", str(remote))
        hook = remote / ".git" / "hooks" / "post-checkout"
        hook.write_text(f"#!/bin/sh\ntouch {self.root / 'hook-ran'}\n")
        hook.chmod(0o755)
        self.run_git("-C", str(remote), "add", "-A")
        self.run_git("-C", str(remote), "commit", "-q", "-m", "one")
        head = self.run_git("-C", str(remote), "rev-parse", "HEAD")
        target = self.root / "fetched"
        spec = self.spec(head, **{'snapshot = "snapshot"': 'snapshot = "fetched"', 'license = "MIT"': f'license = "MIT"\ngit_url = "file://{remote}"'})
        acquire(spec)
        self.assertEqual(verify_revision(spec), "verified-git-head")
        self.assertTrue((target / "docs" / "guide.md").is_file())
        self.assertFalse((self.root / "hook-ran").exists(), "no hook from the source ever runs")
        with self.assertRaisesRegex(ManifestError, "already exists"):
            acquire(spec)
        missing = self.spec("c" * 40, **{'snapshot = "snapshot"': 'snapshot = "other"', 'license = "MIT"': f'license = "MIT"\ngit_url = "file://{remote}"'})
        with self.assertRaises(PrerequisiteError):
            acquire(missing)
        self.assertFalse((self.root / "other").exists(), "a failed acquisition removes the directory it created")

    @unittest.skipUnless(GIT, "git is not installed")
    def test_the_pipeline_acquires_a_missing_snapshot_when_allowed(self):
        remote = self.root / "remote"
        shutil.copytree(self.snapshot, remote, symlinks=True)
        self.run_git("init", "-q", str(remote))
        self.run_git("-C", str(remote), "add", "-A")
        self.run_git("-C", str(remote), "commit", "-q", "-m", "one")
        head = self.run_git("-C", str(remote), "rev-parse", "HEAD")
        shutil.rmtree(self.snapshot)
        config = self.config(CONFIG.format(revision=head).replace('license = "MIT"', f'license = "MIT"\ngit_url = "file://{remote}"'))
        report = self.prepare(config, allow_acquire=True).report
        self.assertEqual(report["sources"][0]["revision_verification"], "verified-git-head")

    def test_inventory_skips_symlinks_and_git_metadata(self):
        (self.snapshot / ".git").mkdir()
        (self.snapshot / ".git" / "config.md").write_text("# not a doc\n")
        found = inventory(self.spec(), 100)
        self.assertNotIn(".git/config.md", found.included)
        self.assertEqual(found.symlinks, ("docs/link.md",))
        too_many = self.spec()
        with self.assertRaisesRegex(ManifestError, "more than 3 files"):
            inventory(too_many, 3)


@requires_ml
class CommandTests(PrepareCase):
    def setUp(self):
        super().setUp()
        from devsearch import prepare
        from devsearch.modeling.tokenization import write_diagnostic_tokenizer
        self.module = prepare
        words = {w for text in (GUIDE, NOTEBOOK) for w in text.replace("#", " ").replace("`", " ").replace(".", " ").split()}
        write_diagnostic_tokenizer(self.root / "tokenizer", sorted(words))
        self.report = self.root / "corpus.prepare-report.json"

    def run_cli(self, *extra, output=None):
        out, err = io.StringIO(), io.StringIO()
        argv = ["--config", str(self.config_path), "--output", str(output or self.output), "--runtime-root", str(self.root), *extra]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = self.module.main(argv)
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue(), err.getvalue()

    def test_the_command_writes_the_corpus_and_a_report_with_the_tokenizer_identity(self):
        code, out, err = self.run_cli("--offline")
        self.assertEqual((code, err), (0, ""), err)
        summary = json.loads(out)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(Path(summary["report"]), self.report)
        report = json.loads(self.report.read_text())
        self.assertEqual(report["tokenizer"]["asset_kind"], None)
        self.assertEqual(len(report["tokenizer"]["fingerprint"]), 64)
        self.assertEqual(report["policy"]["passage_budget"], 64 - 16 - report["tokenizer"]["special_tokens_per_pair"])
        corpus = load_corpus(self.output / "corpus-manifest.json")
        self.assertEqual(summary["corpus_id"], corpus.corpus_id)
        self.assertTrue(corpus.chunks)

    def test_a_rerun_is_idempotent_and_a_changed_report_is_never_overwritten(self):
        self.assertEqual(self.run_cli("--offline")[0], 0)
        before = self.report.read_bytes()
        code, _, err = self.run_cli("--offline")
        self.assertEqual((code, err), (0, ""), err)
        self.assertEqual(self.report.read_bytes(), before)
        self.report.write_text("{}")
        code, _, err = self.run_cli("--offline")
        self.assertEqual(code, 1)
        self.assertIn("never overwritten", err)

    def test_failures_exit_1_with_a_reason_and_no_traceback(self):
        shutil.rmtree(self.snapshot)
        code, out, err = self.run_cli("--offline")
        self.assertEqual((code, out), (1, ""))
        self.assertIn("--offline forbids", err)
        self.assertNotIn("Traceback", err)
        self.snapshot.mkdir()
        code, _, err = self.run_cli("--offline", output=self.root / "nope")
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)
        outside = Path(tempfile.mkdtemp(prefix="devsearch-outside-"))
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        code, _, err = self.run_cli("--offline", output=outside)
        self.assertEqual(code, 1)
        self.assertIn("corpus output", err)


if __name__ == "__main__":
    unittest.main()
