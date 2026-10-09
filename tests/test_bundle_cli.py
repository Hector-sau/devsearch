"""`devsearch.bundle register-assets`: hashing already-present model files, pinned only by a full commit id."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from devsearch import bundle
from devsearch.modeling.assets import ASSETS_MANIFEST, verify_assets

REVISION = "0123456789abcdef0123456789abcdef01234567"


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = bundle.main([str(a) for a in argv])
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


class RegisterAssetsTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="devsearch-assets-")).resolve()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        (self.dir / "config.json").write_text(json.dumps({"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 12, "vocab_size": 30522, "max_position_embeddings": 512}))
        (self.dir / "model.safetensors").write_bytes(b"not real weights")
        (self.dir / "tokenizer.json").write_text("{}")

    def register(self, **overrides):
        values = {"--directory": self.dir, "--repo": "microsoft/MiniLM-L12-H384-uncased", "--revision": REVISION, **overrides}
        return run("register-assets", *[item for pair in values.items() for item in pair if item is not None])

    def test_a_full_commit_registers_pinned_assets_and_they_verify(self):
        code, out, err = self.register(**{"--license": "mit"})
        self.assertEqual((code, err), (0, ""), err)
        report = json.loads(out)
        self.assertEqual((report["pinned"], report["asset_kind"], report["source"]["revision"]), (True, "pretrained", REVISION))
        self.assertIn("nothing here proves", report["note"])
        self.assertEqual(verify_assets(self.dir)["manifest_sha256"], report["manifest_sha256"])
        again = self.register(**{"--license": "mit"})
        self.assertEqual(again[0], 0, "an identical registration is a no-op")

    def test_a_branch_name_is_not_a_pin_and_nothing_is_written(self):
        code, out, err = self.register(**{"--revision": "main"})
        self.assertEqual((code, out), (1, ""))
        self.assertIn("40-hex", err)
        self.assertFalse((self.dir / ASSETS_MANIFEST).exists())

    def test_missing_files_a_different_registration_and_missing_options_fail_cleanly(self):
        (self.dir / "model.safetensors").unlink()
        code, _, err = self.register()
        self.assertEqual(code, 1)
        self.assertIn("model.safetensors", err)
        (self.dir / "model.safetensors").write_bytes(b"x")
        self.assertEqual(self.register()[0], 0)
        code, _, err = self.register(**{"--repo": "someone/else"})
        self.assertEqual(code, 1)
        self.assertIn("immutable", err)
        self.assertEqual(run("register-assets", "--directory", self.dir)[0], 2)
        self.assertEqual(run("assemble")[0], 2)


if __name__ == "__main__":
    unittest.main()
