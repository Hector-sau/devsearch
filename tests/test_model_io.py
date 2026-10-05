"""Inference bundle: export, strict reload, tamper detection, immutability, crash safety and parity."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch.errors import ManifestError
from helpers import RootTestCase
from mltest import HAVE_ML

if HAVE_ML:
    import torch
    from devsearch.modeling import io as model_io
    from devsearch.modeling.io import check_parity, export_model, load_model_bundle, read_manifest, verify_bundle
    from devsearch.modeling.scoring import score_pairs
    from mlworld import MLWorld

CPU = None


@unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")
class BundleTests(RootTestCase):
    def setUp(self):
        super().setUp()
        self.w = MLWorld.build_ml(self)
        self.model = self.w.model(dropout=0.1, encoder_seed=3, head_seed=5)
        self.model.head.bias.data.fill_(0.25)          # make the head differ from its init so a skipped load shows
        self.out = Path(tempfile.mkdtemp(prefix="devsearch-bundle-")).resolve()
        self.addCleanup(shutil.rmtree, self.out, ignore_errors=True)
        self.examples = [(p.query_text, p.passage_text) for p in self.w.train_pairs[:6]]

    def export(self, name="model", model=None, **kwargs):
        return export_model(
            model or self.model, self.out / name, tokenizer=self.w.tokenizer, policy=self.w.policy,
            provenance={"purpose": "unit-test"}, parity_examples=self.examples, device=torch.device("cpu"), **kwargs)

    def test_reload_reproduces_the_exported_logits_exactly(self):
        manifest = self.export()
        loaded = load_model_bundle(self.out / "model", device=torch.device("cpu"))
        self.assertEqual(loaded.model_id, manifest["model_id"])
        before = score_pairs(self.model, self.w.tokenizer, self.w.policy, self.examples, device=torch.device("cpu"), batch_size=3)
        after = score_pairs(loaded.model, loaded.tokenizer, loaded.policy, self.examples, device=torch.device("cpu"), batch_size=2)
        self.assertEqual(before, after)
        self.assertFalse(loaded.model.training)
        report = check_parity(loaded, device=torch.device("cpu"))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["max_abs_difference"], 0.0)

    def test_the_bundle_contains_state_dicts_not_pickles_and_lists_every_file(self):
        manifest = self.export()
        names = {p.relative_to(self.out / "model").as_posix() for p in (self.out / "model").rglob("*") if p.is_file()}
        self.assertEqual(names, set(manifest["files"]) | {"model-manifest.json"})
        self.assertFalse(any(n.endswith((".pt", ".pth", ".pkl", ".bin")) for n in names))
        self.assertIn("encoder.safetensors", names)
        self.assertTrue(any(n.startswith("tokenizer/") for n in names))

    def test_identity_depends_on_weights_but_not_on_provenance(self):
        first = self.export("a")
        second = export_model(self.model, self.out / "b", tokenizer=self.w.tokenizer, policy=self.w.policy,
                              provenance={"purpose": "another run"}, parity_examples=self.examples, device=torch.device("cpu"))
        self.assertEqual(first["model_id"], second["model_id"])
        self.assertNotEqual(first["manifest_sha256"], second["manifest_sha256"])
        changed = self.w.model(dropout=0.1, encoder_seed=3, head_seed=5)
        changed.head.bias.data.fill_(0.5)
        self.assertNotEqual(self.export("c", model=changed)["model_id"], first["model_id"])

    def test_an_identical_reexport_is_idempotent_and_a_different_one_is_refused(self):
        first = self.export()
        self.assertEqual(self.export()["model_id"], first["model_id"])
        changed = self.w.model(dropout=0.1, encoder_seed=4, head_seed=5)
        with self.assertRaisesRegex(ManifestError, "immutable"):
            self.export(model=changed)
        self.assertEqual(verify_bundle(self.out / "model")["model_id"], first["model_id"])

    def test_a_crash_before_the_final_rename_leaves_no_bundle_and_is_retryable(self):
        with mock.patch.object(model_io.os, "rename", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.export()
        self.assertEqual(list(self.out.iterdir()), [])
        self.assertEqual(self.export()["kind"], "model")

    def test_export_refuses_missing_parent_and_bad_parity_counts(self):
        with self.assertRaisesRegex(ManifestError, "never creates parent"):
            export_model(self.model, self.out / "missing" / "m", tokenizer=self.w.tokenizer, policy=self.w.policy,
                         provenance={}, parity_examples=self.examples, device=torch.device("cpu"))
        with self.assertRaisesRegex(ManifestError, "parity examples"):
            export_model(self.model, self.out / "m", tokenizer=self.w.tokenizer, policy=self.w.policy,
                         provenance={}, parity_examples=[], device=torch.device("cpu"))

    def test_tampering_is_detected_before_any_tensor_is_loaded(self):
        self.export()
        bundle = self.out / "model"
        weights = bundle / "encoder.safetensors"
        original = weights.read_bytes()
        weights.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with self.assertRaisesRegex(ManifestError, "encoder.safetensors"):
            load_model_bundle(bundle, device=torch.device("cpu"))
        weights.write_bytes(original)
        load_model_bundle(bundle, device=torch.device("cpu"))
        (bundle / "extra.txt").write_text("x")
        with self.assertRaisesRegex(ManifestError, "extra"):
            verify_bundle(bundle)
        (bundle / "extra.txt").unlink()
        (bundle / "tokenizer" / "tokenizer.json").write_text("{}")
        with self.assertRaisesRegex(ManifestError, "tokenizer"):
            verify_bundle(bundle)

    def test_manifest_edits_and_escaping_names_are_rejected(self):
        self.export()
        path = self.out / "model" / "model-manifest.json"
        original = path.read_text(encoding="utf-8")
        data = json.loads(original)
        data["provenance"] = {"purpose": "edited"}
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ManifestError, "manifest_sha256"):
            verify_bundle(self.out / "model")
        from devsearch.artifacts.canonical import seal_manifest
        data = json.loads(original)
        data["files"]["../escape"] = data["files"]["encoder.safetensors"]
        path.write_text(json.dumps(seal_manifest(data)))
        with self.assertRaisesRegex(ManifestError, "escapes"):
            verify_bundle(self.out / "model")
        data = json.loads(original)
        data["unexpected"] = 1
        path.write_text(json.dumps(data))
        with self.assertRaises(ManifestError):
            verify_bundle(self.out / "model")

    def test_a_symlinked_file_is_not_accepted(self):
        self.export()
        bundle = self.out / "model"
        target = self.out / "elsewhere.safetensors"
        shutil.copyfile(bundle / "head.safetensors", target)
        (bundle / "head.safetensors").unlink()
        (bundle / "head.safetensors").symlink_to(target)
        with self.assertRaisesRegex(ManifestError, "head.safetensors"):
            verify_bundle(bundle)

    def test_wrong_shape_weights_with_a_consistent_manifest_fail_strictly(self):
        manifest = self.export()
        other = self.w.model(hidden=48)                 # a different architecture, same file names
        self.export("other", model=other)
        bundle, donor = self.out / "model", self.out / "other"
        shutil.copyfile(donor / "encoder.safetensors", bundle / "encoder.safetensors")
        with self.assertRaises(ManifestError):          # hash no longer matches the manifest it was exported with
            load_model_bundle(bundle, device=torch.device("cpu"))
        self.assertEqual(manifest["kind"], "model")

    def test_parity_fails_when_the_recorded_logits_no_longer_match(self):
        self.export()
        loaded = load_model_bundle(self.out / "model", device=torch.device("cpu"))
        with torch.no_grad():
            loaded.model.head.bias.add_(1.0)
        report = check_parity(loaded, device=torch.device("cpu"))
        self.assertFalse(report["passed"])
        self.assertGreater(report["max_abs_difference"], 0.5)
        looser = check_parity(loaded, device=torch.device("cpu"), atol=2.0)
        self.assertTrue(looser["passed"])
        self.assertEqual(looser["atol"], 2.0)

    def test_the_manifest_is_independent_of_the_destination_path(self):
        first = self.export("one")
        second = self.export("two")
        self.assertEqual(first["files"], second["files"])
        self.assertNotIn(str(self.out), json.dumps(first))

    def test_read_manifest_rejects_a_missing_or_foreign_manifest(self):
        with self.assertRaises(ManifestError):
            read_manifest(self.out / "absent")
        (self.out / "foreign").mkdir()
        (self.out / "foreign" / "model-manifest.json").write_text(json.dumps({"schema_version": 2}))
        with self.assertRaises(ManifestError):
            read_manifest(self.out / "foreign")


if __name__ == "__main__":
    unittest.main()
