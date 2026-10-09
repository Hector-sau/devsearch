"""Exercise grouped input loading and the installed CLI with synthetic, local-only model assets."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mltest import HAVE_ML, requires_ml

if HAVE_ML:
    import torch
    from devsearch import train
    from devsearch.artifacts.canonical import pretty_json, seal_manifest, sha256_file
    from devsearch.errors import ConfigError, ManifestError
    from devsearch.execution import rehearsal
    from devsearch.training.inputs import load_training_inputs
    from devsearch.training.job import load_train_job
    from opsworld import TRAIN_TOML, call


@requires_ml
class GroupedOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(tempfile.mkdtemp(prefix="devsearch-grouped-cli-")).resolve()
        cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        # Only this original synthetic fixture changes mining composition. Production data stays frozen.
        data_toml = rehearsal.DATA_TOML.replace("hard_negatives = 2", "hard_negatives = 3").replace("random_negatives = 2", "random_negatives = 1")
        question = rehearsal._question
        # Add a shared fixture term so every synthetic query retrieves >=3 train negatives.
        # The ordinary rehearsal intentionally retrieves only its two sibling sections.
        with mock.patch.object(rehearsal, "DATA_TOML", data_toml), mock.patch.object(
                rehearsal, "_question", side_effect=lambda *args: question(*args) + " handle"):
            rehearsal.build_rehearsal(cls.root, topics=24)
        cls.manifest = cls.root / "bundle/prepared/prepared-data-manifest.json"
        cls.alternate = cls.root / "runtime/reviewed-pairs"
        shutil.copytree(cls.manifest.parent / "pairs", cls.alternate)
        cls.pairs_manifest = cls.alternate / "pairs-manifest.json"
        cls.pairs_sha = sha256_file(cls.pairs_manifest)

    def config(self, kind, name):
        text = (TRAIN_TOML.format(epochs=3).replace('mode = "frozen"', 'mode = "full"')
                .replace("learning_rate = 0.01", "learning_rate = 0.00005")
                .replace("effective_batch = 8", "effective_batch = 20").replace("microbatch = 4", "microbatch = 5")
                .replace("dropout = 0.0", "dropout = 0.1").replace("head_seed = 0", "head_seed = 1234")
                .replace('pos_weight = "auto"', f'pos_weight = 4\nkind = "{kind}"'))
        path = self.root / (name + ".toml")
        path.write_text(text)
        return path

    def test_grouped_loader_retains_ids_and_verified_sources(self):
        inputs = load_training_inputs(self.manifest, self.root, grouped=True)
        data = inputs.grouped_data
        self.assertGreater(len(data.groups), 0)
        self.assertEqual(tuple(inputs.train_pairs), data.pairs)
        self.assertEqual(dict(data.source_sha256)["prepared_manifest"], sha256_file(self.manifest))
        self.assertEqual(dict(data.source_sha256)["assignment"], sha256_file(self.manifest.parent / "assignment.json"))
        for group in data.groups:
            self.assertEqual([p.record.label for p in group.records], [1, 0, 0, 0, 0])
            self.assertTrue(all(p.owner_split == "train" for p in group.records))
        self.assertIsNone(load_training_inputs(self.manifest, self.root).grouped_data)

    def test_both_cli_objectives_run_and_resume_outside_source_assumptions(self):
        outcomes = []
        for i, kind in enumerate(("group_bce_weight4", "pairwise_logistic_scale_4_over_5")):
            config = self.config(kind, f"grouped-{i}")
            config.write_text(config.read_text().replace('[data]', '[data]\npairs_manifest = "runtime/reviewed-pairs/pairs-manifest.json"\n'
                                                        f'pairs_manifest_sha256 = "{self.pairs_sha}"'))
            directory = self.root / "runtime" / f"grouped-{i}"
            directory.mkdir()
            args = ("--config", config, "--run-dir", directory, "--runtime-root", self.root)
            code, out, err = call(train, *args)
            self.assertEqual(code, 0, err)
            summary = json.loads(out)
            self.assertEqual(summary["best_epoch"], 3)
            self.assertEqual(summary["completed_epochs"], 3)
            receipt = json.loads((directory / "training-summary.json").read_text())
            self.assertEqual(receipt["identity"]["checkpoint_selection"], "fixed-final-epoch-3-v1")
            outcomes.append(receipt)
            code, out, err = call(train, *args, "--resume", "latest")
            self.assertEqual(code, 0, err)
            self.assertEqual(json.loads(out)["global_step"], summary["global_step"])
        self.assertEqual(outcomes[0]["plan"], outcomes[1]["plan"])
        self.assertEqual(outcomes[0]["lr_trace"], outcomes[1]["lr_trace"])
        self.assertEqual(outcomes[0]["identity"]["grouped_data"], outcomes[1]["identity"]["grouped_data"])
        self.assertNotEqual(outcomes[0]["identity"]["config_sha256"], outcomes[1]["identity"]["config_sha256"])

    def test_modified_train_file_is_rejected_before_model_loading(self):
        path = self.manifest.parent / "data/queries-train.jsonl"
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        path.write_bytes(original + b"\n")
        with self.assertRaises(ManifestError):
            load_training_inputs(self.manifest, self.root, grouped=True)

    def test_missing_group_ownership_is_explicit_failure(self):
        from devsearch.data.prepare import load_prepared
        prepared = load_prepared(self.manifest, self.root)
        prepared["split"]["merged_of"] = {}
        with mock.patch("devsearch.training.inputs.load_prepared", return_value=prepared):
            with self.assertRaisesRegex(ManifestError, "complete merged-group ownership"):
                load_training_inputs(self.manifest, self.root, grouped=True)

    def test_pairs_override_requires_hash_grouped_mode_and_root_containment(self):
        kw = dict(grouped=True, pairs_manifest=self.pairs_manifest, pairs_manifest_sha256=self.pairs_sha)
        for change in ({"pairs_manifest_sha256": None}, {"pairs_manifest_sha256": "f" * 64}, {"grouped": False},
                       {"pairs_manifest": self.root.parent / "outside-pairs.json"}):
            with self.subTest(change=change), self.assertRaises(ManifestError):
                load_training_inputs(self.manifest, self.root, **{**kw, **change})
        cfg = self.config("group_bce_weight4", "bad-override")
        original = cfg.read_text()
        for fields in ('pairs_manifest = "x"', 'pairs_manifest_sha256 = "a"',
                       'pairs_manifest = "x"\npairs_manifest_sha256 = "bad-hash"'):
            cfg.write_text(original.replace('[data]', '[data]\n' + fields))
            with self.subTest(fields=fields), self.assertRaises(ConfigError):
                load_train_job(cfg)

    def test_validly_hashed_override_cannot_reference_other_query_qrel_or_index(self):
        original = self.pairs_manifest.read_bytes()
        self.addCleanup(self.pairs_manifest.write_bytes, original)
        for field in ("queries_sha256", "qrels_sha256", "index_id"):
            changed = json.loads(original)
            changed[field] = "f" * 64 if field != "index_id" else "idx-other"
            self.pairs_manifest.write_text(pretty_json(seal_manifest(changed)))
            with self.subTest(field=field), self.assertRaisesRegex(ManifestError, "do not bind"):
                load_training_inputs(self.manifest, self.root, grouped=True, pairs_manifest=self.pairs_manifest,
                                     pairs_manifest_sha256=sha256_file(self.pairs_manifest))

    def test_pair_child_symlinks_cannot_escape_runtime_root_even_with_matching_bytes(self):
        with tempfile.TemporaryDirectory(prefix="devsearch-outside-root-") as outside:
            for name in ("pairs.jsonl", "sampling.jsonl", "mining-pool.json"):
                child = self.alternate / name
                original = child.read_bytes()
                target = Path(outside) / name
                target.write_bytes(original)
                child.unlink()
                child.symlink_to(target)
                try:
                    with self.subTest(name=name), self.assertRaisesRegex(ManifestError, "escapes the root"):
                        load_training_inputs(self.manifest, self.root, grouped=True, pairs_manifest=self.pairs_manifest,
                                             pairs_manifest_sha256=self.pairs_sha)
                finally:
                    child.unlink()
                    child.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
