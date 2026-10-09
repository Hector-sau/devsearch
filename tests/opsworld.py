"""Shared fixtures for the operator-command tests: a diagnostic tree with one trained run, plus a CLI call helper."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mltest import HAVE_ML, requires_ml

if HAVE_ML:
    from devsearch import train
    from devsearch.training.inputs import load_training_inputs
    from rehearsal_fixture import copy_rehearsal

TRAIN_TOML = """schema_version = 1

[run]
recipe_id = "cli-frozen"
mode = "frozen"
seed = 42
epochs = {epochs}

[optimizer]
learning_rate = 0.01
weight_decay = 0.01
betas = [0.9, 0.999]
eps = 1e-8
max_grad_norm = 1.0
warmup_fraction = 0.1

[batch]
effective_batch = 8
microbatch = 4

[model]
dropout = 0.0
head_seed = 0

[loss]
pos_weight = "auto"

[validation]
batch_size = 8

[data]
prepared_manifest = "bundle/prepared/prepared-data-manifest.json"

[assets]
directory = "bundle/model-assets"

[device]
profile = "cpu-diagnostic"

[policy]
max_pair_tokens = 128
max_query_tokens = 32
"""

MODEL_TOML = """schema_version = 1

[device]
profile = "cpu-diagnostic"

[model]
batch_size = 4

[assets]
directory = "bundle/model-assets"

[policy]
max_pair_tokens = 128
max_query_tokens = 32
"""


def call(module, *argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = module.main([str(a) for a in argv])
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


@requires_ml
class OperatorCase(unittest.TestCase):
    epochs = 1

    @classmethod
    def setUpClass(cls):
        """One trained run shared by the read-only tests (training is the slow part)."""
        cls.holder = Path(tempfile.mkdtemp(prefix="devsearch-ops-")).resolve()
        cls.addClassCleanup(shutil.rmtree, cls.holder, ignore_errors=True)
        cls.root = cls.holder / "tree"
        base = copy_rehearsal(unittest.TestCase(), topics=36)
        shutil.copytree(base, cls.root, symlinks=True)
        cls.config = cls.root / "train.toml"
        cls.config.write_text(TRAIN_TOML.format(epochs=cls.epochs))
        cls.model_config = cls.root / "predict.toml"
        cls.model_config.write_text(MODEL_TOML)
        cls.run_dir = cls.root / "runtime" / "run" / "first"
        cls.run_dir.mkdir()
        cls.trained = call(train, "--config", cls.config, "--run-dir", cls.run_dir, "--runtime-root", cls.root)
        cls.prepared = cls.root / "bundle" / "prepared"

    def setUp(self):
        self.out_dir = Path(tempfile.mkdtemp(dir=self.root / "runtime"))
        self.addCleanup(shutil.rmtree, self.out_dir, ignore_errors=True)

    def checkpoint(self, name="best.json"):
        return self.run_dir / "checkpoints" / name

    def pairs_file(self, count=3, overrides=None, name="pairs.jsonl"):
        inputs = load_training_inputs(self.prepared / "prepared-data-manifest.json", self.root)
        query = inputs.validation_queries[0].text
        rows = [{"pair_id": f"p{i}", "query": query, "passage": chunk.text} for i, chunk in enumerate(inputs.loaded.corpus.chunks[:count])]
        for index, change in (overrides or {}).items():
            rows[index] = {**rows[index], **change}
        path = self.out_dir / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path
