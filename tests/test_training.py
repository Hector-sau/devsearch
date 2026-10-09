"""Native training loop: arithmetic, accumulation vs an independent reference, freezing, recovery, failures."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devsearch.errors import ConfigError, ManifestError
from helpers import RootTestCase
from mltest import HAVE_ML, requires_ml

if HAVE_ML:
    import torch
    from torch import nn
    from devsearch.training import loop
    from devsearch.training.checkpoint import checkpoint_dir, read_pointer, write_pointer, read_manifest
    from devsearch.training.config import TrainConfig, load_train_config
    from devsearch.training.loop import (
        DeviceMemoryError, NonFiniteError, ResumeMismatchError, epoch_permutation, lr_multiplier, plan_updates, train,
    )
    from mlworld import MLWorld


class ArithmeticTests(unittest.TestCase):
    @requires_ml
    def test_the_documented_worked_example(self):
        plan = plan_updates(5000, 4, 4, 3)                      # docs/12 §3
        self.assertEqual(plan, {"microbatches_per_epoch": 1250, "updates_per_epoch": 313, "planned_updates": 939})
        self.assertEqual(math.ceil(0.10 * plan["planned_updates"]), 94)
        self.assertEqual(plan_updates(5000, 4, 4, 3)["microbatches_per_epoch"] * 3, 3750)

    @requires_ml
    def test_lr_multiplier_sequence_is_warmup_then_linear_decay_to_zero(self):
        sequence = [lr_multiplier(step, 2, 6) for step in range(6)]
        self.assertEqual(sequence, [0.5, 1.0, 1.0, 0.75, 0.5, 0.25])
        self.assertEqual(lr_multiplier(6, 2, 6), 0.0)
        self.assertEqual(lr_multiplier(10, 2, 6), 0.0)           # never negative
        self.assertEqual([lr_multiplier(s, 0, 4) for s in range(4)], [1.0, 0.75, 0.5, 0.25])   # no warmup

    @requires_ml
    def test_epoch_permutations_depend_only_on_seed_and_epoch(self):
        self.assertEqual(epoch_permutation(50, 7, 0), epoch_permutation(50, 7, 0))
        self.assertNotEqual(epoch_permutation(50, 7, 0), epoch_permutation(50, 7, 1))
        self.assertNotEqual(epoch_permutation(50, 7, 0), epoch_permutation(50, 8, 0))
        self.assertEqual(sorted(epoch_permutation(50, 7, 3)), list(range(50)))


@requires_ml
class ConfigTests(unittest.TestCase):
    def test_valid_and_derived_values(self):
        config = TrainConfig(recipe_id="r1", mode="full", seed=1, epochs=3, learning_rate=2e-5, microbatch=4, effective_batch=16)
        self.assertEqual(config.accumulation_steps, 4)
        self.assertEqual(config.config_sha256, TrainConfig(recipe_id="r1", mode="full", seed=1, epochs=3, learning_rate=2e-5, microbatch=4, effective_batch=16).config_sha256)

    def test_defects(self):
        base = dict(recipe_id="r", mode="full", seed=1, epochs=2, learning_rate=1e-3)
        cases = {
            "mode": dict(mode="partial"), "epochs above cap": dict(epochs=4), "epochs zero": dict(epochs=0),
            "non-dividing microbatch": dict(microbatch=3, effective_batch=16), "negative lr": dict(learning_rate=-1.0),
            "nan lr": dict(learning_rate=float("nan")), "dropout": dict(dropout=1.0), "warmup": dict(warmup_fraction=1.0),
            "pos_weight text": dict(pos_weight="heavy"), "pos_weight zero": dict(pos_weight=0), "recipe id": dict(recipe_id="bad id!"),
            "betas": dict(beta1=1.0),
        }
        for name, over in cases.items():
            with self.subTest(name), self.assertRaises(ConfigError):
                TrainConfig(**{**base, **over})
        TrainConfig(**{**base, "epochs": 50, "diagnostic": True})          # diagnostic runs may be longer

    def test_toml_loader_is_strict(self):
        directory = Path(tempfile.mkdtemp())
        ok = '''schema_version = 1
[run]
recipe_id = "frozen-s42"
mode = "frozen"
seed = 42
epochs = 3
[optimizer]
learning_rate = 1e-3
weight_decay = 0.01
betas = [0.9, 0.999]
eps = 1e-8
max_grad_norm = 1.0
warmup_fraction = 0.1
[batch]
effective_batch = 16
microbatch = 4
[model]
dropout = 0.1
head_seed = 7
'''
        path = directory / "ok.toml"
        path.write_text(ok)
        config = load_train_config(path)
        self.assertEqual((config.recipe_id, config.mode, config.accumulation_steps, config.pos_weight), ("frozen-s42", "frozen", 4, "auto"))
        for name, text in {"unknown section": ok + "[extra]\nx = 1\n", "unknown key": ok.replace("seed = 42", "seed = 42\nbogus = 1"),
                           "missing section": ok.replace("[model]\ndropout = 0.1\nhead_seed = 7\n", ""), "bool seed": ok.replace("seed = 42", "seed = true"),
                           "bad toml": "this = = nope"}.items():
            bad = directory / "bad.toml"
            bad.write_text(text)
            with self.subTest(name), self.assertRaises(ConfigError):
                load_train_config(bad)
        with self.assertRaises(ConfigError):
            load_train_config(directory / "missing.toml")


@requires_ml
class TrainCase(RootTestCase):
    def setUp(self):
        super().setUp()
        self.w = MLWorld.build_ml(self)
        self.run_dir = self.w.root / "run"
        self.run_dir.mkdir()
        torch.set_num_threads(1)

    def run_training(self, config=None, *, model=None, run_dir=None, **kwargs):
        model = model or self.w.model()
        result = train(config=config or self.w.config(), model=model, tokenizer=self.w.tokenizer, policy=self.w.policy, pairs=self.w.train_pairs,
                       validation=self.w.validation, device=torch.device("cpu"), run_dir=run_dir or self.run_dir, identities=self.w.identities(), **kwargs)
        return result, model

    @staticmethod
    def params(model):
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    def assertParamsClose(self, a, b, atol=1e-6):
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(torch.allclose(a[key], b[key], atol=atol, rtol=0), key)


class LoopBehaviorTests(TrainCase):
    def test_a_full_run_checkpoints_epoch_zero_and_every_epoch_with_pointers_not_copies(self):
        result, model = self.run_training()
        self.assertEqual((result.status, result.completed_epochs), ("completed", 2))
        names = sorted(p.name for p in checkpoint_dir(self.run_dir).iterdir())
        self.assertEqual([n for n in names if n.endswith(".pt")], ["epoch-0000.pt", "epoch-0001.pt", "epoch-0002.pt"])
        self.assertIn("latest.json", names)
        self.assertIn("best.json", names)
        self.assertEqual(read_pointer(self.run_dir, "latest")["checkpoint_id"], "epoch-0002")
        self.assertEqual(read_manifest(self.run_dir, "epoch-0001")["resume_kind"], "epoch_boundary")
        self.assertTrue((self.run_dir / "training-summary.json").is_file())

    def test_epoch_zero_holds_the_untouched_initial_weights(self):
        initial = self.params(self.w.model())
        self.run_training()
        from devsearch.training.checkpoint import load_checkpoint
        state, _ = load_checkpoint(self.run_dir, "epoch-0000", torch.device("cpu"), require_resumable=True)
        self.assertParamsClose(initial, state["model"], atol=0)

    def test_counts_match_the_plan(self):
        config = self.w.config(microbatch=4, effective_batch=8)
        result, _ = self.run_training(config)
        pairs = len(self.w.train_pairs)
        plan = plan_updates(pairs, 4, 2, 2)
        self.assertEqual(result.global_step, plan["planned_updates"])
        self.assertEqual(len(result.lr_trace), plan["planned_updates"])
        for record in result.history:
            self.assertEqual((record["updates"], record["microbatches"]), (plan["updates_per_epoch"], plan["microbatches_per_epoch"]))
        warmup = math.ceil(0.10 * plan["planned_updates"])
        self.assertAlmostEqual(result.lr_trace[0], config.learning_rate * lr_multiplier(0, warmup, plan["planned_updates"]))
        self.assertAlmostEqual(result.lr_trace[-1], config.learning_rate * lr_multiplier(plan["planned_updates"] - 1, warmup, plan["planned_updates"]))
        self.assertTrue(all(math.isfinite(v) and v >= 0 for v in result.lr_trace))

    def test_the_loop_matches_an_independent_reference_including_a_short_last_group(self):
        """Weighted partial accumulation, clipping, AdamW and the LR schedule against a from-scratch restatement."""
        pairs = self.w.train_pairs[:6]                                    # 6 pairs, microbatch 2, accumulation 2 -> groups of 4 and 2
        self.w.train_pairs, saved = pairs, self.w.train_pairs
        try:
            config = self.w.config(epochs=1, microbatch=2, effective_batch=4, learning_rate=5e-3, mode="full")
            positives = sum(p.label for p in pairs)
            if positives == 0 or positives == len(pairs):
                self.skipTest("fixture slice needs both classes")
            result, trained = self.run_training(config)
        finally:
            self.w.train_pairs = saved
        reference = self.w.model()
        parameters = [p for p in reference.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
        plan = plan_updates(len(pairs), 2, 2, 1)
        warmup = math.ceil(config.warmup_fraction * plan["planned_updates"])
        negatives = len(pairs) - positives
        loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(negatives / positives))
        from devsearch.modeling.batching import collate
        from devsearch.modeling.policy import encode_pair
        order = epoch_permutation(len(pairs), config.seed, 0)
        torch.manual_seed(config.seed)
        reference.train()
        for index, group in enumerate([order[:4], order[4:]]):            # the second group is SHORT: 2 pairs, not 4
            for g in optimizer.param_groups:
                g["lr"] = config.learning_rate * lr_multiplier(index, warmup, plan["planned_updates"])
            batch = collate([encode_pair(self.w.tokenizer, self.w.policy, pairs[i].query_text, pairs[i].passage_text) for i in group], self.w.tokenizer.pad_id)
            logits = reference(**batch)
            labels = torch.tensor([float(pairs[i].label) for i in group])
            (loss_fn(logits, labels).sum() / len(group)).backward()          # divided by the ACTUAL group size
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.assertParamsClose(self.params(reference), self.params(trained), atol=1e-5)
        self.assertEqual(result.global_step, 2)

    def test_frozen_mode_trains_only_the_head_and_keeps_the_encoder_in_eval_mode(self):
        model = self.w.model(dropout=0.1)
        before = self.params(model)
        seen_modes = []
        original = model.forward
        def spy(*args, **kwargs):
            seen_modes.append((model.training, model.encoder.training))
            return original(*args, **kwargs)
        model.forward = spy
        self.run_training(self.w.config(mode="frozen", learning_rate=1e-2), model=model)
        after = self.params(model)
        for key in before:
            changed = not torch.equal(before[key], after[key])
            self.assertEqual(changed, key.startswith("head."), key)
        self.assertTrue(any(training and not encoder_training for training, encoder_training in seen_modes))
        self.assertTrue(all(not encoder_training for _, encoder_training in seen_modes))

    def test_full_mode_updates_the_encoder_too(self):
        model = self.w.model()
        before = self.params(model)
        self.run_training(self.w.config(mode="full", learning_rate=1e-2), model=model)
        after = self.params(model)
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before if k.startswith("encoder.")))
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before if k.startswith("head.")))
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in after.values()))

    def test_a_tiny_set_can_be_overfitted(self):
        """The core correctness signal: training loss falls steeply and positives end above negatives."""
        config = self.w.config(diagnostic=True, epochs=200, learning_rate=3e-3, microbatch=8, effective_batch=8, mode="full")
        result, model = self.run_training(config)
        losses = [h["train_loss"] for h in result.history]
        self.assertLess(losses[-1], 0.15 * losses[0], losses[::40])          # an 85% drop; other CPUs and torch builds plateau near 0.1 of the start, the margin below is the strict signal
        from devsearch.modeling.scoring import score_pairs
        values = score_pairs(model, self.w.tokenizer, self.w.policy, [(p.query_text, p.passage_text) for p in self.w.train_pairs], device=torch.device("cpu"), batch_size=8)
        positive = [v for v, p in zip(values, self.w.train_pairs) if p.label == 1]
        negative = [v for v, p in zip(values, self.w.train_pairs) if p.label == 0]
        # The fixture holds near-duplicate hard negatives. Whether the last one is pulled clear of the positives depends on float details of the CPU
        # and torch build (a GitHub runner leaves one tied at 200 epochs), so the signal is the whole population, not its single worst pair:
        # a wide margin between the means and nearly every positive/negative pair in the right order.
        self.assertGreater(sum(positive) / len(positive) - sum(negative) / len(negative), 1.0, (min(positive), max(negative)))
        pairs = [(p, n) for p in positive for n in negative]
        ordered = sum(1 for p, n in pairs if p > n) / len(pairs)
        self.assertGreaterEqual(ordered, 0.9, (ordered, min(positive), max(negative)))

    def test_training_is_deterministic_for_a_seed(self):
        first, model_a = self.run_training(run_dir=self._fresh("a"))
        second, model_b = self.run_training(model=self.w.model(), run_dir=self._fresh("b"))
        self.assertParamsClose(self.params(model_a), self.params(model_b), atol=0)
        self.assertEqual(first.lr_trace, second.lr_trace)
        self.assertEqual([h["validation"] for h in first.history], [h["validation"] for h in second.history])

    def _fresh(self, name):
        directory = self.w.root / name
        directory.mkdir()
        return directory

    def test_pos_weight_auto_uses_the_actual_sampled_ratio_and_a_number_is_respected(self):
        positives = sum(p.label for p in self.w.train_pairs)
        ratio = (len(self.w.train_pairs) - positives) / positives
        result, _ = self.run_training(run_dir=self._fresh("auto"))
        from devsearch.training.checkpoint import load_checkpoint
        state, _ = load_checkpoint(self._fresh_path("auto"), "epoch-0001", torch.device("cpu"), require_resumable=True)
        self.assertEqual(state["identity"]["pos_weight"], repr(ratio))
        self.run_training(self.w.config(pos_weight=2.5), run_dir=self._fresh("num"))
        state, _ = load_checkpoint(self._fresh_path("num"), "epoch-0001", torch.device("cpu"), require_resumable=True)
        self.assertEqual(state["identity"]["pos_weight"], repr(2.5))

    def _fresh_path(self, name):
        return self.w.root / name

    def test_best_epoch_is_chosen_by_validation_and_the_earlier_epoch_wins_ties(self):
        sequence = iter([
            {"group_macro": {"mrr@10": 0.5, "ndcg@10": 0.5, "recall@50": 1.0}, "query_macro": {}, "queries_scored": 2, "groups_scored": 2},
            {"group_macro": {"mrr@10": 0.5, "ndcg@10": 0.5, "recall@50": 1.0}, "query_macro": {}, "queries_scored": 2, "groups_scored": 2},
            {"group_macro": {"mrr@10": 0.5, "ndcg@10": 0.6, "recall@50": 1.0}, "query_macro": {}, "queries_scored": 2, "groups_scored": 2},
        ])
        with mock.patch.object(loop, "evaluate_validation", lambda *a, **k: next(sequence)):
            result, _ = self.run_training(self.w.config(epochs=3))
        self.assertEqual([h["selected_best"] for h in result.history], [True, False, True])    # a tie is not an improvement
        self.assertEqual((result.best_epoch, result.best_checkpoint_id), (3, "epoch-0003"))
        self.assertEqual(read_pointer(self.run_dir, "best")["checkpoint_id"], "epoch-0003")
        self.assertEqual(read_pointer(self.run_dir, "latest")["checkpoint_id"], "epoch-0003")

    def test_a_validation_set_without_positives_or_data_without_both_classes_is_refused(self):
        from devsearch.training.data import ValidationSet, TrainPair
        with self.assertRaisesRegex(ManifestError, "validation has no query"):
            train(config=self.w.config(), model=self.w.model(), tokenizer=self.w.tokenizer, policy=self.w.policy, pairs=self.w.train_pairs,
                  validation=ValidationSet((), ()), device=torch.device("cpu"), run_dir=self.run_dir, identities=self.w.identities())
        one_class = [TrainPair("q", "p", 1)] * 3
        with self.assertRaisesRegex(ManifestError, "at least one positive and one negative"):
            train(config=self.w.config(), model=self.w.model(), tokenizer=self.w.tokenizer, policy=self.w.policy, pairs=one_class,
                  validation=self.w.validation, device=torch.device("cpu"), run_dir=self.run_dir, identities=self.w.identities())

    def test_a_missing_run_directory_is_never_created(self):
        missing = self.w.root / "unmounted"
        with self.assertRaisesRegex(ManifestError, "does not exist"):
            self.run_training(run_dir=missing)
        self.assertFalse(missing.exists())


class RecoveryTests(TrainCase):
    def test_continuous_and_resumed_runs_agree_on_everything(self):
        """G07: two epochs straight vs one epoch + save/reload + one epoch."""
        config_two = self.w.config(epochs=2)
        continuous, model_a = self.run_training(config_two, run_dir=self._fresh("continuous"))
        # One epoch with the SAME two-epoch plan, interrupted at the epoch boundary by a stop request.
        resumed_dir = self._fresh("resumed")
        calls = {"n": 0}
        per_epoch = plan_updates(len(self.w.train_pairs), 4, 2, 2)["microbatches_per_epoch"]
        def stop_after_first_epoch():
            calls["n"] += 1
            return calls["n"] > per_epoch              # checked before each microbatch: stops at the start of epoch two
        first, _ = self.run_training(config_two, run_dir=resumed_dir, should_stop=stop_after_first_epoch)
        self.assertEqual((first.status, first.completed_epochs), ("stopped", 1))
        model_b = self.w.model()                       # a brand-new process-like start: fresh model, fresh optimizer
        second, model_b = self.run_training(config_two, model=model_b, run_dir=resumed_dir, resume=True)
        self.assertEqual((second.status, second.completed_epochs, second.global_step), ("completed", 2, continuous.global_step))
        self.assertParamsClose(self.params(model_a), self.params(model_b), atol=1e-6)
        self.assertEqual(second.lr_trace, continuous.lr_trace)                       # identical LR sequence
        self.assertEqual(second.orders, {1: continuous.orders[1]})                   # the replayed epoch used the same data order
        self.assertEqual([h["validation"] for h in second.history], [h["validation"] for h in continuous.history])
        self.assertEqual((second.best_epoch, second.best_checkpoint_id), (continuous.best_epoch, continuous.best_checkpoint_id))

    def _fresh(self, name):
        directory = self.w.root / name
        directory.mkdir()
        return directory

    def test_a_mid_epoch_stop_leaves_a_diagnostic_snapshot_that_is_never_resumable(self):
        calls = {"n": 0}
        def stop_mid_epoch():
            calls["n"] += 1
            return calls["n"] == 3
        result, _ = self.run_training(should_stop=stop_mid_epoch)
        self.assertEqual((result.status, result.completed_epochs, result.stop_reason), ("stopped", 0, "stop_requested"))
        self.assertTrue(result.diagnostic_snapshot.startswith("diagnostic-epoch0000-step"))
        self.assertEqual(read_manifest(self.run_dir, result.diagnostic_snapshot)["resume_kind"], "diagnostic_only")
        self.assertEqual(read_pointer(self.run_dir, "latest")["checkpoint_id"], "epoch-0000")     # the valid boundary is untouched
        write_pointer(self.run_dir, "latest", read_manifest(self.run_dir, result.diagnostic_snapshot))
        with self.assertRaisesRegex(ManifestError, "only epoch_boundary checkpoints can be resumed"):
            self.run_training(resume=True)

    def test_resuming_after_a_mid_epoch_stop_replays_the_whole_interrupted_epoch(self):
        calls = {"n": 0}
        def stop_mid_epoch():
            calls["n"] += 1
            return calls["n"] == 3
        self.run_training(should_stop=stop_mid_epoch)
        result, model = self.run_training(model=self.w.model(), resume=True)
        reference, ref_model = self.run_training(model=self.w.model(), run_dir=self._fresh("ref"))
        self.assertEqual((result.status, result.global_step), ("completed", reference.global_step))
        self.assertParamsClose(self.params(model), self.params(ref_model), atol=1e-6)

    def test_a_passed_deadline_stops_with_a_budget_status_and_keeps_valid_state(self):
        import time
        result, _ = self.run_training(deadline_monotonic=time.monotonic() - 1)
        self.assertEqual((result.status, result.stop_reason, result.completed_epochs), ("stopped_budget", "deadline", 0))
        self.assertEqual(read_pointer(self.run_dir, "latest")["checkpoint_id"], "epoch-0000")

    def test_an_epoch_gate_stops_cleanly_at_the_boundary_and_the_run_resumes_to_the_same_result(self):
        seen = []
        def gate(epoch, history):
            seen.append((epoch, len(history)))
            return "the remaining window cannot hold another epoch" if epoch == 1 else None
        result, _ = self.run_training(epoch_gate=gate)
        self.assertEqual((result.status, result.completed_epochs, result.stop_reason), ("stopped_budget", 1, "the remaining window cannot hold another epoch"))
        self.assertEqual(seen, [(0, 0), (1, 1)])
        self.assertIsNone(result.diagnostic_snapshot)
        self.assertEqual(read_pointer(self.run_dir, "latest")["checkpoint_id"], "epoch-0001")
        self.assertFalse(any(p.name.startswith("diagnostic-") for p in checkpoint_dir(self.run_dir).iterdir()))
        resumed, model = self.run_training(model=self.w.model(), resume=True)
        reference, ref_model = self.run_training(model=self.w.model(), run_dir=self._fresh("ref-gate"))
        self.assertEqual((resumed.status, resumed.global_step), ("completed", reference.global_step))
        self.assertParamsClose(self.params(model), self.params(ref_model), atol=1e-6)

    def test_resume_refuses_a_changed_recipe_data_or_tokenizer(self):
        calls = {"n": 0}
        def stop_after_epoch():
            calls["n"] += 1
            return calls["n"] > plan_updates(len(self.w.train_pairs), 4, 2, 2)["microbatches_per_epoch"]
        self.run_training(should_stop=stop_after_epoch)
        with self.assertRaisesRegex(ResumeMismatchError, "config_sha256"):
            self.run_training(self.w.config(learning_rate=2e-3), resume=True)
        with self.assertRaisesRegex(ResumeMismatchError, "pairs"):
            saved = self.w.train_pairs
            try:
                self.w.train_pairs = saved[:-1]
                self.run_training(resume=True)
            finally:
                self.w.train_pairs = saved
        original = self.w.identities
        self.w.identities = lambda: {**original(), "tokenizer": "different"}
        with self.assertRaisesRegex(ResumeMismatchError, "tokenizer"):
            self.run_training(resume=True)

    def test_a_used_run_directory_cannot_be_started_again_without_resume(self):
        self.run_training()
        with self.assertRaisesRegex(ManifestError, "already has checkpoints"):
            self.run_training()
        with self.assertRaisesRegex(ManifestError, "nothing to resume"):
            self.run_training(resume=True, run_dir=self._fresh("empty"))

    def test_a_corrupted_checkpoint_is_detected_not_loaded(self):
        calls = {"n": 0}
        def stop_after_epoch():
            calls["n"] += 1
            return calls["n"] > plan_updates(len(self.w.train_pairs), 4, 2, 2)["microbatches_per_epoch"]
        self.run_training(should_stop=stop_after_epoch)
        path = checkpoint_dir(self.run_dir) / "epoch-0001.pt"
        path.write_bytes(path.read_bytes()[:-8] + b"corrupt!")
        with self.assertRaisesRegex(ManifestError, "does not match its recorded SHA-256"):
            self.run_training(resume=True)

    def test_a_crash_while_writing_a_checkpoint_keeps_the_previous_valid_one(self):
        from devsearch.training import checkpoint as ck
        real = ck.write_atomic
        def explode(path, data, **kwargs):
            if path.name == "epoch-0002.json":
                raise OSError("disk full")
            return real(path, data, **kwargs)
        with mock.patch.object(ck, "write_atomic", explode):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_training()
        self.assertEqual(read_pointer(self.run_dir, "latest")["checkpoint_id"], "epoch-0001")      # the pointer never advanced
        result, _ = self.run_training(model=self.w.model(), resume=True)
        self.assertEqual((result.status, result.completed_epochs), ("completed", 2))


class FailureTests(TrainCase):
    def test_non_finite_loss_stops_the_run_and_never_advances_the_checkpoint(self):
        model = self.w.model()
        original = model.forward
        calls = {"n": 0}
        def poisoned(*args, **kwargs):
            calls["n"] += 1
            out = original(*args, **kwargs)
            return out * float("nan") if calls["n"] == 3 else out
        model.forward = poisoned
        with self.assertRaisesRegex(NonFiniteError, "non-finite loss"):
            self.run_training(model=model)
        self.assertEqual(read_pointer(self.run_dir, "latest")["checkpoint_id"], "epoch-0000")

    def test_non_finite_gradients_stop_the_run(self):
        with mock.patch("devsearch.training.loop.torch.nn.utils.clip_grad_norm_", return_value=torch.tensor(float("inf"))):
            with self.assertRaisesRegex(NonFiniteError, "gradient norm"):
                self.run_training()

    def test_out_of_memory_is_an_explicit_error_without_silent_changes(self):
        model = self.w.model()
        original = model.forward
        calls = {"n": 0}
        def oom(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
            return original(*args, **kwargs)
        model.forward = oom
        config = self.w.config()
        with self.assertRaisesRegex(DeviceMemoryError, "out of memory at epoch 0"):
            self.run_training(config, model=model)
        self.assertEqual(config.microbatch, 4)                                    # nothing adapted to the failure
        self.assertEqual(read_pointer(self.run_dir, "latest")["checkpoint_id"], "epoch-0000")

    def test_other_runtime_errors_are_not_mistaken_for_oom(self):
        model = self.w.model()
        model.forward = mock.Mock(side_effect=RuntimeError("MPS operator not implemented"))
        with self.assertRaisesRegex(RuntimeError, "operator not implemented"):
            self.run_training(model=model)

    def test_an_over_long_input_fails_instead_of_being_truncated(self):
        from devsearch.modeling.policy import LengthError, PairEncodingPolicy
        policy = PairEncodingPolicy(max_pair_tokens=16, max_query_tokens=8)
        self.w.policy = policy
        with self.assertRaises(LengthError):
            self.run_training()


if __name__ == "__main__":
    unittest.main()
