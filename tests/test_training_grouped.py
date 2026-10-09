"""Offline CPU checks for the opt-in query-group objectives; no pretrained model downloads."""

import hashlib
import math
import unittest
from dataclasses import replace
from unittest import mock

from helpers import RootTestCase
from mltest import requires_ml, HAVE_ML

if HAVE_ML:
    import torch
    from devsearch.artifacts.canonical import digest_of
    from devsearch.data.groups import owner_of_chunks
    from devsearch.data.pairs import PairRecord
    from devsearch.errors import ConfigError, ManifestError
    from devsearch.modeling.batching import collate
    from devsearch.modeling.policy import encode_pair
    from devsearch.training.config import TrainConfig
    from devsearch.training.data import build_grouped_train_set
    from devsearch.training.grouped import SOURCE_KEYS, grouped_order, query_loss, validate_grouped_data
    from devsearch.training.loop import NonFiniteError, ResumeMismatchError, train
    from devsearch.training.checkpoint import read_pointer
    from mlworld import MLWorld


@requires_ml
class GroupedTrainingTests(RootTestCase):
    def setUp(self):
        super().setUp()
        torch.set_num_threads(1)
        self.w = MLWorld.build_ml(self)
        w = self.w
        self.queries = {q.query_id: q for q in w.queries["train"]}
        self.owners = owner_of_chunks(w.corpus, w.merge, w.assignment)
        # Structural fixtures: the tiny lexical corpus cannot always supply three BM25 hits.
        # These explicit toy records test training semantics, never actual mining quality.
        train_chunks = sorted(c for c, owner in self.owners.items() if owner == "train")
        chunk_groups = {c.chunk_id: w.merge.merged_of[c.evidence_group_id] for c in w.corpus.chunks}
        records = []
        for qrel in w.qrels["train"]:
            chosen = [qrel.chunk_id] + [c for c in train_chunks if chunk_groups[c] != chunk_groups[qrel.chunk_id]][:4]
            for index, chunk_id in enumerate(chosen):
                records.append(PairRecord(f"pair-{qrel.query_id}-{index}", qrel.query_id, chunk_id, int(index == 0),
                                          "reviewed_positive" if index == 0 else "assumed_negative",
                                          None if index == 0 else "random" if index == 4 else "bm25", "toy-sampling"))
        self.records = tuple(records)
        self.sources = {key: hashlib.sha256(key.encode()).hexdigest() for key in SOURCE_KEYS}
        self.data = self.build()

    def build(self, **changes):
        args = dict(pairs=self.records, queries=self.queries, corpus=self.w.corpus, qrels=self.w.qrels["train"],
                    chunk_owners=self.owners, source_sha256=self.sources, merged_of=self.w.merge.merged_of)
        return build_grouped_train_set(**{**args, **changes})

    def config(self, kind="pairwise_logistic_scale_4_over_5", **changes):
        return TrainConfig(**{**dict(recipe_id="grouped-test", mode="full", seed=42, epochs=3,
                                     learning_rate=5e-5, microbatch=5, effective_batch=20, pos_weight=4,
                                     dropout=.1, head_seed=1234, loss_kind=kind), **changes})

    def model(self):
        return self.w.model(dropout=.1, head_seed=1234)

    def run_model(self, name, *, model=None, data=None, config=None, **kwargs):
        directory = self.w.root / name
        directory.mkdir(exist_ok=True)
        data = data or self.data
        model = model or self.model()
        result = train(config=config or self.config(), model=model, tokenizer=self.w.tokenizer, policy=self.w.policy,
                       pairs=data.pairs, grouped_data=data, validation=self.w.validation, device=torch.device("cpu"),
                       run_dir=directory, identities=self.w.identities(), **kwargs)
        return result, model, directory

    def assert_params(self, a, b):
        for name, value in a.state_dict().items():
            self.assertTrue(torch.allclose(value, b.state_dict()[name], atol=1e-7, rtol=0), name)

    def test_old_default_config_hash_uses_original_fields(self):
        from dataclasses import asdict
        cfg = TrainConfig(recipe_id="old", mode="full", seed=42, epochs=3, learning_rate=5e-5)
        previous = asdict(cfg)
        del previous["loss_kind"]
        self.assertEqual(cfg.to_dict(), previous)
        self.assertEqual(cfg.config_sha256, digest_of(previous))

    def test_grouped_config_rejects_unmatched_batch_weight_and_selection(self):
        for change in ({"microbatch": 4}, {"effective_batch": 10}, {"pos_weight": 1}, {"pos_weight": "auto"},
                       {"epochs": 2}, {"mode": "frozen"}, {"diagnostic": True}, {"loss_kind": "unknown"}):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                self.config(**change)

    def test_identity_binds_objective(self):
        self.assertNotEqual(self.config().config_sha256, self.config("group_bce_weight4").config_sha256)

    def test_training_group_counts_and_exposures(self):
        self.assertEqual(len(self.data.groups), 5)
        self.assertEqual(len(self.data.pairs), 25)
        for group in self.data.groups:
            self.assertEqual([x.pair.label for x in group.records], [1, 0, 0, 0, 0])
        self.assertEqual(sorted(grouped_order(self.data, 42, 0)), list(range(5)))
        self.assertNotEqual(grouped_order(self.data, 42, 0), grouped_order(self.data, 43, 0))

    def test_rejects_heldout_queries_and_chunks(self):
        qid = next(iter(self.queries))
        for split in ("validation", "test", "development"):
            with self.subTest(split=split), self.assertRaises(ManifestError):
                self.build(queries={**self.queries, qid: replace(self.queries[qid], split=split)})
            cid = self.records[0].chunk_id
            with self.subTest(owner=split), self.assertRaises(ManifestError):
                self.build(chunk_owners={**self.owners, cid: split})

    def test_rejects_duplicates_invalid_labels_missing_sources_and_mix(self):
        with self.assertRaises(ManifestError):
            self.build(pairs=(*self.records, self.records[0]))
        with self.assertRaises(ManifestError):
            self.build(pairs=(*self.records, replace(self.records[0], label=2)))
        with self.assertRaises(ManifestError):
            self.build(source_sha256={})
        idx = next(i for i, p in enumerate(self.records) if p.negative_method == "random")
        records = list(self.records)
        records[idx] = replace(records[idx], negative_method="bm25")
        with self.assertRaises(ManifestError):
            self.build(pairs=records)

    def test_rejects_negative_in_any_positive_merged_group(self):
        group = self.data.groups[0]
        positive, negative = group.records[:2]
        merged = dict(self.w.merge.merged_of)
        merged[negative.evidence_group_id] = positive.merged_group_id
        with self.assertRaisesRegex(ManifestError, "conflicting provenance"):
            self.build(merged_of=merged)
        with self.assertRaisesRegex(ManifestError, "merged-group identity"):
            self.build(merged_of={})

    def test_autograd_matches_independent_finite_difference(self):
        values = [-1.3, .2, -.4, 2., -2.]
        def sp(x):
            return max(x, 0) + math.log1p(math.exp(-abs(x)))
        for kind in ("group_bce_weight4", "pairwise_logistic_scale_4_over_5"):
            def reference(v):
                return ((4 * sp(-v[0]) + sum(sp(x) for x in v[1:])) if kind == "group_bce_weight4"
                        else sum(sp(x - v[0]) for x in v[1:])) / 5
            logits = torch.tensor(values, dtype=torch.float64, requires_grad=True)
            loss = query_loss(logits, kind)
            loss.backward()
            self.assertAlmostEqual(float(loss.detach()), reference(values), places=12)
            for i in range(5):
                plus, minus = list(values), list(values)
                plus[i] += 1e-6
                minus[i] -= 1e-6
                self.assertAlmostEqual(float(logits.grad[i]), (reference(plus) - reference(minus)) / 2e-6, places=7)

    def test_zero_logit_gradients_match_and_extreme_losses_are_finite(self):
        for kind in ("group_bce_weight4", "pairwise_logistic_scale_4_over_5"):
            logits = torch.zeros(5, dtype=torch.float64, requires_grad=True)
            query_loss(logits, kind).backward()
            self.assertTrue(torch.allclose(logits.grad, torch.tensor([-.4, .1, .1, .1, .1], dtype=torch.float64)))
            self.assertTrue(torch.isfinite(query_loss(torch.tensor([-1000., 1000., 500., -500., 0.]), kind)))
        with self.assertRaises(ManifestError):
            query_loss(torch.zeros(4), "group_bce_weight4")

    def test_matched_forward_calls_order_updates_and_fixed_epoch(self):
        all_results = []
        for kind in ("group_bce_weight4", "pairwise_logistic_scale_4_over_5"):
            model = self.model()
            calls = []
            handle = model.register_forward_pre_hook(lambda m, a, kw: calls.append(kw["input_ids"].shape[0]) if m.training else None,
                                                     with_kwargs=True)
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            result, model, directory = self.run_model(kind, config=self.config(kind), model=model)
            handle.remove()
            self.assertEqual(calls, [5] * 15)
            self.assertEqual(result.global_step, 6)
            self.assertEqual(result.best_epoch, 3)
            self.assertEqual([h["selected_best"] for h in result.history], [False, False, True])
            self.assertEqual(read_pointer(directory, "best")["checkpoint_id"], "epoch-0003")
            self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()))
            all_results.append(result)
        self.assertEqual(all_results[0].orders, all_results[1].orders)
        self.assertEqual(all_results[0].lr_trace, all_results[1].lr_trace)

    def test_short_final_accumulation_matches_independent_training_reference(self):
        config = self.config()
        result, trained, _ = self.run_model("actual", config=config)
        model = self.model()
        model.set_encoder_frozen(False)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay, betas=(config.beta1, config.beta2), eps=config.eps)
        step = 0
        torch.manual_seed(config.seed)
        for epoch in range(3):
            order = sorted(self.data.groups, key=lambda g: (hashlib.sha256(f"group-order-v1|42|{epoch}|{g.query_id}".encode()).hexdigest(), g.query_id))
            model.train()
            for start in range(0, 5, 4):
                queries = order[start:start + 4]
                optimizer.zero_grad(set_to_none=True)
                for group in queries:
                    inputs = [encode_pair(self.w.tokenizer, self.w.policy, x.pair.query_text, x.pair.passage_text) for x in group.records]
                    logits = model(**collate(inputs, self.w.tokenizer.pad_id))
                    # Independently assemble the objective and accumulation, without query_loss.
                    # Use the same elementary softplus kernel: logaddexp has roundoff differences
                    # that Adam amplifies on shift-invariant pairwise head-bias gradients (~0).
                    # The separate float64 finite-difference test checks the loss algebra.
                    independent = torch.nn.functional.softplus(logits[1:] - logits[0]).sum() / 5
                    (independent / len(queries)).backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), config.max_grad_norm)
                lr = config.learning_rate * (1. if step == 0 else (6 - step) / 5)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
                step += 1
        self.assertEqual(step, result.global_step)
        self.assert_params(trained, model)

    def test_epoch_resume_equivalent_and_changed_input_refused(self):
        _, continuous, _ = self.run_model("continuous")
        stopped, _, directory = self.run_model("resumed", epoch_gate=lambda epoch, history: "test boundary stop" if epoch == 1 else None)
        self.assertEqual(stopped.completed_epochs, 1)
        self.assertIsNone(read_pointer(directory, "best"))
        resumed, model, _ = self.run_model("resumed", resume=True)
        self.assertEqual(resumed.global_step, 6)
        self.assert_params(continuous, model)
        changed = replace(self.data, source_sha256=tuple(sorted({**dict(self.data.source_sha256), "assignment": "a" * 64}.items())))
        with self.assertRaises(ResumeMismatchError):
            self.run_model("resumed", resume=True, data=changed)
        with self.assertRaises(ResumeMismatchError):
            self.run_model("resumed", resume=True, config=self.config("group_bce_weight4"))

    def test_missing_groups_and_nonfinite_logits_fail(self):
        directory = self.w.root / "missing"
        directory.mkdir()
        with self.assertRaises(ManifestError):
            train(config=self.config(), model=self.model(), tokenizer=self.w.tokenizer, policy=self.w.policy,
                  pairs=self.data.pairs, validation=self.w.validation, device=torch.device("cpu"), run_dir=directory,
                  identities=self.w.identities())
        with mock.patch("devsearch.training.grouped.query_loss", return_value=torch.tensor(float("nan"))):
            with self.assertRaises(NonFiniteError):
                self.run_model("nonfinite")
