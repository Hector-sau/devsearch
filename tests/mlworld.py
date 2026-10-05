"""A tiny but real ML world for CPU tests: local diagnostic tokenizer, tiny BERT, train pairs and validation set."""

import re
import shutil
import tempfile
from pathlib import Path

from devsearch.data.candidates import build_candidates
from devsearch.data.pairs import MiningConfig, build_mining_pool, mine_pairs, sampling_identity
from devsearch.modeling.cross_encoder import CrossEncoder, build_tiny_encoder
from devsearch.modeling.policy import PairEncodingPolicy
from devsearch.modeling.tokenization import WordPieceTokenizer, write_diagnostic_tokenizer
from devsearch.training.config import TrainConfig
from devsearch.training.data import build_train_pairs, build_validation_set

from dataset import World

POLICY = PairEncodingPolicy(max_pair_tokens=256, max_query_tokens=64)


class MLWorld(World):
    """`World` plus a tokenizer, train pairs, a validation set and a model factory."""

    @classmethod
    def build_ml(cls, testcase) -> "MLWorld":
        world = cls.build(testcase)
        world.__class__ = cls
        words = set()
        for chunk in world.corpus.chunks:
            words.update(re.findall(r"[A-Za-z0-9]+", chunk.text.lower()))
        for split in world.queries.values():
            for query in split:
                words.update(re.findall(r"[A-Za-z0-9]+", query.text.lower()))
        directory = Path(tempfile.mkdtemp(prefix="devsearch-tok-")).resolve()
        testcase.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        world.tokenizer_dir = write_diagnostic_tokenizer(directory, sorted(words))
        world.tokenizer = WordPieceTokenizer(world.tokenizer_dir)
        world.policy = POLICY
        pool = build_mining_pool(world.corpus, world.merge, world.assignment)
        config = MiningConfig(hard_negatives=2, random_negatives=2, seed=42)
        sampling = sampling_identity(config, pool, corpus_id="c", index_id="i", queries_sha256="q", qrels_sha256="r")
        world.pairset = mine_pairs(corpus=world.corpus, retriever=world.loaded.retriever(), merge=world.merge, pool=pool,
                                   queries=world.queries["train"], qrels=world.qrels["train"], config=config, sampling_id=sampling)
        by_id = {q.query_id: q for q in world.queries["train"]}
        world.train_pairs = build_train_pairs(world.pairset.pairs, by_id, world.corpus)
        validation_queries = {q.query_id: q for q in world.queries["validation"]}
        candidates = build_candidates(world.loaded, world.queries["validation"], 50)
        world.validation_candidates = candidates
        world.validation = build_validation_set(candidates, validation_queries, world.qrels["validation"], world.corpus)
        return world

    def model(self, *, dropout=0.0, encoder_seed=0, head_seed=1, hidden=32) -> CrossEncoder:
        encoder = build_tiny_encoder(self.tokenizer.vocab_size, hidden_size=hidden, seed=encoder_seed, hidden_dropout=dropout, attention_dropout=dropout)
        return CrossEncoder(encoder, dropout=dropout, head_seed=head_seed)

    def config(self, **over) -> TrainConfig:
        values = dict(recipe_id="test", mode="full", seed=7, epochs=2, learning_rate=1e-3, microbatch=4, effective_batch=8, dropout=0.0)
        values.update(over)
        return TrainConfig(**values)

    def identities(self):
        return {"pairs_manifest": "p" * 8, "tokenizer": self.tokenizer.fingerprint, "model": "tiny-diagnostic"}
