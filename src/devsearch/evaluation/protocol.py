"""The frozen experiment protocol (docs/10 §6, docs/12 §1): recipes, seeds, selection gates, latency, bootstrap.

Everything a later stage may be tempted to adjust after seeing results is fixed here, before training,
and its hash goes into the execution plan. A formal protocol keeps the documented bounded matrix: at most
three epochs, seeds that include the deployment seed, and a deployment seed that is not chosen by score.

`protocol.toml` sections (all keys required unless noted; unknown keys fail):

    schema_version = 1
    [length]      max_pair_tokens = 256   max_query_tokens = 64
    [matrix]      seeds = [42]   deployment_seed = 42   epochs = 3   head_seed = 1234
    [[recipe]]    id = "frozen"   mode = "frozen"   learning_rate = 1e-3
    [[recipe]]    id = "full"     mode = "full"     learning_rate = 2e-5
    [training]    dropout weight_decay beta1 beta2 eps max_grad_norm warmup_fraction effective_batch
                  pos_weight ("auto" or a number)   validation_batch
    [selection]   min_group_mrr_gain = 0.02   max_p95_ms = 2000
    [latency]     warmup = 10   passes = 3   queries = 50   seed = 20261005
    [bootstrap]   confidence = 0.95   repetitions = 10000   seed = 20261005
    [inference]   batch_candidates = [4, 2, 1, 8, 16]
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import digest_of, sha256_file
from devsearch.artifacts.tomlconfig import (
    as_choice, as_int, as_int_list, as_number, as_string, as_table, check_keys, check_version, load_toml,
)
from devsearch.errors import ConfigError

MAX_EPOCHS = 3
DEFAULT_INFERENCE_BATCHES = (4, 2, 1, 8, 16)


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    mode: str
    learning_rate: float


@dataclass(frozen=True)
class Protocol:
    max_pair_tokens: int
    max_query_tokens: int
    seeds: tuple[int, ...]
    deployment_seed: int
    epochs: int
    head_seed: int
    recipes: tuple[Recipe, ...]
    training: tuple[tuple[str, Any], ...]
    min_group_mrr_gain: float
    max_p95_ms: float
    latency_warmup: int
    latency_passes: int
    latency_queries: int
    latency_seed: int
    bootstrap_confidence: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    inference_batches: tuple[int, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "length": {"max_pair_tokens": self.max_pair_tokens, "max_query_tokens": self.max_query_tokens},
            "matrix": {"seeds": list(self.seeds), "deployment_seed": self.deployment_seed, "epochs": self.epochs, "head_seed": self.head_seed},
            "recipes": [{"id": r.recipe_id, "mode": r.mode, "learning_rate": r.learning_rate} for r in self.recipes],
            "training": dict(self.training),
            "selection": {"min_group_mrr_gain": self.min_group_mrr_gain, "max_p95_ms": self.max_p95_ms},
            "latency": {"warmup": self.latency_warmup, "passes": self.latency_passes, "queries": self.latency_queries, "seed": self.latency_seed},
            "bootstrap": {"confidence": self.bootstrap_confidence, "repetitions": self.bootstrap_repetitions, "seed": self.bootstrap_seed},
            "inference": {"batch_candidates": list(self.inference_batches)},
        }

    @property
    def protocol_id(self) -> str:
        return "protocol-" + digest_of(self.to_record())[:32]

    def run_ids(self) -> list[tuple[str, int]]:
        """Every required (recipe, seed) run, in a stable order."""
        return [(recipe.recipe_id, seed) for recipe in self.recipes for seed in self.seeds]


def _training(table: Any, where: str) -> tuple[tuple[str, Any], ...]:
    table = as_table(table, where)
    check_keys(table, ("dropout", "weight_decay", "beta1", "beta2", "eps", "max_grad_norm", "warmup_fraction", "effective_batch",
                       "pos_weight", "validation_batch"), (), where)
    pos_weight = table["pos_weight"]
    if pos_weight != "auto":
        pos_weight = as_number(pos_weight, f"{where}.pos_weight", 1e-6, 1e6)
    return tuple(sorted({
        "dropout": as_number(table["dropout"], f"{where}.dropout", 0.0, 0.99),
        "weight_decay": as_number(table["weight_decay"], f"{where}.weight_decay", 0.0, 10.0),
        "beta1": as_number(table["beta1"], f"{where}.beta1", 1e-6, 0.999999),
        "beta2": as_number(table["beta2"], f"{where}.beta2", 1e-6, 0.999999),
        "eps": as_number(table["eps"], f"{where}.eps", 1e-12, 1.0),
        "max_grad_norm": as_number(table["max_grad_norm"], f"{where}.max_grad_norm", 1e-6, 1e6),
        "warmup_fraction": as_number(table["warmup_fraction"], f"{where}.warmup_fraction", 0.0, 0.99),
        "effective_batch": as_int(table["effective_batch"], f"{where}.effective_batch", 1, 4096),
        "pos_weight": pos_weight,
        "validation_batch": as_int(table["validation_batch"], f"{where}.validation_batch", 1, 4096),
    }.items()))


def load_protocol(path: Path) -> Protocol:
    path = Path(path)
    where = str(path)
    table = load_toml(path)
    check_keys(table, ("schema_version", "length", "matrix", "recipe", "training", "selection", "latency", "bootstrap", "inference"), (), where)
    check_version(table, where)
    length = as_table(table["length"], f"{where}: [length]")
    check_keys(length, ("max_pair_tokens", "max_query_tokens"), (), f"{where}: [length]")
    matrix = as_table(table["matrix"], f"{where}: [matrix]")
    check_keys(matrix, ("seeds", "deployment_seed", "epochs", "head_seed"), (), f"{where}: [matrix]")
    seeds = as_int_list(matrix["seeds"], f"{where}: matrix.seeds", 0, 2**31 - 1)
    deployment = as_int(matrix["deployment_seed"], f"{where}: matrix.deployment_seed", 0, 2**31 - 1)
    if deployment not in seeds:
        raise ConfigError(f"{where}: the deployment seed must be one of the declared seeds")
    if not isinstance(table["recipe"], list) or not table["recipe"]:
        raise ConfigError(f"{where}: declare at least one [[recipe]]")
    recipes = []
    for index, entry in enumerate(table["recipe"]):
        label = f"{where}: recipe[{index}]"
        entry = as_table(entry, label)
        check_keys(entry, ("id", "mode", "learning_rate"), (), label)
        recipes.append(Recipe(as_string(entry["id"], f"{label}.id", max_length=64), as_choice(entry["mode"], f"{label}.mode", ("frozen", "full")),
                              as_number(entry["learning_rate"], f"{label}.learning_rate", 1e-9, 1.0)))
    ids = [r.recipe_id for r in recipes]
    if len(set(ids)) != len(ids) or not all(all(c.isalnum() or c in "-_." for c in i) for i in ids):
        raise ConfigError(f"{where}: recipe ids must be unique identifiers")
    selection = as_table(table["selection"], f"{where}: [selection]")
    check_keys(selection, ("min_group_mrr_gain", "max_p95_ms"), (), f"{where}: [selection]")
    latency = as_table(table["latency"], f"{where}: [latency]")
    check_keys(latency, ("warmup", "passes", "queries", "seed"), (), f"{where}: [latency]")
    bootstrap = as_table(table["bootstrap"], f"{where}: [bootstrap]")
    check_keys(bootstrap, ("confidence", "repetitions", "seed"), (), f"{where}: [bootstrap]")
    inference = as_table(table["inference"], f"{where}: [inference]")
    check_keys(inference, ("batch_candidates",), (), f"{where}: [inference]")
    protocol = Protocol(
        max_pair_tokens=as_int(length["max_pair_tokens"], f"{where}: length.max_pair_tokens", 16, 4096),
        max_query_tokens=as_int(length["max_query_tokens"], f"{where}: length.max_query_tokens", 8, 4096),
        seeds=seeds, deployment_seed=deployment, epochs=as_int(matrix["epochs"], f"{where}: matrix.epochs", 1, MAX_EPOCHS),
        head_seed=as_int(matrix["head_seed"], f"{where}: matrix.head_seed", 0, 2**31 - 1), recipes=tuple(recipes),
        training=_training(table["training"], f"{where}: [training]"),
        min_group_mrr_gain=as_number(selection["min_group_mrr_gain"], f"{where}: selection.min_group_mrr_gain", 0.0, 1.0),
        max_p95_ms=as_number(selection["max_p95_ms"], f"{where}: selection.max_p95_ms", 1.0, 3_600_000.0),
        latency_warmup=as_int(latency["warmup"], f"{where}: latency.warmup", 1, 1000),
        latency_passes=as_int(latency["passes"], f"{where}: latency.passes", 1, 100),
        latency_queries=as_int(latency["queries"], f"{where}: latency.queries", 1, 10_000),
        latency_seed=as_int(latency["seed"], f"{where}: latency.seed", 0, 2**63 - 1),
        bootstrap_confidence=as_number(bootstrap["confidence"], f"{where}: bootstrap.confidence", 0.5, 0.999),
        bootstrap_repetitions=as_int(bootstrap["repetitions"], f"{where}: bootstrap.repetitions", 100, 1_000_000),
        bootstrap_seed=as_int(bootstrap["seed"], f"{where}: bootstrap.seed", 0, 2**63 - 1),
        inference_batches=as_int_list(inference["batch_candidates"], f"{where}: inference.batch_candidates", 1, 4096),
    )
    if not 8 <= protocol.max_query_tokens < protocol.max_pair_tokens:
        raise ConfigError(f"{where}: need 8 <= max_query_tokens < max_pair_tokens")
    return protocol


def protocol_file_sha256(path: Path) -> str:
    return sha256_file(Path(path))
