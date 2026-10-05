"""Strict `data.toml` for `prepare_data`: where drafts come from and every preparation choice, explicit.

Example (relative paths resolve against the config file; see `configs/data.toml`):

    schema_version = 1
    protocol = "prejudged_fixed_candidates_v1"
    [inputs.queries]  train = "queries-train.jsonl" ...  (one file per split, as the loaders require)
    [inputs.qrels]    train = "qrels-train.jsonl"   ...  [inputs] links = "links.jsonl" (optional)
    [groups]      near_duplicate_threshold = 0.9  shingle_size = 5  seed = 20261005
    [groups.targets]  train = 0.7  validation = 0.15  test = 0.15
    [mining]      hard_negatives = 2  random_negatives = 2  seed = 42  hard_search_depth = 100
    [candidates]  depth = 50
    [index]       config = "index.toml"
    [review]      salt = "<at least 8 characters, kept with the evaluation inputs>"
    [lengths]     tokenizer_assets = "model-assets"  max_pair_tokens = 256  max_query_tokens = 64     (optional, strongly recommended)

With `[lengths]`, `prepare_data --stage materialize` measures every passage and every question of every split (the held-out
split included) with the pinned tokenizer BEFORE anything is written, so an over-long test question is found while the data
is being prepared rather than failing the campaign after the models are locked. Without it nothing is measured here.

The review salt blinds item IDs; it is configuration for the evaluation side, never shown to reviewers.
"""

from dataclasses import dataclass
from pathlib import Path

from devsearch.artifacts.canonical import sha256_file
from devsearch.artifacts.tomlconfig import (
    as_choice, as_int, as_number, as_string, as_table, check_keys, check_version, load_toml,
)
from devsearch.data.corpus import SPLITS
from devsearch.data.pairs import MiningConfig
from devsearch.data.review import PROTOCOLS
from devsearch.errors import ConfigError, ManifestError
from devsearch.retrieval.config import IndexConfig, load_index_config


@dataclass(frozen=True)
class LengthCheck:
    tokenizer_assets: Path
    max_pair_tokens: int
    max_query_tokens: int


@dataclass(frozen=True)
class DataConfig:
    protocol: str
    queries: tuple[tuple[str, Path], ...]
    qrels: tuple[tuple[str, Path], ...]
    links: Path | None
    near_duplicate_threshold: float
    shingle_size: int
    group_seed: int
    targets: tuple[tuple[str, float], ...]
    mining: MiningConfig
    candidate_depth: int
    index_config_path: Path
    index: IndexConfig
    salt: str
    config_sha256: str
    lengths: LengthCheck | None = None

    @property
    def target_map(self) -> dict[str, float]:
        return dict(self.targets)


def load_data_config(path: Path) -> DataConfig:
    path = Path(path).resolve()
    where = str(path)
    table = load_toml(path)
    check_keys(table, ("schema_version", "protocol", "inputs", "groups", "mining", "candidates", "index", "review"), ("lengths",), where)
    check_version(table, where)
    base = path.parent

    def resolve(value: object, label: str) -> Path:
        return (base / as_string(value, f"{where}: {label}")).resolve()

    inputs = as_table(table["inputs"], f"{where}: [inputs]")
    check_keys(inputs, ("queries", "qrels"), ("links",), f"{where}: [inputs]")

    def split_files(name: str) -> tuple[tuple[str, Path], ...]:
        files = as_table(inputs[name], f"{where}: [inputs.{name}]")
        check_keys(files, ("train", "validation", "test"), ("development",), f"{where}: [inputs.{name}]")
        return tuple(sorted((split, resolve(path, f"inputs.{name}.{split}")) for split, path in files.items()))

    groups = as_table(table["groups"], f"{where}: [groups]")
    check_keys(groups, ("near_duplicate_threshold", "shingle_size", "seed", "targets"), (), f"{where}: [groups]")
    targets = as_table(groups["targets"], f"{where}: [groups.targets]")
    if not targets or set(targets) - set(SPLITS):
        raise ConfigError(f"{where}: [groups.targets] must name splits among {', '.join(SPLITS)}")
    shares = tuple(sorted((name, as_number(value, f"{where}: groups.targets.{name}", 1e-9, 1.0)) for name, value in targets.items()))
    if not {"train", "validation", "test"} <= set(targets):
        raise ConfigError(f"{where}: [groups.targets] needs train, validation and test shares")
    mining = as_table(table["mining"], f"{where}: [mining]")
    check_keys(mining, ("hard_negatives", "random_negatives", "seed"), ("hard_search_depth",), f"{where}: [mining]")
    candidates = as_table(table["candidates"], f"{where}: [candidates]")
    check_keys(candidates, ("depth",), (), f"{where}: [candidates]")
    index = as_table(table["index"], f"{where}: [index]")
    check_keys(index, ("config",), (), f"{where}: [index]")
    review = as_table(table["review"], f"{where}: [review]")
    check_keys(review, ("salt",), (), f"{where}: [review]")
    salt = as_string(review["salt"], f"{where}: review.salt", max_length=200)
    if len(salt) < 8:
        raise ConfigError(f"{where}: review.salt needs at least 8 characters")
    try:
        mining_config = MiningConfig(
            hard_negatives=as_int(mining["hard_negatives"], f"{where}: mining.hard_negatives", 0, 50),
            random_negatives=as_int(mining["random_negatives"], f"{where}: mining.random_negatives", 0, 50),
            seed=as_int(mining["seed"], f"{where}: mining.seed", 0, 2**31 - 1),
            hard_search_depth=as_int(mining.get("hard_search_depth", 100), f"{where}: mining.hard_search_depth", 1, 10_000),
        )
    except ManifestError as error:                               # MiningConfig rejects impossible negative counts
        raise ConfigError(f"{where}: [mining]: {error}") from error
    index_path = resolve(index["config"], "index.config")
    lengths = None
    if "lengths" in table:
        section = as_table(table["lengths"], f"{where}: [lengths]")
        check_keys(section, ("tokenizer_assets", "max_pair_tokens", "max_query_tokens"), (), f"{where}: [lengths]")
        lengths = LengthCheck(resolve(section["tokenizer_assets"], "lengths.tokenizer_assets"), as_int(section["max_pair_tokens"], f"{where}: lengths.max_pair_tokens", 16, 4096),
                              as_int(section["max_query_tokens"], f"{where}: lengths.max_query_tokens", 8, 4096))
        if lengths.max_query_tokens >= lengths.max_pair_tokens:
            raise ConfigError(f"{where}: [lengths] needs max_query_tokens < max_pair_tokens")
    return DataConfig(
        protocol=as_choice(table["protocol"], f"{where}: protocol", tuple(sorted(PROTOCOLS))),
        queries=split_files("queries"), qrels=split_files("qrels"),
        links=resolve(inputs["links"], "inputs.links") if "links" in inputs else None,
        near_duplicate_threshold=as_number(groups["near_duplicate_threshold"], f"{where}: groups.near_duplicate_threshold", 0.5, 1.0),
        shingle_size=as_int(groups["shingle_size"], f"{where}: groups.shingle_size", 1, 50),
        group_seed=as_int(groups["seed"], f"{where}: groups.seed", 0, 2**63 - 1), targets=shares,
        mining=mining_config, candidate_depth=as_int(candidates["depth"], f"{where}: candidates.depth", 1, 100),
        index_config_path=index_path, index=load_index_config(index_path), salt=salt, config_sha256=sha256_file(path), lengths=lengths,
    )
