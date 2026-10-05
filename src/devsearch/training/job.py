"""The standalone `train` job: a training recipe plus where its inputs live (docs/03 §4).

Extra sections beyond the recipe (strict, all required):

    [data]    prepared_manifest = "path/to/prepared-data-manifest.json"      relative to the config file
    [assets]  directory = "path/to/model-assets"                              pinned encoder + tokenizer
    [device]  profile = "cpu-diagnostic"                                      apple-mps | nvidia-cuda | cpu-diagnostic
    [policy]  max_pair_tokens = 128   max_query_tokens = 32

A campaign takes the same values from its plan and the frozen protocol instead; this exists so one recipe can be run,
resumed and inspected by hand, with the same loop, checkpoints and refusals.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from devsearch.artifacts.tomlconfig import as_int, as_string, as_table, check_keys
from devsearch.errors import ConfigError
from devsearch.training.config import JOB_SECTIONS, TrainConfig, load_train_config


@dataclass(frozen=True)
class TrainJob:
    config: TrainConfig
    prepared_manifest: Path
    assets: Path
    profile: str
    max_pair_tokens: int
    max_query_tokens: int


def load_train_job(path: Path) -> TrainJob:
    path = Path(path).resolve()
    config = load_train_config(path)
    where = str(path)
    table = tomllib.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in JOB_SECTIONS if name not in table]
    if missing:
        raise ConfigError(f"{where}: the standalone train command also needs section(s) {', '.join(missing)}")
    base = path.parent
    data, assets = as_table(table["data"], f"{where}: [data]"), as_table(table["assets"], f"{where}: [assets]")
    device, policy = as_table(table["device"], f"{where}: [device]"), as_table(table["policy"], f"{where}: [policy]")
    check_keys(data, ("prepared_manifest",), (), f"{where}: [data]")
    check_keys(assets, ("directory",), (), f"{where}: [assets]")
    check_keys(device, ("profile",), (), f"{where}: [device]")
    check_keys(policy, ("max_pair_tokens", "max_query_tokens"), (), f"{where}: [policy]")
    return TrainJob(
        config=config, prepared_manifest=(base / as_string(data["prepared_manifest"], f"{where}: data.prepared_manifest")).resolve(),
        assets=(base / as_string(assets["directory"], f"{where}: assets.directory")).resolve(), profile=as_string(device["profile"], f"{where}: device.profile", max_length=64),
        max_pair_tokens=as_int(policy["max_pair_tokens"], f"{where}: policy.max_pair_tokens", 16, 4096),
        max_query_tokens=as_int(policy["max_query_tokens"], f"{where}: policy.max_query_tokens", 8, 4096))
