"""Strict configuration shared by the commands that load a model: `predict`, `evaluate`, `export` and `serve`.

    schema_version = 1
    [device]  profile = "cpu-diagnostic"                  explicit; an unavailable device is an error, never a CPU fallback
    [model]   batch_size = 8                              inference batch (online and offline must agree: it is part of the scoring identity)
              dropout = 0.1                               optional: the training recipe's head dropout, recorded when a model is rebuilt from a checkpoint
                                                          (never applied at inference, but part of an exported model's identity)
    [assets]  directory = "model-assets"                  needed only to rebuild a model from a *training* checkpoint
    [policy]  max_pair_tokens = 128  max_query_tokens = 32   likewise; an inference bundle carries its own policy

Command-specific sections are handled by each command. Unknown keys fail; relative paths resolve against the config file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from devsearch.artifacts.tomlconfig import as_int, as_number, as_string, as_table, check_keys, check_version, load_toml
from devsearch.errors import ConfigError
from devsearch.modeling.policy import PairEncodingPolicy


@dataclass(frozen=True)
class ModelJob:
    profile: str
    batch_size: int
    assets: Path | None
    policy: PairEncodingPolicy | None
    dropout: float
    table: Mapping[str, Any]
    base: Path


def load_model_job(path: Path, *, extra_sections: tuple[str, ...] = (), required_extra: tuple[str, ...] = ()) -> ModelJob:
    path = Path(path).resolve()
    where = str(path)
    table = load_toml(path)
    check_keys(table, ("schema_version", "device", "model", *required_extra), ("assets", "policy", *extra_sections), where)
    check_version(table, where)
    device, model = as_table(table["device"], f"{where}: [device]"), as_table(table["model"], f"{where}: [model]")
    check_keys(device, ("profile",), (), f"{where}: [device]")
    check_keys(model, ("batch_size",), ("dropout",), f"{where}: [model]")
    assets = policy = None
    if "assets" in table:
        section = as_table(table["assets"], f"{where}: [assets]")
        check_keys(section, ("directory",), (), f"{where}: [assets]")
        assets = (path.parent / as_string(section["directory"], f"{where}: assets.directory")).resolve()
    if "policy" in table:
        section = as_table(table["policy"], f"{where}: [policy]")
        check_keys(section, ("max_pair_tokens", "max_query_tokens"), (), f"{where}: [policy]")
        try:
            policy = PairEncodingPolicy(as_int(section["max_pair_tokens"], f"{where}: policy.max_pair_tokens", 16, 4096),
                                        as_int(section["max_query_tokens"], f"{where}: policy.max_query_tokens", 8, 4096))
        except ValueError as error:
            raise ConfigError(f"{where}: [policy]: {error}") from error
    return ModelJob(as_string(device["profile"], f"{where}: device.profile", max_length=64), as_int(model["batch_size"], f"{where}: model.batch_size", 1, 4096),
                    assets, policy, as_number(model.get("dropout", 0.1), f"{where}: model.dropout", 0.0, 0.99), table, path.parent)
