"""Strict training configuration (docs/03 §2, docs/12 §1): every choice explicit, nothing hidden.

A run is fully described by its config plus the identities of its inputs. The config's canonical
hash is part of a checkpoint's immutable identity: resuming with a different hyperparameter is a
*different run* (a fork), never a silent continuation.
"""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import digest_of
from devsearch.errors import ConfigError

MAX_COMPLETED_EPOCHS = 3          # epoch zero plus at most three completed epochs are retained (docs/10 §5)
MAX_DIAGNOSTIC_EPOCHS = 300       # tiny-set overfit checks only; never a formal recipe


@dataclass(frozen=True)
class TrainConfig:
    recipe_id: str
    mode: str                         # "frozen" (head only, encoder in eval) or "full"
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.10
    effective_batch: int = 16
    microbatch: int = 4
    dropout: float = 0.1
    head_seed: int = 0
    pos_weight: str | float = "auto"  # "auto" = actual sampled negative/positive ratio, or an explicit number
    validation_batch: int = 8
    diagnostic: bool = False          # allows long tiny-set overfit checks; formal recipes keep epochs <= 3

    def __post_init__(self) -> None:
        problems = []
        if not self.recipe_id or not all(c.isalnum() or c in "-_." for c in self.recipe_id):
            problems.append("recipe_id must be a non-empty identifier")
        if self.mode not in ("frozen", "full"):
            problems.append("mode must be 'frozen' or 'full'")
        limit = MAX_DIAGNOSTIC_EPOCHS if self.diagnostic else MAX_COMPLETED_EPOCHS
        if not 1 <= self.epochs <= limit:
            problems.append(f"epochs must be between 1 and {limit}" + ("" if self.diagnostic else " (diagnostic runs may use more)"))
        for name in ("learning_rate", "weight_decay", "eps", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                problems.append(f"{name} must be finite and non-negative")
        if self.learning_rate <= 0 or self.eps <= 0 or self.max_grad_norm <= 0:
            problems.append("learning_rate, eps and max_grad_norm must be positive")
        if not (0 < self.beta1 < 1 and 0 < self.beta2 < 1):
            problems.append("betas must be in (0, 1)")
        if not 0 <= self.warmup_fraction < 1:
            problems.append("warmup_fraction must be in [0, 1)")
        if self.effective_batch < 1 or self.microbatch < 1 or self.effective_batch % self.microbatch != 0:
            problems.append("microbatch must divide effective_batch (both positive)")
        if not 0 <= self.dropout < 1:
            problems.append("dropout must be in [0, 1)")
        if self.validation_batch < 1:
            problems.append("validation_batch must be positive")
        if not (self.pos_weight == "auto" or (isinstance(self.pos_weight, (int, float)) and not isinstance(self.pos_weight, bool) and self.pos_weight > 0 and math.isfinite(self.pos_weight))):
            problems.append("pos_weight must be 'auto' or a positive finite number")
        if problems:
            raise ConfigError("invalid training config: " + "; ".join(problems))

    @property
    def accumulation_steps(self) -> int:
        return self.effective_batch // self.microbatch

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_sha256(self) -> str:
        return digest_of(self.to_dict())


_REQUIRED = ("schema_version", "run", "optimizer", "batch", "model")


def load_train_config(path: Path) -> TrainConfig:
    """TOML sections: [run] recipe_id mode seed epochs; [optimizer] ...; [batch] ...; [model] ...; [loss]; [validation]."""
    try:
        table = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot read training config {path}: {error}") from error
    where = str(path)
    allowed = {*_REQUIRED, "loss", "validation"}
    if table.keys() - allowed or set(_REQUIRED) - table.keys():
        raise ConfigError(f"{where}: unknown {sorted(table.keys() - allowed)} or missing {sorted(set(_REQUIRED) - table.keys())} section(s)")
    if table["schema_version"] != 1:
        raise ConfigError(f"{where}: unsupported schema_version")
    sections = {
        "run": ({"recipe_id", "mode", "seed", "epochs"}, {"recipe_id", "mode", "seed", "epochs"}),
        "optimizer": ({"learning_rate", "weight_decay", "betas", "eps", "max_grad_norm", "warmup_fraction"}, {"learning_rate", "weight_decay", "betas", "eps", "max_grad_norm", "warmup_fraction"}),
        "batch": ({"effective_batch", "microbatch"}, {"effective_batch", "microbatch"}),
        "model": ({"dropout", "head_seed"}, {"dropout", "head_seed"}),
        "loss": (set(), {"pos_weight"}), "validation": (set(), {"batch_size"}),
    }
    values: dict[str, Any] = {}
    for name, (required, allowed_keys) in sections.items():
        section = table.get(name, {})
        if not isinstance(section, dict) or section.keys() - allowed_keys or required - section.keys():
            raise ConfigError(f"{where}: [{name}] has unknown or missing keys")
        values[name] = section
    betas = values["optimizer"]["betas"]
    if not (isinstance(betas, list) and len(betas) == 2):
        raise ConfigError(f"{where}: optimizer.betas must be a two-element list")
    for key in ("seed", "epochs"):
        if isinstance(values["run"][key], bool) or not isinstance(values["run"][key], int):
            raise ConfigError(f"{where}: run.{key} must be an integer")
    try:
        return TrainConfig(
            recipe_id=values["run"]["recipe_id"], mode=values["run"]["mode"], seed=values["run"]["seed"], epochs=values["run"]["epochs"],
            learning_rate=float(values["optimizer"]["learning_rate"]), weight_decay=float(values["optimizer"]["weight_decay"]),
            beta1=float(betas[0]), beta2=float(betas[1]), eps=float(values["optimizer"]["eps"]),
            max_grad_norm=float(values["optimizer"]["max_grad_norm"]), warmup_fraction=float(values["optimizer"]["warmup_fraction"]),
            effective_batch=values["batch"]["effective_batch"], microbatch=values["batch"]["microbatch"],
            dropout=float(values["model"]["dropout"]), head_seed=values["model"]["head_seed"],
            pos_weight=values["loss"].get("pos_weight", "auto"), validation_batch=values["validation"].get("batch_size", 8),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where}: {error}") from error
