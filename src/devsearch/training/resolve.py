"""Turn a `--checkpoint` reference into a scoring model: an exported inference bundle or a training checkpoint.

* a directory holding `model-manifest.json` is an exported inference bundle (verified, loaded without any other input);
* a `<run>/checkpoints/<id>.json` manifest (or `best.json` / `latest.json`) names a training checkpoint: its file hash is
  verified, only tensors are loaded (`weights_only`), and the encoder architecture comes from the pinned assets.

Either way the model is returned in eval mode on the requested device, with its tokenizer and pair-encoding policy.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from devsearch.artifacts.canonical import sha256_file
from devsearch.errors import ManifestError
from devsearch.modeling.assets import load_encoder, verify_assets
from devsearch.modeling.cross_encoder import CrossEncoder
from devsearch.modeling.io import MANIFEST_NAME, load_model_bundle
from devsearch.modeling.policy import PairEncodingPolicy
from devsearch.modeling.tokenization import WordPieceTokenizer


@dataclass(frozen=True)
class ResolvedModel:
    model: CrossEncoder
    tokenizer: WordPieceTokenizer
    policy: PairEncodingPolicy
    model_id: str
    source: str
    checkpoint_sha256: str | None
    details: Mapping[str, Any]          # what the checkpoint manifest recorded (empty for an inference bundle); used as export provenance


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: expected a JSON object")
    return value


def resolve_model(
    reference: Path, *, device: torch.device, assets: Path | None, policy: PairEncodingPolicy | None, dropout: float = 0.1, head_seed: int = 0,
) -> ResolvedModel:
    reference = Path(reference)
    if reference.is_dir() and (reference / MANIFEST_NAME).is_file():
        loaded = load_model_bundle(reference, device=device)
        return ResolvedModel(loaded.model, loaded.tokenizer, loaded.policy, loaded.model_id, "inference-bundle", loaded.manifest["provenance"].get("checkpoint_sha256"), {})
    if not reference.is_file() or reference.suffix != ".json":
        raise ManifestError(f"{reference} is neither an inference bundle directory nor a checkpoint manifest (.json)")
    if assets is None or policy is None:
        raise ManifestError("a training checkpoint needs the pinned assets directory and the pair-encoding policy to be rebuilt")
    manifest = _read_json(reference)
    if set(manifest) <= {"checkpoint_id", "sha256"}:                       # a best/latest pointer
        name = manifest.get("checkpoint_id")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ManifestError(f"{reference}: the pointer names no valid checkpoint")
        pointed = manifest.get("sha256")
        reference = reference.with_name(f"{name}.json")
        manifest = _read_json(reference)
        if pointed != manifest.get("sha256"):
            raise ManifestError(f"the pointer and checkpoint {name} disagree about the file hash")
    file_name, expected = manifest.get("file"), manifest.get("sha256")
    if not isinstance(file_name, str) or Path(file_name).name != file_name or not isinstance(expected, str):
        raise ManifestError(f"{reference}: not a training-checkpoint manifest (needs a plain `file` name and a `sha256`)")
    file = reference.parent / file_name
    if not file.is_file() or sha256_file(file) != expected:
        raise ManifestError(f"checkpoint {manifest.get('checkpoint_id')} is missing or does not match its recorded SHA-256")
    assets_manifest = verify_assets(assets)
    model = CrossEncoder(load_encoder(assets), dropout=dropout, head_seed=head_seed)
    try:
        state = torch.load(file, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
    except (RuntimeError, KeyError, TypeError, OSError, EOFError) as error:
        raise ManifestError(f"checkpoint {manifest.get('checkpoint_id')} does not fit the pinned encoder assets: {error}") from error
    model.eval()
    model.to(device)
    details = {key: manifest[key] for key in ("checkpoint_id", "resume_kind", "recipe_id", "completed_epoch", "global_step", "identity_sha256") if key in manifest}
    details["assets_manifest_sha256"] = assets_manifest["manifest_sha256"]
    return ResolvedModel(model, WordPieceTokenizer(assets), policy, "checkpoint-" + expected[:24], "training-checkpoint", expected, details)
