"""Pinned model assets: the encoder weights, config and tokenizer files, hashed and loadable offline (docs/10 §3).

`<assets>/` holds `config.json`, `model.safetensors`, the tokenizer files (`vocab.txt` and `tokenizer_config.json`,
or `tokenizer.json`) and `assets-manifest.json`. Weights are safetensors only: a pickled `.bin` is never loaded.
`register_assets` hashes an already-present directory; it downloads nothing (formal compute does no network
asset access, and acquisition is a one-time preparation step the operator does on a connected machine).

`kind` is `pretrained` for a real pinned model (a 40-hex commit revision is required to count as pinned) or
`diagnostic` for the tiny local encoder used by rehearsals and tests. A diagnostic asset set can never satisfy a
backend gate: it proves the pipeline, not the model.
"""

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import pretty_json, seal_manifest, sha256_file, verify_manifest_hash
from devsearch.artifacts.files import write_atomic
from devsearch.errors import ManifestError
from devsearch.execution.records import SCHEMA_VERSION, read_json

ASSETS_MANIFEST = "assets-manifest.json"
REQUIRED = ("config.json", "model.safetensors")
TOKENIZER_FILES = ("tokenizer.json", "vocab.txt", "tokenizer_config.json")
FIELDS = ("asset_kind", "source", "pinned", "files", "encoder", "tokenizer_files")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _inventory(directory: Path) -> dict[str, dict[str, Any]]:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.name == ASSETS_MANIFEST or path.name.startswith("."):
            continue
        if path.is_symlink() or not path.is_file():
            raise ManifestError(f"{directory}: {path.name} must be a regular file")
        files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def register_assets(directory: Path, *, asset_kind: str, source: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the files in `directory` and write `assets-manifest.json` (never overwrites a different manifest)."""
    directory = Path(directory)
    if asset_kind not in ("pretrained", "diagnostic"):
        raise ManifestError("asset_kind must be 'pretrained' or 'diagnostic'")
    files = _inventory(directory)
    missing = [name for name in REQUIRED if name not in files]
    tokenizer_files = [name for name in TOKENIZER_FILES if name in files]
    if missing or not ({"tokenizer.json", "vocab.txt"} & set(tokenizer_files)):
        raise ManifestError(f"{directory}: needs {', '.join(REQUIRED)} and a tokenizer (vocab.txt or tokenizer.json); missing {missing or 'tokenizer'}")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    encoder = {key: config.get(key) for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size", "max_position_embeddings")}
    if not all(isinstance(value, int) for value in encoder.values()):
        raise ManifestError(f"{directory}/config.json lacks the integer encoder fields {sorted(encoder)}")
    pinned = asset_kind == "pretrained" and bool(_COMMIT.match(str(source.get("revision", ""))))
    manifest = seal_manifest({
        "schema_version": SCHEMA_VERSION, "kind": "model-assets", "asset_kind": asset_kind, "source": dict(source), "pinned": pinned,
        "files": files, "encoder": encoder, "tokenizer_files": tokenizer_files,
    })
    path = directory / ASSETS_MANIFEST
    if path.exists():
        existing = read_json(path, str(path))
        if existing.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ManifestError(f"{path} already describes different assets; asset sets are immutable")
        return existing
    write_atomic(path, pretty_json(manifest).encode("utf-8"), exclusive=True)
    return manifest


def verify_assets(directory: Path) -> dict[str, Any]:
    """Structure and every file hash; reads no tensors."""
    directory = Path(directory)
    where = str(directory / ASSETS_MANIFEST)
    manifest = read_json(directory / ASSETS_MANIFEST, where)
    expected = {"schema_version", "kind", *FIELDS, "manifest_sha256"}
    if set(manifest) != expected or manifest["kind"] != "model-assets" or manifest["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(f"{where}: not a schema-{SCHEMA_VERSION} model-assets manifest")
    verify_manifest_hash(manifest, where=where)
    actual = _inventory(directory)
    if actual != manifest["files"]:
        differing = sorted(set(actual) ^ set(manifest["files"]) | {n for n in actual if n in manifest["files"] and actual[n] != manifest["files"][n]})
        raise ManifestError(f"{where}: files differ from the manifest: {differing}")
    return manifest


def build_diagnostic_assets(directory: Path, words: list[str], *, seed: int = 0, hidden_size: int = 32, layers: int = 2) -> dict[str, Any]:
    """A tiny random BERT plus a local tokenizer, saved as safetensors. Strictly a rehearsal/test asset, never a model."""
    from safetensors.torch import save_file

    from devsearch.modeling.cross_encoder import build_tiny_encoder
    from devsearch.modeling.tokenization import write_diagnostic_tokenizer

    directory = Path(directory)
    if (directory / ASSETS_MANIFEST).exists() or any(directory.iterdir()):
        raise ManifestError(f"{directory} is not empty; diagnostic assets are written into a fresh directory")
    tokenizer_dir = write_diagnostic_tokenizer(directory, words)
    vocab_size = len(json.loads((tokenizer_dir / "tokenizer.json").read_text())["model"]["vocab"])
    encoder = build_tiny_encoder(vocab_size, hidden_size=hidden_size, layers=layers, seed=seed, hidden_dropout=0.1, attention_dropout=0.1)
    state = {key: value.detach().to("cpu").contiguous().clone() for key, value in encoder.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"))
    config = encoder.config
    (directory / "config.json").write_text(json.dumps({
        "model_type": "bert", "vocab_size": config.vocab_size, "hidden_size": config.hidden_size, "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads, "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings, "hidden_act": config.hidden_act,
        "hidden_dropout_prob": config.hidden_dropout_prob, "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
        "type_vocab_size": config.type_vocab_size, "initializer_range": config.initializer_range, "layer_norm_eps": config.layer_norm_eps,
        "pad_token_id": config.pad_token_id,
    }, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return register_assets(directory, asset_kind="diagnostic", source={"diagnostic": True, "seed": seed, "note": "random weights; never a trained model"})


def load_encoder(directory: Path):
    """Load the pinned encoder from local safetensors only. Returns a `BertModel` without a pooler."""
    from safetensors.torch import load_file
    from transformers import BertConfig, BertModel

    directory = Path(directory)
    verify_assets(directory)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    from devsearch.modeling.io import ARCH_FIELDS
    config = BertConfig(**{key: raw[key] for key in ARCH_FIELDS if key in raw})
    model = BertModel(config, add_pooling_layer=False)
    state = load_file(str(directory / "model.safetensors"), device="cpu")
    state = {(key[len("bert."):] if key.startswith("bert.") else key): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=False)
    allowed_extra = [key for key in result.unexpected_keys if key.startswith(("cls.", "pooler.", "classifier."))]     # heads of the original checkpoint
    unexpected = [key for key in result.unexpected_keys if key not in allowed_extra]
    missing = [key for key in result.missing_keys if "position_ids" not in key]
    if missing or unexpected:
        raise ManifestError(f"{directory}: weights do not fit the declared encoder (missing {missing[:3]}, unexpected {unexpected[:3]})")
    return model
