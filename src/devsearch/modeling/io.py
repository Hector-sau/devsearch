"""Offline inference bundle: safetensors weights, explicit model construction, a hashed manifest.

Layout of `<bundle>/` (docs/03 §3): `model-manifest.json`, `encoder.safetensors`, `head.safetensors`,
`encoder-config.json`, `tokenizer/` (every tokenizer file) and `parity-examples.json`. Weights are plain
state dictionaries rebuilt through an explicit constructor, never a pickled live model. Loading
needs no network and no Hub: every file is checked against the manifest before a tensor is read.

`model_id` is content-addressed over weights, configs, tokenizer and pair-encoding policy; run and
checkpoint provenance is recorded in the manifest but does not change the identity, so the same
weights exported twice are the same model. A bundle is immutable: an identical re-export returns the
existing manifest, a different one at the same place is refused. Publishing stages everything in a
sibling directory and renames it into place, so a crash leaves no half-written bundle.
"""

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import BertConfig, BertModel

from devsearch.artifacts.canonical import (
    canonical_json, digest_of, loads_strict, pretty_json, read_utf8, seal_manifest, sha256_file, verify_manifest_hash,
)
from devsearch.artifacts.schema import as_int, as_object, as_sha256, exact_keys
from devsearch.errors import ManifestError
from devsearch.modeling.cross_encoder import CrossEncoder
from devsearch.modeling.policy import PairEncodingPolicy
from devsearch.modeling.scoring import score_pairs
from devsearch.modeling.tokenization import WordPieceTokenizer

MODEL_SCHEMA_VERSION = 1
MANIFEST_NAME = "model-manifest.json"
ENCODER_FILE, HEAD_FILE, CONFIG_FILE, PARITY_FILE, TOKENIZER_DIR = (
    "encoder.safetensors", "head.safetensors", "encoder-config.json", "parity-examples.json", "tokenizer",
)
# The architecture fields that define a BERT encoder here. Stored explicitly (not as a diff against library
# defaults) so rebuilding the model never depends on what a future library version defaults to.
ARCH_FIELDS = (
    "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "hidden_act",
    "hidden_dropout_prob", "attention_probs_dropout_prob", "max_position_embeddings", "type_vocab_size",
    "initializer_range", "layer_norm_eps", "pad_token_id",
)
DEFAULT_PARITY_ATOL = 1e-4          # same device and same software; looser values must be declared by the caller
MAX_PARITY_EXAMPLES = 16


@dataclass(frozen=True)
class LoadedModel:
    model: CrossEncoder
    tokenizer: WordPieceTokenizer
    policy: PairEncodingPolicy
    manifest: dict[str, Any]
    directory: Path

    @property
    def model_id(self) -> str:
        return self.manifest["model_id"]


def _architecture(config: BertConfig) -> dict[str, Any]:
    return {name: getattr(config, name) for name in ARCH_FIELDS}


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().to("cpu").contiguous().clone() for key, value in module.state_dict().items()}


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _identity(files: dict[str, dict[str, Any]], architecture: dict[str, Any], head: dict[str, Any], tokenizer: str,
              policy: PairEncodingPolicy) -> str:
    return "model-" + digest_of({
        "files": files, "architecture": architecture, "head": head, "tokenizer": tokenizer,
        "policy": {"max_pair_tokens": policy.max_pair_tokens, "max_query_tokens": policy.max_query_tokens},
    })[:32]


def export_model(
    model: CrossEncoder, destination: Path, *, tokenizer: WordPieceTokenizer, policy: PairEncodingPolicy,
    provenance: dict[str, Any], parity_examples: list[tuple[str, str]], device: torch.device | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Write an immutable inference bundle at `destination` (which must not exist unless identical).

    `parity_examples` are fixed (query, passage) pairs whose logits are recorded now and re-checked after
    every reload. They must come from diagnostic or validation material, never from the test split.
    """
    destination = Path(destination)
    if not 1 <= len(parity_examples) <= MAX_PARITY_EXAMPLES:
        raise ManifestError(f"export needs between 1 and {MAX_PARITY_EXAMPLES} parity examples")
    device = device or next(model.parameters()).device
    staging = destination.with_name(f".{destination.name}.staging-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    if not destination.parent.is_dir():
        raise ManifestError(f"{destination.parent} does not exist; the exporter never creates parent directories")
    staging.mkdir()
    try:
        save_file(_cpu_state(model.encoder), str(staging / ENCODER_FILE))
        save_file(_cpu_state(model.head), str(staging / HEAD_FILE))
        architecture = _architecture(model.encoder.config)
        (staging / CONFIG_FILE).write_text(canonical_json(architecture) + "\n", encoding="utf-8")
        tokenizer_dir = staging / TOKENIZER_DIR
        tokenizer_dir.mkdir()
        for source in tokenizer.files():
            shutil.copyfile(source, tokenizer_dir / source.name)
        copied = WordPieceTokenizer(tokenizer_dir)
        if copied.fingerprint != tokenizer.fingerprint:
            raise ManifestError("copied tokenizer files do not reproduce the tokenizer fingerprint")
        logits = score_pairs(model, copied, policy, parity_examples, device=device, batch_size=batch_size)
        (staging / PARITY_FILE).write_text(
            pretty_json({"examples": [{"query": q, "passage": p} for q, p in parity_examples], "logits": logits,
                         "ranks": _ranks(logits), "atol": DEFAULT_PARITY_ATOL}),
            encoding="utf-8")
        files = {
            name: _file_record(staging / name)
            for name in (ENCODER_FILE, HEAD_FILE, CONFIG_FILE, PARITY_FILE)
        }
        files.update({f"{TOKENIZER_DIR}/{path.name}": _file_record(path) for path in sorted(tokenizer_dir.iterdir())})
        head = {"in_features": model.head.in_features, "dropout": model.dropout.p}
        model_id = _identity(files, architecture, head, tokenizer.fingerprint, policy)
        manifest = seal_manifest({
            "schema_version": MODEL_SCHEMA_VERSION, "kind": "model", "model_id": model_id, "files": files,
            "architecture": architecture, "head": head, "tokenizer": {"fingerprint": tokenizer.fingerprint},
            "policy": {"max_pair_tokens": policy.max_pair_tokens, "max_query_tokens": policy.max_query_tokens},
            "provenance": provenance,
        })
        (staging / MANIFEST_NAME).write_text(pretty_json(manifest), encoding="utf-8")
        _verify_directory(staging)                      # the staged bundle must pass the same check a consumer runs
        if destination.exists():
            existing = _verify_directory(destination)
            if existing["model_id"] != model_id:
                raise ManifestError(f"{destination} already holds a different model ({existing['model_id']}); bundles are immutable")
            return existing
        os.rename(staging, destination)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _ranks(logits: list[float]) -> list[int]:
    order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
    ranks = [0] * len(logits)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def read_manifest(directory: Path) -> dict[str, Any]:
    """Parse and structurally validate the manifest; file hashes are checked by `verify_bundle`."""
    path = Path(directory) / MANIFEST_NAME
    manifest = as_object(loads_strict(read_utf8(path, where=str(path)), where=str(path)), str(path))
    exact_keys(manifest, {"schema_version", "kind", "model_id", "files", "architecture", "head", "tokenizer", "policy",
                          "provenance", "manifest_sha256"}, str(path))
    if manifest["schema_version"] != MODEL_SCHEMA_VERSION or manifest["kind"] != "model":
        raise ManifestError(f"{path}: not a schema-{MODEL_SCHEMA_VERSION} model manifest")
    verify_manifest_hash(manifest, where=str(path))
    files = as_object(manifest["files"], f"{path}: files")
    for name, record in files.items():
        if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
            raise ManifestError(f"{path}: file name {name!r} escapes the bundle")
        record = as_object(record, f"{path}: files[{name}]")
        exact_keys(record, {"sha256", "bytes"}, f"{path}: files[{name}]")
        as_sha256(record["sha256"], f"{path}: files[{name}].sha256")
        as_int(record["bytes"], f"{path}: files[{name}].bytes")
    required = {ENCODER_FILE, HEAD_FILE, CONFIG_FILE, PARITY_FILE}
    if not required <= set(files) or not any(name.startswith(TOKENIZER_DIR + "/") for name in files):
        raise ManifestError(f"{path}: a required bundle file is not listed")
    policy = as_object(manifest["policy"], f"{path}: policy")
    exact_keys(policy, {"max_pair_tokens", "max_query_tokens"}, f"{path}: policy")
    return dict(manifest)


def _verify_directory(directory: Path) -> dict[str, Any]:
    manifest = read_manifest(directory)
    listed = set(manifest["files"]) | {MANIFEST_NAME}
    present = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if present != listed:
        raise ManifestError(f"{directory}: files differ from the manifest (extra: {sorted(present - listed)}, missing: {sorted(listed - present)})")
    for name, record in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ManifestError(f"{directory}: {name} does not match its recorded size and SHA-256")
    policy = PairEncodingPolicy(**manifest["policy"])
    expected = _identity(manifest["files"], manifest["architecture"], manifest["head"], manifest["tokenizer"]["fingerprint"], policy)
    if manifest["model_id"] != expected:
        raise ManifestError(f"{directory}: model_id does not match the bundle contents")
    return manifest


def verify_bundle(directory: Path) -> dict[str, Any]:
    """Check structure and every file hash without reading any tensor."""
    return _verify_directory(Path(directory))


def load_model_bundle(directory: Path, *, device: torch.device) -> LoadedModel:
    """Verify, then rebuild the model explicitly and load weights strictly. Never touches the network."""
    directory = Path(directory)
    manifest = _verify_directory(directory)
    tokenizer = WordPieceTokenizer(directory / TOKENIZER_DIR)
    if tokenizer.fingerprint != manifest["tokenizer"]["fingerprint"]:
        raise ManifestError(f"{directory}: the tokenizer files do not match the recorded fingerprint")
    architecture = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
    if architecture != manifest["architecture"]:
        raise ManifestError(f"{directory}: encoder-config.json disagrees with the manifest")
    encoder = BertModel(BertConfig(**architecture), add_pooling_layer=False)
    model = CrossEncoder(encoder, dropout=float(manifest["head"]["dropout"]))
    try:
        encoder.load_state_dict(load_file(str(directory / ENCODER_FILE), device="cpu"), strict=True)
        model.head.load_state_dict(load_file(str(directory / HEAD_FILE), device="cpu"), strict=True)
    except (RuntimeError, ValueError) as error:
        raise ManifestError(f"{directory}: weights do not fit the declared architecture: {error}") from error
    model.eval()
    model.to(device)
    return LoadedModel(model, tokenizer, PairEncodingPolicy(**manifest["policy"]), manifest, directory)


def check_parity(loaded: LoadedModel, *, device: torch.device, atol: float | None = None, batch_size: int = 8) -> dict[str, Any]:
    """Re-score the recorded fixed examples and compare logits and ranks with the export-time record.

    Same-device reload should agree within the recorded tolerance; a different device or library may need a
    looser tolerance, which the caller states explicitly and the report records. Ranks must match exactly
    unless two recorded logits are closer than the tolerance (then the tie is reported, not ignored).
    """
    record = json.loads((loaded.directory / PARITY_FILE).read_text(encoding="utf-8"))
    tolerance = float(record["atol"] if atol is None else atol)
    pairs = [(item["query"], item["passage"]) for item in record["examples"]]
    logits = score_pairs(loaded.model, loaded.tokenizer, loaded.policy, pairs, device=device, batch_size=batch_size)
    differences = [abs(a - b) for a, b in zip(logits, record["logits"])]
    ranks = _ranks(logits)
    near_ties = sorted(
        (a, b) for a in range(len(logits)) for b in range(a + 1, len(logits)) if abs(record["logits"][a] - record["logits"][b]) <= tolerance
    )
    rank_ok = ranks == record["ranks"] or bool(near_ties)
    passed = max(differences) <= tolerance and rank_ok
    return {"passed": passed, "max_abs_difference": max(differences), "atol": tolerance, "examples": len(pairs),
            "ranks_equal": ranks == record["ranks"], "near_ties": [list(t) for t in near_ties], "device": str(device)}
