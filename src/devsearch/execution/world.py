"""The campaign's runtime context: verified inputs, tokenizer, device and model construction (docs/10 §3–§5).

`World` is built once per process from the immutable plan and the active host binding. Everything it exposes to
training and selection comes from the *prepared* (training-visible) bundle: train pairs, validation candidates and
their labels. The test side is reachable only through `evaluation(ctx)`, which refuses until the `select` stage has
committed — a code-level guarantee that no test query, candidate or judgment is opened before the models and the
default mode are locked on validation alone.

It also applies the frozen resource limits: a per-process device memory fraction on CUDA/MPS (so an allocation beyond
the cap fails as an out-of-memory error instead of silently growing), and a host-memory check between units of work.
Neither limit is raised, lowered or worked around at run time.
"""

from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any

import torch

from devsearch.artifacts.identity import code_identity
from devsearch.data.candidates import load_candidates
from devsearch.data.prepare import EVALUATION_MANIFEST, load_evaluation
from devsearch.data.seals import load_bank, load_seal
from devsearch.errors import ExecutionFailure, ManifestError, PrerequisiteError
from devsearch.evaluation.protocol import Protocol
from devsearch.modeling.assets import load_encoder
from devsearch.execution.bundle import verify_bundle
from devsearch.modeling.cross_encoder import CrossEncoder
from devsearch.modeling.devices import DeviceProfile, backend_fingerprint, load_profile, memory_stats, resolve_device
from devsearch.modeling.policy import PairEncodingPolicy
from devsearch.modeling.tokenization import WordPieceTokenizer
from devsearch.training.config import TrainConfig
from devsearch.training.data import ValidationSet
from devsearch.training.inputs import TrainingInputs, load_training_inputs

def apply_memory_cap(device: torch.device, cap_bytes: int, total_bytes: int | None) -> dict[str, Any]:
    """Limit this process's device memory to `cap_bytes`. Returns what was applied (nothing for CPU or an unknown capacity)."""
    if device.type == "cpu" or not cap_bytes:
        return {"applied": False, "reason": "no device cap applies to this device"}
    if not total_bytes:
        raise PrerequisiteError(f"the capacity of {device} is unknown, so a {cap_bytes}-byte cap cannot be turned into a limit")
    if cap_bytes > total_bytes:
        raise PrerequisiteError(f"the device cap {cap_bytes} bytes exceeds the observed capacity {total_bytes} bytes")
    fraction = cap_bytes / total_bytes
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    else:
        torch.mps.set_per_process_memory_fraction(fraction)
    return {"applied": True, "fraction": fraction, "cap_bytes": cap_bytes, "capacity_bytes": total_bytes}


def host_memory_check(device: torch.device, cap_bytes: int) -> None:
    """Fail (never adapt) if this process's peak resident memory passed the frozen host cap."""
    peak = memory_stats(device)["host_peak_rss_bytes"]
    if cap_bytes and peak > cap_bytes:
        raise ExecutionFailure(f"the host memory cap of {cap_bytes} bytes was exceeded ({peak} bytes resident at peak)")


class World:
    def __init__(self, plan: Mapping[str, Any], binding: Mapping[str, Any], *, resolver: Callable[[DeviceProfile], torch.device] = resolve_device) -> None:
        self.plan, self.binding, self._resolver = plan, binding, resolver
        self.bundle_dir = Path(binding["paths"]["bundle"])
        self.evaluation_root = Path(binding["paths"]["evaluation_root"])
        self.run_root = Path(binding["paths"]["run_root"])
        self.backup_dir = Path(binding["paths"]["backup_dir"])
        self.host_memory_cap = int(binding["resources"]["host_memory_cap_bytes"])
        self.device_memory_cap = int(binding["resources"]["device_memory_cap_bytes"])
        self.limits_applied: dict[str, Any] | None = None

    # -- verified inputs --------------------------------------------------------------------------------
    @cached_property
    def bundle(self) -> dict[str, Any]:
        verified = verify_bundle(self.bundle_dir)
        if verified["manifest"]["manifest_sha256"] != self.plan["bundle"]["manifest_sha256"]:
            raise ManifestError("the bundle on disk is not the bundle this plan was made for")
        if verified["protocol"].protocol_id != self.plan["bundle"]["protocol_id"]:
            raise ManifestError("the protocol on disk is not the one this plan froze")
        return verified

    def verify_inputs(self) -> None:
        """Verify every bundle hash, the plan binding and the tokenizer identity (each raises ManifestError on a mismatch)."""
        self.bundle
        self.tokenizer

    @property
    def protocol(self) -> Protocol:
        return self.bundle["protocol"]

    @property
    def loaded(self):
        return self.bundle["prepared"]["loaded"]

    @property
    def prepared_dir(self) -> Path:
        return self.bundle["prepared"]["directory"]

    @property
    def assets_dir(self) -> Path:
        return self.bundle_dir / "model-assets"

    @cached_property
    def tokenizer(self) -> WordPieceTokenizer:
        tokenizer = WordPieceTokenizer(self.assets_dir)
        if tokenizer.fingerprint != self.plan["model_assets"]["tokenizer_fingerprint"]:
            raise ManifestError("the tokenizer files differ from the ones this plan froze")
        return tokenizer

    @cached_property
    def policy(self) -> PairEncodingPolicy:
        training = self.plan["training"]
        return PairEncodingPolicy(training["max_pair_tokens"], training["max_query_tokens"])

    @cached_property
    def inputs(self) -> TrainingInputs:
        """Train pairs and the validation set from the verified prepared directory (never anything from the test side)."""
        return load_training_inputs(self.bundle_dir / "prepared" / "prepared-data-manifest.json", self.bundle_dir)

    @property
    def train_pairs(self):
        return self.inputs.train_pairs

    @property
    def pairs_identity(self) -> str:
        return self.inputs.pairs_identity

    @property
    def validation_candidates(self) -> list[dict[str, Any]]:
        return self.inputs.validation_candidates

    @property
    def validation(self) -> ValidationSet:
        return self.inputs.validation

    def validation_queries(self) -> list:
        return self.inputs.validation_queries

    # -- device and models ------------------------------------------------------------------------------
    @cached_property
    def profile(self) -> DeviceProfile:
        return load_profile(self.plan["profile"]["name"])

    @cached_property
    def device(self) -> torch.device:
        """The explicitly requested device; an unavailable one raises PrerequisiteError (never a CPU fallback)."""
        device = self._resolver(self.profile)
        fingerprint = backend_fingerprint(self.profile, device)
        planned = self.plan["backend"]["fingerprint"] or {}
        for key in ("profile", "device", "torch", "machine", "platform", "python", "tf32"):
            if key in planned and planned[key] != fingerprint.get(key):
                raise PrerequisiteError(f"the {key} differs from the doctor's calibration ({planned[key]!r} then, {fingerprint.get(key)!r} now); recalibrate")
        total = fingerprint.get("total_memory_bytes") or fingerprint.get("recommended_max_memory_bytes")
        self.limits_applied = apply_memory_cap(device, self.device_memory_cap, total)
        return device

    def train_config(self, entry: Mapping[str, Any]) -> TrainConfig:
        shared = self.plan["training"]["shared"]
        return TrainConfig(
            recipe_id=f"{entry['recipe_id']}-seed{entry['seed']}", mode=entry["mode"], seed=entry["seed"], epochs=entry["epochs"],
            learning_rate=entry["learning_rate"], weight_decay=shared["weight_decay"], beta1=shared["beta1"], beta2=shared["beta2"], eps=shared["eps"],
            max_grad_norm=shared["max_grad_norm"], warmup_fraction=shared["warmup_fraction"], effective_batch=shared["effective_batch"],
            microbatch=self.plan["training"]["microbatch"], dropout=shared["dropout"], head_seed=self.protocol.head_seed, pos_weight=shared["pos_weight"],
            validation_batch=shared["validation_batch"])

    def new_model(self, mode: str) -> CrossEncoder:
        """A fresh model: the pinned encoder re-loaded from disk and the shared, seeded scoring head. Never a diagnostic weight."""
        model = CrossEncoder(load_encoder(self.assets_dir), dropout=self.plan["training"]["shared"]["dropout"], head_seed=self.protocol.head_seed)
        model.set_encoder_frozen(mode == "frozen")
        return model

    def identities(self) -> dict[str, str]:
        return {"pairs_manifest": self.pairs_identity, "tokenizer": self.tokenizer.fingerprint, "model": self.plan["model_assets"]["manifest_sha256"],
                "plan": "plan-" + self.plan["manifest_sha256"][:32], "code": code_identity()["source_snapshot_sha256"]}

    # -- the test side ----------------------------------------------------------------------------------
    def evaluation(self, ctx) -> dict[str, Any]:
        """Verified test inputs. Refuses unless `select` has committed: nothing about the test split is read before the lock."""
        ctx.dependency_manifest("select")
        manifest_path = self.evaluation_root / EVALUATION_MANIFEST
        evaluation = load_evaluation(manifest_path, self.evaluation_root, self.loaded)
        manifest = evaluation["manifest"]
        if manifest["manifest_sha256"] != self.plan["data"]["evaluation_commitment_sha256"]:
            raise ManifestError("the evaluation inputs are not the ones committed to by the plan")
        seal = load_seal(self.prepared_dir / "initial-seal.json", 1)
        if seal["manifest_sha256"] != self.plan["data"]["initial_seal_sha256"]:
            raise ManifestError("the initial seal is not the one this plan froze")
        checks = {"test_queries_sha256": manifest["queries"]["sha256"], "qrels_v1_sha256": manifest["qrels_v1"]["sha256"],
                  "candidates_manifest_sha256": manifest["candidates"]["manifest_sha256"], "test_groups_sha256": manifest["test_groups_sha256"]}
        for key, actual in checks.items():
            if seal[key] != actual:
                raise ManifestError(f"the sealed {key} does not match the evaluation inputs; the test side changed after sealing")
        base = evaluation["directory"]
        _, candidates = load_candidates(base / manifest["candidates"]["path"], self.loaded)
        bank = None
        if seal["bank_manifest_sha256"] is not None:
            bank_manifest, bank_qrels = load_bank(base / "bank" / "judgment-bank-manifest.json", self.loaded, evaluation["queries"])
            if bank_manifest["manifest_sha256"] != seal["bank_manifest_sha256"]:
                raise ManifestError("the judgment bank is not the one sealed with the initial test inputs")
            bank = {"manifest": bank_manifest, "qrels": list(bank_qrels)}
        return {**evaluation, "seal": seal, "candidates": candidates, "bank": bank}
