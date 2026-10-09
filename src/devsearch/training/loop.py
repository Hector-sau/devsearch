"""Native PyTorch training loop with epoch-boundary recovery (docs/03 §2, docs/12 §3).

Everything that decides what the model learns is written out here rather than hidden in a trainer:

* Per epoch the pair order is a deterministic permutation seeded by `(seed, epoch)`.
* Micro-batches of `microbatch` pairs are accumulated into groups of `accumulation_steps`. Each
  micro-batch contributes `sum(per-pair loss) / group_pair_count`, so a short last group is divided
  by its **actual** size, never by the planned size. Gradients are clipped once per group, then the
  optimizer and scheduler step once and gradients are cleared.
* The LR multiplier is a pure function: linear warmup over `ceil(warmup_fraction * planned_updates)`
  updates, then linear decay to zero.
* Validation (frozen candidates, group-macro MRR@10 then nDCG@10) runs after every complete epoch;
  the earlier epoch wins ties. Checkpoints are written only at epoch boundaries (plus an epoch-zero
  checkpoint) and `resume` restores model, optimizer, scheduler, RNG and counters, then *replays* the
  interrupted epoch. Mid-epoch stops leave `diagnostic_only` snapshots that resume refuses.
* OOM, non-finite loss/gradient and device errors stop the run with an explicit error. Nothing
  silently changes batch size, sequence length, precision, data or device.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from devsearch.artifacts.canonical import digest_of, pretty_json
from devsearch.artifacts.files import write_atomic
from devsearch.errors import DevSearchError, ExecutionFailure, ManifestError
from devsearch.modeling.batching import collate
from devsearch.modeling.cross_encoder import CrossEncoder
from devsearch.modeling.devices import memory_stats, synchronize
from devsearch.modeling.policy import PairEncodingPolicy, PairTokenizer, encode_pair
from devsearch.training.checkpoint import (
    checkpoint_id, load_checkpoint, read_pointer, save_checkpoint, write_pointer,
)
from devsearch.training.config import TrainConfig
from devsearch.training.data import TrainPair, ValidationSet
from devsearch.training.validation import evaluate_validation


class NonFiniteError(ExecutionFailure):
    """A loss or gradient norm was NaN/infinite; the run stops and keeps its evidence."""


class DeviceMemoryError(ExecutionFailure):
    """The device ran out of memory. The run stops; batch size and length are never changed silently."""


class ResumeMismatchError(DevSearchError):
    """The checkpoint's immutable identity differs from the requested run; start a new run instead."""


def lr_multiplier(step: int, warmup: int, total: int) -> float:
    """LR multiplier before optimizer update number `step` (0-based)."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def plan_updates(pairs: int, microbatch: int, accumulation: int, epochs: int) -> dict[str, int]:
    microbatches = math.ceil(pairs / microbatch)
    groups = math.ceil(microbatches / accumulation)
    return {"microbatches_per_epoch": microbatches, "updates_per_epoch": groups, "planned_updates": epochs * groups}


def epoch_permutation(num_pairs: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator()
    generator.manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(num_pairs, generator=generator).tolist()


@dataclass
class TrainResult:
    status: str                              # completed | stopped | stopped_budget
    completed_epochs: int
    global_step: int
    best_epoch: int | None
    best_checkpoint_id: str | None
    history: list[dict[str, Any]]
    lr_trace: list[float]
    plan: dict[str, int]
    stop_reason: str | None = None
    diagnostic_snapshot: str | None = None
    orders: dict[int, list[int]] = field(default_factory=dict)


def _capture_rng(device: torch.device) -> dict[str, Any]:
    rng: dict[str, Any] = {"torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        rng["mps"] = torch.mps.get_rng_state()
    return rng


def _restore_rng(rng: Mapping[str, Any]) -> None:
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if "cuda" in rng:
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    if "mps" in rng:
        torch.mps.set_rng_state(rng["mps"].cpu())


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, RuntimeError) and ("out of memory" in text or "mps backend out of memory" in text)


def train(
    *, config: TrainConfig, model: CrossEncoder, tokenizer: PairTokenizer, policy: PairEncodingPolicy, pairs: Sequence[TrainPair],
    validation: ValidationSet, device: torch.device, run_dir: Path, identities: Mapping[str, str], resume: bool = False,
    should_stop: Callable[[], bool] | None = None, deadline_monotonic: float | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    epoch_gate: Callable[[int, Sequence[dict[str, Any]]], str | None] | None = None,
) -> TrainResult:
    """Train for `config.epochs` complete epochs (or resume from the last complete one)."""
    emit = on_event or (lambda event: None)
    if not run_dir.is_dir():
        raise ManifestError(f"run directory does not exist: {run_dir}")
    positives = sum(1 for p in pairs if p.label == 1)
    negatives = len(pairs) - positives
    if not pairs or positives == 0 or negatives == 0:
        raise ManifestError("training needs at least one positive and one negative pair")
    if not validation.queries:
        raise ManifestError("validation has no query with a confirmed positive; selection would be undefined")
    pos_weight = negatives / positives if config.pos_weight == "auto" else float(config.pos_weight)
    plan = plan_updates(len(pairs), config.microbatch, config.accumulation_steps, config.epochs)
    warmup = math.ceil(config.warmup_fraction * plan["planned_updates"])
    run_identity = {
        "config_sha256": config.config_sha256, "pairs": str(len(pairs)), "pos_weight": repr(pos_weight),
        "validation": validation.identity, "policy": f"{policy.max_pair_tokens}/{policy.max_query_tokens}", **dict(identities),
    }

    model.set_encoder_frozen(config.mode == "frozen")
    model.to(device)
    parameters = model.trainable_parameters()
    if not parameters:
        raise ManifestError("no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(config.beta1, config.beta2), eps=config.eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, warmup, plan["planned_updates"]))
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(pos_weight, device=device))

    history: list[dict[str, Any]] = []
    lr_trace: list[float] = []
    global_step, start_epoch = 0, 0
    best_key: tuple[float, float] | None = None
    best_epoch: int | None = None
    best_id: str | None = None
    summary_base = {"recipe_id": config.recipe_id, "identity": run_identity}

    def capture(completed: int) -> dict[str, Any]:
        return {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "completed_epoch": completed, "next_epoch": completed, "global_step": global_step, "rng": _capture_rng(device),
            "identity": run_identity, "best_key": list(best_key) if best_key else None, "best_epoch": best_epoch,
            "best_checkpoint_id": best_id, "history": history, "lr_trace": lr_trace,
        }

    def checkpoint_summary(completed: int) -> dict[str, Any]:
        return {"completed_epoch": completed, "next_epoch": completed, "global_step": global_step, "recipe_id": config.recipe_id,
                "identity_sha256": digest_of(run_identity)}

    if resume:
        pointer = read_pointer(run_dir, "latest")
        if pointer is None:
            raise ManifestError("nothing to resume: this run has no checkpoint")
        state, manifest = load_checkpoint(run_dir, pointer["checkpoint_id"], device, require_resumable=True)
        if manifest["sha256"] != pointer["sha256"]:
            raise ManifestError("the latest pointer disagrees with its checkpoint manifest")
        if state["identity"] != run_identity:
            differing = sorted(k for k in set(state["identity"]) | set(run_identity) if state["identity"].get(k) != run_identity.get(k))
            raise ResumeMismatchError(f"checkpoint identity differs ({', '.join(differing)}); a changed recipe needs a new run, not a resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
        global_step, start_epoch = state["global_step"], state["next_epoch"]
        best_key = tuple(state["best_key"]) if state["best_key"] else None
        best_epoch, best_id = state["best_epoch"], state["best_checkpoint_id"]
        history, lr_trace = list(state["history"]), list(state["lr_trace"])
        emit({"event": "resumed", "from_epoch": start_epoch, "global_step": global_step})
    else:
        if read_pointer(run_dir, "latest") is not None:
            raise ManifestError("this run directory already has checkpoints; resume it or use a new directory")
        torch.manual_seed(config.seed)
        zero = save_checkpoint(run_dir, checkpoint_id(0), capture(0), resume_kind="epoch_boundary", summary=checkpoint_summary(0))
        write_pointer(run_dir, "latest", zero)
        emit({"event": "checkpoint_saved", "checkpoint_id": zero["checkpoint_id"], "epoch": 0})

    orders: dict[int, list[int]] = {}
    stop_reason: str | None = None
    snapshot: str | None = None

    def stopping() -> str | None:
        if should_stop is not None and should_stop():
            return "stop_requested"
        if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
            return "deadline"
        return None

    for epoch in range(start_epoch, config.epochs):
        if epoch_gate is not None:
            reason = epoch_gate(epoch, history)         # e.g. "the remaining window cannot hold an epoch + validation + save + reserve"
            if reason is not None:
                emit({"event": "stopped", "reason": reason, "epoch": epoch, "global_step": global_step, "diagnostic_snapshot": None})
                return TrainResult("stopped_budget", epoch, global_step, best_epoch, best_id, history, lr_trace, plan, reason, None, orders)
        order = epoch_permutation(len(pairs), config.seed, epoch)
        orders[epoch] = order
        batches = [order[i:i + config.microbatch] for i in range(0, len(order), config.microbatch)]
        groups = [batches[i:i + config.accumulation_steps] for i in range(0, len(batches), config.accumulation_steps)]
        emit({"event": "epoch_start", "epoch": epoch, "pairs": len(pairs), "updates": len(groups), "microbatches": len(batches)})
        model.train()
        synchronize(device)
        started = time.perf_counter()
        loss_sum, microbatch_count, max_norm = 0.0, 0, 0.0
        first_lr_index = len(lr_trace)
        for group in groups:
            group_total = sum(len(b) for b in group)
            for batch in group:
                reason = stopping()
                if reason is not None:
                    optimizer.zero_grad(set_to_none=True)
                    name = f"diagnostic-epoch{epoch:04d}-step{global_step:06d}"
                    save_checkpoint(run_dir, name, {**capture(epoch), "partial_epoch": epoch, "partial_updates": global_step},
                                    resume_kind="diagnostic_only", summary={**checkpoint_summary(epoch), "note": "mid-epoch diagnostic snapshot; never resumable"})
                    stop_reason, snapshot = reason, name
                    emit({"event": "stopped", "reason": reason, "epoch": epoch, "global_step": global_step, "diagnostic_snapshot": name})
                    status = "stopped_budget" if reason == "deadline" else "stopped"
                    return TrainResult(status, epoch, global_step, best_epoch, best_id, history, lr_trace, plan, stop_reason, snapshot, orders)
                try:
                    encoded = [encode_pair(tokenizer, policy, pairs[i].query_text, pairs[i].passage_text) for i in batch]
                    tensors = {k: v.to(device) for k, v in collate(encoded, tokenizer.pad_id).items()}
                    labels = torch.tensor([float(pairs[i].label) for i in batch], dtype=torch.float32, device=device)
                    logits = model(**tensors)
                    per_pair = loss_fn(logits, labels)
                    group_loss = per_pair.sum() / group_total
                    if not bool(torch.isfinite(group_loss)):
                        raise NonFiniteError(f"non-finite loss at epoch {epoch}, update {global_step}")
                    group_loss.backward()
                except RuntimeError as error:
                    if _is_oom(error):
                        raise DeviceMemoryError(f"out of memory at epoch {epoch}, update {global_step}: {error}") from error
                    raise
                loss_sum += float(per_pair.detach().sum().item())
                microbatch_count += 1
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm))
            if not math.isfinite(norm):
                raise NonFiniteError(f"non-finite gradient norm at epoch {epoch}, update {global_step}")
            max_norm = max(max_norm, norm)
            lr_trace.append(float(optimizer.param_groups[0]["lr"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        synchronize(device)
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        try:
            metrics = evaluate_validation(model, tokenizer, policy, validation, device, config.validation_batch)
        except RuntimeError as error:
            if _is_oom(error):
                raise DeviceMemoryError(f"out of memory during validation after epoch {epoch}: {error}") from error
            raise
        synchronize(device)
        validation_seconds = time.perf_counter() - started
        key = (metrics["group_macro"]["mrr@10"], metrics["group_macro"]["ndcg@10"])
        improved = best_key is None or key > best_key           # strictly better only: the earlier epoch wins ties
        completed = epoch + 1
        if improved:
            best_key, best_epoch, best_id = key, completed, checkpoint_id(completed)
        record = {
            "epoch": completed, "train_loss": loss_sum / len(pairs), "updates": len(groups), "microbatches": microbatch_count,
            "lr_first": lr_trace[first_lr_index], "lr_last": lr_trace[-1], "max_grad_norm_seen": max_norm,
            "validation": metrics, "selected_best": improved, "memory": memory_stats(device),
            "seconds": {"train": train_seconds, "validation": validation_seconds},
        }
        history.append(record)
        started = time.perf_counter()
        manifest = save_checkpoint(run_dir, checkpoint_id(completed), capture(completed), resume_kind="epoch_boundary", summary=checkpoint_summary(completed))
        write_pointer(run_dir, "latest", manifest)
        if improved:
            write_pointer(run_dir, "best", manifest)
        record["seconds"]["save"] = time.perf_counter() - started
        emit({"event": "epoch_end", "epoch": completed, "train_loss": record["train_loss"], "validation": metrics,
              "selected_best": improved, "seconds": record["seconds"], "global_step": global_step})

    write_atomic(run_dir / "training-summary.json", pretty_json({
        **summary_base, "status": "completed", "completed_epochs": config.epochs, "global_step": global_step, "plan": plan,
        "best_epoch": best_epoch, "best_checkpoint_id": best_id, "history": history, "lr_trace": lr_trace,
    }).encode("utf-8"))
    return TrainResult("completed", config.epochs, global_step, best_epoch, best_id, history, lr_trace, plan, None, None, orders)

