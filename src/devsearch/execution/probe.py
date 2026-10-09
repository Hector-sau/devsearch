"""Calibration worker: one isolated process per probe (docs/10 §4, docs/12 §6).

`python -m devsearch.execution.probe <spec.json> <result.json>` runs exactly one probe with the pinned encoder
from the model-assets directory on the profile's explicitly requested device, then writes a result file and
exits. A fresh process per probe keeps device state, allocator fragmentation and a possible out-of-memory
crash away from the doctor. A probe never opens test queries, candidates or labels: its inputs are synthetic
token-id sequences with exact, representative or near-limit lengths drawn from the tokenizer's own vocabulary.

Kinds: `train_probe` (three optimizer updates at the maximum length, full fine-tuning), `train_timing` (warm-up
plus measured updates at representative lengths, per recipe mode), `infer_probe` (one batch at the maximum length),
`infer_timing` (warm-up plus measured inference batches) and `io` (save/load/export/reload timings and parity).
Diagnostic weights are discarded: the doctor reports numbers, never a checkpoint a formal run could start from.
"""

import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from devsearch.artifacts.files import write_atomic
from devsearch.errors import DevSearchError
from devsearch.modeling.assets import load_encoder
from devsearch.modeling.batching import collate
from devsearch.modeling.cross_encoder import CrossEncoder
from devsearch.modeling.devices import (
    backend_fingerprint, load_profile, memory_stats, reset_peak_memory, resolve_device, synchronize,
)
from devsearch.modeling.io import check_parity, export_model, load_model_bundle
from devsearch.modeling.policy import EncodedPair, PairEncodingPolicy
from devsearch.modeling.tokenization import WordPieceTokenizer

KINDS = ("train_probe", "train_timing", "infer_probe", "infer_timing", "io")
WINDOWS, WINDOW_SIZE = 5, 10                 # five timing windows of ten measured units (docs/10 §4)


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "mps backend out of memory" in text or "cuda error: out of memory" in text


def synthetic_pairs(tokenizer: WordPieceTokenizer, policy: PairEncodingPolicy, count: int, *, rng: random.Random, representative: bool) -> list[EncodedPair]:
    """Exact-length pairs from the tokenizer vocabulary. `representative` spreads lengths over 40%..100% of the limit."""
    vocab = tokenizer.vocab()                                                           # the pinned tokenizer's own tokens
    words = sorted(token for token in vocab if token.isalpha() and not token.startswith("[") and not token.startswith("##"))
    if len(words) < 8:
        raise DevSearchError("the tokenizer has too few plain tokens to build probe sequences")
    ids = [vocab[w] for w in words]
    cls, sep = vocab["[CLS]"], vocab["[SEP]"]
    special = tokenizer.pair_special_tokens
    pairs = []
    for _ in range(count):
        total = policy.max_pair_tokens if not representative else rng.randint(int(0.4 * policy.max_pair_tokens), policy.max_pair_tokens)
        query_len = min(policy.max_query_tokens, max(4, total // 4)) if representative else policy.max_query_tokens
        passage_len = max(total - special - query_len, 1)
        query = [rng.choice(ids) for _ in range(query_len)]
        passage = [rng.choice(ids) for _ in range(passage_len)]
        sequence = [cls, *query, sep, *passage, sep]
        types = [0] * (query_len + 2) + [1] * (passage_len + 1)
        pairs.append(EncodedPair(tuple(sequence), tuple(types)))
    return pairs


def _build(spec: dict[str, Any], device: torch.device, mode: str) -> tuple[CrossEncoder, WordPieceTokenizer, PairEncodingPolicy]:
    assets = Path(spec["assets"])
    tokenizer = WordPieceTokenizer(assets)
    policy = PairEncodingPolicy(spec["max_pair_tokens"], spec["max_query_tokens"])
    encoder = load_encoder(assets)
    if policy.max_pair_tokens > encoder.config.max_position_embeddings:
        raise DevSearchError(f"max_pair_tokens {policy.max_pair_tokens} exceeds the encoder's {encoder.config.max_position_embeddings} positions")
    torch.manual_seed(spec["seed"])
    model = CrossEncoder(encoder, dropout=spec.get("dropout", 0.1), head_seed=spec["head_seed"])
    model.set_encoder_frozen(mode == "frozen")
    model.to(device)
    return model, tokenizer, policy


def _peak(device: torch.device) -> dict[str, int]:
    stats = memory_stats(device)
    stats["device_peak_bytes"] = max(stats.get("device_peak_allocated_bytes", 0), stats.get("device_peak_reserved_bytes", 0), stats.get("device_driver_allocated_bytes", 0))
    return stats


def _train_updates(spec: dict[str, Any], device: torch.device, *, mode: str, microbatch: int, warmup: int, measured: int, representative: bool) -> dict[str, Any]:
    model, tokenizer, policy = _build(spec, device, mode)
    effective = spec["effective_batch"]
    if effective % microbatch:
        raise DevSearchError(f"microbatch {microbatch} does not divide the effective batch {effective}")
    steps = effective // microbatch
    parameters = model.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=spec["learning_rate"], weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    rng = random.Random(spec["seed"])
    model.train()
    reset_peak_memory(device)
    durations, losses, norms, peaks = [], [], [], []
    for update in range(warmup + measured):
        pairs = synthetic_pairs(tokenizer, policy, effective, rng=rng, representative=representative)
        labels = [float(i % 2) for i in range(effective)]
        synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step in range(steps):
            chunk = pairs[step * microbatch:(step + 1) * microbatch]
            batch = {k: v.to(device) for k, v in collate(chunk, tokenizer.pad_id).items()}
            target = torch.tensor(labels[step * microbatch:(step + 1) * microbatch], dtype=torch.float32, device=device)
            loss = loss_fn(model(**batch), target).sum() / effective
            if not bool(torch.isfinite(loss)):
                raise DevSearchError(f"non-finite loss at update {update}")
            loss.backward()
            losses.append(float(loss.detach().item()) * effective / microbatch)
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
        if not math.isfinite(norm):
            raise DevSearchError(f"non-finite gradient norm at update {update}")
        norms.append(norm)
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - started
        peaks.append(_peak(device))
        if update >= warmup:
            durations.append(elapsed)
    trainable = sum(p.numel() for p in parameters)
    result: dict[str, Any] = {
        "mode": mode, "microbatch": microbatch, "accumulation_steps": steps, "updates": warmup + measured, "warmup": warmup, "measured": len(durations),
        "microbatches_per_update": steps, "finite": True, "max_grad_norm_seen": max(norms), "trainable_parameters": trainable,
        "memory": {key: max(p.get(key, 0) for p in peaks) for key in peaks[0]},
    }
    if durations:
        result["seconds_per_update"] = {"mean": sum(durations) / len(durations), "min": min(durations), "max": max(durations)}
        result["windows"] = _windows(durations)
        result["pairs_per_second"] = effective / (sum(durations) / len(durations))
    del model, optimizer
    return result


def _windows(durations: list[float]) -> list[float]:
    size = max(len(durations) // WINDOWS, 1)
    return [sum(durations[i:i + size]) / len(durations[i:i + size]) for i in range(0, min(len(durations), size * WINDOWS), size)]


def _infer(spec: dict[str, Any], device: torch.device, *, batch: int, warmup: int, measured: int, representative: bool) -> dict[str, Any]:
    model, tokenizer, policy = _build(spec, device, "frozen")
    model.eval()
    rng = random.Random(spec["seed"] + 1)
    reset_peak_memory(device)
    durations, peaks = [], []
    with torch.no_grad():
        for index in range(warmup + measured):
            pairs = synthetic_pairs(tokenizer, policy, batch, rng=rng, representative=representative)
            tensors = {k: v.to(device) for k, v in collate(pairs, tokenizer.pad_id).items()}
            synchronize(device)
            started = time.perf_counter()
            logits = model(**tensors)
            synchronize(device)
            values = logits.detach().float().cpu().tolist()                         # host copy: the same synchronization serving pays
            elapsed = time.perf_counter() - started
            if not all(math.isfinite(v) for v in values):
                raise DevSearchError(f"non-finite logit at inference batch {index}")
            peaks.append(_peak(device))
            if index >= warmup:
                durations.append(elapsed)
    result: dict[str, Any] = {"batch": batch, "warmup": warmup, "measured": len(durations), "finite": True,
                              "memory": {key: max(p.get(key, 0) for p in peaks) for key in peaks[0]}}
    if durations:
        result["seconds_per_batch"] = {"mean": sum(durations) / len(durations), "min": min(durations), "max": max(durations)}
        result["windows"] = _windows(durations)
        result["pairs_per_second"] = batch / (sum(durations) / len(durations))
    return result


def _io(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model, tokenizer, policy = _build(spec, device, "full")
    rng = random.Random(spec["seed"] + 2)
    words = sorted(token for token in tokenizer.vocab() if token.isalpha() and not token.startswith("["))
    texts = [(" ".join(rng.sample(words, 3)), " ".join(rng.sample(words, 6))) for _ in range(2)]
    with tempfile.TemporaryDirectory(dir=spec.get("scratch")) as scratch:
        scratch_path = Path(scratch)
        started = time.perf_counter()
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, scratch_path / "state.pt")
        save_seconds = time.perf_counter() - started
        started = time.perf_counter()
        torch.load(scratch_path / "state.pt", map_location=device, weights_only=True)
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        export_model(model, scratch_path / "bundle", tokenizer=tokenizer, policy=policy, provenance={"purpose": "doctor io probe", "diagnostic_weights": True},
                     parity_examples=texts, device=device, batch_size=2)
        export_seconds = time.perf_counter() - started
        started = time.perf_counter()
        loaded = load_model_bundle(scratch_path / "bundle", device=device)
        reload_seconds = time.perf_counter() - started
        parity = check_parity(loaded, device=device, batch_size=2)
        size = sum(p.stat().st_size for p in scratch_path.rglob("*") if p.is_file())
    return {"checkpoint_save_seconds": save_seconds, "checkpoint_load_seconds": load_seconds, "export_seconds": export_seconds,
            "reload_seconds": reload_seconds, "reload_parity": parity, "scratch_bytes": size, "finite": bool(parity["passed"])}


def run_probe(spec: dict[str, Any]) -> dict[str, Any]:
    kind = spec["kind"]
    if kind not in KINDS:
        raise DevSearchError(f"unknown probe kind {kind!r}")
    profile = load_profile(spec["profile"])
    device = resolve_device(profile)                              # explicit; an unavailable device is an error, never a CPU fallback
    fingerprint = backend_fingerprint(profile, device)
    if kind == "train_probe":
        body = _train_updates(spec, device, mode="full", microbatch=spec["microbatch"], warmup=0, measured=spec.get("updates", 3), representative=False)
    elif kind == "train_timing":
        body = _train_updates(spec, device, mode=spec["mode"], microbatch=spec["microbatch"], warmup=spec.get("warmup", 5),
                              measured=spec.get("measured", 50), representative=True)
    elif kind == "infer_probe":
        body = _infer(spec, device, batch=spec["batch"], warmup=0, measured=1, representative=False)
    elif kind == "infer_timing":
        body = _infer(spec, device, batch=spec["batch"], warmup=spec.get("warmup", 5), measured=spec.get("measured", 50), representative=True)
    else:
        body = _io(spec, device)
    return {"ok": True, "kind": kind, "device": str(device), "backend": fingerprint, **body}


def main(argv: list[str]) -> int:
    spec_path, result_path = Path(argv[0]), Path(argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        result = run_probe(spec)
    except BaseException as error:                                 # the parent classifies; this process only reports
        reason = "out_of_memory" if _is_oom(error) else type(error).__name__
        if isinstance(error, KeyboardInterrupt):
            raise
        result = {"ok": False, "kind": spec.get("kind"), "reason": reason, "error": str(error)[:2000],
                  "device_unavailable": type(error).__name__ == "DeviceUnavailable"}
    write_atomic(result_path, (json.dumps(result, sort_keys=True) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
