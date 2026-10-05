"""Validation-only model and default-mode selection (docs/10 §6). Pure functions, standard library only.

The rule is fixed in the protocol before training and uses *validation* evidence alone; test results never
choose a model, a recipe or the default mode.

1. Within each (recipe, seed), pick the completed-epoch checkpoint with the highest validation group-macro
   MRR@10, then the highest group-macro nDCG@10, then the earliest epoch.
2. A recipe qualifies only if BOTH its mean over the declared seeds AND its designated deployment seed gain at
   least `min_group_mrr_gain` MRR over BM25 with a non-negative nDCG change, and the deployment seed's measured
   target-device p95 latency is within `max_p95_ms`.
3. Among qualifying recipes choose the highest mean MRR, then mean nDCG, then lower deployment-seed p95, then
   the stable recipe ID. If none qualifies the default is BM25.

A failed or missing required seed, or missing latency evidence, *blocks* the decision instead of selecting
among the survivors. The deployment seed is a declaration, never the seed with the best score.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from devsearch.artifacts.canonical import digest_of
from devsearch.errors import DevSearchError
from devsearch.evaluation.protocol import Protocol

TOLERANCE = 1e-12            # float noise guard for threshold comparisons


class SelectionBlocked(DevSearchError):
    """The selection cannot be made honestly: a required run or latency measurement is missing or incomplete."""


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    group_mrr: float
    group_ndcg: float
    checkpoint_id: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class RunResult:
    recipe_id: str
    seed: int
    epochs: tuple[EpochResult, ...]          # completed epochs, 1..E in order


def best_epoch(run: RunResult) -> EpochResult:
    if not run.epochs:
        raise SelectionBlocked(f"run {run.recipe_id}/{run.seed} has no completed epoch")
    return min(run.epochs, key=lambda e: (-e.group_mrr, -e.group_ndcg, e.epoch))


def _check_run(protocol: Protocol, key: tuple[str, int], run: RunResult | None) -> RunResult:
    if run is None:
        raise SelectionBlocked(f"required run {key[0]}/seed {key[1]} is missing; a partial matrix is not a comparison")
    if (run.recipe_id, run.seed) != key:
        raise SelectionBlocked(f"run result {run.recipe_id}/{run.seed} is filed under {key[0]}/{key[1]}")
    if [e.epoch for e in run.epochs] != list(range(1, protocol.epochs + 1)):
        raise SelectionBlocked(
            f"run {key[0]}/seed {key[1]} completed epochs {[e.epoch for e in run.epochs]}; the protocol requires 1..{protocol.epochs}")
    for epoch in run.epochs:
        for value in (epoch.group_mrr, epoch.group_ndcg):
            if not (value == value) or value in (float("inf"), float("-inf")):
                raise SelectionBlocked(f"run {key[0]}/seed {key[1]} epoch {epoch.epoch} has a non-finite validation metric")
    return run


def _quality(gain_mrr: float, change_ndcg: float, protocol: Protocol) -> bool:
    return gain_mrr >= protocol.min_group_mrr_gain - TOLERANCE and change_ndcg >= -TOLERANCE


def select(
    protocol: Protocol, *, baseline: Mapping[str, float], runs: Mapping[tuple[str, int], RunResult | None],
    latency_p95_ms: Mapping[str, float | None], plan_id: str,
) -> dict[str, Any]:
    """The immutable validation-selection record. `latency_p95_ms[recipe]` is the deployment seed's measured p95."""
    for name in ("group_mrr", "group_ndcg"):
        if name not in baseline or baseline[name] != baseline[name]:
            raise SelectionBlocked(f"the BM25 validation baseline lacks a finite {name}")
    checked = {key: _check_run(protocol, key, runs.get(key)) for key in protocol.run_ids()}
    recipes: list[dict[str, Any]] = []
    for recipe in protocol.recipes:
        per_seed = []
        for seed in protocol.seeds:
            run = checked[(recipe.recipe_id, seed)]
            chosen = best_epoch(run)
            per_seed.append({
                "seed": seed, "best_epoch": chosen.epoch, "checkpoint_id": chosen.checkpoint_id,
                "checkpoint_sha256": chosen.checkpoint_sha256, "group_mrr": chosen.group_mrr, "group_ndcg": chosen.group_ndcg,
                "mrr_gain": chosen.group_mrr - baseline["group_mrr"], "ndcg_change": chosen.group_ndcg - baseline["group_ndcg"],
                "epochs": [{"epoch": e.epoch, "group_mrr": e.group_mrr, "group_ndcg": e.group_ndcg} for e in run.epochs],
            })
        mean_mrr = sum(s["group_mrr"] for s in per_seed) / len(per_seed)
        mean_ndcg = sum(s["group_ndcg"] for s in per_seed) / len(per_seed)
        deployment = next(s for s in per_seed if s["seed"] == protocol.deployment_seed)
        p95 = latency_p95_ms.get(recipe.recipe_id)
        mean_ok = _quality(mean_mrr - baseline["group_mrr"], mean_ndcg - baseline["group_ndcg"], protocol)
        deployment_ok = _quality(deployment["mrr_gain"], deployment["ndcg_change"], protocol)
        recipes.append({
            "recipe_id": recipe.recipe_id, "mode": recipe.mode, "learning_rate": recipe.learning_rate, "seeds": per_seed,
            "mean_group_mrr": mean_mrr, "mean_group_ndcg": mean_ndcg, "mean_mrr_gain": mean_mrr - baseline["group_mrr"],
            "mean_ndcg_change": mean_ndcg - baseline["group_ndcg"], "mean_quality_ok": mean_ok,
            "deployment_seed_quality_ok": deployment_ok, "deployment_p95_ms": p95, "latency_evidence": p95 is not None,
        })
    for entry in recipes:
        if entry["mean_quality_ok"] and entry["deployment_seed_quality_ok"] and entry["deployment_p95_ms"] is None:
            raise SelectionBlocked(f"recipe {entry['recipe_id']} meets the quality gates but has no latency measurement; selection is blocked")
        entry["latency_ok"] = entry["deployment_p95_ms"] is not None and entry["deployment_p95_ms"] <= protocol.max_p95_ms
        entry["qualifies"] = entry["mean_quality_ok"] and entry["deployment_seed_quality_ok"] and entry["latency_ok"]
    qualifying = [e for e in recipes if e["qualifies"]]
    qualifying.sort(key=lambda e: (-e["mean_group_mrr"], -e["mean_group_ndcg"], e["deployment_p95_ms"], e["recipe_id"]))
    if qualifying:
        winner = qualifying[0]
        deployment = next(s for s in winner["seeds"] if s["seed"] == protocol.deployment_seed)
        decision = {"default_mode": "rerank", "recipe_id": winner["recipe_id"], "seed": protocol.deployment_seed,
                    "checkpoint_id": deployment["checkpoint_id"], "checkpoint_sha256": deployment["checkpoint_sha256"],
                    "reason": "qualified under the predeclared validation gates"}
    else:
        decision = {"default_mode": "baseline", "recipe_id": None, "seed": None, "checkpoint_id": None, "checkpoint_sha256": None,
                    "reason": "no recipe met every predeclared validation gate; BM25 stays the default"}
    record = {
        "schema_version": 1, "kind": "validation-selection", "plan_id": plan_id, "protocol_id": protocol.protocol_id,
        "baseline_validation": {"group_mrr": baseline["group_mrr"], "group_ndcg": baseline["group_ndcg"]},
        "thresholds": {"min_group_mrr_gain": protocol.min_group_mrr_gain, "max_p95_ms": protocol.max_p95_ms,
                       "ndcg_change_must_be_nonnegative": True, "deployment_seed": protocol.deployment_seed},
        "recipes": recipes, "decision": decision,
    }
    return {**record, "selection_id": "selection-" + digest_of(record)[:32]}
