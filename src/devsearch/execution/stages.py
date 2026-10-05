"""The campaign's stage implementations (docs/10 §2, §6). Each stage calls existing module APIs; none re-implements
training, scoring, search or sealing.

Order and dependencies come from the plan (`plan.stage_graph`); estimates come from the plan too. A stage writes only
into its own attempt directory (or, for training, a persistent run directory that resumes from the last complete epoch)
and returns the names of the files that become its committed outputs.

Test-side material is reachable only from `rank_test` onward, through `World.evaluation`, which refuses until `select`
has committed. Training and selection therefore cannot open a test query, candidate or judgment.
"""

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from devsearch.artifacts.canonical import digest_of, pretty_json, sha256_file
from devsearch.artifacts.identity import code_identity
from devsearch.data.review import build_review_pool, parse_response, postrank_universe, resolve_bank
from devsearch.data.seals import build_final_seal, load_bank, write_bank, write_review_pool
from devsearch.errors import BudgetStop, DeadlineExceeded, Interrupted, ManifestError, PrerequisiteError
from devsearch.evaluation.latency import measure, workload
from devsearch.evaluation.outputs import MANIFEST_NAME as RANKED_MANIFEST, load_ranked_outputs, write_ranked_outputs
from devsearch.evaluation.ranking import bm25_ranked_records, order_by_score, ranked_record, score_with_positives
from devsearch.evaluation.report import build_final_report
from devsearch.evaluation.selection import EpochResult, RunResult, select
from devsearch.execution import acceptance, backup as backup_module
from devsearch.execution.controller import Stage, StageContext, StageOutcome
from devsearch.execution.hostfile import run_checks
from devsearch.execution.plan import UNCERTAINTY_MARGIN, profile_sha256
from devsearch.execution.records import read_json, read_stage_manifest, utc_now
from devsearch.execution.world import World, host_memory_check
from devsearch.contracts import SearchRequest
from devsearch.retrieval.config import SearchConfig
from devsearch.serving.pipeline import SearchPipeline
from devsearch.serving.release import check_smoke, export_release, load_release, read_smoke, record_smoke, verify_release
from devsearch.training.checkpoint import checkpoint_id, load_checkpoint, read_manifest, read_pointer
from devsearch.training.loop import train
from devsearch.modeling.io import check_parity, export_model, load_model_bundle
from devsearch.modeling.policy import LengthError, encode_pair
from devsearch.modeling.rerank import Reranker

SELECTION_FILE = "validation-selection.json"


def _put(ctx: StageContext, name: str, value: Mapping[str, Any]) -> str:
    (ctx.work_dir / name).parent.mkdir(parents=True, exist_ok=True)
    (ctx.work_dir / name).write_text(pretty_json(value), encoding="utf-8")
    return name


def _tree(directory: Path, prefix: str) -> list[str]:
    return sorted(f"{prefix}/{p.relative_to(directory).as_posix()}" for p in directory.rglob("*") if p.is_file())


def _search_config(plan: Mapping[str, Any]) -> SearchConfig:
    search = plan["policies"]["search"]
    return SearchConfig(candidate_k=search["candidate_k"], top_k=search["top_k"], max_candidate_k=search["max_candidate_k"],
                        max_query_chars=search["max_query_chars"], sources=(), versions=())


def _search_toml(plan: Mapping[str, Any]) -> str:
    search = plan["policies"]["search"]
    return (f"schema_version = 1\ncandidate_k = {search['candidate_k']}\ntop_k = {search['top_k']}\nmax_candidate_k = {search['max_candidate_k']}\n"
            f"max_query_chars = {search['max_query_chars']}\n[filters]\nsources = []\n")


class Runner:
    """Builds the stages for one plan. Holds the `World`; stages are methods bound into `Stage` objects."""

    def __init__(self, plan: Mapping[str, Any], world: World) -> None:
        self.plan, self.world = plan, world
        self.protocol = world.protocol
        self.estimates = plan["budget"]["estimates"]["stage_seconds"]
        self.plan_id = "plan-" + plan["manifest_sha256"][:32]

    # -- construction ------------------------------------------------------------------------------------
    def stages(self) -> list[Stage]:
        table: dict[str, Any] = {
            "preflight": self.preflight, "data_check": self.data_check, "baseline_validation": self.baseline_validation,
            "latency": self.latency, "select": self.select, "rank_test": self.rank_test, "pool": self.pool, "review_gate": self.review_gate,
            "seal_final": self.seal_final, "score": self.score, "export_release": self.export_release, "reload_verify": self.reload_verify,
            "backup": self.backup, "summary": self.summary,
        }
        built = []
        for entry in self.plan["stages"]:
            stage_id = entry["stage_id"]
            if stage_id.startswith("train:"):
                function = self._bind(self.train, stage_id)
            elif stage_id.startswith("export:"):
                function = self._bind(self.export_model_stage, stage_id)
            else:
                function = table[stage_id]
            built.append(Stage(stage_id, tuple(entry["needs"]), function, self.estimates.get(stage_id)))
        return built

    @staticmethod
    def _bind(method, stage_id: str):
        return lambda ctx: method(ctx, stage_id)

    def run_entries(self) -> dict[str, Mapping[str, Any]]:
        return {entry["stage_id"]: entry for entry in self.plan["matrix"]}

    # -- preflight and data -------------------------------------------------------------------------------
    def preflight(self, ctx: StageContext) -> StageOutcome:
        world, plan = self.world, self.plan
        ctx.guard()
        running = code_identity()
        if running["source_snapshot_sha256"] != plan["code"]["identity"]["source_snapshot_sha256"]:
            raise ManifestError("the running code is not the code this plan identifies")
        world.verify_inputs()                                   # every bundle hash, the plan binding and the tokenizer identity
        if profile_sha256(world.profile) != plan["profile"]["sha256"]:
            raise ManifestError("the device profile changed after the plan was made")
        offline = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        if offline["HF_HUB_OFFLINE"] != "1":
            raise PrerequisiteError("formal compute runs offline: HF_HUB_OFFLINE=1 must be set before the run starts")
        device = world.device
        required = int((plan["capacity"] or {}).get("bytes") or 0)
        checks = run_checks(ctx.binding, required_free_bytes=required)
        failed = [c for c in checks if c["status"] == "failed"]
        if failed:
            raise PrerequisiteError("; ".join(f"{c['name']}: {c['detail']}" for c in failed))
        window = ctx.remaining_seconds()
        known = plan["budget"]["estimates"]["total_seconds"]
        if known > window:
            raise BudgetStop(f"about {known:.0f}s of known work against {window:.0f}s remaining after the reserve")
        return StageOutcome((_put(ctx, "preflight.json", {
            "checks": checks, "device": str(device), "limits_applied": world.limits_applied, "remaining_seconds": window,
            "estimated_seconds": known, "offline_env": offline, "rehearsal": plan["rehearsal"]}),), {"device": str(device)})

    def data_check(self, ctx: StageContext) -> StageOutcome:
        world, plan = self.world, self.plan
        pairs, validation = world.train_pairs, world.validation
        if len(pairs) != plan["data"]["counts"]["pairs"]:
            raise ManifestError("the number of training pairs differs from the plan")
        longest = 0
        try:
            for pair in pairs:
                longest = max(longest, len(encode_pair(world.tokenizer, world.policy, pair.query_text, pair.passage_text).ids))
            for query in validation.queries:
                for text in query.candidate_texts:
                    longest = max(longest, len(encode_pair(world.tokenizer, world.policy, query.text, text).ids))
        except LengthError as error:
            raise ManifestError(f"a training or validation pair does not fit the length policy ({error}); evidence is never truncated during a run") from error
        positives = sum(p.label for p in pairs)
        return StageOutcome((_put(ctx, "data-check.json", {
            "train_pairs": len(pairs), "positives": positives, "negatives": len(pairs) - positives, "auto_pos_weight": (len(pairs) - positives) / max(positives, 1),
            "validation_queries_scored": len(validation.queries), "validation_queries_without_positive": list(validation.excluded_without_positive),
            "validation_groups": len({q.group_id for q in validation.queries}), "longest_pair_tokens": longest, "limit": world.policy.max_pair_tokens,
            "pairs_manifest_sha256": world.pairs_identity}),), {"train_pairs": len(pairs)})

    def baseline_validation(self, ctx: StageContext) -> StageOutcome:
        world = self.world
        validation = world.validation
        group_of = {q.query_id: q.group_id for q in world.validation_queries()}
        scored = score_with_positives(bm25_ranked_records(world.validation_candidates), {q.query_id: q.positives for q in validation.queries}, group_of)
        record = {k: scored[k] for k in ("group_macro", "query_macro", "queries_scored", "groups_scored", "queries_without_positive")}
        record["group_mrr"], record["group_ndcg"] = record["group_macro"]["mrr@10"], record["group_macro"]["ndcg@10"]
        return StageOutcome((_put(ctx, "baseline-validation.json", record),), {"group_mrr": record["group_mrr"]})

    # -- training and export ------------------------------------------------------------------------------
    def train(self, ctx: StageContext, stage_id: str) -> StageOutcome:
        world = self.world
        entry = self.run_entries()[stage_id]
        config = world.train_config(entry)
        run_dir = ctx.persistent_dir(stage_id.replace(":", "-"))
        resume = read_pointer(run_dir, "latest") is not None
        device = world.device
        model = world.new_model(entry["mode"])
        estimate = self.estimates.get(stage_id)
        per_epoch = (estimate / config.epochs) if estimate else None

        def gate(epoch: int, history: Sequence[Mapping[str, Any]]) -> str | None:
            ctx.guard()                                         # mount, space and deadline are re-checked at every epoch boundary
            host_memory_check(device, world.host_memory_cap)
            seen = [sum(h["seconds"].values()) for h in history]
            expected = (sum(seen) / len(seen)) if seen else per_epoch
            if expected is not None and ctx.remaining_seconds() < expected * UNCERTAINTY_MARGIN:
                return (f"the remaining {ctx.remaining_seconds():.0f}s cannot hold another epoch plus validation and save "
                        f"(about {expected * UNCERTAINTY_MARGIN:.0f}s with margin) and the export/backup reserve")
            return None

        def on_event(event: Mapping[str, Any]) -> None:
            ctx.event(event["event"], **{k: v for k, v in event.items() if k in ("epoch", "global_step", "train_loss", "reason")})
            if event["event"] == "epoch_end":
                host_memory_check(device, world.host_memory_cap)

        def should_stop() -> bool:
            ctx.crash_point("mid_stage")                         # only active under fault injection
            return False

        result = train(config=config, model=model, tokenizer=world.tokenizer, policy=world.policy, pairs=world.train_pairs, validation=world.validation,
                       device=device, run_dir=run_dir, identities=world.identities(), resume=resume, should_stop=should_stop,
                       deadline_monotonic=time.monotonic() + max(ctx.remaining_seconds(), 0), on_event=on_event, epoch_gate=gate)
        if result.status == "stopped_budget":
            raise BudgetStop(result.stop_reason or "the training budget is exhausted; the last complete epoch is kept and resumable")
        if result.status != "completed":
            raise Interrupted(f"training stopped ({result.stop_reason}); resume replays from the last complete epoch")
        epochs = []
        for record in result.history:
            manifest = read_manifest(run_dir, checkpoint_id(record["epoch"]))
            epochs.append({"epoch": record["epoch"], "group_mrr": record["validation"]["group_macro"]["mrr@10"], "group_ndcg": record["validation"]["group_macro"]["ndcg@10"],
                           "checkpoint_id": manifest["checkpoint_id"], "checkpoint_sha256": manifest["sha256"], "train_loss": record["train_loss"],
                           "seconds": record["seconds"], "memory": record["memory"]})
        summary = {"stage_id": stage_id, "recipe_id": entry["recipe_id"], "seed": entry["seed"], "mode": entry["mode"], "status": result.status,
                   "completed_epochs": result.completed_epochs, "global_step": result.global_step, "plan": result.plan, "best_epoch": result.best_epoch,
                   "best_checkpoint_id": result.best_checkpoint_id, "epochs": epochs, "lr_trace_sha256": digest_of(result.lr_trace),
                   "config_sha256": config.config_sha256, "run_dir": run_dir.name}
        return StageOutcome((_put(ctx, "run-summary.json", summary),), {"best_epoch": result.best_epoch, "completed_epochs": result.completed_epochs})

    def _best_model(self, ctx: StageContext, train_stage: str):
        summary = read_json(ctx.dependency_dir(train_stage) / "run-summary.json", "run-summary.json")
        run_dir = ctx.run_root / "work" / train_stage.replace(":", "-")
        epoch = next(e for e in summary["epochs"] if e["epoch"] == summary["best_epoch"])
        state, manifest = load_checkpoint(run_dir, epoch["checkpoint_id"], torch.device("cpu"), require_resumable=False)
        if manifest["sha256"] != epoch["checkpoint_sha256"]:
            raise ManifestError(f"checkpoint {epoch['checkpoint_id']} no longer matches the hash recorded when training finished")
        model = self.world.new_model(summary["mode"])
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        return model, summary, epoch

    def export_model_stage(self, ctx: StageContext, stage_id: str) -> StageOutcome:
        world = self.world
        _, recipe, seed = stage_id.split(":")
        model, summary, epoch = self._best_model(ctx, f"train:{recipe}:{seed}")
        model.to(world.device)
        examples = [(q.text, q.candidate_texts[0]) for q in world.validation.queries[:8]]
        destination = ctx.work_dir / "model"
        manifest = export_model(model, destination, tokenizer=world.tokenizer, policy=world.policy, device=world.device,
                                batch_size=self.plan["training"]["inference_batch"], parity_examples=examples,
                                provenance={"recipe_id": recipe, "seed": int(seed), "best_epoch": summary["best_epoch"], "checkpoint_id": epoch["checkpoint_id"],
                                            "checkpoint_sha256": epoch["checkpoint_sha256"], "plan_id": self.plan_id,
                                            "assets_manifest_sha256": self.plan["model_assets"]["manifest_sha256"], "diagnostic": self.plan["rehearsal"]})
        loaded = load_model_bundle(destination, device=world.device)
        parity = check_parity(loaded, device=world.device, batch_size=self.plan["training"]["inference_batch"])
        if not parity["passed"]:
            raise ManifestError(f"the exported model does not reproduce its recorded logits after reload: {parity}")
        names = _tree(destination, "model")
        names.append(_put(ctx, "export.json", {"model_id": manifest["model_id"], "recipe_id": recipe, "seed": int(seed), "checkpoint_id": epoch["checkpoint_id"],
                                              "checkpoint_sha256": epoch["checkpoint_sha256"], "parity": parity}))
        return StageOutcome(tuple(names), {"model_id": manifest["model_id"]})

    # -- latency, selection ---------------------------------------------------------------------------------
    def _pipeline(self, model_dir: Path | None) -> SearchPipeline:
        reranker = None
        if model_dir is not None:
            reranker = Reranker(load_model_bundle(model_dir, device=self.world.device), device=self.world.device, batch_size=self.plan["training"]["inference_batch"])
        return SearchPipeline(self.world.loaded, _search_config(self.plan), reranker)

    def latency(self, ctx: StageContext) -> StageOutcome:
        world, protocol = self.world, self.protocol
        queries = {q.query_id: q for q in world.validation_queries()}
        chosen = workload(sorted(queries), size=protocol.latency_queries, seed=protocol.latency_seed)
        config = _search_config(self.plan)

        def search_for(pipeline: SearchPipeline, mode: str):
            def search(query_id: str) -> dict[str, Any]:
                query = queries[query_id]
                return pipeline.search(SearchRequest(query=query.text, sources=query.sources, versions=query.versions,
                                                     candidate_k=config.candidate_k, top_k=config.top_k, mode=mode))
            return search
        result: dict[str, Any] = {"workload": chosen, "inference_batch": self.plan["training"]["inference_batch"], "device": str(world.device),
                                  "candidate_k": config.candidate_k, "protocol": "docs/10 §6: warm-up, seeded sequential passes, concurrency 1, no cache"}
        result["baseline"] = measure(search_for(self._pipeline(None), "baseline"), chosen, warmup=protocol.latency_warmup, passes=protocol.latency_passes, seed=protocol.latency_seed)
        p95: dict[str, float | None] = {}
        result["recipes"] = {}
        for recipe in protocol.recipes:
            dependency = ctx.dependency_dir(f"export:{recipe.recipe_id}:{protocol.deployment_seed}")
            measured = measure(search_for(self._pipeline(dependency / "model"), "rerank"), chosen, warmup=protocol.latency_warmup,
                               passes=protocol.latency_passes, seed=protocol.latency_seed)
            result["recipes"][recipe.recipe_id] = measured
            p95[recipe.recipe_id] = measured["p95_ms"] if measured["valid"] else None
        result["p95_ms"] = p95
        return StageOutcome((_put(ctx, "latency.json", result),), {"p95_ms": p95})

    def select(self, ctx: StageContext) -> StageOutcome:
        protocol = self.protocol
        baseline = read_json(ctx.dependency_dir("baseline_validation") / "baseline-validation.json", "baseline-validation.json")
        latency = read_json(ctx.dependency_dir("latency") / "latency.json", "latency.json")
        runs: dict[tuple[str, int], RunResult | None] = {}
        for recipe, seed in protocol.run_ids():
            summary = read_json(ctx.dependency_dir(f"train:{recipe}:{seed}") / "run-summary.json", "run-summary.json")
            runs[(recipe, seed)] = RunResult(recipe, seed, tuple(
                EpochResult(e["epoch"], e["group_mrr"], e["group_ndcg"], e["checkpoint_id"], e["checkpoint_sha256"]) for e in summary["epochs"]))
        record = select(protocol, baseline={"group_mrr": baseline["group_mrr"], "group_ndcg": baseline["group_ndcg"]}, runs=runs,
                        latency_p95_ms=latency["p95_ms"], plan_id=self.plan_id)
        record["latency_sha256"] = sha256_file(ctx.dependency_dir("latency") / "latency.json")
        return StageOutcome((_put(ctx, SELECTION_FILE, record),), {"default_mode": record["decision"]["default_mode"], "selection_id": record["selection_id"]})

    # -- the locked comparison on the test side ---------------------------------------------------------------
    def _selection(self, ctx: StageContext) -> dict[str, Any]:
        return read_json(ctx.dependency_dir("select") / SELECTION_FILE, SELECTION_FILE)

    def rank_test(self, ctx: StageContext) -> StageOutcome:
        world, protocol = self.world, self.protocol
        evaluation = world.evaluation(ctx)                     # refuses unless `select` committed
        selection = self._selection(ctx)
        candidates = evaluation["candidates"]
        queries = {q.query_id: q for q in evaluation["queries"]}
        texts = {c.chunk_id: c.text for c in world.loaded.corpus.chunks}
        systems: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {"bm25": ({"role": "baseline"}, bm25_ranked_records(candidates))}
        by_run = {(recipe["recipe_id"], entry["seed"]): entry for recipe in selection["recipes"] for entry in recipe["seeds"]}
        for recipe, seed in protocol.run_ids():
            run = by_run[(recipe, seed)]
            dependency = ctx.dependency_dir(f"export:{recipe}:{seed}")
            loaded = load_model_bundle(dependency / "model", device=world.device)
            reranker = Reranker(loaded, device=world.device, batch_size=self.plan["training"]["inference_batch"])
            records = []
            for record in candidates:                          # one all-or-nothing score per query; a failure discards the whole system
                ids = [c["chunk_id"] for c in record["candidates"]]
                try:
                    scores = reranker.score(queries[record["query_id"]].text, [texts[i] for i in ids], should_stop=lambda: ctx.remaining_seconds() <= 0)
                except DeadlineExceeded as error:
                    raise BudgetStop(f"the window closed while ranking {recipe}/{seed}; partial rankings were discarded") from error
                records.append(ranked_record(record["query_id"], record["candidate_set_id"], order_by_score(ids, scores)))
            systems[f"{recipe}-seed{seed}"] = ({"role": "locked", "recipe_id": recipe, "seed": seed, "checkpoint_id": run["checkpoint_id"],
                                                "checkpoint_sha256": run["checkpoint_sha256"], "model_id": loaded.model_id}, records)
        directory = ctx.work_dir / "ranked"
        directory.mkdir()
        write_ranked_outputs(directory, systems, candidates, candidates_manifest_sha256=evaluation["manifest"]["candidates"]["manifest_sha256"],
                             selection_id=selection["selection_id"], protocol_id=protocol.protocol_id)
        return StageOutcome(tuple(_tree(directory, "ranked")), {"systems": sorted(systems)})

    def _ranked(self, ctx: StageContext, evaluation: Mapping[str, Any]):
        directory = ctx.dependency_dir("rank_test") / "ranked"
        return load_ranked_outputs(directory / RANKED_MANIFEST, evaluation["candidates"]), directory

    def pool(self, ctx: StageContext) -> StageOutcome:
        evaluation = self.world.evaluation(ctx)
        (manifest, ranked), _ = self._ranked(ctx, evaluation)
        selection = self._selection(ctx)
        universe = postrank_universe(ranked, evaluation["qrels"], depth=10)
        salt = digest_of(["postrank-salt", self.plan_id, selection["selection_id"]])
        pool = build_review_pool("postrank_pool_v1", salt, universe)
        directory = ctx.work_dir / "review"
        directory.mkdir()
        pool_manifest = write_review_pool(directory, pool, self.world.loaded, {q.query_id: q for q in evaluation["queries"]})
        names = _tree(directory, "review")
        names.append(_put(ctx, "pool.json", {"pool_id": pool.pool_id, "items": len(pool.items), "ranked_outputs_manifest_sha256": manifest["manifest_sha256"],
                                            "review_pool_manifest_sha256": pool_manifest["manifest_sha256"], "pool_depth": 10}))
        return StageOutcome(tuple(names), {"items": len(pool.items)})

    def review_gate(self, ctx: StageContext) -> StageOutcome:
        world = self.world
        evaluation = world.evaluation(ctx)
        pool_dir = ctx.dependency_dir("pool")
        pool_info = read_json(pool_dir / "pool.json", "pool.json")
        reviews = ctx.options.get("reviews")
        if not reviews:
            return StageOutcome((_put(ctx, "review-request.json", {
                "protocol": "postrank_pool_v1", "plan_id": self.plan_id, "selection_id": self._selection(ctx)["selection_id"],
                "ranked_outputs_manifest_sha256": pool_info["ranked_outputs_manifest_sha256"], "review_pool_manifest_sha256": pool_info["review_pool_manifest_sha256"],
                "pool_id": pool_info["pool_id"], "missing_reviewed_records": pool_info["items"], "request_file": "review/review-request.jsonl",
                "instructions": "judge every item blind (query, filters, passage, source evidence only), then run with --resume --reviews <responses.jsonl>"}),),
                {"waiting_for": pool_info["items"]}, status="awaiting_review")
        (manifest, ranked), _ = self._ranked(ctx, evaluation)
        selection = self._selection(ctx)
        salt = digest_of(["postrank-salt", self.plan_id, selection["selection_id"]])
        pool = build_review_pool("postrank_pool_v1", salt, postrank_universe(ranked, evaluation["qrels"], depth=10))
        if pool.pool_id != pool_info["pool_id"]:
            raise ManifestError("the review pool rebuilt from the locked outputs differs from the pool that was sent for review")
        chunks = {c.chunk_id: c for c in world.loaded.corpus.chunks}
        lengths = {item.item_id: len(chunks[item.chunk_id].text) for item in pool.items}
        from devsearch.artifacts.files import read_jsonl
        responses = [parse_response(record, label, lengths) for label, record in read_jsonl(Path(reviews), "reviews")]
        bank = resolve_bank(pool, responses, world.loaded)
        directory = ctx.work_dir / "bank"
        directory.mkdir()
        write_bank(directory, pool, bank, evaluation["qrels"], world.loaded, list(evaluation["queries"]))
        return StageOutcome(tuple(_tree(directory, "bank")), {"items": len(bank.qrels)})

    def seal_final(self, ctx: StageContext) -> StageOutcome:
        world = self.world
        evaluation = world.evaluation(ctx)
        (manifest, ranked), directory = self._ranked(ctx, evaluation)
        outputs = {name: (entry["sha256"], ranked[name]) for name, entry in manifest["systems"].items()}
        if evaluation["bank"] is not None:
            bank_manifest, bank_qrels = evaluation["bank"]["manifest"], evaluation["bank"]["qrels"]
        else:
            bank_manifest, bank_qrels_loaded = load_bank(ctx.dependency_dir("review_gate") / "bank" / "judgment-bank-manifest.json", world.loaded, list(evaluation["queries"]))
            bank_qrels = list(bank_qrels_loaded)
        selection = self._selection(ctx)
        seal = build_final_seal(initial_seal=evaluation["seal"], selection_id=selection["selection_id"], outputs=outputs, candidate_records=evaluation["candidates"],
                                bank_manifest=bank_manifest, bank_qrels=bank_qrels, qrels_v1=evaluation["qrels"])
        names = [_put(ctx, "test-seal-final.json", seal)]
        return StageOutcome(tuple(names), {"coverage": seal["coverage"]})

    def score(self, ctx: StageContext) -> StageOutcome:
        world = self.world
        evaluation = world.evaluation(ctx)
        (manifest, ranked), _ = self._ranked(ctx, evaluation)
        selection = self._selection(ctx)
        seal = read_json(ctx.dependency_dir("seal_final") / "test-seal-final.json", "test-seal-final.json")
        if seal["selection_id"] != selection["selection_id"] or sorted(seal["systems"]) != sorted(ranked):
            raise ManifestError("the final seal does not bind the locked selection and systems")
        if evaluation["bank"] is not None:
            bank_manifest, qrels = evaluation["bank"]["manifest"], evaluation["bank"]["qrels"]
        else:
            bank_manifest, loaded_qrels = load_bank(ctx.dependency_dir("review_gate") / "bank" / "judgment-bank-manifest.json", world.loaded, list(evaluation["queries"]))
            qrels = list(loaded_qrels)
        if seal["qrels_v2_sha256"] != bank_manifest["files"]["qrels-v2.jsonl"]["sha256"]:
            raise ManifestError("the final seal and the judgment bank disagree about qrels-v2")
        groups = {q.query_id: q.group_id for q in evaluation["queries"]}
        origins: dict[str, int] = {}
        for query in evaluation["queries"]:
            origins[query.origin] = origins.get(query.origin, 0) + 1
        provenance = {"query_origins": origins, "reviewers": dict(bank_manifest["reviewers"]), "diagnostic_rehearsal": bool(self.plan["rehearsal"]),
                      "evaluation_protocol": evaluation["manifest"]["protocol"], "corpus_id": world.loaded.corpus.corpus_id, "index_id": world.loaded.index_id}
        built = build_final_report(
            plan_id=self.plan_id, protocol=self.protocol, selection=selection, systems=ranked, system_meta={n: manifest["systems"][n]["meta"] for n in ranked},
            qrels=qrels, group_of=groups, provenance=provenance,
            seals={"initial": evaluation["seal"]["manifest_sha256"], "final": seal["manifest_sha256"], "bank": bank_manifest["manifest_sha256"],
                   "ranked_outputs": manifest["manifest_sha256"]})
        (ctx.work_dir / "final-report.json").write_text(pretty_json(built.report), encoding="utf-8")
        (ctx.work_dir / "per-query.jsonl").write_bytes(built.per_query)
        (ctx.work_dir / "report.md").write_text(built.markdown, encoding="utf-8")
        return StageOutcome(("final-report.json", "per-query.jsonl", "report.md"), {"final_report_id": built.report["final_report_id"],
                                                                                     "default_mode": built.report["default_mode"]})

    # -- release, reload, backup ----------------------------------------------------------------------------
    def export_release(self, ctx: StageContext) -> StageOutcome:
        world = self.world
        selection = self._selection(ctx)
        decision = selection["decision"]
        model_dir = ctx.dependency_dir(f"export:{decision['recipe_id']}:{decision['seed']}") / "model" if decision["default_mode"] == "rerank" else None
        (ctx.work_dir / "search.toml").write_text(_search_toml(self.plan), encoding="utf-8")
        pipeline = self._pipeline(model_dir)
        smoke_queries = [{"query": q.text, "sources": list(q.sources), "versions": [list(v) for v in q.versions]} for q in world.validation_queries()[:8]]
        smoke = record_smoke(pipeline, smoke_queries, _search_config(self.plan))
        prepared = world.prepared_dir
        manifest = export_release(
            ctx.work_dir / "release", status="evaluated", default_mode=decision["default_mode"], corpus_dir=prepared / "corpus", index_dir=prepared / "index",
            model_dir=model_dir, search_config_path=ctx.work_dir / "search.toml", report_dir=ctx.dependency_dir("score"), smoke=smoke,
            provenance={"plan_id": self.plan_id, "selection_id": selection["selection_id"], "rehearsal": self.plan["rehearsal"],
                        "inference_batch": self.plan["training"]["inference_batch"],
                        "validation_only_smoke": True})
        names = _tree(ctx.work_dir / "release", "release")
        return StageOutcome(tuple(names), {"release_id": manifest["release_id"], "status": manifest["status"], "default_mode": manifest["default_mode"]})

    def reload_verify(self, ctx: StageContext) -> StageOutcome:
        world = self.world
        release_dir = ctx.dependency_dir("export_release") / "release"
        checks: dict[str, Any] = {}
        manifest = verify_release(release_dir, allow_candidate=False)
        checks["hashes_and_activation_rules"] = {"status": "passed", "release_id": manifest["release_id"]}

        def factory(path: Path):
            return Reranker(load_model_bundle(path, device=world.device), device=world.device, batch_size=self.plan["training"]["inference_batch"])
        loaded = load_release(release_dir, allow_candidate=False, reranker_factory=factory)
        if loaded.pipeline.reranker is not None:
            parity = check_parity(loaded.pipeline.reranker.loaded, device=world.device, batch_size=self.plan["training"]["inference_batch"])
            checks["model_parity_after_reload"] = {"status": "passed" if parity["passed"] else "failed", **parity}
            checks["rerank_ready"] = {"status": "passed" if loaded.pipeline.self_check() else "failed"}
        else:
            checks["model_parity_after_reload"] = {"status": "not_applicable", "detail": "the release defaults to BM25 and ships no model"}
        smoke = check_smoke(loaded.pipeline, read_smoke(release_dir), loaded.search_config)
        checks["smoke_replay"] = {"status": "passed" if smoke["passed"] else "failed", **smoke}
        checks["api_smoke"] = self._api_smoke(loaded)
        failed = [name for name, check in checks.items() if check["status"] == "failed"]
        _put(ctx, "reload-report.json", {"release_id": manifest["release_id"], "checks": checks, "failed": failed})
        if failed:
            raise ManifestError(f"the exported release does not reload cleanly: {failed}")
        return StageOutcome(("reload-report.json",), {"release_id": manifest["release_id"], "api_smoke": checks["api_smoke"]["status"]})

    def _api_smoke(self, loaded) -> dict[str, Any]:
        try:
            from fastapi.testclient import TestClient

            from devsearch.serving.api import SearchService, create_app
        except ImportError as error:
            return {"status": "unverified", "detail": f"the serve extra is not installed ({error.name}); the API was not exercised"}
        manifest = loaded.manifest
        service = SearchService(loaded.pipeline, deadline_s=30.0, release={"status": manifest["status"], "release_id": manifest["release_id"],
                                                                           "default_mode": manifest["default_mode"]})
        try:
            client = TestClient(create_app(service))
            health = client.get("/health").json()
            ready = client.get("/ready", params={"mode": "baseline"}).status_code
            query = self.world.validation_queries()[0].text
            baseline = client.post("/search", json={"query": query, "mode": "baseline"})
            rerank_status = None
            if loaded.pipeline.reranker is not None:
                response = client.post("/search", json={"query": query, "mode": "rerank"})
                rerank_status = (response.status_code, response.json().get("status"))
            ok = health["alive"] and ready == 200 and baseline.status_code == 200 and (rerank_status is None or rerank_status == (200, "ok"))
            return {"status": "passed" if ok else "failed", "health": health["release"]["status"], "ready_baseline": ready, "baseline_status": baseline.status_code,
                    "rerank": rerank_status, "transport": "in-process ASGI test client (loopback sockets are checked by the rehearsal)"}
        finally:
            service.close()

    def backup(self, ctx: StageContext) -> StageOutcome:
        world = self.world
        staging = ctx.work_dir / "evidence"
        staging.mkdir()
        manifests = staging / "stage-manifests"
        manifests.mkdir()
        for stage in self.plan["stages"]:
            path = ctx.run_root / "stages" / stage["stage_id"] / "stage-manifest.json"
            if path.is_file():
                (manifests / f"{stage['stage_id'].replace(':', '_')}.json").write_bytes(path.read_bytes())
        (staging / "events.jsonl").write_bytes((ctx.run_root / "events.jsonl").read_bytes())
        sources: dict[str, Path] = {
            "release": ctx.dependency_dir("export_release") / "release", "evaluation": ctx.dependency_dir("score"), "selection": ctx.dependency_dir("select"),
            "reload": ctx.dependency_dir("reload_verify"), "evidence": staging,
        }
        sources["plan"] = ctx.campaign.plan_dir                  # the immutable plan and every host-binding revision
        for stage_id in self.run_entries():
            summary = read_json(ctx.dependency_dir(stage_id) / "run-summary.json", "run-summary.json")
            run_dir = ctx.run_root / "work" / stage_id.replace(":", "-") / "checkpoints"
            for suffix in (".pt", ".json"):
                sources[f"best-checkpoints/{stage_id.replace(':', '_')}-{summary['best_checkpoint_id']}{suffix}"] = run_dir / f"{summary['best_checkpoint_id']}{suffix}"
        destination = world.backup_dir / self.plan_id
        receipt = backup_module.backup(sources, destination, plan_id=self.plan_id)
        backup_module.verify_backup(destination)
        names = [_put(ctx, "backup-receipt.json", receipt),
                 _put(ctx, "backup-location.json", {"destination": str(destination), "files": receipt["totals"]["files"], "bytes": receipt["totals"]["bytes"],
                                                    "independent_device_required": bool(ctx.binding["limits"]["require_independent_backup"])})]
        return StageOutcome(tuple(names), {"files": receipt["totals"]["files"]})

    def summary(self, ctx: StageContext) -> StageOutcome:
        plan = self.plan
        events = [json.loads(line) for line in (ctx.run_root / "events.jsonl").read_text().splitlines() if line]
        observed: dict[str, float] = {}
        for event in events:
            if event["event"] == "stage_succeeded":
                observed[event["stage_id"]] = observed.get(event["stage_id"], 0.0) + event["seconds"]
        estimates = plan["budget"]["estimates"]["stage_seconds"]
        selection = self._selection(ctx)
        report = read_json(ctx.dependency_dir("score") / "final-report.json", "final-report.json")
        release = verify_release(ctx.dependency_dir("export_release") / "release", allow_candidate=False)
        receipt = read_json(ctx.dependency_dir("backup") / "backup-receipt.json", "backup-receipt.json")
        stages = {}
        for stage in plan["stages"]:
            manifest_path = ctx.run_root / "stages" / stage["stage_id"] / "stage-manifest.json"
            if manifest_path.is_file():
                manifest = read_stage_manifest(manifest_path)
                stages[stage["stage_id"]] = {"manifest_sha256": manifest["manifest_sha256"], "status": manifest["status"]}
        summary = {
            "plan_id": self.plan_id, "status": "complete", "rehearsal": plan["rehearsal"],
            "recipes_completed": [m["stage_id"] for m in plan["matrix"]], "recipes_missing": [],
            "validation_decision": selection["decision"], "selection_id": selection["selection_id"], "final_report_id": report["final_report_id"],
            "release_id": release["release_id"], "release_status": release["status"], "default_mode": release["default_mode"],
            "estimates_vs_observed_seconds": {sid: {"estimate": estimates.get(sid), "observed": observed.get(sid)} for sid in sorted(observed)},
            "estimate_label": "estimates come from measured calibration rates; observed values are measured durations of this run",
            "backup": {"status": "verified", "files": receipt["totals"]["files"], "bytes": receipt["totals"]["bytes"], "receipt_sha256": receipt["manifest_sha256"]},
            "limitations": report["limitations"],
        }
        gates = {name: dict(gate) for name, gate in plan["gates"].items()}      # the verdicts frozen when the plan was made
        acceptance_report = acceptance.build_report(
            plan=plan, gates=gates, profile=plan["profile"]["name"], diagnostic=bool(plan["rehearsal"]), stages=stages, commands=[],
            artifacts={"final_report_id": report["final_report_id"], "release_id": release["release_id"], "backup_receipt_sha256": receipt["manifest_sha256"],
                       "selection_id": selection["selection_id"]}, created_at=utc_now())
        names = (_put(ctx, "campaign-summary.json", summary), _put(ctx, "acceptance-report.json", acceptance_report))
        return StageOutcome(names, {"final_report_id": report["final_report_id"]})
