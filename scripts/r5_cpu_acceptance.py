"""CPU-only inference acceptance for the separately preregistered R5 holdout.

Ranks and timings only: no training, label inference, test tuning, or quality
metrics before pooled judgments are sealed. Existing campaign test gates remain.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def frozen_inputs(directory: Path, expected_plan_sha: str):
    from devsearch.artifacts.canonical import sha256_file
    from devsearch.errors import ManifestError
    from devsearch.evaluation.fusion import RankingPolicy
    plan_path, seal_path = directory / "frozen-evaluation-plan.json", directory / "questions-seal.json"
    if sha256_file(plan_path) != expected_plan_sha:
        raise ManifestError("the preregistered evaluation plan changed")
    plan, seal = json.loads(plan_path.read_text()), json.loads(seal_path.read_text())
    if (plan["kind"] != "r5-independent-evaluation-preregistration" or plan["deployment_seed"] != 42
            or plan["checkpoint"]["checkpoint_sha256"] != "74b7fd525d75a0aab2a126e49eb759704cd0cf51a2c955122db938d4cd9f2a3c"
            or RankingPolicy.from_record(plan["ranking_policy"]) != RankingPolicy("rank_fusion")
            or plan["candidate_k"] != 50 or plan["batch_size"] != 8):
        raise ManifestError("this runner accepts only the frozen seed42 R5+fusion comparison")
    questions_path = directory / "questions-sealed.json"
    if (seal["kind"] != "new_r5_holdout_questions_seal" or seal["status"] != "sealed_before_ranking"
            or seal["frozen_evaluation_plan_sha256"] != expected_plan_sha
            or seal["questions_sha256"] != sha256_file(questions_path)):
        raise ManifestError("the pre-ranking question seal does not match")
    questions = json.loads(questions_path.read_text())
    ids = [q["query_id"] for q in questions]
    if (not questions or len(set(ids)) != len(ids) or len(questions) != seal["queries"]
            or len({q["group_id"] for q in questions}) != len(questions)
            or any(q["review_status"] != "accepted" or q["reviewer_type"] != "agent" for q in questions)):
        raise ManifestError("question review or unique-group coverage is incomplete")
    return plan, seal, questions


async def exercise(args, plan, seal, questions):
    import httpx
    from devsearch.artifacts.canonical import pretty_json, sha256_file
    from devsearch.artifacts.files import write_atomic
    from devsearch.artifacts.identity import code_identity
    from devsearch.artifacts.paths import ensure_inside
    from devsearch.errors import ManifestError
    from devsearch.execution.serving import ServeSettings, start_service
    from devsearch.serving.api import create_app

    if ServeSettings.load(args.config).job.profile != "cpu-diagnostic":
        raise ManifestError("this acceptance runner refuses every accelerator profile")
    output = ensure_inside(args.output, args.runtime_root, what="acceptance output")
    if output.exists():
        raise ManifestError("output exists; use a new directory, never overwrite acceptance evidence")
    output.mkdir()
    started = start_service(config=args.config, manifest=args.release, allow_candidate=True, runtime_root=args.runtime_root)
    try:
        checkpoint = started.release["provenance"].get("model_source", {}).get("checkpoint_sha256")
        if checkpoint != plan["checkpoint"]["checkpoint_sha256"] or started.service.pipeline.ranking.to_record() != plan["ranking_policy"]:
            raise ManifestError("the candidate does not ship the locked R5 checkpoint and fusion policy")
        if started.settings.job.batch_size != plan["batch_size"]:
            raise ManifestError("the inference batch changed")
        # This API exercises real request validation, admission, worker and serialization.
        app = create_app(started.service)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local", trust_env=False, timeout=None) as client:
            async def request(question):
                before = time.perf_counter()
                response = await client.post("/search", json={"query": question["text"], "mode": "compare", "candidate_k": 50, "top_k": 50,
                                                             "sources": question.get("sources", []), "versions": question.get("versions", {})})
                elapsed = (time.perf_counter() - before) * 1000
                doc = response.json()
                if response.status_code != 200 or doc.get("status") not in ("ok", "empty"):
                    raise ManifestError(f"{question['query_id']}: real-model API failed/degraded: {doc.get('warnings', doc)}")
                return doc, elapsed

            if args.phase == "rank":
                rows = []
                for question in questions:
                    doc, elapsed = await request(question)
                    from devsearch.evaluation.fusion import RankingPolicy
                    neural = doc["reranked_results"]
                    lexical = doc["baseline_results"]
                    if {r["chunk_id"] for r in neural} != {r["chunk_id"] for r in lexical}:
                        raise ManifestError("API changed the frozen natural candidate set")
                    offline = RankingPolicy("rank_fusion").order([r["chunk_id"] for r in neural], [r["rerank_logit"] for r in neural],
                                                                [r["baseline_rank"] for r in neural])
                    if [c for c, _, _ in offline] != [r["chunk_id"] for r in neural]:
                        raise ManifestError("offline/online fusion order differs")
                    row = {"query_id": question["query_id"], "group_id": question["group_id"], "response": doc, "api_wall_ms": elapsed}
                    write_atomic(output / f"{question['query_id']}.json", pretty_json(row).encode(), exclusive=True)
                    rows.append(row)
                    print(json.dumps({"completed": len(rows), "total": len(questions)}), flush=True)
                result = {"kind": "r5_real_cpu_api_rankings", "questions": len(rows), "offline_online_order_mismatches": 0,
                          "quality_status": "pending blinded pooled review; no metrics computed"}
            else:
                chosen = random.Random(plan["latency"]["seed"]).sample(questions, min(plan["latency"]["queries"], len(questions)))
                for i in range(plan["latency"]["warmup"]):
                    await request(chosen[i % len(chosen)])
                values = []
                for _ in range(plan["latency"]["passes"]):
                    for question in chosen:
                        _, elapsed = await request(question)
                        values.append(elapsed)
                import torch
                ordered = sorted(values)
                import math
                result = {"kind": "r5_real_cpu_api_latency", "samples": len(values), "queries": len(chosen),
                          "warmup": plan["latency"]["warmup"], "passes": plan["latency"]["passes"], "concurrency": 1,
                          "api_wall_ms": values, "median_ms": statistics.median(values), "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
                          "device": "cpu", "host": platform.platform(), "processor": platform.processor(),
                          "torch_threads": torch.get_num_threads(), "inference_batch": started.settings.job.batch_size,
                          "scope": "in-process real API including admission, model, fusion and response serialization; no network transit or GPU SLA"}
            result.update({"plan_sha256": args.plan_sha256, "question_seal_sha256": sha256_file(args.inputs / "questions-seal.json"),
                           "checkpoint_sha256": checkpoint, "release_id": started.release["release_id"], "source": code_identity(),
                           "startup": started.startup, "gpu_used": False, "training_performed": False})
            write_atomic(output / "receipt.json", pretty_json(result).encode(), exclusive=True)
            return result
    finally:
        started.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("rank", "benchmark"))
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    from devsearch.errors import DevSearchError
    try:
        plan, seal, questions = frozen_inputs(args.inputs, args.plan_sha256)
        result = asyncio.run(exercise(args, plan, seal, questions))
        print(json.dumps({"kind": result["kind"], "receipt": str(args.output / "receipt.json")}))
        return 0
    except (DevSearchError, OSError, ValueError, KeyError) as error:
        print(f"r5_cpu_acceptance: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
