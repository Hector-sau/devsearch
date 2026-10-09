"""`python -m devsearch.evaluate`: score BM25 or a reranker on a training-visible split (train, development, validation).

    evaluate --config eval.toml --mode baseline|rerank --index <prepared/index/index-manifest.json> [--checkpoint REF] --split validation --output metrics.json

Both modes rank the SAME fixed natural BM25 candidate lists and are scored over the same denominator: queries with at
least one confirmed positive (a retrieval miss scores zero and stays in; queries without a positive are counted
separately). Known positives are never inserted into a result list; `positives_missing_from_candidates` reports how many
retrieval missed. Reports MRR@10, nDCG@10 and Recall@50 of known positives, query- and group-macro.

The final-test stages (rank, pool, score) are deliberately NOT available as free-standing commands: they run only inside a
campaign (`devsearch.run`), after the validation lock, so test results can never be produced before the models and the
default mode are fixed on validation alone. `--split test` and `--stage` exit 2 with that explanation.
Exit: 0 written, 1 invalid input or failure, 2 usage / unsupported operation.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser

TEST_MESSAGE = ("devsearch.evaluate: the test split and the rank/pool/score stages run only inside a campaign (`python -m devsearch.run`), after the "
                "validation selection has committed; they are not available as free-standing commands. / 最终测试仅在实验流程内、验证选择锁定后运行。")


def build_parser():
    parser = new_parser("evaluate", "Evaluate BM25 or a reranker on a training-visible split.")
    parser.add_argument("--config", type=Path, required=True, help="Evaluation TOML (see configs/eval.toml).")
    parser.add_argument("--mode", choices=("baseline", "rerank"), required=True)
    parser.add_argument("--index", type=Path, required=True, help="The prepared index manifest (<prepared>/index/index-manifest.json).")
    parser.add_argument("--checkpoint", type=Path, help="rerank: an exported inference bundle directory or a training checkpoint manifest (.json).")
    parser.add_argument("--split", choices=("development", "train", "validation", "test"), default="validation")
    parser.add_argument("--stage", choices=("rank", "pool", "score"), help="Not available standalone; see the module help.")
    parser.add_argument("--output", type=Path, required=True, help="New metrics JSON (never overwritten).")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain every artifact path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.split == "test" or args.stage is not None:
        print(TEST_MESSAGE, file=sys.stderr)
        return 2
    from devsearch.errors import DevSearchError
    try:
        report = _run(args)
    except (DevSearchError, OSError) as error:
        return fail("evaluate", error)
    print(json.dumps({"status": "ok", "mode": report["mode"], "split": report["split"], "group_macro": report["group_macro"], "output": str(args.output.resolve())}))
    return 0


def _run(args) -> dict:
    import time
    from devsearch.artifacts.canonical import pretty_json, sha256_file
    from devsearch.artifacts.files import write_atomic
    from devsearch.artifacts.paths import ensure_inside, resolve_root
    from devsearch.data.candidates import build_candidates, load_candidates
    from devsearch.data.prepare import PREPARED_MANIFEST, load_prepared
    from devsearch.data.queries import load_qrels, load_queries
    from devsearch.errors import ManifestError
    from devsearch.evaluation.ranking import bm25_ranked_records, order_by_score, positives_by_query, ranked_record, score_with_positives

    root = resolve_root(args.runtime_root, args.output.resolve().parent)
    output = ensure_inside(args.output, root, what="metrics output")
    if output.exists():
        raise ManifestError(f"{output} already exists; metrics are never overwritten")
    if not output.parent.is_dir():
        raise ManifestError(f"output directory does not exist: {output.parent}")
    from devsearch.modeling.jobs import load_model_job
    job = load_model_job(args.config)                       # strict even for the baseline, which then never touches a device or an encoder
    index_path = ensure_inside(args.index, root, what="index manifest")
    prepared_manifest = index_path.parent.parent / PREPARED_MANIFEST
    if not prepared_manifest.is_file():
        raise ManifestError(f"{prepared_manifest} not found: --index must be <prepared>/index/index-manifest.json of a prepared-data directory")
    prepared = load_prepared(prepared_manifest, root)
    loaded, base = prepared["loaded"], prepared["directory"]
    split_file = base / "data" / f"queries-{args.split}.jsonl"
    if not split_file.is_file():
        raise ManifestError(f"the prepared data holds no {args.split} queries")
    queries = list(load_queries(split_file))
    qrels = list(load_qrels(base / "data" / f"qrels-{args.split}.jsonl"))
    group_of = {q.query_id: q.group_id for q in queries}
    depth = 50
    if args.split == "validation":
        manifest, records = load_candidates(base / "candidates" / "candidates-validation-manifest.json", loaded)
        depth, source = manifest["candidate_k"] or 50, "stored (frozen) validation candidates"
    else:
        records, source = build_candidates(loaded, queries, depth), "regenerated natural BM25 candidates (deterministic)"
    started = time.perf_counter()
    model_info = {"model_id": None, "source": None, "checkpoint_sha256": None, "device": None, "batch_size": None}
    if args.mode == "baseline":
        ranked = bm25_ranked_records(records)
    else:
        from devsearch.modeling.devices import load_profile, resolve_device
        from devsearch.modeling.scoring import score_pairs
        from devsearch.training.resolve import resolve_model
        if args.checkpoint is None:
            raise ManifestError("rerank mode needs --checkpoint")
        device = resolve_device(load_profile(job.profile))
        resolved = resolve_model(args.checkpoint, device=device, assets=job.assets, policy=job.policy, dropout=job.dropout)
        texts = {c.chunk_id: c.text for c in loaded.corpus.chunks}
        by_id = {q.query_id: q for q in queries}
        ranked = []
        for record in records:
            ids = [c["chunk_id"] for c in record["candidates"]]
            scores = score_pairs(resolved.model, resolved.tokenizer, resolved.policy, [(by_id[record["query_id"]].text, texts[i]) for i in ids],
                                 device=device, batch_size=job.batch_size)
            ranked.append(ranked_record(record["query_id"], record["candidate_set_id"], order_by_score(ids, scores)))
        model_info = {"model_id": resolved.model_id, "source": resolved.source, "checkpoint_sha256": resolved.checkpoint_sha256, "device": str(device),
                      "batch_size": job.batch_size}
    elapsed = time.perf_counter() - started
    positives = positives_by_query(qrels)
    scored = score_with_positives(ranked, positives, group_of)
    candidate_ids = {r["query_id"]: {c["chunk_id"] for c in r["candidates"]} for r in records}
    missing = sum(len(pos - candidate_ids.get(q, set())) for q, pos in positives.items() if q in candidate_ids)
    origins: dict[str, int] = {}
    for query in queries:
        origins[query.origin] = origins.get(query.origin, 0) + 1
    report = {
        "schema_version": 1, "kind": "evaluation", "mode": args.mode, "split": args.split, "corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id,
        "candidate_source": source, "candidate_depth": depth, "config_sha256": sha256_file(args.config), **model_info,
        "queries_scored": scored["queries_scored"], "groups_scored": scored["groups_scored"], "queries_without_positive": scored["queries_without_positive"],
        "positives_missing_from_candidates": missing, "query_macro": scored["query_macro"], "group_macro": scored["group_macro"],
        "per_query": scored["per_query"], "label_provenance": {"query_origins": origins},
        "recall_note": "Recall@50 is recall of KNOWN positives inside the fixed candidate list, not exhaustive relevance; positives are never inserted into results.",
        "seconds": round(elapsed, 3),
    }
    write_atomic(output, pretty_json(report).encode("utf-8"), exclusive=True)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
