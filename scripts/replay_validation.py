#!/usr/bin/env python3
"""Replay frozen validation-v4 ranking arithmetic on CPU; never score a model.

Private owner-held inputs are required. The CLI verifies exact original byte
identities and structural/coverage gates, then writes aggregate metrics only.
This does not re-audit the original review ledger or establish new test quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from devsearch.evaluation.metrics import METRICS, aggregate, paired_group_bootstrap, query_metrics

QUERY_COUNT, GROUP_COUNT, PAIR_COUNT, CANDIDATE_K = 202, 202, 4571, 50
SEEDS = (42, 43, 44)
SYSTEMS = frozenset({"BM25"} | {
    f"R{round_number}-seed{seed}-{mode}"
    for round_number in (4, 5, 6, 7) for seed in SEEDS for mode in ("raw", "fused")
})
PINNED_SHA256 = {
    "queries": "a45a64dc9753b8d84442620e8f8f134e6ad66d99409241dccc3d67dbeee54ee5",
    "rankings": "5e28564770f93de98f1752a94845088714d41a6cd72f1ebc245e5163078c84ba",
    "label_release": "bee1c2bcd08b4085c6391fc022fca5b0355bce67c0e2a8317cb1c1bcccf2e038",
}
METRIC_NAMES = {"mrr@10": "mrr@10", "ndcg@10": "pooled_ndcg@10", "recall@50": "known_positive_recall@50"}
BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED = 10_000, 20261008


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def parse_json(data: bytes) -> object:
    return json.loads(data, object_pairs_hook=unique_object)


def read_pinned(path: Path, expected: str, kind: str) -> tuple[bytes, str]:
    require(path.is_file(), f"missing {kind} input file")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    require(actual == expected, f"{kind} SHA-256 mismatch")
    return data, actual


def identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_inputs(
    queries: object, rankings: object, release: object, labels: object, *,
    expected_queries: int = QUERY_COUNT, expected_groups: int = GROUP_COUNT, expected_pairs: int = PAIR_COUNT,
) -> dict:
    """Structural checks are separate from byte binding for focused fixture tests."""
    require(isinstance(queries, list) and len(queries) == expected_queries, "query count mismatch")
    group_of, evidence_groups = {}, set()
    for query in queries:
        require(isinstance(query, dict), "query record must be an object")
        query_id, group_id = query.get("query_id"), query.get("group_id")
        require(identifier(query_id) and identifier(group_id), "invalid query/group identifier")
        require(query_id not in group_of, "duplicate query identifier")
        require(query.get("split") == "validation" and query.get("language") == "en", "query is outside English validation scope")
        groups = query.get("evidence_group_ids")
        require(isinstance(groups, list) and len(groups) == 1 and identifier(groups[0]), "invalid evidence-group membership")
        require(groups[0] not in evidence_groups, "duplicate evidence-group membership")
        evidence_groups.add(groups[0])
        group_of[query_id] = group_id
    require(len(set(group_of.values())) == expected_groups, "group count or unique group membership mismatch")

    require(isinstance(release, dict), "label release must be an object")
    require(release.get("schema_version") == "devsearch-sampled-agent-human-assisted-release-v4", "wrong label-release schema")
    require(release.get("status") == "sealed", "label release is not sealed")
    require(release.get("query_count") == expected_queries and release.get("items") == expected_pairs, "label-release counts mismatch")
    require(isinstance(labels, list) and len(labels) == expected_pairs, "resolved-label count mismatch")
    judgments, positives = {}, {query_id: set() for query_id in group_of}
    for label in labels:
        require(isinstance(label, dict), "resolved label must be an object")
        query_id, chunk_id, relevance = label.get("query_id"), label.get("chunk_id"), label.get("relevance")
        require(query_id in group_of and identifier(chunk_id), "label has invalid query/chunk membership")
        require(type(relevance) is int and relevance in (0, 1), "resolved label is not binary")
        pair = (query_id, chunk_id)
        require(pair not in judgments, "duplicate resolved query/chunk pair")
        judgments[pair] = relevance
        if relevance == 1:
            positives[query_id].add(chunk_id)
    require(all(positives.values()), "a query has no known positive; replay refuses to exclude it")

    require(isinstance(rankings, dict), "compact rankings must be an object")
    require(rankings.get("exact_order_equality_verified_on_server") is True, "compact ranking order provenance missing")
    require(rankings.get("ranking_scores_omitted") is True, "unexpected ranking-score representation")
    sources = rankings.get("rank_source_sha256")
    require(isinstance(sources, dict) and set(sources) == SYSTEMS, "ranking-source system inventory mismatch")
    require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in sources.values()), "invalid ranking-source identity")
    orders = rankings.get("ranking_orders")
    require(isinstance(orders, dict) and set(orders) == set(group_of), "ranking query membership mismatch")
    reconstructed = {name: {} for name in sorted(SYSTEMS)}
    top10_positions, top50_judged = 0, 0
    for query_id, row in orders.items():
        require(isinstance(row, dict), "ranking row must be an object")
        chunks, permutations = row.get("chunk_ids"), row.get("system_permutations")
        require(isinstance(chunks, list) and len(chunks) == CANDIDATE_K, "candidate pool must contain exactly 50 chunks")
        require(all(identifier(chunk) for chunk in chunks) and len(set(chunks)) == CANDIDATE_K, "candidate pool has invalid or duplicate chunks")
        require(isinstance(permutations, dict) and set(permutations) == SYSTEMS, "per-query system inventory mismatch")
        for system, order in permutations.items():
            require(isinstance(order, list) and len(order) == CANDIDATE_K, "system ranking must have 50 positions")
            require(all(type(position) is int for position in order) and set(order) == set(range(CANDIDATE_K)), "malformed system permutation")
            ranked = [chunks[position] for position in order]
            require(set(ranked) == set(chunks), "system candidate-pool mismatch")
            require(all((query_id, chunk) in judgments for chunk in ranked[:10]), "incomplete labeled top-10 coverage")
            reconstructed[system][query_id] = ranked
            top10_positions += 10
            top50_judged += sum((query_id, chunk) in judgments for chunk in ranked)
    return {"group_of": group_of, "ranked": reconstructed, "positives": positives, "pair_count": len(judgments),
            "positive_pairs": sum(judgments.values()), "top10_positions": top10_positions, "top50_judged": top50_judged}


def load_frozen(queries: Path, rankings: Path, release_directory: Path) -> tuple[dict, dict[str, str]]:
    require(release_directory.is_dir(), "--release must be the sealed v4 release directory")
    query_bytes, query_hash = read_pinned(queries, PINNED_SHA256["queries"], "queries")
    ranking_bytes, ranking_hash = read_pinned(rankings, PINNED_SHA256["rankings"], "rankings")
    release_bytes, release_hash = read_pinned(release_directory / "label-release.json", PINNED_SHA256["label_release"], "label release")
    manifest = parse_json(release_bytes)
    require(isinstance(manifest, dict), "label release must be an object")
    labels_hash = manifest.get("resolved_labels_sha256")
    require(isinstance(labels_hash, str) and re.fullmatch(r"[0-9a-f]{64}", labels_hash), "invalid resolved-label hash binding")
    label_bytes, verified_label_hash = read_pinned(release_directory / "resolved-labels.json", labels_hash, "resolved labels")
    query_rows = [parse_json(line) for line in query_bytes.splitlines() if line.strip()]
    valid = validate_inputs(query_rows, parse_json(ranking_bytes), manifest, parse_json(label_bytes),
                            expected_queries=QUERY_COUNT, expected_groups=GROUP_COUNT, expected_pairs=PAIR_COUNT)
    return valid, {"queries": query_hash, "compact_rankings": ranking_hash, "label_release": release_hash,
                   "resolved_labels": verified_label_hash}


def metric_names(values: dict[str, float]) -> dict[str, float]:
    return {METRIC_NAMES[name]: values[name] for name in METRICS}


def summarize(valid: dict, hashes: dict[str, str], *, bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS) -> dict:
    group_of, ranked, positives = valid["group_of"], valid["ranked"], valid["positives"]
    summaries, grouped = {}, {}
    for system in sorted(ranked):
        per_query = {query_id: query_metrics(ranked[system][query_id], positives[query_id]) for query_id in sorted(group_of)}
        result = aggregate(per_query, group_of)
        summaries[system] = {"query_macro": metric_names(result["query_macro"]), "group_macro": metric_names(result["group_macro"])}
        grouped[system] = result["group_scores"]

    family = {"seeds": list(SEEDS), "query_macro": {}, "group_macro": {}}
    for scope in ("query_macro", "group_macro"):
        for metric in METRIC_NAMES.values():
            values = [summaries[f"R5-seed{seed}-fused"][scope][metric] for seed in SEEDS]
            baseline = summaries["BM25"][scope][metric]
            family[scope][metric] = {"per_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
                                    "mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                                    "mean_delta_vs_BM25": statistics.mean(values) - baseline}
    family["all_three_mrr_improved"] = all(
        summaries[f"R5-seed{seed}-fused"]["group_macro"]["mrr@10"] > summaries["BM25"]["group_macro"]["mrr@10"]
        for seed in SEEDS
    )
    intervals = {}
    for metric in METRICS:
        # Average the three seed scores *inside each group* before resampling groups.
        mean_seed_groups = {group: statistics.mean(grouped[f"R5-seed{seed}-fused"][group][metric] for seed in SEEDS)
                            for group in sorted(grouped["BM25"])}
        baseline_groups = {group: value[metric] for group, value in grouped["BM25"].items()}
        intervals[METRIC_NAMES[metric]] = paired_group_bootstrap(
            mean_seed_groups, baseline_groups, repetitions=bootstrap_repetitions, seed=BOOTSTRAP_SEED, confidence=.95)
    return {
        "schema_version": "devsearch-validation-v4-arithmetic-replay-v1", "status": "verified_sealed_input_arithmetic_replay",
        "label_version": "validation-v2-sampled202-human6-v4", "input_sha256": hashes,
        "counts": {"queries": len(group_of), "groups": len(set(group_of.values())), "excluded_queries": 0,
                   "labeled_pairs": valid["pair_count"], "positive_pairs": valid["positive_pairs"], "systems": len(ranked),
                   "candidates_per_query_per_system": CANDIDATE_K},
        "coverage": {"top10_judged_fraction": 1.0, "top10_judged_positions": valid["top10_positions"],
                     "top50_judged_fraction": valid["top50_judged"] / (len(group_of) * len(ranked) * CANDIDATE_K),
                     "same_candidate_pool_for_all_systems": True},
        "systems": summaries, "R5_fused_three_seed_summary": family,
        "R5_fused_vs_BM25_paired_group_bootstrap": intervals,
        "scope": {"model_scoring": False, "new_training": False, "new_label_review": False,
                  "full_original_review_ledger_reaudited": False, "fresh_holdout": False,
                  "bootstrap_unit": "evidence group, with seeds averaged inside each group", "synthetic_queries": True},
        "limitations": ["Arithmetic replay of pinned, sealed owner-held inputs; original adjudication audit remains its original evidence.",
                        "The validation benchmark was repeatedly used during development; bootstrap intervals are descriptive.",
                        "nDCG and recall use the known pooled positives; full corpus relevance was not exhaustively labeled.",
                        "No private question, label, ranking, group identifier or filesystem path is exported."]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path, help="Original frozen validation-v1 queries JSONL")
    parser.add_argument("--rankings", required=True, type=Path, help="Original compact-ranking-orders.json")
    parser.add_argument("--release", required=True, type=Path, help="Sealed v4 directory containing label-release.json and resolved-labels.json")
    parser.add_argument("--output", required=True, type=Path, help="New aggregate JSON file; its parent directory must exist")
    args = parser.parse_args(argv)
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        require(not args.output.exists(), "output already exists; refusing overwrite")
        require(args.output.parent.is_dir(), "output parent directory does not exist")
        valid, hashes = load_frozen(args.queries, args.rankings, args.release)
        report = summarize(valid, hashes)
        report["runtime"] = {"python": sys.version.split()[0], "device": "cpu", "wall_seconds": time.perf_counter() - started,
                             "process_cpu_seconds": time.process_time() - cpu_started}
        payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except (ValueError, OSError, UnicodeError, KeyError, TypeError) as error:
        # Gate messages intentionally omit private question/chunk identifiers and paths.
        message = str(error) if isinstance(error, ValueError) else "input/output or schema failure"
        print(f"validation replay refused: {message}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "queries": report["counts"]["queries"],
                      "groups": report["counts"]["groups"], "excluded_queries": 0,
                      "BM25_mrr@10": report["systems"]["BM25"]["group_macro"]["mrr@10"],
                      "R5_fused_seed_mrr@10": report["R5_fused_three_seed_summary"]["group_macro"]["mrr@10"]["per_seed"],
                      "R5_fused_mean_mrr@10": report["R5_fused_three_seed_summary"]["group_macro"]["mrr@10"]["mean"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
