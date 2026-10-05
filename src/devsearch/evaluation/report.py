"""The measured final report (docs/10 §6): every system scored on the same sealed bank, nothing hidden.

Inputs are the locked systems' permutations of the frozen candidate lists, the final qrels (the complete
judgment bank) and the frozen group map. Output is a JSON report, per-query rows and a short Markdown
summary. Rules enforced here:

* one shared qrels version for every system; a query counts only if it has at least one confirmed positive,
  the rest are listed separately (never a division by zero, never silently dropped);
* group-macro metrics average inside each frozen group first; intervals come from a paired *group* bootstrap
  with the protocol's seed and repetition count, never from group-by-seed rows;
* the report states plainly when a reranker is worse than, or statistically indistinguishable from, BM25 —
  the outcome label comes from the interval, not from the authors' hopes;
* Recall@50 is recall of *known* positives inside the fixed candidate list; it is not exhaustive relevance;
* synthetic queries stay labelled synthetic and agent review stays labelled agent review.

Standard library only. The report id is a hash of the report body, so any edit is detectable.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from devsearch.artifacts.canonical import digest_of, sha256_hex
from devsearch.artifacts.files import jsonl_bytes
from devsearch.data.queries import Qrel
from devsearch.errors import ManifestError
from devsearch.evaluation.metrics import METRICS, paired_group_bootstrap, per_group
from devsearch.evaluation.protocol import Protocol
from devsearch.evaluation.ranking import positives_by_query, score_with_positives

REPORT_SCHEMA_VERSION = 1
BASELINE = "bm25"
PRIMARY_METRIC = "mrr@10"


@dataclass(frozen=True)
class FinalReport:
    report: dict[str, Any]
    per_query: bytes
    markdown: str


def _outcome(interval: Mapping[str, Any]) -> str:
    if interval["interval"][0] > 0.0:
        return "improved: the 95% interval for the group-macro difference excludes zero (positive)"
    if interval["interval"][1] < 0.0:
        return "worse: the 95% interval for the group-macro difference excludes zero (negative)"
    return "inconclusive: the interval for the group-macro difference spans zero; a stable gain is not supported"


def _compare(a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]], group_of: Mapping[str, str], protocol: Protocol) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in ("mrr@10", "ndcg@10"):
        interval = paired_group_bootstrap(
            per_group(a, group_of, metric), per_group(b, group_of, metric), repetitions=protocol.bootstrap_repetitions,
            seed=protocol.bootstrap_seed, confidence=protocol.bootstrap_confidence)
        out[metric] = {**interval, "outcome": _outcome(interval)}
    return out


def build_final_report(
    *, plan_id: str, protocol: Protocol, selection: Mapping[str, Any], systems: Mapping[str, Sequence[Mapping[str, Any]]],
    system_meta: Mapping[str, Mapping[str, Any]], qrels: Sequence[Qrel], group_of: Mapping[str, str],
    provenance: Mapping[str, Any], seals: Mapping[str, str],
) -> FinalReport:
    if BASELINE not in systems:
        raise ManifestError("the report needs the BM25 baseline among the scored systems")
    if set(system_meta) != set(systems):
        raise ManifestError("every scored system needs its metadata, and nothing else may carry any")
    positives = positives_by_query(qrels)
    scored = {name: score_with_positives(records, positives, group_of) for name, records in sorted(systems.items())}
    reference = scored[BASELINE]
    for name, result in scored.items():
        if set(result["per_query"]) != set(reference["per_query"]) or result["queries_without_positive"] != reference["queries_without_positive"]:
            raise ManifestError(f"system {name!r} is not scored over the same queries as the baseline")
    system_records = []
    for name, result in scored.items():
        system_records.append({
            "system": name, **dict(system_meta[name]), "queries_scored": result["queries_scored"], "groups_scored": result["groups_scored"],
            "query_macro": result["query_macro"], "group_macro": result["group_macro"],
        })
    selected = selection["decision"]
    primary_recipe = selected["recipe_id"]
    comparisons = []
    for name in sorted(scored):
        if name == BASELINE:
            continue
        meta = system_meta[name]
        is_deployment_seed = meta.get("seed") == protocol.deployment_seed
        primary = bool(primary_recipe is not None and meta.get("recipe_id") == primary_recipe and is_deployment_seed)
        comparisons.append({
            "system": name, "baseline": BASELINE, "recipe_id": meta.get("recipe_id"), "seed": meta.get("seed"),
            "primary": primary, "deployment_seed_comparison": is_deployment_seed,
            "differences": _compare(scored[name]["per_query"], reference["per_query"], group_of, protocol),
        })
    recipe_means = []
    if len(protocol.seeds) > 1:
        for recipe in protocol.recipes:
            names = [n for n, m in sorted(system_meta.items()) if m.get("recipe_id") == recipe.recipe_id]
            if len(names) == len(protocol.seeds):
                averaged: dict[str, dict[str, float]] = {}
                for query in reference["per_query"]:
                    averaged[query] = {m: sum(scored[n]["per_query"][query][m] for n in names) / len(names) for m in METRICS}
                recipe_means.append({
                    "recipe_id": recipe.recipe_id, "systems": names, "differences": _compare(averaged, reference["per_query"], group_of, protocol),
                    "note": "conditional on the declared seeds: scores are averaged across seeds inside each group, then only groups are resampled; "
                            "this is not a training-variability interval",
                })
    rows = []
    for name, result in scored.items():
        by_id = {r["query_id"]: r for r in systems[name]}
        for query_id in sorted(result["per_query"]):
            rows.append({"system": name, "query_id": query_id, "group_id": group_of[query_id], **result["per_query"][query_id],
                         "top10": [item["chunk_id"] for item in by_id[query_id]["ranking"][:10]]})
    per_query = jsonl_bytes(rows)
    body = {
        "schema_version": REPORT_SCHEMA_VERSION, "kind": "final-report", "status": "complete", "plan_id": plan_id,
        "protocol_id": protocol.protocol_id, "selection_id": selection["selection_id"], "default_mode": selected["default_mode"],
        "selected": {k: selected[k] for k in ("recipe_id", "seed", "checkpoint_id", "checkpoint_sha256")},
        "denominator": {"queries_scored": reference["queries_scored"], "groups_scored": reference["groups_scored"],
                        "queries_without_confirmed_positive": list(reference["queries_without_positive"]),
                        "rule": "queries with at least one confirmed positive in the final qrels; a retrieval miss scores zero and stays in"},
        "systems": system_records, "comparisons": comparisons, "recipe_means": recipe_means,
        "bootstrap": {"kind": "paired group bootstrap", "confidence": protocol.bootstrap_confidence,
                      "repetitions": protocol.bootstrap_repetitions, "seed": protocol.bootstrap_seed},
        "provenance": dict(provenance), "seals": dict(seals), "per_query_sha256": sha256_hex(per_query),
        "limitations": _limitations(provenance),
    }
    report = {**body, "final_report_id": "report-" + digest_of(body)[:32]}
    return FinalReport(report, per_query, render_markdown(report))


def _limitations(provenance: Mapping[str, Any]) -> list[str]:
    notes = [
        "Recall@50 is recall of known positives inside the fixed natural BM25 top-50; it is not exhaustive relevance.",
        "Intervals describe the sampled evidence groups, not future users; a single-split result is not a population guarantee.",
        "Rerankers only reorder the same frozen candidate lists; known positives outside a list were judged but never inserted into retrieval.",
    ]
    if provenance.get("query_origins", {}).get("synthetic"):
        notes.append("Queries are SYNTHETIC (generated by an agent workflow); they are not real user questions.")
    if provenance.get("reviewers", {}).get("agent") and not provenance.get("reviewers", {}).get("human"):
        notes.append("All judgments come from AGENT review; no human reviewer judged any item.")
    elif provenance.get("reviewers", {}).get("human"):
        notes.append("Judgments mix agent and human review; the counts are in `provenance.reviewers`.")
    return notes


def verify_report(report: Mapping[str, Any], per_query: bytes) -> None:
    """Detect edits: the id must hash the body and the per-query file must match its recorded hash."""
    body = {k: v for k, v in report.items() if k != "final_report_id"}
    if report.get("final_report_id") != "report-" + digest_of(body)[:32]:
        raise ManifestError("final report: final_report_id does not match the report content")
    if sha256_hex(per_query) != report.get("per_query_sha256"):
        raise ManifestError("final report: per-query rows do not match their recorded SHA-256")
    if report.get("status") != "complete":
        raise ManifestError("final report: only a complete report can be verified")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# DevSearch final report `{report['final_report_id']}`", "",
             f"Default mode chosen on validation only: **{report['default_mode']}**"
             + (f" (recipe `{report['selected']['recipe_id']}`, seed {report['selected']['seed']})" if report["selected"]["recipe_id"] else ""), "",
             f"Scored {report['denominator']['queries_scored']} queries in {report['denominator']['groups_scored']} independent groups; "
             f"{len(report['denominator']['queries_without_confirmed_positive'])} query/queries without a confirmed positive are listed in the JSON report.", "",
             "| system | group MRR@10 | group nDCG@10 | group Recall@50 | query MRR@10 |", "| --- | ---: | ---: | ---: | ---: |"]
    for system in report["systems"]:
        group, query = system["group_macro"], system["query_macro"]
        lines.append(f"| {system['system']} | {group['mrr@10']:.4f} | {group['ndcg@10']:.4f} | {group['recall@50']:.4f} | {query['mrr@10']:.4f} |")
    lines += ["", "## Paired group-bootstrap differences against BM25", ""]
    for comparison in report["comparisons"]:
        tag = " (primary)" if comparison["primary"] else ""
        for metric, result in comparison["differences"].items():
            low, high = result["interval"]
            lines.append(f"- `{comparison['system']}`{tag}, {metric}: {result['observed_difference']:+.4f} [{low:+.4f}, {high:+.4f}] — {result['outcome']}")
    lines += ["", "## Limitations", ""] + [f"- {note}" for note in report["limitations"]]
    return "\n".join(lines) + "\n"
