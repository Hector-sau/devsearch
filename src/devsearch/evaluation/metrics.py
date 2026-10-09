"""Ranking metrics and paired group bootstrap (docs/02 §5, docs/10 §6). Pure standard library.

Conventions, all fixed by the data specification:

* Binary relevance. A missing judgment contributes zero gain; it is *unjudged*, not proof of
  irrelevance. `positives` is the set of confirmed relevant chunk IDs for the query.
* MRR@k is the reciprocal rank of the first confirmed positive within the top k (else 0).
* nDCG@k uses gain/log2(rank+1); the ideal DCG places min(|positives|, k) positives first, where
  `positives` is *every* known positive (including ones retrieval missed).
* Recall@k is the fraction of known positives found in the top k of the ranked list.
* The ranking denominator is queries with at least one confirmed positive; a retrieval miss
  scores zero rather than leaving the denominator. Queries without a positive are counted
  separately and never divided by zero.
* Group-macro averages inside each frozen `group_id` first, then weights groups equally.
* The paired bootstrap resamples *groups* with replacement (never group-by-seed rows) using the
  same sampled groups for both systems, with a prespecified seed and repetition count.
"""

import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

METRICS = ("mrr@10", "ndcg@10", "recall@50")


def mrr_at(ranked: Sequence[str], positives: frozenset[str] | set[str], k: int = 10) -> float:
    for position, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in positives:
            return 1.0 / position
    return 0.0


def ndcg_at(ranked: Sequence[str], positives: frozenset[str] | set[str], k: int = 10) -> float:
    if not positives:
        raise ValueError("nDCG is undefined without a positive; the caller excludes such queries")
    dcg = sum(1.0 / math.log2(position + 1) for position, c in enumerate(ranked[:k], start=1) if c in positives)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(positives), k) + 1))
    return dcg / ideal


def recall_at(ranked: Sequence[str], positives: frozenset[str] | set[str], k: int = 50) -> float:
    if not positives:
        raise ValueError("recall is undefined without a positive; the caller excludes such queries")
    return len(positives & set(ranked[:k])) / len(positives)


def query_metrics(ranked: Sequence[str], positives: frozenset[str] | set[str]) -> dict[str, float]:
    return {"mrr@10": mrr_at(ranked, positives), "ndcg@10": ndcg_at(ranked, positives), "recall@50": recall_at(ranked, positives)}


def aggregate(
    per_query: Mapping[str, Mapping[str, float]], group_of: Mapping[str, str], *, excluded_without_positive: Sequence[str] = (),
) -> dict[str, Any]:
    """Query-macro and group-macro means over the queries that have a confirmed positive."""
    ids = sorted(per_query)
    if not ids:
        raise ValueError("no scored queries")
    query_macro = {m: sum(per_query[q][m] for q in ids) / len(ids) for m in METRICS}
    by_group: dict[str, list[str]] = defaultdict(list)
    for query_id in ids:
        by_group[group_of[query_id]].append(query_id)
    group_scores = {g: {m: sum(per_query[q][m] for q in members) / len(members) for m in METRICS} for g, members in by_group.items()}
    group_macro = {m: sum(s[m] for s in group_scores.values()) / len(group_scores) for m in METRICS}
    return {
        "queries_scored": len(ids), "groups_scored": len(by_group), "queries_without_positive": sorted(excluded_without_positive),
        "query_macro": query_macro, "group_macro": group_macro, "group_scores": {g: group_scores[g] for g in sorted(group_scores)},
    }


def per_group(per_query: Mapping[str, Mapping[str, float]], group_of: Mapping[str, str], metric: str) -> dict[str, float]:
    members: dict[str, list[float]] = defaultdict(list)
    for query_id, scores in per_query.items():
        members[group_of[query_id]].append(scores[metric])
    return {g: sum(v) / len(v) for g, v in sorted(members.items())}


def paired_group_bootstrap(
    scores_a: Mapping[str, float], scores_b: Mapping[str, float], *, repetitions: int = 10_000, seed: int = 20261005,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Interval for mean(a - b) over groups; both systems share every resample.

    The inputs are *per-group* scores. Paraphrases inside a group are not independent units, so
    resampling happens only at the group level. An interval that spans zero does not support a
    stable-gain claim. The result is descriptive of the sampled evidence groups, not of future users.
    """
    if scores_a.keys() != scores_b.keys():
        raise ValueError("both systems must be scored on the same groups")
    groups = sorted(scores_a)
    if not groups:
        raise ValueError("no groups to resample")
    if repetitions < 100 or not 0.5 < confidence < 1.0:
        raise ValueError("need at least 100 repetitions and 0.5 < confidence < 1")
    diffs = [scores_a[g] - scores_b[g] for g in groups]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(repetitions))
    tail = (1.0 - confidence) / 2.0
    low = means[int(math.floor(tail * repetitions))]
    high = means[min(int(math.ceil((1.0 - tail) * repetitions)) - 1, repetitions - 1)]
    observed = sum(diffs) / n
    return {
        "observed_difference": observed, "interval": [low, high], "confidence": confidence, "repetitions": repetitions,
        "seed": seed, "groups": n, "interval_excludes_zero": low > 0.0 or high < 0.0,
    }
