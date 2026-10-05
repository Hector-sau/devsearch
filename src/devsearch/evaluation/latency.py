"""The declared latency protocol (docs/10 §6): warm-up, then seeded sequential passes, no response cache.

`measure` times a caller-supplied `search(query) -> response` one request at a time. The workload is a
fixed, seeded list drawn from development/validation queries (never test), identical for both modes.
Percentiles use the nearest-rank rule so p95 of a given sample is unambiguous. A degraded response means
the model did not do its job: the evidence is marked invalid rather than silently averaged in.
Standard library only; timing a device call is only meaningful if `search` synchronizes (scoring returns
host floats, which does).
"""

import hashlib
import math
import random
import time
from collections.abc import Callable, Sequence
from typing import Any

from devsearch.artifacts.canonical import digest_of


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least q of the sample at or below it."""
    if not values or not 0.0 < q <= 1.0:
        raise ValueError("need a non-empty sample and 0 < q <= 1")
    ordered = sorted(values)
    return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]


def workload(query_ids: Sequence[str], *, size: int, seed: int) -> list[str]:
    """A fixed list of `size` queries (or all of them if fewer): hash-seeded, independent of input order."""
    unique = sorted(set(query_ids))
    ordered = sorted(unique, key=lambda q: hashlib.sha256(f"{seed}|workload|{q}".encode()).hexdigest())
    return ordered[:size]


def pass_order(queries: Sequence[str], *, seed: int, index: int) -> list[str]:
    shuffled = list(queries)
    random.Random(f"{seed}|pass|{index}").shuffle(shuffled)
    return shuffled


def measure(
    search: Callable[[str], dict[str, Any]], queries: Sequence[str], *, warmup: int, passes: int, seed: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if not queries or warmup < 1 or passes < 1:
        raise ValueError("need queries, at least one warm-up search and at least one pass")
    for index in range(warmup):                                        # warm-up cycles through the list; results discarded
        search(queries[index % len(queries)])
    timings: list[float] = []
    candidate_counts: list[int] = []
    degraded = 0
    for index in range(passes):
        for query in pass_order(queries, seed=seed, index=index):
            started = clock()
            response = search(query)
            timings.append((clock() - started) * 1000.0)
            candidate_counts.append(int(response.get("candidate_count", 0)))
            degraded += response.get("status") == "degraded"
    return {
        "queries": len(queries), "workload_id": "workload-" + digest_of(list(queries))[:24], "warmup": warmup, "passes": passes,
        "requests": len(timings), "concurrency": 1, "response_cache": False, "seed": seed,
        "p50_ms": percentile(timings, 0.50), "p95_ms": percentile(timings, 0.95), "max_ms": max(timings), "mean_ms": sum(timings) / len(timings),
        "candidate_count": {"min": min(candidate_counts), "median": percentile(candidate_counts, 0.5), "max": max(candidate_counts)},
        "degraded_requests": degraded, "valid": degraded == 0,
    }
