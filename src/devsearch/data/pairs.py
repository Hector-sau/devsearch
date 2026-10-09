"""Train-only negative mining and unique BCE pair materialization (docs/02 §4, R03).

* The mining pool holds only chunks whose merged group is owned by `train`. Held-out
  groups, held-out questions and held-out judgments are never read here.
* Per training query: every confirmed positive once (label 1), up to `hard_negatives`
  BM25-ranked and `random_negatives` seeded-random negatives from the pool (label 0), none from a
  group that holds any known positive of that query (duplicates included).
* Unknown passages remain *assumed* negatives and keep that status. A shortage is recorded,
  never filled from outside the pool.
* Records are unique `(query_id, chunk_id)`; positive/negative conflicts are errors; counts are
  reconciled; `pos_weight` is the actual sampled negative/positive ratio.
"""

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import (
    digest_of, loads_strict, pretty_json, read_utf8, seal_manifest, sha256_file, sha256_hex, verify_manifest_hash,
)
from devsearch.artifacts.files import jsonl_bytes, read_jsonl, write_atomic
from devsearch.artifacts.schema import as_id, as_object, as_sha256, exact_keys
from devsearch.contracts import SearchRequest
from devsearch.data.corpus import Corpus
from devsearch.data.groups import GroupMerge, owner_of_chunks
from devsearch.data.queries import Qrel, Query
from devsearch.errors import ManifestError
from devsearch.retrieval.bm25 import Bm25Retriever

PAIRS_SCHEMA_VERSION = 1
PAIR_FIELDS = ("pair_id", "query_id", "chunk_id", "label", "label_status", "negative_method", "sampling_manifest_id")
SAMPLING_FIELDS = ("query_id", "positive_chunk_ids", "negative_chunk_id", "negative_method", "label_status", "seed")
NEGATIVE_METHODS = ("bm25", "random")


@dataclass(frozen=True)
class MiningConfig:
    hard_negatives: int = 2
    random_negatives: int = 2
    seed: int = 42
    hard_search_depth: int = 100            # BM25 candidates examined per query when mining hard negatives

    def __post_init__(self) -> None:
        if min(self.hard_negatives, self.random_negatives) < 0 or self.hard_negatives + self.random_negatives < 1:
            raise ManifestError("need at least one negative per query and no negative counts below zero")
        if self.hard_search_depth < self.hard_negatives or self.hard_search_depth > 10_000:
            raise ManifestError("hard_search_depth must cover hard_negatives and stay at most 10000")


@dataclass(frozen=True)
class MiningPool:
    chunk_ids: tuple[str, ...]

    @property
    def pool_id(self) -> str:
        return "pool-" + digest_of(list(self.chunk_ids))[:32]


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    query_id: str
    chunk_id: str
    label: int
    label_status: str
    negative_method: str | None
    sampling_manifest_id: str

    def as_record(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id, "query_id": self.query_id, "chunk_id": self.chunk_id, "label": self.label,
            "label_status": self.label_status, "negative_method": self.negative_method,
            "sampling_manifest_id": self.sampling_manifest_id,
        }


@dataclass(frozen=True)
class PairSet:
    pairs: tuple[PairRecord, ...]
    sampling: tuple[Mapping[str, Any], ...]
    report: Mapping[str, Any]
    pool: MiningPool
    sampling_manifest_id: str


def build_mining_pool(
    corpus: Corpus, merge: GroupMerge, assignment: Mapping[str, str], *, excluded_chunk_ids: Iterable[str] = (),
) -> MiningPool:
    """Chunks owned by `train` (minus e.g. token-budget exclusions): the only negative source."""
    excluded = set(excluded_chunk_ids)
    owners = owner_of_chunks(corpus, merge, assignment)
    return MiningPool(tuple(sorted(c for c, owner in owners.items() if owner == "train" and c not in excluded)))


def _pair_id(sampling_id: str, query_id: str, chunk_id: str, label: int) -> str:
    return "pr_" + digest_of({"sampling": sampling_id, "query_id": query_id, "chunk_id": chunk_id, "label": label})[:32]


def sampling_identity(
    config: MiningConfig, pool: MiningPool, *, corpus_id: str, index_id: str, queries_sha256: str, qrels_sha256: str,
) -> str:
    return "sampling-" + digest_of({
        "hard": config.hard_negatives, "random": config.random_negatives, "seed": config.seed,
        "depth": config.hard_search_depth, "pool": pool.pool_id, "corpus": corpus_id, "index": index_id,
        "queries": queries_sha256, "qrels": qrels_sha256,
    })[:32]


def mine_pairs(
    *, corpus: Corpus, retriever: Bm25Retriever, merge: GroupMerge, pool: MiningPool,
    queries: Sequence[Query], qrels: Sequence[Qrel], config: MiningConfig, sampling_id: str,
) -> PairSet:
    """Materialize unique BCE pairs for `train` queries. Never touches any other split."""
    stray = sorted({q.query_id for q in qrels} - {q.query_id for q in queries})
    if stray:
        raise ManifestError(
            "judgments belong to queries that are not in the supplied train queries "
            f"(held-out judgments must never reach mining): {', '.join(stray[:5])}"
        )
    pool_set = frozenset(pool.chunk_ids)
    chunk_group = {chunk.chunk_id: merge.merged_of[chunk.evidence_group_id] for chunk in corpus.chunks}
    qrels_by_query: dict[str, list[Qrel]] = defaultdict(list)
    for qrel in qrels:
        qrels_by_query[qrel.query_id].append(qrel)
    exclusions: dict[str, int] = defaultdict(int)
    pairs: list[PairRecord] = []
    sampling: list[dict[str, Any]] = []
    shortages: list[dict[str, Any]] = []
    counts = {"positives": 0, "negatives_bm25": 0, "negatives_random": 0}
    used = 0
    for query in sorted(queries, key=lambda q: q.query_id):
        if query.split != "train":
            raise ManifestError(f"query {query.query_id} has split {query.split!r}: only train queries may be mined")
        if not query.accepted:
            exclusions["query_not_accepted"] += 1
            continue
        judged = qrels_by_query.get(query.query_id, [])
        positives = sorted({q.chunk_id for q in judged if q.confirmed_positive})
        known_positive_groups = {chunk_group[q.chunk_id] for q in judged if q.relevance == 1 and q.chunk_id in chunk_group}
        outside = [c for c in positives if c not in pool_set]
        if outside:
            exclusions["positive_outside_pool"] += len(outside)
            positives = [c for c in positives if c in pool_set]
        if not positives:
            exclusions["no_usable_positive"] += 1
            continue
        forbidden_groups = known_positive_groups
        usable = lambda chunk_id: chunk_group[chunk_id] not in forbidden_groups          # noqa: E731
        request = SearchRequest(query=query.text, sources=query.sources, versions=query.versions, candidate_k=config.hard_search_depth)
        ranked = [c.chunk.chunk_id for c in retriever.retrieve_within(request, pool_set)]
        hard = [c for c in ranked if usable(c)][: config.hard_negatives]
        taken = set(hard)
        view = retriever.eligible_chunk_ids(request)
        candidates = [c for c in pool.chunk_ids if c not in taken and usable(c) and (view is None or c in view)]
        rng = random.Random(f"{config.seed}:{query.query_id}:random")
        randoms = sorted(rng.sample(candidates, min(config.random_negatives, len(candidates))))
        if len(hard) < config.hard_negatives or len(randoms) < config.random_negatives:
            shortages.append({"query_id": query.query_id, "hard": len(hard), "random": len(randoms)})
        for chunk_id in positives:
            pairs.append(PairRecord(_pair_id(sampling_id, query.query_id, chunk_id, 1), query.query_id, chunk_id, 1, "reviewed_positive", None, sampling_id))
        for method, chosen in (("bm25", hard), ("random", randoms)):
            for chunk_id in chosen:
                pairs.append(PairRecord(_pair_id(sampling_id, query.query_id, chunk_id, 0), query.query_id, chunk_id, 0, "assumed_negative", method, sampling_id))
                sampling.append({"query_id": query.query_id, "positive_chunk_ids": positives, "negative_chunk_id": chunk_id,
                                 "negative_method": method, "label_status": "assumed_negative", "seed": config.seed})
        counts["positives"] += len(positives)
        counts["negatives_bm25"] += len(hard)
        counts["negatives_random"] += len(randoms)
        used += 1
    pairs.sort(key=lambda p: (p.query_id, p.chunk_id))
    keys = [(p.query_id, p.chunk_id) for p in pairs]
    if len(set(keys)) != len(keys):
        raise ManifestError("positive/negative conflict: a (query, chunk) pair was materialized twice")
    negatives = counts["negatives_bm25"] + counts["negatives_random"]
    report = {
        "queries_total": len(queries), "queries_used": used, "exclusions": dict(sorted(exclusions.items())),
        **counts, "negatives": negatives, "pairs_total": len(pairs), "shortage_queries": shortages,
        "pos_weight": (negatives / counts["positives"]) if counts["positives"] else None,
        "mining_pool_size": len(pool.chunk_ids), "mining_pool_id": pool.pool_id,
    }
    if report["pairs_total"] != counts["positives"] + negatives:
        raise ManifestError("pair count reconciliation failed")
    return PairSet(tuple(pairs), tuple(sorted(sampling, key=lambda r: (r["query_id"], r["negative_chunk_id"]))), report, pool, sampling_id)


def write_pair_artifacts(
    directory: Path, pairset: PairSet, *, corpus_id: str, index_id: str, queries_sha256: str, qrels_sha256: str,
    config: MiningConfig,
) -> dict[str, Any]:
    """Write pairs.jsonl, sampling.jsonl, mining-pool.json and a sealed manifest (manifest last)."""
    if not directory.is_dir():
        raise ManifestError(f"directory does not exist: {directory}")
    files = {
        "pairs.jsonl": jsonl_bytes(p.as_record() for p in pairset.pairs),
        "sampling.jsonl": jsonl_bytes(pairset.sampling),
        "mining-pool.json": pretty_json({"pool_id": pairset.pool.pool_id, "chunk_ids": list(pairset.pool.chunk_ids)}).encode("utf-8"),
    }
    manifest = seal_manifest({
        "schema_version": PAIRS_SCHEMA_VERSION, "kind": "pairs", "sampling_manifest_id": pairset.sampling_manifest_id,
        "corpus_id": corpus_id, "index_id": index_id, "queries_sha256": queries_sha256, "qrels_sha256": qrels_sha256,
        "mining": {"hard_negatives": config.hard_negatives, "random_negatives": config.random_negatives, "seed": config.seed, "hard_search_depth": config.hard_search_depth},
        "files": {name: {"sha256": sha256_hex(data)} for name, data in files.items()},
        "report": dict(pairset.report),
    })
    for name, data in files.items():
        write_atomic(directory / name, data, exclusive=True)
    write_atomic(directory / "pairs-manifest.json", pretty_json(manifest).encode("utf-8"), exclusive=True)
    return manifest


def load_pairs(manifest_path: Path, corpus: Corpus) -> tuple[dict[str, Any], tuple[PairRecord, ...], MiningPool]:
    """Load and fully validate pair artifacts: hashes, uniqueness, pool membership, count reconciliation."""
    where = "pairs manifest"
    base = manifest_path.resolve(strict=True).parent
    manifest = dict(as_object(loads_strict(read_utf8(manifest_path, where=where), where=where), where))
    exact_keys(manifest, ("schema_version", "kind", "sampling_manifest_id", "corpus_id", "index_id", "queries_sha256",
                          "qrels_sha256", "mining", "files", "report", "manifest_sha256"), where)
    if manifest["schema_version"] != PAIRS_SCHEMA_VERSION or manifest["kind"] != "pairs":
        raise ManifestError(f"{where}: expected kind='pairs', schema_version={PAIRS_SCHEMA_VERSION}")
    verify_manifest_hash(manifest, where=where)
    if manifest["corpus_id"] != corpus.corpus_id:
        raise ManifestError(f"{where}: built for a different corpus")
    listed = as_object(manifest["files"], f"{where}.files")
    exact_keys(listed, ("pairs.jsonl", "sampling.jsonl", "mining-pool.json"), f"{where}.files")
    for name, ref in listed.items():
        exact_keys(as_object(ref, f"{where}.files.{name}"), ("sha256",), f"{where}.files.{name}")
        path = base / name
        if not path.is_file() or sha256_file(path) != as_sha256(ref["sha256"], f"{where}.files.{name}.sha256"):
            raise ManifestError(f"{where}: {name} is missing or does not match its recorded SHA-256")
    pool_doc = as_object(loads_strict(read_utf8(base / "mining-pool.json", where="mining pool"), where="mining pool"), "mining pool")
    pool = MiningPool(tuple(pool_doc["chunk_ids"]))
    if pool_doc["pool_id"] != pool.pool_id or list(pool.chunk_ids) != sorted(set(pool.chunk_ids)):
        raise ManifestError("mining pool: pool_id or ordering does not match its chunk list")
    known = {chunk.chunk_id for chunk in corpus.chunks}
    if not set(pool.chunk_ids) <= known:
        raise ManifestError("mining pool: names chunks that are not in the corpus")
    pool_set, pairs, seen = set(pool.chunk_ids), [], set()
    for label, record in read_jsonl(base / "pairs.jsonl", "pairs.jsonl"):
        exact_keys(record, PAIR_FIELDS, label)
        value = record["label"]
        if isinstance(value, bool) or value not in (0, 1):
            raise ManifestError(f"{label}.label: must be 0 or 1")
        pair = PairRecord(as_id(record["pair_id"], f"{label}.pair_id"), as_id(record["query_id"], f"{label}.query_id"),
                          as_id(record["chunk_id"], f"{label}.chunk_id"), value, record["label_status"],
                          record["negative_method"], as_id(record["sampling_manifest_id"], f"{label}.sampling_manifest_id"))
        expected = (("reviewed_positive", None) if value == 1 else ("assumed_negative", pair.negative_method))
        if (pair.label_status, pair.negative_method if value == 0 else None) != expected or (value == 0 and pair.negative_method not in NEGATIVE_METHODS):
            raise ManifestError(f"{label}: label/label_status/negative_method are inconsistent")
        if pair.sampling_manifest_id != manifest["sampling_manifest_id"]:
            raise ManifestError(f"{label}: belongs to a different sampling manifest")
        if pair.pair_id != _pair_id(pair.sampling_manifest_id, pair.query_id, pair.chunk_id, pair.label):
            raise ManifestError(f"{label}: pair_id does not match its derived ID")
        if pair.chunk_id not in pool_set:
            raise ManifestError(f"{label}: chunk is outside the train-owned mining pool")
        if (pair.query_id, pair.chunk_id) in seen:
            raise ManifestError(f"{label}: duplicate (query_id, chunk_id)")
        seen.add((pair.query_id, pair.chunk_id))
        pairs.append(pair)
    report = as_object(manifest["report"], f"{where}.report")
    positives = sum(1 for p in pairs if p.label == 1)
    negatives = len(pairs) - positives
    if (report["pairs_total"], report["positives"], report["negatives"]) != (len(pairs), positives, negatives):
        raise ManifestError(f"{where}: reported counts do not reconcile with pairs.jsonl")
    return manifest, tuple(pairs), pool
