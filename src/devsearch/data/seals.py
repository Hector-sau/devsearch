"""Review artifacts, the judgment bank and the two test seals (docs/02 §5, docs/10).

Artifacts (each a sealed JSON manifest plus hashed files; files are written exclusively, manifest last):

* `review-pool`     blinded `review-request.jsonl`, private `hidden-provenance.jsonl`
* `judgment-bank`   `qrels-v2.jsonl` (final shared judgments), `qrels-diff.json`
* `test-seal` stage 1: freezes test queries/groups/corpus/index/candidates/qrels-v1 and, for the
  prejudged protocol, the complete bank, before any training comparison
* `test-seal` stage 2: after systems are locked, verifies every ranked output is a permutation of its
  frozen candidate list, that the *entire* bank still covers every candidate and known positive, binds the
  locked systems to the unchanged bank and records the final qrels hash. It adds no judgment and
  selects no model.
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import (
    digest_of, loads_strict, pretty_json, read_utf8, seal_manifest, sha256_file, sha256_hex, verify_manifest_hash,
)
from devsearch.artifacts.files import jsonl_bytes, write_atomic
from devsearch.artifacts.schema import as_object, as_sha256, exact_keys
from devsearch.data.queries import Qrel, Query, load_qrels, qrel_record, validate_qrels
from devsearch.data.review import Bank, ReviewPool, hidden_records, qrels_diff, request_records
from devsearch.errors import ManifestError
from devsearch.retrieval.index_store import LoadedIndex

SEAL_SCHEMA_VERSION = 1


def _manifest(path: Path, kind: str, fields: Sequence[str], where: str) -> dict[str, Any]:
    manifest = dict(as_object(loads_strict(read_utf8(path, where=where), where=where), where))
    exact_keys(manifest, ("schema_version", "kind", *fields, "manifest_sha256"), where)
    if manifest["schema_version"] != SEAL_SCHEMA_VERSION or manifest["kind"] != kind:
        raise ManifestError(f"{where}: expected kind={kind!r}, schema_version={SEAL_SCHEMA_VERSION}")
    verify_manifest_hash(manifest, where=where)
    return manifest


def _check_files(base: Path, files: Mapping[str, Any], where: str) -> None:
    for name, ref in files.items():
        path = base / name
        if Path(name).name != name or not path.is_file() or sha256_file(path) != as_sha256(as_object(ref, where)["sha256"], where):
            raise ManifestError(f"{where}: {name} is missing or does not match its recorded SHA-256")


def write_review_pool(directory: Path, pool: ReviewPool, loaded: LoadedIndex, queries: Mapping[str, Query]) -> dict[str, Any]:
    if not directory.is_dir():
        raise ManifestError(f"directory does not exist: {directory}")
    requests = jsonl_bytes(request_records(pool, loaded, queries))
    hidden = jsonl_bytes([{"item_id": "_pool", "salt": pool.salt, "protocol": pool.protocol}] + hidden_records(pool))
    manifest = seal_manifest({
        "schema_version": SEAL_SCHEMA_VERSION, "kind": "review-pool", "protocol": pool.protocol, "pool_id": pool.pool_id,
        "salt_sha256": sha256_hex(pool.salt.encode()), "corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id,
        "files": {"review-request.jsonl": {"sha256": sha256_hex(requests)}, "hidden-provenance.jsonl": {"sha256": sha256_hex(hidden)}},
        "counts": {"items": len(pool.items), "queries": len({i.query_id for i in pool.items}),
                   "known_positive_items": sum(1 for i in pool.items if i.provenance.get("known_positive"))},
    })
    write_atomic(directory / "review-request.jsonl", requests, exclusive=True)
    write_atomic(directory / "hidden-provenance.jsonl", hidden, exclusive=True)
    write_atomic(directory / "review-pool-manifest.json", pretty_json(manifest).encode("utf-8"), exclusive=True)
    return manifest


def write_bank(
    directory: Path, pool: ReviewPool, bank: Bank, qrels_v1: Sequence[Qrel], loaded: LoadedIndex, queries: Sequence[Query],
) -> dict[str, Any]:
    """Seal the complete bank. Unresolved or missing items block sealing (never become label 0)."""
    if not directory.is_dir():
        raise ManifestError(f"directory does not exist: {directory}")
    if not bank.ready:
        shown = ", ".join(f"{item}:{reason}" for item, reason in bank.unresolved[:5])
        raise ManifestError(f"{len(bank.unresolved)} review item(s) unresolved or missing; sealing is blocked ({shown})")
    if len(bank.qrels) != len(pool.items):
        raise ManifestError("the bank does not hold exactly one judgment per review item")
    diff = qrels_diff(qrels_v1, bank.qrels, [q.query_id for q in queries])
    files = {
        "qrels-v2.jsonl": jsonl_bytes(qrel_record(q) for q in bank.qrels),
        "qrels-diff.json": pretty_json(diff).encode("utf-8"),
    }
    manifest = seal_manifest({
        "schema_version": SEAL_SCHEMA_VERSION, "kind": "judgment-bank", "protocol": pool.protocol, "pool_id": pool.pool_id,
        "corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id,
        "files": {name: {"sha256": sha256_hex(data)} for name, data in files.items()},
        "counts": {"items": len(bank.qrels), "positives": sum(q.relevance for q in bank.qrels), "queries": len(queries)},
        "reviewers": dict(bank.reviewer_counts), "diff_summary": {k: (len(v) if isinstance(v, list) else v) for k, v in diff.items()},
    })
    for name, data in files.items():
        write_atomic(directory / name, data, exclusive=True)
    write_atomic(directory / "judgment-bank-manifest.json", pretty_json(manifest).encode("utf-8"), exclusive=True)
    return manifest


def load_bank(manifest_path: Path, loaded: LoadedIndex, queries: Sequence[Query]) -> tuple[dict[str, Any], tuple[Qrel, ...]]:
    where = "judgment bank manifest"
    manifest = _manifest(manifest_path, "judgment-bank", ("protocol", "pool_id", "corpus_id", "index_id", "files", "counts", "reviewers", "diff_summary"), where)
    if (manifest["corpus_id"], manifest["index_id"]) != (loaded.corpus.corpus_id, loaded.index_id):
        raise ManifestError(f"{where}: built for a different corpus/index")
    base = manifest_path.resolve(strict=True).parent
    _check_files(base, as_object(manifest["files"], where), where)
    qrels = load_qrels(base / "qrels-v2.jsonl")
    validate_qrels(qrels, {q.query_id: q for q in queries}, loaded.corpus)
    if len(qrels) != manifest["counts"]["items"] or sum(q.relevance for q in qrels) != manifest["counts"]["positives"]:
        raise ManifestError(f"{where}: counts do not match qrels-v2.jsonl")
    return manifest, qrels


def verify_bank_covers(
    bank_qrels: Sequence[Qrel], candidate_records: Sequence[Mapping[str, Any]], qrels_v1: Sequence[Qrel],
) -> dict[str, int]:
    """Every frozen candidate and every known positive must have a bank judgment."""
    judged = {(q.query_id, q.chunk_id) for q in bank_qrels}
    required = {(r["query_id"], c["chunk_id"]) for r in candidate_records for c in r["candidates"]}
    required |= {(q.query_id, q.chunk_id) for q in qrels_v1 if q.confirmed_positive}
    missing = sorted(required - judged)
    if missing:
        raise ManifestError(f"the judgment bank does not cover {len(missing)} required pair(s), e.g. {missing[0]}")
    return {"required_pairs": len(required), "judged_pairs": len(judged)}


def verify_pooled_coverage(
    bank_qrels: Sequence[Qrel], outputs: Mapping[str, Sequence[Mapping[str, Any]]], qrels_v1: Sequence[Qrel], depth: int,
) -> dict[str, int]:
    """Post-rank protocol: every top-`depth` result of every locked system, and every known positive, must be judged."""
    judged = {(q.query_id, q.chunk_id) for q in bank_qrels}
    required = {(r["query_id"], item["chunk_id"]) for records in outputs.values() for r in records for item in r["ranking"][:depth]}
    required |= {(q.query_id, q.chunk_id) for q in qrels_v1 if q.confirmed_positive}
    missing = sorted(required - judged)
    if missing:
        raise ManifestError(f"the reviewed pool does not cover {len(missing)} required pair(s), e.g. {missing[0]}")
    return {"required_pairs": len(required), "judged_pairs": len(judged), "pool_depth": depth}


def verify_permutation_output(records: Sequence[Mapping[str, Any]], candidate_records: Sequence[Mapping[str, Any]], system: str) -> None:
    """A system may only reorder its query's frozen list: same chunks, one finite score each, dense ranks."""
    frozen = {r["query_id"]: [c["chunk_id"] for c in r["candidates"]] for r in candidate_records}
    if [r["query_id"] for r in records] != sorted(frozen):
        raise ManifestError(f"system {system!r}: output does not cover exactly the frozen queries")
    for record in records:
        ids = [item["chunk_id"] for item in record["ranking"]]
        if sorted(ids) != sorted(frozen[record["query_id"]]):
            raise ManifestError(f"system {system!r}, query {record['query_id']}: not a permutation of the frozen candidates")
        if [item["rank"] for item in record["ranking"]] != list(range(1, len(ids) + 1)):
            raise ManifestError(f"system {system!r}, query {record['query_id']}: ranks must be dense from 1")
        if not all(isinstance(item["score"], (int, float)) and not isinstance(item["score"], bool) and math.isfinite(item["score"]) for item in record["ranking"]):
            raise ManifestError(f"system {system!r}, query {record['query_id']}: every candidate needs one finite score")


def build_initial_seal(
    *, protocol: str, loaded: LoadedIndex, test_queries_sha256: str, test_group_ids: Sequence[str], candidates_manifest_sha256: str,
    qrels_v1_sha256: str, bank_manifest_sha256: str | None, eligibility_rules: Mapping[str, Any],
) -> dict[str, Any]:
    """Stage 1: frozen before any training comparison. The prejudged protocol requires the complete bank."""
    if protocol == "prejudged_fixed_candidates_v1" and bank_manifest_sha256 is None:
        raise ManifestError("the prejudged protocol seals the complete judgment bank with the initial test inputs")
    return seal_manifest({
        "schema_version": SEAL_SCHEMA_VERSION, "kind": "test-seal", "stage": 1, "protocol": protocol,
        "corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id, "test_queries_sha256": test_queries_sha256,
        "test_groups_sha256": digest_of(sorted(test_group_ids)), "candidates_manifest_sha256": candidates_manifest_sha256,
        "qrels_v1_sha256": qrels_v1_sha256, "bank_manifest_sha256": bank_manifest_sha256,
        "eligibility_rules": dict(eligibility_rules),
    })


def load_seal(path: Path, stage: int) -> dict[str, Any]:
    where = f"test seal (stage {stage})"
    common = ("stage", "protocol", "corpus_id", "index_id")
    fields = {1: (*common, "test_queries_sha256", "test_groups_sha256", "candidates_manifest_sha256", "qrels_v1_sha256", "bank_manifest_sha256", "eligibility_rules"),
              2: (*common, "initial_seal_sha256", "selection_id", "systems", "bank_manifest_sha256", "qrels_v2_sha256", "coverage")}[stage]
    seal = _manifest(path, "test-seal", fields, where)
    if seal["stage"] != stage:
        raise ManifestError(f"{where}: this is a stage {seal['stage']} seal")
    return seal


def build_final_seal(
    *, initial_seal: Mapping[str, Any], selection_id: str, outputs: Mapping[str, tuple[str, Sequence[Mapping[str, Any]]]],
    candidate_records: Sequence[Mapping[str, Any]], bank_manifest: Mapping[str, Any], bank_qrels: Sequence[Qrel], qrels_v1: Sequence[Qrel],
    pool_depth: int = 10,
) -> dict[str, Any]:
    """Stage 2: bind locked systems to the unchanged bank. `outputs`: system -> (file sha256, records).

    Prejudged protocol: the bank was sealed with the initial inputs and must be byte-identical now; it must cover every
    frozen candidate and known positive. Post-rank protocol: the initial seal holds no bank; the reviewed pool supplies it
    now and must cover every top-`pool_depth` result of every locked system plus the known positives. Neither path adds a
    judgment or selects a model.
    """
    prejudged = initial_seal["protocol"] == "prejudged_fixed_candidates_v1"
    if prejudged and initial_seal["bank_manifest_sha256"] != bank_manifest["manifest_sha256"]:
        raise ManifestError("the bank changed after the initial seal; sealing is refused")
    if not prejudged and initial_seal["bank_manifest_sha256"] is not None:
        raise ManifestError("the post-rank protocol seals no bank with the initial inputs")
    if not outputs:
        raise ManifestError("no locked system outputs to seal")
    for system, (_sha, records) in sorted(outputs.items()):
        verify_permutation_output(records, candidate_records, system)
    if prejudged:
        coverage = verify_bank_covers(bank_qrels, candidate_records, qrels_v1)
    else:
        coverage = verify_pooled_coverage(bank_qrels, {system: records for system, (_sha, records) in outputs.items()}, qrels_v1, pool_depth)
    return seal_manifest({
        "schema_version": SEAL_SCHEMA_VERSION, "kind": "test-seal", "stage": 2, "protocol": initial_seal["protocol"],
        "corpus_id": initial_seal["corpus_id"], "index_id": initial_seal["index_id"],
        "initial_seal_sha256": initial_seal["manifest_sha256"], "selection_id": selection_id,
        "systems": {system: sha for system, (sha, _r) in sorted(outputs.items())},
        "bank_manifest_sha256": bank_manifest["manifest_sha256"],
        "qrels_v2_sha256": bank_manifest["files"]["qrels-v2.jsonl"]["sha256"], "coverage": coverage,
    })
