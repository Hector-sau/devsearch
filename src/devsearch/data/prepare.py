"""Dataset preparation stages behind `python -m devsearch.prepare_data` (docs/02, docs/10 §3).

Three stages, each producing immutable, hash-sealed artifacts that later stages and the execution
controller verify instead of trusting:

1. `prepare_groups`      corpus -> merged evidence groups -> seeded split assignment -> independent audit.
                         Writes `split-manifest.json` and an ownership-annotated `corpus/`.
2. `prepare_materialize` drafts (queries/qrels) -> validated per-split files, BM25 index, train-only
                         mining pool + unique BCE pairs, fixed natural candidates, and (prejudged protocol)
                         the blinded review pool. Training-visible output goes to the *prepared* directory;
                         test queries, qrels-v1, test candidates and the review pool go to the separate
                         *evaluation* directory.
3. `prepare_bank`        reviewer responses -> complete judgment bank -> stage-1 test seal. Missing or
                         unresolved items block sealing; they never become label 0.

Nothing here calls a generator or reviewer: drafts and review responses are inputs, labelled by their
own provenance. Test material never enters the prepared directory.
"""

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import (
    digest_of, loads_strict, pretty_json, read_utf8, seal_manifest, sha256_file, sha256_hex, verify_manifest_hash,
)
from devsearch.artifacts.files import jsonl_bytes, new_staging, publish_staged, read_jsonl, write_atomic
from devsearch.artifacts.paths import ensure_inside
from devsearch.artifacts.schema import as_object, exact_keys
from devsearch.data.candidates import build_candidates, load_candidates, write_candidate_artifacts
from devsearch.data.config import DataConfig
from devsearch.data.corpus import MANIFEST_NAME as CORPUS_MANIFEST, Corpus, load_corpus, reowned_documents, write_corpus
from devsearch.data.groups import GroupMerge, assign_groups, audit_assignment, merge_evidence_groups, owner_of_chunks
from devsearch.data.pairs import build_mining_pool, mine_pairs, sampling_identity, write_pair_artifacts
from devsearch.data.queries import Qrel, Query, load_qrels, load_queries, qrel_record, query_record, validate_qrels, validate_queries
from devsearch.data.review import build_review_pool, parse_response, prejudged_universe, resolve_bank
from devsearch.data.seals import build_initial_seal, write_bank, write_review_pool
from devsearch.errors import ManifestError
from devsearch.retrieval.index_store import LoadedIndex, build_index, load_index

PREPARE_SCHEMA_VERSION = 1
SPLIT_MANIFEST = "split-manifest.json"
PREPARED_MANIFEST = "prepared-data-manifest.json"
EVALUATION_MANIFEST = "evaluation-manifest.json"
INITIAL_SEAL = "initial-seal.json"
TRAINING_SPLITS = ("train", "validation", "development")        # splits whose files may live in the prepared directory
ELIGIBILITY_RULES = {
    "queries_without_confirmed_positive": "kept and counted; excluded from ranking denominators, never silently deleted",
    "unresolved_review_items": "block sealing; never converted to label 0",
    "exclusions": "only by a predeclared answerability/evidence rule, logged independently of ranking success",
}


def _read_json(path: Path, where: str) -> dict[str, Any]:
    return dict(as_object(loads_strict(read_utf8(path, where=where), where=where), where))


def _sealed(path: Path, kind: str, fields: Sequence[str], where: str) -> dict[str, Any]:
    manifest = _read_json(path, where)
    exact_keys(manifest, ("schema_version", "kind", *fields, "manifest_sha256"), where)
    if manifest["schema_version"] != PREPARE_SCHEMA_VERSION or manifest["kind"] != kind:
        raise ManifestError(f"{where}: expected kind={kind!r}, schema_version={PREPARE_SCHEMA_VERSION}")
    verify_manifest_hash(manifest, where=where)
    return manifest


def _load_links(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return []
    links = []
    for label, record in read_jsonl(path, str(path)):
        exact_keys(record, ("a", "b"), label)
        if not (isinstance(record["a"], str) and isinstance(record["b"], str)):
            raise ManifestError(f"{label}: a and b must be evidence group IDs")
        links.append((record["a"], record["b"]))
    return links


def _merge(config: DataConfig, corpus: Corpus) -> GroupMerge:
    return merge_evidence_groups(
        corpus, near_duplicate_threshold=config.near_duplicate_threshold, shingle_size=config.shingle_size,
        extra_links=_load_links(config.links),
    )


def _counts(corpus: Corpus, merge: GroupMerge, assignment: Mapping[str, str]) -> dict[str, dict[str, int]]:
    owners = owner_of_chunks(corpus, merge, assignment)
    counts: dict[str, dict[str, int]] = {}
    for merged, split in assignment.items():
        entry = counts.setdefault(split, {"merged_groups": 0, "evidence_groups": 0, "chunks": 0})
        entry["merged_groups"] += 1
        entry["evidence_groups"] += len(merge.members[merged])
    for split in owners.values():
        counts[split]["chunks"] += 1
    return dict(sorted(counts.items()))


# -- stage 1 ---------------------------------------------------------------------------------------------
def prepare_groups(config: DataConfig, corpus_manifest: Path, directory: Path, root: Path) -> dict[str, Any]:
    """Merge, assign and audit evidence groups; publish `split-manifest.json` and the assigned `corpus/`."""
    root = root.resolve(strict=True)
    directory = ensure_inside(directory, root, what="prepared directory")
    corpus_path = ensure_inside(corpus_manifest, root, what="corpus manifest")
    if not directory.is_dir():
        raise ManifestError(f"prepared directory does not exist: {directory}")
    corpus = load_corpus(corpus_path)
    merge = _merge(config, corpus)
    declared = {chunk.chunk_id: chunk.owner_split for chunk in corpus.chunks}
    if all(owner is None for owner in declared.values()):
        assignment, origin = assign_groups(merge, targets=config.target_map, seed=config.group_seed), "assigned"
    elif all(owner is not None for owner in declared.values()):
        assignment, origin = {}, "adopted"
        for merged_id, chunk_ids in merge.chunks_of.items():
            owners = {declared[chunk_id] for chunk_id in chunk_ids}
            if len(owners) != 1:
                raise ManifestError(f"merged group {merged_id} spans declared owners {sorted(owners)}; fix the corpus or its links")
            assignment[merged_id] = next(iter(owners))
    else:
        raise ManifestError("the corpus is partially assigned (some chunks have an owner, some do not); assign all or none")
    audit = audit_assignment(
        corpus, merge, assignment, near_duplicate_threshold=config.near_duplicate_threshold, shingle_size=config.shingle_size,
    )
    if not audit.ok:
        shown = "; ".join(audit.violations[:3])
        raise ManifestError(f"the split audit found {len(audit.violations)} violation(s): {shown}")
    owners = owner_of_chunks(corpus, merge, assignment)
    staging = new_staging(directory)
    try:
        (staging / "corpus").mkdir()
        documents, chunker = reowned_documents(corpus, owners)
        corpus_out = write_corpus(staging / "corpus", documents, chunker_version=chunker)
        assignment_bytes = pretty_json({
            "merged_of": dict(sorted(merge.merged_of.items())),
            "members": {k: list(v) for k, v in merge.members.items()}, "assignment": dict(sorted(assignment.items())),
        }).encode("utf-8")
        audit_bytes = pretty_json({"ok": True, "violations": [], "counts": dict(audit.counts)}).encode("utf-8")
        write_atomic(staging / "assignment.json", assignment_bytes, exclusive=True)
        write_atomic(staging / "audit.json", audit_bytes, exclusive=True)
        manifest = seal_manifest({
            "schema_version": PREPARE_SCHEMA_VERSION, "kind": "split", "owner_origin": origin,
            "corpus_in_id": corpus.corpus_id,
            "corpus": {"corpus_id": corpus_out["corpus_id"], "manifest_path": f"corpus/{CORPUS_MANIFEST}",
                       "manifest_file_sha256": sha256_file(staging / "corpus" / CORPUS_MANIFEST)},
            "method": {**dict(merge.method), "merges_by_rule": dict(merge.merges_by_rule), "seed": config.group_seed,
                       "targets": {k: v for k, v in config.targets}},
            "files": {"assignment.json": {"sha256": sha256_hex(assignment_bytes)}, "audit.json": {"sha256": sha256_hex(audit_bytes)}},
            "counts": _counts(corpus, merge, assignment), "config_sha256": config.config_sha256,
        })
        write_atomic(staging / SPLIT_MANIFEST, pretty_json(manifest).encode("utf-8"), exclusive=True)
        publish_staged(staging, directory, manifests=[SPLIT_MANIFEST])
    finally:
        _discard(staging)
    return load_split(directory / SPLIT_MANIFEST)["manifest"]


def _discard(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def load_split(manifest_path: Path) -> dict[str, Any]:
    """Verify a split manifest, its files and corpus; returns manifest, corpus, merge input and assignment."""
    where = "split manifest"
    manifest = _sealed(manifest_path, "split", ("owner_origin", "corpus_in_id", "corpus", "method", "files", "counts", "config_sha256"), where)
    base = manifest_path.resolve(strict=True).parent
    for name, ref in as_object(manifest["files"], where).items():
        if Path(name).name != name or sha256_file(base / name) != ref["sha256"]:
            raise ManifestError(f"{where}: {name} is missing or does not match its recorded SHA-256")
    corpus_path = base / manifest["corpus"]["manifest_path"]
    if sha256_file(corpus_path) != manifest["corpus"]["manifest_file_sha256"]:
        raise ManifestError(f"{where}: the corpus manifest changed since the split was written")
    corpus = load_corpus(corpus_path)
    if corpus.corpus_id != manifest["corpus"]["corpus_id"]:
        raise ManifestError(f"{where}: corpus_id does not match the referenced corpus")
    assignment_doc = _read_json(base / "assignment.json", "assignment.json")
    return {"manifest": manifest, "corpus": corpus, "corpus_path": corpus_path, "assignment": dict(assignment_doc["assignment"]),
            "merged_of": dict(assignment_doc["merged_of"]), "members": {k: tuple(v) for k, v in assignment_doc["members"].items()}}


# -- stage 2 ---------------------------------------------------------------------------------------------
def _split_files(queries: Sequence[Query], qrels: Sequence[Qrel]) -> dict[str, dict[str, bytes]]:
    by_split: dict[str, dict[str, bytes]] = {}
    split_of = {q.query_id: q.split for q in queries}
    for split in sorted({q.split for q in queries}):
        members = [q for q in queries if q.split == split]
        judged = [r for r in qrels if split_of.get(r.query_id) == split]
        by_split[split] = {
            f"queries-{split}.jsonl": jsonl_bytes(query_record(q) for q in sorted(members, key=lambda q: q.query_id)),
            f"qrels-{split}.jsonl": jsonl_bytes(qrel_record(r) for r in sorted(judged, key=lambda r: (r.query_id, r.chunk_id))),
        }
    return by_split


def _require_consistent(config: DataConfig, split: Mapping[str, Any], queries: Sequence[Query], qrels: Sequence[Qrel]) -> GroupMerge:
    corpus: Corpus = split["corpus"]
    merge = _merge(config, corpus)
    if dict(merge.merged_of) != split["merged_of"]:
        raise ManifestError("regrouping the corpus no longer reproduces the recorded groups; the links or thresholds changed")
    by_id = {q.query_id: q for q in queries}
    if len(by_id) != len(queries):
        raise ManifestError("duplicate query_id in the query draft")
    validate_queries(queries, corpus)
    validate_qrels(qrels, by_id, corpus)
    audit = audit_assignment(
        corpus, merge, split["assignment"], queries, near_duplicate_threshold=config.near_duplicate_threshold,
        shingle_size=config.shingle_size,
    )
    if not audit.ok:
        raise ManifestError(f"the split audit with queries found {len(audit.violations)} violation(s): " + "; ".join(audit.violations[:3]))
    owners = owner_of_chunks(corpus, merge, split["assignment"])
    for qrel in qrels:
        if qrel.confirmed_positive and owners.get(qrel.chunk_id) != by_id[qrel.query_id].split:
            raise ManifestError(f"qrel ({qrel.query_id}, {qrel.chunk_id}): a positive owned by another split would leak evidence across splits")
    for split_name in ("train", "validation", "test"):
        members = [q for q in queries if q.split == split_name]
        if not members:
            raise ManifestError(f"the draft holds no {split_name} queries; every formal split needs questions")
        if not any(r.confirmed_positive and by_id[r.query_id].split == split_name for r in qrels):
            raise ManifestError(f"the {split_name} split has no confirmed positive; nothing can be ranked or trained on")
    return merge


def prepare_materialize(
    config: DataConfig, split_manifest: Path, evaluation_directory: Path, root: Path, evaluation_root: Path | None = None,
) -> dict[str, Any]:
    """Validate the drafts and write every derived artifact; returns the prepared-data manifest."""
    root = root.resolve(strict=True)
    evaluation_root = (evaluation_root or root).resolve(strict=True)
    split_manifest = ensure_inside(split_manifest, root, what="split manifest")
    prepared = split_manifest.parent
    evaluation = ensure_inside(evaluation_directory, evaluation_root, what="evaluation directory")
    if not evaluation.is_dir():
        raise ManifestError(f"evaluation directory does not exist: {evaluation}")
    if evaluation == prepared or prepared in evaluation.parents or evaluation in prepared.parents:
        raise ManifestError("the evaluation directory must be separate from the prepared (training-visible) directory")
    split = load_split(split_manifest)
    corpus: Corpus = split["corpus"]
    queries = tuple(q for _, path in config.queries for q in load_queries(path))
    qrels = tuple(r for _, path in config.qrels for r in load_qrels(path))
    for split_name, path in config.queries:
        if any(q.split != split_name for q in load_queries(path)):
            raise ManifestError(f"{path.name} is listed for the {split_name} split but holds queries of another split")
    merge = _require_consistent(config, split, queries, qrels)
    by_id = {q.query_id: q for q in queries}
    files = _split_files(queries, qrels)

    (prepared / "index").mkdir(exist_ok=True)
    index_manifest, _ = build_index(config.index, split["corpus_path"], prepared / "index" / "index-manifest.json", root)
    loaded = load_index(prepared / "index" / "index-manifest.json", root)

    train_q = [q for q in queries if q.split == "train"]
    train_r = [r for r in qrels if by_id[r.query_id].split == "train"]
    q_sha = {s: sha256_hex(f[f"queries-{s}.jsonl"]) for s, f in files.items()}
    r_sha = {s: sha256_hex(f[f"qrels-{s}.jsonl"]) for s, f in files.items()}
    pool = build_mining_pool(corpus, merge, split["assignment"])
    sampling = sampling_identity(config.mining, pool, corpus_id=corpus.corpus_id, index_id=loaded.index_id,
                                 queries_sha256=q_sha["train"], qrels_sha256=r_sha["train"])
    pairset = mine_pairs(corpus=corpus, retriever=loaded.retriever(), merge=merge, pool=pool, queries=train_q, qrels=train_r,
                         config=config.mining, sampling_id=sampling)

    validation_q = [q for q in queries if q.split == "validation"]
    test_q = [q for q in queries if q.split == "test"]
    test_r = [r for r in qrels if by_id[r.query_id].split == "test"]
    validation_candidates = build_candidates(loaded, validation_q, config.candidate_depth)
    test_candidates = build_candidates(loaded, test_q, config.candidate_depth)

    staging, eval_staging = new_staging(prepared), new_staging(evaluation)
    try:
        (staging / "data").mkdir()
        for split_name in TRAINING_SPLITS:
            for filename, data in files.get(split_name, {}).items():
                write_atomic(staging / "data" / filename, data, exclusive=True)
        (staging / "pairs").mkdir()
        pairs_manifest = write_pair_artifacts(
            staging / "pairs", pairset, corpus_id=corpus.corpus_id, index_id=loaded.index_id, queries_sha256=q_sha["train"],
            qrels_sha256=r_sha["train"], config=config.mining)
        (staging / "candidates").mkdir()
        validation_manifest = write_candidate_artifacts(
            staging / "candidates", "validation", validation_candidates, loaded, queries_sha256=q_sha["validation"])
        # test side: separate directory, never copied into the prepared one
        for filename, data in files["test"].items():
            write_atomic(eval_staging / filename, data, exclusive=True)
        (eval_staging / "candidates").mkdir()
        test_manifest = write_candidate_artifacts(
            eval_staging / "candidates", "test", test_candidates, loaded, queries_sha256=q_sha["test"])
        review_manifest = None
        if config.protocol == "prejudged_fixed_candidates_v1":
            universe = prejudged_universe(loaded, test_q, test_candidates, test_r)
            review_pool = build_review_pool(config.protocol, config.salt, universe)
            (eval_staging / "review").mkdir()
            review_manifest = write_review_pool(eval_staging / "review", review_pool, loaded, by_id)
        test_groups = sorted({g for q in test_q for g in q.evidence_group_ids})
        evaluation_manifest = seal_manifest({
            "schema_version": PREPARE_SCHEMA_VERSION, "kind": "evaluation-inputs", "protocol": config.protocol,
            "corpus_id": corpus.corpus_id, "index_id": loaded.index_id,
            "queries": {"path": "queries-test.jsonl", "sha256": q_sha["test"], "count": len(test_q)},
            "qrels_v1": {"path": "qrels-test.jsonl", "sha256": r_sha["test"], "count": len(test_r)},
            "candidates": {"path": "candidates/candidates-test-manifest.json", "manifest_sha256": test_manifest["manifest_sha256"]},
            "review_pool": None if review_manifest is None else {"path": "review/review-pool-manifest.json", "manifest_sha256": review_manifest["manifest_sha256"]},
            "test_groups_sha256": digest_of(test_groups), "config_sha256": config.config_sha256,
        })
        write_atomic(eval_staging / EVALUATION_MANIFEST, pretty_json(evaluation_manifest).encode("utf-8"), exclusive=True)
        prepared_manifest = seal_manifest({
            "schema_version": PREPARE_SCHEMA_VERSION, "kind": "prepared-data", "protocol": config.protocol,
            "split_manifest_sha256": sha256_file(split_manifest), "config_sha256": config.config_sha256,
            "corpus": {"corpus_id": corpus.corpus_id, "manifest_path": "corpus/corpus-manifest.json"},
            "index": {"index_id": loaded.index_id, "manifest_path": "index/index-manifest.json",
                      "manifest_sha256": index_manifest["manifest_sha256"]},
            "data": {name: {"sha256": sha256_hex(data), "count": sum(1 for _ in data.splitlines())}
                     for split_name in TRAINING_SPLITS for name, data in sorted(files.get(split_name, {}).items())},
            "pairs": {"manifest_path": "pairs/pairs-manifest.json", "manifest_sha256": pairs_manifest["manifest_sha256"],
                      "sampling_manifest_id": pairs_manifest["sampling_manifest_id"]},
            "validation_candidates": {"manifest_path": "candidates/candidates-validation-manifest.json",
                                      "manifest_sha256": validation_manifest["manifest_sha256"]},
            "evaluation_commitment": {"manifest_sha256": evaluation_manifest["manifest_sha256"], "test_queries": len(test_q),
                                      "test_groups": len(test_groups)},
            "counts": {split_name: {"queries": sum(1 for q in queries if q.split == split_name),
                                    "confirmed_positives": sum(1 for r in qrels if r.confirmed_positive and by_id[r.query_id].split == split_name)}
                       for split_name in ("train", "validation", "test", "development") if any(q.split == split_name for q in queries)},
        })
        write_atomic(staging / PREPARED_MANIFEST, pretty_json(prepared_manifest).encode("utf-8"), exclusive=True)
        publish_staged(eval_staging, evaluation, manifests=[EVALUATION_MANIFEST])
        publish_staged(staging, prepared, manifests=[PREPARED_MANIFEST])
    finally:
        _discard(staging)
        _discard(eval_staging)
    return load_prepared(prepared / PREPARED_MANIFEST, root)["manifest"]


_PREPARED_FIELDS = ("protocol", "split_manifest_sha256", "config_sha256", "corpus", "index", "data", "pairs",
                    "validation_candidates", "evaluation_commitment", "counts")


def load_prepared(manifest_path: Path, root: Path) -> dict[str, Any]:
    """Verify the training-visible prepared data: hashes of every listed file and nested manifest."""
    where = "prepared-data manifest"
    root = root.resolve(strict=True)
    path = ensure_inside(manifest_path, root, what=where)
    manifest = _sealed(path, "prepared-data", _PREPARED_FIELDS, where)
    base = path.parent
    split = load_split(base / SPLIT_MANIFEST)
    if sha256_file(base / SPLIT_MANIFEST) != manifest["split_manifest_sha256"]:
        raise ManifestError(f"{where}: the split manifest changed since the data was materialized")
    loaded = load_index(base / manifest["index"]["manifest_path"], root)
    if loaded.manifest["manifest_sha256"] != manifest["index"]["manifest_sha256"] or loaded.index_id != manifest["index"]["index_id"]:
        raise ManifestError(f"{where}: the index changed since the data was materialized")
    if loaded.corpus.corpus_id != manifest["corpus"]["corpus_id"]:
        raise ManifestError(f"{where}: the index was built for a different corpus")
    for name, ref in manifest["data"].items():
        if Path(name).name != name or sha256_file(base / "data" / name) != ref["sha256"]:
            raise ManifestError(f"{where}: data/{name} is missing or does not match its recorded SHA-256")
    for key in ("pairs", "validation_candidates"):
        if sha256_manifest(base / manifest[key]["manifest_path"]) != manifest[key]["manifest_sha256"]:
            raise ManifestError(f"{where}: {key} manifest does not match its recorded hash")
    return {"manifest": manifest, "split": split, "loaded": loaded, "directory": base}


def sha256_manifest(path: Path) -> str:
    """The `manifest_sha256` field of a sealed manifest, after checking it matches the content."""
    manifest = _read_json(path, str(path))
    verify_manifest_hash(manifest, where=str(path))
    return str(manifest["manifest_sha256"])


def load_evaluation(manifest_path: Path, root: Path, loaded: LoadedIndex) -> dict[str, Any]:
    """Verify the test-side inputs against the loaded index. Called only by evaluation-phase code."""
    where = "evaluation manifest"
    root = root.resolve(strict=True)
    path = ensure_inside(manifest_path, root, what=where)
    manifest = _sealed(path, "evaluation-inputs", ("protocol", "corpus_id", "index_id", "queries", "qrels_v1", "candidates",
                                                    "review_pool", "test_groups_sha256", "config_sha256"), where)
    base = path.parent
    if (manifest["corpus_id"], manifest["index_id"]) != (loaded.corpus.corpus_id, loaded.index_id):
        raise ManifestError(f"{where}: built for a different corpus/index")
    for key in ("queries", "qrels_v1"):
        ref = manifest[key]
        if sha256_file(base / ref["path"]) != ref["sha256"]:
            raise ManifestError(f"{where}: {ref['path']} does not match its recorded SHA-256")
    queries, qrels = load_queries(base / manifest["queries"]["path"]), load_qrels(base / manifest["qrels_v1"]["path"])
    if sha256_manifest(base / manifest["candidates"]["path"]) != manifest["candidates"]["manifest_sha256"]:
        raise ManifestError(f"{where}: candidates manifest does not match its recorded hash")
    return {"manifest": manifest, "queries": queries, "qrels": qrels, "directory": base}


# -- stage 3 ---------------------------------------------------------------------------------------------
def _rebuild_pool(config: DataConfig, evaluation: Mapping[str, Any], loaded: LoadedIndex):
    base = evaluation["directory"]
    queries = evaluation["queries"]
    _, candidates = load_candidates(base / evaluation["manifest"]["candidates"]["path"], loaded)
    universe = prejudged_universe(loaded, queries, candidates, evaluation["qrels"])
    pool = build_review_pool(config.protocol, config.salt, universe)
    recorded = _read_json(base / evaluation["manifest"]["review_pool"]["path"], "review pool manifest")
    if recorded["pool_id"] != pool.pool_id:
        raise ManifestError("the review pool rebuilt from the evaluation inputs and salt differs from the recorded pool")
    return pool, candidates


def prepare_bank(
    config: DataConfig, prepared_manifest: Path, evaluation_manifest: Path, reviews: Path, root: Path, evaluation_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve reviewer responses into the complete bank and write the stage-1 test seal.

    The seal lands in the prepared directory (it holds hashes only); the bank stays in the evaluation
    directory. Unresolved, missing or disagreeing items block sealing and write nothing.
    """
    if config.protocol != "prejudged_fixed_candidates_v1":
        raise ManifestError("the post-rank protocol has no pre-training bank; its seal is written by `seal_initial`")
    root = root.resolve(strict=True)
    evaluation_root = (evaluation_root or root).resolve(strict=True)
    prepared = load_prepared(prepared_manifest, root)
    loaded: LoadedIndex = prepared["loaded"]
    evaluation = load_evaluation(ensure_inside(evaluation_manifest, evaluation_root, what="evaluation manifest"), evaluation_root, loaded)
    pool, candidates = _rebuild_pool(config, evaluation, loaded)
    lengths = {item.item_id: len(next(c.text for c in loaded.corpus.chunks if c.chunk_id == item.chunk_id)) for item in pool.items}
    responses = [parse_response(record, label, lengths) for label, record in read_jsonl(reviews, "reviews")]
    bank = resolve_bank(pool, responses, loaded, min_primaries=config.min_primary_reviews, protocol=config.review_protocol)
    queries = list(evaluation["queries"])
    base = evaluation["directory"]
    bank_dir = base / "bank"
    bank_dir.mkdir(exist_ok=True)
    staging = new_staging(bank_dir)
    try:
        bank_manifest = write_bank(staging, pool, bank, evaluation["qrels"], loaded, queries)
        publish_staged(staging, bank_dir, manifests=["judgment-bank-manifest.json"])
    finally:
        _discard(staging)
    seal = build_initial_seal(
        protocol=config.protocol, loaded=loaded, test_queries_sha256=evaluation["manifest"]["queries"]["sha256"],
        test_group_ids=_test_group_ids(queries), candidates_manifest_sha256=evaluation["manifest"]["candidates"]["manifest_sha256"],
        qrels_v1_sha256=evaluation["manifest"]["qrels_v1"]["sha256"], bank_manifest_sha256=bank_manifest["manifest_sha256"],
        eligibility_rules=ELIGIBILITY_RULES)
    seal_dir = prepared["directory"]
    staging = new_staging(seal_dir)
    try:
        write_atomic(staging / INITIAL_SEAL, pretty_json(seal).encode("utf-8"), exclusive=True)
        publish_staged(staging, seal_dir, manifests=[INITIAL_SEAL])
    finally:
        _discard(staging)
    return seal


def prepare_initial_seal_postrank(config: DataConfig, prepared_manifest: Path, evaluation_manifest: Path, root: Path, evaluation_root: Path | None = None) -> dict[str, Any]:
    """Post-rank protocol: the stage-1 seal freezes test inputs now; judging happens after ranking."""
    if config.protocol != "postrank_pool_v1":
        raise ManifestError("this seal is for the post-rank protocol; the prejudged protocol seals through `prepare_bank`")
    root = root.resolve(strict=True)
    evaluation_root = (evaluation_root or root).resolve(strict=True)
    prepared = load_prepared(prepared_manifest, root)
    loaded: LoadedIndex = prepared["loaded"]
    evaluation = load_evaluation(ensure_inside(evaluation_manifest, evaluation_root, what="evaluation manifest"), evaluation_root, loaded)
    seal = build_initial_seal(
        protocol=config.protocol, loaded=loaded, test_queries_sha256=evaluation["manifest"]["queries"]["sha256"],
        test_group_ids=_test_group_ids(list(evaluation["queries"])), candidates_manifest_sha256=evaluation["manifest"]["candidates"]["manifest_sha256"],
        qrels_v1_sha256=evaluation["manifest"]["qrels_v1"]["sha256"], bank_manifest_sha256=None, eligibility_rules=ELIGIBILITY_RULES)
    seal_dir = prepared["directory"]
    staging = new_staging(seal_dir)
    try:
        write_atomic(staging / INITIAL_SEAL, pretty_json(seal).encode("utf-8"), exclusive=True)
        publish_staged(staging, seal_dir, manifests=[INITIAL_SEAL])
    finally:
        _discard(staging)
    return seal


def _test_group_ids(queries: Sequence[Query]) -> list[str]:
    return sorted({g for q in queries for g in q.evidence_group_ids})
