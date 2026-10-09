"""Training-visible inputs materialized from a verified prepared-data directory: train pairs and the validation set.

One loader shared by the campaign's `World` and the standalone `train`/`evaluate` commands, so every path that trains or
validates sees exactly the same pairs, candidates and labels. Only the prepared (training-visible) directory is read: no
test query, candidate or judgment is reachable from here.
"""

from dataclasses import dataclass
from pathlib import Path

from devsearch.artifacts.canonical import sha256_file
from devsearch.artifacts.paths import ensure_inside
from devsearch.data.candidates import load_candidates
from devsearch.data.pairs import load_pairs
from devsearch.data.prepare import load_prepared
from devsearch.data.queries import Query, load_qrels, load_queries
from devsearch.errors import ManifestError
from devsearch.retrieval.index_store import LoadedIndex
from devsearch.training.data import GroupedTrainSet, TrainPair, ValidationSet, build_grouped_train_set, build_train_pairs, build_validation_set


@dataclass(frozen=True)
class TrainingInputs:
    directory: Path
    loaded: LoadedIndex
    protocol: str
    train_pairs: list[TrainPair]
    validation: ValidationSet
    validation_queries: list[Query]
    pairs_identity: str
    validation_candidates: list[dict]
    grouped_data: GroupedTrainSet | None = None


def load_training_inputs(prepared_manifest: Path, root: Path, *, grouped: bool = False,
                         pairs_manifest: Path | None = None, pairs_manifest_sha256: str | None = None) -> TrainingInputs:
    if ((pairs_manifest is None) != (pairs_manifest_sha256 is None)
            or (pairs_manifest is not None and not grouped)):
        raise ManifestError("pairs override requires grouped training and a pinned manifest SHA256")
    prepared = load_prepared(Path(prepared_manifest), root)
    directory, loaded = prepared["directory"], prepared["loaded"]
    manifest_path = directory / "pairs" / "pairs-manifest.json"
    if pairs_manifest is not None:
        manifest_path = ensure_inside(Path(pairs_manifest), root, what="grouped pairs manifest")
        if sha256_file(manifest_path) != pairs_manifest_sha256:
            raise ManifestError("grouped pairs manifest differs from the pinned SHA256")
    if grouped:
        for name in ("pairs.jsonl", "sampling.jsonl", "mining-pool.json"):
            ensure_inside(manifest_path.parent / name, root, what=f"grouped pair artifact {name}")
    pairs_doc, records, _ = load_pairs(manifest_path, loaded.corpus)
    train_queries = {q.query_id: q for q in load_queries(directory / "data" / "queries-train.jsonl")}
    validation_queries = list(load_queries(directory / "data" / "queries-validation.jsonl"))
    _, candidates = load_candidates(directory / "candidates" / "candidates-validation-manifest.json", loaded)
    validation = build_validation_set(candidates, {q.query_id: q for q in validation_queries},
                                      list(load_qrels(directory / "data" / "qrels-validation.jsonl")), loaded.corpus)
    grouped_data = None
    pairs = build_train_pairs(records, train_queries, loaded.corpus)
    if grouped:
        from devsearch.data.prepare import SPLIT_MANIFEST
        if (pairs_doc["index_id"] != loaded.index_id
                or pairs_doc["queries_sha256"] != sha256_file(directory / "data" / "queries-train.jsonl")
                or pairs_doc["qrels_sha256"] != sha256_file(directory / "data" / "qrels-train.jsonl")):
            raise ManifestError("grouped pairs do not bind the prepared index and training query/qrel files")
        split = prepared["split"]
        try:
            owners = {c.chunk_id: split["assignment"][split["merged_of"][c.evidence_group_id]] for c in loaded.corpus.chunks}
        except (KeyError, TypeError) as error:
            raise ManifestError("grouped inputs require complete merged-group ownership") from error
        sources = {"pairs_manifest": sha256_file(manifest_path),
                   "train_queries": sha256_file(directory / "data" / "queries-train.jsonl"),
                   "train_qrels": sha256_file(directory / "data" / "qrels-train.jsonl"),
                   "split_manifest": sha256_file(directory / SPLIT_MANIFEST),
                   "assignment": sha256_file(directory / "assignment.json"),
                   "prepared_manifest": sha256_file(prepared_manifest)}
        grouped_data = build_grouped_train_set(records, train_queries, loaded.corpus,
                                              list(load_qrels(directory / "data" / "qrels-train.jsonl")), owners, sources,
                                              merged_of=split["merged_of"])
        pairs = list(grouped_data.pairs)
    return TrainingInputs(directory, loaded, prepared["manifest"]["protocol"], pairs, validation,
                          validation_queries, sha256_file(manifest_path), candidates, grouped_data)
