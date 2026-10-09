"""Training-visible inputs materialized from a verified prepared-data directory: train pairs and the validation set.

One loader shared by the campaign's `World` and the standalone `train`/`evaluate` commands, so every path that trains or
validates sees exactly the same pairs, candidates and labels. Only the prepared (training-visible) directory is read: no
test query, candidate or judgment is reachable from here.
"""

from dataclasses import dataclass
from pathlib import Path

from devsearch.artifacts.canonical import sha256_file
from devsearch.data.candidates import load_candidates
from devsearch.data.pairs import load_pairs
from devsearch.data.prepare import load_prepared
from devsearch.data.queries import Query, load_qrels, load_queries
from devsearch.retrieval.index_store import LoadedIndex
from devsearch.training.data import TrainPair, ValidationSet, build_train_pairs, build_validation_set


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


def load_training_inputs(prepared_manifest: Path, root: Path) -> TrainingInputs:
    prepared = load_prepared(Path(prepared_manifest), root)
    directory, loaded = prepared["directory"], prepared["loaded"]
    manifest_path = directory / "pairs" / "pairs-manifest.json"
    _, records, _ = load_pairs(manifest_path, loaded.corpus)
    train_queries = {q.query_id: q for q in load_queries(directory / "data" / "queries-train.jsonl")}
    validation_queries = list(load_queries(directory / "data" / "queries-validation.jsonl"))
    _, candidates = load_candidates(directory / "candidates" / "candidates-validation-manifest.json", loaded)
    validation = build_validation_set(candidates, {q.query_id: q for q in validation_queries},
                                      list(load_qrels(directory / "data" / "qrels-validation.jsonl")), loaded.corpus)
    return TrainingInputs(directory, loaded, prepared["manifest"]["protocol"], build_train_pairs(records, train_queries, loaded.corpus), validation,
                          validation_queries, sha256_file(manifest_path), candidates)

