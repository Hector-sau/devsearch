"""Ranked-output artifacts: each locked system's permutation of the frozen candidate lists, hash-sealed.

`ranked-<system>.jsonl` holds one record per query (`ranked_record` shape); `ranked-output-manifest.json`
binds every file by SHA-256, together with the candidate manifest, the validation selection and the
protocol it was produced under. Loading re-checks that each system only *permutes* its query's frozen
list, with dense ranks and one finite score per candidate (the seal does the same independently).
Standard library only.
"""

import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import loads_strict, pretty_json, read_utf8, seal_manifest, sha256_file, sha256_hex, verify_manifest_hash
from devsearch.artifacts.files import jsonl_bytes, new_staging, publish_staged, read_jsonl, write_atomic
from devsearch.artifacts.schema import as_object, exact_keys
from devsearch.errors import ManifestError

RANKED_SCHEMA_VERSION = 1
MANIFEST_NAME = "ranked-output-manifest.json"


def _check_permutation(records: Sequence[Mapping[str, Any]], candidate_records: Sequence[Mapping[str, Any]], system: str) -> None:
    frozen = {r["query_id"]: [c["chunk_id"] for c in r["candidates"]] for r in candidate_records}
    if [r["query_id"] for r in records] != sorted(frozen):
        raise ManifestError(f"system {system!r}: output does not cover exactly the frozen queries")
    for record in records:
        items = record["ranking"]
        if sorted(i["chunk_id"] for i in items) != sorted(frozen[record["query_id"]]):
            raise ManifestError(f"system {system!r}, query {record['query_id']}: not a permutation of the frozen candidates")
        if [i["rank"] for i in items] != list(range(1, len(items) + 1)):
            raise ManifestError(f"system {system!r}, query {record['query_id']}: ranks must be dense from 1")
        if not all(isinstance(i["score"], (int, float)) and not isinstance(i["score"], bool) and math.isfinite(i["score"]) for i in items):
            raise ManifestError(f"system {system!r}, query {record['query_id']}: every candidate needs one finite score")


def write_ranked_outputs(
    directory: Path, systems: Mapping[str, tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]], candidate_records: Sequence[Mapping[str, Any]],
    *, candidates_manifest_sha256: str, selection_id: str, protocol_id: str,
) -> dict[str, Any]:
    """`systems`: name -> (metadata, ranked records). Nothing is written unless every system is a valid permutation."""
    if not directory.is_dir():
        raise ManifestError(f"directory does not exist: {directory}")
    if not systems:
        raise ManifestError("no system outputs to write")
    for name, (_meta, records) in sorted(systems.items()):
        if Path(name).name != name or not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise ManifestError(f"system name {name!r} is not a plain identifier")
        _check_permutation(records, candidate_records, name)
    staging = new_staging(directory)
    try:
        entries = {}
        for name, (meta, records) in sorted(systems.items()):
            data = jsonl_bytes(records)
            write_atomic(staging / f"ranked-{name}.jsonl", data, exclusive=True)
            entries[name] = {"file": f"ranked-{name}.jsonl", "sha256": sha256_hex(data), "meta": dict(meta), "queries": len(records)}
        manifest = seal_manifest({
            "schema_version": RANKED_SCHEMA_VERSION, "kind": "ranked-outputs", "candidates_manifest_sha256": candidates_manifest_sha256,
            "selection_id": selection_id, "protocol_id": protocol_id, "systems": entries,
        })
        write_atomic(staging / MANIFEST_NAME, pretty_json(manifest).encode("utf-8"), exclusive=True)
        publish_staged(staging, directory, manifests=[MANIFEST_NAME])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest


def load_ranked_outputs(manifest_path: Path, candidate_records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    where = "ranked-output manifest"
    manifest = dict(as_object(loads_strict(read_utf8(manifest_path, where=where), where=where), where))
    exact_keys(manifest, ("schema_version", "kind", "candidates_manifest_sha256", "selection_id", "protocol_id", "systems", "manifest_sha256"), where)
    if manifest["schema_version"] != RANKED_SCHEMA_VERSION or manifest["kind"] != "ranked-outputs":
        raise ManifestError(f"{where}: expected kind='ranked-outputs', schema_version={RANKED_SCHEMA_VERSION}")
    verify_manifest_hash(manifest, where=where)
    base = manifest_path.resolve(strict=True).parent
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, entry in as_object(manifest["systems"], where).items():
        exact_keys(entry, ("file", "sha256", "meta", "queries"), f"{where}.systems.{name}")
        path = base / entry["file"]
        if Path(entry["file"]).name != entry["file"] or not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ManifestError(f"{where}: {entry['file']} is missing or does not match its recorded SHA-256")
        records = [record for _, record in read_jsonl(path, path.name)]
        _check_permutation(records, candidate_records, name)
        loaded[name] = records
    return manifest, loaded
