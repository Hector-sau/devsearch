"""The prepared bundle: every training-visible input of a campaign, tied together by one sealed manifest (docs/10 §3).

Layout (all relative, no absolute host paths):

    <bundle>/bundle-manifest.json
    <bundle>/prepared/...            the prepare_data output: corpus, index, split, drafts, pairs, validation candidates, initial seal
    <bundle>/model-assets/...        pinned encoder + tokenizer (or the diagnostic assets of a rehearsal)
    <bundle>/protocol.toml           the frozen experiment protocol

The *test* material is not in the bundle: it lives in a separately mounted evaluation root, and the bundle only
carries the hash commitment to it (inside the sealed prepared-data manifest and the stage-1 seal). Assembling
verifies every part; verifying again later detects any change. Readiness never rests on this file alone: the
doctor re-verifies and the plan binds the hashes.
"""

from pathlib import Path
from typing import Any

from devsearch.artifacts.canonical import pretty_json, seal_manifest, sha256_file
from devsearch.artifacts.files import write_atomic
from devsearch.artifacts.identity import code_identity
from devsearch.data.prepare import INITIAL_SEAL, PREPARED_MANIFEST, load_prepared
from devsearch.data.seals import load_seal
from devsearch.errors import ManifestError
from devsearch.evaluation.protocol import Protocol, load_protocol
from devsearch.modeling.assets import ASSETS_MANIFEST, verify_assets
from devsearch.execution.records import SCHEMA_VERSION, read_sealed

BUNDLE_MANIFEST = "bundle-manifest.json"
FIELDS = ("code", "prepared", "initial_seal", "protocol", "model_assets", "diagnostic", "lock")


def assemble_bundle(bundle: Path, *, lock_file: Path | None = None) -> dict[str, Any]:
    """Verify every part under `bundle` and write `bundle-manifest.json`. Fails (writes nothing) on any defect."""
    bundle = Path(bundle).resolve(strict=True)
    prepared = load_prepared(bundle / "prepared" / PREPARED_MANIFEST, bundle)
    seal_path = bundle / "prepared" / INITIAL_SEAL
    if not seal_path.is_file():
        raise ManifestError(f"{seal_path} is missing: the initial test seal must exist before the bundle is ready")
    seal = load_seal(seal_path, 1)
    loaded = prepared["loaded"]
    if (seal["corpus_id"], seal["index_id"]) != (loaded.corpus.corpus_id, loaded.index_id):
        raise ManifestError("the initial seal was written for a different corpus or index")
    if seal["protocol"] != prepared["manifest"]["protocol"]:
        raise ManifestError("the initial seal and the prepared data use different evaluation protocols")
    protocol = load_protocol(bundle / "protocol.toml")
    assets = verify_assets(bundle / "model-assets")
    manifest = seal_manifest({
        "schema_version": SCHEMA_VERSION, "kind": "bundle", "code": code_identity(),
        "prepared": {"manifest_path": f"prepared/{PREPARED_MANIFEST}", "manifest_sha256": prepared["manifest"]["manifest_sha256"],
                     "corpus_id": loaded.corpus.corpus_id, "index_id": loaded.index_id, "protocol": prepared["manifest"]["protocol"]},
        "initial_seal": {"path": f"prepared/{INITIAL_SEAL}", "manifest_sha256": seal["manifest_sha256"]},
        "protocol": {"path": "protocol.toml", "sha256": sha256_file(bundle / "protocol.toml"), "protocol_id": protocol.protocol_id},
        "model_assets": {"manifest_path": f"model-assets/{ASSETS_MANIFEST}", "manifest_sha256": assets["manifest_sha256"],
                         "asset_kind": assets["asset_kind"], "pinned": assets["pinned"]},
        "diagnostic": assets["asset_kind"] == "diagnostic",
        "lock": None if lock_file is None else {"sha256": sha256_file(Path(lock_file)), "name": Path(lock_file).name},
    })
    path = bundle / BUNDLE_MANIFEST
    if path.exists():
        existing = read_sealed(path, "bundle", FIELDS, str(path))
        if existing["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ManifestError(f"{path} already describes a different bundle; bundles are immutable")
        return existing
    write_atomic(path, pretty_json(manifest).encode("utf-8"), exclusive=True)
    return manifest


def verify_bundle(bundle: Path) -> dict[str, Any]:
    """Re-verify every part against the sealed manifest. Returns the manifest plus the loaded prepared data and protocol."""
    bundle = Path(bundle).resolve(strict=True)
    manifest = read_sealed(bundle / BUNDLE_MANIFEST, "bundle", FIELDS, str(bundle / BUNDLE_MANIFEST))
    prepared = load_prepared(bundle / manifest["prepared"]["manifest_path"], bundle)
    if prepared["manifest"]["manifest_sha256"] != manifest["prepared"]["manifest_sha256"]:
        raise ManifestError("the prepared-data manifest changed after the bundle was assembled")
    seal = load_seal(bundle / manifest["initial_seal"]["path"], 1)
    if seal["manifest_sha256"] != manifest["initial_seal"]["manifest_sha256"]:
        raise ManifestError("the initial seal changed after the bundle was assembled")
    protocol_path = bundle / manifest["protocol"]["path"]
    if sha256_file(protocol_path) != manifest["protocol"]["sha256"]:
        raise ManifestError("protocol.toml changed after the bundle was assembled; its hash is part of the plan")
    protocol: Protocol = load_protocol(protocol_path)
    assets = verify_assets(bundle / "model-assets")
    if assets["manifest_sha256"] != manifest["model_assets"]["manifest_sha256"]:
        raise ManifestError("the model assets changed after the bundle was assembled")
    return {"manifest": manifest, "prepared": prepared, "seal": seal, "protocol": protocol, "assets": assets, "directory": bundle}
