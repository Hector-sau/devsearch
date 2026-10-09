"""`python -m devsearch.prepare`: pinned sources in, a validated corpus artifact and a preparation report out (T03, docs/02 §3).

    prepare --config sources.toml --output <existing empty corpus dir> [--report FILE] [--offline] [--runtime-root R]

Each `[[source]]` is a snapshot of one exact revision. Without `--offline` a *missing* snapshot is fetched with git (exact
commit, no hooks, no LFS, nothing executed; needs network); an existing snapshot is verified, never changed, and `--offline`
forbids any acquisition. Files are read as data (Markdown/MDX, reStructuredText, notebook JSON, tutorial Python), cut into
passages that fit the pinned tokenizer's budget (never truncated), and published as an immutable corpus. The report lists
what was quarantined, dropped or excluded and why. The tokenizer needs the `ml` extra's `tokenizers` library but no model
weights. Exit: 0 written (or identical to what exists), 1 invalid input or failure, 2 usage.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser


def build_parser():
    parser = new_parser("prepare", "Acquire/read pinned sources and build a token-validated corpus artifact.")
    parser.add_argument("--config", type=Path, required=True, help="Sources TOML (see configs/sources.toml).")
    parser.add_argument("--output", type=Path, required=True, help="An existing, empty directory that receives the corpus artifact.")
    parser.add_argument("--report", type=Path, help="Preparation report JSON (default: <output>.prepare-report.json beside the corpus directory).")
    parser.add_argument("--offline", action="store_true", help="Never acquire: every snapshot must already exist.")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain the output, report and every snapshot.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import DevSearchError
    try:
        summary = _run(args)
    except (DevSearchError, OSError) as error:
        return fail("prepare", error)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run(args) -> dict:
    from devsearch.artifacts.canonical import pretty_json, sha256_file
    from devsearch.artifacts.files import write_atomic
    from devsearch.artifacts.paths import ensure_inside, resolve_root
    from devsearch.errors import ManifestError
    from devsearch.ingestion.config import load_prepare_config
    from devsearch.ingestion.pipeline import prepare_corpus
    from devsearch.modeling.assets import ASSETS_MANIFEST, read_json
    from devsearch.modeling.policy import PairEncodingPolicy
    from devsearch.modeling.tokenization import WordPieceTokenizer

    config = load_prepare_config(args.config)
    output = args.output.resolve()
    root = resolve_root(args.runtime_root, output)
    ensure_inside(output, root, what="corpus output")
    if not output.is_dir():
        raise ManifestError(f"{output} does not exist; the output directory is never created implicitly")
    report_path = ensure_inside(args.report if args.report is not None else output.with_name(output.name + ".prepare-report.json"), root, what="preparation report")
    if not report_path.parent.is_dir():
        raise ManifestError(f"{report_path.parent} does not exist")
    for spec in config.sources:
        ensure_inside(spec.snapshot, root, what=f"snapshot of {spec.source_id}")
    ensure_inside(config.tokenizer_assets, root, what="tokenizer assets")
    tokenizer = WordPieceTokenizer(config.tokenizer_assets)
    budget = PairEncodingPolicy(config.max_pair_tokens, config.max_query_tokens).passage_budget(tokenizer)
    info = {"fingerprint": tokenizer.fingerprint, "special_tokens_per_pair": tokenizer.pair_special_tokens, "asset_kind": None, "pinned": None, "source": None}
    manifest_path = config.tokenizer_assets / ASSETS_MANIFEST
    if manifest_path.is_file():
        assets = read_json(manifest_path, ASSETS_MANIFEST)
        info.update(asset_kind=assets.get("asset_kind"), pinned=assets.get("pinned"), source=assets.get("source"))
    prepared = prepare_corpus(config, tokenizer, budget, output, allow_acquire=not args.offline, tokenizer_info=info, config_sha256=sha256_file(args.config))
    body = pretty_json(prepared.report).encode("utf-8")
    if report_path.exists():
        if report_path.read_bytes() != body:
            raise ManifestError(f"{report_path} already exists with different content; reports are never overwritten")
    else:
        write_atomic(report_path, body, exclusive=True)
    totals = prepared.report["totals"]
    return {"status": "ok", "corpus_id": prepared.manifest["corpus_id"], "documents": totals["documents"], "passages": totals["passages"],
            "passages_excluded": totals["passages_excluded"], "files": totals["files"], "tokenizer_asset_kind": info["asset_kind"],
            "report": str(report_path), "report_sha256": prepared.report["manifest_sha256"]}


if __name__ == "__main__":
    raise SystemExit(main())
