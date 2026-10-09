"""`python -m devsearch.prepare_data`: audited splits, materialized training data, review bank and seals.

Stages (default `groups`, which is the documented `--config/--corpus/--output` form):

  groups        corpus -> merged evidence groups -> seeded assignment -> audit; writes split-manifest.json
  materialize   drafts -> validated queries/qrels, index, train-only BCE pairs, fixed candidates, review pool
  bank          reviewer responses -> complete judgment bank -> stage-1 test seal (prejudged protocol)
  seal-initial  stage-1 test seal without a bank (post-rank protocol)

No stage calls a generator or a reviewer and none contacts a network: drafts and reviews are inputs with
their own recorded provenance. `--output` must be the stage's manifest file (the name is checked); every
artifact path must stay inside the runtime root. Failures write nothing visible and exit 1.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser
from devsearch.artifacts.paths import resolve_root
from devsearch.data.config import load_data_config
from devsearch.data.prepare import (
    EVALUATION_MANIFEST, INITIAL_SEAL, PREPARED_MANIFEST, SPLIT_MANIFEST, prepare_bank, prepare_groups,
    prepare_initial_seal_postrank, prepare_materialize,
)
from devsearch.errors import DevSearchError, ManifestError

STAGES = ("groups", "materialize", "bank", "seal-initial")
OUTPUT_NAMES = {"groups": SPLIT_MANIFEST, "materialize": PREPARED_MANIFEST, "bank": INITIAL_SEAL, "seal-initial": INITIAL_SEAL}


def build_parser():
    parser = new_parser("prepare_data", "Prepare audited splits, training data, the review bank and the test seals.")
    parser.add_argument("--config", type=Path, required=True, help="Data TOML (see configs/data.toml).")
    parser.add_argument("--stage", choices=STAGES, default="groups")
    parser.add_argument("--corpus", type=Path, help="groups: input corpus manifest.")
    parser.add_argument("--split-manifest", type=Path, help="materialize: the split manifest written by `groups`.")
    parser.add_argument("--evaluation-dir", type=Path, help="materialize: existing directory for the test-side inputs (separate from the prepared one).")
    parser.add_argument("--evaluation-root", type=Path, help="Root that must contain the evaluation directory (default: the runtime root).")
    parser.add_argument("--prepared-manifest", type=Path, help="bank / seal-initial: prepared-data-manifest.json.")
    parser.add_argument("--evaluation-manifest", type=Path, help="bank / seal-initial: evaluation-manifest.json.")
    parser.add_argument("--reviews", type=Path, help="bank: reviewer responses (JSONL).")
    parser.add_argument("--output", type=Path, required=True, help="The stage's manifest file; its name is fixed per stage.")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain every artifact path.")
    return parser


def _need(args, *names: str) -> None:
    missing = [f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None]
    if missing:
        raise ManifestError(f"stage {args.stage!r} also needs {', '.join(missing)}")


def _seal_beside_prepared(args) -> None:
    """The seal is written into the prepared directory, so `--output` must name exactly that file (the printed path is then the real one)."""
    if args.output.resolve() != args.prepared_manifest.resolve().parent / INITIAL_SEAL:
        raise ManifestError(f"--output must be {INITIAL_SEAL} inside the prepared directory of --prepared-manifest (the seal is written there)")


def _check_lengths(config, split_manifest: Path) -> bool:
    """With `[lengths]`: measure every passage and every question (all splits) with the pinned tokenizer before anything is written."""
    if config.lengths is None:
        return False
    from devsearch.data.prepare import load_split
    from devsearch.data.queries import load_queries
    from devsearch.modeling.policy import PairEncodingPolicy, check_corpus_budget
    from devsearch.modeling.tokenization import WordPieceTokenizer

    policy = PairEncodingPolicy(config.lengths.max_pair_tokens, config.lengths.max_query_tokens)
    tokenizer = WordPieceTokenizer(config.lengths.tokenizer_assets)
    passages = check_corpus_budget(load_split(split_manifest)["corpus"].chunks, tokenizer, policy)
    problems = [f"passage {item['chunk_id']} has {item['tokens']} tokens (budget {passages['passage_budget']})" for item in passages["excluded"][:10]]
    problems += [f"question {q.query_id} ({q.split}) has {tokenizer.count(q.text)} tokens (reserve {policy.max_query_tokens})"
                 for _, path in config.queries for q in load_queries(path) if tokenizer.count(q.text) > policy.max_query_tokens][:10]
    if problems:
        raise ManifestError("the data does not fit the pinned tokenizer's budget; nothing was written:\n  " + "\n  ".join(problems[:10]))
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checked = False
    try:
        if args.output.name != OUTPUT_NAMES[args.stage]:
            raise ManifestError(f"stage {args.stage!r} writes {OUTPUT_NAMES[args.stage]}; --output must have that name")
        config = load_data_config(args.config)
        root = resolve_root(args.runtime_root, args.output.resolve())
        if args.stage == "groups":
            _need(args, "corpus")
            manifest = prepare_groups(config, args.corpus, args.output.resolve().parent, root)
        elif args.stage == "materialize":
            _need(args, "split_manifest", "evaluation_dir")
            if args.split_manifest.resolve() != (args.output.resolve().parent / SPLIT_MANIFEST):
                raise ManifestError("--split-manifest must sit beside --output (the prepared directory)")
            checked = _check_lengths(config, args.split_manifest)
            manifest = prepare_materialize(config, args.split_manifest, args.evaluation_dir, root, args.evaluation_root)
        elif args.stage == "bank":
            _need(args, "prepared_manifest", "evaluation_manifest", "reviews")
            _seal_beside_prepared(args)
            manifest = prepare_bank(config, args.prepared_manifest, args.evaluation_manifest, args.reviews, root, args.evaluation_root)
        else:
            _need(args, "prepared_manifest", "evaluation_manifest")
            _seal_beside_prepared(args)
            manifest = prepare_initial_seal_postrank(config, args.prepared_manifest, args.evaluation_manifest, root, args.evaluation_root)
    except (DevSearchError, OSError) as error:
        return fail("prepare_data", error)
    print(json.dumps({"status": "ok", "stage": args.stage, "kind": manifest["kind"], "manifest_sha256": manifest["manifest_sha256"],
                      "manifest": str(args.output.resolve()), "evaluation_manifest_name": EVALUATION_MANIFEST,
                      "lengths_checked": checked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
