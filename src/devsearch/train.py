"""`python -m devsearch.train`: run (or resume) one native-PyTorch training recipe on prepared data.

    train --config train.toml --run-dir <existing dir> [--resume latest|<checkpoint id>] [--runtime-root R]

The same loop, checkpoints and refusals as a campaign: epoch-boundary checkpoints (epoch zero plus at most three completed
epochs for a formal recipe), `latest`/`best` as pointers, resume only from the last complete epoch and only with unchanged
data, tokenizer, model assets and hyperparameters (a changed recipe is a different run, not a continuation). The run
directory must already exist; nothing is created implicitly. Only training-visible data is read. Exit: 0 completed,
1 invalid input or a validation/runtime failure (reported without a stack trace), 2 usage.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import new_parser


def build_parser():
    parser = new_parser("train", "Train one cross-encoder recipe (frozen head or full fine-tuning) with checkpoints.")
    parser.add_argument("--config", type=Path, required=True, help="Training TOML (see configs/train.toml).")
    parser.add_argument("--run-dir", type=Path, required=True, help="An existing directory that receives checkpoints and logs.")
    parser.add_argument("--resume", metavar="CHECKPOINT", help="'latest' or the id of the latest complete-epoch checkpoint; nothing else can be resumed.")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain the inputs and the run directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch._cli import fail
    from devsearch.errors import DevSearchError
    try:
        summary = _run(args)
    except (DevSearchError, OSError) as error:
        return fail("train", error)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "completed" else 1


def _run(args) -> dict:
    from devsearch.artifacts.canonical import canonical_json
    from devsearch.artifacts.identity import code_identity
    from devsearch.artifacts.paths import ensure_inside, resolve_root
    from devsearch.errors import ManifestError
    from devsearch.modeling.assets import load_encoder, verify_assets
    from devsearch.modeling.cross_encoder import CrossEncoder
    from devsearch.modeling.devices import load_profile, resolve_device
    from devsearch.modeling.policy import PairEncodingPolicy
    from devsearch.modeling.tokenization import WordPieceTokenizer
    from devsearch.training.checkpoint import read_pointer
    from devsearch.training.inputs import load_training_inputs
    from devsearch.training.job import load_train_job
    from devsearch.training.loop import train

    job = load_train_job(args.config)
    run_dir = args.run_dir.resolve()
    root = resolve_root(args.runtime_root, run_dir)
    ensure_inside(run_dir, root, what="run directory")
    if not run_dir.is_dir():
        raise ManifestError(f"run directory does not exist: {run_dir} (it is never created implicitly)")
    ensure_inside(job.prepared_manifest, root, what="prepared-data manifest")
    ensure_inside(job.assets, root, what="model assets")
    if args.resume is not None:
        latest = read_pointer(run_dir, "latest")
        if latest is None:
            raise ManifestError("nothing to resume: this run directory has no checkpoint")
        if args.resume not in ("latest", latest["checkpoint_id"]):
            raise ManifestError(f"only the latest complete-epoch checkpoint ({latest['checkpoint_id']}) can be resumed; start a new run to fork from {args.resume}")
    elif read_pointer(run_dir, "latest") is not None:
        raise ManifestError("this run directory already has checkpoints; resume it with --resume or choose a new directory")
    inputs = (load_training_inputs(job.prepared_manifest, root, grouped=True, pairs_manifest=job.pairs_manifest,
                                   pairs_manifest_sha256=job.pairs_manifest_sha256) if job.config.grouped
              else load_training_inputs(job.prepared_manifest, root))
    assets_manifest = verify_assets(job.assets)
    tokenizer = WordPieceTokenizer(job.assets)
    policy = PairEncodingPolicy(job.max_pair_tokens, job.max_query_tokens)
    device = resolve_device(load_profile(job.profile))
    config = job.config
    model = CrossEncoder(load_encoder(job.assets), dropout=config.dropout, head_seed=config.head_seed)
    log_path = run_dir / "events.jsonl"

    def on_event(event) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(event) + "\n")

    identities = {"pairs_manifest": inputs.pairs_identity, "tokenizer": tokenizer.fingerprint, "model": assets_manifest["manifest_sha256"],
                  "code": code_identity()["source_snapshot_sha256"]}
    result = train(config=config, model=model, tokenizer=tokenizer, policy=policy, pairs=inputs.train_pairs, validation=inputs.validation, device=device,
                   run_dir=run_dir, identities=identities, resume=args.resume is not None, on_event=on_event,
                   **({"grouped_data": inputs.grouped_data} if config.grouped else {}))
    return {"status": result.status, "stop_reason": result.stop_reason, "completed_epochs": result.completed_epochs, "global_step": result.global_step,
            "best_epoch": result.best_epoch, "best_checkpoint_id": result.best_checkpoint_id, "plan": result.plan, "device": str(device),
            "recipe_id": config.recipe_id, "config_sha256": config.config_sha256, "diagnostic": config.diagnostic,
            "epochs": [{"epoch": h["epoch"], "train_loss": h["train_loss"], "validation_group_mrr": h["validation"]["group_macro"]["mrr@10"],
                        "validation_group_ndcg": h["validation"]["group_macro"]["ndcg@10"]} for h in result.history]}


if __name__ == "__main__":
    raise SystemExit(main())
