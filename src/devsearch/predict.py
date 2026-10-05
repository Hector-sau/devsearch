"""`python -m devsearch.predict`: score question/passage pairs with a checkpoint or an exported model.

    predict --config predict.toml --checkpoint <inference bundle dir | checkpoint .json> --pairs pairs.jsonl --output scores.jsonl

Input lines are `{"pair_id", "query", "passage"}` (unique pair IDs); output lines are `{"pair_id", "logit", "model_id"}` in
input order. A logit is an uncalibrated relevance score, not a probability and not comparable to a BM25 score. Scoring is
all-or-nothing: a pair that does not fit the length policy (nothing is truncated), a non-finite logit or any failure writes
no output. The output file is new (never overwritten) and must lie inside the runtime root. Exit: 0 ok, 1 failure, 2 usage.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from devsearch._cli import fail, new_parser


def build_parser():
    parser = new_parser("predict", "Score supplied question/passage pairs.")
    parser.add_argument("--config", type=Path, required=True, help="Model-job TOML (see configs/predict.toml).")
    parser.add_argument("--checkpoint", type=Path, required=True, help="An exported inference bundle directory, or a training checkpoint manifest (.json).")
    parser.add_argument("--pairs", type=Path, required=True, help="JSONL of {pair_id, query, passage}.")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL file of {pair_id, logit, model_id}.")
    parser.add_argument("--runtime-root", type=Path, help="Directory that must contain the output (default: DEVSEARCH_RUNTIME_ROOT, then its parent).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from devsearch.errors import DevSearchError
    try:
        count, model_id = _run(args)
    except (DevSearchError, OSError) as error:
        return fail("predict", error)
    print(json.dumps({"status": "ok", "pairs": count, "model_id": model_id, "output": str(args.output.resolve())}))
    return 0


def _run(args) -> tuple[int, str]:
    from devsearch.artifacts.files import jsonl_bytes, read_jsonl, write_atomic
    from devsearch.artifacts.paths import ensure_inside, resolve_root
    from devsearch.artifacts.schema import as_id, as_str, exact_keys
    from devsearch.errors import ManifestError
    from devsearch.modeling.devices import load_profile, resolve_device
    from devsearch.modeling.jobs import load_model_job
    from devsearch.modeling.scoring import score_pairs
    from devsearch.training.resolve import resolve_model

    job = load_model_job(args.config)
    root = resolve_root(args.runtime_root, args.output.resolve().parent)
    output = ensure_inside(args.output, root, what="prediction output")
    if output.exists():
        raise ManifestError(f"{output} already exists; predictions are never overwritten")
    if not output.parent.is_dir():
        raise ManifestError(f"output directory does not exist: {output.parent}")
    pairs, seen = [], set()
    for label, record in read_jsonl(args.pairs, args.pairs.name):
        exact_keys(record, ("pair_id", "query", "passage"), label)
        pair_id = as_id(record["pair_id"], f"{label}.pair_id")
        if pair_id in seen:
            raise ManifestError(f"{label}: duplicate pair_id {pair_id!r}")
        seen.add(pair_id)
        pairs.append((pair_id, as_str(record["query"], f"{label}.query", max_length=10_000), as_str(record["passage"], f"{label}.passage", max_length=1_000_000)))
    if not pairs:
        raise ManifestError("no pairs to score")
    device = resolve_device(load_profile(job.profile))
    resolved = resolve_model(args.checkpoint, device=device, assets=job.assets, policy=job.policy, dropout=job.dropout)
    logits = score_pairs(resolved.model, resolved.tokenizer, resolved.policy, [(q, p) for _, q, p in pairs], device=device, batch_size=job.batch_size)
    write_atomic(output, jsonl_bytes({"pair_id": pid, "logit": value, "model_id": resolved.model_id} for (pid, _q, _p), value in zip(pairs, logits)), exclusive=True)
    return len(pairs), resolved.model_id


if __name__ == "__main__":
    raise SystemExit(main())
