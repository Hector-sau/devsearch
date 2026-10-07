#!/usr/bin/env python3
"""CPU check of a registered model-assets directory before any GPU is rented (docs/GPU_QUICKSTART.md section 10).

    check_model_cpu.py --assets DIR [--require-pretrained] [--output report.json]

Verifies, on CPU and offline: every file hash against `assets-manifest.json` (and, for a pretrained set, that it is pinned to a 40-hex
commit); that the safetensors weights load into the declared BERT encoder with no missing or unexpected encoder keys; that the
tokenizer loads with the project's WordPiece rules and encodes a pair as `[CLS] q [SEP] p [SEP]` with the expected type ids; and that a
tiny training step works: finite logits and loss, finite gradients on every trainable parameter, a parameter update, identical
logits for two forward passes in eval mode, and no gradient reaching the encoder when it is frozen. It proves the files are usable,
not that the model ranks well. Exit 0 all checks passed, 1 a check failed, 2 usage.
"""

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PAIRS = [("How do I save a model checkpoint?", "Use torch.save(model.state_dict(), PATH) to save the learned parameters."),
         ("How do I save a model checkpoint?", "The DataLoader shuffles the dataset at every epoch when shuffle is true."),
         ("What does zero_grad do?", "Call optimizer.zero_grad() so gradients from the previous step do not accumulate."),
         ("What does zero_grad do?", "Tensors can be moved to a device with the to() method.")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--require-pretrained", action="store_true", help="fail unless the set is pretrained and pinned to a 40-hex commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    import torch

    from devsearch.modeling.assets import load_encoder, verify_assets
    from devsearch.modeling.batching import collate
    from devsearch.modeling.cross_encoder import CrossEncoder, parameter_counts
    from devsearch.modeling.policy import PairEncodingPolicy, encode_pair
    from devsearch.modeling.tokenization import WordPieceTokenizer

    checks, report = [], {"assets": str(args.assets)}

    def check(name, ok, detail):
        checks.append({"name": name, "passed": bool(ok), "detail": detail})
        print(f"  [{'pass' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    try:
        manifest = verify_assets(args.assets)
    except Exception as error:                                     # noqa: BLE001 (reported, not hidden)
        check("hashes", False, str(error))
        return finish(args, report, checks)
    check("hashes", True, f"{len(manifest['files'])} files match assets-manifest.json ({manifest['manifest_sha256'][:16]}...)")
    report["source"], report["asset_kind"], report["pinned"] = manifest["source"], manifest["asset_kind"], manifest["pinned"]
    if args.require_pretrained:
        check("pinned", manifest["asset_kind"] == "pretrained" and manifest["pinned"],
              f"asset_kind {manifest['asset_kind']}, pinned {manifest['pinned']}, revision {manifest['source'].get('revision')}")

    torch.manual_seed(0)
    try:
        encoder = load_encoder(args.assets)
    except Exception as error:                                     # noqa: BLE001
        check("load", False, str(error))
        return finish(args, report, checks)
    model = CrossEncoder(encoder, dropout=0.1, head_seed=1234)
    counts = parameter_counts(model)
    report["parameters"] = counts
    check("load", True, f"weights fit the declared encoder; {counts['total']:,} parameters")

    tokenizer = WordPieceTokenizer(args.assets)
    pair = tokenizer.encode_pair(*PAIRS[0])
    cls_sep = pair.ids[0] == tokenizer.encode_pair("a", "b").ids[0] and pair.ids.count(pair.ids[-1]) >= 2
    first_b = pair.type_ids.index(1) if 1 in pair.type_ids else -1
    check("tokenizer", cls_sep and first_b > 0 and pair.type_ids[-1] == 1 and set(pair.type_ids) == {0, 1},
          f"pair of {len(pair.ids)} tokens; segment B starts at {first_b}; fingerprint {tokenizer.fingerprint[:16]}...")
    unk = tokenizer.vocab().get("[UNK]")
    sample = [i for q, p in PAIRS for i in tokenizer.encode_pair(q, p).ids]
    unknown = sum(1 for i in sample if i == unk)
    detail = f"{unknown} of {len(sample)} sample tokens are [UNK] (a wrong vocabulary file shows up here)"
    if args.require_pretrained:                                   # a real English vocabulary must cover plain technical English
        check("vocabulary", unk is not None and unknown <= 0.1 * len(sample), detail + "; more than 10% fails")
    else:
        print(f"  [info] vocabulary: {detail}; binding only with --require-pretrained")
        report["vocabulary"] = detail

    policy = PairEncodingPolicy(256, 64)
    batch = collate([encode_pair(tokenizer, policy, q, p) for q, p in PAIRS], tokenizer.pad_id)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    model.train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=2e-5)
    logits = model(**batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    check("forward", torch.isfinite(logits).all().item() and math.isfinite(loss.item()), f"logits {[round(v, 4) for v in logits.tolist()]}, loss {loss.item():.4f}")
    loss.backward()
    grads = [p.grad for p in model.trainable_parameters()]
    finite = all(g is not None and torch.isfinite(g).all().item() for g in grads)
    check("backward", finite, f"{len(grads)} trainable tensors, every gradient present and finite: {finite}")
    before = model.head.weight.detach().clone()
    optimizer.step()
    check("update", not torch.equal(before, model.head.weight.detach()), "an optimizer step changed the parameters")

    model.eval()
    with torch.no_grad():
        first, second = model(**batch), model(**batch)
    check("eval determinism", torch.equal(first, second), "two forward passes in eval mode give identical logits (dropout off)")

    model.train()
    model.set_encoder_frozen(True)
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(model(**batch), labels).backward()
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    check("frozen encoder", not encoder_grads and model.head.weight.grad is not None, f"{len(encoder_grads)} encoder tensors received gradients when frozen")
    return finish(args, report, checks)


def finish(args, report, checks) -> int:
    report["checks"] = checks
    report["passed"] = all(c["passed"] for c in checks)
    report["note"] = "CPU usability check of the files; not a GPU check and not a ranking result"
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
