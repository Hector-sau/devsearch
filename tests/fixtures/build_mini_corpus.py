"""Deterministic builder for `tests/fixtures/mini_corpus/`.

The text is original fixture prose written for this project, not copied or
paraphrased documentation. Source/revision labels (`fixture-*`) and URLs under the
reserved `.invalid` domain are placeholders, never real sources or links. These
files test code paths only; they are not a domain corpus and give no quality metric.

    python tests/fixtures/build_mini_corpus.py DIRECTORY   # DIRECTORY must exist
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from devsearch.data.corpus import ChunkSpan, DocInput, write_corpus
from devsearch.data.sourcemap import SourceSegment

CHUNKER_VERSION = "fixture-chunker-1"
LICENSE = "project-original-fixture"
TIE_NOTE = (
    "Fixture note: the token tiebreakmarker appears in exactly two chunks with identical "
    "text, so a model-free score cannot separate them and the chunk id decides."
)


@dataclass(frozen=True)
class Block:
    text: str
    lines: tuple[int, int]
    headings: tuple[str, ...]
    group: str
    code_fragment: bool = False
    joiner: str = "\n\n"


def make_doc(doc_id, source_id, title, revision, path, url, blocks, synthetic_title=None) -> DocInput:
    text = ""
    segments: list[SourceSegment] = []
    spans: list[ChunkSpan] = []
    if synthetic_title:
        text = f"# {synthetic_title}"
        segments.append(SourceSegment(0, len(text), path, True, start_line=1, end_line=1))
    for block in blocks:
        if text:
            text += block.joiner
        start = len(text)
        text += block.text
        segments.append(SourceSegment(start, len(text), path, False, start_line=block.lines[0], end_line=block.lines[1]))
        spans.append(ChunkSpan(start, len(text), block.group, block.headings, None, block.code_fragment))
    return DocInput(
        doc_id=doc_id, source_id=source_id, title=title, revision=revision, source_url=url,
        license=LICENSE, parser_version="fixture-parser-1", text=text + "\n",
        segments=tuple(segments), chunks=tuple(spans),
    )


def documents() -> list[DocInput]:
    torch_url = "https://docs.fixture.invalid/torch/"
    hf_url = "https://docs.fixture.invalid/transformers/"
    checkpoints = make_doc(
        "fixture-torch:docs/checkpoints.md", "fixture-torch", "Saving and resuming checkpoints",
        "fixture-r1", "docs/checkpoints.md", torch_url + "checkpoints", [
            Block(
                "Call torch.save on the dictionary returned by model.state_dict() to write the "
                "learned parameters to disk. A checkpoint that will be used to resume training "
                "should also store the optimizer state and the number of finished epochs, not "
                "only the model weights.",
                (3, 6), ("Checkpoints", "Saving"), "eg-ckpt-save"),
            Block(
                "To restore a model, build the same architecture first, then call "
                "model.load_state_dict(torch.load(path)). The call copies tensors into the "
                "existing parameters and raises an error when a key is missing or unexpected, "
                "unless strict loading is switched off.",
                (8, 11), ("Checkpoints", "Loading"), "eg-ckpt-load"),
            Block(
                "Resuming training means loading three things: the model parameters, the "
                "optimizer state and the scheduler state, then continuing from the epoch after "
                "the last finished one. Seeding the data loader again gives a repeatable order "
                "of batches for the epoch that is replayed.",
                (13, 16), ("Checkpoints", "Resuming"), "eg-ckpt-resume"),
            Block(
                '```python\ncheckpoint = {\n    "epoch": epoch,\n    "model": model.state_dict(),\n'
                '    "optimizer": optimizer.state_dict(),\n}\ntorch.save(checkpoint, "last.pt")\n```',
                (18, 26), ("Checkpoints", "Code example"), "eg-ckpt-code"),
        ], synthetic_title="Checkpoints (generated title)")
    train_eval = make_doc(
        "fixture-torch:docs/train-eval.md", "fixture-torch", "Train and eval modes",
        "fixture-r1", "docs/train-eval.md", torch_url + "train-eval", [
            Block(
                "A model starts in training mode. Call model.eval() before validation so that "
                "dropout layers stop dropping units and batch normalization uses its stored "
                "running statistics. Call model.train() again before the next training epoch.",
                (3, 6), ("Modes", "Switching"), "eg-mode-switch"),
            Block(
                "Wrap validation in torch.no_grad() to avoid building the autograd graph. "
                "Evaluation mode does not disable gradients by itself; the two switches are "
                "independent. A parameter with requires_grad set to False is frozen and is "
                "skipped by the optimizer.",
                (8, 11), ("Modes", "Gradients"), "eg-mode-grad"),
            Block(
                "Gradient accumulation runs several small batches before one optimizer step. "
                "Divide each loss by the number of accumulation steps, call backward on every "
                "batch, and call optimizer.zero_grad() only after the step. This trades model "
                "speed for memory.",
                (13, 17), ("Modes", "Accumulation"), "eg-mode-accum"),
            Block(TIE_NOTE + " The model is irrelevant here.", (19, 20), ("Notes",), "eg-note-torch"),
        ])
    long_loop = make_doc(
        "fixture-torch:docs/long-loop.md", "fixture-torch", "A long training loop",
        "fixture-r1", "docs/long-loop.md", torch_url + "long-loop", [
            Block(
                "The loop below is longer than one passage, so it is split at line boundaries "
                "into fragments that keep their indentation. Each fragment links back to the "
                "complete example.",
                (1, 3), ("Training loop",), "eg-loop-intro"),
            Block(
                "for epoch in range(start_epoch, num_epochs):\n    model.train()\n"
                "    for batch in train_loader:\n        optimizer.zero_grad()\n"
                '        logits = model(batch["inputs"])\n'
                '        loss = loss_fn(logits, batch["labels"])\n'
                "        loss.backward()\n        optimizer.step()",
                (6, 13), ("Training loop", "Code"), "eg-loop-code-1", True),
            Block(
                "    scheduler.step()\n    model.eval()\n    with torch.no_grad():\n"
                "        for batch in val_loader:\n"
                '            logits = model(batch["inputs"])\n'
                '            val_loss += loss_fn(logits, batch["labels"]).item()',
                (14, 19), ("Training loop", "Code"), "eg-loop-code-2", True, "\n"),
            Block(
                '    torch.save({"epoch": epoch, "model": model.state_dict(),\n'
                '                "optimizer": optimizer.state_dict()}, "last.pt")',
                (20, 21), ("Training loop", "Code"), "eg-loop-code-3", True, "\n"),
        ])
    padding = Block(
        "Padding makes sequences in a batch the same length, and truncation cuts sequences "
        "that are longer than max_length. Pass padding=True and truncation=True to the "
        "tokenizer call. A model reads the attention mask to ignore padded positions.",
        (8, 11), ("Tokenizers", "Padding"), "eg-tok-padding")
    tokenizers_v1 = make_doc(
        "fixture-transformers:docs/tokenizers.md", "fixture-transformers", "Tokenizers",
        "fixture-v1", "docs/tokenizers.md", hf_url + "tokenizers", [
            Block(
                "Load a tokenizer with AutoTokenizer.from_pretrained(name). The tokenizer turns "
                "text into input ids and an attention mask; special tokens such as [CLS] and "
                "[SEP] are added around a pair of sentences. A question and a passage can be "
                "encoded together as one pair.",
                (3, 6), ("Tokenizers", "Loading"), "eg-tok-load"),
            padding,
            Block(
                "Words that are not in the vocabulary map to the unknown token <unk>, or are "
                "split into smaller word pieces. Inspect the result with "
                "tokenizer.convert_ids_to_tokens(ids).",
                (13, 15), ("Tokenizers", "Unknown"), "eg-tok-unknown"),
        ])
    tokenizers_v2 = make_doc(
        "fixture-transformers:docs/tokenizers.md", "fixture-transformers", "Tokenizers",
        "fixture-v2", "docs/tokenizers.md", hf_url + "tokenizers", [
            Block(
                "Load a tokenizer with AutoTokenizer.from_pretrained(name). Version two of this "
                "fixture also documents return_tensors, which asks the tokenizer to return "
                "PyTorch tensors instead of Python lists.",
                (3, 6), ("Tokenizers", "Loading"), "eg-tok-load-v2"),
            padding,
        ])
    models = make_doc(
        "fixture-transformers:docs/models.md", "fixture-transformers", "Models",
        "fixture-v1", "docs/models.md", hf_url + "models", [
            Block(
                "AutoModel.from_pretrained(name) downloads the configuration and the weights of "
                "a pretrained model and returns a PyTorch module. Call "
                "model.save_pretrained(directory) to store both, then load the saved directory "
                "later without a network connection.",
                (3, 7), ("Models", "Loading"), "eg-model-load"),
            Block(
                "AutoModelForSequenceClassification adds a classification head on top of the "
                "encoder, which is randomly initialized when it is not part of the checkpoint. "
                "The head must be trained before its scores are useful.",
                (9, 12), ("Models", "Heads"), "eg-model-head"),
            Block(TIE_NOTE + " The model is irrelevant here.", (14, 15), ("Notes",), "eg-note-hf"),
        ])
    return [checkpoints, train_eval, long_loop, tokenizers_v1, tokenizers_v2, models]


def build(directory: Path) -> None:
    write_corpus(directory, documents(), chunker_version=CHUNKER_VERSION)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    build(Path(sys.argv[1]))
