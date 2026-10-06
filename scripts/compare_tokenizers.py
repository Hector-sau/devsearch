"""Compare a tokenizer directory with references: its complete effective configuration AND the actual encodings of real texts.

    compare_tokenizers.py --candidate DIR [--reference DIR] --texts FILE [--texts FILE ...] [--max-texts N] [--output report.json]

`--texts` files are JSONL with a `text` field (corpus `chunks.jsonl`, query drafts) or plain text, one item per line. Every text is
encoded alone (no special tokens) and paired (query-like texts with passage-like texts, in file order, as `[CLS] a [SEP] b [SEP]`).
The candidate is encoded by the project's own `WordPieceTokenizer` and compared with:

- `project-reference`: the project tokenizer built from `--reference DIR` (for example the stand-in vocabulary against the pinned
  MiniLM files) - this is what decides whether artifacts built with one directory stay valid for the other;
- `bert-legacy`: Transformers' pure-Python BERT tokenizer (independent code) configured from the candidate's own `tokenizer_config.json`,
  for a `vocab.txt` directory;
- `transformers-auto`: `AutoTokenizer.from_pretrained(candidate, local_files_only=True)`, which applies the directory's configuration
  the way the Hub model card intends; it is reported as unavailable when the directory is not a Hub-style tokenizer folder.

The report lists both effective configurations and every difference, then the number of texts and pairs whose ids or type ids differ, with
the first examples as tokens. Exit 0 only when every available comparison is identical, at least one independent reference
(`bert-legacy` or `transformers-auto`) ran, and the configurations do not differ. Nothing is downloaded.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from devsearch.modeling.tokenization import WordPieceTokenizer  # noqa: E402

CONFIG_FILES = ("tokenizer.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "config.json")


def read_texts(paths: list[Path], limit: int | None) -> list[str]:
    texts = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            texts.append(json.loads(line)["text"] if line.lstrip().startswith("{") else line)
    return texts[:limit] if limit else texts


def describe(directory: Path) -> dict:
    """Files and the effective pipeline the project tokenizer builds from them (what actually encodes)."""
    tokenizer = WordPieceTokenizer(directory)
    pipeline = json.loads(tokenizer._tokenizer.to_str())
    model = pipeline.get("model") or {}
    vocab = sorted(tokenizer._tokenizer.get_vocab().items(), key=lambda item: item[1])
    declared = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        path = directory / name
        if path.is_file():
            declared[name] = json.loads(path.read_text(encoding="utf-8"))
    return {
        "directory": str(directory), "fingerprint": tokenizer.fingerprint,
        "files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in CONFIG_FILES if (directory / name).is_file()},
        "declared": declared,
        "effective": {"normalizer": pipeline.get("normalizer"), "pre_tokenizer": pipeline.get("pre_tokenizer"), "post_processor": pipeline.get("post_processor"),
                      "model": {key: model.get(key) for key in ("type", "unk_token", "continuing_subword_prefix", "max_input_chars_per_word")},
                      "added_tokens": pipeline.get("added_tokens"), "vocab_size": tokenizer.vocab_size,
                      "vocab_sha256": hashlib.sha256("\n".join(token for token, _ in vocab).encode("utf-8")).hexdigest(),
                      "pair_special_tokens": tokenizer.pair_special_tokens, "pad_id": tokenizer.pad_id},
    }


def differences(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            out += differences(a.get(key), b.get(key), f"{path}.{key}" if path else key)
        return out
    return [] if a == b else [f"{path}: {json.dumps(a)[:120]} != {json.dumps(b)[:120]}"]


class Project:
    def __init__(self, directory: Path):
        self.tokenizer = WordPieceTokenizer(directory)

    def single(self, text: str) -> list[int]:
        return list(self.tokenizer._tokenizer.encode(text, add_special_tokens=False).ids)

    def pair(self, a: str, b: str) -> tuple[list[int], list[int]]:
        encoded = self.tokenizer.encode_pair(a, b)
        return list(encoded.ids), list(encoded.type_ids)

    def tokens(self, ids: list[int]) -> list[str]:
        return [self.tokenizer._tokenizer.id_to_token(i) for i in ids]


class Legacy:
    def __init__(self, directory: Path):
        from transformers.models.bert.tokenization_bert_legacy import BertTokenizerLegacy
        config_path = directory / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        self.tokenizer = BertTokenizerLegacy(vocab_file=str(directory / "vocab.txt"), do_lower_case=config.get("do_lower_case", True),
                                             strip_accents=config.get("strip_accents"), tokenize_chinese_chars=config.get("tokenize_chinese_chars", True))

    def single(self, text: str) -> list[int]:
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))

    def pair(self, a: str, b: str) -> tuple[list[int], list[int]]:
        first, second = self.single(a), self.single(b)
        return self.tokenizer.build_inputs_with_special_tokens(first, second), self.tokenizer.create_token_type_ids_from_sequences(first, second)


class Auto:
    def __init__(self, directory: Path):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)

    def single(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def pair(self, a: str, b: str) -> tuple[list[int], list[int]]:
        encoded = self.tokenizer(a, b, return_token_type_ids=True)
        return list(encoded["input_ids"]), list(encoded["token_type_ids"])


def compare(candidate: Project, other, texts: list[str], pairs: list[tuple[str, str]], examples: int = 5) -> dict:
    single_bad, pair_bad, shown = 0, 0, []
    for text in texts:
        mine, theirs = candidate.single(text), other.single(text)
        if mine != theirs:
            single_bad += 1
            if len(shown) < examples:
                shown.append({"text": text[:160], "candidate": candidate.tokens(mine)[:40], "reference_ids": theirs[:40]})
    for a, b in pairs:
        if candidate.pair(a, b) != tuple(other.pair(a, b)):
            pair_bad += 1
    return {"texts": len(texts), "texts_differing": single_bad, "pairs": len(pairs), "pairs_differing": pair_bad, "examples": shown,
            "identical": single_bad == 0 and pair_bad == 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--texts", type=Path, action="append", required=True)
    parser.add_argument("--pair-with", type=Path, action="append", help="Passage-like texts to pair with --texts (default: --texts paired with itself, shifted by one).")
    parser.add_argument("--max-texts", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    texts = read_texts(args.texts, args.max_texts)
    passages = read_texts(args.pair_with, args.max_texts) if args.pair_with else texts[1:] + texts[:1]
    pairs = [(texts[i], passages[i % len(passages)]) for i in range(len(texts))] if passages else []
    candidate = Project(args.candidate)
    report = {"candidate": describe(args.candidate), "comparisons": {}, "unavailable": {}}
    if args.reference:
        report["reference"] = describe(args.reference)
        report["config_differences"] = differences(report["reference"]["effective"], report["candidate"]["effective"])
        report["comparisons"]["project-reference"] = compare(candidate, Project(args.reference), texts, pairs)
    for name, factory, applies in (("bert-legacy", Legacy, (args.candidate / "vocab.txt").is_file()), ("transformers-auto", Auto, True)):
        if not applies:
            report["unavailable"][name] = "not a vocab.txt directory"
            continue
        try:
            reference = factory(args.candidate)
        except Exception as error:                              # an unavailable reference is reported, never counted as agreement
            report["unavailable"][name] = f"{type(error).__name__}: {str(error)[:200]}"
            continue
        report["comparisons"][name] = compare(candidate, reference, texts, pairs)
    independent = [name for name in ("bert-legacy", "transformers-auto") if name in report["comparisons"]]
    report["verdict"] = {"identical": bool(report["comparisons"]) and all(c["identical"] for c in report["comparisons"].values())
                         and not report.get("config_differences"), "independent_references": independent}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "config_differences": report.get("config_differences"), "unavailable": report["unavailable"],
                      "comparisons": {k: {key: v[key] for key in ("texts", "texts_differing", "pairs", "pairs_differing", "identical")} for k, v in report["comparisons"].items()}},
                     indent=2, ensure_ascii=False))
    return 0 if report["verdict"]["identical"] and independent else 1


if __name__ == "__main__":
    raise SystemExit(main())
