"""WordPiece tokenizer built on the `tokenizers` library (the fast core Transformers wraps).

Wrapping the core library directly keeps pair encoding independent of Transformers' tokenizer
classes, whose constructors changed between major versions (one silently ignored a vocabulary
file in testing). A tokenizer directory holds either `tokenizer.json` or `vocab.txt` (+ optional
`tokenizer_config.json` for `do_lower_case`); its identity is the SHA-256 of those files.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

from devsearch.artifacts.canonical import digest_of, sha256_file
from devsearch.errors import ManifestError
from devsearch.modeling.policy import EncodedPair

SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
_FILES = ("tokenizer.json", "vocab.txt", "tokenizer_config.json")


def build_wordpiece(vocab: list[str], *, lowercase: bool = True) -> Tokenizer:
    """A BERT-style WordPiece tokenizer from an ordered vocabulary that contains the specials."""
    mapping = {token: index for index, token in enumerate(vocab)}
    if len(mapping) != len(vocab):
        raise ManifestError("vocabulary contains duplicate tokens")
    for special in SPECIALS[1:4]:
        if special not in mapping:
            raise ManifestError(f"vocabulary lacks {special}")
    tokenizer = Tokenizer(models.WordPiece(mapping, unk_token="[UNK]", max_input_chars_per_word=100))
    tokenizer.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=lowercase)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", mapping["[CLS]"]), ("[SEP]", mapping["[SEP]"])],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def write_diagnostic_tokenizer(directory: Path, words: list[str]) -> Path:
    """A tiny local tokenizer for tests and rehearsals: the given words plus `##s`. Never a real vocabulary."""
    vocab = list(SPECIALS) + sorted({w.lower() for w in words if w.isalnum()}) + ["##s"]
    build_wordpiece(vocab).save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": True, "kind": "diagnostic"}, sort_keys=True))
    return directory


class WordPieceTokenizer:
    """The pinned pair tokenizer. `pair_special_tokens` is measured, not assumed."""

    def __init__(self, directory: Path) -> None:
        directory = Path(directory)
        present = {name: directory / name for name in _FILES if (directory / name).is_file()}
        if "tokenizer.json" in present:
            self._tokenizer = Tokenizer.from_file(str(present["tokenizer.json"]))
        elif "vocab.txt" in present:
            lowercase = True
            if "tokenizer_config.json" in present:
                lowercase = bool(json.loads(present["tokenizer_config.json"].read_text()).get("do_lower_case", True))
            vocab = present["vocab.txt"].read_text(encoding="utf-8").splitlines()
            self._tokenizer = build_wordpiece(vocab, lowercase=lowercase)
        else:
            raise ManifestError(f"tokenizer directory {directory} has neither tokenizer.json nor vocab.txt")
        self.directory = directory
        self.fingerprint = digest_of({name: sha256_file(path) for name, path in sorted(present.items())})
        vocab_map = self._tokenizer.get_vocab()
        self.pad_id = vocab_map.get("[PAD]", 0)
        self.vocab_size = self._tokenizer.get_vocab_size()
        probe_q, probe_p = "alpha", "beta"
        pair = self._tokenizer.encode(probe_q, probe_p)
        raw = len(self._tokenizer.encode(probe_q, add_special_tokens=False).ids) + len(self._tokenizer.encode(probe_p, add_special_tokens=False).ids)
        self.pair_special_tokens = len(pair.ids) - raw

    def encode_pair(self, query: str, passage: str) -> EncodedPair:
        encoding = self._tokenizer.encode(query, passage)
        return EncodedPair(tuple(encoding.ids), tuple(encoding.type_ids))

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def files(self) -> list[Path]:
        return [self.directory / name for name in _FILES if (self.directory / name).is_file()]
