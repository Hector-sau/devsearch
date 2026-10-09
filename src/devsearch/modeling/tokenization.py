"""WordPiece tokenizer built on the `tokenizers` library (the fast core Transformers wraps).

Wrapping the core library directly keeps pair encoding independent of Transformers' tokenizer
classes, whose constructors changed between major versions (one silently ignored a vocabulary
file in testing). A tokenizer directory holds either `tokenizer.json` (the complete pipeline) or `vocab.txt` with optional
`tokenizer_config.json`, `special_tokens_map.json` and `added_tokens.json`; its identity is the SHA-256 of those files.
For a `vocab.txt` directory the BERT pipeline honours `do_lower_case`, `strip_accents` and `tokenize_chinese_chars`, and
REFUSES any declared setting it cannot reproduce (no basic tokenization, `never_split`, non-default special tokens, added
tokens), so a pinned repository can never be encoded differently from its own configuration without an error.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

from devsearch.artifacts.canonical import digest_of, sha256_file
from devsearch.errors import ManifestError
from devsearch.modeling.policy import EncodedPair

SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
_FILES = ("tokenizer.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
DEFAULT_SPECIALS = {"unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]", "mask_token": "[MASK]"}


def _token_text(value):
    return value.get("content") if isinstance(value, dict) else value


def vocab_settings(config: dict, special_map: dict, added: dict, vocab: list[str]) -> dict:
    """The encoding settings a `vocab.txt` directory declares, as `build_wordpiece` arguments; anything it cannot reproduce is refused."""
    def flag(name: str, default, allow_none: bool = False):
        value = config.get(name, default)
        if not (isinstance(value, bool) or (allow_none and value is None)):
            raise ManifestError(f"tokenizer_config.json: {name} must be true or false{' or null' if allow_none else ''}, got {value!r}")
        return value
    lowercase = flag("do_lower_case", True)
    strip_accents = flag("strip_accents", None, allow_none=True)
    chinese = flag("tokenize_chinese_chars", True)
    if flag("do_basic_tokenize", True) is not True:
        raise ManifestError("tokenizer_config.json: do_basic_tokenize false is not supported by this BERT pipeline")
    if config.get("never_split"):
        raise ManifestError("tokenizer_config.json: never_split is not supported by this BERT pipeline")
    for source_name, source in (("tokenizer_config.json", config), ("special_tokens_map.json", special_map)):
        for key, default in DEFAULT_SPECIALS.items():
            if key in source and _token_text(source[key]) != default:
                raise ManifestError(f"{source_name}: {key} {_token_text(source[key])!r} differs from the BERT default {default!r}, which this pipeline uses")
        if source.get("additional_special_tokens"):
            raise ManifestError(f"{source_name}: additional special tokens are not supported by this BERT pipeline")
    if added:
        raise ManifestError("added_tokens.json adds tokens to the vocabulary; this BERT pipeline does not handle added tokens")
    for index, entry in (config.get("added_tokens_decoder") or {}).items():
        content = _token_text(entry)
        if not (str(index).isdigit() and int(index) < len(vocab) and vocab[int(index)] == content):
            raise ManifestError(f"tokenizer_config.json: added token {content!r} at id {index} is not that vocabulary entry")
    return {"lowercase": lowercase, "strip_accents": strip_accents, "handle_chinese_chars": chinese}


def build_wordpiece(vocab: list[str], *, lowercase: bool = True, strip_accents: bool | None = None, handle_chinese_chars: bool = True) -> Tokenizer:
    """A BERT-style WordPiece tokenizer from an ordered vocabulary that contains the specials (`strip_accents=None` follows `lowercase`, as in BERT)."""
    mapping = {token: index for index, token in enumerate(vocab)}
    if len(mapping) != len(vocab):
        raise ManifestError("vocabulary contains duplicate tokens")
    for special in SPECIALS[1:4]:
        if special not in mapping:
            raise ManifestError(f"vocabulary lacks {special}")
    tokenizer = Tokenizer(models.WordPiece(mapping, unk_token="[UNK]", max_input_chars_per_word=100))
    tokenizer.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=handle_chinese_chars, strip_accents=strip_accents, lowercase=lowercase)
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
            def read(name: str) -> dict:
                if name not in present:
                    return {}
                value = json.loads(present[name].read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ManifestError(f"{present[name]} must hold a JSON object")
                return value
            vocab = present["vocab.txt"].read_text(encoding="utf-8").splitlines()
            settings = vocab_settings(read("tokenizer_config.json"), read("special_tokens_map.json"), read("added_tokens.json"), vocab)
            self._tokenizer = build_wordpiece(vocab, **settings)
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

    def vocab(self) -> dict[str, int]:
        """Token -> id, for building exact-length probe sequences without touching real text."""
        return self._tokenizer.get_vocab()

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def files(self) -> list[Path]:
        return [self.directory / name for name in _FILES if (self.directory / name).is_file()]
