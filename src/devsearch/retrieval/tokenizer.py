"""Lexical tokenizer `lexical-v1`, shared by indexing and querying (docs/04 "Index").

Rules, in order:

1. Unicode NFKC normalization.
2. A *run* is a word-character sequence that may continue across dots
   (`state_dict`, `torch.nn.Module`, `DataLoader`).
3. Each run emits, case-folded and without repeats: its **full token**; each
   dot-separated **component** (`AutoModel.from_pretrained` -> `automodel`,
   `from_pretrained`); then its **parts**, split further at `_`, camelCase, acronym
   and letter/digit boundaries (`auto`, `model`, `pretrained`; `float32` -> `float`,
   `32`). An exact identifier query therefore matches more tokens than plain words.
4. Tokens shorter than two characters, tokens without a letter or digit, and
   English stopwords are dropped.

There is deliberately no stemming. Changing any rule changes the tokenizer
fingerprint, which index manifests record and loaders compare.
"""

import re
import unicodedata

from devsearch.artifacts.canonical import digest_of

NAME = "lexical-v1"
ALGORITHM_REVISION = 1
MIN_TOKEN_LENGTH = 2

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers him his how i if in into is it its just me more most my
no nor not now of off on once only or other our out over own same she should so some such
than that the their them then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your
""".split())

_RUN = re.compile(r"\w+(?:\.\w+)*")
_SEPARATORS = re.compile(r"[._]+")


def _split_segment(segment: str) -> list[str]:
    """Split at camelCase, acronym-to-word and letter/digit boundaries."""
    classes = ["D" if c.isdigit() else "U" if c.isupper() else "L" for c in segment]
    parts, start = [], 0
    for i in range(1, len(segment)):
        before, here = classes[i - 1], classes[i]
        boundary = (
            (before == "L" and here == "U")
            or ((before == "D") != (here == "D"))
            or (before == "U" and here == "U" and i + 1 < len(segment) and classes[i + 1] == "L")
        )
        if boundary:
            parts.append(segment[start:i])
            start = i
    parts.append(segment[start:])
    return parts


def _keep(token: str) -> bool:
    return (
        len(token) >= MIN_TOKEN_LENGTH
        and any(ch.isalnum() for ch in token)
        and token not in STOPWORDS
    )


def tokenize(text: str) -> list[str]:
    """Tokens in document order, duplicates kept (term frequency counts them)."""
    tokens: list[str] = []
    for run in _RUN.findall(unicodedata.normalize("NFKC", text)):
        components = run.split(".")
        parts = [
            part for component in components
            for segment in _SEPARATORS.split(component) if segment
            for part in _split_segment(segment)
        ]
        emitted = dict.fromkeys(token.casefold() for token in [run, *components, *parts])
        tokens.extend(token for token in emitted if _keep(token))
    return tokens


def unique_query_terms(text: str) -> list[str]:
    """Query terms with binary weight: unique, in first-occurrence order."""
    return list(dict.fromkeys(tokenize(text)))


_PROBES = (
    "load_state_dict", "torch.nn.Module.eval()", "AutoTokenizer.from_pretrained('x')",
    "HTTPServer float32 __init__ 2.8.0", "How do I resume training?", "<unk> [CLS] no_grad",
    "ＮＦＫＣ ﬁle", "the of and a ?! x",
)


def fingerprint() -> str:
    """Hash of the rule set *and* its behavior on fixed probe strings.

    Index manifests record it and loaders compare it, so a behavior change that
    forgot to bump ALGORITHM_REVISION still invalidates old indexes.
    """
    return digest_of({
        "name": NAME, "algorithm_revision": ALGORITHM_REVISION,
        "normalization": "NFKC+casefold", "run_pattern": _RUN.pattern,
        "separators": _SEPARATORS.pattern, "min_token_length": MIN_TOKEN_LENGTH,
        "stopwords": sorted(STOPWORDS), "probe_outputs": [tokenize(probe) for probe in _PROBES],
    })
