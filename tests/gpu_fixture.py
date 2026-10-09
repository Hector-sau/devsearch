"""A small, original fixture world for `scripts/gpu.py`: a local git source repository, a WordPiece model directory with a tiny
random encoder, and a dataset package built by `gpu.py package` itself. Everything is synthetic and says so; nothing is downloaded.
"""

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from helpers import REPO

GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid", "GIT_AUTHOR_DATE": "2026-10-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2026-10-01T00:00:00Z"}
SYLLABLES = ("ba", "ko", "ri", "tu", "ve", "mi", "so", "la", "ne", "du", "pa", "xi", "zo", "fre", "gli", "tho")
FILLER = "the value is stored when a request arrives and each stage keeps its own state for later use so that retries stay safe".split()
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

SOURCES = """schema_version = 1
parser_version = "static-extract-v2"
chunker_version = "chunker-v1"
[policy]
max_pair_tokens = 256
max_query_tokens = 64
min_chunk_tokens = 8
overlong_code = "split"
max_file_bytes = 2000000
max_files = 200
[tokenizer]
assets = "@PINNED_TOKENIZER@"
[[source]]
source_id = "fixture-docs"
revision = "{revision}"
snapshot = "@SNAPSHOTS@/fixture-docs"
git_url = "file://{repo}"
url_template = "https://docs.fixture.invalid/blob/{{revision}}/{{path}}"
license = "fixture-original"
include = ["docs/**/*.md"]
"""

DATA = """# Fixture dataset (synthetic). @PINNED_TOKENIZER@ is filled in by scripts/gpu.py.
schema_version = 1
protocol = "prejudged_fixed_candidates_v1"
[inputs.queries]
train = "drafts/queries-train.jsonl"
validation = "drafts/queries-validation.jsonl"
test = "drafts/queries-test.jsonl"
[inputs.qrels]
train = "drafts/qrels-train.jsonl"
validation = "drafts/qrels-validation.jsonl"
test = "drafts/qrels-test.jsonl"
[groups]
near_duplicate_threshold = 0.9
shingle_size = 5
seed = 11
[groups.targets]
train = 0.5
validation = 0.25
test = 0.25
[mining]
hard_negatives = 2
random_negatives = 2
seed = 42
hard_search_depth = 100
[candidates]
depth = 50
[index]
config = "index.toml"
[review]
salt = "fixture-salt-not-a-secret"
[lengths]
tokenizer_assets = "@PINNED_TOKENIZER@"
max_pair_tokens = 256
max_query_tokens = 64
"""

INDEX = """schema_version = 1
tokenizer = "lexical-v1"
[bm25]
idf = "lucene"
k1 = 1.2
b = 0.75
"""


def load_gpu():
    spec = importlib.util.spec_from_file_location("devsearch_gpu_script", REPO / "scripts" / "gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, env={**os.environ, **GIT_ENV}, capture_output=True, text=True, check=True).stdout.strip()


def _words(rng, count):
    seen = set()
    while len(seen) < count:
        seen.add("".join(rng.choice(SYLLABLES) for _ in range(rng.choice((2, 3)))))
    return sorted(seen)


def source_repo(directory: Path, topics: int = 24, seed: int = 5) -> str:
    """Original markdown documents, three sections each; returns the commit."""
    rng = random.Random(seed)
    names = _words(rng, topics * 4)
    (directory / "docs").mkdir(parents=True)
    for index in range(topics):
        a, b, c, d = names[index * 4:index * 4 + 4]
        (directory / "docs" / f"{a}.md").write_text(
            f"# Topic {a}\n\n## Overview\n\nThe {a} component keeps a {b} register and a {c} routine for each {d}. "
            f"{' '.join(rng.sample(FILLER, 10))}.\n\n## Usage\n\nCall {a} with a {b} handle, then flush the {d} queue. "
            f"{' '.join(rng.sample(FILLER, 9))}.\n\n## Notes\n\nDo not share a {c} between two {a} instances or reset the {b} register. "
            f"{' '.join(rng.sample(FILLER, 8))}.\n", encoding="utf-8")
    git("init", "-q", "-b", "main", cwd=directory)
    git("add", "-A", cwd=directory)
    git("commit", "-q", "-m", "fixture documents", cwd=directory)
    return git("rev-parse", "HEAD", cwd=directory)


def model_dir(directory: Path, source: Path) -> None:
    """vocab.txt (every lowercase word of the fixture docs and questions), tokenizer_config.json and a tiny random BERT."""
    from safetensors.torch import save_file

    from devsearch.modeling.cross_encoder import build_tiny_encoder

    import re
    words = set()
    for path in sorted(source.rglob("*.md")):
        words.update(re.findall(r"[a-z0-9]+", path.read_text(encoding="utf-8").lower()))
    words.update("how does the work with and in case".split())
    punctuation = list(".,?!()_-#:;'\"/")
    vocab = SPECIALS + sorted(words) + punctuation + [str(n) for n in range(10)]
    directory.mkdir(parents=True)
    (directory / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)) + "\n", encoding="utf-8")
    (directory / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": True}), encoding="utf-8")
    size = len((directory / "vocab.txt").read_text(encoding="utf-8").splitlines())
    encoder = build_tiny_encoder(size, hidden_size=32, layers=2, seed=1, hidden_dropout=0.1, attention_dropout=0.1)
    save_file({k: v.detach().contiguous().clone() for k, v in encoder.state_dict().items()}, str(directory / "model.safetensors"))
    c = encoder.config
    (directory / "config.json").write_text(json.dumps({
        "model_type": "bert", "vocab_size": c.vocab_size, "hidden_size": c.hidden_size, "num_hidden_layers": c.num_hidden_layers,
        "num_attention_heads": c.num_attention_heads, "intermediate_size": c.intermediate_size, "max_position_embeddings": c.max_position_embeddings,
        "hidden_act": c.hidden_act, "hidden_dropout_prob": c.hidden_dropout_prob, "attention_probs_dropout_prob": c.attention_probs_dropout_prob,
        "type_vocab_size": c.type_vocab_size, "initializer_range": c.initializer_range, "layer_norm_eps": c.layer_norm_eps,
        "pad_token_id": c.pad_token_id}, indent=1), encoding="utf-8")


def run_module(args, root: Path):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "DEVSEARCH_RUNTIME_ROOT": str(root)}
    result = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, env=env, timeout=1200)
    if result.returncode != 0:
        raise AssertionError(f"{args} exited {result.returncode}: {result.stderr[-2000:]}")
    return result


class World:
    """Everything a `gpu.py` test needs, under one temporary directory."""

    def __init__(self, base: Path):
        self.base = base
        self.repo = base / "source-repo"
        self.commit = source_repo(self.repo)
        self.model = base / "model"
        model_dir(self.model, self.repo)
        self.package = base / "dataset.tar.gz"
        self.package_sha256 = self._package()

    def _package(self) -> str:
        """Drafts need the split, so a scratch build comes first; then `gpu.py package` rebuilds and freezes the values."""
        from devsearch.data.prepare import load_split
        from devsearch.execution.rehearsal import draft_queries

        src = self.base / "package-src"
        (src / "drafts").mkdir(parents=True)
        (src / "sources.toml").write_text(SOURCES.format(revision=self.commit, repo=self.repo), encoding="utf-8")
        (src / "data.toml").write_text(DATA, encoding="utf-8")
        (src / "index.toml").write_text(INDEX, encoding="utf-8")
        scratch = self.base / "scratch"
        (scratch / "snapshots").mkdir(parents=True)
        (scratch / "corpus").mkdir()
        (scratch / "prepared").mkdir()
        (scratch / "tokenizer").mkdir()
        for name in ("vocab.txt", "tokenizer_config.json"):
            shutil.copyfile(self.model / name, scratch / "tokenizer" / name)
        sources = scratch / "sources.toml"
        sources.write_text((src / "sources.toml").read_text().replace("@PINNED_TOKENIZER@", str(scratch / "tokenizer")).replace("@SNAPSHOTS@", str(scratch / "snapshots")))
        data = scratch / "data.toml"
        data.write_text(DATA.replace("@PINNED_TOKENIZER@", str(scratch / "tokenizer")).replace('"drafts/', '"' + str(src / "drafts") + "/")
                        .replace('"index.toml"', '"' + str(src / "index.toml") + '"'))
        run_module(["-m", "devsearch.prepare", "--config", sources, "--output", scratch / "corpus", "--runtime-root", self.base], self.base)
        run_module(["-m", "devsearch.prepare_data", "--config", data, "--stage", "groups", "--corpus", scratch / "corpus" / "corpus-manifest.json",
                    "--output", scratch / "prepared" / "split-manifest.json", "--runtime-root", self.base], self.base)
        counts = draft_queries(load_split(scratch / "prepared" / "split-manifest.json"), src / "drafts", seed=3)
        assert min(counts.values()) >= 3, counts
        work = self.base / "package-work"
        work.mkdir()
        result = subprocess.run([sys.executable, REPO / "scripts" / "gpu.py", "package", "--sources", src / "sources.toml", "--data", src / "data.toml",
                                 "--index", src / "index.toml", "--drafts", src / "drafts", "--reference-tokenizer", scratch / "tokenizer",
                                 "--work", work, "--output", self.package, "--notice", "fixture"],
                                capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=1200)
        if result.returncode != 0:
            raise AssertionError(f"gpu.py package failed: {result.stderr[-2000:]}")
        return json.loads(result.stdout)["sha256"]

    def config(self, root: Path, **overrides) -> Path:
        values = {"profile": "cpu-diagnostic", "kind": "diagnostic", "sha256": self.package_sha256, "record": "false"}
        values.update(overrides)
        path = root.parent / f"{root.name}-{len(list(root.parent.glob(root.name + '-*.toml')))}.toml"
        path.write_text(f"""schema_version = 1
[runtime]
root = "{root}"
time_limit_hours = 2
min_free_gib = 0.1
step_timeout_minutes = 30
[backend]
profile = "{values['profile']}"
[environment]
mode = "current"
[model]
repo = "fixture/tiny-wordpiece"
revision = "{'1' * 40}"
license = "fixture-original"
source = "local"
local_dir = "{self.model}"
kind = "{values['kind']}"
[dataset]
package = "inbox/dataset.tar.gz"
sha256 = "{values['sha256']}"
[receipts]
record = {values['record']}
""", encoding="utf-8")
        return path

    def runtime(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="gpu-root-", dir=self.base)).resolve()
        (root / "inbox").mkdir()
        shutil.copyfile(self.package, root / "inbox" / "dataset.tar.gz")
        return root


def oracle_judgments(root: Path, batch_dir: Path) -> list[dict]:
    """Scripted stand-in judge for the fixture only: relevant iff the passage is the drafted positive of the question."""
    dataset = root / "dataset"
    queries = {}
    for split in ("train", "validation", "test"):
        for line in (dataset / "drafts" / f"queries-{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            queries[record["query_id"]] = record["text"]
    chunks = {json.loads(line)["chunk_id"]: json.loads(line)["text"] for line in (root / "data" / "corpus" / "chunks.jsonl").read_text().splitlines()}
    positive = set()
    for split in ("train", "validation", "test"):
        for line in (dataset / "drafts" / f"qrels-{split}.jsonl").read_text().splitlines():
            record = json.loads(line)
            positive.add((queries[record["query_id"]], chunks[record["chunk_id"]]))
    lines = []
    for path in sorted(batch_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            relevant = (item["question"], item["text"]) in positive
            lines.append({"item_id": item["item_id"], "relevance": 1 if relevant else 0,
                          "evidence_quote": " ".join(item["text"].split()[:6]) if relevant else "", "note": "fixture oracle"})
    return lines
