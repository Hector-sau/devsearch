"""Pick the passages that questions will be written for (one per evidence group), deterministically, and cut them into agent batches.

    select_passages.py --split-manifest prepared/split-manifest.json --output-dir work/ [--train 800] [--validation 250] [--test 300]
                       [--batch-size 20] [--seed 20261005]

A passage is eligible when it looks like explanatory prose: not a code fragment, at least 15 words outside code-looking lines, at most
70% code-looking lines, and not licence or boilerplate text. The filter is deliberately loose; the question writers skip what cannot
support a question, and those skips are counted. Each evidence group contributes at most one passage, so no two questions of a split share a
group, and a group's split is the owner the audited assignment gave it (nothing here can move evidence between splits). The agent
batches carry only `ref`, the document title, the section and the passage text: no chunk id, no split, no score.

Outputs in --output-dir: `selection.jsonl` (private: ref -> chunk id, group, split), `batches/<split>-NNN.jsonl` (what an agent sees)
and `selection-summary.json`.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from devsearch.data.prepare import load_split  # noqa: E402

CODE_LINE = re.compile(r"^\s*(import|from|def|class|for|while|if|return|print|with|try|except)\b|^\s*(@|#|>>>|\.\.\.|Traceback|File \")|[{}=;]|\w\(.*\)\s*$|^\s{4,}\S")
BOILERPLATE = re.compile(r"copyright|license|all rights reserved|\bauthor\b:|created on:|last updated:|prerequisites?\b", re.I)
SPLITS = ("train", "validation", "test")


def prose_words(text: str) -> int:
    return sum(len(line.split()) for line in text.splitlines() if line.strip() and not CODE_LINE.search(line))


def eligible(chunk) -> bool:
    text = chunk.text
    if chunk.code_fragment or not 100 <= len(text) <= 1800:
        return False
    if BOILERPLATE.search(text[:400]):
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    code_lines = sum(1 for line in lines if CODE_LINE.search(line))
    return prose_words(text) >= 15 and code_lines <= 0.7 * len(lines)


def rank_key(seed: int, group: str) -> str:
    return hashlib.sha256(f"{seed}|select|{group}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=int, default=800)
    parser.add_argument("--validation", type=int, default=250)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} must be an existing, empty directory")
    corpus = load_split(args.split_manifest)["corpus"]
    wanted = {"train": args.train, "validation": args.validation, "test": args.test}
    groups: dict[str, list] = {}
    for chunk in corpus.chunks:
        if chunk.owner_split in SPLITS and eligible(chunk):
            groups.setdefault(chunk.evidence_group_id, []).append(chunk)
    rows, summary = [], {"seed": args.seed, "eligible_groups": {}, "selected": {}}
    batches = args.output_dir / "batches"
    batches.mkdir()
    for split in SPLITS:
        pool = sorted((g for g, chunks in groups.items() if chunks[0].owner_split == split), key=lambda g: rank_key(args.seed, g))
        summary["eligible_groups"][split] = len(pool)
        picked = []
        for group in pool[:wanted[split]]:
            chunk = sorted(groups[group], key=lambda c: (-prose_words(c.text), c.chunk_id))[0]       # the most explanatory passage of the group
            picked.append((group, chunk))
        summary["selected"][split] = len(picked)
        for number, start in enumerate(range(0, len(picked), args.batch_size), start=1):
            batch = []
            for offset, (group, chunk) in enumerate(picked[start:start + args.batch_size]):
                ref = f"{split[:2]}-{start + offset + 1:04d}"
                doc = corpus.doc_of(chunk)
                rows.append({"ref": ref, "split": split, "chunk_id": chunk.chunk_id, "evidence_group_id": group, "doc_id": doc.doc_id})
                batch.append({"ref": ref, "title": doc.title, "section": " > ".join(chunk.heading_path), "text": chunk.text})
            (batches / f"{split}-{number:03d}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in batch), encoding="utf-8")
    (args.output_dir / "selection.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (args.output_dir / "selection-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
