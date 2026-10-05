"""Check agent-written questions mechanically, then cut the survivors into independent review batches.

    validate_generations.py --work-dir work/ --tokenizer-assets DIR [--max-query-tokens 64] [--batch-size 20]

Reads `selection.jsonl`, `batches/*.jsonl` (the passages) and `gen/*.jsonl` (the agents' output). A question survives only if: its
ref exists once, it is a trimmed single-line question of 6 to 35 words that fits the query-token reserve, it never refers to hidden
context ("the passage", "this tutorial" ...), shares no run of six consecutive words with its passage, is not a near duplicate of an
earlier question, and its evidence quote occurs in the passage (whitespace-tolerant), at most 400 characters. Evidence offsets are
computed here from the quote, never trusted from an agent.

Writes `accepted-gen.jsonl`, `rejected-gen.jsonl`, `needs-regeneration.jsonl` (refs without a usable line) and `review-batches/`.
The review batches show a reviewer only the question and the passage: never the quote, the generator or any label.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

HIDDEN = re.compile(r"\b(the|this|that)\s+(passage|text|tutorial|section|snippet|notebook|example above|code above|code below|video|figure)\b|\b(above|below) example\b", re.I)
FAMILIES = ("how-to", "why", "what-is", "api", "error", "comparison", "other")
WORD = re.compile(r"\w+")


def grams(text: str, n: int = 6) -> set[tuple[str, ...]]:
    words = [w.lower() for w in WORD.findall(text)]
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def locate(quote: str, text: str):
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(w) for w in words), text)
    return (match.start(), match.end()) if match else None


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-assets", type=Path, required=True)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    from devsearch.modeling.tokenization import WordPieceTokenizer
    tokenizer = WordPieceTokenizer(args.tokenizer_assets)
    work = args.work_dir
    selection = {r["ref"]: r for r in map(json.loads, (work / "selection.jsonl").read_text().splitlines())}
    passages = {}
    for path in sorted((work / "batches").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            passages[record["ref"]] = record
    accepted, rejected, seen_refs, kept_words = [], [], set(), {}
    for path in sorted((work / "gen").glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{number}"
            try:
                record = json.loads(line)
                ref = record["ref"]
            except (ValueError, KeyError, TypeError):
                rejected.append({"where": where, "reasons": ["not a JSON object with a ref"]})
                continue
            reasons = []
            if ref not in selection:
                rejected.append({"where": where, "ref": ref, "reasons": ["unknown ref"]})
                continue
            if ref in seen_refs:
                rejected.append({"where": where, "ref": ref, "reasons": ["duplicate ref"]})
                continue
            seen_refs.add(ref)
            if "skip" in record:
                rejected.append({"where": where, "ref": ref, "reasons": [f"agent skipped: {str(record['skip'])[:120]}"]})
                continue
            question, quote, family = record.get("question"), record.get("evidence_quote"), record.get("task_family")
            text = passages[ref]["text"]
            if not isinstance(question, str) or not isinstance(quote, str):
                rejected.append({"where": where, "ref": ref, "reasons": ["question and evidence_quote must be strings"]})
                continue
            if question != question.strip() or not question.isprintable() or not question.endswith("?"):
                reasons.append("not a trimmed single-line question ending with ?")
            count = len(question.split())
            if not 6 <= count <= 35:
                reasons.append(f"{count} words")
            if HIDDEN.search(question):
                reasons.append("refers to hidden context")
            if grams(question) & grams(text):
                reasons.append("copies six or more consecutive words from the passage")
            if tokenizer.count(question) > args.max_query_tokens:
                reasons.append("exceeds the query-token reserve")
            if family not in FAMILIES:
                reasons.append("unknown task_family")
            span = locate(quote, text) if len(quote) <= 400 else None
            if span is None:
                reasons.append("evidence quote is not an exact (whitespace-tolerant) part of the passage or longer than 400 characters")
            words = set(w.lower() for w in WORD.findall(question))
            split = selection[ref]["split"]
            if any(other_split == split and jaccard(words, other) >= 0.85 for other_split, other in kept_words.values()):
                reasons.append("near duplicate of an earlier question")
            if reasons:
                rejected.append({"where": where, "ref": ref, "reasons": reasons})
                continue
            kept_words[ref] = (split, words)
            accepted.append({"ref": ref, "question": question, "task_family": family, "quote_start": span[0], "quote_end": span[1]})
    missing = sorted(set(selection) - seen_refs)
    (work / "accepted-gen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sorted(accepted, key=lambda r: r["ref"])))
    (work / "rejected-gen.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejected))
    regenerate = sorted({r["ref"] for r in rejected if "ref" in r} | set(missing))
    (work / "needs-regeneration.jsonl").write_text("".join(json.dumps({"ref": ref, "split": selection[ref]["split"]}) + "\n" for ref in regenerate if ref in selection))
    review = work / "review-batches"
    review.mkdir(exist_ok=True)
    by_split: dict[str, list] = {}
    for item in sorted(accepted, key=lambda r: r["ref"]):
        by_split.setdefault(selection[item["ref"]]["split"], []).append(item)
    for split, items in by_split.items():
        for number, start in enumerate(range(0, len(items), args.batch_size), start=1):
            rows = [{"ref": it["ref"], "question": it["question"], "passage": {k: passages[it["ref"]][k] for k in ("title", "section", "text")}} for it in items[start:start + args.batch_size]]
            (review / f"{split}-{number:03d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"selected": len(selection), "generated_lines": len(seen_refs), "accepted": len(accepted), "rejected": len(rejected), "never_answered": len(missing),
                      "per_split_accepted": {s: len(v) for s, v in by_split.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
