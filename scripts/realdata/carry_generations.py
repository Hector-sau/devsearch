"""Reuse generated questions after the corpus was rebuilt (a parser fix, a real tokenizer), and cut the rest into generation batches.

    carry_generations.py --old-work-dir qwork/ --new-work-dir qwork2/ [--batch-size 40]

`qwork2` is a fresh `select_passages.py` run on the rebuilt corpus (new refs, new owner splits). A generation line is carried over only when the
new selection holds a chunk with the SAME chunk id and the SAME passage text as an old selected passage: the question then still describes
exactly that text, whichever split the rebuilt assignment gave it. The carried lines go to `gen/carried.jsonl` under their new refs; the
passages without one go to `batches-todo/` (same record shape as `batches/`) for the generation agents. Nothing is judged here, and the old
work directory is never modified. Reviews are not carried: run this before the review step.
"""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_dir(path: Path) -> list[dict]:
    return [record for file in sorted(path.glob("*.jsonl")) for record in read_jsonl(file)]


def carry(old_selection: list[dict], old_passages: dict, old_gen: dict, new_selection: list[dict], new_passages: dict) -> tuple[list[dict], list[str]]:
    """The carried lines (with new refs) and the new refs that still need generation."""
    by_chunk = {row["chunk_id"]: row["ref"] for row in old_selection}
    carried, todo = [], []
    for row in new_selection:
        ref = row["ref"]
        old_ref = by_chunk.get(row["chunk_id"])
        if old_ref is not None and old_ref in old_gen and old_ref in old_passages and old_passages[old_ref]["text"] == new_passages[ref]["text"]:
            carried.append({**old_gen[old_ref], "ref": ref})
        else:
            todo.append(ref)
    return carried, todo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-work-dir", type=Path, required=True)
    parser.add_argument("--new-work-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()
    new = args.new_work_dir
    if (new / "gen").exists() or (new / "batches-todo").exists():
        raise SystemExit(f"{new} already has gen/ or batches-todo/; carrying over never overwrites")
    old_selection = read_jsonl(args.old_work_dir / "selection.jsonl")
    old_passages = {r["ref"]: r for r in read_dir(args.old_work_dir / "batches")}
    old_gen = {r["ref"]: r for r in read_dir(args.old_work_dir / "gen") if "ref" in r}
    new_selection = read_jsonl(new / "selection.jsonl")
    new_passages = {r["ref"]: r for r in read_dir(new / "batches")}
    carried, todo = carry(old_selection, old_passages, old_gen, new_selection, new_passages)
    (new / "gen").mkdir()
    (new / "gen" / "carried.jsonl").write_text("".join(json.dumps(line) + "\n" for line in carried), encoding="utf-8")
    (new / "batches-todo").mkdir()
    split_of = {row["ref"]: row["split"] for row in new_selection}
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        refs = [ref for ref in todo if split_of[ref] == split]
        counts[split] = len(refs)
        for number, start in enumerate(range(0, len(refs), args.batch_size), start=1):
            rows = [new_passages[ref] for ref in refs[start:start + args.batch_size]]
            (new / "batches-todo" / f"{split}-{number:03d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    report = {"new_selection": len(new_selection), "carried": len(carried), "to_generate": len(todo), "to_generate_per_split": counts}
    (new / "carry-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
