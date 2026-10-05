"""Turn reviewed questions into the query and qrel drafts that `prepare_data` consumes.

    build_drafts.py --work-dir work/ --split-manifest prepared/split-manifest.json --output-dir drafts/

A question becomes a draft only when it passed the mechanical checks AND an independent agent review said `answerable: yes`,
`self_contained: true`, `generic: false`. Every record is labelled for what it is: `origin: synthetic` with the generator's prompt hash,
and an `agent` review carrying the reviewer prompt's revision; nothing here can pass agent review off as human review. The positive
qrel's evidence span is the program-located quote, as document offsets. Writes `queries-<split>.jsonl`, `qrels-<split>.jsonl` and
`drafts-summary.json` (counts, rejection reasons). Never overwrites.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from devsearch.artifacts.canonical import canonical_json  # noqa: E402
from devsearch.data.prepare import load_split  # noqa: E402
from devsearch.data.queries import parse_qrel, parse_query  # noqa: E402

PROMPTS = Path(__file__).resolve().parent / "prompts"
SPLITS = ("train", "validation", "test")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    work = args.work_dir
    corpus = load_split(args.split_manifest)["corpus"]
    chunks = {c.chunk_id: c for c in corpus.chunks}
    selection = {r["ref"]: r for r in map(json.loads, (work / "selection.jsonl").read_text().splitlines())}
    accepted = {r["ref"]: r for r in map(json.loads, (work / "accepted-gen.jsonl").read_text().splitlines())}
    reviews, stamps = {}, {}
    for path in sorted((work / "rev").glob("*.jsonl")):
        stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                reviews[record["ref"]] = record
                stamps[record["ref"]] = stamp
    generator = {"model": "agent-workflow-generate-v1", "prompt_sha256": sha(PROMPTS / "generate-v1.md"), "seed": None}
    revision = "review-question-v1:" + sha(PROMPTS / "review-question-v1.md")[:16]
    out = {split: ([], []) for split in SPLITS}
    reasons = Counter()
    for ref, item in sorted(accepted.items()):
        review = reviews.get(ref)
        if review is None:
            reasons["not reviewed"] += 1
            continue
        if review.get("answerable") != "yes":
            reasons[f"reviewer: answerable={review.get('answerable')}"] += 1
            continue
        if review.get("self_contained") is not True:
            reasons["reviewer: not self-contained"] += 1
            continue
        if review.get("generic") is not False:
            reasons["reviewer: generic"] += 1
            continue
        row = selection[ref]
        chunk = chunks[row["chunk_id"]]
        doc = corpus.doc_of(chunk)
        split = row["split"]
        stamp = stamps[ref]
        review_record = {"reviewer_type": "agent", "reviewer_id": "agent-question-reviewer", "status": "accepted", "reviewed_at": stamp, "revision": revision}
        query_id = f"q-{ref}"
        query = {"query_id": query_id, "text": item["question"], "language": "en", "group_id": f"pg-{ref}", "evidence_group_ids": [row["evidence_group_id"]],
                 "parent_query_id": None, "task_family": item["task_family"], "split": split, "filters": {"sources": [], "versions": {}}, "origin": "synthetic",
                 "generator": generator, "review_status": "accepted", "reviews": [review_record]}
        qrel = {"query_id": query_id, "chunk_id": chunk.chunk_id, "relevance": 1, "label_source": "synthetic", "review_status": "accepted", "reviews": [review_record],
                "evidence_spans": [{"doc_id": doc.doc_id, "revision": doc.revision, "content_sha256": doc.content_sha256,
                                    "start_char": chunk.start_char + item["quote_start"], "end_char": chunk.start_char + item["quote_end"]}]}
        parse_query(query, f"query {ref}")                         # the real validators, before anything is written
        parse_qrel(qrel, f"qrel {ref}")
        out[split][0].append(query)
        out[split][1].append(qrel)
    args.output_dir.mkdir(exist_ok=True)
    for split, (queries, qrels) in out.items():
        for name, rows in (("queries", queries), ("qrels", qrels)):
            path = args.output_dir / f"{name}-{split}.jsonl"
            if path.exists():
                raise SystemExit(f"{path} already exists; drafts are never overwritten")
            path.write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
    summary = {"questions": {s: len(out[s][0]) for s in SPLITS}, "dropped": dict(reasons), "generator": generator, "reviewer_revision": revision,
               "labels": "synthetic questions written by an agent workflow and accepted by an independent AGENT review; no human reviewed any of them"}
    (args.output_dir / "drafts-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
