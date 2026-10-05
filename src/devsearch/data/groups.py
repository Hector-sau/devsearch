"""Evidence-group merging, deterministic split assignment and the cross-split audit (docs/02 §4).

Splits are assigned to *merged groups*, never to individual passages or questions:

1. `merge_evidence_groups` unions the chunker's declared evidence groups whenever their
   passages are exact duplicates, near duplicates (shingle Jaccard >= threshold, found with
   a small bottom-k sketch and then verified exactly), overlapping spans of one document,
   or explicitly linked (paraphrase/parent links between questions).
2. `assign_groups` partitions merged groups across splits with a seeded greedy rule that
   depends only on group identities, so it is reproducible and independent of input order.
3. `audit_assignment` re-derives the dangerous overlaps *independently of the merger* and
   reports every violation, so a bug in merging cannot silently put one piece of evidence
   on both sides of the train/test boundary.
"""

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from devsearch.artifacts.canonical import digest_of
from devsearch.data.corpus import SPLITS, Corpus
from devsearch.data.queries import Query
from devsearch.errors import ManifestError

_WORDS = re.compile(r"\w+")
SKETCH_SIZE = 8                      # smallest shingle hashes kept per passage as LSH keys
MAX_BUCKET = 200                     # larger buckets are boilerplate; exact duplicates are handled separately


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def shingles(text: str, size: int) -> frozenset[int]:
    words = _WORDS.findall(text.casefold())
    grams = [" ".join(words[i:i + size]) for i in range(max(len(words) - size + 1, 1))] if words else []
    return frozenset(int.from_bytes(hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest(), "big") for g in grams)


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


@dataclass(frozen=True)
class GroupMerge:
    corpus_id: str
    merged_of: Mapping[str, str]                 # evidence_group_id -> merged_group_id
    members: Mapping[str, tuple[str, ...]]       # merged_group_id -> evidence_group_ids
    chunks_of: Mapping[str, tuple[str, ...]]     # merged_group_id -> chunk_ids
    method: Mapping[str, Any]
    merges_by_rule: Mapping[str, int] = field(default_factory=dict)


def merge_evidence_groups(
    corpus: Corpus, *, near_duplicate_threshold: float = 0.9, shingle_size: int = 5,
    extra_links: Iterable[tuple[str, str]] = (),
) -> GroupMerge:
    """Union declared evidence groups through duplicate/overlap/explicit-link rules."""
    if not 0.5 <= near_duplicate_threshold <= 1.0:
        raise ManifestError("near_duplicate_threshold must be between 0.5 and 1.0")
    groups = sorted({chunk.evidence_group_id for chunk in corpus.chunks})
    union = _UnionFind(groups)
    merges: dict[str, int] = defaultdict(int)

    by_text: dict[str, list[str]] = defaultdict(list)
    for chunk in corpus.chunks:
        by_text[normalize_text(chunk.text)].append(chunk.evidence_group_id)
    for members in by_text.values():
        for other in members[1:]:
            merges["exact_duplicate"] += union.union(members[0], other)

    sets = {chunk.chunk_id: shingles(chunk.text, shingle_size) for chunk in corpus.chunks}
    group_of = {chunk.chunk_id: chunk.evidence_group_id for chunk in corpus.chunks}
    buckets: dict[int, list[str]] = defaultdict(list)
    for chunk_id, grams in sets.items():
        for key in sorted(grams)[:SKETCH_SIZE]:
            buckets[key].append(chunk_id)
    checked: set[tuple[str, str]] = set()
    for bucket in buckets.values():
        if len(bucket) < 2 or len(bucket) > MAX_BUCKET:
            continue
        ordered = sorted(bucket)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if group_of[a] == group_of[b] or (a, b) in checked:
                    continue
                checked.add((a, b))
                if jaccard(sets[a], sets[b]) >= near_duplicate_threshold:
                    merges["near_duplicate"] += union.union(group_of[a], group_of[b])

    spans: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for chunk in corpus.chunks:
        spans[(chunk.doc_id, chunk.revision)].append((chunk.start_char, chunk.end_char, chunk.evidence_group_id))
    for items in spans.values():
        items.sort()
        reach, holder = -1, ""
        for start, end, group in items:
            if start < reach:
                merges["overlap"] += union.union(holder, group)
            if end > reach:
                reach, holder = end, group

    for a, b in extra_links:
        if a not in union.parent or b not in union.parent:
            raise ManifestError(f"explicit group link names an unknown evidence group: {a!r}, {b!r}")
        merges["explicit_link"] += union.union(a, b)

    members: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        members[union.find(group)].append(group)
    merged_of: dict[str, str] = {}
    named_members: dict[str, tuple[str, ...]] = {}
    for root, items in members.items():
        merged_id = "mg-" + digest_of(sorted(items))[:16]
        named_members[merged_id] = tuple(sorted(items))
        for group in items:
            merged_of[group] = merged_id
    chunks_of: dict[str, list[str]] = defaultdict(list)
    for chunk in corpus.chunks:
        chunks_of[merged_of[chunk.evidence_group_id]].append(chunk.chunk_id)
    return GroupMerge(
        corpus_id=corpus.corpus_id, merged_of=merged_of, members=dict(sorted(named_members.items())),
        chunks_of={k: tuple(v) for k, v in sorted(chunks_of.items())},
        method={"near_duplicate_threshold": near_duplicate_threshold, "shingle_size": shingle_size,
                "sketch_size": SKETCH_SIZE, "rules": ["exact_duplicate", "near_duplicate", "overlap", "explicit_link"]},
        merges_by_rule=dict(merges),
    )


def assign_groups(
    merge: GroupMerge, *, targets: Mapping[str, float], seed: int,
    weights: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """merged_group_id -> split. Seeded, order-independent greedy fill toward target shares."""
    if not targets or any(split not in SPLITS for split in targets) or any(v <= 0 for v in targets.values()):
        raise ManifestError(f"targets must be positive shares over {', '.join(SPLITS)}")
    total_share = sum(targets.values())
    shares = {split: value / total_share for split, value in targets.items()}
    order = sorted(shares, key=SPLITS.index)
    load = {split: 0.0 for split in order}
    assignment: dict[str, str] = {}
    for merged_id in sorted(merge.members, key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).hexdigest()):
        weight = max((weights or {}).get(merged_id, len(merge.chunks_of[merged_id])), 1)
        total = sum(load.values()) + weight
        # the split currently furthest below its target share of the load after adding this group
        chosen = max(order, key=lambda s: (shares[s] * total - load[s], -order.index(s)))
        assignment[merged_id] = chosen
        load[chosen] += weight
    return dict(sorted(assignment.items()))


def owner_of_chunks(corpus: Corpus, merge: GroupMerge, assignment: Mapping[str, str]) -> dict[str, str]:
    """chunk_id -> owner split, via the chunk's evidence group and its merged group."""
    try:
        return {chunk.chunk_id: assignment[merge.merged_of[chunk.evidence_group_id]] for chunk in corpus.chunks}
    except KeyError as error:
        raise ManifestError(f"assignment does not cover merged group {error.args[0]!r}") from error


@dataclass(frozen=True)
class AuditReport:
    violations: tuple[str, ...]
    counts: Mapping[str, int]

    @property
    def ok(self) -> bool:
        return not self.violations


def audit_assignment(
    corpus: Corpus, merge: GroupMerge, assignment: Mapping[str, str], queries: Sequence[Query] = (),
    *, near_duplicate_threshold: float = 0.9, shingle_size: int = 5,
) -> AuditReport:
    """Independent cross-split checks; recomputes overlaps without using `merge` logic."""
    violations: list[str] = []
    missing, unknown = sorted(set(merge.members) - set(assignment)), sorted(set(assignment) - set(merge.members))
    if missing or unknown:
        return AuditReport((
            f"assignment does not cover every merged group: {len(missing)} missing, {len(unknown)} unknown",
        ), {})
    owner = owner_of_chunks(corpus, merge, assignment)

    # (1) a declared owner on a chunk record must agree with the assignment
    for chunk in corpus.chunks:
        if chunk.owner_split is not None and chunk.owner_split != owner[chunk.chunk_id]:
            violations.append(f"chunk {chunk.chunk_id}: record says {chunk.owner_split!r} but assignment says {owner[chunk.chunk_id]!r}")
    # (2) one evidence group, one owner (re-derived straight from chunks)
    group_owners: dict[str, set[str]] = defaultdict(set)
    for chunk in corpus.chunks:
        group_owners[chunk.evidence_group_id].add(owner[chunk.chunk_id])
    for group, owners in group_owners.items():
        if len(owners) > 1:
            violations.append(f"evidence group {group!r} spans splits {sorted(owners)}")
    # (3) exact-duplicate passages on both sides of a boundary
    text_owners: dict[str, set[str]] = defaultdict(set)
    for chunk in corpus.chunks:
        text_owners[normalize_text(chunk.text)].add(owner[chunk.chunk_id])
    exact = sum(1 for owners in text_owners.values() if len(owners) > 1)
    violations += [f"{exact} exact-duplicate passage text(s) owned by more than one split"] if exact else []
    # (4) overlapping spans in different splits
    spans: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for chunk in corpus.chunks:
        spans[(chunk.doc_id, chunk.revision)].append((chunk.start_char, chunk.end_char, owner[chunk.chunk_id]))
    overlaps = 0
    for items in spans.values():
        items.sort()
        for i, (s1, e1, o1) in enumerate(items):
            for s2, e2, o2 in items[i + 1:]:
                if s2 >= e1:
                    break
                overlaps += o1 != o2
    violations += [f"{overlaps} overlapping passage pair(s) owned by different splits"] if overlaps else []
    # (5) near-duplicate passages across splits (verified exactly, like the merger but recomputed)
    sets = {c.chunk_id: shingles(c.text, shingle_size) for c in corpus.chunks}
    buckets: dict[int, list[str]] = defaultdict(list)
    for chunk_id, grams in sets.items():
        for key in sorted(grams)[:SKETCH_SIZE]:
            buckets[key].append(chunk_id)
    near = set()
    for bucket in buckets.values():
        if 1 < len(bucket) <= MAX_BUCKET:
            for i, a in enumerate(sorted(bucket)):
                for b in sorted(bucket)[i + 1:]:
                    if owner[a] != owner[b] and jaccard(sets[a], sets[b]) >= near_duplicate_threshold:
                        near.add((a, b))
    violations += [f"{len(near)} near-duplicate passage pair(s) owned by different splits"] if near else []
    # (6) questions: split must equal the owner of every evidence group; paraphrases stay together
    group_split = {group: next(iter(owners)) for group, owners in group_owners.items() if len(owners) == 1}
    by_group: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, set[str]] = defaultdict(set)
    for query in queries:
        for group in query.evidence_group_ids:
            if group_split.get(group) != query.split:
                violations.append(f"query {query.query_id} (split {query.split}) uses evidence group {group!r} owned by {group_split.get(group)!r}")
        by_group[query.group_id].add(query.split)
        texts[normalize_text(query.text)].add(query.split)
    violations += [f"paraphrase group {g!r} spans splits {sorted(s)}" for g, s in sorted(by_group.items()) if len(s) > 1]
    duplicate_questions = sum(1 for splits in texts.values() if len(splits) > 1)
    violations += [f"{duplicate_questions} identical question text(s) appear in more than one split"] if duplicate_questions else []
    counts = {f"groups_{s}": 0 for s in SPLITS} | {f"chunks_{s}": 0 for s in SPLITS} | {f"queries_{s}": 0 for s in SPLITS}
    for split in assignment.values():
        counts[f"groups_{split}"] += 1
    for split in owner.values():
        counts[f"chunks_{split}"] += 1
    for query in queries:
        counts[f"queries_{query.split}"] += 1
    return AuditReport(tuple(violations), counts)
