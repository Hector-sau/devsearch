"""Which judgments make a review item final: the declared review protocol of the judgment bank (docs/02 "Review protocol revision 5").

Two modes, chosen in the data config's `[review]` table before any formal judging and bound into its hash:

* `double_primary` (the original rule): every item needs an adjudicator's resolved judgment, or resolved, unanimous primary
  judgments from at least `min_primary_reviews` distinct reviewers.
* `primary_audit_v1` (revision 5): one complete primary pass by the declared `primary_reviewer` over the whole fixed pool; then a
  seeded, stratified random second review by other reviewers; then adjudication of everything uncertain, invalid or disputed.

  - Strata: the primary label (1 or 0) crossed with the hidden candidate-rank band (1-5, 6-10, 11-20, 21+, and known positives
    outside the candidate list), so positive and negative labels are covered across ranks. Each stratum contributes
    `max(audit_min_per_stratum, ceil(rate * size))` items (all of them when smaller), `rate` being `audit_rate_positive` or
    `audit_rate_negative`; within a stratum items are taken in the order of `sha256("<audit_seed>|second-review|<item_id>")`.
    Every question with a resolved primary judgment then gets at least one sampled item (a primary positive when it has one).
  - The sample is drawn only once the primary pass covers every item, so it cannot adapt to partial results, and it depends only
    on the seed, the primary labels and the hidden ranks: never on any system's test score.
  - Expansion: once every sampled item has a second review, a stratum whose disagreement rate (second review unresolved or with
    the other label) exceeds `escalate_disagreement_rate` is escalated: every item of that stratum then needs a second review.
  - Adjudication: every unresolved primary judgment (unsure, an unlocatable quote, an unusable line) and every second review that is
    unresolved or disagrees goes to an adjudicator who is neither the primary reviewer nor a second reviewer of that item.
  - Final: an independent adjudicator's resolved judgment; else a resolved primary judgment that every second review (required
    ones present) confirms. Until the escalation is decided, an item outside the sample is not final.

Reviewers never see the strata, ranks or earlier labels: the second-review batches are cut from the blinded request like any other.
Nothing here fills a gap: a missing judgment keeps the item unresolved.
"""

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from devsearch.errors import ConfigError

MODES = ("double_primary", "primary_audit_v1")
RANK_BANDS = ((1, 5), (6, 10), (11, 20), (21, None))
OUTSIDE = "outside-candidates"


@dataclass(frozen=True)
class ReviewProtocol:
    mode: str = "double_primary"
    min_primary_reviews: int = 2
    primary_reviewer: str | None = None
    audit_seed: int = 0
    audit_rate_positive: float = 0.25
    audit_rate_negative: float = 0.08
    audit_min_per_stratum: int = 30
    escalate_disagreement_rate: float = 0.10

    def describe(self) -> dict[str, Any]:
        if self.mode == "double_primary":
            return {"mode": self.mode, "min_primary_reviews": self.min_primary_reviews}
        return {"mode": self.mode, "primary_reviewer": self.primary_reviewer, "audit_seed": self.audit_seed,
                "audit_rate_positive": self.audit_rate_positive, "audit_rate_negative": self.audit_rate_negative,
                "audit_min_per_stratum": self.audit_min_per_stratum, "escalate_disagreement_rate": self.escalate_disagreement_rate,
                "rank_bands": [f"{a}-{b}" if b else f"{a}+" for a, b in RANK_BANDS] + [OUTSIDE]}


AUDIT_KEYS = ("primary_reviewer", "audit_seed", "audit_rate_positive", "audit_rate_negative", "audit_min_per_stratum", "escalate_disagreement_rate")


def protocol_from_table(review: Mapping[str, Any], where: str) -> ReviewProtocol:
    """The protocol from a `[review]` table (keys other than these are checked by the caller)."""
    mode = review.get("mode", "double_primary")
    if mode not in MODES:
        raise ConfigError(f"{where}: review.mode must be one of {', '.join(MODES)}")
    if mode == "double_primary":
        extra = [k for k in AUDIT_KEYS if k in review]
        if extra:
            raise ConfigError(f"{where}: review.{extra[0]} belongs to review.mode = \"primary_audit_v1\"")
        value = review.get("min_primary_reviews", 2)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ConfigError(f"{where}: review.min_primary_reviews must be an integer from 1 to 5")
        return ReviewProtocol(mode, value)
    if "min_primary_reviews" in review:
        raise ConfigError(f"{where}: review.min_primary_reviews belongs to review.mode = \"double_primary\"")
    missing = [k for k in AUDIT_KEYS if k not in review]
    if missing:
        raise ConfigError(f"{where}: review.mode = \"primary_audit_v1\" needs {', '.join(missing)} (declared before judging)")

    def number(key: str, low: float, high: float) -> float:
        value = review[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            raise ConfigError(f"{where}: review.{key} must be a number from {low} to {high}")
        return float(value)

    reviewer, seed, minimum = review["primary_reviewer"], review["audit_seed"], review["audit_min_per_stratum"]
    if not isinstance(reviewer, str) or not reviewer or len(reviewer) > 100 or reviewer.startswith("pilot"):
        raise ConfigError(f"{where}: review.primary_reviewer must be a reviewer id (not a pilot id)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ConfigError(f"{where}: review.audit_seed must be a non-negative integer")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= 10_000:
        raise ConfigError(f"{where}: review.audit_min_per_stratum must be an integer from 1 to 10000")
    return ReviewProtocol(mode, 1, reviewer, seed, number("audit_rate_positive", 0.01, 1.0), number("audit_rate_negative", 0.01, 1.0), minimum,
                          number("escalate_disagreement_rate", 0.0, 1.0))


def rank_band(provenance: Mapping[str, Any]) -> str:
    rank = provenance.get("candidate_rank")
    if rank is None:
        return OUTSIDE
    for low, high in RANK_BANDS:
        if rank >= low and (high is None or rank <= high):
            return f"{low}-{high}" if high else f"{low}+"
    raise ValueError(f"candidate rank {rank!r} is not positive")


def _order(seed: int, item: str) -> str:
    return hashlib.sha256(f"{seed}|second-review|{item}".encode()).hexdigest()


def evaluate(protocol: ReviewProtocol, items: Mapping[str, Mapping[str, Any]], responses: Sequence[Any]) -> dict[str, Any]:
    """Per-item state and the second-review plan under `protocol`.

    `items` maps every item id of the pool to `{"query_id", "provenance"}` (hidden; never shown to reviewers). `responses` are parsed
    responses with `item_id`, `role`, `relevance`, `resolved` and `reviewer_id`. Returns `states` (item -> state; final states start
    with "final"), `basis` (final item -> the responses its judgment rests on) and, in audit mode, `plan`.
    """
    grouped: dict[str, list] = defaultdict(list)
    for response in responses:
        grouped[response.item_id].append(response)
    states, basis = {}, {}
    if protocol.mode == "double_primary":
        for item in sorted(items):
            mine = grouped.get(item, [])
            adjudicators = [r for r in mine if r.role == "adjudicator"]
            primaries: dict[str, Any] = {}
            for response in mine:
                if response.role == "primary":
                    primaries.setdefault(response.reviewer_id, response)          # one vote per reviewer, the first recorded
            if len(adjudicators) > 1:
                states[item] = "more than one adjudicator"
            elif adjudicators:
                states[item] = "final: adjudicated" if adjudicators[0].resolved else "adjudicator left it unresolved"
                basis[item] = adjudicators
            elif len(primaries) < protocol.min_primary_reviews:
                states[item] = f"needs primary reviews ({len(primaries)} of {protocol.min_primary_reviews})"
            elif not all(r.resolved for r in primaries.values()):
                states[item] = "needs adjudication: unresolved primary"
            elif len({r.relevance for r in primaries.values()}) != 1:
                states[item] = "needs adjudication: primaries disagree"
            else:
                states[item] = "final: unanimous primaries"
                basis[item] = list(primaries.values())
        return {"states": states, "basis": {k: v for k, v in basis.items() if states[k].startswith("final")}, "plan": None}

    primary_id = protocol.primary_reviewer
    primary, secondaries, adjudicators = {}, defaultdict(dict), defaultdict(list)
    for item in items:
        for response in grouped.get(item, []):
            if response.role == "adjudicator":
                adjudicators[item].append(response)
            elif response.reviewer_id == primary_id:
                primary.setdefault(item, response)
            else:
                secondaries[item].setdefault(response.reviewer_id, response)

    def agrees(item: str) -> bool:
        return all(r.resolved and r.relevance == primary[item].relevance for r in secondaries[item].values())

    complete_primary = len(primary) == len(items)
    strata: dict[str, list[str]] = defaultdict(list)
    for item, response in primary.items():
        if response.resolved:
            strata[f"label-{response.relevance}|rank-{rank_band(items[item]['provenance'])}"].append(item)
    plan: dict[str, Any] = {"protocol": protocol.describe(), "primary_pass": {"items": len(items), "judged": len(primary),
                            "resolved": sum(1 for r in primary.values() if r.resolved), "complete": complete_primary}}
    sample: set[str] = set()
    required: set[str] = set()
    decided = False
    if complete_primary:
        table = {}
        for name, members in sorted(strata.items()):
            rate = protocol.audit_rate_positive if name.startswith("label-1") else protocol.audit_rate_negative
            size = min(len(members), max(protocol.audit_min_per_stratum, math.ceil(rate * len(members))))
            chosen = sorted(members, key=lambda i: _order(protocol.audit_seed, i))[:size]
            sample.update(chosen)
            table[name] = {"items": len(members), "rate": rate, "sampled": len(chosen)}
        by_query: dict[str, list[str]] = defaultdict(list)
        for item, response in primary.items():
            if response.resolved:
                by_query[items[item]["query_id"]].append(item)
        added = []
        for query, members in sorted(by_query.items()):
            if not sample.intersection(members):
                positives = [i for i in members if primary[i].relevance == 1]
                pick = min(positives or members, key=lambda i: _order(protocol.audit_seed, i))
                added.append(pick)
        sample.update(added)
        for name, members in strata.items():
            table[name]["sampled_with_question_coverage"] = len(sample.intersection(members))
        reviewed = {i for i in sample if secondaries[i]}
        decided = reviewed == sample
        escalated = []
        for name, members in sorted(strata.items()):
            mine = sample.intersection(members)
            done = [i for i in mine if secondaries[i]]
            disagreements = sum(1 for i in done if not agrees(i))
            table[name].update(second_reviewed=len(done), disagreements=disagreements,
                               disagreement_rate=round(disagreements / len(done), 4) if done else None)
            if decided and mine and disagreements / len(mine) > protocol.escalate_disagreement_rate:
                table[name]["escalated"] = True
                escalated.append(name)
                required.update(members)
            else:
                table[name]["escalated"] = False if decided else None
        required.update(sample)
        plan.update(strata=table, sample=sorted(sample), question_coverage_added=sorted(added), questions=len(by_query),
                    questions_in_sample=len({items[i]["query_id"] for i in sample}), escalation_decided=decided, escalated_strata=escalated,
                    required_second_reviews=len(required), second_reviews_missing=sorted(i for i in required if not secondaries[i]))
    for item in sorted(items):
        others = {primary_id} | set(secondaries[item])
        judges = [r for r in adjudicators[item] if r.resolved]          # an unsure adjudicator settles nothing; another one may
        if any(r.reviewer_id in others for r in adjudicators[item]):
            states[item] = "adjudicator is not independent"
        elif len(judges) > 1:
            states[item] = "more than one adjudicator"
        elif judges:
            states[item] = "final: adjudicated"
            basis[item] = judges
        elif adjudicators[item] and (item not in primary or not primary[item].resolved or (secondaries[item] and not agrees(item))):
            states[item] = "adjudicator left it unresolved"
        elif item not in primary:
            states[item] = "needs primary review"
        elif not primary[item].resolved:
            states[item] = "needs adjudication: unresolved primary"
        elif secondaries[item] and not agrees(item):
            states[item] = "needs adjudication: second review disagrees"
        elif not complete_primary:
            states[item] = "awaiting the complete primary pass"
        elif item in required and not secondaries[item]:
            states[item] = "needs second review"
        elif not decided:
            states[item] = "awaiting the second-review sample"
        else:
            states[item] = "final: primary confirmed by second review" if secondaries[item] else "final: primary, not sampled"
            basis[item] = [primary[item], *secondaries[item].values()]
    return {"states": states, "basis": basis, "plan": plan}


ADJUDICATION_STATES = ("needs adjudication", "adjudicator left it unresolved")      # states whose items go to the (next) adjudicator
