"""
services/disambiguation.py — the ONLY place "should we ask the student
which one they meant?" is decided. Handbook, FAQ, and date/department
lookups all call into this instead of each having their own ad-hoc
gap-threshold check, so the behavior is consistent and only needs tuning
in one place.

WHY THE OLD LOGIC OVER-TRIGGERED
---------------------------------
The old code asked "is the gap between candidate #1 and #2 small?" and
nothing else. That fires constantly, because:
  - Two candidates can be close together while BOTH are mediocre matches
    (e.g. 0.31 vs 0.29) — neither is a good answer, but the gap looks
    "tied" so it asked anyway instead of just saying "I don't know".
  - A candidate can be a clearly, obviously correct match (0.91) with a
    weak, barely-related runner-up (0.85) purely because of shared common
    words — still "close" by raw gap, but not actually ambiguous.

This module fixes both by requiring ALL of:
  1. The runner-up must itself be a plausible answer (min_contender_score) —
     a weak #2 doesn't count as a real alternative.
  2. The leader must NOT be decisively confident (decisive_score) — if the
     top match is clearly good, we trust it and skip the picker even if
     something else is nearby.
  3. The gap between #1 and #2 must be genuinely small (max_gap).
  4. Optionally, only offer contenders whose score is within `max_gap` of
     the leader — a #3/#4 that's technically "close to #2" but far from #1
     doesn't get listed either.
"""
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DisambiguationPolicy:
    decisive_score: float = 0.75      # >= this on #1 -> never disambiguate
    min_contender_score: float = 0.45  # #2 must clear this to count
    max_gap: float = 0.07              # #1 - #2 must be <= this
    max_options: int = 4               # cap on picker size


DEFAULT_POLICY = DisambiguationPolicy()
# Stricter: once the top handbook chunk is clearly ahead, trust it instead
# of forcing a picker. Over-triggering on vision/mission/history was a
# major source of remaining failures.
HANDBOOK_POLICY = DisambiguationPolicy(decisive_score=0.58, min_contender_score=0.50, max_gap=0.12)
FAQ_POLICY = DisambiguationPolicy(decisive_score=0.85, min_contender_score=0.58, max_gap=0.06)
DATE_POLICY = DisambiguationPolicy(decisive_score=0.65, min_contender_score=0.50, max_gap=0.10)

# "Soft" clarification tier -- used ONLY at the point the bot is about to
# give up entirely and say "I don't have a confirmed answer for that."
# Normal disambiguation (above) only fires when there's a genuinely close
# race between good candidates; it deliberately does NOT fire on two
# mediocre, spread-out candidates, because asking "which did you mean?"
# between two bad guesses is worse than just answering the better one.
# But when NEITHER guess is good enough to answer confidently, that
# same logic means both get silently dropped and the student gets a flat
# non-answer -- even though a person skimming the same 2-3 candidates
# would often recognize the topic and just ask "did you mean X or Y?"
# instead of shrugging. decisive_score is set above 1.0 (unreachable) so
# select_contenders() never short-circuits into "just trust the leader" --
# by the time this policy is consulted, nothing WAS trusted, so there is
# no leader to fall back on. min_contender_score is deliberately much
# lower than the normal policies' (a candidate only needs to be
# vaguely-plausible, not close-to-confident, to be worth offering as an
# option) and max_gap is wide (candidates don't need to be near-tied,
# just all be in the "maybe" zone). Only ever consulted with >=2
# contenders required by the caller -- a single weak guess still isn't
# worth interrupting the conversation for.
FAQ_SOFT_POLICY = DisambiguationPolicy(
    decisive_score=1.01, min_contender_score=0.32, max_gap=0.28, max_options=3,
)
HANDBOOK_SOFT_POLICY = DisambiguationPolicy(
    decisive_score=1.01, min_contender_score=0.22, max_gap=0.30, max_options=3,
)


def with_overrides(policy: DisambiguationPolicy, **overrides) -> DisambiguationPolicy:
    return replace(policy, **overrides)


def should_disambiguate(scores: Sequence[float], policy: DisambiguationPolicy = DEFAULT_POLICY) -> bool:
    """`scores` must be pre-sorted descending and normalized to roughly
    0..1. Returns True only when there are genuinely two-or-more plausible,
    similarly-confident candidates worth asking the student about."""
    if len(scores) < 2:
        return False
    top, second = scores[0], scores[1]
    if top <= 0:
        return False
    if top >= policy.decisive_score:
        return False
    if second < policy.min_contender_score:
        return False
    if (top - second) > policy.max_gap:
        return False
    return True


def select_contenders(items: Sequence[T], scores: Sequence[float],
                       policy: DisambiguationPolicy = DEFAULT_POLICY) -> List[T]:
    """Given items and their (pre-sorted descending) scores, return the
    subset worth showing in the disambiguation picker: the leader plus any
    runner-up within `max_gap` of the leader AND above `min_contender_score`,
    capped at `max_options`."""
    if not items:
        return []
    top_score = scores[0]
    out = []
    for item, score in zip(items, scores):
        if score < policy.min_contender_score:
            continue
        if (top_score - score) > policy.max_gap:
            continue
        out.append(item)
        if len(out) >= policy.max_options:
            break
    return out


def decisive_answer(scores: Sequence[float], policy: DisambiguationPolicy = DEFAULT_POLICY) -> bool:
    """True when the top score alone is confident enough to answer
    directly, with no disambiguation and no need to check a runner-up at
    all. Useful as a fast-path before even computing `should_disambiguate`."""
    return bool(scores) and scores[0] >= policy.decisive_score