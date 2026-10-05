"""The judge's ordering, fitted from pairwise comparisons.

Comparisons accumulate across a run and are stored append-only next to the lineage, for
the same reasons the lineage is: a run can be stopped and resumed, two runs can be
concatenated, and the ordering can be refitted later without asking the model anything
again.

Fitted with the same Bradley-Terry maximum likelihood the agent ladder uses. That is not
an economy, it is the point -- a preference ordering and a strength ladder are the same
estimation problem, and having one implementation means the ordering cannot quietly
acquire different behaviour on ties or on a player who has won everything.

**An ordering has an interval too, and this one is thin.** A handful of comparisons per
candidate cannot support a confident ranking, and a ranker that is confidently arbitrary
is worse than no ranker at all, because it looks like signal. So the table resamples its
own comparisons, reports a standard deviation per candidate, and ``beats`` requires both
a minimum number of comparisons and non-overlapping intervals before it will say one
candidate is preferred to another.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chess_variant_sandbox.llm.compare import Comparison
from chess_variant_sandbox.tournament.ratings import ELO_PER_LOG10, _fit_bradley_terry

PREFERENCE_FILE = "preferences.jsonl"

# Below this, a candidate's rating is one or two opinions wearing a number. It is still
# fitted and still reported -- it is simply not allowed to decide anything.
MIN_COMPARISONS = 3

# Enough resamples to put two decimal places on a standard deviation, cheap because a
# fit over a few dozen comparisons is microseconds.
RESAMPLES = 200


@dataclass
class PreferenceTable:
    """A fitted ordering over rulesets, with the interval that says how much to trust it."""

    ratings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    sd: dict[str, float] = field(default_factory=dict)
    # How often the two orders of a comparison disagreed. The honest headline number for
    # a pairwise judge: at 0.5 it is answering by position rather than by preference.
    flip_rate: float = 0.0
    comparisons: int = 0

    def rating(self, digest: str) -> float:
        return self.ratings.get(digest, 0.0)

    def confident(self, digest: str) -> bool:
        return self.counts.get(digest, 0) >= MIN_COMPARISONS

    def beats(self, digest: str, other: str) -> bool:
        """Whether one candidate is preferred to another by more than the noise in both.

        Deliberately the same shape of test as `metrics.bootstrap.clears`, and for the
        same reason: an ordering whose gaps are inside its own error bars is an ordering
        that will be climbed rather than followed.
        """
        if not (self.confident(digest) and self.confident(other)):
            return False
        mine, theirs = self.rating(digest), self.rating(other)
        return mine - 1.96 * self.sd.get(digest, 0.0) > theirs + 1.96 * self.sd.get(other, 0.0)

    def ranked(self, digests: list[str]) -> list[str]:
        """``digests`` best first. Unrated candidates sort last, not first."""
        return sorted(
            digests,
            key=lambda digest: (-self.rating(digest) if self.confident(digest) else 1e9),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ratings": {k: round(v, 1) for k, v in self.ratings.items()},
            "counts": dict(self.counts),
            "sd": {k: round(v, 1) for k, v in self.sd.items()},
            "flip_rate": round(self.flip_rate, 3),
            "comparisons": self.comparisons,
        }


class PreferenceLog:
    """Every comparison this run has paid for, and the ordering they add up to."""

    def __init__(self, directory: str | Path) -> None:
        self.path = Path(directory) / PREFERENCE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[Comparison] = list(self._load())

    def _load(self) -> list[Comparison]:
        if not self.path.exists():
            return []
        found: list[Comparison] = []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    found.append(
                        Comparison(
                            first=data["first"],
                            second=data["second"],
                            winner=data["winner"],
                            agreed=bool(data.get("agreed", True)),
                            reasons=tuple(data.get("reasons", ())),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return found

    @property
    def comparisons(self) -> list[Comparison]:
        return list(self._entries)

    def seen(self, first: str, second: str) -> bool:
        """Whether this pair has already been compared, in either order."""
        pair = {first, second}
        return any({c.first, c.second} == pair for c in self._entries)

    def record(self, comparison: Comparison) -> Comparison:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(comparison.as_dict(), separators=(",", ":")) + "\n")
        self._entries.append(comparison)
        return comparison

    def table(self, seed: int = 0) -> PreferenceTable:
        return fit(self._entries, seed=seed)


def fit(comparisons: list[Comparison], seed: int = 0) -> PreferenceTable:
    """Fit an ordering, and resample it to find out how much of it is real."""
    if not comparisons:
        return PreferenceTable()

    ratings, counts = _fit(comparisons)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(RESAMPLES):
        resampled = [rng.choice(comparisons) for _ in comparisons]
        drawn, _ = _fit(resampled)
        for name, value in drawn.items():
            draws[name].append(value)

    sd = {
        name: (
            math.sqrt(sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values))
            if len(values) > 1
            else 0.0
        )
        for name, values in draws.items()
    }
    flips = sum(1 for c in comparisons if not c.agreed)
    return PreferenceTable(
        ratings=ratings,
        counts=counts,
        sd=sd,
        flip_rate=flips / len(comparisons),
        comparisons=len(comparisons),
    )


def _fit(comparisons: list[Comparison]) -> tuple[dict[str, float], dict[str, int]]:
    wins: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    pair_games: dict[tuple[str, str], int] = defaultdict(int)

    for comparison in comparisons:
        first, second = comparison.first, comparison.second
        score = comparison.score_for_first
        wins[first] += score
        wins[second] += 1.0 - score
        counts[first] += 1
        counts[second] += 1
        pair_games[(first, second)] += 1

    names = sorted(counts)
    if not names:
        return {}, {}

    strengths = _fit_bradley_terry(names, wins, pair_games)
    mean_log = sum(math.log10(s) for s in strengths.values()) / len(strengths)
    # Reported on the same scale as the agent ladder, so a reader who knows what 100
    # points means there knows what it means here.
    ratings = {
        name: ELO_PER_LOG10 * (math.log10(strengths[name]) - mean_log) for name in names
    }
    return ratings, dict(counts)
