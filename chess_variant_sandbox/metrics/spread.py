"""How far a failed check's number could move by chance, from the games it read (R-S17j).

The design loop's rule for a *narrow* refusal (`docs/reference/wrong-no.md` §10.2): a failed
check is narrow when its margin past the line is at most two standard errors of the number
it read, taken from that refusal's own games. Three kinds of number, one standard error
each:

- **a rate, or a count out of n games**: binomial *at the line*, sqrt(L(1 - L)/n), so a
  rate of 1.0 does not report a spread of nothing (`binomial_se`, `count_se`);
- **a mean over games** (plies; a score with draws as halves): the sample sd of the
  per-game values over sqrt(n), so a game whose every game is the same length has no
  spread (`mean_se`);
- **the Elo gradient**, a fitted slope: the sd of the slope over the resamples the index's
  interval already draws (`bootstrap_index` keeps `skill_gradient` among its components of
  interest, so `IndexResult.interval["component_sd"]` holds it).

This module reads persisted records and the interval a score already carries; it plays
nothing. **It reaches no score**: only the loop's refusals import it, and it is left out of
every fingerprint's code reading as the glossary is (`measurement.LOOP_ONLY`), so adding it
or changing it stales no stored measurement. `tests/test_measurement.py` pins that no
fingerprinted source imports it.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from chess_variant_sandbox.metrics.components import (
    WORST_HALF_RATES,
    WORST_HALF_SUFFIX,
    _budget,
    _is_weak,
    _outcome_rates,
    _split_rung,
)
from chess_variant_sandbox.tournament.ratings import RatingTable
from chess_variant_sandbox.tournament.record import GameRecord


def binomial_se(line: float, n: int) -> float | None:
    """The standard error of a rate over ``n`` games, at the line ``line`` (a rate)."""
    if n <= 0:
        return None
    rate = min(max(float(line), 0.0), 1.0)
    return math.sqrt(rate * (1.0 - rate) / n)


def count_se(line: float, n: int) -> float | None:
    """The standard error of a count out of ``n`` games, at the line ``line`` (a count):
    the binomial at the line as a rate, in games."""
    if n <= 0:
        return None
    se = binomial_se(float(line) / n, n)
    return None if se is None else se * n


def mean_se(values: Sequence[float]) -> float | None:
    """The standard error of a mean: the sample sd of the values over sqrt(n). Zero when
    every value is the same, and None under two values, where there is no spread to read."""
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


# The scored checks this reads, by the kind of number each is (`wrong-no.md` §10.2's table).
# Every other check, the seat gates and `mean_branching` among them, has no spread here and
# is never narrow.
WORST_HALF_RATE_CHECKS = {
    "stall_rate" + WORST_HALF_SUFFIX: "stall_rate",
    "decisive_rate" + WORST_HALF_SUFFIX: "decisive_rate",
    "quick_kill_rate" + WORST_HALF_SUFFIX: "quick_kill_rate",
}
WORST_HALF_MEAN_CHECKS = {"mean_plies" + WORST_HALF_SUFFIX: "mean_plies"}
SEARCH_ADVANTAGE = "search_advantage"
SKILL_GRADIENT = "skill_gradient"
FULL_CHECKS = frozenset(
    {*WORST_HALF_RATE_CHECKS, *WORST_HALF_MEAN_CHECKS, SEARCH_ADVANTAGE, SKILL_GRADIENT}
)


def _graded(records: Sequence[GameRecord]) -> list[GameRecord]:
    """The games the outcome gates read: those with no random mover in either seat, or all
    of them where there are none (`compute_components`)."""
    strong = [r for r in records if not _is_weak(r.agents[0]) and not _is_weak(r.agents[1])]
    return strong or list(records)


def worst_half_games(records: Sequence[GameRecord], rate: str) -> list[GameRecord]:
    """The games of the half of the ladder whose value the gate on ``rate`` reported, chosen
    as `components._worst_half_rates` chooses it: the half reading worse in the rate's own
    direction, or every graded game where the ladder cannot be halved."""
    graded = _graded(records)
    split = _split_rung(graded)
    halves: dict[str, list[GameRecord]] = {}
    if split is not None:
        weak = [r for r in graded if min(_budget(a) for a in r.agents) < split]
        strong = [r for r in graded if min(_budget(a) for a in r.agents) >= split]
        halves = {name: half for name, half in (("weak", weak), ("strong", strong)) if half}
    if len(halves) < 2:
        halves = {"pooled": graded}
    measured = {name: _outcome_rates(half) for name, half in halves.items()}
    pick = max if WORST_HALF_RATES[rate] == "high" else min
    worst = pick(measured, key=lambda name: measured[name][rate])
    return halves[worst]


def _search_advantage_scores(ratings: RatingTable, records: Sequence[GameRecord]) -> list[float]:
    """Each game's score for the strongest searching player against a random mover, the
    games `components._search_advantage` reads, with a draw as a half."""
    randoms = {s.name for s in ratings.standings if _is_weak(s.name)}
    best = next((s.name for s in ratings.standings if not _is_weak(s.name)), None)
    if not randoms or best is None:
        return []
    scores: list[float] = []
    for record in records:
        first, second = record.agents
        if first == best and second in randoms:
            scores.append(record.score_for(0))
        elif second == best and first in randoms:
            scores.append(record.score_for(1))
    return scores


def check_spread(
    check: str,
    line: float | None,
    records: Sequence[GameRecord],
    ratings: RatingTable,
    interval: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """``n`` (the games the check read) and ``se`` (its standard error) for one failed check
    of the full roster, or None for a check that has none (§10.2: never narrow)."""
    if check in WORST_HALF_RATE_CHECKS:
        if line is None:
            return None
        half = worst_half_games(records, WORST_HALF_RATE_CHECKS[check])
        return {"n": len(half), "se": binomial_se(line, len(half))}
    if check in WORST_HALF_MEAN_CHECKS:
        half = worst_half_games(records, WORST_HALF_MEAN_CHECKS[check])
        return {"n": len(half), "se": mean_se([float(r.plies) for r in half])}
    if check == SEARCH_ADVANTAGE:
        scores = _search_advantage_scores(ratings, records)
        return {"n": len(scores), "se": mean_se(scores)}
    if check == SKILL_GRADIENT:
        sd = ((interval or {}).get("component_sd") or {}).get(SKILL_GRADIENT)
        return {"n": len(records), "se": None if sd is None else float(sd)}
    return None


__all__ = [
    "FULL_CHECKS",
    "binomial_se",
    "check_spread",
    "count_se",
    "mean_se",
    "worst_half_games",
]
