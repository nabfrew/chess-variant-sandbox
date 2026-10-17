"""How far a score rests on repeated games: the share of different games, beside the score.

Q953a, `docs/reference/repeated-games.md` §6 (option D, pre-registered in §1). Every game a
schedule plays has its own seed and its own seat streams, but players whose draws are inert
on a position play the same game again, move for move, and a library score is the mean of
four runs of such players. This reads, from a measurement's own records:

- **the distinct share** of its games, pooled over its runs, over three sets -- every game,
  the *searching band* (both seats a searching rung) and the *top two* (both seats one of
  the two strongest rungs). A game's identity is its ordered pair of players and its whole
  move list (§1.1): the seed is not in it, so two seeds whose players play the same moves
  are one game played twice;
- **the distinct re-score**: each run scored again with each identity kept once within the
  run (the first in schedule order), its analyses and blunder trials kept with it, under the
  same weights and with no interval, and the mean of those totals over the runs whose own
  total the index averages;
- **whether the score rests on repeated games**: the distinct mean is at least
  ``REST_SD`` published sds from the index (§1.3), that is counting each game once moves the
  score outside its own interval.

**It moves no number.** No total, component, interval or verdict reads it; it is reported
beside the score, and the page says the sentence. A repeat *lowers* a score, since
`position_divergence` reads a replay as a neighbour at distance zero on purpose
(`components.py`), so the distinct mean is an upper reading, not a correction.

A placeholder run (a scorer that produced no combination) is no total and is never averaged
here, as `metrics/seeds.py` never averages one; Q953's reading script averaged it as 0.0
(`repeated-games.md` §8 has the grading).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.metrics.components import _budget
from chess_variant_sandbox.metrics.scoring import Weights
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis
from chess_variant_sandbox.tournament.record import GameRecord

# §1.3's rule, fixed before any library record was read: a score rests on repeated games
# when counting each game once moves it by at least this many of its published sds.
REST_SD = 2.0

IDENTITY = "the ordered players and the whole move list"


def identity(record: GameRecord) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Which game a record is, for counting repeats: its players in order and its moves."""
    return (tuple(record.agents), tuple(record.moves))


def distinct_share(records: Sequence[GameRecord]) -> float | None:
    """Different games over games, four places; ``None`` for no games."""
    if not records:
        return None
    return round(len({identity(record) for record in records}) / len(records), 4)


def shares(records: Sequence[GameRecord], agents: Iterable[str]) -> dict[str, float | None]:
    """The three shares over a measurement's games pooled: all, searching band, top two.

    ``agents`` is the roster: a rung searches when its budget is above zero, and the top two
    are the two largest budgets on it (`components._budget`, the ladder the seat reads).
    """
    roster = list(agents)
    searching = {agent for agent in roster if _budget(agent) > 0}
    top = set(sorted({_budget(agent) for agent in roster})[-2:])
    return {
        "all": distinct_share(records),
        "searching": distinct_share(
            [record for record in records if set(record.agents) <= searching]
        ),
        "top_two": distinct_share(
            [record for record in records if all(_budget(a) in top for a in record.agents)]
        ),
    }


def distinct_records(
    records: Sequence[GameRecord], analysis: CorpusAnalysis | None
) -> tuple[list[GameRecord], CorpusAnalysis | None]:
    """One run's records with each identity once, the first in schedule order, and the
    analyses and blunder trials of the records kept (by ``(seed, agents)``)."""
    seen: set[Any] = set()
    kept: list[GameRecord] = []
    for record in records:
        key = identity(record)
        if key not in seen:
            seen.add(key)
            kept.append(record)
    if analysis is None:
        return kept, None
    games = {(record.seed, tuple(record.agents)) for record in kept}
    return kept, CorpusAnalysis(
        games=[g for g in analysis.games if (g.seed, tuple(g.agents)) in games],
        blunders=[b for b in analysis.blunders if (b.seed, tuple(b.agents)) in games],
    )


def distinct_total(
    spec: dict[str, Any],
    records: Sequence[GameRecord],
    analysis: CorpusAnalysis | None,
    weights: Weights | str,
) -> tuple[float | None, int]:
    """One run scored with each game once, no interval: the total (or ``None`` where the
    scorer produced no combination) and how many games were kept."""
    from chess_variant_sandbox.metrics.index import score_ruleset
    from chess_variant_sandbox.tournament.ratings import rate

    kept, kept_analysis = distinct_records(records, analysis)
    if not kept:
        return None, 0
    try:
        result = score_ruleset(
            spec, kept, rate(kept), weights=weights, analysis=kept_analysis, interval=False
        )
    except Exception:
        # A report beside the score never fails the measurement it reports on: a run the
        # scorer refuses (`WeightsError`) or cannot read counted once has no total here.
        return None, len(kept)
    if not result.scored:
        return None, len(kept)
    return round(float(result.headline), 4), len(kept)


@dataclass(frozen=True)
class Run:
    """One run's records as the reading takes them."""

    seed: int
    records: Sequence[GameRecord]
    analysis: CorpusAnalysis | None
    # Whether the index averages this run's own total. The distinct mean is taken over
    # exactly these runs, so the two numbers compared rest on the same runs.
    counted: bool


def reading(
    spec: dict[str, Any],
    runs: Sequence[Run],
    agents: Iterable[str],
    weights: Weights | str,
    index: float | None,
    sd: float | None,
    minimum: int,
) -> dict[str, Any]:
    """The `repeats` block of a `canonical.json`.

    ``index`` and ``sd`` are the measurement's own, ``minimum`` the number of totals a mean
    needs (`seeds.minimum_totals`). Pure in its inputs: the runs are taken in seed order.
    """
    ordered = sorted(runs, key=lambda run: run.seed)
    pooled = [record for run in ordered for record in run.records]
    per_run: list[dict[str, Any]] = []
    totals: list[float] = []
    for run in ordered:
        total, kept = (
            distinct_total(spec, run.records, run.analysis, weights)
            if run.counted else (None, len(distinct_records(run.records, None)[0]))
        )
        per_run.append({
            "seed": run.seed, "games": len(run.records), "kept": kept, "distinct": total,
        })
        if run.counted and total is not None:
            totals.append(total)
    distinct_index = (
        round(statistics.fmean(totals), 4) if totals and len(totals) >= minimum else None
    )
    return {
        "identity": IDENTITY,
        "games": len(pooled),
        **shares(pooled, agents),
        "distinct_index": distinct_index,
        "runs": per_run,
        "rest_sd": REST_SD,
        "rests": rests(distinct_index, index, sd),
    }


def rests(distinct_index: float | None, index: float | None, sd: float | None) -> bool:
    """§1.3's rule: |distinct mean - index| >= `REST_SD` x sd, all three known."""
    if distinct_index is None or index is None or not sd:
        return False
    # Rounded to the four places both numbers are published to, so the rule reads the
    # numbers a reader sees and a difference of exactly two sds is not lost to a bit.
    return round(abs(distinct_index - index), 4) >= round(REST_SD * sd, 4)


def summary(block: Mapping[str, Any] | None) -> str:
    """The block in one line for somebody watching `measure` run, or nothing."""
    if not block:
        return ""
    found = [
        f"{label} {block[key]:.0%}" for key, label in
        (("all", "all"), ("searching", "searching"), ("top_two", "top two"))
        if block.get(key) is not None
    ]
    line = "  different games: " + ", ".join(found)
    if block.get("distinct_index") is not None:
        line += f"; counted once, {block['distinct_index']:.3f}"
    if block.get("rests"):
        line += " (rests on repeated games)"
    return line
