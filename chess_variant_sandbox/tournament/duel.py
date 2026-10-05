"""Comparing two players closely enough that a search can be run against the answer.

``head_to_head`` answers "did this player beat that one" at a standard error of about
0.072 over 48 games, which is the resolution the fit's acceptance test was written at and
is fine for a *verdict*. It is useless as an objective. The differences a coordinate
ascent has to walk through are around 0.02 to 0.05, and hill-climbing a statistic whose
noise is three times its signal optimises the noise -- which is exactly what
``runs/stage30/pilots.py`` measured when six adjacent values of one parameter came back
0.58, 0.44, 0.39, 0.71 on xiangqi.

**The pairing.** A game is a pure function of ``(ruleset, seed, agents)`` and
``play.seat_rng`` derives each seat's stream from the seed and the *seat*. So playing
``(A, B)`` at seed ``s`` and ``(B, A)`` at the same ``s`` puts the same two streams in
the same two seats and swaps only who sits there. If A and B choose the same move in
every position they reach, the two games are move for move identical, A wins one and
loses the other, and the pair contributes exactly 0.5 with **no variance whatsoever**.

**How much that is worth was measured, and it is a fraction of what the plan assumed.**
Two spellings of one player come back at exactly 0.5 with an error of exactly 0.0000, on
both rulesets and every seed -- the pairing is *exact* on a null arm, which is a real
property and the reason a null arm needs no games to interpret. Between players that
genuinely differ it buys far less, because chess-like games are chaotic in the way that
matters here: multiplying one piece's value by 1.1 changes an early move choice, and from
that ply the two games share nothing but their seed. Measured over 33 one-piece steps on
chess and xiangqi, ``identical`` is 0 to 2 pairs in 24 and the standard error falls from
the binomial 0.0722 to a median of **0.0541 to 0.0611** -- about 20%, which is a factor of
1.6 in games and not the factor of six that 0.03 at this spend would have needed.

So the honest summary is that this estimator is *better calibrated* rather than
dramatically sharper, and Stage A's stated exit was missed. What made the search
affordable anyway was the other half of the same measurement: the per-piece effects on
the ruleset where piece values matter are 0.10 to 0.36, not the 0.02 to 0.05 the plan
assumed, so 0.04 of noise resolves them at four to nine sigma. See
``docs/reference/piece-values.md``.

**The error is measured, not assumed**, and that is what survives. A win rate's standard
error is ``sqrt(0.25 / games)`` because the games are independent draws from a binomial;
the paired estimate's is not, and using the binomial formula on it would report a
resolution the estimator does not have -- badly wrong on a null arm, where the truth is
zero. So the sample's own standard deviation over the seed-pairs is what is reported, and
a ``Duel`` cannot be constructed without one. A search that accepts a step at two of
*these* standard errors is holding itself to the noise it actually met.

**Graded outcomes are implemented here and switched off by measurement.** The plan this
module comes from proposed scoring the margin -- a win by mate in 20 is not a win by mate
in 200 -- and ``_graded`` does it, blending the result with the final material lead.
``MARGIN_WEIGHT`` is the answer the calibration gave and it is zero; the sweep is in
``docs/reference/piece-values.md``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import CompiledSpec
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

# How much of a graded outcome is the material lead rather than the result.
#
# Zero, by measurement, and the measurement is not the one the idea predicts. Blending in
# the margin does lower the error -- on xiangqi's known-positive arm, from 0.0734 at
# weight 0 to 0.0578 at weight 0.5 -- but it lowers the *signal* faster: 0.156 above a
# draw becomes 0.085, and the ratio of the two falls monotonically from 2.13 to 1.47 at
# every weight tried. A term that shrinks what is being measured faster than it shrinks
# the noise is not a lower-variance estimator of the same quantity; it is a lower-variance
# estimator of a different one, and the difference is every game won from behind.
#
# Swept over {0, 0.1, 0.25, 0.5} on both a null arm and a known-positive arm at three
# seeds each. The null reads exactly 0.5 at exactly zero error at every weight, so it
# cannot separate them -- the pairing has already made that arm exact.
#
# Kept as a constant with the blend behind it rather than deleted, because the number is
# the finding and a future estimator may want to sweep it again on other rulesets.
MARGIN_WEIGHT = 0.0

# The material lead, as a share of one side's opening army, that counts as completely
# winning. Only read when ``MARGIN_WEIGHT`` is non-zero.
#
# A quarter: on chess's derived table that is a little more than a queen, which is about
# where a game stops being a contest. Squashed with ``tanh`` rather than clipped so the
# term has no flat region for a search to sit on.
MARGIN_FULL = 0.25


@dataclass(frozen=True)
class Duel:
    """What one paired comparison found, and how much of it the sample could resolve.

    ``score`` is on the same ``[0, 1]`` scale as ``head_to_head`` and means the same
    thing -- above 0.5 is the candidate winning -- so a reader does not have to learn a
    second convention. What differs is how it was arrived at, and what ``error`` is: the
    standard error of *this sample* over seed-pairs, rather than a binomial constant.
    """

    candidate: str
    reference: str
    score: float
    error: float
    # Every pair's own estimate, in seed order. Kept so a calibration can re-grade the
    # same games at another margin weight without paying for them twice, and so a
    # surprising verdict can be looked at rather than re-run.
    pairs: tuple[float, ...]
    # Pairs whose two games were move for move identical, which is to say pairs where the
    # two players never once disagreed. These contribute exactly 0.5 and exactly no
    # variance, and their share is the whole reason this estimator is affordable.
    identical: int
    margin_weight: float = MARGIN_WEIGHT

    @property
    def games(self) -> int:
        return 2 * len(self.pairs)

    @property
    def decided(self) -> bool:
        """Whether the sample can tell this apart from a draw at one standard error.

        The bar the coordinate ascent uses, and it is deliberately the *measured* error
        rather than a margin somebody chose: a fixed threshold would be a threshold
        picked while looking at which steps it accepts.
        """
        return abs(self.score - 0.5) > self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "reference": self.reference,
            "score": round(self.score, 6),
            "error": round(self.error, 6),
            "games": self.games,
            "identical": self.identical,
            "margin_weight": self.margin_weight,
        }


def duel(
    spec: dict[str, Any],
    candidate: str,
    reference: str,
    pairs: int,
    seed: int = 0,
    workers: int = 0,
    margin_weight: float = MARGIN_WEIGHT,
) -> Duel:
    """Play ``pairs`` matched seeds of ``candidate`` against ``reference`` and score them.

    Two games per pair -- the candidate at each seat on one seed -- so ``pairs`` of 24 is
    the same 48 games the acceptance test plays, at a fraction of its noise.

    Everything goes through the ordinary tournament runner, which is not a detail: the
    runner fixes every seed before any thread starts, so the answer is identical at any
    worker count, and the games then run in parallel in the core. A loop over
    ``play_game`` here would be a second scheduler to keep in step with the first.
    """
    if pairs < 1:
        raise ValueError("a duel needs at least one seed-pair")
    if candidate == reference:
        # Nothing downstream forbids it, and it would return exactly 0.5 at exactly zero
        # error -- correct, and also the one answer that cannot mean what a caller
        # wanted. The self-duel is worth playing; it just has to be spelled as two names
        # for one player, which is what a null arm is.
        raise ValueError(
            f"a duel compares two players and both sides are {candidate!r}; a null arm "
            "is two spellings of one player, not one spelling twice"
        )

    result = run_tournament(
        spec,
        TournamentConfig(
            agents=(candidate, reference),
            games_per_pairing=pairs,
            seed=seed,
            workers=workers,
            include_self_play=False,
            collect_series=margin_weight != 0.0,
            paired=True,
        ),
    )
    scale = _army_scale(Engine(spec).spec) if margin_weight else 1.0
    return grade(result.records, candidate, reference, scale, margin_weight)


def grade(
    records: list[GameRecord],
    candidate: str,
    reference: str,
    scale: float = 1.0,
    margin_weight: float = MARGIN_WEIGHT,
) -> Duel:
    """Score a paired set of records for ``candidate``. Separate so it can be re-run.

    The records are grouped by seed and each seed must carry exactly two games, one with
    the candidate at each seat. A seed with one game is a schedule that was not paired,
    and it is refused here rather than averaged: an unpaired half-sample would produce a
    number of the right shape with none of the variance reduction, which is the worst
    failure available to this module.
    """
    by_seed: dict[int, list[GameRecord]] = {}
    for record in records:
        by_seed.setdefault(record.seed, []).append(record)

    estimates: list[float] = []
    identical = 0
    for seed in sorted(by_seed):
        games = by_seed[seed]
        if len(games) != 2:
            raise ValueError(
                f"seed {seed} carries {len(games)} games and a paired comparison needs "
                "two, one with the candidate at each seat; the schedule was not paired"
            )
        seats = [_seat_of(game, candidate, reference) for game in games]
        if sorted(seats) != [0, 1]:
            raise ValueError(
                f"seed {seed} does not put {candidate!r} at each seat once; the two "
                "games of a pair are not each other's mirror"
            )
        estimates.append(
            sum(
                _graded(game, seat, scale, margin_weight)
                for game, seat in zip(games, seats, strict=True)
            )
            / 2.0
        )
        if games[0].moves == games[1].moves:
            identical += 1

    if not estimates:
        raise ValueError("a duel needs at least one seed-pair; none were played")
    return Duel(
        candidate=candidate,
        reference=reference,
        score=sum(estimates) / len(estimates),
        error=_standard_error(estimates),
        pairs=tuple(estimates),
        identical=identical,
        margin_weight=margin_weight,
    )


def _standard_error(estimates: list[float]) -> float:
    """The sample's own standard error. Exactly zero when every pair agreed.

    One pair has no spread to measure and reports zero, which is honest about the sample
    rather than about the estimator: a single pair says nothing, and ``decided`` on it
    reads whatever that pair did. Callers ask for more than one.
    """
    if len(estimates) < 2:
        return 0.0
    return statistics.stdev(estimates) / math.sqrt(len(estimates))


def _seat_of(record: GameRecord, candidate: str, reference: str) -> int:
    if record.agents[0] == candidate and record.agents[1] == reference:
        return 0
    if record.agents[0] == reference and record.agents[1] == candidate:
        return 1
    raise ValueError(
        f"a game between {record.agents[0]!r} and {record.agents[1]!r} is not part of a "
        f"duel between {candidate!r} and {reference!r}"
    )


def _graded(record: GameRecord, seat: int, scale: float, margin_weight: float) -> float:
    """The outcome, optionally blended with how far ahead the game finished.

    At the default weight of zero this is the result and nothing else. Above zero the
    blend is a *grading* of the result only while it cannot reorder one, and the bound is
    **one third**, arrived at rather than assumed: the worst win scores ``1 - w`` and the
    best draw ``0.5 + 0.5 w``, which cross at ``w = 1/3``, and the draw-against-loss pair
    crosses at the same place. At or above a third a drawn game with a big lead outscores
    a won one, which is a different quantity wearing this one's scale. Every weight the
    calibration swept is below it.
    """
    outcome = record.score_for(seat)
    if not margin_weight:
        return outcome
    return (1.0 - margin_weight) * outcome + margin_weight * _margin(record, seat, scale)


def _margin(record: GameRecord, seat: int, scale: float) -> float:
    """How far ahead ``seat`` finished, squashed onto ``[0, 1]`` with 0.5 for level.

    Read off the last entry of the per-ply series, which is the position before the final
    move -- the last one anybody had a choice in. A record with no series reads level,
    which is what an unscored game should contribute and never a zero.
    """
    if not record.material or scale <= 0.0:
        return 0.5
    lead = record.material[-1] if seat == 0 else -record.material[-1]
    return 0.5 + 0.5 * math.tanh(lead / (scale * MARGIN_FULL))


def _army_scale(spec: CompiledSpec) -> float:
    """One side's opening army, valued by the derived table. The unit a lead is in.

    Derived and never an override, for the reason ``total_material`` is: a scale that
    moved with the player being measured would make the same lead read differently
    depending on which of them was ahead.
    """
    values = Evaluator.for_spec(spec).values
    n_types = spec.n_types
    total = 0.0
    for value in spec.initial_board:
        if value == 0:
            continue
        _owner, piece_type = divmod(value - 1, n_types)
        total += abs(values[piece_type])
    return total / 2.0
