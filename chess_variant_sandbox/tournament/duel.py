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

**A steering duel may stop early; a stamped one never does.** ``steer`` is a
group-sequential rule with looks at 8, 16 and 24 seed-pairs (``docs/reference/compute-economy.md``
§4.3), pre-registered in ``docs/reference/learned-evaluator.md`` §9 before anything used
it. It is a different test from the fixed one, so it changes which comparisons pass near
the bar, and it is for duels whose result only chooses the next experiment. Nothing that
writes a stamp, an acceptance or a score calls it; ``tests/test_duel_steer.py`` holds that.

**A stamped duel may be curtailed, exactly.** ``curtailed_duel`` (``docs/reference/compute-economy.md``
§4.2) plays the fixed rule's seed-pairs in schedule order, a batch of the worker count at a
time, and stops at the first prefix after which every possible result of the unplayed pairs
gives the same verdict (``settled``). The verdict is then the full duel's by construction,
not by a bound on an error. The stopping point is read pair by pair along the schedule, never
from which games came back first, so it is the same at any worker count; games past it are
discarded. What moves is the score, which is the prefix's, stored with "decided after k of n".
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from chess_variant_sandbox.agents.evaluation import Evaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import CompiledSpec
from chess_variant_sandbox.tournament.checkpoint import run_tournament
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig

# The steering rule's design, pre-registered in `docs/reference/learned-evaluator.md` §9.
# Changing any of these is a new rule, re-registered there with its error rates first.
#
# The looks, in seed-pairs: reference/compute-economy.md §4.3's 8, 16 and 24.
STEER_LOOKS = (8, 16, 24)
# Stop for a side when z = (W - L) / sqrt(W + L) reaches this, W and L the pairs each side
# won outright and ties (a pair at exactly 0.5) left out. Pocock-shaped: one bound at every
# look, the smallest bound to one decimal whose worst one-sided error over every tie
# pattern is under `STEER_ALPHA` (2.4 reaches 0.0313; 2.5's worst is 0.0210, with 7, 10
# and 16 decisive pairs at the three looks).
STEER_BOUND = 2.5
# Stop early as *too close* when |z| is under these at the first two looks: a duel that 24 pairs
# would very likely leave unresolved, not a finding that the two are equal. Non-binding:
# the error rates hold whether or not a caller obeys it.
STEER_FUTILITY = (0.5, 1.0)
# The ceiling on each side's false call, under any tie rate, which a test proves exactly.
STEER_ALPHA = 0.025

CANDIDATE = "candidate"
REFERENCE = "reference"
TOO_CLOSE = "too close"
INCONCLUSIVE = "inconclusive"
CONTINUE = "continue"

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
    score, error = summary(estimates)
    return Duel(
        candidate=candidate,
        reference=reference,
        score=score,
        error=error,
        pairs=tuple(estimates),
        identical=identical,
        margin_weight=margin_weight,
    )


def summary(estimates: Sequence[float]) -> tuple[float, float]:
    """A duel's score and error from its seed-pairs: what ``grade`` stores, and what every
    verdict over a duel reads, so a curtailed verdict and a full one are one computation."""
    return sum(estimates) / len(estimates), _standard_error(list(estimates))


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


@dataclass(frozen=True)
class Steer:
    """Where a steering duel stands: the call, and the look it was made at.

    ``call`` is one of ``CANDIDATE`` or ``REFERENCE`` (that side won, at the registered
    error), ``TOO_CLOSE`` (stopped early, too close for more pairs to be worth it),
    ``INCONCLUSIVE`` (all 24 pairs and no side reached the bound) or ``CONTINUE`` (play to
    ``next_look`` and ask again). A steer is a reading for choosing the next step and has
    no score to store, by design.
    """

    call: str
    pairs: int
    wins: int
    losses: int
    z: float
    next_look: int | None


def steer(pairs: Sequence[float]) -> Steer:
    """Read a steering duel's seed-pairs at the registered looks, in schedule order.

    ``pairs`` is ``Duel.pairs`` (or ``SteeringDuel.estimates``), each pair's estimate in
    seed order. The rule reads the prefix at each look in turn and returns the first call;
    pairs past the deciding look are ignored, so the call never depends on how many extra
    games happened to come back. Fewer pairs than the next look is ``CONTINUE``.
    """
    if len(pairs) > STEER_LOOKS[-1]:
        raise ValueError(
            f"a steering duel reads at most {STEER_LOOKS[-1]} seed-pairs and was given "
            f"{len(pairs)}; more is a different rule, with different error rates"
        )
    wins = losses = 0
    read = 0
    z = 0.0
    for index, look in enumerate(STEER_LOOKS):
        if len(pairs) < look:
            return Steer(CONTINUE, read, wins, losses, z, look)
        for estimate in pairs[read:look]:
            if estimate > 0.5:
                wins += 1
            elif estimate < 0.5:
                losses += 1
        read = look
        z = _steer_z(wins, losses)
        call = steer_call(index, wins, losses)
        if call != CONTINUE:
            return Steer(call, look, wins, losses, z, None)
    raise AssertionError("the last look always makes a call")


def steer_call(look: int, wins: int, losses: int, futility: bool = True) -> str:
    """The call at look ``look`` (0, 1 or 2) after ``wins`` and ``losses`` outright pairs.

    The rule's one definition: ``steer`` walks the looks through it, and
    ``steer_characteristics`` integrates over it. ``futility=False`` is the rule played on
    regardless of the too-close stop.
    """
    z = _steer_z(wins, losses)
    if z >= STEER_BOUND:
        return CANDIDATE
    if z <= -STEER_BOUND:
        return REFERENCE
    if futility and look < len(STEER_FUTILITY) and abs(z) < STEER_FUTILITY[look]:
        return TOO_CLOSE
    if look == len(STEER_LOOKS) - 1:
        return INCONCLUSIVE
    return CONTINUE


def _steer_z(wins: int, losses: int) -> float:
    decisive = wins + losses
    return (wins - losses) / math.sqrt(decisive) if decisive else 0.0


@dataclass(frozen=True)
class SteerCharacteristics:
    """The exact probability of each call, and the expected pairs played, at one truth."""

    candidate: float
    reference: float
    too_close: float
    inconclusive: float
    expected_pairs: float


def steer_characteristics(win: float, loss: float, futility: bool = True) -> SteerCharacteristics:
    """``steer``'s operating characteristics when each pair independently goes to the
    candidate with probability ``win``, to the reference with ``loss``, and ties otherwise.

    Exact, by walking the distribution of (wins, losses) through the looks; no sampling.
    ``futility=False`` reads the rule with the futility stop ignored, which is the rule a
    caller who plays on regardless gets, and the one whose error is the registered bound.
    """
    if win < 0 or loss < 0 or win + loss > 1:
        raise ValueError(f"win {win} and loss {loss} are not the chances of one pair")
    tie = 1.0 - win - loss
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}
    played = 0
    out = {CANDIDATE: 0.0, REFERENCE: 0.0, TOO_CLOSE: 0.0, INCONCLUSIVE: 0.0}
    expected = 0.0
    for index, look in enumerate(STEER_LOOKS):
        while played < look:
            grown: dict[tuple[int, int], float] = {}
            for (wins, losses), chance in states.items():
                for key, step in (((wins + 1, losses), win), ((wins, losses + 1), loss),
                                  ((wins, losses), tie)):
                    if step:
                        grown[key] = grown.get(key, 0.0) + chance * step
            states = grown
            played += 1
        going: dict[tuple[int, int], float] = {}
        for (wins, losses), chance in states.items():
            call = steer_call(index, wins, losses, futility)
            if call == CONTINUE:
                going[(wins, losses)] = chance
                continue
            out[call] += chance
            expected += chance * look
        states = going
    return SteerCharacteristics(
        candidate=out[CANDIDATE],
        reference=out[REFERENCE],
        too_close=out[TOO_CLOSE],
        inconclusive=out[INCONCLUSIVE],
        expected_pairs=expected,
    )


@dataclass(frozen=True)
class SteeringDuel:
    """A steering duel as played: the call, and every pair's estimate up to it."""

    candidate: str
    reference: str
    steer: Steer
    estimates: tuple[float, ...]
    identical: int

    @property
    def games(self) -> int:
        return 2 * len(self.estimates)


def steering_duel(
    spec: dict[str, Any],
    candidate: str,
    reference: str,
    seed: int = 0,
    workers: int = 0,
    margin_weight: float = MARGIN_WEIGHT,
) -> SteeringDuel:
    """Play look by look until ``steer`` makes a call. For steering only, never a verdict.

    Each look plays the next block of seed-pairs through ``duel`` with its seed offset by
    the pairs already played. Two players get one block of seeds starting at ``seed``, so
    the pairs are those of one long duel's prefix: the same games at any worker count,
    and the same call as reading a full 24-pair duel with ``steer``.
    """
    estimates: list[float] = []
    identical = 0
    while True:
        reading = steer(estimates)
        if reading.call != CONTINUE:
            return SteeringDuel(candidate, reference, reading, tuple(estimates), identical)
        assert reading.next_look is not None
        block = duel(
            spec,
            candidate,
            reference,
            pairs=reading.next_look - len(estimates),
            seed=seed + len(estimates),
            workers=workers,
            margin_weight=margin_weight,
        )
        estimates.extend(block.pairs)
        identical += block.identical


# ------------------------------------------------------------------ exact curtailment

# Single games fit the same reading unchanged (Q800): a game's 0, 1/2 or 1 is a whole
# number of quarters, and a verdict over their sum alone meets both conditions below, so
# `settled` over games is exact for it (`fitting.acceptance`, the fit's `score > 0.5`).
Verdict = Callable[[Sequence[float]], bool]
"""A stamped duel's verdict, read off all of its seed-pairs.

``settled`` asks it only of complete duels, and relies on two properties every verdict here
has: it reads the pairs as a multiset, through their count, sum and sum of squares (as
``summary`` does, whose sum of quarters and exact variance are order-free), and at a fixed
sum it is monotone in the sum of squares (a larger spread is a larger error, never a
better score)."""

PlayPairs = Callable[[int, int], list[GameRecord]]
"""``(first seed, pairs)`` to the records of that block of a paired schedule's seeds."""


def floor_verdict(floor: float) -> Verdict:
    """``score >= floor``: the strength check's two duels."""

    def passes(estimates: Sequence[float]) -> bool:
        return summary(estimates)[0] >= floor

    return passes


def settled(prefix: Sequence[float], planned: int, verdict: Verdict) -> bool | None:
    """The verdict every completion of ``prefix`` to ``planned`` pairs gives, else ``None``.

    Exact. A pair's estimate is one of {0, 1/4, 1/2, 3/4, 1} (two games of 0, 1/2 or 1,
    ``MARGIN_WEIGHT`` zero), so the unplayed pairs reach a total ``t`` quarters in many ways
    but, at each ``t``, their sum of squares lies between the even spread's and the heaped
    one's (as many whole points as fit). The verdict is monotone in that sum at a fixed
    total, so the two extremes at every ``t`` agree with everything between them, and the
    question is ``2 x (4r + 1)`` verdicts for ``r`` unplayed pairs. The two ends, all lost
    and all won, are asked first: an open duel nearly always disagrees there at once.
    """
    played = len(prefix)
    if played > planned:
        raise ValueError(f"{played} pairs played of a duel of {planned}")
    for estimate in prefix:
        quarters = estimate * 4
        if quarters != int(quarters) or not 0 <= quarters <= 4:
            raise ValueError(
                f"a pair scored {estimate}, not a whole number of quarters: curtailment "
                "reads ungraded duels only (MARGIN_WEIGHT 0)"
            )
    rest = planned - played
    head = list(prefix)
    if not rest:
        return verdict(head)
    first = verdict(head + [0.0] * rest)
    for total in (4 * rest, *range(1, 4 * rest)):
        for completion in (_spread(total, rest), _heaped(total, rest)):
            if verdict(head + completion) != first:
                return None
    return first


def _spread(total: int, rest: int) -> list[float]:
    """``total`` quarters over ``rest`` pairs as evenly as they go: the least spread."""
    each, over = divmod(total, rest)
    return [(each + 1) / 4] * over + [each / 4] * (rest - over)


def _heaped(total: int, rest: int) -> list[float]:
    """``total`` quarters as whole points first, the remainder in one pair: the most."""
    whole, part = divmod(total, 4)
    tail = [part / 4] if part else []
    return [1.0] * whole + tail + [0.0] * (rest - whole - len(tail))


@dataclass(frozen=True)
class Curtailed:
    """A stamped duel played to its decision: the full duel's verdict, the prefix's grade.

    ``found`` grades only the pairs up to the deciding one, so its score and error are the
    prefix's; ``planned`` is the fixed rule's size. ``decided_after`` of ``planned`` is
    what gets stored beside the verdict.
    """

    found: Duel
    verdict: bool
    planned: int
    # The kept games, every game of the pairs up to the deciding one, by seed and then with
    # the candidate's first seat first, whatever the batches were. Out of equality: a
    # record carries its wall time.
    records: tuple[GameRecord, ...] = field(default=(), compare=False)

    @property
    def decided_after(self) -> int:
        return len(self.found.pairs)

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.found.as_dict(),
            "decided_after": self.decided_after,
            "planned_pairs": self.planned,
        }


def curtailed_duel(
    spec: dict[str, Any],
    candidate: str,
    reference: str,
    pairs: int,
    verdict: Verdict,
    seed: int = 0,
    workers: int = 0,
    play: PlayPairs | None = None,
) -> Curtailed:
    """``duel``'s ``pairs`` seed-pairs from ``seed``, stopped once ``verdict`` is settled.

    A batch is ``workers`` (resolved as the runner resolves it) seed-pairs, the next block
    of the one paired schedule, played as ``duel`` plays it, or by ``play``. After each
    batch the pairs are read one at a time in seed order and the first prefix ``settled``
    answers is the decision; the rest of that batch is discarded. So the stopping point,
    the kept games and the verdict are functions of the schedule alone, and the verdict is
    ``verdict`` of the full duel whatever the unplayed games would have been.
    """
    if pairs < 1:
        raise ValueError("a duel needs at least one seed-pair")
    config = TournamentConfig(
        agents=(candidate, reference),
        games_per_pairing=1,
        workers=workers,
        include_self_play=False,
        paired=True,
    )
    batch = config.resolved_workers()

    def standard(first: int, count: int) -> list[GameRecord]:
        return duel_records(spec, candidate, reference, count, first, workers)

    play = play or standard
    estimates: list[float] = []
    kept: list[GameRecord] = []
    while True:
        first = seed + len(estimates)
        count = min(batch, pairs - len(estimates))
        records = play(first, count)
        block = grade(records, candidate, reference)
        if len(block.pairs) != count:
            raise ValueError(f"a batch of {count} seed-pairs came back with {len(block.pairs)}")
        for index, estimate in enumerate(block.pairs):
            estimates.append(estimate)
            call = settled(estimates, pairs, verdict)
            if call is None:
                continue
            cut = first + index
            kept.extend(record for record in records if record.seed <= cut)
            kept.sort(key=lambda record: (record.seed, record.agents[0] != candidate))
            return Curtailed(grade(kept, candidate, reference), call, pairs, tuple(kept))
        kept.extend(records)


def duel_records(
    spec: dict[str, Any],
    candidate: str,
    reference: str,
    pairs: int,
    seed: int = 0,
    workers: int = 0,
) -> list[GameRecord]:
    """The games of ``duel``'s paired schedule, ungraded: seeds ``seed`` to ``seed + pairs``.

    A paired schedule gives the one pairing one block of seeds from the config's, so the
    block from ``seed + k`` is the same games as the tail of the block from ``seed``.
    """
    result = run_tournament(
        spec,
        TournamentConfig(
            agents=(candidate, reference),
            games_per_pairing=pairs,
            seed=seed,
            workers=workers,
            include_self_play=False,
            collect_series=False,
            paired=True,
        ),
    )
    return list(result.records)
