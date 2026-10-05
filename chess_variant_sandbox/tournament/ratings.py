"""Strength ratings from a game corpus.

Fitted with Bradley-Terry maximum likelihood rather than sequential Elo updates, for
one reason that matters here more than accuracy: sequential Elo depends on the order
games were played and on an arbitrary K-factor, so replaying the same tournament with
its games shuffled gives different ratings. This project feeds ratings straight into
the interestingness index, and an index that moves when nothing about the game changed
is worthless. A likelihood fit is order-independent and deterministic.

Ratings are reported on the familiar Elo scale, centred on zero, so the numbers read
the way anyone expects even though they were not produced the usual way.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from chess_variant_sandbox.tournament.record import GameRecord

ELO_PER_LOG10 = 400.0

# A small number of virtual drawn games against an average opponent. Without it a
# player who won every game has unbounded strength and the fit never converges --
# which is not hypothetical, since an early sweep often pairs a searching agent
# against a random one on a game where the search never slips.
PRIOR_GAMES = 2.0
# A cap, not the usual stop: the fit stops when no renormalised log-strength moved by
# more than `TOLERANCE` in a sweep, which is 1.7e-8 Elo (400 / ln 10 per unit) and takes
# tens of sweeps. Pre-registered in `docs/reference/perf-survey-2026-09.md` §5.8 (Q535).
MAX_ITERATIONS = 500
TOLERANCE = 1e-10


@dataclass
class Standing:
    name: str
    rating: float
    games: int
    wins: float
    draws: int
    as_first: int
    first_player_score: float

    @property
    def score_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0


@dataclass
class RatingTable:
    standings: list[Standing]
    head_to_head: dict[tuple[str, str], tuple[float, int]] = field(default_factory=dict)

    def rating_of(self, name: str) -> float:
        for standing in self.standings:
            if standing.name == name:
                return standing.rating
        raise KeyError(name)

    def spread(self) -> float:
        """Distance from strongest to weakest. Zero means the game ignores skill."""
        if len(self.standings) < 2:
            return 0.0
        ratings = [s.rating for s in self.standings]
        return max(ratings) - min(ratings)

    def as_dict(self) -> dict:
        return {
            "standings": [vars(s) for s in self.standings],
            "head_to_head": {f"{a}|{b}": v for (a, b), v in self.head_to_head.items()},
        }


def rate(records: Iterable[GameRecord]) -> RatingTable:
    """Fit ratings to a corpus of games."""
    wins: dict[str, float] = defaultdict(float)
    games: dict[str, int] = defaultdict(int)
    draws: dict[str, int] = defaultdict(int)
    as_first: dict[str, int] = defaultdict(int)
    first_score: dict[str, float] = defaultdict(float)
    pair_games: dict[tuple[str, str], int] = defaultdict(int)
    pair_score: dict[tuple[str, str], float] = defaultdict(float)

    for record in records:
        first, second = record.agents
        for seat, name in enumerate(record.agents):
            score = record.score_for(seat)
            wins[name] += score
            games[name] += 1
            if record.winner is None:
                draws[name] += 1
        as_first[first] += 1
        first_score[first] += record.score_for(0)
        pair_games[(first, second)] += 1
        pair_score[(first, second)] += record.score_for(0)

    names = sorted(games)
    if not names:
        return RatingTable(standings=[])

    strengths = _fit_bradley_terry(names, wins, pair_games)
    mean_log = sum(math.log10(s) for s in strengths.values()) / len(strengths)

    standings = [
        Standing(
            name=name,
            rating=round(ELO_PER_LOG10 * (math.log10(strengths[name]) - mean_log), 1),
            games=games[name],
            wins=round(wins[name], 2),
            draws=draws[name],
            as_first=as_first[name],
            first_player_score=(
                round(first_score[name] / as_first[name], 4) if as_first[name] else 0.0
            ),
        )
        for name in names
    ]
    standings.sort(key=lambda s: -s.rating)

    head_to_head = {
        pair: (round(pair_score[pair] / count, 4), count)
        for pair, count in pair_games.items()
    }
    return RatingTable(standings=standings, head_to_head=head_to_head)


def _fit_bradley_terry(
    names: list[str],
    wins: dict[str, float],
    pair_games: dict[tuple[str, str], int],
) -> dict[str, float]:
    """Minorization-maximization fit of Bradley-Terry strengths.

    Each iteration sets a player's strength to their score divided by the sum, over
    opponents, of games weighted by how likely each matchup was to be close. It
    converges monotonically and needs no learning rate.
    """
    opponents: dict[str, dict[str, int]] = {name: defaultdict(int) for name in names}
    for (first, second), count in pair_games.items():
        opponents[first][second] += count
        opponents[second][first] += count

    # The loop runs 201 times a score, so it works on positions rather than names. Every
    # float operation is the one the name-keyed loop did, on the same operands in the
    # same order: the opponents keep their dict's order, the sweep updates in place, and
    # the renormalisation sums the same logs in `names` order. `tests/test_scoring_batch.py`
    # holds the result equal to that loop's, bit for bit.
    #
    # Convergence is read on the renormalised strengths, in log space. The raw update
    # cannot be the measure: the prior's virtual opponent has strength 1 whatever the
    # scale, so every sweep moves the whole field by the same scale step and the raw
    # shift settles near 0.07, never under any tolerance (`perf-survey-2026-09.md` R3).
    index = {name: position for position, name in enumerate(names)}
    rows = [
        [(index[other], count) for other, count in opponents[name].items()] for name in names
    ]
    numerators = [wins[name] + PRIOR_GAMES / 2.0 for name in names]
    strengths = [1.0] * len(names)
    players = range(len(names))

    logs = [0.0] * len(names)

    for _ in range(MAX_ITERATIONS):
        for player in players:
            own = strengths[player]
            denominator = PRIOR_GAMES / (own + 1.0)
            for other, count in rows[player]:
                denominator += count / (own + strengths[other])
            if denominator <= 0.0:
                continue
            strengths[player] = numerators[player] / denominator

        # Renormalise to a geometric mean of one, so that successive sweeps compare.
        updated_logs = list(map(math.log, strengths))
        log_mean = sum(updated_logs) / len(strengths)
        scale = math.exp(-log_mean)
        strengths = [s * scale for s in strengths]

        shift = max(
            abs(new - log_mean - old) for new, old in zip(updated_logs, logs, strict=True)
        )
        logs = [value - log_mean for value in updated_logs]
        if shift < TOLERANCE:
            break

    return dict(zip(names, strengths, strict=True))


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expectation, for reporting and sanity checks."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_PER_LOG10))
