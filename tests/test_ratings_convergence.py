"""The Bradley-Terry fit stops when it has converged, not after 500 sweeps (Q535).

`docs/reference/perf-survey-2026-09.md` R3 found the old test never fired: it measured the
raw update, which carries the prior's constant scale step, so every fit ran the full 500
sweeps. The test now reads the renormalised log-strengths, at the tolerance pre-registered
in §5.8. The oracle here is the old loop, run its 500 sweeps: the new fit has to land within
a hair of it, and no rating rounds differently.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

import pytest

from chess_variant_sandbox.tournament import ratings

ELO_PER_LOG = 400.0 / math.log(10)


def _fixed_sweeps(
    names: list[str], wins: dict[str, float], pair_games: dict[tuple[str, str], int]
) -> tuple[dict[str, float], int]:
    """The fit as it ran before Q535: every sweep up to 500, renormalised each time."""
    opponents: dict[str, dict[str, int]] = {name: defaultdict(int) for name in names}
    for (first, second), count in pair_games.items():
        opponents[first][second] += count
        opponents[second][first] += count
    strengths = {name: 1.0 for name in names}
    for _ in range(500):
        for name in names:
            denominator = ratings.PRIOR_GAMES / (strengths[name] + 1.0)
            for other, count in opponents[name].items():
                denominator += count / (strengths[name] + strengths[other])
            strengths[name] = (wins[name] + ratings.PRIOR_GAMES / 2.0) / denominator
        log_mean = sum(math.log(s) for s in strengths.values()) / len(strengths)
        strengths = {name: s * math.exp(-log_mean) for name, s in strengths.items()}
    return strengths, 500


def _ratings(strengths: dict[str, float]) -> dict[str, float]:
    mean = sum(math.log(s) for s in strengths.values()) / len(strengths)
    return {name: ELO_PER_LOG * (math.log(s) - mean) for name, s in strengths.items()}


def _corpus(players: int, gap: float, seed: int) -> tuple[list[str], dict, dict]:
    """A ladder: each pair plays a few games, the stronger winning by `gap` per rung."""
    rng = random.Random(seed)
    names = [f"p{index}" for index in range(players)]
    wins: dict[str, float] = defaultdict(float)
    pair_games: dict[tuple[str, str], int] = defaultdict(int)
    for a, first in enumerate(names):
        for b, second in enumerate(names):
            for _ in range(rng.randrange(1, 5)):
                edge = 1.0 / (1.0 + 10 ** (gap * (a - b) / 400.0))
                roll = rng.random()
                score = 1.0 if roll < edge * 0.9 else 0.5 if roll < edge * 0.9 + 0.1 else 0.0
                wins[first] += score
                wins[second] += 1.0 - score
                pair_games[(first, second)] += 1
    return names, wins, pair_games


CASES = [(players, gap, seed) for players in (2, 4, 8) for gap in (0.0, 150.0, 600.0)
         for seed in (0, 1)]


@pytest.mark.parametrize(("players", "gap", "seed"), CASES)
def test_the_fit_lands_on_the_500_sweep_answer(players: int, gap: float, seed: int) -> None:
    names, wins, pair_games = _corpus(players, gap, seed)
    new = _ratings(ratings._fit_bradley_terry(names, wins, pair_games))
    old = _ratings(_fixed_sweeps(names, wins, pair_games)[0])
    for name in names:
        assert abs(new[name] - old[name]) < 1e-5, name
        assert round(new[name], 1) == round(old[name], 1), name


def test_a_level_field_stops_in_tens_of_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the change: the stop fires. Counted by the sweeps' `math.exp` calls."""
    names, wins, pair_games = _corpus(8, 0.0, 3)
    calls = 0
    exp = math.exp

    def counted(value: float) -> float:
        nonlocal calls
        calls += 1
        return exp(value)

    monkeypatch.setattr(ratings.math, "exp", counted)
    ratings._fit_bradley_terry(names, wins, pair_games)
    assert 0 < calls < 100


def test_the_cap_still_holds() -> None:
    """A fit that has not converged by the cap stops there, as it always did."""
    names, wins, pair_games = _corpus(8, 600.0, 0)
    assert ratings.MAX_ITERATIONS == 500
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ratings, "TOLERANCE", 0.0)
        capped = ratings._fit_bradley_terry(names, wins, pair_games)
    old = _fixed_sweeps(names, wins, pair_games)[0]
    for name in names:
        assert capped[name] == pytest.approx(old[name], rel=1e-12)


def test_the_tolerance_is_the_pre_registered_one() -> None:
    assert ratings.TOLERANCE == 1e-10
