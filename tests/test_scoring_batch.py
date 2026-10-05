"""The scoring batch (Q497) is invisible: every score and rating is the old path's, bit for bit.

Two changes, both to how much interpreter work a score does and neither to what it
computes (`docs/reference/perf-survey-2026-09.md` R2, R3):

* the Bradley-Terry fit works on positions and lists rather than names and dicts, with
  every float operation kept in its order;
* the bootstrap reads each game's own numbers once (`components.GameReadings`) rather
  than once per resample, and `_budget` and `_is_weak` are cached.

The old path is kept here as the oracle: `_old_fit` is the fit as it stood, and
`_the_old_path` puts it back together with a reading that memoises nothing and the two
name readers uncached. Equality is `==` on floats and on whole results, never approx.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from chess_variant_sandbox.metrics import bootstrap, components
from chess_variant_sandbox.metrics.index import score_ruleset
from chess_variant_sandbox.metrics.scoring import load_weights
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament import ratings
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, analyse_corpus
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.tournament.runner import TournamentConfig, run_tournament

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
LADDER = ("random", "greedy", "minimax:nodes=100", "minimax:nodes=400")


def _old_fit(
    names: list[str],
    wins: dict[str, float],
    pair_games: dict[tuple[str, str], int],
) -> dict[str, float]:
    """`ratings._fit_bradley_terry` as it stood before Q497, keyed on names.

    Its stopping rule is Q535's (the renormalised log-strengths moved by less than the
    tolerance), so this still pins what Q497 claimed: positions for names changes no bit.
    The fixed 500 sweeps it replaced are `tests/test_ratings_convergence.py`'s oracle.
    """
    opponents: dict[str, dict[str, int]] = {name: defaultdict(int) for name in names}
    for (first, second), count in pair_games.items():
        opponents[first][second] += count
        opponents[second][first] += count

    strengths = {name: 1.0 for name in names}
    logs = {name: 0.0 for name in names}

    for _ in range(ratings.MAX_ITERATIONS):
        for name in names:
            numerator = wins[name] + ratings.PRIOR_GAMES / 2.0
            denominator = ratings.PRIOR_GAMES / (strengths[name] + 1.0)
            for other, count in opponents[name].items():
                denominator += count / (strengths[name] + strengths[other])
            if denominator <= 0.0:
                continue
            strengths[name] = numerator / denominator

        updated = {name: math.log(s) for name, s in strengths.items()}
        log_mean = sum(updated.values()) / len(strengths)
        scale = math.exp(-log_mean)
        strengths = {name: s * scale for name, s in strengths.items()}

        shift = max(abs(updated[name] - log_mean - logs[name]) for name in names)
        logs = {name: updated[name] - log_mean for name in names}
        if shift < ratings.TOLERANCE:
            break

    return strengths


class _FreshReadings(components.GameReadings):
    """The old path's reading: every game read afresh on every call."""

    def of(self, record: GameRecord) -> components.GameReading:
        return components.GameReading(
            lead_changes=components._lead_changes(record),
            volatility=components._volatility(record),
            irreversibility=components._irreversibility(record),
        )


@contextmanager
def _the_old_path(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with monkeypatch.context() as patch:
        patch.setattr(ratings, "_fit_bradley_terry", _old_fit)
        patch.setattr(bootstrap, "GameReadings", _FreshReadings)
        patch.setattr(components, "GameReadings", _FreshReadings)
        patch.setattr(components, "_budget", components._budget.__wrapped__)
        patch.setattr(components, "_is_weak", components._is_weak.__wrapped__)
        yield


def _synthetic(shape: str, seed: int) -> list[GameRecord]:
    """A ladder corpus with the shapes a fit meets: draws, sweeps, self-play, gaps."""
    rng = random.Random(seed)
    agents = list(LADDER) + ["minimax:nodes=1600"]
    records = []
    game = 0
    for first in agents:
        for second in agents:
            if shape == "no_self_play" and first == second:
                continue
            count = rng.randrange(1, 6) if shape == "uneven" else 4
            for _ in range(count):
                if shape == "sweep":
                    winner: int | None = 0 if agents.index(first) > agents.index(second) else 1
                else:
                    winner = rng.choice([0, 1, None])
                plies = rng.randrange(8, 60)
                records.append(
                    GameRecord(
                        ruleset_hash="fixture", ruleset_name="fixture", seed=game,
                        agents=(first, second), moves=[rng.randrange(40) for _ in range(plies)],
                        winner=winner, reason="checkmate" if winner is not None else "ply_limit",
                        plies=plies, branching=[rng.randrange(1, 40) for _ in range(plies)],
                        material=[rng.uniform(-9, 9) for _ in range(plies)],
                        captures=[rng.randrange(2) for _ in range(plies)],
                    )
                )
                game += 1
    return records


SHAPES = ("level", "sweep", "uneven", "no_self_play")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_fit_is_the_old_fit_to_the_last_bit(shape: str, seed: int) -> None:
    records = _synthetic(shape, seed)
    wins: dict[str, float] = defaultdict(float)
    pair_games: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        for position, name in enumerate(record.agents):
            wins[name] += record.score_for(position)
        pair_games[record.agents] += 1
    names = sorted(wins)
    new = ratings._fit_bradley_terry(names, wins, pair_games)
    old = _old_fit(names, wins, pair_games)
    assert list(new) == list(old)
    assert new == old


def test_a_lone_player_fits_as_before() -> None:
    wins = {"a": 3.0}
    pair_games = {("a", "a"): 6}
    assert ratings._fit_bradley_terry(["a"], wins, pair_games) == _old_fit(
        ["a"], wins, pair_games
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_the_bootstrap_is_the_old_bootstrap(
    shape: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_spec(RULESETS / "dull.json")
    records = _synthetic(shape, 5)
    weights = load_weights("v2")
    wanted = ("skill_gradient", "lead_changes", "material_volatility",
              "progress_irreversibility", "decisive_rate_worst_half")

    def run() -> bootstrap.BootstrapResult | None:
        return bootstrap.bootstrap_index(
            spec, records, weights, resamples=40, seed=3, components_of_interest=wanted
        )

    new = run()
    with _the_old_path(monkeypatch):
        old = run()
    assert new is not None
    assert new == old


@pytest.mark.parametrize("shape", SHAPES)
def test_every_resample_reads_as_it_did_unrounded(shape: str) -> None:
    """The bootstrap's result is rounded to four places; each draw's components are not."""
    spec = load_spec(RULESETS / "dull.json")
    records = _synthetic(shape, 9)
    shared = components.GameReadings()
    rng = random.Random(1)
    for _ in range(20):
        drawn = rng.choices(records, k=len(records))
        table = ratings.rate(drawn)
        new = components.compute_components(spec, drawn, table, readings=shared)
        old = components.compute_components(spec, drawn, table, readings=_FreshReadings())
        assert new.raw == old.raw
        assert new.diagnostics == old.diagnostics
        # And against the three lines as they were written before the memo.
        graded = [
            r for r in drawn if "random" not in (r.agents[0], r.agents[1])
        ] or drawn
        assert new.raw["lead_changes"] == components._mean(
            [components._lead_changes(r) for r in graded]
        )
        assert new.raw["material_volatility"] == components._mean(
            [components._volatility(r) for r in graded]
        )
        assert new.raw["progress_irreversibility"] == components._mean(
            [components._irreversibility(r) for r in graded if r.irreversible]
        )


@pytest.fixture(scope="module")
def played() -> tuple[dict, list[GameRecord], CorpusAnalysis]:
    """A real tournament and its analysis, small enough for the fast suite."""
    spec = load_spec(RULESETS / "gardner.json")
    result = run_tournament(
        spec, TournamentConfig(agents=LADDER, games_per_pairing=2, seed=3, workers=1)
    )
    analysis = analyse_corpus(spec, result.records, blunder_sample=8, workers=1)
    return spec, result.records, analysis


def test_a_played_corpus_scores_as_it_did(
    played: tuple[dict, list[GameRecord], CorpusAnalysis], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole of `score`, interval included, under the current weights."""
    spec, records, analysis = played

    def run() -> dict:
        table = ratings.rate(records)
        scored = score_ruleset(spec, records, table, analysis=analysis)
        return {"ratings": table.as_dict(), "index": scored.as_dict()}

    new = run()
    with _the_old_path(monkeypatch):
        old = run()
    assert new["index"]["interval"] is not None
    assert new == old


def test_each_game_is_read_once_a_bootstrap(
    played: tuple[dict, list[GameRecord], CorpusAnalysis], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saving R2 is for, as a count: one reading per game, not one per game per draw."""
    spec, records, analysis = played
    calls = 0
    volatility = components._volatility

    def counted(record: GameRecord) -> float:
        nonlocal calls
        calls += 1
        return volatility(record)

    monkeypatch.setattr(components, "_volatility", counted)
    result = bootstrap.bootstrap_index(
        spec, records, load_weights(), resamples=50, analysis=analysis
    )
    assert result is not None and result.resamples == 50
    assert 0 < calls <= len(records)
