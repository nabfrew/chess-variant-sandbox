"""The winning-chances fit (Q701, `docs/reference/win-probability.md`), on fakes and fixtures.

Nothing here reads `runs/`: the readings come from a scripted fake engine, and the fit and
its verdict are pure functions of readings and a seed. What only the box can prove (the
verdict on real library runs) is the record's §2.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pytest

from chess_variant_sandbox import measurement, winprob
from chess_variant_sandbox.tournament.record import GameRecord
from chess_variant_sandbox.winprob import (
    CALIBRATED,
    ROUGH,
    TOO_FEW,
    GameReading,
    Position,
    Ruler,
    calibration,
    chances,
    fit,
    fit_slope,
    held_out,
    read_game,
    read_run,
)

RULER = Ruler(
    nodes=10, max_depth=2, goal=0.0, immune=0.0, unit=2.0, decided_at=500_000.0, build="fake"
)


# ------------------------------------------------------------------ a scripted engine


@dataclass(frozen=True)
class _State:
    ply: int
    side_to_move: int


class FakeEngine:
    """Every position has ``widths[ply]`` legal moves numbered 10*ply + i; sides alternate.

    Its root scores are ``table[ply]``. It is the engine's question-answering stood in by
    a script, so a reading can be checked number for number.
    """

    def __init__(self, widths: list[int], table: dict[int, list[float]]) -> None:
        self.widths = widths
        self.table = table
        self.asked: list[int] = []

    def initial_state(self) -> _State:
        return _State(0, 0)

    def legal_moves(self, state: _State) -> list[int]:
        return [10 * state.ply + i for i in range(self.widths[state.ply])]

    def apply(self, state: _State, move: int) -> _State:
        return _State(state.ply + 1, 1 - state.side_to_move)


def scripted(engine: FakeEngine, state: _State, ruler: Ruler) -> list[float]:
    engine.asked.append(state.ply)
    return engine.table[state.ply]


def record(moves: list[int], winner: int | None, seed: int = 0) -> GameRecord:
    return GameRecord(
        ruleset_hash="fake",
        ruleset_name="fake",
        seed=seed,
        agents=("a", "b"),
        moves=moves,
        winner=winner,
        reason="test",
        plies=len(moves),
    )


# ---------------------------------------------------------------------------- readings


def test_a_reading_takes_the_best_and_the_played_score_from_the_engine() -> None:
    engine = FakeEngine([3, 1, 2], {0: [1.0, 4.0, -2.0], 1: [0.5], 2: [-1.0, 3.0]})
    reading = read_game(engine, record([2, 10, 20], winner=0), RULER, scores=scripted)
    first, forced, third = reading.positions
    assert (first.best, first.played, first.best_move, first.mover) == (4.0, -2.0, 1, 0)
    assert forced.forced and forced.mover == 1 and forced.choices == 1
    assert (third.best, third.played, third.best_move) == (3.0, -1.0, 21)
    assert reading.result_for(0) == 1.0 and reading.result_for(1) == 0.0


def test_a_draw_counts_half_for_both_sides() -> None:
    reading = GameReading(0, 0, ("a", "b"), None, ())
    assert reading.result_for(0) == reading.result_for(1) == 0.5


def test_a_move_the_engine_calls_illegal_is_an_error() -> None:
    engine = FakeEngine([2], {0: [0.0, 0.0]})
    with pytest.raises(ValueError, match="illegal"):
        read_game(engine, record([7], winner=0), RULER, scores=scripted)


def test_a_score_list_that_does_not_match_the_legal_moves_is_an_error() -> None:
    engine = FakeEngine([3], {0: [0.0, 1.0]})
    with pytest.raises(ValueError, match="3 moves and 2 scores"):
        read_game(engine, record([0], winner=0), RULER, scores=scripted)


def test_a_sampled_run_asks_only_for_its_plies_and_the_sample_is_seeded() -> None:
    widths = [2] * 12
    table = {ply: [0.0, 1.0] for ply in range(12)}
    games = [record([10 * p for p in range(12)], winner=0, seed=s) for s in range(3)]

    def plies(seed: int) -> list[list[int]]:
        engine = FakeEngine(widths, table)
        run = read_run(engine, games, RULER, seed=seed, positions_per_game=4, scores=scripted)
        assert all(len(r.positions) == 4 for r in run)
        assert len(engine.asked) == 12
        return [[p.ply for p in r.positions] for r in run]

    assert plies(5) == plies(5)
    assert plies(5) != plies(6)
    whole = read_run(FakeEngine(widths, table), games, RULER, scores=scripted)
    assert [len(r.positions) for r in whole] == [12, 12, 12]


# ----------------------------------------------------------------------------- the map


def test_the_map_is_monotone_and_symmetric() -> None:
    xs = [x / 4 for x in range(-40, 41)]
    for k in (0.0, 0.3, 2.0, 50.0):
        ps = [chances(k, x) for x in xs]
        assert ps == sorted(ps)
        for x in xs:
            assert chances(k, -x) == pytest.approx(1.0 - chances(k, x))
    assert chances(50.0, 1e6) == 1.0 and chances(50.0, -1e6) == 0.0


def test_the_fit_recovers_a_known_slope() -> None:
    rng = random.Random(0)
    pairs = []
    for _ in range(20_000):
        x = rng.uniform(-4.0, 4.0)
        pairs.append((x, 1.0 if rng.random() < chances(1.5, x) else 0.0))
    assert fit_slope(pairs) == pytest.approx(1.5, abs=0.1)


def test_soft_labels_recover_the_slope_exactly() -> None:
    pairs = [(x / 10, chances(0.8, x / 10)) for x in range(-50, 51)]
    assert fit_slope(pairs) == pytest.approx(0.8, abs=1e-6)


def test_no_data_or_backwards_data_gives_even_chances() -> None:
    assert fit_slope([]) == 0.0
    assert fit_slope([(1.0, 0.0), (-1.0, 1.0)]) == 0.0


def test_a_decided_value_is_certain_and_left_out_of_the_fit() -> None:
    games = [
        GameReading(i, i, ("a", "b"), 0, (Position(0, 0, 2, 600_000.0, 0.0, 0),)) for i in range(4)
    ]
    win = fit(games, RULER, 0, "fake", "digest")
    assert win.basis.decided == 4
    assert win.basis.fit_positions + win.basis.held_positions == 0
    assert win.chances(600_000.0) == 1.0 and win.chances(-600_000.0) == 0.0


# ------------------------------------------------------------------- split and basis


def _readings(n: int, k: float, seed: int = 0) -> list[GameReading]:
    """``n`` games of 30 positions whose results follow ``chances(k, .)`` per game."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        positions = tuple(Position(p, p % 2, 3, rng.uniform(-6.0, 6.0), 0.0, 0) for p in range(30))
        x = RULER.scale(positions[0].best)
        winner = 0 if rng.random() < chances(k, x) else 1
        out.append(GameReading(i, 1000 + i, ("a", "b"), winner, positions))
    return out


def test_the_split_is_by_game_deterministic_and_moved_only_by_the_seed() -> None:
    readings = _readings(200, 1.0)
    halves = [held_out(3, r) for r in readings]
    assert halves == [held_out(3, r) for r in readings]
    assert halves != [held_out(4, r) for r in readings]
    assert 60 < sum(halves) < 140


def test_a_fit_is_a_pure_function_of_readings_and_seed_and_carries_its_basis() -> None:
    readings = _readings(60, 1.0)
    one = fit(readings, RULER, 7, "fake", "digest")
    assert one.as_dict() == fit(readings, RULER, 7, "fake", "digest").as_dict()
    assert one.as_dict() != fit(readings, RULER, 8, "fake", "digest").as_dict()
    basis = one.as_dict()["basis"]
    assert basis["method"] == winprob.METHOD
    assert basis["seed"] == 7 and basis["ruleset_hash"] == "fake"
    assert basis["fit_games"] + basis["held_games"] == 60
    assert basis["ruler"]["unit"] == 2.0 and basis["ruler"]["build"] == "fake"


def test_the_records_digest_reads_moves_results_seeds_and_players() -> None:
    a = record([1, 2], winner=0)
    assert winprob.records_digest([a]) == winprob.records_digest([record([1, 2], winner=0)])
    assert winprob.records_digest([a]) != winprob.records_digest([record([1, 2], winner=1)])
    assert winprob.records_digest([a]) != winprob.records_digest([record([2, 1], winner=0)])


# ---------------------------------------------------------------------------- verdict


def _exact(k: float, n: int, span: float = 3.0) -> list[tuple[float, float]]:
    """Positions in ``[-span, span]`` whose observed result is exactly the map's chances."""
    xs = [-span + 2.0 * span * i / (n - 1) for i in range(n)]
    return [(x, chances(k, x)) for x in xs]


def test_perfect_chances_are_calibrated() -> None:
    result = calibration(1.0, _exact(1.0, 1000), games=20)
    assert result.verdict == CALIBRATED and result.failures == ()
    assert result.mean_error == pytest.approx(0.0, abs=0.01)
    assert sum(b.count for b in result.bins) == 1000


def test_too_few_positions_or_games_reads_nothing_else() -> None:
    assert calibration(1.0, _exact(1.0, 199), games=20).verdict == TOO_FEW
    assert calibration(1.0, _exact(1.0, 1000), games=9).verdict == TOO_FEW


def test_a_wrong_slope_fails_the_mean_error() -> None:
    result = calibration(4.0, _exact(1.0, 1000), games=20)
    assert result.verdict == ROUGH
    assert any("mean calibration error" in f for f in result.failures)


def test_one_bad_bin_fails_alone() -> None:
    pairs = _exact(1.0, 4000, span=1.0)
    # 60 positions predicted near 0.95, in a bin of their own, that the side to move lost.
    pairs += [(3.0, 0.0)] * 60
    result = calibration(1.0, pairs, games=20)
    assert result.mean_error <= winprob.MAX_MEAN_ERROR
    assert result.verdict == ROUGH
    assert [f for f in result.failures if f.startswith("bin 0.9")]


def test_knowing_nothing_is_not_calibrated() -> None:
    result = calibration(0.0, _exact(1.0, 1000), games=20)
    assert result.mean_error == pytest.approx(0.0, abs=1e-9)
    assert result.verdict == ROUGH
    assert result.failures == (
        f"Brier {result.brier:.4f} is not below the constant's {result.brier_constant:.4f}",
    )


def test_a_fit_on_well_behaved_games_is_calibrated_and_finds_its_slope() -> None:
    readings = _readings(400, 0.0)
    # Results that follow each position's own chances, game by game, at k = 1 per unit.
    rng = random.Random(1)
    honest = []
    for r in readings:
        x = RULER.scale(r.positions[0].best)
        winner = 0 if rng.random() < chances(1.0, x) else 1
        honest.append(GameReading(r.index, r.seed, r.agents, winner, r.positions[:1]))
    win = fit(honest, RULER, 0, "fake", "digest")
    assert win.k == pytest.approx(1.0, abs=0.35)
    assert math.isfinite(win.calibration.brier)


# ------------------------------------------------------------------ it moves nothing


@pytest.mark.parametrize("kind", ["engine", "metrics", "analysis", "agents"])
def test_the_new_modules_are_in_no_fingerprinted_set(kind: str) -> None:
    names = {path.name for path in measurement.fingerprinted(kind)}
    assert "winprob.py" not in names and "accuracy.py" not in names
