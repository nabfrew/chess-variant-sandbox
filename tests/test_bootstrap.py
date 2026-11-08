"""`bootstrap_index` and its result, pinned on small synthetic corpora (Q1145).

The fixtures are synthetic records on ``rulesets/dull.json``: three ladder rungs, six
ordered pairings of three games each, whose outcomes are chosen so that the pairings
hold 1, 2, 2, 3, 3 and 3 distinct ``(winner, plies)`` outcomes. One game, and only that
one, carries an irreversibility series, so ``progress_irreversibility`` is present in a
resample only when that game is drawn.

The pool's own behaviour (``pool_workers``, ``_score_draws``, ``_percentile``,
``component_noise``) is ``tests/test_bootstrap_pool.py``'s; this file reaches the pool
only through the constants that size it and the worker's one entry point.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.metrics import bootstrap
from chess_variant_sandbox.metrics.bootstrap import (
    BootstrapResult,
    bootstrap_index,
    clears,
    draw_resamples,
)
from chess_variant_sandbox.metrics.scoring import load_weights
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.record import GameRecord

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"
RUNGS = ("minimax:nodes=100", "minimax:nodes=200", "minimax:nodes=400")
PAIRINGS = [
    (RUNGS[0], RUNGS[1]), (RUNGS[1], RUNGS[0]), (RUNGS[0], RUNGS[2]),
    (RUNGS[2], RUNGS[0]), (RUNGS[1], RUNGS[2]), (RUNGS[2], RUNGS[1]),
]
# (winner, plies) per game, one row a pairing: 1, 2, 2, 3, 3 and 3 distinct outcomes.
OUTCOMES = [
    [(0, 20), (0, 20), (0, 20)],
    [(1, 24), (1, 24), (None, 30)],
    [(0, 22), (1, 22), (1, 22)],
    [(0, 18), (1, 26), (None, 30)],
    [(1, 21), (0, 23), (None, 25)],
    [(None, 30), (0, 19), (1, 27)],
]
TRACKED = ("ply_spread", "progress_irreversibility")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return load_spec(RULESETS / "dull.json")


def _corpus() -> list[GameRecord]:
    rng = random.Random(4)
    records = []
    seed = 0
    for pairing, outcomes in zip(PAIRINGS, OUTCOMES, strict=True):
        for winner, plies in outcomes:
            records.append(
                GameRecord(
                    ruleset_hash="fixture", ruleset_name="fixture", seed=seed,
                    agents=pairing, moves=[rng.randrange(50) for _ in range(plies)],
                    winner=winner, reason="fixture", plies=plies,
                    branching=[rng.randrange(5, 30) for _ in range(plies)],
                    material=[rng.uniform(-3, 3) for _ in range(plies)],
                    captures=([1] + [0] * (plies - 1)) if seed == 0 else [],
                )
            )
            seed += 1
    return records


def _result(**overrides: Any) -> BootstrapResult:
    fields: dict[str, Any] = {
        "resamples": 200, "mean": 0.5, "sd": 0.01, "ci_low": 0.48, "ci_high": 0.52,
        "deterministic_pairings": 0.0, "component_sd": {},
    }
    fields.update(overrides)
    return BootstrapResult(**fields)


class TestTheWholeResult:
    def test_two_resamples_pin_every_field(self, spec: dict[str, Any]) -> None:
        """Seed 8 draws the one irreversible game into exactly one of the two resamples,
        so ``progress_irreversibility`` has one value and is left out of both component
        dicts, while ``ply_spread`` has two and is in. Every number carries a fourth
        decimal that a third or a fifth would change."""
        draws = draw_resamples([3] * 6, None, 2, 8)
        assert [0 in chosen for chosen, _ in draws] == [True, False]
        result = bootstrap_index(
            spec, _corpus(), load_weights("v2"), resamples=2, seed=8,
            components_of_interest=TRACKED,
        )
        assert result is not None
        assert result.as_dict() == {
            "resamples": 2,
            "mean": 0.3517,
            "sd": 0.0181,
            "ci_low": 0.3344,
            "ci_high": 0.3689,
            "deterministic_pairings": 0.167,
            "component_sd": {"ply_spread": 0.0019},
            "component_mean": {"ply_spread": 0.1489},
            "stratum_correction": 1.2247,
        }

    def test_one_scored_resample_is_no_interval(self, spec: dict[str, Any]) -> None:
        assert bootstrap_index(spec, _corpus(), load_weights("v2"), resamples=1, seed=8) is None

    def test_a_dropped_resample_is_skipped_not_the_end(
        self, spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A degenerate first resample is dropped and the rest are still read."""
        monkeypatch.setattr(
            bootstrap, "_score_draws", lambda scorer, draws, workers: [None, (0.4, {}), (0.6, {})]
        )
        result = bootstrap_index(spec, _corpus(), load_weights("v2"), resamples=3)
        assert result is not None
        assert result.resamples == 2
        assert result.mean == 0.5


class TestHowFewGames:
    def _eight(self) -> list[GameRecord]:
        records = _corpus()
        return records[:6] + [replace(records[3], seed=90), replace(records[0], seed=91)]

    def test_eight_games_are_enough(self, spec: dict[str, Any]) -> None:
        result = bootstrap_index(spec, self._eight(), load_weights("v2"), resamples=3)
        assert result is not None
        assert result.resamples == 3
        assert result.deterministic_pairings == 0.5

    def test_seven_games_are_not(self, spec: dict[str, Any]) -> None:
        assert bootstrap_index(spec, self._eight()[:7], load_weights("v2"), resamples=3) is None

    def test_one_pairing_that_always_ends_alike_is_wholly_deterministic(
        self, spec: dict[str, Any]
    ) -> None:
        one = [replace(_corpus()[0], seed=100 + i) for i in range(8)]
        result = bootstrap_index(spec, one, load_weights("v2"), resamples=3)
        assert result is not None
        assert result.deterministic_pairings == 1.0


class TestDefaults:
    def test_a_bare_call_draws_two_hundred_from_seed_zero_and_lets_the_pool_decide(
        self, spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``DEFAULT_RESAMPLES`` is load-bearing: every `measure` score is 200 resamples,
        and the stored interval is the one those draws give."""
        seen: dict[str, Any] = {}
        real_draw = bootstrap.draw_resamples

        def draw(sizes: list[int], blunders: int | None, resamples: int, seed: int) -> Any:
            seen["resamples"], seen["seed"] = resamples, seed
            return real_draw(sizes, blunders, resamples, seed)

        def score(scorer: Any, draws: list[Any], workers: int) -> list[Any]:
            seen["workers"] = workers
            return [(0.4, {}), (0.6, {})]

        monkeypatch.setattr(bootstrap, "draw_resamples", draw)
        monkeypatch.setattr(bootstrap, "_score_draws", score)
        assert bootstrap_index(spec, _corpus(), load_weights("v2")) is not None
        assert seen == {"resamples": 200, "seed": 0, "workers": 0}

    def test_a_result_stored_before_the_correction_reads_as_uncorrected(self) -> None:
        assert _result().stratum_correction == 1.0

    def test_a_result_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _result().mean = 0.9  # type: ignore[misc]


class TestSeparation:
    def test_touching_intervals_do_not_clear(self) -> None:
        assert clears(0.5, 0.5) is False
        assert clears(0.5001, 0.5) is True
        assert clears(0.4999, 0.5) is False

    def test_a_clear_gap_beats_below_the_deterministic_share(self) -> None:
        high = _result(ci_low=0.7, ci_high=0.8, deterministic_pairings=0.79)
        low = _result(ci_low=0.1, ci_high=0.2, deterministic_pairings=0.0)
        assert high.beats(low) is True
        assert high.separates(low) is True
        assert low.beats(high) is False

    def test_a_clear_gap_is_refused_at_the_deterministic_share(self) -> None:
        """At 0.8 exactly, and on either side, a sweep-built interval is no evidence."""
        high = _result(ci_low=0.7, ci_high=0.8, deterministic_pairings=0.8)
        low = _result(ci_low=0.1, ci_high=0.2, deterministic_pairings=0.0)
        assert high.beats(low) is False
        assert high.separates(low) is False
        assert _result(ci_low=0.7, ci_high=0.8).beats(
            _result(ci_low=0.1, ci_high=0.2, deterministic_pairings=0.85)
        ) is False


class TestTheScorer:
    def test_a_resample_that_cannot_be_scored_is_none(self, spec: dict[str, Any]) -> None:
        """v12 asks for second-pass components; without an analysis every resample is
        degenerate and dropped, not raised."""
        scorer = bootstrap._Scorer(spec, _corpus(), load_weights("v12"), None, {}, ())
        assert scorer.score((list(range(18)), None)) is None

    def test_a_worker_loads_its_call_once_and_a_new_call_anew(
        self, spec: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = _corpus()
        weights = load_weights("v2")
        first = bootstrap._Scorer(spec, records, weights, None, {}, ("ply_spread",))
        second = bootstrap._Scorer(spec, records[::-1], weights, None, {}, ("ply_spread",))
        draw = (list(range(9)), None)
        expected_first = first.score(draw)
        expected_second = second.score(draw)
        assert expected_first is not None and expected_second is not None
        assert expected_first != expected_second

        def dump(scorer: bootstrap._Scorer, name: str) -> str:
            path = tmp_path / name
            path.write_bytes(pickle.dumps(scorer))
            return str(path)

        monkeypatch.setattr(bootstrap, "_WORKER_SCORER", None)
        one = dump(first, "one.pickle")
        assert bootstrap._score_chunk(("call-1", one, [draw])) == [expected_first]
        # The same call again reads the scorer it holds, not the file.
        os.remove(one)
        assert bootstrap._score_chunk(("call-1", one, [draw, draw])) == [expected_first] * 2
        # A new call reads its own file.
        two = dump(second, "two.pickle")
        assert bootstrap._score_chunk(("call-2", two, [draw])) == [expected_second]


class TestThePoolsSize:
    def test_the_pool_starts_at_twenty_thousand_drawn_games(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        assert bootstrap.pool_workers(0, 19_999) == 1
        assert bootstrap.pool_workers(0, 20_000) == 4

    def test_the_pool_is_four_workers_at_most(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(os, "cpu_count", lambda: 16)
        assert bootstrap.pool_workers(0, 10**6) == 4
        monkeypatch.setattr(os, "cpu_count", lambda: 4)
        assert bootstrap.pool_workers(0, 10**6) == 3
