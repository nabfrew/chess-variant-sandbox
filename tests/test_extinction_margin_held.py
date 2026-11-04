"""SR4d2 (`docs/plan-schema-next.md` §3.2 batch B, under SR4d): the evaluator's
`extinction_margin` counts the hand under `per_type: "own"`, as the rule does.

SR4d made `Game::short_of` add a side's hand to its board under `"own"` (a held piece of a
kind is not gone), and made a kind in the opening stock one the side starts with. The
evaluator's margin read the board alone, so a side whose last wazir sat in its hand read as
having none left, and a side whose wazirs all start in its stock read as extinct at ply
zero. Now the margin's count of a kind is the board plus the hand, read against the opening
board plus the opening stock, in the core (`rust/src/features.rs`) and in its oracle
(`agents/features.py`) alike.

Three things are asked: the oracle's number on hand-built positions is the new reading and
not the old; the core values those positions and a played game's as the oracle does; and
games without a hand beside `"own"` (or with a hand and another spelling) are valued to the
bit as before the change, pinned off the core built just before it. Hand-built specs, not
chess.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import pytest

from chess_variant_sandbox.agents.base import native_spec
from chess_variant_sandbox.agents.features import VOCABULARY, FeatureEvaluator
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import eval_values, play_game, play_game_python
from tests.test_feature_terms import (
    _assert_valued_alike,
    _beside_material,
    _gardner,
    _term_alone,
)
from tests.test_held_kind_counts import _mirrored as _own_mirrored
from tests.test_schema_twenty_seven import gap

TERM = "extinction_margin"
INDEX = VOCABULARY.index(TERM)
HAND = {"from_captures": True}
# Player 0's wazirs start in its hand, none on the board.
STOCKED = {"from_captures": True, "stock_by_player": {"0": {"A": 2}, "1": {}}}


def _mirrored(per_type: bool) -> dict[str, Any]:
    """SR4d's mirrored armies beside a capture hand, under `per_type: true`: the board alone."""
    spec = _own_mirrored()
    spec["rules"]["extinction"]["per_type"] = per_type
    return spec


def _stocked_spec() -> dict[str, Any]:
    spec = gap(drops=STOCKED)
    spec["setup"]["placements"] = [p for p in spec["setup"]["placements"] if p["piece"] != "A"]
    return spec


def _margin(engine: Engine, state: GameState, player: int) -> float:
    reader = FeatureEvaluator.for_engine(engine, (1.0,) * len(VOCABULARY))
    return reader.features(engine, state, player)[INDEX]


def _held_position(engine: Engine, wazirs_in_hand: int) -> GameState:
    """Player 0 with no wazir on the board and `wazirs_in_hand` in its hand; else whole."""
    placement = {"c1": ("B", 0), "d1": ("B", 0), "b6": ("C", 1), "e6": ("C", 1),
                 "c6": ("E", 1), "d6": ("E", 1)}
    state = engine.make_state(placement, side_to_move=0, ply=10)
    hands = [list(hand) for hand in engine.initial_state().hands]
    hands[0][engine.spec.piece_ids.index("A")] = wazirs_in_hand
    return dataclasses.replace(state, hands=(tuple(hands[0]), tuple(hands[1])))


class TestTheReading:
    def test_the_last_wazir_in_hand_is_not_gone(self) -> None:
        """Old: no wazir on the board, 0 of 2 left, margin -1 (the side reads extinct).
        New: one of its two in hand, half left, margin -0.5; the rule agrees it is alive."""
        engine = Engine(gap(drops=HAND))
        state = _held_position(engine, 1)
        assert engine.terminal(state) is None
        assert _margin(engine, state, 0) == -0.5
        assert _margin(engine, state, 1) == 0.5
        both = _held_position(engine, 2)
        assert _margin(engine, both, 0) == 0.0

    def test_an_empty_hand_reads_as_the_board(self) -> None:
        engine = Engine(gap(drops=HAND))
        state = _held_position(engine, 0)
        assert _margin(engine, state, 0) == -1.0

    def test_a_stocked_kind_starts_whole(self) -> None:
        """Old: the stocked wazirs read 0 of max(0, 1) at ply zero, margin -1. New: 2 of 2."""
        engine = Engine(_stocked_spec())
        state = engine.initial_state()
        assert engine.terminal(state) is None
        assert _margin(engine, state, 0) == 0.0
        assert _margin(engine, state, 1) == 0.0


def _drop_game_positions(engine: Engine, seed: int, limit: int = 24) -> list[GameState]:
    record = play_game(engine, ("random", "random"), seed=seed)
    states, state = [], engine.initial_state()
    for move in record.moves[:limit]:
        states.append(state)
        state = engine.apply(state, move)
    return states


class TestTheCoreAgrees:
    def test_the_hand_built_positions(self) -> None:
        engine = Engine(gap(drops=HAND))
        positions = [_held_position(engine, n) for n in (1, 2)]
        _assert_valued_alike(engine, _term_alone(TERM), positions)

    def test_a_stocked_game_s_positions(self) -> None:
        """Random play through the stocked game: drops and captures move the hands."""
        engine = Engine(_stocked_spec())
        positions = _drop_game_positions(engine, seed=2, limit=60)
        reads = {_margin(engine, state, 0) for state in positions}
        assert len(reads) >= 3, "the guard: the term moves across these positions"
        assert any(any(state.hands[0]) for state in positions), "the guard: a hand held"
        _assert_valued_alike(engine, _term_alone(TERM), positions)

    @pytest.mark.parametrize("build", [lambda: gap(drops=HAND), _stocked_spec],
                             ids=["capture-hand", "stocked"])
    def test_a_played_game_is_the_same_game(self, build: Any) -> None:
        engine = Engine(build())
        agents = (_beside_material(TERM), "random")
        native = play_game(engine, agents, seed=2)
        in_python = play_game_python(engine, agents, seed=2)
        assert native.moves == in_python.moves
        assert native.nodes == in_python.nodes


# Games the change must not move: `"own"` with no hand, `true` beside a hand (the board
# alone), and a summed count beside a hand. The core's one-ply values with the term alone
# over a random game's first 24 positions, digested; pinned off the core built at SR4d
# (b1ecfee0), before this change.
PLAIN: list[tuple[str, Any]] = [
    ("own-no-hand", lambda: gap()),
    ("per-type-true-beside-a-hand", lambda: _mirrored(True)),
    ("summed-beside-a-hand", lambda: _gardner(
        extinction={"pieces": ["N"]}, drops={"from_captures": True})),
]
PINNED = {
    "own-no-hand": "de45881a6f4cb461",
    "per-type-true-beside-a-hand": "e46ab9a53271782c",
    "summed-beside-a-hand": "f4a7387e233c0ef6",
}


def plain_digest(engine: Engine) -> str:
    states = [s for s in _drop_game_positions(engine, seed=1) if engine.terminal(s) is None]
    native = native_spec(_term_alone(TERM), engine.spec.hash)
    assert native is not None
    seeds = [5 + index for index in range(len(states))]
    answers = engine.core.search_values(
        [_pack(state) for state in states], native, eval_values(engine.spec), seeds
    )
    text = repr([(value.hex(), move, depth) for value, move, depth in answers])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.mark.parametrize("label,build", PLAIN, ids=[p[0] for p in PLAIN])
def test_games_without_a_held_count_are_valued_as_before(label: str, build: Any) -> None:
    assert plain_digest(Engine(build())) == PINNED[label]
