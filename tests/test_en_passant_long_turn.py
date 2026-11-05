"""SR7a (`docs/plan-schema-next.md` §2.2 row L1, ruled 2026-10-27): en passant in a long turn.

The ruled reading: the window is the opponent's, opened by the mover's *last* double step of
the turn, and open for the opponent's first move only. Inside the mover's own turn a later
move keeps it (the mover may not take their own pawn in passing, Q531), unless that move
disturbs it: the pawn moves on, is removed, or a piece is set down on the square it passed,
which closes it for good. The core is `Game::keeps_ep_window`.

The oracle is this file's `_expected_window`, which reads the window off the history of the
game rather than off the core's state (a declared test oracle, rule 3): the last double step
of the previous turn whose pawn still stands where it landed and has not moved since, with
nothing on the square it passed, while it is the first move of the answering turn or a later
move of the turn that made it. Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import (
    MOVE_EN_PASSANT,
    MOVE_NORMAL,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import NO_SQUARE, GameState
from chess_variant_sandbox.spec.schema import validate_spec

FIXTURE = Path(__file__).parent / "_en_passant_in_a_long_turn.json"
WIDTH = 6
PAWN = 2  # the type index of `P` in `_spec`

KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
PAWN_MOVES = [
    {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
    {"kind": "slide", "vectors": [[0, 1]], "range": 2, "min_range": 2, "mode": "move",
     "condition": {"from_rank": 1}},
    {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
]


def _spec(turn: dict[str, Any], placements: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """6x6: kings, a knight to promote to, pawns that double-step from their second rank."""
    return copy.deepcopy({
        "schema_version": 14, "name": "two steps at a time",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "N", "value": 3, "moves": [
                {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}]},
            {"id": "P", "value": 1, "moves": PAWN_MOVES},
        ],
        "setup": {"mirror": False, "placements": placements or [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "P", "player": 0, "squares": ["b2", "c2", "d2", "e2"]},
            {"piece": "P", "player": 1, "squares": ["b5", "c5", "d5", "e5"]},
        ]},
        "rules": {
            "win": "royal_capture", "stalemate": "draw", "max_plies": 80,
            "promotion": {"piece": "P", "rank": "last", "to": ["N"]},
            "en_passant": True, "turn": turn,
        },
    })


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _passing(engine: Engine, state: GameState) -> list[str]:
    return sorted(text for text in engine.move_texts(state) if "e.p." in text)


def _play(engine: Engine, state: GameState, *texts: str) -> GameState:
    for text in texts:
        state = engine.apply(state, engine.find_move(state, text))
    return state


class TestTheValidatorLiftsTheRefusal:
    def test_the_fuzzers_fixture_validates(self) -> None:
        spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
        assert validate_spec(spec) == []

    def test_a_ramp_is_a_long_turn_and_validates_too(self) -> None:
        spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
        spec["rules"]["turn"] = {"moves": 1, "ramp": [2]}
        assert validate_spec(spec) == []


class TestTheFixture:
    """The fuzzer's spec: White double-steps b2-b4 with a second move to come, the pawn on a2
    beside the window. Before Q531 `Pa2xb3 e.p.` was legal and, with the strike, panicked."""

    def test_the_mover_never_takes_their_own_double_step(self) -> None:
        engine = Engine(compile_spec(json.loads(FIXTURE.read_text(encoding="utf-8"))))
        state = _play(engine, engine.initial_state(), "Pb2-b4")
        assert state.side_to_move == 0, "the turn has a second move"
        assert state.ep_target == _sq5("b3"), "the window is open, and stays the opponent's"
        assert _passing(engine, state) == []

    def test_the_window_survives_the_movers_second_move(self) -> None:
        engine = Engine(compile_spec(json.loads(FIXTURE.read_text(encoding="utf-8"))))
        state = _play(engine, engine.initial_state(), "Pb2-b4", "Ka1-b1")
        assert state.side_to_move == 1
        assert (state.ep_target, state.ep_victim) == (_sq5("b3"), _sq5("b4"))


def _sq5(name: str) -> int:
    return (int(name[1:]) - 1) * 5 + ord(name[0]) - ord("a")


class TestTheRuledReading:
    """One case per clause of the reading, on a 6x6 board, two moves a turn."""

    def _engine(self) -> Engine:
        return _engine(_spec({"moves": 2}, [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "P", "player": 0, "squares": ["a2", "c2"]},
            {"piece": "P", "player": 1, "squares": ["d4"]},
        ]))

    def test_a_double_step_on_the_first_move_is_answered(self) -> None:
        """Opened on the turn's first move, kept by its second, answered by the next first."""
        engine = self._engine()
        state = _play(engine, engine.initial_state(), "Pc2-c4", "Ka1-b1")
        assert state.side_to_move == 1
        assert _passing(engine, state) == ["Pd4xc3 e.p."]
        taken = _play(engine, state, "Pd4xc3 e.p.")
        assert engine.spec.piece_at(taken.board, _sq("c4")) is None

    def test_a_double_step_on_the_last_move_is_answered(self) -> None:
        engine = self._engine()
        state = _play(engine, engine.initial_state(), "Ka1-b1", "Pc2-c4")
        assert _passing(engine, state) == ["Pd4xc3 e.p."]

    def test_only_the_last_double_step_of_the_turn_opens_it(self) -> None:
        engine = self._engine()
        state = _play(engine, engine.initial_state(), "Pc2-c4", "Pa2-a4")
        assert (state.ep_target, state.ep_victim) == (_sq("a3"), _sq("a4"))
        assert _passing(engine, state) == []

    def test_the_window_is_open_for_the_opponents_first_move_only(self) -> None:
        engine = self._engine()
        state = _play(engine, engine.initial_state(), "Pc2-c4", "Ka1-b1", "Kf6-f5")
        assert state.side_to_move == 1, "the opponent's second move"
        assert (state.ep_target, state.ep_victim) == (NO_SQUARE, NO_SQUARE)
        assert _passing(engine, state) == []

    def test_a_pawn_that_moves_on_closes_it(self) -> None:
        engine = self._engine()
        state = _play(engine, engine.initial_state(), "Pc2-c4", "Pc4-c5")
        assert (state.ep_target, state.ep_victim) == (NO_SQUARE, NO_SQUARE)
        assert _passing(engine, state) == []

    def test_a_piece_set_down_where_it_passed_closes_it(self) -> None:
        """The king walks onto the passed square on the turn's last move (three moves)."""
        engine = _engine(_spec({"moves": 3}, [
            {"piece": "K", "player": 0, "squares": ["b2"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "P", "player": 0, "squares": ["c2"]},
            {"piece": "P", "player": 1, "squares": ["d4"]},
        ]))
        state = _play(engine, engine.initial_state(), "Pc2-c4", "Kb2-b3")
        assert state.ep_target == _sq("c3"), "a move that does not touch it keeps it"
        state = _play(engine, state, "Kb3-c3")
        assert (state.ep_target, state.ep_victim) == (NO_SQUARE, NO_SQUARE)
        assert _passing(engine, state) == []

    def test_one_move_turns_are_unchanged(self) -> None:
        engine = _engine(_spec({"moves": 1}, [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": ["f6"]},
            {"piece": "P", "player": 0, "squares": ["c2"]},
            {"piece": "P", "player": 1, "squares": ["d4"]},
        ]))
        state = _play(engine, engine.initial_state(), "Pc2-c4")
        assert _passing(engine, state) == ["Pd4xc3 e.p."]
        state = _play(engine, state, "Kf6-f5", "Ka1-b1")
        assert _passing(engine, state) == []


def _is_double_step(engine: Engine, before: GameState, move: int) -> bool:
    origin, target = move_from(move), move_to(move)
    held = engine.spec.piece_at(before.board, origin)
    return (
        move_type(move) == MOVE_NORMAL
        and held is not None and held[1] == PAWN
        and origin % WIDTH == target % WIDTH
        and abs(origin // WIDTH - target // WIDTH) == 2
    )


Step = tuple[GameState, int, GameState]


def _length(turn: dict[str, Any], index: int) -> int:
    """How many moves turn `index` has, read off the spec's `rules.turn`."""
    ramp = turn.get("ramp", [])
    return ramp[index] if index < len(ramp) else turn["moves"]


def _expected_window(
    engine: Engine, turn: dict[str, Any], history: list[Step], state: GameState
) -> tuple[int, int]:
    """The window `state` should carry, read off the moves that led to it.

    The turn that may have opened it is the previous turn when `state` begins a turn (the
    other player's), and the turn under way otherwise (the side to move's own, so far). Its
    window is that turn's last double step, closed by a later move of the turn that moves the
    pawn or sets a piece on the square it passed, and by the pawn being gone or no longer the
    mover's now.
    """
    first = state.remaining == _length(turn, state.turns_taken)
    turn_of = state.turns_taken - 1 if first else state.turns_taken
    window: tuple[int, int] | None = None
    for before, m, after in history:
        if before.turns_taken != turn_of:
            continue
        if _is_double_step(engine, before, m):
            window = ((move_from(m) + move_to(m)) // 2, move_to(m))
        elif window is not None and (
            move_from(m) == window[1] or engine.spec.piece_at(after.board, window[0]) is not None
        ):
            window = None
    if window is None:
        return NO_SQUARE, NO_SQUARE
    passed, victim = window
    mover = 1 - state.side_to_move if first else state.side_to_move
    if engine.spec.piece_at(state.board, victim) != (mover, PAWN):
        return NO_SQUARE, NO_SQUARE
    if engine.spec.piece_at(state.board, passed) is not None:
        return NO_SQUARE, NO_SQUARE
    return passed, victim


@pytest.mark.parametrize("turn", [{"moves": 2}, {"moves": 2, "ramp": [1]}, {"moves": 3}])
def test_seeded_walks_hold_the_window_to_the_oracle(turn: dict[str, Any]) -> None:
    """Seeded games: at every position the core's window is the oracle's, every en passant
    offered takes the window's pawn and never the mover's own, and the key is the slow one.
    The walks meet the window both on a turn's first move and inside the turn that opened
    it, and take a pawn in passing."""
    engine = _engine(_spec(turn))
    answered = inside = 0
    for seed in range(24):
        rng = random.Random(seed)
        state = engine.initial_state()
        history: list[Step] = []
        for _ in range(60):
            assert (state.ep_target, state.ep_victim) == _expected_window(
                engine, turn, history, state
            ), (seed, len(history))
            assert state.key == engine._compute_hash(state)
            if engine.terminal(state) is not None:
                break
            inside += state.ep_target != NO_SQUARE and state.remaining != _length(
                turn, state.turns_taken
            )
            moves = engine.legal_moves(state)
            passing = [m for m in moves if move_type(m) == MOVE_EN_PASSANT]
            if passing:
                victim = engine.spec.piece_at(state.board, state.ep_victim)
                assert victim is not None and victim[0] != state.side_to_move
            # Lean towards double steps and their answers, so the walk meets the window.
            leaning = passing or [m for m in moves if _is_double_step(engine, state, m)]
            move = rng.choice(leaning if leaning and rng.random() < 0.6 else moves)
            answered += move_type(move) == MOVE_EN_PASSANT
            after = engine.apply(state, move)
            history.append((state, move, after))
            state = after
    assert answered > 0, "no walk took a pawn in passing"
    assert inside > 0, "no walk kept a window inside the turn that opened it"


def test_perft_matches_an_apply_built_count() -> None:
    """The core's perft from mid-turn, against a count built by `apply` (no terminal test:
    perft has none)."""
    engine = _engine(_spec({"moves": 2}))
    state = _play(engine, engine.initial_state(), "Pc2-c4")

    def count(position: GameState, depth: int) -> int:
        if depth == 0:
            return 1
        return sum(count(engine.apply(position, m), depth - 1)
                   for m in engine.legal_moves(position))

    assert engine.perft(state, 3) == count(state, 3)
