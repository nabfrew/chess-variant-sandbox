"""SR2 (`docs/plan-schema-next.md` §2.2, rows L5 and L6): majority control and the purse in a
turn longer than one move.

Both were refused beside a long turn; the probe fed the core the pair past the validator and
found it already plays the ruled reading, so the lift is the validator's alone.

* **L5.** The holder of a nobody's piece is read off the board at the start of each move,
  which movegen does per move anyway: a first move that changes the count hands the piece
  to the player (or takes it away) for the second.
* **L6.** A buy, a hire or a bribe is a move; two in a turn are two moves, each paid from the
  purse as the move before left it.

The declared oracles are `tests/neutral_oracle.py` (who holds a neutral, what it may do, a
hire's price) and `tests/test_bounty_purse.py::_paid` (what a move pays). Hand-built specs,
not chess.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_BUY,
    MOVE_HIRE,
    move_foreign,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import PURSE_MAX, validate_spec
from chess_variant_sandbox.tournament.play import play_game
from tests import test_bounty_purse as purse_tests
from tests import test_neutral_pieces as neutral_tests
from tests.neutral_oracle import NeutralOracle

ROOT = Path(__file__).resolve().parents[1]
TURNS = [{"moves": 2}, {"moves": 3, "ramp": [1, 2]}]


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _count(engine: Engine, state: GameState, depth: int) -> int:
    """The perft by `apply`, a fresh position each node, against the core's make and unmake.
    No terminal test, as the core's perft has none: in a long turn a royal left attacked on a
    turn's last move can be taken on the next turn's first, and perft walks on past it."""
    if depth == 0:
        return 1
    return sum(_count(engine, engine.apply(state, m), depth - 1) for m in engine.legal_moves(state))


def _turn_start(engine: Engine, state: GameState) -> GameState:
    """A hand-built position at the start of a two-move turn (`make_state` builds one with a
    single move left)."""
    state = state.with_(remaining=2)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def _scrum(turn: dict[str, Any]) -> dict[str, Any]:
    spec = neutral_tests._walk_spec()
    spec["rules"]["turn"] = turn
    return spec


def _neutral_walk(spec: dict[str, Any], seed: int, plies: int, *, buy: bool = False) -> int:
    """A seeded walk, leaning toward held neutrals and hires: the core's legal list and
    holders are the oracle's at every ply, the key the full scan's. Returns the plies on
    which the side to move was the side that just moved (the second half of a turn) and a
    held neutral or a hire was on offer."""
    engine = _engine(spec)
    oracle = NeutralOracle(spec, engine)
    rng = random.Random(seed)
    state = engine.initial_state()
    width = engine.spec.width * engine.spec.height
    seen = 0
    last = None
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        core = sorted(engine.legal_moves(state))
        assert core == oracle.legal_moves(state), (seed, state.ply)
        for square in range(width):
            assert engine.controller_of(state, square) == oracle.holder(state, square)
        assert engine._compute_hash(state) == state.key
        special = [
            m for m in core if move_foreign(m) or (buy and move_type(m) == MOVE_HIRE)
        ]
        seen += bool(special) and last == state.side_to_move
        last = state.side_to_move
        move = rng.choice(special) if special and rng.random() < 0.5 else rng.choice(core)
        state = engine.apply(state, move)
    return seen


class TestTheValidator:
    @pytest.mark.parametrize("turn", TURNS)
    def test_majority_control_in_a_long_turn_is_valid(self, turn: dict[str, Any]) -> None:
        assert validate_spec(_scrum(turn)) == []

    @pytest.mark.parametrize("turn", TURNS)
    def test_a_purse_in_a_long_turn_is_valid(self, turn: dict[str, Any]) -> None:
        assert validate_spec(purse_tests._purse_spec(5, turn=turn)) == []


class TestTheHolderInALongTurn:
    def test_the_first_move_hands_the_neutral_to_the_mover_for_the_second(self) -> None:
        """6x6, two moves a turn. A guard each beside the condottiere on c3: a tie, so nobody
        holds it. White's first move brings a second guard next to it; still white to move,
        white now holds it."""
        engine = _engine(_scrum({"moves": 2}))
        placement = {
            "a1": ("K", 0), "f6": ("K", 1), "c3": ("C", 2),
            "b2": ("G", 0), "e2": ("G", 0), "d4": ("G", 1),
        }
        state = _turn_start(engine, neutral_tests._position(engine, placement))
        assert engine.controller_of(state, neutral_tests._sq("c3")) == -1
        assert neutral_tests._foreign(engine, state) == []
        step = next(m for m in engine.legal_moves(state)
                    if (m & 0xFF, (m >> 8) & 0xFF) == (neutral_tests._sq("e2"), neutral_tests._sq("d2")))
        after = engine.apply(state, step)
        assert after.side_to_move == 0
        assert engine.controller_of(after, neutral_tests._sq("c3")) == 0
        assert neutral_tests._foreign(engine, after)

    def test_the_first_move_can_take_it_away_for_the_second(self) -> None:
        """The mirror of the case above, for the second player: black holds it, then black's
        first move walks a guard away and leaves a tie."""
        engine = _engine(_scrum({"moves": 2}))
        placement = {
            "a1": ("K", 0), "f6": ("K", 1), "c4": ("C", 2),
            "b5": ("G", 1), "d5": ("G", 1), "c3": ("G", 0),
        }
        state = _turn_start(engine, neutral_tests._position(engine, placement, side=1))
        assert engine.controller_of(state, neutral_tests._sq("c4")) == 1
        away = next(m for m in engine.legal_moves(state)
                    if (m & 0xFF, (m >> 8) & 0xFF) == (neutral_tests._sq("d5"), neutral_tests._sq("e6")))
        after = engine.apply(state, away)
        assert after.side_to_move == 1
        assert engine.controller_of(after, neutral_tests._sq("c4")) == -1
        assert neutral_tests._foreign(engine, after) == []

    @pytest.mark.parametrize("turn", TURNS)
    def test_a_walk_agrees_with_the_oracle(self, turn: dict[str, Any]) -> None:
        assert sum(_neutral_walk(_scrum(turn), seed, 80) for seed in range(4)) > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("turn", TURNS)
    def test_twenty_walks_agree_with_the_oracle(self, turn: dict[str, Any]) -> None:
        assert sum(_neutral_walk(_scrum(turn), seed, 120) for seed in range(20)) > 50

    def test_unmake_puts_a_mid_turn_position_back(self) -> None:
        engine = _engine(_scrum({"moves": 2}))
        rng = random.Random(3)
        state = engine.initial_state()
        for _ in range(5):
            state = engine.apply(state, rng.choice(engine.legal_moves(state)))
        assert engine.perft(state, 3) == _count(engine, state, 3)

    def test_a_searching_game_replays(self) -> None:
        engine = _engine(_scrum({"moves": 2, "ramp": [1]}))
        record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), 2)
        state = engine.initial_state()
        for move in record.moves:
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)


class TestThePurseInALongTurn:
    @pytest.mark.parametrize("turn", TURNS)
    def test_every_move_pays_and_spends_as_the_move_before_left_it(self, turn: dict[str, Any]) -> None:
        """Walks under a market: after every move, the mover's purse is the oracle's (paid
        bounties in, capped, a buy's price out); two buys in one turn are seen (the market
        sells on the second rank, empty from the start)."""
        spec = purse_tests._purse_spec(90, max_plies=80, turn=turn)
        spec["rules"]["purse"]["region"] = ["a2", "b2", "c2", "d2", "e2", "f2"]
        engine = _engine(spec)
        twice = 0
        for walk in range(6):
            rng = random.Random(walk)
            state = engine.initial_state()
            bought = False
            for _ in range(80):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                buys = [m for m in moves if move_type(m) == MOVE_BUY]
                move = rng.choice(buys) if buys and rng.random() < 0.5 else rng.choice(moves)
                after = engine.apply(state, move)
                mover = state.side_to_move
                expected = list(state.purse)
                expected[mover] = min(PURSE_MAX, expected[mover] + purse_tests._paid(engine, state, after))
                if move_type(move) == MOVE_BUY:
                    expected[mover] -= purse_tests._PRICE
                    twice += bought
                bought = move_type(move) == MOVE_BUY and after.side_to_move == mover
                assert after.purse == tuple(expected)
                assert after.key == engine._compute_hash(after)
                state = after
        assert twice > 0

    def test_hires_and_bribes_agree_with_the_oracle(self) -> None:
        spec = json.loads((ROOT / "tests" / "_purse_hire.json").read_text(encoding="utf-8"))
        spec["rules"]["turn"] = {"moves": 2}
        spec["rules"]["purse"]["start"] = 20
        assert sum(_neutral_walk(spec, seed, 60, buy=True) for seed in range(6)) > 0

    def test_unmake_puts_the_purse_back_mid_turn(self) -> None:
        engine = _engine(purse_tests._purse_spec(0, turn={"moves": 2}))
        state = purse_tests._position(
            engine,
            {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0), "f5": ("R", 1),
             "c4": ("G", 1), "c2": ("G", 0)},
            purse=(PURSE_MAX - 2, PURSE_MAX - 1),
        )
        assert engine.perft(state, 3) == _count(engine, state, 3)

    def test_a_searching_game_replays(self) -> None:
        engine = _engine(purse_tests._purse_spec(97, max_plies=24, turn={"moves": 2, "ramp": [1]}))
        record = play_game(engine, ("minimax:nodes=200", "minimax:nodes=200"), 1)
        state = engine.initial_state()
        for move in record.moves:
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)
