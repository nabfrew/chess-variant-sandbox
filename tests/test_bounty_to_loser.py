"""TC9c (`docs/plan-turncoat.md` §5.10): a bounty paid to the player who lost the piece.

`rules.purse.bounty_to: "loser"` sends a capture's bounty (`pieces[].bounty`) to the player
who owned the piece taken, into the purse (capped at `PURSE_MAX`) and the points tally alike;
`"taker"` is the off value, today's rule. A piece nobody owns pays nobody; an own piece taken
pays nothing. The core asks `Tables::bounty_payee` in `make` and `unmake`.

Hand-built specs, not chess: `tests/test_bounty_purse.py`'s 6x6 hunt with a purse and a market.
The oracle for each ply is this file's `_owed`: the bounties of the pieces each player lost on
the move, read off two boards (a declared test oracle, rule 3).
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import build_tables, canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import MOVE_BUY, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.schema import PURSE_MAX, validate_spec
from chess_variant_sandbox.spec.transform import settle_inert_rules
from chess_variant_sandbox.tournament.play import play_game

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
_PRICE = 3


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(start: int = 0, payee: str | None = "loser", **rules: Any) -> dict[str, Any]:
    """6x6: kings a1/a6, a rook (bounty 4) f1/f6, guards (bounty 1) b1-d1/b6-d6, a scout sold
    for 3 coins onto the first rank; a purse of ``start`` each, bounties paid to ``payee``."""
    purse: dict[str, Any] = {"start": start, "market": ["S"]}
    if payee is not None:
        purse["bounty_to"] = payee
    return {
        "schema_version": 31,
        "name": "the hunt, paid to the loser",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK], "bounty": 4},
            {"id": "G", "value": 2, "moves": [KING], "bounty": 1},
            {"id": "S", "value": 1, "moves": [KING], "price": _PRICE},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["b1", "c1", "d1"]},
            {"piece": "S", "squares": ["e1"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 60,
                  "purse": purse, **rules},
    }


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
              purse: tuple[int, int] | None = None) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement), purse=purse)


def _move(engine: Engine, state: GameState, origin: str, target: str) -> int:
    for move in engine.legal_moves(state):
        if (move & 0xFF, (move >> 8) & 0xFF) == (_sq(origin), _sq(target)):
            return move
    raise AssertionError(f"no move {origin}-{target}")


def _owed(engine: Engine, before: GameState, after: GameState) -> list[int]:
    """The oracle: per player, the bounties of their own pieces that left the board on this
    move (what the loser is owed), taken by the other player."""
    spec = engine.spec
    mover = before.side_to_move

    def lost(board: bytes, player: int) -> int:
        return sum(
            spec.bounty[spec.type_of(value)] for value in board
            if value and spec.owner_of(value) == player
        )

    return [
        0 if player == mover else lost(before.board, player) - lost(after.board, player)
        for player in (0, 1)
    ]


class TestTheKey:
    def test_the_loser_is_valid_and_the_taker_is_the_game_without_the_key(self) -> None:
        assert validate_spec(_spec()) == []
        written = _spec(payee="taker")
        assert validate_spec(written) == []
        assert spec_hash(written) == spec_hash(_spec(payee=None))
        assert spec_hash(_spec()) != spec_hash(_spec(payee=None))

    @pytest.mark.parametrize("bad", ["owner", True, 1, None])
    def test_only_the_two_payees(self, bad: Any) -> None:
        spec = _spec()
        spec["rules"]["purse"]["bounty_to"] = bad
        assert any("rules.purse.bounty_to" in error for error in validate_spec(spec))

    def test_the_loser_needs_a_bounty(self) -> None:
        spec = _spec()
        for piece in spec["pieces"]:
            piece.pop("bounty", None)
        assert any('"loser" with no pieces[].bounty' in e for e in validate_spec(spec))
        settled = copy.deepcopy(spec)
        settle_inert_rules(settled)
        assert "bounty_to" not in settled["rules"]["purse"]

    def test_the_loser_is_refused_beside_a_wild_bounty(self) -> None:
        spec = _spec()
        spec["pieces"].append({"id": "W", "value": 2, "moves": []})
        spec["rules"]["npcs"] = [{"piece": "W", "route": ["c3", "d3"], "bounty": 1}]
        assert any("could never be paid" in e for e in validate_spec(spec))

    def test_the_description_says_the_loser_is_paid(self) -> None:
        assert "lost" in describe_spec(_spec()).split("bount", 1)[1]


class TestTheTables:
    def test_the_flag_crosses_only_under_the_loser(self) -> None:
        assert build_tables(compile_spec(_spec()))["purse_to_loser"] is True
        assert build_tables(compile_spec(_spec(payee=None)))["purse_to_loser"] is False
        assert "purse_to_loser" not in canonical_tables(compile_spec(_spec(payee="taker")))
        assert canonical_tables(compile_spec(_spec(payee="taker"))) == canonical_tables(
            compile_spec(_spec(payee=None))
        )


class TestPaying:
    def test_the_side_that_loses_a_piece_is_paid_its_bounty(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0),
                                   "f5": ("R", 1)}, purse=(2, 3))
        after = engine.apply(state, _move(engine, state, "f1", "f5"))
        assert after.purse == (2, 7) and after.points == (0.0, 4.0)
        assert after.key == engine._compute_hash(after)

    def test_the_first_player_is_paid_when_the_second_takes(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f2": ("G", 0),
                                   "f6": ("R", 1)}, side=1, purse=(0, 0))
        after = engine.apply(state, _move(engine, state, "f6", "f2"))
        assert after.purse == (1, 0) and after.points == (1.0, 0.0)

    def test_the_taker_rule_is_unchanged(self) -> None:
        engine = Engine(_spec(payee=None))
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0),
                                   "f5": ("R", 1)}, purse=(2, 3))
        after = engine.apply(state, _move(engine, state, "f1", "f5"))
        assert after.purse == (6, 3)

    def test_the_cap_loses_what_passes_it(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0),
                                   "f5": ("R", 1)}, purse=(0, PURSE_MAX - 1))
        after = engine.apply(state, _move(engine, state, "f1", "f5"))
        assert after.purse == (0, PURSE_MAX)
        assert after.key == engine._compute_hash(after)

    def test_a_piece_nobody_owns_pays_nobody(self) -> None:
        spec = _spec()
        spec["pieces"].append({"id": "C", "value": 3, "moves": [KING], "bounty": 2})
        spec["setup"]["placements"].append({"piece": "C", "player": "nobody", "squares": ["c3"]})
        assert validate_spec(spec) == [], validate_spec(spec)
        engine = Engine(spec)
        state = engine.initial_state()
        for origin, target in (("c1", "c2"), ("c6", "c5"), ("c2", "c3")):
            state = engine.apply(state, _move(engine, state, origin, target))
        assert state.purse == (0, 0) and state.points == (0.0, 0.0)

    def test_the_purse_follows_the_oracle_over_twenty_walks(self) -> None:
        engine = Engine(_spec(start=1, max_plies=80))
        paid_somewhere = False
        for walk in range(20):
            rng = random.Random(walk)
            state = engine.initial_state()
            for _ in range(80):
                moves = engine.legal_moves(state)
                if engine.terminal(state, moves) is not None:
                    break
                move = rng.choice(moves)
                after = engine.apply(state, move)
                owed = _owed(engine, state, after)
                mover = state.side_to_move
                expected = [min(PURSE_MAX, state.purse[p] + owed[p]) for p in (0, 1)]
                if move_type(move) == MOVE_BUY:
                    expected[mover] -= _PRICE
                assert after.purse == tuple(expected)
                assert after.points == tuple(state.points[p] + owed[p] for p in (0, 1))
                assert after.key == engine._compute_hash(after)
                paid_somewhere |= any(owed)
                state = after
        assert paid_somewhere

    def test_unmake_puts_the_loser_s_coins_back(self) -> None:
        engine = Engine(_spec())
        state = _position(engine, {"a1": ("K", 0), "a6": ("K", 1), "f1": ("R", 0),
                                   "f5": ("R", 1), "c4": ("G", 1), "c2": ("G", 0)},
                          purse=(PURSE_MAX - 2, PURSE_MAX - 1))

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            if engine.terminal(position) is not None:
                return 0
            return sum(count(engine.apply(position, move), depth - 1)
                       for move in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_a_searching_game_replays(self) -> None:
        engine = Engine(_spec(start=2, max_plies=24))
        record = play_game(engine, ("minimax:nodes=300", "minimax:nodes=300"), 1)
        state = engine.initial_state()
        for move in record.moves:
            state = engine.apply(state, move)
            assert state.key == engine._compute_hash(state)
        outcome = engine.terminal(state)
        assert outcome is not None
        assert (outcome.winner, outcome.reason) == (record.winner, record.reason)
