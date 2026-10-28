"""SR2 (`docs/plan-schema-next.md` §2.5, rows H1, H2 and H6): what a side has lost, and what a
blast puts in a hand.

* **H1, H2.** `promotion.only_lost` reads the board (*"fewer of that type on the board than it
  started with"*), and the core counts the board. So beside a hand that fills from captures a
  piece in a hand is lost (it is not on the board), and beside a circe strike a piece reborn on
  its home square is not (it is). Both pairs were refused; the probe found the core already
  counts the board, both seats, so the lift is the validator's.
* **H6.** `drops.either_face` beside `explode_on_capture`: a blast's victims leave the board
  without entering a hand, and only the piece the capture took goes to the mover's hand
  (demoted), where it may be dropped as either face. The probe found the core already plays
  this, so the lift is the validator's.

The oracle for H6 is this file's `_blast_walk`: after a capture the mover's hand gained the
taken piece's demoted type and nothing else (a declared test oracle, rule 3). Hand-built
specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import validate_spec

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
PAWN = [{"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}]
RANKS_2_TO_5 = [f"{f}{r}" for r in range(2, 6) for f in "abcdef"]


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return f"{chr(ord('a') + square % WIDTH)}{square // WIDTH + 1}"


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0,
              hands: dict[int, dict[str, int]] | None = None) -> GameState:
    state = engine.make_state(placement, side_to_move=side, moved=list(placement))
    held = [list(hand) for hand in engine.initial_state().hands]
    for player, pieces in (hands or {}).items():
        for pid, n in pieces.items():
            held[player][engine.spec.piece_ids.index(pid)] = n
    return dataclasses.replace(state, hands=(tuple(held[0]), tuple(held[1])))


def _mirror(placement: dict[str, tuple[str, int]]) -> dict[str, tuple[str, int]]:
    """The same position from the second player's seat: ranks reflected, owners swapped."""
    return {f"{name[0]}{WIDTH + 1 - int(name[1:])}": (pid, 1 - player)
            for name, (pid, player) in placement.items()}


# --------------------------------------------------------------------------- only_lost


def _only_lost(**rules: Any) -> dict[str, Any]:
    """6x6: a king, a rook, a guard and two pawns a side; a pawn promotes on the last rank to a
    rook or a guard, only into what its side has lost."""
    return copy.deepcopy({
        "schema_version": 31, "name": "lost and found", "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "P", "value": 1, "moves": PAWN},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["b1"]}, {"piece": "P", "squares": ["c2", "d2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80,
                  "promotion": {"piece": "P", "rank": "last", "to": ["R", "G"], "only_lost": True},
                  **rules},
    })


HAND = {"drops": {"from_captures": True, "region": {"P": RANKS_2_TO_5}}}
CIRCE = {"strike": {"effect": "capture", "position": "home", "immune": ["K"]}}


def _offered(engine: Engine, state: GameState, origin: str) -> set[str]:
    """The targets the pawn on ``origin`` is offered."""
    return {
        engine.describe_move(state, m).rsplit("=", 1)[1]
        for m in engine.legal_moves(state)
        if (m & 0xFF) == _sq(origin) and "=" in engine.describe_move(state, m)
    }


class TestOnlyLostBesideAHand:
    def test_it_is_valid(self) -> None:
        assert validate_spec(_only_lost(**HAND)) == []

    @pytest.mark.parametrize("seat", [0, 1])
    @pytest.mark.parametrize("on_board, in_hand, offered", [
        (False, 0, {"R"}),   # the rook is lost (the guard on f4 stands)
        (True, 0, set()),    # the rook stands
        (False, 1, {"R"}),   # a rook in hand is not on the board: still lost
        (True, 1, set()),    # one on the board is the opening count, in hand or not
    ])
    def test_a_piece_in_hand_is_lost(self, seat: int, on_board: bool, in_hand: int,
                                     offered: set[str]) -> None:
        engine = _engine(_only_lost(**HAND))
        placement = {"a1": ("K", 0), "a6": ("K", 1), "c5": ("P", 0), "f6": ("R", 1),
                     "e6": ("G", 1), "f4": ("G", 0)}
        if on_board:
            placement["e1"] = ("R", 0)
        promoting = "c5"
        if seat:
            placement, promoting = _mirror(placement), "c2"
        state = _position(engine, placement, side=seat, hands={seat: {"R": in_hand}})
        assert _offered(engine, state, promoting) == offered

    def test_a_dropped_piece_is_on_the_board_again(self) -> None:
        engine = _engine(_only_lost(**HAND))
        placement = {"a1": ("K", 0), "a6": ("K", 1), "c5": ("P", 0), "f6": ("R", 1),
                     "e6": ("G", 1), "f4": ("G", 0)}
        state = _position(engine, placement, hands={0: {"R": 1}})
        assert "R" in _offered(engine, state, "c5")
        drop = next(m for m in engine.legal_moves(state)
                    if move_type(m) == MOVE_DROP and (m >> 8) & 0xFF == _sq("e3"))
        after = engine.apply(state, drop)
        after = engine.apply(after, next(m for m in engine.legal_moves(after)
                                         if engine.describe_move(after, m).startswith("K")))
        assert "R" not in _offered(engine, after, "c5")


class TestOnlyLostBesideCirce:
    def test_it_is_valid(self) -> None:
        assert validate_spec(_only_lost(**CIRCE)) == []

    @pytest.mark.parametrize("seat", [0, 1])
    def test_a_reborn_piece_is_not_lost_and_a_removed_one_is(self, seat: int) -> None:
        """The opponent takes the rook on e3: reborn on f1, its home, it is not lost; with a
        guard on f1 it is removed, and it is."""
        engine = _engine(_only_lost(**CIRCE))
        base = {"a1": ("K", 0), "a6": ("K", 1), "c5": ("P", 0), "e3": ("R", 0), "e6": ("R", 1),
                "b1": ("G", 0)}
        for blocked, offered in ((False, set()), (True, {"R"})):
            placement = {**base, "f1": ("G", 0)} if blocked else dict(base)
            taker, taken, promoting = "e6", "e3", "c5"
            if seat:
                placement = _mirror(placement)
                taker, taken, promoting = "e1", "e4", "c2"
            state = _position(engine, placement, side=1 - seat)
            move = next(m for m in engine.legal_moves(state)
                        if (m & 0xFF, (m >> 8) & 0xFF) == (_sq(taker), _sq(taken)))
            text = engine.describe_move(state, move)
            assert ("removed" in text) == blocked, text
            after = engine.apply(state, move)
            assert _offered(engine, after, promoting) == offered


# --------------------------------------------------------------------------- either_face


def _faces(**explosion: Any) -> dict[str, Any]:
    """6x6: kings, a rook, two guards a side, and `H`, a guard's promoted face."""
    return copy.deepcopy({
        "schema_version": 31, "name": "two-faced atoms", "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "H", "value": 4, "moves": [ROOK, KING], "promoted_from": "G"},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["b2", "c2"]}, {"piece": "H", "squares": ["e2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80,
                  "drops": {"from_captures": True, "either_face": True},
                  "explode_on_capture": explosion or {"radius": 1, "immune": ["K"]}},
    })


def _blast_walk(engine: Engine, seeds: range, plies: int) -> int:
    """After every capture, the mover's hand gained the taken piece's demoted type and nothing
    else: the blast's victims, the capturer and the mover's own pieces enter no hand."""
    spec = engine.spec
    blasts = 0
    for seed in seeds:
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ in range(plies):
            moves = engine.legal_moves(state)
            if engine.terminal(state, moves) is not None:
                break
            takes = [m for m in moves
                     if move_type(m) != MOVE_DROP and state.board[(m >> 8) & 0xFF]]
            move = rng.choice(takes) if takes and rng.random() < 0.6 else rng.choice(moves)
            after = engine.apply(state, move)
            mover = state.side_to_move
            if move_type(move) != MOVE_DROP:
                expected = list(state.hands[mover])
                taken = state.board[(move >> 8) & 0xFF]
                if taken:
                    expected[spec.demotions[spec.type_of(taken)]] += 1
                    gone = sum(1 for a, b in zip(state.board, after.board, strict=True) if a and not b)
                    blasts += gone > 2
                assert tuple(after.hands[mover]) == tuple(expected), (seed, state.ply)
                assert after.hands[1 - mover] == state.hands[1 - mover]
            assert after.key == engine._compute_hash(after)
            state = after
    return blasts


class TestEitherFaceBesideABlast:
    def test_it_is_valid(self) -> None:
        assert validate_spec(_faces()) == []

    @pytest.mark.parametrize("seat", [0, 1])
    def test_the_victims_skip_the_hand_and_the_taken_piece_comes_back_either_face(
        self, seat: int,
    ) -> None:
        """The rook takes a guard on c5 beside an enemy champion and a guard of its own: the
        champion, the own guard and the rook leave the board for no hand; the guard goes to
        the hand, and comes back as a guard or a champion."""
        engine = _engine(_faces())
        placement = {"a1": ("K", 0), "f6": ("K", 1), "c3": ("R", 0), "c5": ("G", 1),
                     "d5": ("H", 1), "b6": ("G", 0)}
        origin, target = "c3", "c5"
        if seat:
            placement = _mirror(placement)
            origin, target = "c4", "c2"
        state = _position(engine, placement, side=seat)
        move = next(m for m in engine.legal_moves(state)
                    if (m & 0xFF, (m >> 8) & 0xFF) == (_sq(origin), _sq(target)))
        after = engine.apply(state, move)
        ids = engine.spec.piece_ids
        assert sorted(ids[engine.spec.type_of(v)] for v in after.board if v) == ["K", "K"]
        assert {ids[t]: n for t, n in enumerate(after.hands[seat]) if n} == {"G": 1}
        assert not any(after.hands[1 - seat])
        mine = dataclasses.replace(after, side_to_move=seat)
        dropped = {engine.describe_move(mine, m)[0]
                   for m in engine.legal_moves(mine) if move_type(m) == MOVE_DROP}
        assert dropped == {"G", "H"}

    def test_a_walk_deposits_the_taken_piece_alone(self) -> None:
        assert _blast_walk(_engine(_faces()), range(8), 80) > 0

    @pytest.mark.slow
    def test_twenty_walks(self) -> None:
        assert _blast_walk(_engine(_faces()), range(20), 160) > 20
