"""SR2 (`docs/plan-schema-next.md` §2.4, rows T2 and T5): a strike or a contact that removes its
victim, beside a hand that fills from captures.

`strike` and `contacts` were refused beside `drops` because a struck piece cannot both stand
on the board and sit in a hand. The ruled reading lifts the part with no contradiction: a
strike that only removes (effect `capture`, position `stay`: a shot, `schema.strike_only_removes`)
and a contact table whose every link removes its target (no `onto_to`,
`schema.contacts_only_remove`) send the victim to the mover's hand as any capture does,
demoted to its `promoted_from`. The probe fed the core the pair past the validator and found it
already plays that, both seats, so the lift is the validator's. The kept rows (a transforming
strike, circe, Andernach, a link that keeps its target) stay refused by name.

The oracle for what a move deposits is this file's `_hand_walk`: the opponent's pieces that left
the board on the move, by their demoted type, go to the mover's hand and nowhere else (a declared
test oracle, rule 3). Hand-built specs, not chess, but for the archers' fixture.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
BISHOP = {"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}
SHOT = {"effect": "capture", "position": "stay", "immune": ["K"], "by": ["B"]}
REMOVING_LINKS = {"links": [{"mover": "B", "onto": "G", "mover_to": "R"},
                            {"mover": "R", "onto": "B", "mover_to": "G"}], "immune": ["K"]}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(**rules: Any) -> dict[str, Any]:
    """6x6: kings a1/a6 (mirrored), a rook, a bishop, two guards, and a champion `H` that is a
    promoted guard (it goes to a hand as a guard)."""
    return copy.deepcopy({
        "schema_version": 31, "name": "the shooting gallery",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "B", "value": 3, "moves": [BISHOP]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "H", "value": 4, "moves": [ROOK, KING], "promoted_from": "G"},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["f1"]},
            {"piece": "B", "squares": ["c1"]}, {"piece": "G", "squares": ["b2", "c2", "d2", "e2"]},
        ]},
        "rules": {"win": "checkmate", "stalemate": "draw", "max_plies": 80,
                  "drops": {"from_captures": True}, **rules},
    })


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(spec)


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int = 0) -> GameState:
    state = engine.make_state(placement, side_to_move=side, moved=list(placement))
    return dataclasses.replace(state, hands=engine.initial_state().hands)


def _play(engine: Engine, state: GameState, origin: str, target: str) -> tuple[str, GameState]:
    for move in engine.legal_moves(state):
        if (move & 0xFF, (move >> 8) & 0xFF) == (_sq(origin), _sq(target)):
            return engine.describe_move(state, move), engine.apply(state, move)
    raise AssertionError(f"no move {origin}-{target}")


def _held(engine: Engine, state: GameState, player: int) -> dict[str, int]:
    return {engine.spec.piece_ids[t]: n for t, n in enumerate(state.hands[player]) if n}


def _count(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(_count(engine, engine.apply(state, m), depth - 1) for m in engine.legal_moves(state))


def _hand_walk(engine: Engine, seeds: range, plies: int) -> int:
    """Seeded walks leaning toward captures; after every move that is not a drop, the mover's
    hand gained exactly the opponent's pieces that left the board, by demoted type, and the
    other hand did not move. Returns the pieces deposited."""
    spec = engine.spec
    deposited = 0
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

            def theirs(board: bytes, mover: int = mover) -> Counter[int]:
                return Counter(spec.demotions[spec.type_of(v)] for v in board
                               if v and spec.owner_of(v) == 1 - mover)

            if move_type(move) != MOVE_DROP:
                expected = list(state.hands[mover])
                for piece_type, n in (theirs(state.board) - theirs(after.board)).items():
                    expected[piece_type] += n
                    deposited += n
                assert tuple(after.hands[mover]) == tuple(expected), (seed, state.ply)
                assert after.hands[1 - mover] == state.hands[1 - mover]
            assert after.key == engine._compute_hash(after)
            state = after
    return deposited


# --------------------------------------------------------------------------- the validator


class TestTheValidator:
    @pytest.mark.parametrize("strike", [
        SHOT,
        {"effect": "capture", "position": "stay", "immune": ["K"]},
        # Under "capture" a chain is never read: the shot still removes.
        {"effect": "capture", "position": "stay", "immune": ["K"], "chain": [{"from": "R", "to": "G"}]},
    ])
    def test_a_shot_beside_a_hand_is_valid(self, strike: dict[str, Any]) -> None:
        assert validate_spec(_spec(strike=strike)) == []

    @pytest.mark.parametrize("position", ["advance", "stay", "swap"])
    def test_removing_links_beside_a_hand_are_valid(self, position: str) -> None:
        assert validate_spec(_spec(contacts={**REMOVING_LINKS, "position": position})) == []

    def test_an_own_removing_link_is_valid(self) -> None:
        own = {"links": [{"mover": "B", "onto": "G", "mover_to": "R", "own": True}],
               "position": "stay", "immune": ["K"]}
        assert validate_spec(_spec(contacts=own)) == []

    @pytest.mark.parametrize("strike", [
        {"effect": "demote", "position": "stay", "immune": ["K"], "chain": [{"from": "R", "to": "G"}]},
        {"effect": "capture", "position": "home", "immune": ["K"]},
        {"effect": "capture", "switches_sides": ["B"], "immune": ["K"]},
    ])
    def test_a_strike_that_keeps_its_victim_is_still_refused(self, strike: dict[str, Any]) -> None:
        assert "T2" in refused_ids(validate_spec(_spec(strike=strike)))

    def test_a_link_that_keeps_its_target_is_still_refused(self) -> None:
        keeps = {"links": [{"mover": "B", "onto": "G", "onto_to": "R"}],
                 "position": "stay", "immune": ["K"]}
        assert "T5" in refused_ids(validate_spec(_spec(contacts=keeps)))

    def test_a_shot_beside_either_face_is_still_refused(self) -> None:
        """H4 stays: either_face is refused beside a strike whose position is not advance."""
        spec = _spec(strike=SHOT)
        spec["rules"]["drops"]["either_face"] = True
        assert any("either_face beside a strike" in e for e in validate_spec(spec))


# --------------------------------------------------------------------------- the hand


class TestTheHand:
    @pytest.mark.parametrize("side", [0, 1])
    def test_a_shot_sends_its_victim_to_the_shooters_hand(self, side: int) -> None:
        engine = _engine(_spec(strike=SHOT))
        state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c3": ("B", side),
                                   "e5": ("G", 1 - side), "f3": ("R", 1 - side)}, side)
        text, after = _play(engine, state, "c3", "e5")
        assert "stays" in text
        assert after.board[_sq("c3")] and not after.board[_sq("e5")]
        assert _held(engine, after, side) == {"G": 1} and _held(engine, after, 1 - side) == {}

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_shot_promoted_piece_goes_to_the_hand_as_what_it_was(self, side: int) -> None:
        engine = _engine(_spec(strike=SHOT))
        state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c3": ("B", side),
                                   "e5": ("H", 1 - side)}, side)
        _, after = _play(engine, state, "c3", "e5")
        assert _held(engine, after, side) == {"G": 1}

    @pytest.mark.parametrize("side", [0, 1])
    def test_a_removing_link_sends_its_target_to_the_hand(self, side: int) -> None:
        engine = _engine(_spec(contacts={**REMOVING_LINKS, "position": "stay"}))
        state = _position(engine, {"a1": ("K", side), "f6": ("K", 1 - side), "c3": ("B", side),
                                   "e5": ("G", 1 - side)}, side)
        _, after = _play(engine, state, "c3", "e5")
        assert engine.spec.piece_ids[engine.spec.type_of(after.board[_sq("c3")])] == "R"
        assert not after.board[_sq("e5")]
        assert _held(engine, after, side) == {"G": 1} and _held(engine, after, 1 - side) == {}

    def test_an_own_removing_link_puts_the_piece_in_its_owners_hand(self) -> None:
        own = {"links": [{"mover": "B", "onto": "G", "mover_to": "R", "own": True}],
               "position": "stay", "immune": ["K"]}
        engine = _engine(_spec(contacts=own))
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c3": ("B", 0), "e5": ("G", 0)})
        _, after = _play(engine, state, "c3", "e5")
        assert _held(engine, after, 0) == {"G": 1} and _held(engine, after, 1) == {}

    def test_the_archers_walk_deposits_what_left_the_board(self) -> None:
        spec = json.loads((ROOT / "tests" / "_strike_archers.json").read_text(encoding="utf-8"))
        spec["rules"]["drops"] = {"from_captures": True}
        assert _hand_walk(_engine(spec), range(4), 100) > 0

    def test_the_contacts_walk_deposits_what_left_the_board(self) -> None:
        assert _hand_walk(_engine(_spec(contacts={**REMOVING_LINKS, "position": "stay"})),
                          range(6), 80) > 0

    @pytest.mark.slow
    def test_twenty_archers_walks(self) -> None:
        spec = json.loads((ROOT / "tests" / "_strike_archers.json").read_text(encoding="utf-8"))
        spec["rules"]["drops"] = {"from_captures": True}
        assert _hand_walk(_engine(spec), range(20), 160) > 200

    def test_unmake_puts_the_hand_back(self) -> None:
        engine = _engine(_spec(strike=SHOT))
        state = _position(engine, {"a1": ("K", 0), "f6": ("K", 1), "c3": ("B", 0), "e5": ("G", 1),
                                   "b4": ("H", 1), "d1": ("B", 1), "e2": ("G", 0)})
        assert engine.perft(state, 3) == _count(engine, state, 3)
