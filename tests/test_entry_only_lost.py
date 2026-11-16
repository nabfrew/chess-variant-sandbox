"""SR8a (`docs/plan-schema-next.md` §2.5, row H3): `promotion.only_lost` beside `promote_on_entry`.

A piece entering play inside its zone (dropped there, bought, gated, or left standing there by a
strike or a link) has no move in which to choose, so `promote_on_entry` gives it the first target
its promotion lists. Beside `only_lost` (H3, ruled 2026-10-27) it takes the first target **still
offered** -- one its side has fewer of on the board than it started with, in the order of
`promotion.to` (or `to_by_player` for its player) -- and none if none is: it enters as itself.
The count is the board before the move, as a moving promotion's is (`Game::promotion_offered`),
and for a round the board the round begins on (`Game::tally_entries`). So a guard a strike turns
into a pawn on its own last rank still counts as standing: the side has lost no guard yet.

Rounds are checked against the declared oracle (`tests/simultaneous_oracle.py::_arriving`).
Hand-built specs, not chess.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest
from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, decode_move, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
PAWN = [{"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"}]


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _mirror(name: str) -> str:
    return f"{name[0]}{WIDTH + 1 - int(name[1:])}"


def _seat(placement: dict[str, tuple[str, int]], seat: int) -> dict[str, tuple[str, int]]:
    """The same position from the second player's seat: ranks reflected, owners swapped."""
    if not seat:
        return dict(placement)
    return {_mirror(name): (pid, 1 - player) for name, (pid, player) in placement.items()}


def _spec(**rules: Any) -> dict[str, Any]:
    """6x6: a king, a rook, a guard and two pawns a side; a pawn promotes on the last rank to a
    rook or a guard, only into what its side has lost, and on entering its zone too."""
    return copy.deepcopy({
        "schema_version": 31, "name": "lost on entry", "board": {"width": WIDTH, "height": WIDTH},
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
                  "promote_on_entry": True, **rules},
    })


HAND = {"drops": {"from_captures": True}}
# A guard taken is struck into a pawn where it stands, the taker staying put.
STRIKE = {"strike": {"effect": "demote", "chain": [{"from": "G", "to": "P"}],
                     "position": "stay", "immune": ["K"]}}


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
    state = dataclasses.replace(state, hands=(tuple(held[0]), tuple(held[1])))
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def _at(engine: Engine, state: GameState, name: str) -> tuple[str, int] | None:
    value = state.board[_sq(name)]
    if value == 0:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def _drop(engine: Engine, state: GameState, square: str, moves: list[int] | None = None) -> int:
    return next(
        m for m in (engine.legal_moves(state) if moves is None else moves)
        if move_type(m) == MOVE_DROP and decode_move(m)[1] == _sq(square)
    )


# Which of the dropper's rook and guard stand, and what a pawn dropped on its last rank becomes.
CASES = [
    ((False, True), "R"),   # the rook is lost: the first target
    ((True, False), "G"),   # the rook stands, the guard is lost: the second
    ((True, True), "P"),    # nothing lost: it enters as itself
    ((False, False), "R"),  # both lost: the first listed
]


def _army(rook: bool, guard: bool) -> dict[str, tuple[str, int]]:
    placement = {"a1": ("K", 0), "f4": ("K", 1)}
    if rook:
        placement["e1"] = ("R", 0)
    if guard:
        placement["c1"] = ("G", 0)
    return placement


class TestTheLift:
    @pytest.mark.parametrize("rules", [HAND, STRIKE], ids=["hand", "strike"])
    def test_the_pair_is_valid(self, rules: dict[str, Any]) -> None:
        errors = validate_spec(_spec(**rules))
        assert "H3" not in refused_ids(errors), errors
        assert errors == [], errors


class TestADroppedPiece:
    @pytest.mark.parametrize("seat", [0, 1])
    @pytest.mark.parametrize(("standing", "becomes"), CASES)
    def test_it_takes_the_first_target_still_offered(
        self, seat: int, standing: tuple[bool, bool], becomes: str
    ) -> None:
        engine = _engine(_spec(**HAND))
        state = _position(engine, _seat(_army(*standing), seat), side=seat,
                          hands={seat: {"P": 1}})
        square = _mirror("b6") if seat else "b6"
        after = engine.apply(state, _drop(engine, state, square))
        assert _at(engine, after, square) == (becomes, seat)
        assert after.key == engine._compute_hash(after)
        # The hand gives up the pawn whatever it became.
        assert after.hands[seat][engine.spec.piece_ids.index("P")] == 0

    def test_a_drop_off_the_zone_is_a_pawn(self) -> None:
        engine = _engine(_spec(**HAND))
        state = _position(engine, _army(False, True), hands={0: {"P": 1}})
        after = engine.apply(state, _drop(engine, state, "b5"))
        assert _at(engine, after, "b5") == ("P", 0)


class TestAStruckPiece:
    @pytest.mark.parametrize("seat", [0, 1])
    @pytest.mark.parametrize(("rook", "becomes"), [(True, "P"), (False, "R")])
    def test_it_counts_the_board_before_the_move(
        self, seat: int, rook: bool, becomes: str
    ) -> None:
        """The opponent's rook on d4 takes the guard on d6, the guard's own last rank: struck
        into a pawn there, it enters its zone. Its own guard still stood before the move, so
        the guard is not offered; the rook is, where the side has lost it."""
        engine = _engine(_spec(**STRIKE))
        placement = {"a6": ("K", 0), "e2": ("K", 1), "d6": ("G", 0), "d4": ("R", 1)}
        if rook:
            placement["f6"] = ("R", 0)
        placement = _seat(placement, seat)
        taker, taken = ("d4", "d6") if not seat else (_mirror("d4"), _mirror("d6"))
        state = _position(engine, placement, side=1 - seat)
        move = next(
            m for m in engine.legal_moves(state)
            if decode_move(m)[:2] == (_sq(taker), _sq(taken))
        )
        after = engine.apply(state, move)
        assert _at(engine, after, taker) == ("R", 1 - seat)
        assert _at(engine, after, taken) == (becomes, seat)
        assert after.key == engine._compute_hash(after)


class TestInRounds:
    @pytest.mark.parametrize(("standing", "becomes"), CASES)
    def test_both_drops_read_the_rounds_board(
        self, standing: tuple[bool, bool], becomes: str
    ) -> None:
        """Both players drop a pawn on their last rank in one round, each army as `standing`
        says: each becomes what the round's start board offers, as the oracle says."""
        spec = _spec(**HAND)
        spec["rules"]["win"] = "royal_capture"
        spec["rules"]["simultaneous"] = {}
        engine = _engine(spec)
        army = _army(*standing)
        placement = {**army, **_seat(army, 1)}
        placement.pop("f4")
        placement.pop(_mirror("f4"))
        state = _position(engine, placement, hands={0: {"P": 1}, 1: {"P": 1}})
        first = _drop(engine, state, "b6", engine.legal_moves_for(state, 0))
        second = _drop(engine, state, "b1", engine.legal_moves_for(state, 1))
        after, _, _ = engine.apply_round_with_phase(state, first, second)
        assert after == resolve_round(engine, state, first, second)
        assert after.key == engine._compute_hash(after)
        assert _at(engine, after, "b6") == (becomes, 0)
        assert _at(engine, after, "b1") == (becomes, 1)


class TestTheLoopReachesIt:
    """The second of the two edits by which the loop reaches the pair (no stored game is one
    edit from it: `tests/test_operators.py::_CANNOT_ENTER`)."""

    @pytest.mark.parametrize(("operator", "drop"), [
        ("promote-what-arrives", "promote_on_entry"), ("promote-only-lost", "only_lost"),
    ])
    def test_each_half_enters_it_from_a_game_with_the_other(self, operator: str, drop: str) -> None:
        import random

        from chess_variant_sandbox.loop.operators import OPERATORS
        from tests.feature_inventory import _schema_one_hundred_eighteen_uses

        apply = next(op for op in OPERATORS if op.name == operator)
        parent = _spec(**HAND)
        if drop == "promote_on_entry":
            del parent["rules"]["promote_on_entry"]
        else:
            del parent["rules"]["promotion"]["only_lost"]
        assert validate_spec(parent) == [], validate_spec(parent)
        assert not _schema_one_hundred_eighteen_uses(parent)
        for seed in range(20):
            child = copy.deepcopy(parent)
            if apply.apply(child, random.Random(seed)) is None:
                continue
            assert validate_spec(child) == [], validate_spec(child)
            if _schema_one_hundred_eighteen_uses(child):
                return
        pytest.fail(f"{operator} never entered the pair in 20 seeds")
