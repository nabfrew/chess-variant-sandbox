"""SR5j (`docs/plan-schema-next.md` §2.11 row O12): a held piece's blast is the mover's.

The ruling (2026-10-28): a blast set off by a held piece (a piece nobody owns, moved by the
player with the majority beside it) is charged to the mover; a blast that would take the mover's
royal is illegal as any.

The probe (the pair fed past the validator on SR5h's build `619bf98c974c66a0`; a 6x6 board, a
king's-step blast, a condottiere nobody owns held by the side to move): the core already played
it as ruled. The legality filter's `player` is the side to move, which a held move's mover is,
and `blast_sets_off` and the filter's blast arm read the capturer's type off the board, nobody's
or not, so a held capture next to the mover's own royal was not offered under checkmate, and one
next to the other royal ended the game for the mover (`royal_destroyed`). No core change.

Hand-built 6x6 specs, not chess. Every walk checks the incremental key against `_compute_hash`
each ply, and `perft` against an `apply` count puts every cleared piece back.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import WILD, compile_spec
from chess_variant_sandbox.engine.moves import MOVE_FOREIGN, move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_eighty_seven_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _spec(win: str = "checkmate") -> dict[str, Any]:
    return copy.deepcopy({
        "name": "the held blast (SR5j test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "N", "name": "guard", "value": 3, "moves": [KING]},
            # A stone attacks nothing, so a royal beside it is not in check.
            {"id": "T", "name": "stone", "value": 1, "moves": []},
            {"id": "C", "name": "condottiere", "value": 3, "moves": [KING],
             "control": "majority"},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "N", "squares": ["b1"]},
            {"piece": "T", "squares": ["f1"]},
            {"piece": "C", "player": "nobody", "squares": ["c3"]},
        ]},
        "rules": {"win": win, "stalemate": "draw", "max_plies": 80,
                  "explode_on_capture": {"radius": 1}},
    })


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _at(engine: Engine, state: GameState, square: str) -> tuple[str, int] | None:
    value = state.board[_sq(square)]
    if not value:
        return None
    spec = engine.spec
    return spec.piece_ids[spec.type_of(value)], spec.owner_of(value)


def _position(engine: Engine, side: int, mine: str, theirs: str) -> GameState:
    """The condottiere on c3, held by ``side`` (two of its guards on b2 and b3), a stone of the
    other side on d4; ``side``'s royal on ``mine`` and the other's on ``theirs``."""
    placement = {
        "b2": ("N", side), "b3": ("N", side), "c3": ("C", WILD), "d4": ("T", 1 - side),
        mine: ("K", side), theirs: ("K", 1 - side),
    }
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


def _held_capture(engine: Engine, state: GameState) -> list[int]:
    return [
        m for m in engine.legal_moves(state)
        if (move_from(m), move_to(m)) == (_sq("c3"), _sq("d4"))
    ]


# --------------------------------------------------------------------------- the acceptance


class TestAHeldBlastIsTheMovers:
    @pytest.mark.parametrize("side", [0, 1])
    @pytest.mark.parametrize("win", ["checkmate", "royal_capture"])
    def test_one_that_takes_the_other_royal_wins_for_the_mover(self, side: int, win: str) -> None:
        engine = _engine(_spec(win))
        state = _position(engine, side, mine="a1", theirs="e5")
        (move,) = _held_capture(engine, state)
        assert move & MOVE_FOREIGN
        after = engine.apply(state, move)
        # The condottiere, the stone and the other royal are gone; the guards stand.
        for square in ("c3", "d4", "e5"):
            assert _at(engine, after, square) is None
        assert _at(engine, after, "b2") == ("N", side)
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == side
        assert after.key == engine._compute_hash(after)
        # Undo exact: the position played from is the position it was.
        assert engine.apply(state, move).board == after.board

    @pytest.mark.parametrize("side", [0, 1])
    def test_one_that_would_take_the_movers_royal_is_not_offered(self, side: int) -> None:
        engine = _engine(_spec("checkmate"))
        barred = _position(engine, side, mine="e5", theirs="a6" if side == 0 else "a1")
        assert _held_capture(engine, barred) == []
        # The condottiere's quiet moves stand; only the capture goes.
        assert any(move_from(m) == _sq("c3") for m in engine.legal_moves(barred))

    @pytest.mark.parametrize("side", [0, 1])
    def test_under_royal_capture_it_is_a_move_that_loses(self, side: int) -> None:
        """Where a royal may be lost, a move that loses it is a move, held or not: the reading
        a player's own piece has beside a blast."""
        engine = _engine(_spec("royal_capture"))
        state = _position(engine, side, mine="e5", theirs="a6" if side == 0 else "a1")
        (move,) = _held_capture(engine, state)
        outcome = engine.terminal(engine.apply(state, move))
        assert outcome is not None and outcome.winner == 1 - side

    @pytest.mark.parametrize("side", [0, 1])
    def test_the_opponents_crowd_is_not_the_movers(self, side: int) -> None:
        """The other side holds the condottiere: the side to move has no held capture at all."""
        engine = _engine(_spec("checkmate"))
        placement = {
            "b2": ("N", 1 - side), "b3": ("N", 1 - side), "c3": ("C", WILD),
            "d4": ("T", 1 - side), "a1": ("K", side), "a6": ("K", 1 - side),
            "f1": ("N", side),
        }
        state = engine.make_state(placement, side_to_move=side, moved=list(placement))
        assert not any(move_from(m) == _sq("c3") for m in engine.legal_moves(state))


# --------------------------------------------------------------------------- the validator


class TestTheLift:
    def test_the_pair_is_valid(self) -> None:
        errors = validate_spec(_spec())
        assert errors == []
        assert "O12" not in refused_ids(errors)

    def test_the_inventory_names_it(self) -> None:
        name = "pieces.control.majority.explode_on_capture"
        assert _schema_eighty_seven_uses(_spec()) == {name}
        no_blast = _spec()
        del no_blast["rules"]["explode_on_capture"]
        assert _schema_eighty_seven_uses(no_blast) == set()
        standing = _spec()
        del standing["pieces"][3]["control"]
        assert _schema_eighty_seven_uses(standing) == set()

    def test_describe_says_it(self) -> None:
        assert "the blast is the mover's" in describe_spec(_spec())
        no_blast = _spec()
        del no_blast["rules"]["explode_on_capture"]
        assert "the blast is the mover's" not in describe_spec(no_blast)

    def test_fairy_stockfish_refuses_it(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec())


# --------------------------------------------------------------------------- the walks


def _crowded(engine: Engine) -> GameState:
    """Both royals back, the condottiere between the armies, guards and stones round it."""
    placement = {
        "a1": ("K", 0), "f6": ("K", 1), "c3": ("C", WILD),
        "b2": ("N", 0), "b3": ("N", 0), "d4": ("N", 1), "d5": ("T", 1), "c2": ("T", 0),
        "e4": ("N", 1),
    }
    return engine.make_state(placement, moved=list(placement))


class TestTheWalks:
    @pytest.mark.parametrize("win", ["checkmate", "royal_capture"])
    def test_walks_keep_the_key(self, win: str) -> None:
        """Seeded walks, held captures preferred; the key is the full scan's each ply."""
        engine = _engine(_spec(win))
        held = 0
        for seed in range(8):
            rng = random.Random(seed)
            state = _crowded(engine)
            for _ in range(60):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                loud = [m for m in moves if state.board[move_to(m)]]
                foreign = [m for m in loud if m & MOVE_FOREIGN]
                if foreign and rng.random() < 0.9:
                    move = rng.choice(foreign)
                    held += 1
                else:
                    move = rng.choice(loud if loud and rng.random() < 0.5 else moves)
                after = engine.apply(state, move)
                assert after.key == engine._compute_hash(after), (win, seed)
                state = after
        assert held > 0, win

    @pytest.mark.parametrize("win", ["checkmate", "royal_capture"])
    def test_perft_counts_what_apply_counts(self, win: str) -> None:
        engine = _engine(_spec(win))
        state = _crowded(engine)

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(
                count(engine.apply(position, m), depth - 1) for m in engine.legal_moves(position)
            )

        assert engine.perft(state, 3) == count(state, 3)
