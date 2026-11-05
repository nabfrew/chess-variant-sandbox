"""SR5b (`docs/plan-schema-next.md` §2.7 row V2): a line of sight beside a blast.

The ruling (2026-10-27, the user's words): *a shield also shields from a blast*. A shielded
piece in a blast survives as if its type were in `explode_on_capture.immune`, so a shielded
royal is never lost to one. The row's second half stands: a frozen or disarmed piece makes no
capture, so sets off no blast. A commanded capture (`captures_own`) explodes as any capture,
and is the commander's move: a blast that would take the commander's own royal makes it
illegal.

The probe (fed past the validator on SR5a's build `47768a9737c82fe0`, a rook taking a guard
beside a blast of a king's step): the core cleared a shielded guard and a shielded king as if
no shield stood, so a capture next to a shielded royal won the game, and a capture next to the
mover's own shielded royal was not offered. The frozen capturer's capture was already not
offered (the sight filter drops every move of a frozen piece), and so was a disarmed one's.

Readings the row left open, settled on the plan's principles and written in the record: the
shields are read **once, on the board the capture leaves** (the capturer standing on the capture
square, nothing cleared yet), so a shield whose source the same blast destroys still spares
what it covered (the blast is one event, as its victims are); the capturer always goes, as an
immune capturer does; a round reads every blast of the round on the board its captures left.
Hand-built 6x6 specs, not chess.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import move_from, move_to
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.refusals import refused_ids
from chess_variant_sandbox.spec.schema import validate_spec
from tests.feature_inventory import _schema_seventy_nine_uses

WIDTH = 6
KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"}
ROOK = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}
AROUND = {"vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "range": 1}
IDS = ("K", "R", "G", "S", "F", "D", "M")


def _sq(name: str) -> int:
    return (int(name[1:]) - 1) * WIDTH + ord(name[0]) - ord("a")


def _name(square: int) -> str:
    return f"{'abcdef'[square % WIDTH]}{square // WIDTH + 1}"


def _spec(win: str = "checkmate", **rules: Any) -> dict[str, Any]:
    """Kings, rooks and guards; a shielder `S` (its own side's pieces a king's step away), a
    freezer `F` and a disarmer `D` (the enemy's, a king's step away), and a preacher `M` who
    commands the first enemy piece on each rook line and may make it take its own side. A blast
    of a king's step on every capture."""
    spec: dict[str, Any] = {
        "name": "the shielded blast (SR5b test spec)",
        "board": {"width": WIDTH, "height": WIDTH},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [KING]},
            {"id": "R", "value": 5, "moves": [ROOK]},
            {"id": "G", "value": 2, "moves": [KING]},
            {"id": "S", "value": 2, "moves": [KING],
             "sight": [{**AROUND, "effect": "shield", "whose": "own"}]},
            {"id": "F", "value": 2, "moves": [KING],
             "sight": [{**AROUND, "effect": "freeze", "whose": "enemy"}]},
            {"id": "D", "value": 2, "moves": [KING],
             "sight": [{**AROUND, "effect": "disarm", "whose": "enemy"}]},
            {"id": "M", "value": 3, "moves": [KING],
             "sight": [{"vectors": [[1, 0]], "symmetry": "rotate4", "effect": "command",
                        "whose": "enemy", "captures_own": True}]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["f1"]},
            {"piece": "G", "squares": ["b1"]}, {"piece": "S", "squares": ["c1"]},
            {"piece": "F", "squares": ["d1"]}, {"piece": "D", "squares": ["e1"]},
            {"piece": "M", "squares": ["b2"]},
        ]},
        "rules": {"win": win, "stalemate": "draw", "max_plies": 60,
                  "explode_on_capture": {"radius": 1}},
    }
    spec["rules"].update(rules)
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    return Engine(compile_spec(spec))


def _board(engine: Engine, state: GameState) -> dict[str, tuple[str, int]]:
    spec = engine.spec
    return {
        _name(sq): (IDS[spec.type_of(v)], spec.owner_of(v)) for sq, v in enumerate(state.board) if v
    }


def _find(engine: Engine, state: GameState, origin: str, target: str) -> int | None:
    for move in engine.legal_moves(state):
        if (move_from(move), move_to(move)) == (_sq(origin), _sq(target)):
            return move
    return None


def _play(placement: dict[str, tuple[str, int]], origin: str = "f3", target: str = "d3",
          spec: dict[str, Any] | None = None) -> tuple[Engine, GameState, GameState]:
    engine = _engine(spec or _spec())
    state = engine.make_state(placement, side_to_move=0, moved=list(placement))
    move = _find(engine, state, origin, target)
    assert move is not None, f"no move {origin}-{target}"
    after = engine.apply(state, move)
    assert after.key == engine._compute_hash(after)
    return engine, state, after


# The rook on f3 takes the guard on d3; the blast reaches c2-e4.
KINGS = {"a1": ("K", 0), "a6": ("K", 1)}


class TestAShieldShieldsFromABlast:
    def test_a_shielded_piece_survives_and_an_unshielded_one_goes(self) -> None:
        """An enemy guard on c3 shielded by its shielder on b3 survives; the guard on d4, which
        nothing shields, goes with the capturer and the piece taken."""
        placement = {**KINGS, "f3": ("R", 0), "d3": ("G", 1), "c3": ("G", 1),
                     "b3": ("S", 1), "d4": ("G", 1)}
        engine, _, after = _play(placement)
        assert _board(engine, after) == {**KINGS, "c3": ("G", 1), "b3": ("S", 1)}

    def test_a_shield_destroyed_by_the_same_blast_still_spares(self) -> None:
        """The shielder on c4 (covering guards only, so not the rook taken on d3) stands in the
        blast and goes; the guards it covered on c3 and d4 survive, the shields being read
        before anything is cleared; the guard on e4, out of its reach, goes."""
        spec = _spec()
        spec["pieces"][3]["sight"][0]["affects"] = ["G"]
        placement = {**KINGS, "f3": ("R", 0), "d3": ("R", 1), "c3": ("G", 1),
                     "c4": ("S", 1), "d4": ("G", 1), "e4": ("G", 1)}
        engine, _, after = _play(placement, spec=spec)
        assert _board(engine, after) == {**KINGS, "c3": ("G", 1), "d4": ("G", 1)}

    def test_a_shield_spares_the_mover_own_pieces_too(self) -> None:
        """The mover's guard on c2, shielded by its own shielder on b1, survives its own blast;
        the capturer itself goes though the shielder on e3 covers it on d3 (as an immune
        capturer does), and that shielder, which nothing covers, goes."""
        placement = {**KINGS, "d1": ("R", 0), "e3": ("S", 0), "d3": ("G", 1), "c2": ("G", 0),
                     "b1": ("S", 0)}
        engine, _, after = _play(placement, "d1", "d3")
        board = _board(engine, after)
        assert board == {**KINGS, "c2": ("G", 0), "b1": ("S", 0)}, board

    def test_a_shielded_enemy_royal_next_to_a_capture_is_not_lost(self) -> None:
        unshielded = {"a1": ("K", 0), "e4": ("K", 1), "f3": ("R", 0), "d3": ("G", 1)}
        engine, _, after = _play(unshielded)
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "royal_destroyed"
        engine, _, after = _play({**unshielded, "f5": ("S", 1)})
        assert _board(engine, after) == {"a1": ("K", 0), "e4": ("K", 1), "f5": ("S", 1)}
        assert engine.terminal(after) is None

    def test_a_capture_next_to_the_mover_own_shielded_royal_is_legal(self) -> None:
        """Without the shield the capture would destroy the mover's king, so it is not offered;
        with it the king survives and the capture is a move."""
        placement = {"c2": ("K", 0), "a6": ("K", 1), "f3": ("R", 0), "d3": ("G", 1)}
        engine = _engine(_spec())
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        assert _find(engine, state, "f3", "d3") is None
        engine, _, after = _play({**placement, "b1": ("S", 0)})
        assert _board(engine, after) == {"c2": ("K", 0), "b1": ("S", 0), "a6": ("K", 1)}
        assert engine.terminal(after) is None


class TestARestrainedPieceSetsOffNothing:
    @pytest.mark.parametrize("restrainer", ["F", "D"])
    def test_a_frozen_or_disarmed_piece_capture_is_not_offered(self, restrainer: str) -> None:
        placement = {**KINGS, "f3": ("R", 0), "d3": ("G", 1), "f4": (restrainer, 1)}
        engine = _engine(_spec())
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        assert _find(engine, state, "f3", "d3") is None
        freed = {**placement}
        freed.pop("f4")
        state = engine.make_state(freed, side_to_move=0, moved=list(freed))
        assert _find(engine, state, "f3", "d3") is not None


class TestACommandedCapture:
    # The preacher on f1 commands the enemy rook on f3 (the first enemy piece up the f-file),
    # which may be made to take its own side's guard on d3.
    BASE = {"a6": ("K", 1), "f1": ("M", 0), "f3": ("R", 1), "d3": ("G", 1)}

    def test_it_explodes_as_any_capture(self) -> None:
        placement = {**self.BASE, "a1": ("K", 0), "e4": ("G", 1)}
        engine, _, after = _play(placement)
        assert _board(engine, after) == {"a1": ("K", 0), "a6": ("K", 1), "f1": ("M", 0)}

    def test_one_whose_blast_takes_the_commander_royal_is_illegal(self) -> None:
        placement = {**self.BASE, "c2": ("K", 0)}
        engine = _engine(_spec())
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        assert _find(engine, state, "f3", "d3") is None
        # The commander's royal shielded: the blast spares it, and the capture is a move.
        engine, _, after = _play({**placement, "b1": ("S", 0)})
        assert _board(engine, after) == {
            "c2": ("K", 0), "b1": ("S", 0), "a6": ("K", 1), "f1": ("M", 0),
        }


class TestARound:
    def test_a_round_blast_spares_a_shielded_piece(self) -> None:
        """In rounds every blast of the round reads the shields on the board its captures left:
        the guard on c3 under its shielder on b3 survives, the one on d4 goes."""
        # No preacher: a command is refused in rounds (R17).
        spec = _spec("royal_capture", simultaneous={})
        spec["pieces"] = [p for p in spec["pieces"] if p["id"] != "M"]
        spec["setup"]["placements"] = [p for p in spec["setup"]["placements"] if p["piece"] != "M"]
        engine = _engine(spec)
        placement = {**KINGS, "f3": ("R", 0), "d3": ("G", 1), "c3": ("G", 1),
                     "b3": ("S", 1), "d4": ("G", 1)}
        state = engine.make_state(placement, side_to_move=0, moved=list(placement))
        ours = next(m for m in engine.legal_moves_for(state, 0)
                    if (move_from(m), move_to(m)) == (_sq("f3"), _sq("d3")))
        theirs = next(m for m in engine.legal_moves_for(state, 1)
                      if (move_from(m), move_to(m)) == (_sq("a6"), _sq("a5")))
        after = engine.apply_round(state, ours, theirs)
        board = {
            _name(sq): (("K", "R", "G", "S", "F", "D")[engine.spec.type_of(v)],
                        engine.spec.owner_of(v))
            for sq, v in enumerate(after.board) if v
        }
        assert board == {"a1": ("K", 0), "a5": ("K", 1), "c3": ("G", 1), "b3": ("S", 1)}


class TestUndo:
    PLACEMENT = {"a1": ("K", 0), "f6": ("K", 1), "f3": ("R", 0), "d3": ("G", 1), "c3": ("G", 1),
                 "c4": ("S", 1), "d4": ("G", 1), "b1": ("S", 0), "c2": ("G", 0), "e5": ("R", 1)}

    def test_perft_counts_what_apply_counts(self) -> None:
        """A tree of shielded blasts: an unmake that restored a spared or a cleared piece wrong
        would move the next sibling's count."""
        engine = _engine(_spec())
        state = engine.make_state(self.PLACEMENT, side_to_move=0, moved=list(self.PLACEMENT))

        def count(position: GameState, depth: int) -> int:
            if depth == 0:
                return 1
            return sum(count(engine.apply(position, m), depth - 1)
                       for m in engine.legal_moves(position))

        assert engine.perft(state, 3) == count(state, 3)

    def test_walks_keep_the_key(self) -> None:
        """Seeded walks, captures preferred: the key is the full scan's every ply, and `apply`
        leaves its argument as it was."""
        engine = _engine(_spec())
        for seed in range(4):
            rng = random.Random(seed)
            state = engine.make_state(self.PLACEMENT, side_to_move=0, moved=list(self.PLACEMENT))
            for _ in range(40):
                moves = engine.legal_moves(state)
                if not moves or engine.terminal(state, moves) is not None:
                    break
                loud = [m for m in moves if state.board[move_to(m)]]
                move = rng.choice(loud if loud and rng.random() < 0.7 else moves)
                board = bytes(state.board)
                after = engine.apply(state, move)
                assert bytes(state.board) == board
                assert after.key == engine._compute_hash(after)
                state = after


class TestTheSurfaces:
    def test_the_validator_lifts_v2(self) -> None:
        errors = validate_spec(_spec())
        assert errors == [] and "V2" not in refused_ids(errors)

    def test_the_inventory_names_it(self) -> None:
        assert _schema_seventy_nine_uses(_spec()) == {"pieces.sight.explode_on_capture"}
        spec = _spec()
        spec["rules"].pop("explode_on_capture")
        assert _schema_seventy_nine_uses(spec) == set()

    def test_describe_says_it(self) -> None:
        said = describe_spec(_spec())
        assert "A shielded piece also survives a blast" in said
        assert "A frozen or disarmed piece makes no capture, so sets off no blast." in said
        assert "A commanded capture explodes as any capture does" in said

    def test_fairy_stockfish_refuses_the_sight(self) -> None:
        with pytest.raises(Untranslatable):
            translate(_spec())
