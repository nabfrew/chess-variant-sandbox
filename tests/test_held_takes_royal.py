"""Q1077 (`docs/plan-turncoat.md` §4.7): a held neutral that takes a royal, and so gives check.

The user, 2026-10-31: *"Mercenary pieces don't seem to recognise check. And if it's win by mate
only not capture royal, it robs players of a winning scenario."* Under TC3 a neutral attacked
nothing, so a condottiere the opponent held could stand a knight's leap from a king and the king
could walk onto its squares. `pieces[].takes_royal: true` on a majority type makes a held piece
of it take the royal of the player it is held against, and so attack it: check, mate, and a
legality filter that sees a check handed over by a change of count.

Specs, not chess: `archive/the-crowd.json` (which carries the key since this change) and edits of
it. The oracle is `tests/neutral_oracle.py`, which reads the attack off the frame-0 twin and the
game without the key, never off the core's `held_attacks`.

The other readings of the report are held here too: a hired or bribed piece (TC6) is an ordinary
piece of its new owner's and gives check like one, and a hire is legal only where it leaves the
hirer's royal unattacked (`archive/the-free-companies.json`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_HIRE, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.spec.schema import validate_spec
from tests.neutral_oracle import NeutralOracle

ROOT = Path(__file__).resolve().parent.parent


def _crowd() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((ROOT / "archive" / "the-crowd.json").read_text("utf-8"))
    return loaded


def _without_key(spec: dict[str, Any]) -> dict[str, Any]:
    plain = copy.deepcopy(spec)
    for piece in plain["pieces"]:
        piece.pop("takes_royal", None)
    return plain


def _position(engine: Engine, placement: dict[str, tuple[str, int]], side: int) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement))


def _texts(engine: Engine, state: GameState) -> set[str]:
    return {engine.describe_move(state, move) for move in engine.legal_moves(state)}


def _agrees(spec: dict[str, Any], engine: Engine, state: GameState) -> None:
    assert sorted(engine.legal_moves(state)) == NeutralOracle(spec, engine).legal_moves(state)


# White holds the condottiere on d6 (white pawns c5 and e5 beside it, black's d7 one), a knight's
# leap from the black king on e8.
HELD = {
    "h1": ("K", 0), "e8": ("K", 1), "d6": ("C", 2), "c5": ("P", 0), "e5": ("P", 0),
    "d7": ("P", 1), "a7": ("R", 1), "h7": ("P", 1),
}


class TestTheSpelling:
    def test_the_crowd_carries_the_key(self) -> None:
        assert validate_spec(_crowd()) == []
        assert spec_hash(_crowd()) == "ea7aa465cbfb"

    def test_false_is_the_game_without_it(self) -> None:
        spec = _without_key(_crowd())
        written = copy.deepcopy(spec)
        next(p for p in written["pieces"] if p["id"] == "C")["takes_royal"] = False
        assert validate_spec(written) == []
        assert spec_hash(written) == spec_hash(spec) == "961b230bc5eb"

    def test_it_is_refused_where_nobody_holds_the_type(self) -> None:
        spec = _crowd()
        next(p for p in spec["pieces"] if p["id"] == "N")["takes_royal"] = True
        assert any("takes_royal on 'N'" in error for error in validate_spec(spec))

    def test_it_is_refused_on_a_type_that_stands(self) -> None:
        spec = _crowd()
        next(p for p in spec["pieces"] if p["id"] == "C").pop("control")
        assert any("takes_royal" in error for error in validate_spec(spec))

    def test_it_must_be_a_boolean(self) -> None:
        spec = _crowd()
        next(p for p in spec["pieces"] if p["id"] == "C")["takes_royal"] = 1
        assert any("must be true or false" in error for error in validate_spec(spec))


class TestCheck:
    def test_a_held_condottiere_gives_check(self) -> None:
        spec = _crowd()
        engine = Engine(spec)
        state = _position(engine, HELD, 1)
        assert engine.controller_of(state, engine._square_index("d6")) == 0
        assert engine.in_check(state)
        # f7 is the condottiere's square too: the king may not step onto it.
        assert "Ke8-f7" not in _texts(engine, state)
        _agrees(spec, engine, state)

    def test_without_the_key_it_attacks_nothing(self) -> None:
        spec = _without_key(_crowd())
        engine = Engine(spec)
        state = _position(engine, HELD, 1)
        assert not engine.in_check(state)
        assert "Ke8-f7" in _texts(engine, state)
        _agrees(spec, engine, state)

    def test_a_check_is_answered_by_tying_the_count(self) -> None:
        spec = _crowd()
        engine = Engine(spec)
        state = _position(engine, HELD, 1)
        texts = _texts(engine, state)
        # The rook steps next to the condottiere: two against two, nobody holds it.
        assert "Ra7-c7" in texts
        # Off the neighbours it answers nothing.
        assert "Ra7-b7" not in texts and "Ra7-a1" not in texts
        _agrees(spec, engine, state)

    def test_a_tied_or_own_condottiere_gives_no_check(self) -> None:
        engine = Engine(_crowd())
        tied = dict(HELD) | {"c7": ("P", 1)}
        assert engine.controller_of(_position(engine, tied, 1), engine._square_index("d6")) == -1
        assert not engine.in_check(_position(engine, tied, 1))
        black = dict(tied) | {"e7": ("P", 1)}
        assert engine.controller_of(_position(engine, black, 1), engine._square_index("d6")) == 1
        assert not engine.in_check(_position(engine, black, 1))

    def test_a_move_that_hands_over_the_count_and_exposes_the_king_is_illegal(self) -> None:
        spec = _crowd()
        engine = Engine(spec)
        # Two against two: the knight on e7 is one of black's. Leaving the neighbours hands the
        # condottiere to white, whose leap reaches e8.
        placement = dict(HELD) | {"e7": ("N", 1)}
        state = _position(engine, placement, 1)
        assert not engine.in_check(state)
        texts = _texts(engine, state)
        assert "Ne7-g6" not in texts and "Ne7-g8" not in texts
        assert "Ne7-c6" in texts  # it lands beside the condottiere, so the count still ties
        _agrees(spec, engine, state)

    def test_a_held_condottiere_mates(self) -> None:
        spec = _crowd()
        engine = Engine(spec)
        # The condottiere on f7, held by white's e6 pawn and e7 knight against black's g7 pawn,
        # leaps onto h8; the knight covers g8, where the king would also tie the count.
        placement = {
            "a1": ("K", 0), "h8": ("K", 1), "f7": ("C", 2), "e6": ("P", 0), "e7": ("N", 0),
            "g7": ("P", 1), "h7": ("P", 1),
        }
        state = _position(engine, placement, 1)
        assert engine.in_check(state)
        assert engine.legal_moves(state) == []
        outcome = engine.terminal(state)
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "checkmate"
        _agrees(spec, engine, state)
        assert Engine(_without_key(spec)).terminal(_position(Engine(_without_key(spec)), placement, 1)) is None


class TestTakingTheRoyal:
    def test_under_royal_capture_a_held_condottiere_takes_the_king(self) -> None:
        spec = _crowd()
        spec["rules"]["win"] = "royal_capture"
        assert validate_spec(spec) == []
        engine = Engine(spec)
        state = _position(engine, HELD, 0)
        assert "~Cd6xe8" in _texts(engine, state)
        _agrees(spec, engine, state)
        plain = _without_key(spec)
        plain_engine = Engine(plain)
        assert "~Cd6xe8" not in _texts(plain_engine, _position(plain_engine, HELD, 0))


class TestHiredPiecesCheck:
    """Readings (a) and (b) of the report for the hiring games: no defect, held down."""

    def _engine(self) -> Engine:
        spec = json.loads((ROOT / "archive" / "the-free-companies.json").read_text("utf-8"))
        return Engine(spec)

    def test_a_hired_mercenary_gives_check_and_bounds_the_king(self) -> None:
        engine = self._engine()
        placement = {"e1": ("K", 0), "e8": ("K", 1), "d6": ("M", 0), "a2": ("P", 0), "h7": ("P", 1)}
        state = engine.make_state(placement, side_to_move=1, moved=list(placement), purse=(0, 0))
        assert engine.in_check(state)
        assert _texts(engine, state) == {"Ke8-d8", "Ke8-f8"}

    def test_a_hire_that_mates_is_mate(self) -> None:
        engine = self._engine()
        placement = {
            "a1": ("K", 0), "h8": ("K", 1), "g7": ("P", 1), "h7": ("P", 1), "f7": ("M", 2),
            "e6": ("P", 0),
        }
        state = engine.make_state(placement, side_to_move=0, moved=list(placement), purse=(9, 0))
        hire = next(m for m in engine.legal_moves(state) if move_type(m) == MOVE_HIRE)
        after = engine.apply(state, hire)
        outcome = engine.terminal(after)
        assert outcome is not None and outcome.winner == 0 and outcome.reason == "checkmate"

    def test_a_hire_does_not_answer_a_check_it_leaves_standing(self) -> None:
        engine = self._engine()
        placement = {
            "e1": ("K", 0), "e2": ("R", 0), "e8": ("K", 1), "a5": ("M", 2), "a6": ("P", 1),
            "h2": ("P", 0),
        }
        state = engine.make_state(placement, side_to_move=1, moved=list(placement), purse=(9, 9))
        assert engine.in_check(state)
        assert not any(move_type(m) == MOVE_HIRE for m in engine.legal_moves(state))
