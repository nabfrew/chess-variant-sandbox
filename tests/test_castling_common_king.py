"""Castling with a piece that is not royal (SR1d, `docs/plan-schema-next.md` §1 rank 8, row Z2).

Under a rule with no royalty (`win: "annihilation"`) the validator already lets a commander
castle, and the core reads no attack for it (`Tables::needs_check_filter` is false). Beside
royals under checkmate the validator still refuses it (`schema.py`, Z2), and these tests show
why the refusal cannot simply go: fed to the core past the validator, a castle whose rook
stands pinned to the royal is offered, because `generate_legal`'s pin shortcut asks only a
move *from* the royal or a pinned square, and a castle's second piece moves from the rook's.
The fix is the core's (SR1d, brief held: no), so that test is a strict xfail naming it.
"""

from __future__ import annotations

from typing import Any

import pytest

from chess_variant_sandbox.engine import compile as compile_module
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.spec.schema import validate_spec

KING_STEP = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}
ROOK_LINE = {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "both"}


def _spec(white: dict[str, list[str]], black: dict[str, list[str]], win: str) -> dict[str, Any]:
    """K (royal unless `win` has no royalty), C a commander that castles, R a rook."""
    return {
        "schema_version": 31, "name": "commander castling", "description": "a probe",
        "board": {"width": 8, "height": 8},
        "pieces": [
            {"id": "K", "name": "king", "value": 0, "royal": win != "annihilation",
             "moves": [KING_STEP]},
            {"id": "C", "name": "commander", "value": 3, "moves": [KING_STEP]},
            {"id": "R", "name": "rook", "value": 5, "moves": [ROOK_LINE]},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": piece, "squares": squares, "player": player}
            for player, army in ((0, white), (1, black)) for piece, squares in army.items()
        ]},
        "rules": {"win": win, "stalemate": "draw",
                  "castling": {"king": "C", "rook": "R", "king_steps": 2},
                  "max_plies": 200},
    }


def _castles(spec: dict[str, Any], monkeypatch: pytest.MonkeyPatch | None = None) -> list[str]:
    if monkeypatch is not None:
        monkeypatch.setattr(compile_module, "validate_or_raise", lambda spec: None)
    engine = Engine(compile_spec(spec))
    state = engine.initial_state()
    return [text for text in engine.move_texts(state) if "castle" in text]


def test_a_commander_castles_under_annihilation() -> None:
    spec = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"]}, "annihilation")
    assert validate_spec(spec) == []
    assert _castles(spec) == ["Ce1-g1 (castle)"]


def test_without_check_the_castle_reads_no_attack() -> None:
    """A black rook on f8 attacks f1, the square the commander crosses: no royalty, no check,
    so the castle stands."""
    spec = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"], "R": ["f8"]},
                 "annihilation")
    assert validate_spec(spec) == []
    assert _castles(spec) == ["Ce1-g1 (castle)"]


def test_beside_royals_under_checkmate_it_is_still_refused() -> None:
    spec = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]}, {"K": ["a8"]}, "checkmate")
    assert any("rules.castling.king 'C' is not a royal piece" in error
               for error in validate_spec(spec))


def test_where_check_exists_the_core_reads_attacks_on_the_commander(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the validator, the core tests the castling piece's crossing as a king's: f1
    attacked, no castle; the rook moved off the f-file, the castle."""
    attacked = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]},
                     {"K": ["a8"], "R": ["f8"]}, "checkmate")
    assert _castles(attacked, monkeypatch) == []
    quiet = _spec({"K": ["a1"], "R": ["h1"], "C": ["e1"]},
                  {"K": ["a8"], "R": ["b8"]}, "checkmate")
    assert _castles(quiet, monkeypatch) == ["Ce1-g1 (castle)"]


@pytest.mark.xfail(
    strict=True,
    reason="SR1d needs the core: generate_legal's pin shortcut does not ask a castle whose "
    "rook is pinned to the royal (the castling piece is not the royal)",
)
def test_a_castle_never_bares_the_royal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Royal K on b1, rook b2 pinned by the black rook on b8; the commander on e2 castles
    long, carrying the rook to d2 and leaving the king in check."""
    spec = _spec({"K": ["b1"], "R": ["b2"], "C": ["e2"]},
                 {"K": ["h8"], "R": ["b8"]}, "checkmate")
    assert _castles(spec, monkeypatch) == []
