"""Control and allegiance, unified (TC1, `docs/plan-turncoat.md` §3): the compiled control table.

Every nobody's piece today is one of three kinds -- a route walker (`rules.npcs`), the duck
(`rules.duck`), a portal end (`rules.portals`) -- and `compile.ControlRule` says which, per type,
for the core's `Tables::value_kind`. The table is derived wholly from those rules, so it stays out
of the tables fingerprint (`bridge._DERIVED`) and no stored basis moves for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_variant_sandbox.engine import Engine, bridge
from chess_variant_sandbox.engine.compile import (
    CONTROL_DUCK,
    CONTROL_NONE,
    CONTROL_PORTAL,
    CONTROL_ROUTE,
    compile_spec,
)
from chess_variant_sandbox.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def _compiled(path: str):  # type: ignore[no-untyped-def]
    return compile_spec(load_spec(ROOT / path))


def test_two_armies_cross_no_control_table() -> None:
    spec = _compiled("rulesets/chess.json")
    assert spec.control is None
    assert bridge.build_tables(spec)["control_kind"] == []


def test_a_duck_alone_is_the_only_kind_in_its_block() -> None:
    spec = _compiled("tests/_duck_chess.json")
    assert spec.control is not None and spec.duck is not None
    kinds = spec.control.kind
    assert kinds[spec.duck.piece_type] == CONTROL_DUCK
    assert all(kind == CONTROL_NONE for t, kind in enumerate(kinds) if t != spec.duck.piece_type)


def test_the_duck_beside_wild_pieces_is_not_a_route_walker() -> None:
    """*The shepherd's duck*: the duck and a wild type share the third block; only the wild type
    walks a route, which is what `is_wild_piece` answered by comparing against the duck's byte."""
    spec = _compiled("archive/the-shepherds-duck.json")
    assert spec.control is not None and spec.duck is not None and spec.npcs is not None
    kinds = spec.control.kind
    assert kinds[spec.duck.piece_type] == CONTROL_DUCK
    walkers = {piece for piece in spec.npcs.piece if piece >= 0}
    assert walkers and all(kinds[piece] == CONTROL_ROUTE for piece in walkers)


def test_portal_ends_are_portals() -> None:
    spec = _compiled("archive/carousel.json")
    assert spec.control is not None and spec.portals is not None
    for piece_type, kind in enumerate(spec.control.kind):
        expected = CONTROL_PORTAL if spec.portals.is_end(piece_type) else CONTROL_NONE
        assert kind == expected


@pytest.mark.parametrize(
    "path", ["tests/_duck_chess.json", "archive/the-shepherds-duck.json", "archive/carousel.json"]
)
def test_the_control_table_is_left_out_of_the_fingerprint(path: str) -> None:
    spec = _compiled(path)
    assert bridge.build_tables(spec)["control_kind"]
    assert "control_kind" not in bridge.canonical_tables(spec)


def test_the_core_takes_the_table() -> None:
    """The core reads the kinds (`Tables::from_py` checks the shape) and plays the duck game."""
    engine = Engine(load_spec(ROOT / "archive/the-shepherds-duck.json"))
    assert engine.legal_moves(engine.initial_state())
