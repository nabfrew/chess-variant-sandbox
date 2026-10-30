"""CS2 (R1, `docs/plan-smells-rust.md`; PY1): the move word's field map, held equal to the core's.

`engine/moves.py` and `rust/src/tables.rs` each lay the word out; stored records hold raw
words, so the two copies drifting would misread every record on one side. The core exports
its map (`_core.MOVE_FIELDS`, `_core.MOVE_KINDS`) the way it exports `SPY_PIECES`.
"""

from __future__ import annotations

from chess_variant_sandbox import _core
from chess_variant_sandbox.engine import moves
from chess_variant_sandbox.engine.moves import (
    MOVE_FIELDS,
    MOVE_FOREIGN,
    MOVE_PROMOTION,
    encode_move,
    move_fold,
    move_foreign,
    move_landing,
    move_portal,
)
from chess_variant_sandbox.spec.schema import (
    MAX_LANDING_CANDIDATES,
    MAX_PIECE_TYPES,
    MAX_PORTAL_ENDS,
)


def test_the_fields_are_the_cores() -> None:
    assert {name: (shift, width) for name, shift, width in _core.MOVE_FIELDS} == MOVE_FIELDS


def test_the_kinds_are_the_cores() -> None:
    core = dict(_core.MOVE_KINDS)
    assert {name: getattr(moves, name) for name in core} == core
    # And Python knows no kind the core does not: every named kind is one of the core's.
    assert sorted(core.values()) == sorted(moves.MOVE_TYPE_NAMES)
    assert {f"MOVE_{text.upper()}" for text in moves.MOVE_TYPE_NAMES.values()} == set(core)


def test_the_foreign_bit_is_the_foreign_field() -> None:
    shift, width = MOVE_FIELDS["foreign"]
    assert (width, MOVE_FOREIGN) == (1, 1 << shift)


def test_every_tag_value_fits_its_field() -> None:
    _, width = MOVE_FIELDS["landing"]
    assert 1 << width >= MAX_LANDING_CANDIDATES
    assert 1 << MOVE_FIELDS["portal"][1] >= MAX_PIECE_TYPES
    assert 1 << MOVE_FIELDS["fold"][1] >= MAX_PORTAL_ENDS + 1


def test_each_accessor_reads_its_own_field_only() -> None:
    """PY1: the foreign bit is never read as part of a landing, a portal end or a fold end."""
    base = encode_move(200, 255, MOVE_PROMOTION, 31)
    foreign = base | MOVE_FOREIGN
    assert move_foreign(foreign)
    assert (move_landing(foreign), move_portal(foreign), move_fold(foreign)) == (0, 0, 0)
    tagged = encode_move(200, 255, MOVE_PROMOTION, 31, landing=7) | MOVE_FOREIGN
    assert (move_landing(tagged), move_portal(tagged), move_fold(tagged)) == (7, 7, 7)
    assert not move_foreign(encode_move(200, 255, MOVE_PROMOTION, 31, MAX_LANDING_CANDIDATES - 1))
