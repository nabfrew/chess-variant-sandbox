"""R12 (CS6, `docs/plan-smells-rust.md`): `Core.apply` plays only a word the position offers.

`make` trusts its word. Before CS6 a word of a kind the game does not have reached an
`expect` inside the core (Python saw `PanicException`), and a drop from an empty hand slot
wrapped the count and handed back a corrupted position. `apply` now asks the position's own
generator first (its pseudo-legal list, then its legal list) and refuses anything else as a
`ValueError`. Pseudo-legal words stay playable: the declared oracles judge legality by applying
each one and walking the board after it. One test per kind the box names, on stock rulesets;
castling's own message is held by `test_last_game_corners.py`'s castling-word test.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import (
    DUCK_PLACE,
    MOVE_DROP,
    MOVE_DUCK,
    MOVE_FOLD,
    MOVE_HIRE,
    MOVE_NORMAL,
    encode_move,
    move_type,
)

ROOT = Path(__file__).resolve().parents[1]


@cache
def _engine(name: str) -> Engine:
    spec: dict[str, Any] = json.loads((ROOT / "rulesets" / f"{name}.json").read_text("utf-8"))
    return Engine(spec)


def _sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def test_a_duck_move_without_a_duck_is_refused() -> None:
    engine = _engine("chess")
    e4 = _sq(engine, "e4")
    with pytest.raises(ValueError, match="not a move this position offers"):
        engine.apply(engine.initial_state(), encode_move(e4, e4, MOVE_DUCK, DUCK_PLACE))


def test_a_fold_move_without_folds_is_refused() -> None:
    engine = _engine("chess")
    word = encode_move(_sq(engine, "d4"), _sq(engine, "e5"), MOVE_FOLD)
    with pytest.raises(ValueError, match="not a move this position offers"):
        engine.apply(engine.initial_state(), word)


def test_a_drop_from_an_empty_slot_is_refused() -> None:
    """Crazyhouse opens with both hands empty: every slot's drop is refused, where the core
    would have wrapped the count."""
    engine = _engine("crazyhouse")
    state = engine.initial_state()
    assert not any(move_type(m) == MOVE_DROP for m in engine.legal_moves(state))
    e4 = _sq(engine, "e4")
    for slot in range(len(engine.spec.piece_ids)):
        with pytest.raises(ValueError, match="not a move this position offers"):
            engine.apply(state, encode_move(e4, e4, MOVE_DROP, slot))


def test_a_hire_without_a_purse_is_refused() -> None:
    engine = _engine("chess")
    e2 = _sq(engine, "e2")
    with pytest.raises(ValueError, match="not a move this position offers"):
        engine.apply(engine.initial_state(), encode_move(e2, e2, MOVE_HIRE))


def test_an_ordinary_word_the_position_does_not_offer_is_refused() -> None:
    """A word of a kind the game has, but not here: a pawn three squares forward."""
    engine = _engine("chess")
    word = encode_move(_sq(engine, "e2"), _sq(engine, "e5"), MOVE_NORMAL)
    with pytest.raises(ValueError, match="not a move this position offers"):
        engine.apply(engine.initial_state(), word)


def test_every_offered_word_is_still_played() -> None:
    engine = _engine("crazyhouse")
    state = engine.initial_state()
    for move in engine.legal_moves(state):
        assert engine.apply(state, move) != state


def test_a_pseudo_legal_word_that_leaves_the_royal_attacked_is_still_played() -> None:
    """The oracles' contract: a pinned piece's move is generated pseudo-legally, so `apply`
    plays it though `legal_moves` leaves it out."""
    engine = _engine("chess")
    state = engine.make_state({"e1": ("K", 0), "e2": ("R", 0), "e8": ("R", 1), "a8": ("K", 1)})
    pinned = encode_move(_sq(engine, "e2"), _sq(engine, "a2"), MOVE_NORMAL)
    assert pinned in engine.pseudo_legal_moves(state)
    assert pinned not in engine.legal_moves(state)
    assert engine.apply(state, pinned) != state
