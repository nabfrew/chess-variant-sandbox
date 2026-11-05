"""`Engine.make_state` gives a drop game its hands (the defect SR7j found).

A hand-built position in a game with ``rules.drops`` used to carry `GameState`'s default
``((), ())``, and the core's `deposit_capture` indexed the empty hand at the first capture and
panicked (index out of bounds); tests worked around it with ``with_(hands=...)``. Now
`make_state` sizes the hands to the types, empty unless ``hands`` is given.
"""

from __future__ import annotations

from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_DROP, decode_move


def _spec(drops: bool) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema_version": 27,
        "name": "a small hand",
        "board": {"width": 4, "height": 4},
        "pieces": [
            {
                "id": "K", "name": "king", "value": 0, "royal": True,
                "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
            },
            {
                "id": "R", "name": "rook", "value": 5,
                "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}],
            },
        ],
        "setup": {"mirror": True, "placements": [{"piece": "K", "squares": ["a1"]}, {"piece": "R", "squares": ["d1"]}]},
        "rules": {"win": "royal_capture", "stalemate": "draw", "max_plies": 60},
    }
    if drops:
        spec["rules"]["drops"] = {"from_captures": True}
    return spec


def test_a_drop_game_position_has_empty_hands_sized_to_the_types() -> None:
    engine = Engine(_spec(drops=True))
    state = engine.make_state({"a1": ("K", 0), "d4": ("K", 1), "b1": ("R", 0), "b3": ("R", 1)})
    assert state.hands == ((0, 0), (0, 0))
    assert state.hands == tuple(tuple(0 for _ in hand) for hand in engine.initial_state().hands)


def test_a_capture_from_a_made_position_files_the_taken_piece_and_it_drops() -> None:
    engine = Engine(_spec(drops=True))
    rook = engine.spec.piece_ids.index("R")
    state = engine.make_state({"a1": ("K", 0), "d4": ("K", 1), "b1": ("R", 0), "b3": ("R", 1)})
    take = next(
        move for move in engine.legal_moves(state)
        if decode_move(move)[0] == engine._square_index("b1")
        and decode_move(move)[1] == engine._square_index("b3")
    )
    after = engine.apply(state, take)
    assert after.hands[0][rook] == 1 and after.hands[1] == (0, 0)
    # Black answers; then the taken rook is White's to drop.
    reply = engine.apply(after, engine.legal_moves(after)[0])
    assert any(
        decode_move(move)[2] == MOVE_DROP and decode_move(move)[3] == rook
        for move in engine.legal_moves(reply)
    )


def test_given_hands_are_kept_and_checked() -> None:
    engine = Engine(_spec(drops=True))
    placement = {"a1": ("K", 0), "d4": ("K", 1)}
    state = engine.make_state(placement, hands=((0, 2), (0, 0)))
    assert state.hands == ((0, 2), (0, 0))
    assert any(decode_move(move)[2] == MOVE_DROP for move in engine.legal_moves(state))
    with pytest.raises(ValueError, match="hands"):
        engine.make_state(placement, hands=((1,), (0, 0)))


def test_a_game_without_drops_has_no_hand() -> None:
    engine = Engine(_spec(drops=False))
    state = engine.make_state({"a1": ("K", 0), "d4": ("K", 1)}, hands=((0, 1), (0, 0)))
    assert state.hands == ((), ())
