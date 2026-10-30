"""Off the board is a capture (Q1074b3, `docs/plan-push.md` §3.3, C1-C6, K6).

A push that throws a piece off the board changes material: the no-progress clock resets and
the repetition window clears, and an ending that counts the board reads it. Where it throws
an *enemy* off it takes that piece as a capture does: the piece goes to the pusher's hand
demoted (C2), its bounty is paid to the pusher (C3), it satisfies `must_capture` (C6), and
the searches read it as loud and order it by what it takes (`is_loud`, `capture_gain`, and
their `agents/minimax.py` mirrors). A push that throws the pusher's own piece off takes
nothing (no hand, no bounty, not compelled, not loud). A push that only moves its piece does
none of these. The text writes `>` for a push and `>>` for one that removed.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from chess_variant_sandbox.agents.base import build_agent
from chess_variant_sandbox.agents.minimax import MinimaxAgent
from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_PUSH, decode_move, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec
from chess_variant_sandbox.tournament.analysis import takes


def _flat(**rules: Any) -> dict[str, Any]:
    """A 6x6 board, no check: a rook that pushes any piece on its lines (and never captures),
    a pawn that only steps, and a queen promoted from the pawn, so a hand reads the demotion."""
    spec: dict[str, Any] = {
        "name": "push captures",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
            ]},
            {"id": "R", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                 "push": {"whose": "any"}},
            ]},
            {"id": "P", "value": 1, "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            ]},
            {"id": "Q", "value": 9, "promoted_from": "P", "moves": [
                {"kind": "slide", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "R", "squares": ["c1"]},
            {"piece": "P", "squares": ["b2", "c2", "d2"]},
        ]},
        "rules": {
            "win": "royal_capture",
            "max_plies": 200,
            "promotion": {"piece": "P", "rank": "last", "to": ["Q"]},
        },
    }
    spec["rules"].update(rules)
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == []
    return Engine(spec)


def _push(engine: Engine, state: GameState, origin: str, target: str) -> int:
    o, t = engine._square_index(origin), engine._square_index(target)
    [move] = [
        move for move in engine.legal_moves(state)
        if move_type(move) == MOVE_PUSH and decode_move(move)[:2] == (o, t)
    ]
    return move


# The white rook on c3: c6 holds a black pawn on the edge (pushed off), e3 a black pawn with
# f3 behind it empty (pushed to f3), c1 a white pawn on the edge (its own, pushed off).
_PLACEMENT = {
    "a1": ("K", 0), "c3": ("R", 0), "c1": ("P", 0), "c6": ("P", 1), "e3": ("P", 1),
    "f6": ("K", 1), "a6": ("Q", 1),
}
_OFF, _MOVED, _OWN_OFF = ("c3", "c6"), ("c3", "e3"), ("c3", "c1")


def _position(engine: Engine, **extra: Any) -> GameState:
    # Every piece has moved, so no first-move right is spent and only material clears the
    # window.
    return engine.make_state(
        _PLACEMENT, side_to_move=0, halfmove=5, moved=list(_PLACEMENT), **extra,
    )


def _played(engine: Engine, state: GameState, push: tuple[str, str]) -> GameState:
    return engine.apply(state, _push(engine, state, *push))


def test_a_removing_push_resets_the_clock_and_clears_the_window_c1_k4() -> None:
    engine = _engine(_flat(no_progress_draw=50, repetition_draw=3))
    state = _position(engine)
    for push in (_OFF, _OWN_OFF):
        after = _played(engine, state, push)
        assert after.halfmove == 0, push
        assert after.repetition == (after.key,), push
    moved = _played(engine, state, _MOVED)
    assert moved.halfmove == 6
    assert moved.repetition == (*state.repetition, moved.key)


def test_a_removing_push_fills_the_pusher_s_hand_demoted_c2() -> None:
    engine = _engine(_flat(drops={"from_captures": True}))
    spec = engine.spec
    pawn, queen = spec.piece_ids.index("P"), spec.piece_ids.index("Q")
    placement = dict(_PLACEMENT)
    del placement["a6"]
    placement["a3"] = ("Q", 1)
    state = engine.make_state(placement, side_to_move=0)
    if not state.hands[0]:
        zeros = (0,) * len(spec.piece_ids)
        state = dataclasses.replace(state, hands=(zeros, zeros))
    empty = state.hands[0]
    off = _played(engine, state, _OFF)
    assert off.hands[0][pawn] == empty[pawn] + 1
    # The queen goes to the hand as the pawn it was promoted from.
    queen_off = engine.apply(state, _push(engine, state, "c3", "a3"))
    assert queen_off.hands[0][pawn] == empty[pawn] + 1
    assert queen_off.hands[0][queen] == empty[queen]
    # The pusher's own pawn, and a pawn moved and not removed, reach no hand.
    assert _played(engine, state, _OWN_OFF).hands == state.hands
    assert _played(engine, state, _MOVED).hands == state.hands


def test_a_removing_push_pays_a_bounty_c3() -> None:
    spec = _flat(scoring={"count": "points"})
    for piece in spec["pieces"]:
        if piece["id"] == "P":
            piece["bounty"] = 2
    engine = _engine(spec)
    state = _position(engine)
    assert _played(engine, state, _OFF).points == (2.0, 0.0)
    assert _played(engine, state, _OWN_OFF).points == (0.0, 0.0)
    assert _played(engine, state, _MOVED).points == (0.0, 0.0)


def test_a_removing_push_satisfies_must_capture_c6() -> None:
    engine = _engine(_flat(must_capture=True))
    state = _position(engine)
    legal = engine.legal_moves(state)
    texts = sorted(engine.describe_move(state, move) for move in legal)
    # Only the pushes that throw an enemy off: the pawn on c6 and the queen on a6 is too far
    # (a6 is not on the rook's lines from c3), so the pawn alone.
    assert texts == ["Rc3>>c6"]
    # With no enemy to throw off, a push that moves one, or throws one's own off, compels
    # nothing: every move stays.
    calm = engine.make_state(
        {k: v for k, v in _PLACEMENT.items() if k != "c6"}, side_to_move=0,
    )
    calm_texts = {engine.describe_move(calm, move) for move in engine.legal_moves(calm)}
    assert {"Rc3>e3", "Rc3>>c1", "Ka1-a2"} <= calm_texts


def test_a_removing_push_ends_an_extinction_game() -> None:
    engine = _engine(_flat(extinction={"pieces": ["P"], "result": "loss"}))
    state = engine.make_state(
        {"a1": ("K", 0), "c3": ("R", 0), "c6": ("P", 1), "f6": ("K", 1), "b2": ("P", 0)},
        side_to_move=0,
    )
    assert engine.terminal(state) is None
    after = _played(engine, state, _OFF)
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0
    moved = engine.make_state(
        {"a1": ("K", 0), "c3": ("R", 0), "e3": ("P", 1), "f6": ("K", 1), "b2": ("P", 0)},
        side_to_move=0,
    )
    assert engine.terminal(_played(engine, moved, _MOVED)) is None


def test_a_push_reads_as_a_push_and_a_removal_as_a_throw_k6() -> None:
    engine = _engine(_flat())
    state = _position(engine)
    assert engine.describe_move(state, _push(engine, state, *_OFF)) == "Rc3>>c6"
    assert engine.describe_move(state, _push(engine, state, *_MOVED)) == "Rc3>e3"
    assert engine.describe_move(state, _push(engine, state, *_OWN_OFF)) == "Rc3>>c1"


def test_the_record_s_tally_counts_only_an_enemy_thrown_off_c1() -> None:
    """`tournament.analysis.takes`, the capture tally of the analysis pass (and of the
    legality job, which reads the same function), read off the position each push is played
    in (Q1074b5). Every push lands on an occupied square; only one that throws an enemy off
    takes, as the core's `restrict` and `is_loud` read it."""
    engine = _engine(_flat())
    state = _position(engine)
    board = state.board
    off, moved, own = (_push(engine, state, *p) for p in (_OFF, _MOVED, _OWN_OFF))
    assert takes(engine, board, off)
    assert not takes(engine, board, moved)
    assert not takes(engine, board, own)
    # The same answer as the core gives `must_capture` (C6): of every legal move, the pushes
    # the tally counts are the ones a compelled capture may play.
    counted = {move for move in engine.legal_moves(state) if takes(engine, board, move)}
    compelled = Engine({**_flat(must_capture=True)})
    assert counted == set(compelled.legal_moves(_position(compelled)))


def _minimax(engine: Engine) -> MinimaxAgent:
    agent = build_agent("minimax:nodes=100", engine)
    assert isinstance(agent, MinimaxAgent)
    return agent


def test_the_search_s_mirror_reads_only_an_enemy_thrown_off_as_loud() -> None:
    """`_is_loud` and `_capture_gain`, the oracle `agents.rs` is held to by
    `tests/test_backends.py`'s push arms."""
    engine = _engine(_flat())
    agent = _minimax(engine)
    state = _position(engine)
    board = state.board
    off, moved, own = (_push(engine, state, *p) for p in (_OFF, _MOVED, _OWN_OFF))
    assert agent._is_loud(state, off)
    assert not agent._is_loud(state, moved)
    assert not agent._is_loud(state, own)
    assert agent._capture_gain(board, off) > 0
    assert agent._capture_gain(board, moved) == 0.0
    assert agent._capture_gain(board, own) == 0.0
    spec = engine.spec
    assert spec.push_takes(board, off)
    assert not spec.push_takes(board, moved) and not spec.push_takes(board, own)


@pytest.mark.parametrize(
    "rules",
    [{"drops": {"from_captures": True}}, {"must_capture": True}],
    ids=["hand", "must_capture"],
)
def test_the_bridge_admits_a_push_beside_what_counts_a_capture(rules: dict[str, Any]) -> None:
    """Q1074b2 held these two shut until a removing push counted as a capture."""
    assert rust_supports(compile_spec(_flat(**rules))) is None
