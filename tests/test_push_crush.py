"""Crushing walls and rows on a wrapped board (Q1074b7, `docs/plan-push.md` W1, H4, C1).

Under `rules.push.walls: "crush"` a piece pushed against a wall is removed, as one pushed off
the board is, and counts as a capture where it is the enemy's (C1: `push_takes`, the clock,
the text's `>>`); never the pusher's own royal (R4). The compiler writes such a wall into
`push_next` as `PUSH_CRUSH`; the core reads it as a removal. On a wrapped board a row's walk
stops at a square it already walked: a line that comes back round to the piece landed on is a
full ring and cannot move, and one that comes round to the pusher's own square ends there,
the square being empty (§2). The core's push words and the boards they leave agree with
`tests/push_oracle.py` over seeded games, both seats, with the key checked every ply.
"""

from __future__ import annotations

import random
from typing import Any

import push_oracle
import pytest

from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import PUSH_CRUSH, PUSH_STOP, compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_PUSH, decode_move, move_push, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema


def _walled(chain: str = "one", walls: str = "crush", edge: str = "removes") -> dict[str, Any]:
    """A 7x7 board with walls on c4 and e4, no check: a rook that pushes any piece on its
    lines, a knight that pushes sideways, a pawn that pushes straight ahead and never
    captures. The lists leave the king out of every line."""
    return {
        "name": f"push crush {chain} {walls} {edge}",
        "board": {"width": 7, "height": 7},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
            ]},
            {"id": "R", "value": 5, "pushes": ["R", "N", "P", "K"], "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                 "push": {"whose": "any"}},
            ]},
            {"id": "N", "value": 3, "pushes": ["N", "P"], "moves": [
                {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8",
                 "push": {"along": "rank"}},
            ]},
            {"id": "P", "value": 1, "pushes": ["P"], "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move", "push": True},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["d1"]},
            {"piece": "R", "squares": ["a1", "g1"]},
            {"piece": "N", "squares": ["b1", "f1"]},
            {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2", "f2", "g2"]},
        ]},
        "rules": {
            "win": "royal_capture",
            "obstacles": {"squares": ["c4", "e4"]},
            "push": {"chain": chain, "walls": walls, "edge": edge},
            "max_plies": 200,
        },
    }


def _ring(chain: str = "row") -> dict[str, Any]:
    """A 5x5 board whose files wrap, no check: a rook and a knight (pushing sideways) that
    push rooks and knights, so a rank can fill and the push comes round to itself."""
    return {
        "name": f"push ring {chain}",
        "board": {"width": 5, "height": 5},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
            ]},
            {"id": "R", "value": 5, "pushes": ["R", "N"], "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                 "push": {"whose": "any"}},
            ]},
            {"id": "N", "value": 3, "pushes": ["R", "N"], "moves": [
                {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8",
                 "push": {"along": "rank", "whose": "any"}},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1", "e1"]},
            {"piece": "N", "squares": ["b1", "d1", "a2", "c2", "e2"]},
        ]},
        "rules": {
            "win": "royal_capture",
            "wrap": {"files": "both"},
            "push": {"chain": chain},
            "max_plies": 200,
        },
    }


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == [], validate_spec(spec)
    assert rust_supports(compile_spec(spec)) is None
    return Engine(spec)


def _as_oracle(engine: Engine, move: int) -> push_oracle.PushMove:
    push = engine.spec.push
    assert push is not None
    origin, target, _kind, promoted = decode_move(move)
    _player, dx, dy = push.directions[move_push(move)]
    return origin, target, (dx, dy), promoted


def _pushes_to(engine: Engine, state: GameState, origin: str, target: str) -> list[int]:
    o, t = engine._square_index(origin), engine._square_index(target)
    return [
        move for move in engine.legal_moves(state)
        if move_type(move) == MOVE_PUSH and decode_move(move)[:2] == (o, t)
    ]


def _at(engine: Engine, state: GameState, square: str) -> str | None:
    value = state.board[engine._square_index(square)]
    if not value:
        return None
    spec = engine.spec
    return f"{spec.owner_of(value)}{spec.piece_ids[spec.type_of(value)]}"


def _perft_by_apply(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _perft_by_apply(engine, engine.apply(state, move), depth - 1)
        for move in engine.legal_moves(state)
    )


# ------------------------------------------------------------------ the keys and the gates


def test_crush_is_accepted_beside_a_wall_and_refused_without_one() -> None:
    assert validate_spec(_walled()) == []
    bare = _walled()
    del bare["rules"]["obstacles"]
    assert any("walls 'crush' (W1) on a board with no wall" in e for e in validate_spec(bare))
    # `"block"` is the off value: written out, it is the game without it.
    blocked, plain = _walled(walls="block"), _walled(walls="block")
    del plain["rules"]["push"]["walls"]
    assert spec_hash(blocked) == spec_hash(plain)
    assert spec_hash(_walled()) != spec_hash(blocked)


def test_the_compiler_writes_a_crushing_wall_only_under_crush() -> None:
    crushing, blocking = compile_spec(_walled()), compile_spec(_walled(walls="block"))
    assert crushing.push is not None and blocking.push is not None
    assert any(PUSH_CRUSH in row for row in crushing.push.next)
    assert not any(PUSH_STOP in row for row in crushing.push.next)
    assert not any(PUSH_CRUSH in row for row in blocking.push.next)
    assert any(PUSH_STOP in row for row in blocking.push.next)
    # The crush changes only what a wall does to a push.
    assert [
        [square if square != PUSH_CRUSH else PUSH_STOP for square in row]
        for row in crushing.push.next
    ] == [list(row) for row in blocking.push.next]


def test_the_bridge_admits_the_crush_and_rows_on_a_wrap() -> None:
    for spec in (_walled(), _walled("row"), _ring(), _ring("one"), _weighed_ring()):
        assert validate_spec(spec) == [], validate_spec(spec)
        assert rust_supports(compile_spec(spec)) is None


def test_the_schema_and_the_description_offer_the_crush() -> None:
    push = rule_spec_schema()["properties"]["rules"]["properties"]["push"]
    assert push["properties"]["walls"]["enum"] == ["block", "crush"]
    assert "is crushed" in describe_spec(_walled())
    assert "A wall blocks a push" in describe_spec(_walled(walls="block"))


# --------------------------------------------------------------------------- the crush


def test_a_piece_pressed_on_a_wall_is_crushed_and_counted() -> None:
    """W1, C1: blocked under `"block"`; under `"crush"` the piece is removed, the push takes
    it (the clock resets, the text writes `>>`), and the material is one piece less."""
    placement = {
        "a1": ("K", 0), "a4": ("R", 0), "b4": ("P", 1), "g7": ("K", 1),
    }
    blocking = _engine(_walled(walls="block"))
    state = blocking.make_state(placement, side_to_move=0, halfmove=7)
    assert _pushes_to(blocking, state, "a4", "b4") == []
    engine = _engine(_walled())
    state = engine.make_state(placement, side_to_move=0, halfmove=7)
    [push] = _pushes_to(engine, state, "a4", "b4")
    assert engine.describe_move(state, push) == "Ra4>>b4"
    assert engine.spec.push_takes(state.board, push)
    after = engine.apply(state, push)
    assert [_at(engine, after, s) for s in ("a4", "b4")] == [None, "0R"]
    assert sum(map(bool, after.board)) == sum(map(bool, state.board)) - 1
    assert after.halfmove == 0
    assert engine.hash(after) == engine._compute_hash(after)
    # Under `edge: "blocks"` the wall still crushes; the edge alone blocks.
    edged = _engine(_walled(edge="blocks"))
    state = edged.make_state(placement, side_to_move=0)
    assert len(_pushes_to(edged, state, "a4", "b4")) == 1


def test_a_pusher_s_own_piece_is_crushed_but_not_taken_and_its_royal_never() -> None:
    engine = _engine(_walled())
    own = engine.make_state({
        "a1": ("K", 0), "a4": ("R", 0), "b4": ("P", 0), "g7": ("K", 1),
    }, side_to_move=0, halfmove=7)
    [push] = _pushes_to(engine, own, "a4", "b4")
    assert not engine.spec.push_takes(own.board, push)
    after = engine.apply(own, push)
    assert _at(engine, after, "b4") == "0R" and after.halfmove == 0
    royal = engine.make_state({
        "a1": ("R", 0), "a4": ("R", 0), "b4": ("K", 0), "g7": ("K", 1),
    }, side_to_move=0)
    assert _pushes_to(engine, royal, "a4", "b4") == []


def test_a_row_is_crushed_at_its_last_piece() -> None:
    engine = _engine(_walled("row"))
    state = engine.make_state({
        "a1": ("K", 0), "g4": ("R", 0), "f4": ("P", 1), "d4": ("N", 1), "g7": ("K", 1),
    }, side_to_move=0)
    # g4 onto f4: the line is f4 alone, pressed on the wall on e4.
    [push] = _pushes_to(engine, state, "g4", "f4")
    assert engine.describe_move(state, push) == "Rg4>>f4"
    after = engine.apply(state, push)
    assert [_at(engine, after, s) for s in ("g4", "f4", "d4")] == [None, "0R", "1N"]
    capped = engine.make_state({
        "a1": ("K", 0), "c7": ("R", 0), "c6": ("P", 1), "c5": ("N", 1), "g7": ("K", 1),
    }, side_to_move=0)
    # c7 onto c6 pushes c6 and c5 down the c-file; c5 is pressed on the wall on c4.
    [push] = _pushes_to(engine, capped, "c7", "c6")
    assert engine.spec.push_takes(capped.board, push)
    assert engine.spec.push_taken(capped.board, push) == capped.board[engine._square_index("c5")]
    after = engine.apply(capped, push)
    assert [_at(engine, after, s) for s in ("c7", "c6", "c5")] == [None, "0R", "1P"]
    assert engine.perft(capped, 3) == _perft_by_apply(engine, capped, 3)


# ---------------------------------------------------------------------------- the ring


def test_a_full_ring_cannot_move() -> None:
    """H4: the knight on a1 lands on b3, pushing along rank 3, which every square of fills.
    The line comes back round to b3: no sink, no push. Empty one square and it moves."""
    engine = _engine(_ring())
    full = {
        "c1": ("K", 0), "a1": ("N", 0), "a3": ("R", 1), "b3": ("N", 1), "c3": ("R", 1),
        "d3": ("N", 1), "e3": ("R", 1), "c5": ("K", 1),
    }
    state = engine.make_state(full, side_to_move=0)
    assert _pushes_to(engine, state, "a1", "b3") == []
    spec = _ring()
    decode = _decoder(engine)
    assert not any(
        push[:2] == (engine._square_index("a1"), engine._square_index("b3"))
        for push in push_oracle.push_moves(spec, decode, state.board, 0)
    )
    gap = dict(full)
    del gap["e3"]
    state = engine.make_state(gap, side_to_move=0)
    [push] = _pushes_to(engine, state, "a1", "b3")  # rightwards, the jump's own sign
    after = engine.apply(state, push)
    assert bytes(after.board) == push_oracle.after_push(
        spec, state.board, _as_oracle(engine, push)
    )
    assert [_at(engine, after, s) for s in ("a3", "b3", "c3", "d3", "e3")] == [
        "1R", "0N", "1N", "1R", "1N",
    ]


def test_a_ring_ending_at_the_pusher_s_origin_moves() -> None:
    """H4: the rook on a3 pushes b3 rightwards; the line b3, c3, d3, e3 wraps round onto a3,
    the rook's own square, which it leaves in the same move. Both ways round."""
    engine = _engine(_ring())
    state = engine.make_state({
        "c1": ("K", 0), "a3": ("R", 0), "b3": ("N", 1), "c3": ("R", 1), "d3": ("N", 0),
        "e3": ("R", 1), "c5": ("K", 1),
    }, side_to_move=0, halfmove=7)
    [right] = _pushes_to(engine, state, "a3", "b3")
    assert not engine.spec.push_takes(state.board, right)
    # On a joined board one landing can be pushed two ways, so the text names the way.
    assert engine.describe_move(state, right) == "Ra3>b3 (to c3)"
    after = engine.apply(state, right)
    assert [_at(engine, after, s) for s in ("a3", "b3", "c3", "d3", "e3")] == [
        "1R", "0R", "1N", "1R", "0N",
    ]
    assert after.halfmove == 8
    assert engine.hash(after) == engine._compute_hash(after)
    [left] = _pushes_to(engine, state, "a3", "e3")
    after = engine.apply(state, left)
    assert [_at(engine, after, s) for s in ("a3", "b3", "c3", "d3", "e3")] == [
        "1N", "1R", "0N", "1R", "0R",
    ]
    assert engine.perft(state, 3) == _perft_by_apply(engine, state, 3)


# ---------------------------------------------------------------- the oracle, move by move


def _decoder(engine: Engine) -> Any:
    spec = engine.spec
    return lambda value: (spec.owner_of(value), spec.piece_ids[spec.type_of(value)])


def _weighed_ring() -> dict[str, Any]:
    spec = _ring("weight")
    for piece in spec["pieces"]:
        if piece["id"] != "K":
            piece["weight"] = {"R": 2, "N": 1}[piece["id"]]
        if piece["id"] == "R":
            piece["strength"] = 4
    return spec


CASES = {
    "crush-one": lambda: _walled(),
    "crush-row": lambda: _walled("row"),
    "crush-row-blocks": lambda: _walled("row", edge="blocks"),
    "ring-one": lambda: _ring("one"),
    "ring-row": lambda: _ring(),
    "ring-weight": _weighed_ring,
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_core_equals_the_oracle_over_seeded_games(name: str) -> None:
    """Both seats, every ply: the core's push words are the oracle's, every push leaves the
    board the oracle says, and the key is the one computed from scratch. The games reach the
    case's own reading: a crush on a wall, or a line wrapped round onto the pusher's square."""
    spec = CASES[name]()
    engine = _engine(spec)
    decode = _decoder(engine)
    pushes_seen = removed = reached = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(100):
            player = state.side_to_move
            expected = push_oracle.push_moves(spec, decode, state.board, player)
            core = {
                _as_oracle(engine, move) for move in engine.pseudo_legal_moves(state)
                if move_type(move) == MOVE_PUSH
            }
            assert core == expected, (seed, ply)
            pushes_seen += len(expected)
            if engine.terminal(state) is not None:
                break
            legal = engine.legal_moves(state)
            pushes = [move for move in legal if move_type(move) == MOVE_PUSH]
            move = rng.choice(pushes if pushes and rng.random() < 0.7 else legal)
            after = engine.apply(state, move)
            if move_type(move) == MOVE_PUSH:
                last = engine.spec.push_line(state.board, move)[-1][1]
                reached += last == (
                    PUSH_CRUSH if name.startswith("crush") else decode_move(move)[0]
                )
            if move_type(move) == MOVE_PUSH and not decode_move(move)[3]:
                oracle = push_oracle.after_push(spec, state.board, _as_oracle(engine, move))
                assert bytes(after.board) == oracle, (seed, ply)
                gone = sum(map(bool, state.board)) - sum(map(bool, after.board))
                assert gone == (engine.spec.push_removed(state.board, move) != 0)
                removed += gone
            state = after
            assert engine.hash(state) == engine._compute_hash(state)
    assert pushes_seen > 100 and removed > 0 and reached > 0, (pushes_seen, removed, reached)

