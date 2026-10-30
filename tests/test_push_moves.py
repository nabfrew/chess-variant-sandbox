"""The single push in the core (Q1074b2, `docs/plan-push.md` §1 to §3, K1-K4, R3-R5, E1, E3, N1-N3).

`MOVE_PUSH` words are generated and played by the core and agree with `tests/push_oracle.py`
over seeded games of two push games, both seats, with the incremental key checked every ply
and `make`/`unmake` held to `apply` by perft; a push that puts the pushed piece where it
checks the pusher's royal is refused (R5); a pushed piece loses its first-move rights, so a
pushed rook does not castle (E3); a wild piece and the duck block (N1, N2); a push that
throws one's own royal off the board is no move (R4); `rust_supports` admits the round-1
single push and holds what this stage does not play.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import push_oracle
import pytest

from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_PUSH,
    decode_move,
    move_push,
    move_type,
)
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import validate_spec

ROOT = Path(__file__).resolve().parents[1]


def _sumo() -> dict[str, Any]:
    """Chess with every non-royal capture a push (enemy only, never a capture), the knight
    pushing along the rank as sumo chess's does, and no en passant (§6.1's reading)."""
    spec: dict[str, Any] = json.loads((ROOT / "rulesets/chess.json").read_text(encoding="utf-8"))
    spec["name"] = "sumo test"
    for piece in spec["pieces"]:
        if piece.get("royal"):
            continue
        for move in piece["moves"]:
            if move.get("mode") != "move":
                move["push"] = {"along": "rank"} if piece["id"] == "N" else True
    del spec["rules"]["en_passant"]
    return spec


def _ring() -> dict[str, Any]:
    """A small cylinder (files joined) with no check: a rook that pushes any piece and also
    captures, a knight that pushes pawns and knights sideways, a pawn that pushes on its
    diagonals and promotes by its landing."""
    return {
        "name": "push ring",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
            ]},
            {"id": "R", "value": 5, "moves": [
                {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
                 "push": {"whose": "any", "captures": True}},
            ]},
            {"id": "N", "value": 3, "pushes": ["P", "N"], "moves": [
                {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8",
                 "push": {"along": "rank"}},
            ]},
            {"id": "P", "value": 1, "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture",
                 "push": True},
            ]},
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1", "f1"]},
            {"piece": "N", "squares": ["b1", "e1"]},
            {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2", "f2"]},
        ]},
        "rules": {
            "win": "royal_capture",
            "wrap": {"files": "both"},
            "promotion": {"piece": "P", "rank": "last", "to": ["R", "N"]},
            "max_plies": 200,
        },
    }


def _core_pushes(engine: Engine, state: GameState) -> set[push_oracle.PushMove]:
    push = engine.spec.push
    assert push is not None
    found = set()
    for move in engine.pseudo_legal_moves(state):
        if move_type(move) != MOVE_PUSH:
            continue
        origin, target, _kind, promoted = decode_move(move)
        _player, dx, dy = push.directions[move_push(move)]
        found.add((origin, target, (dx, dy), promoted))
    return found


def _decoder(engine: Engine) -> Any:
    spec = engine.spec
    return lambda value: (spec.owner_of(value), spec.piece_ids[spec.type_of(value)])


@pytest.mark.parametrize("make", [_sumo, _ring], ids=["sumo", "ring"])
def test_push_moves_equal_the_oracle_s_over_seeded_games(make: Any) -> None:
    """Both seats, every ply: the core's push words are the oracle's, the key is the one
    computed from scratch, and every push's text names it uniquely."""
    spec = make()
    assert validate_spec(spec) == []
    engine = Engine(spec)
    decode = _decoder(engine)
    seen = {0: 0, 1: 0}
    removed = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = engine.initial_state()
        for _ply in range(90):
            player = state.side_to_move
            expected = push_oracle.push_moves(spec, decode, state.board, player)
            assert _core_pushes(engine, state) == expected, (seed, _ply)
            seen[player] += len(expected)
            if engine.terminal(state) is not None:
                break
            legal = engine.legal_moves(state)
            texts = [engine.describe_move(state, move) for move in legal]
            assert len(set(texts)) == len(texts)
            pushes = [move for move in legal if move_type(move) == MOVE_PUSH]
            # Lean on the pushes, so the games are about them.
            move = rng.choice(pushes if pushes and rng.random() < 0.7 else legal)
            before = sum(1 for value in state.board if value)
            state = engine.apply(state, move)
            removed += before - sum(1 for value in state.board if value)
            assert engine.hash(state) == engine._compute_hash(state)
    assert seen[0] > 50 and seen[1] > 50
    assert removed > 0


def _perft_by_apply(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _perft_by_apply(engine, engine.apply(state, move), depth - 1)
        for move in engine.legal_moves(state)
    )


def test_make_and_unmake_agree_with_apply_in_a_push_game() -> None:
    """The core's perft plays and takes back every push in place; counting the same tree
    through `apply`, which plays each move on a fresh copy, must give the same number."""
    engine = Engine(_ring())
    state = engine.initial_state()
    rng = random.Random(7)
    for _ in range(12):
        legal = engine.legal_moves(state)
        pushes = [move for move in legal if move_type(move) == MOVE_PUSH]
        state = engine.apply(state, rng.choice(pushes or legal))
    assert any(move_type(m) == MOVE_PUSH for m in engine.legal_moves(state))
    assert engine.perft(state, 3) == _perft_by_apply(engine, state, 3)


def _sumo_engine() -> Engine:
    return Engine(_sumo())


def _push_to(engine: Engine, state: GameState, origin: str, target: str) -> list[int]:
    o, t = engine._square_index(origin), engine._square_index(target)
    return [
        move for move in engine.pseudo_legal_moves(state)
        if move_type(move) == MOVE_PUSH and decode_move(move)[:2] == (o, t)
    ]


def test_a_push_that_puts_the_pushed_piece_where_it_checks_is_refused_r5() -> None:
    """The rook's push throws the black knight from b4 to b3, where it checks the white king
    on a1: offered before the check filter, refused by it, though nothing the pin shortcut
    reads moved (the rook was pinned by nothing, the king was not in check)."""
    engine = _sumo_engine()
    state = engine.make_state(
        {"a1": ("K", 0), "b8": ("R", 0), "b4": ("N", 1), "h8": ("K", 1)}, side_to_move=0,
    )
    assert not engine.in_check(state)
    offered = _push_to(engine, state, "b8", "b4")
    assert len(offered) == 1
    assert offered[0] not in engine.legal_moves(state)
    # The same push with the knight thrown somewhere harmless is legal.
    elsewhere = engine.make_state(
        {"h1": ("K", 0), "b8": ("R", 0), "b4": ("N", 1), "h8": ("K", 1)}, side_to_move=0,
    )
    assert _push_to(engine, elsewhere, "b8", "b4")[0] in engine.legal_moves(elsewhere)


def test_a_push_threat_is_check_r1() -> None:
    """A rook that may push the king attacks it: the side to move is in check."""
    engine = _sumo_engine()
    state = engine.make_state({"a1": ("K", 0), "a8": ("R", 1), "h8": ("K", 1)}, side_to_move=0)
    assert engine.in_check(state)


def _castles(engine: Engine, state: GameState) -> list[int]:
    return [move for move in engine.legal_moves(state) if move_type(move) == MOVE_CASTLE]


def test_a_pushed_piece_loses_its_first_move_rights_e1_e3() -> None:
    """A pushed piece stands where the push put it with no `unmoved` bit: it made no move of
    its own, but it has left its square, so no double step and no castling."""
    engine = _sumo_engine()
    state = engine.make_state(
        {"e1": ("K", 0), "e4": ("R", 1), "e8": ("K", 1), "c4": ("B", 0)}, side_to_move=1,
    )
    # The black rook on e4 pushes the white bishop on c4 to b4 (along the rank, westwards).
    push = _push_to(engine, state, "e4", "c4")
    assert len(push) == 1
    after = engine.apply(state, push[0])
    b4 = engine._square_index("b4")
    assert after.board[b4] == state.board[engine._square_index("c4")]
    assert not after.unmoved >> b4 & 1
    assert not after.unmoved >> engine._square_index("c4") & 1


def test_a_pushed_rook_does_not_castle_e3() -> None:
    """A cylinder lets a rook be pushed off its castling square and back onto it: the rook on
    h1 is pushed round the seam to a1 and home again by its own side's queen-like rook, and
    castling, offered before, is not offered after, the rook back where it started."""
    spec = _sumo()
    spec["rules"]["wrap"] = {"files": "both"}
    spec["rules"]["castling"] = {"king": "K", "rook": "R", "king_steps": 2}
    for piece in spec["pieces"]:
        if piece["id"] == "Q":
            for move in piece["moves"]:
                move["push"] = {"whose": "any"}
    engine = Engine(spec)
    placement = {"e1": ("K", 0), "h1": ("R", 0), "e8": ("K", 1), "g4": ("Q", 0)}
    before = engine.make_state(placement, side_to_move=0)
    assert _castles(engine, before)
    # The queen on b1 pushes the rook on a1 west, round the seam, home to h1.
    pushed = engine.make_state(
        {"e1": ("K", 0), "a1": ("R", 0), "e8": ("K", 1), "c1": ("Q", 0)},
        side_to_move=0,
    )
    push = _push_to(engine, pushed, "c1", "a1")
    assert len(push) == 1
    home = engine.apply(pushed, push[0])
    h1 = engine._square_index("h1")
    assert home.board[h1] == before.board[h1]
    # The queen now stands on a1, so it is not in the castling's way; it is the black king's
    # turn: let it step and hand the move back.
    king_step = next(
        move for move in engine.legal_moves(home)
        if decode_move(move)[0] == engine._square_index("e8")
    )
    back = engine.apply(home, king_step)
    assert not _castles(engine, back)


def test_wild_pieces_and_the_duck_block_n1_n2() -> None:
    """A wild piece on the target is never pushed, and one behind the target blocks; the duck
    likewise."""
    spec = _ring()
    spec["rules"].pop("wrap")
    spec["pieces"].append({"id": "W", "value": 1, "moves": []})
    spec["rules"]["npcs"] = [{"piece": "W", "route": ["c4"]}]
    assert validate_spec(spec) == []
    engine = Engine(spec)
    state = engine.make_state(
        {"a1": ("K", 0), "c1": ("R", 0), "c3": ("P", 1), "f6": ("K", 1), "c4": ("W", 2),
         "a4": ("R", 0)},
        side_to_move=0,
    )
    decode = _decoder(engine)
    assert _core_pushes(engine, state) == push_oracle.push_moves(spec, decode, state.board, 0)
    assert not _push_to(engine, state, "c1", "c3"), "the wild piece behind the pawn blocks"
    assert not _push_to(engine, state, "a4", "c4"), "the wild piece itself is never pushed"
    duck = _ring()
    duck["rules"].pop("wrap")
    duck["pieces"].append({"id": "D", "value": 0, "moves": []})
    duck["rules"]["duck"] = {"piece": "D", "start": "c4"}
    assert validate_spec(duck) == []
    engine = Engine(duck)
    state = engine.make_state(
        {"a1": ("K", 0), "c1": ("R", 0), "c3": ("P", 1), "f6": ("K", 1), "c4": ("D", 2)},
        side_to_move=0,
    )
    assert not _push_to(engine, state, "c1", "c3"), "the duck behind the pawn blocks"


def test_one_s_own_royal_is_never_pushed_off_the_board_r4() -> None:
    engine = Engine(_ring())
    state = engine.make_state(
        {"c1": ("K", 0), "c3": ("R", 0), "f6": ("K", 1), "a3": ("R", 1)}, side_to_move=0,
    )
    # The rook on c3 slides down onto its own king on c1 (past c2) and would throw it off;
    # onto its own pawn there, it may.
    assert not _push_to(engine, state, "c3", "c1")
    pawn = engine.make_state(
        {"a1": ("K", 0), "c1": ("P", 0), "c3": ("R", 0), "f6": ("K", 1)}, side_to_move=0,
    )
    assert len(_push_to(engine, pawn, "c3", "c1")) == 1


def test_a_push_reads_as_a_push() -> None:
    """`>` for a push, `>>` for one that throws the piece off, never the capture's `x`; on a
    joined board, where one landing can be pushed two ways, the text names where it goes."""
    ring = Engine(_ring())
    state = ring.make_state(
        {"c1": ("K", 0), "c3": ("R", 0), "c5": ("P", 1), "f6": ("K", 1)}, side_to_move=0,
    )
    [push] = _push_to(ring, state, "c3", "c5")
    assert ring.describe_move(state, push) == "Rc3>c5 (to c6)"
    flat = _ring()
    flat["rules"].pop("wrap")
    engine = Engine(flat)
    state = engine.make_state(
        {"c1": ("K", 0), "c3": ("R", 0), "c5": ("P", 1), "f6": ("K", 1)}, side_to_move=0,
    )
    [push] = _push_to(engine, state, "c3", "c5")
    assert engine.describe_move(state, push) == "Rc3>c5"
    edge = engine.make_state(
        {"c1": ("K", 0), "c3": ("R", 0), "c6": ("P", 1), "f6": ("K", 1)}, side_to_move=0,
    )
    [off] = _push_to(engine, edge, "c3", "c6")
    assert engine.describe_move(edge, off) == "Rc3>>c6"


def test_the_bridge_admits_the_single_push_and_holds_the_rest() -> None:
    sumo = compile_spec(_sumo())
    assert rust_supports(sumo) is None
    assert rust_supports(compile_spec(_ring())) is None
    # Q1074b4 lifted rows and weights off a wrapped board (`tests/test_push_rows.py`).
    row = copy.deepcopy(_ring())
    row["rules"].pop("wrap")
    row["rules"]["push"] = {"chain": "row"}
    assert rust_supports(compile_spec(row)) is None
    # §3.14's rows, which the validator refuses, held here too should one reach the compiler.
    for field, value, words in [
        ("strike", object(), "push.X2"),
        ("explosion", object(), "push.X3"),
        ("simultaneous", object(), "push.S1"),
        ("sight", object(), "push.N4"),
        ("rest", (1,) * sumo.n_types, "push.N5"),
        ("reach", object(), "push.G1"),
        ("territory", object(), "push.G2"),
        ("purse", object(), "push.M1"),
    ]:
        held = rust_supports(dataclasses.replace(sumo, **{field: value}))
        assert held is not None and words in held, field
    # Q1074b3 lifted `must_capture` (and the capture hand, `tests/test_push_captures.py`).
    assert rust_supports(dataclasses.replace(sumo, must_capture=True)) is None
