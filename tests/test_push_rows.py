"""Rows and weights in the core (Q1074b4, `docs/plan-push.md` §2, K2, C5).

Under `rules.push.chain: "row"` a push moves the whole line, the piece landed on and every
piece behind it up to the first empty square, each one step on, if the pusher may push every
piece in it; the last falls off at the edge. Under `"weight"` the line moves only where the
pusher's strength is at least the line's summed weight (the user, 2026-10-30: greater or
equal; Q1074b4s parted a pusher's strength from its weight, which it defaults to). The
core's push words and the boards they leave agree with `tests/push_oracle.py` over seeded
games, both seats, with the incremental key checked every ply; `make`/`unmake` agree with
`apply` by perft; a row's last piece thrown off is a capture where it is the enemy's
(`Tables::push_takes`, `CompiledSpec.push_takes`), and the text writes `>>` for it.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import push_oracle
import pytest

from chess_variant_sandbox.engine.bridge import rust_supports
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import Engine
from chess_variant_sandbox.engine.moves import MOVE_PUSH, decode_move, move_push, move_type
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import describe_spec
from chess_variant_sandbox.spec.jsonschema import rule_spec_schema
from chess_variant_sandbox.spec.normal import normal_form


def _army(weighed: bool) -> dict[str, Any]:
    """A 7x7 board, no check: a rook that pushes any piece's row on its lines, a knight that
    pushes sideways, a pawn that pushes straight ahead and never captures. Under ``"row"`` the
    lists leave the king out of every row; under ``"weight"`` the weights say it all."""
    rook: dict[str, Any] = {"id": "R", "value": 5, "moves": [
        {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4",
         "push": {"whose": "any"}},
    ]}
    knight: dict[str, Any] = {"id": "N", "value": 3, "moves": [
        {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8",
         "push": {"along": "rank"}},
    ]}
    pawn: dict[str, Any] = {"id": "P", "value": 1, "moves": [
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move", "push": True},
    ]}
    king: dict[str, Any] = {"id": "K", "royal": True, "value": 0, "moves": [
        {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
    ]}
    if weighed:
        king["weight"], rook["weight"], knight["weight"], pawn["weight"] = 2, 3, 2, 1
    else:
        rook["pushes"], knight["pushes"], pawn["pushes"] = ["R", "N", "P"], ["N", "P"], ["P"]
    return {
        "name": "push rows " + ("weight" if weighed else "row"),
        "board": {"width": 7, "height": 7},
        "pieces": [king, rook, knight, pawn],
        "setup": {"placements": [
            {"piece": "K", "squares": ["d1"]},
            {"piece": "R", "squares": ["a1", "g1"]},
            {"piece": "N", "squares": ["b1", "f1"]},
            {"piece": "P", "squares": ["a2", "b2", "c2", "d2", "e2", "f2", "g2"]},
        ]},
        "rules": {
            "win": "royal_capture",
            "push": {"chain": "weight" if weighed else "row"},
            "max_plies": 200,
        },
    }


def _circle() -> dict[str, Any]:
    """Three types each pushing only the next (A pushes B, B pushes C, C pushes A) by a rook's
    step, under ``"row"``: a circular hierarchy by lists (§1.4)."""
    def stepper(piece_id: str, pushes: str) -> dict[str, Any]:
        return {"id": piece_id, "value": 2, "pushes": [pushes], "moves": [
            {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4",
             "push": {"whose": "any"}},
        ]}

    return {
        "name": "push circle",
        "board": {"width": 6, "height": 6},
        "pieces": [
            {"id": "K", "royal": True, "value": 0, "moves": [
                {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "dihedral8"},
            ]},
            stepper("A", "B"), stepper("B", "C"), stepper("C", "A"),
        ],
        "setup": {"placements": [
            {"piece": "K", "squares": ["a1"]},
            {"piece": "A", "squares": ["b1", "e2"]},
            {"piece": "B", "squares": ["c1", "b2", "f2"]},
            {"piece": "C", "squares": ["d1", "c2", "d2"]},
        ]},
        "rules": {"win": "royal_capture", "push": {"chain": "row"}, "max_plies": 200},
    }


def _strong(rook: tuple[int, int] = (1, 6), pawn: int = 3) -> dict[str, Any]:
    """`_army` under ``"weight"`` with strength apart from weight (Q1074b4s): the rook light
    and strong (weight 1, strength 6 by default), the pawn weight 1 and strength 3, the
    knight's strength its weight."""
    spec = _army(True)
    spec["name"] = "push rows strength"
    by_id = {piece["id"]: piece for piece in spec["pieces"]}
    by_id["R"]["weight"], by_id["R"]["strength"] = rook
    by_id["P"]["strength"] = pawn
    return spec


def _engine(spec: dict[str, Any]) -> Engine:
    assert validate_spec(spec) == []
    assert rust_supports(compile_spec(spec)) is None
    return Engine(spec)


def _decoder(engine: Engine) -> Any:
    spec = engine.spec
    return lambda value: (spec.owner_of(value), spec.piece_ids[spec.type_of(value)])


def _as_oracle(engine: Engine, move: int) -> push_oracle.PushMove:
    push = engine.spec.push
    assert push is not None
    origin, target, _kind, promoted = decode_move(move)
    _player, dx, dy = push.directions[move_push(move)]
    return origin, target, (dx, dy), promoted


def _core_pushes(engine: Engine, state: GameState) -> set[push_oracle.PushMove]:
    return {
        _as_oracle(engine, move) for move in engine.pseudo_legal_moves(state)
        if move_type(move) == MOVE_PUSH
    }


def _line_length(engine: Engine, state: GameState, move: int) -> int:
    """How many pieces the push ``move`` moves (or throws off), read by walking the squares
    behind its target on the board before it."""
    spec = engine.spec
    assert spec.push is not None
    origin, square = decode_move(move)[:2]
    following = spec.push.next[move_push(move)]
    length = 1
    while following[square] >= 0 and following[square] != origin and state.board[following[square]]:
        square = following[square]
        length += 1
    return length


@pytest.mark.parametrize("make", [lambda: _army(False), lambda: _army(True), _strong, _circle],
                         ids=["row", "weight", "strength", "circle"])
def test_rows_equal_the_oracle_s_over_seeded_games(make: Any) -> None:
    """Both seats, every ply: the core's push words are the oracle's, every push leaves the
    board the oracle says, and the key is the one computed from scratch."""
    spec = make()
    engine = _engine(spec)
    decode = _decoder(engine)
    seen = {0: 0, 1: 0}
    longest = removed = 0
    for seed in range(4):
        rng = random.Random(seed)
        state = engine.initial_state()
        for ply in range(120):
            player = state.side_to_move
            expected = push_oracle.push_moves(spec, decode, state.board, player)
            assert _core_pushes(engine, state) == expected, (seed, ply)
            seen[player] += len(expected)
            if engine.terminal(state) is not None:
                break
            legal = engine.legal_moves(state)
            texts = [engine.describe_move(state, move) for move in legal]
            assert len(set(texts)) == len(texts)
            pushes = [move for move in legal if move_type(move) == MOVE_PUSH]
            # Lean on the longest pushes, so the games are about rows.
            if pushes and rng.random() < 0.8:
                most = max(_line_length(engine, state, move) for move in pushes)
                move = rng.choice(
                    [m for m in pushes if _line_length(engine, state, m) == most]
                )
            else:
                move = rng.choice(legal)
            after = engine.apply(state, move)
            if move_type(move) == MOVE_PUSH and not decode_move(move)[3]:
                longest = max(longest, _line_length(engine, state, move))
                oracle = push_oracle.after_push(spec, state.board, _as_oracle(engine, move))
                assert bytes(after.board) == oracle, (seed, ply)
            removed += sum(map(bool, state.board)) - sum(map(bool, after.board))
            state = after
            assert engine.hash(state) == engine._compute_hash(state)
    assert seen[0] > 50 and seen[1] > 50
    assert longest >= 2 and removed > 0


def _perft_by_apply(engine: Engine, state: GameState, depth: int) -> int:
    if depth == 0:
        return 1
    return sum(
        _perft_by_apply(engine, engine.apply(state, move), depth - 1)
        for move in engine.legal_moves(state)
    )


@pytest.mark.parametrize("weighed", [False, True], ids=["row", "weight"])
def test_undo_restores_every_square_of_a_row(weighed: bool) -> None:
    """The core's perft plays and takes back every row in place; counting the same tree
    through `apply`, on a fresh copy each move, gives the same number."""
    engine = _engine(_army(weighed))
    state = engine.make_state({
        "a1": ("K", 0), "a4": ("R", 0), "c4": ("P", 1), "d4": ("P", 1), "e4": ("P", 0),
        "g4": ("P", 1), "d2": ("P", 0), "d3": ("P", 1), "g7": ("K", 1), "b6": ("R", 1),
        "e6": ("P", 0), "e5": ("N", 0),
    }, side_to_move=0, moved=["a4", "c4", "d4", "e4", "g4", "d2", "d3", "b6", "e6", "e5"])
    assert max(
        _line_length(engine, state, move) for move in engine.legal_moves(state)
        if move_type(move) == MOVE_PUSH
    ) >= 3
    assert engine.perft(state, 3) == _perft_by_apply(engine, state, 3)


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


def test_a_row_of_three_moves_and_its_last_falls_off() -> None:
    engine = _engine(_army(False))
    state = engine.make_state({
        "a1": ("K", 0), "a4": ("R", 0), "e4": ("P", 1), "f4": ("N", 1), "g4": ("R", 1),
        "g7": ("K", 1),
    }, side_to_move=0, halfmove=7)
    [push] = _pushes_to(engine, state, "a4", "e4")
    assert engine.describe_move(state, push) == "Ra4>>e4"
    assert engine.spec.push_takes(state.board, push)
    after = engine.apply(state, push)
    assert [_at(engine, after, s) for s in ("a4", "e4", "f4", "g4")] == [None, "0R", "1P", "1N"]
    assert sum(map(bool, after.board)) == sum(map(bool, state.board)) - 1
    assert after.halfmove == 0
    # A row that ends at an empty square moves whole and takes nothing.
    short = engine.make_state({
        "a1": ("K", 0), "a4": ("R", 0), "c4": ("P", 1), "d4": ("N", 1), "g7": ("K", 1),
    }, side_to_move=0, halfmove=7)
    [push] = _pushes_to(engine, short, "a4", "c4")
    assert engine.describe_move(short, push) == "Ra4>c4"
    assert not engine.spec.push_takes(short.board, push)
    after = engine.apply(short, push)
    assert [_at(engine, after, s) for s in ("c4", "d4", "e4")] == ["0R", "1P", "1N"]
    assert after.halfmove == 8


def test_a_row_with_one_unlisted_piece_is_refused() -> None:
    """The rook's list leaves the king out, so a row with a king in it does not move, nor
    does a row the pawn's list does not cover (it pushes pawns only)."""
    engine = _engine(_army(False))
    state = engine.make_state({
        "a1": ("K", 0), "a5": ("R", 0), "e5": ("P", 1), "f5": ("K", 1), "c2": ("P", 0),
        "c3": ("P", 1), "c4": ("N", 1),
    }, side_to_move=0)
    assert _pushes_to(engine, state, "a5", "e5") == []
    assert _pushes_to(engine, state, "c2", "c3") == []
    # Without the king, and with the knight gone from behind the pawn, both move.
    free = engine.make_state({
        "a1": ("K", 0), "a5": ("R", 0), "e5": ("P", 1), "f5": ("P", 1), "c2": ("P", 0),
        "c3": ("P", 1), "c4": ("P", 1), "g7": ("K", 1),
    }, side_to_move=0)
    assert len(_pushes_to(engine, free, "a5", "e5")) == 1
    assert len(_pushes_to(engine, free, "c2", "c3")) == 1


def test_a_weight_three_pusher_moves_one_and_two_but_not_two_and_two() -> None:
    engine = _engine(_army(True))

    def rook_pushes(behind: str) -> list[int]:
        state = engine.make_state({
            "a1": ("K", 0), "a4": ("R", 0), "c4": (behind, 1), "d4": ("N", 1), "g7": ("K", 1),
        }, side_to_move=0)
        return _pushes_to(engine, state, "a4", "c4")

    assert len(rook_pushes("P")) == 1  # 1 + 2 = 3: at least the line's weight
    assert rook_pushes("N") == []  # 2 + 2 = 4


def test_a_weighed_row_never_throws_the_pusher_s_own_royal_off_r4() -> None:
    engine = _engine(_army(True))
    state = engine.make_state({
        "a4": ("R", 0), "f4": ("P", 0), "g4": ("K", 0), "a7": ("K", 1),
    }, side_to_move=0)
    assert _pushes_to(engine, state, "a4", "f4") == []
    # One square further from the edge, the same row moves.
    inside = engine.make_state({
        "a4": ("R", 0), "e4": ("P", 0), "f4": ("K", 0), "a7": ("K", 1),
    }, side_to_move=0)
    assert len(_pushes_to(engine, inside, "a4", "e4")) == 1


def test_a_circular_hierarchy_plays_under_row() -> None:
    engine = _engine(_circle())
    state = engine.make_state({
        "a1": ("K", 0), "a3": ("A", 0), "b3": ("B", 1), "c3": ("B", 1), "f6": ("K", 1),
        "a5": ("B", 0), "b5": ("C", 1), "c5": ("A", 1),
    }, side_to_move=0)
    assert len(_pushes_to(engine, state, "a3", "b3")) == 1  # A pushes a row of Bs
    assert _pushes_to(engine, state, "a5", "b5") == []  # B pushes C but not the A behind it
    after = engine.apply(state, _pushes_to(engine, state, "a3", "b3")[0])
    assert [_at(engine, after, s) for s in ("b3", "c3", "d3")] == ["0A", "1B", "1B"]


def test_a_row_whose_last_piece_is_the_pusher_s_own_takes_nothing() -> None:
    """`push_takes` reads the line's last piece, not the target's: an enemy row ending in the
    pusher's own piece at the edge is no capture, though it removes a piece."""
    engine = _engine(_army(False) | {"rules": {
        "win": "royal_capture", "push": {"chain": "row"}, "max_plies": 200,
        "must_capture": True,
    }})
    state = engine.make_state({
        "a1": ("K", 0), "a4": ("R", 0), "f4": ("P", 1), "g4": ("P", 0), "a6": ("R", 0),
        "f6": ("P", 0), "g6": ("P", 1), "g1": ("K", 1),
    }, side_to_move=0)
    own = _pushes_to(engine, state, "a4", "f4")
    enemy = _pushes_to(engine, state, "a6", "f6")
    assert own == [] and len(enemy) == 1  # must_capture keeps only the one that takes
    pseudo = [
        move for move in engine.pseudo_legal_moves(state)
        if move_type(move) == MOVE_PUSH
        and decode_move(move)[:2] == (engine._square_index("a4"), engine._square_index("f4"))
    ]
    assert len(pseudo) == 1 and not engine.spec.push_takes(state.board, pseudo[0])
    assert engine.describe_move(state, pseudo[0]) == "Ra4>>f4"


def test_the_bridge_admits_rows_and_weights_and_holds_rows_on_a_wrap() -> None:
    assert rust_supports(compile_spec(_army(False))) is None
    assert rust_supports(compile_spec(_army(True))) is None
    # The validator refuses a row on a wrapped board (push.H4); one that reached the compiler
    # some other way is held too, until Q1074b7 reads the ring.
    held = rust_supports(dataclasses.replace(compile_spec(_army(False)), wrap=object()))
    assert held is not None and "push.H4" in held
    assert "Q1074b4" not in held


def test_the_schema_offers_every_chain() -> None:
    push = rule_spec_schema()["properties"]["rules"]["properties"]["push"]
    assert push["properties"]["chain"]["enum"] == ["one", "row", "weight"]


# ------------------------------------------------- strength apart from weight (Q1074b4s)


def _errors_with(spec: dict[str, Any], text: str) -> list[str]:
    return [error for error in validate_spec(spec) if text in error]


def _piece(spec: dict[str, Any], pid: str) -> dict[str, Any]:
    found: dict[str, Any] = next(p for p in spec["pieces"] if p["id"] == pid)
    return found


def _unpushed_rook(**numbers: int) -> dict[str, Any]:
    """`_army` under ``"weight"`` where no list names the rook, so nothing may push it: its
    one number is read only as its strength."""
    spec = _army(True)
    del _piece(spec, "R")["weight"]
    _piece(spec, "R").update(numbers)
    _piece(spec, "R")["pushes"] = ["K", "N", "P"]
    _piece(spec, "N")["pushes"] = ["N", "P"]
    _piece(spec, "P")["pushes"] = ["P"]
    return spec


def test_a_light_strong_pusher_moves_a_row_heavier_than_itself() -> None:
    """Weight 1, strength 5: a row of two knights (2 + 2) moves, three (6) does not; and the
    same rook is light enough for a pawn of strength 3 to push it with a pawn behind."""
    engine = _engine(_strong(rook=(1, 5)))

    def rook_pushes(*behind: str) -> list[int]:
        board = {"a1": ("K", 0), "a4": ("R", 0), "g7": ("K", 1)}
        board |= {square: ("N", 1) for square in behind}
        state = engine.make_state(board, side_to_move=0)
        return _pushes_to(engine, state, "a4", "c4")

    assert len(rook_pushes("c4", "d4")) == 1
    assert rook_pushes("c4", "d4", "e4") == []
    blocked = engine.make_state({
        "a1": ("K", 0), "c2": ("P", 0), "c3": ("R", 1), "c4": ("P", 1), "c5": ("N", 1),
        "g7": ("K", 1),
    }, side_to_move=0)
    assert _pushes_to(engine, blocked, "c2", "c3") == []  # 1 + 1 + 2 = 4 > 3
    light = engine.make_state({
        "a1": ("K", 0), "c2": ("P", 0), "c3": ("R", 1), "c4": ("P", 1), "g7": ("K", 1),
    }, side_to_move=0)
    assert len(_pushes_to(engine, light, "c2", "c3")) == 1  # 1 + 1 = 2 <= 3


def test_a_heavy_weak_pusher_moves_only_a_light_row() -> None:
    engine = _engine(_strong(rook=(9, 1), pawn=1))
    state = engine.make_state({
        "a1": ("K", 0), "a4": ("R", 0), "c4": ("P", 1), "a6": ("R", 0), "c6": ("N", 1),
        "g7": ("K", 1),
    }, side_to_move=0)
    assert len(_pushes_to(engine, state, "a4", "c4")) == 1
    assert _pushes_to(engine, state, "a6", "c6") == []


def test_a_stated_strength_compiles_beside_the_weights() -> None:
    compiled = compile_spec(_strong())
    assert compiled.push is not None
    assert compiled.push.weight == (2, 1, 2, 1)
    assert compiled.push.strength == (0, 6, 2, 3)  # the king never pushes
    weighed = compile_spec(_army(True)).push
    assert weighed is not None and weighed.strength == (0, 3, 2, 1)
    row = compile_spec(_army(False)).push
    assert row is not None and row.weight == () and row.strength == ()


def test_a_strength_equal_to_the_weight_is_its_off_value() -> None:
    """Every spec written before the key hashes as it did: a strength written at the weight
    is dropped by the normal form, and one that differs is another game."""
    plain = _army(True)
    same = _army(True)
    _piece(same, "R")["strength"] = 3
    assert validate_spec(same) == []
    assert spec_hash(same) == spec_hash(plain)
    assert "strength" not in _piece(normal_form(same), "R")
    assert spec_hash(_strong()) != spec_hash(plain)
    assert _piece(normal_form(_strong()), "R")["strength"] == 6


def test_a_pusher_nothing_may_push_needs_only_a_strength() -> None:
    """Its weight would be read only as its strength, so the two spellings are one game."""
    strong = _unpushed_rook(strength=4)
    weighed = _unpushed_rook(weight=4)
    assert validate_spec(strong) == [] and validate_spec(weighed) == []
    assert spec_hash(strong) == spec_hash(weighed)
    assert compile_spec(strong).push == compile_spec(weighed).push
    assert validate_spec(_unpushed_rook(weight=4, strength=4)) == []
    assert _errors_with(
        _unpushed_rook(weight=2, strength=4), "piece 'R' states a weight and a different strength"
    )
    assert _errors_with(_unpushed_rook(), "piece 'R' pushes under chain 'weight' and states neither")


def test_a_pusher_that_may_be_pushed_needs_a_weight() -> None:
    spec = _army(True)
    del _piece(spec, "R")["weight"]
    _piece(spec, "R")["strength"] = 4
    assert _errors_with(spec, "piece 'R' may be pushed under chain 'weight' and states no weight")


@pytest.mark.parametrize("chain", ["one", "row"])
def test_a_strength_is_refused_outside_chain_weight(chain: str) -> None:
    spec = _army(False)
    spec["rules"]["push"] = {"chain": chain}
    _piece(spec, "R")["strength"] = 4
    assert _errors_with(spec, "piece 'R' states a strength and rules.push.chain is")
    del _piece(spec, "R")["strength"]
    assert validate_spec(spec) == []


def test_a_strength_nothing_reads_is_refused() -> None:
    king = _army(True)
    _piece(king, "K")["strength"] = 4
    assert _errors_with(king, "piece 'K' states a strength and none of its moves pushes")
    still = _army(True)
    for piece in still["pieces"]:
        piece.pop("weight")
        for move in piece["moves"]:
            move.pop("push", None)
    still["rules"].pop("push")
    _piece(still, "R")["strength"] = 4
    assert _errors_with(still, "piece 'R' states a strength and no move in the game pushes")


@pytest.mark.parametrize("strength", [0, 100, -1, 1.5, True, "2"])
def test_a_strength_is_one_to_ninety_nine(strength: Any) -> None:
    spec = _strong()
    _piece(spec, "R")["strength"] = strength
    assert _errors_with(spec, "strength must be a whole number from 1 to 99")


def test_the_schema_and_the_rules_offer_the_strength() -> None:
    piece = rule_spec_schema()["properties"]["pieces"]["items"]["properties"]["strength"]
    assert (piece["minimum"], piece["maximum"]) == (1, 99)
    text = describe_spec(_strong())
    assert "its strength is at least the line's weights added together" in text
    assert "Strengths, of the pieces that push: R 6, N 2, P 3." in text
