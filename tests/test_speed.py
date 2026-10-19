"""Schema 28 (Q943a): speed in rounds, `docs/plan-speed.md`.

§5's Q943a acceptance, as tests: the key and every refusal of §4.1 in words; the compiled
table and its inert default; each case of §2.5 (A to H) on its position as written there,
and with the speeds turned round and equal; §2.6's endings in the middle of a round; §2.8's
void bar; every round held to the declared oracle (`tests/simultaneous_oracle.py`) and the
key to `_compute_hash`; the joint perft of a speed spec frozen after the oracle agreed; and
a round game without speed byte for byte what it was.

White is player 0 and moves up the board, as in the plan's diagrams.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest
from simultaneous_oracle import resolve_round

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.engine import _square_name
from chess_variant_sandbox.engine.moves import MOVE_DROP, MOVE_ROTATE, decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.spec.schema import _schema_twenty_eight_uses, validate_spec

HERE = Path(__file__).parent

(
    EVENT_COLLISION, EVENT_VOID, EVENT_INITIATIVE, EVENT_BARRED, EVENT_SWAP,
    EVENT_ORDER, EVENT_LOST, EVENT_UNPLAYED,
) = range(8)


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))


GARDNER = load("_simultaneous_gardner")

# Two pieces the plan's cases need beside Gardner's six (xiangqi's cannon and horse), a
# second kind of pawn for a capture-only swap between two speeds, and a pawn with a facing.
CANNON = {
    "id": "C", "name": "cannon", "value": 4,
    "moves": [
        {"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "move"},
        {"kind": "hop", "vectors": [[1, 0]], "symmetry": "rotate4", "mode": "capture"},
    ],
}
HORSE = {
    "id": "H", "name": "horse", "value": 3,
    "moves": [{"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8", "mode": "both"}],
}
PAWN_TWO = {
    "id": "Z", "name": "second pawn", "value": 1,
    "moves": [
        {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
        {"kind": "leap", "vectors": [[1, 1]], "symmetry": "lateral", "mode": "capture"},
    ],
}
FACED = {
    "id": "F", "name": "faced pawn", "value": 1, "facing": {"axes": 4},
    "moves": [{"kind": "leap", "vectors": [[0, 1]], "mode": "both"}],
}


def with_speeds(
    base: dict[str, Any], speeds: dict[str, int] | None, *extra: dict[str, Any], **rules: Any
) -> dict[str, Any]:
    """``base`` at schema 28, with ``extra`` pieces declared, ``rules`` set and these
    speeds (``None``: no speed at all, the round as it was)."""
    spec = copy.deepcopy(base)
    spec["schema_version"] = 28
    spec["pieces"].extend(copy.deepcopy(piece) for piece in extra)
    spec["rules"].update(copy.deepcopy(rules))
    for piece in spec["pieces"]:
        if speeds and piece["id"] in speeds:
            piece["speed"] = speeds[piece["id"]]
    return spec


def gardner(speeds: dict[str, int] | None, *extra: dict[str, Any], **rules: Any) -> dict[str, Any]:
    """Gardner in rounds with these speeds; each extra piece starts in place of a pawn, so
    both players hold it and its speed can order a round."""
    spec = with_speeds(GARDNER, speeds, *extra, **rules)
    pawns = next(p for p in spec["setup"]["placements"] if p["piece"] == "P")
    for piece in extra:
        spec["setup"]["placements"].append({"piece": piece["id"], "squares": [pawns["squares"].pop()]})
    return spec


def place(engine: Engine, pieces: dict[str, tuple[str, int]], **changes: Any) -> GameState:
    state = engine.make_state(pieces).with_(**changes)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def move(
    engine: Engine, state: GameState, player: int, origin: str | None, target: str,
    *, rotate: bool = False, promotion: str | None = None,
) -> int:
    """``player``'s legal move from ``origin`` (``None`` for a drop) to ``target``."""
    to = engine._square_index(target)
    for m in engine.legal_moves_for(state, player):
        o, t, kind, promo = decode_move(m)
        if t != to or (kind == MOVE_ROTATE) != rotate:
            continue
        if origin is None:
            if kind == MOVE_DROP:
                return m
            continue
        if kind == MOVE_DROP or o != engine._square_index(origin):
            continue
        if promotion is not None and engine.spec.piece_ids[promo] != promotion:
            continue
        return m
    raise AssertionError(f"player {player} has no move {origin}->{target}")


def play(
    spec: dict[str, Any], pieces: dict[str, tuple[str, int]],
    first: tuple[Any, ...], second: tuple[Any, ...], **changes: Any,
) -> tuple[Engine, GameState, list[tuple[int, int, int]]]:
    """One round from a hand-built position, through the core, held to the oracle and the
    slow key. ``first`` and ``second`` are each ``(origin, target)`` plus `move`'s options."""
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    state = place(engine, pieces, **changes)

    def word(player: int, said: tuple[Any, ...]) -> int:
        options = said[2] if len(said) > 2 else {}
        return move(engine, state, player, said[0], said[1], **options)

    a, b = word(0, first), word(1, second)
    after, _, events = engine.apply_round_with_phase(state, a, b)
    assert after == resolve_round(engine, state, a, b)
    assert after.key == engine._compute_hash(after)
    return engine, after, [tuple(event) for event in events]


def board(engine: Engine, state: GameState) -> dict[str, tuple[str, int]]:
    """The position as square names to ``(piece id, player)``."""
    spec = engine.spec
    return {
        _square_name(square, spec.width): (spec.piece_ids[spec.type_of(value)], spec.owner_of(value))
        for square, value in enumerate(state.board[: spec.n_squares]) if value
    }


def sq(engine: Engine, name: str) -> int:
    return engine._square_index(name)


def kinds(events: list[tuple[int, int, int]]) -> list[int]:
    return [event[0] for event in events]


def same_board(a: tuple[Engine, GameState, Any], b: tuple[Engine, GameState, Any]) -> bool:
    """Two rounds of two specs (with and without speeds) left the same position, the key
    aside: a spec's Zobrist draw is its own."""
    fields = ("board", "hands", "unmoved", "halfmove", "ply", "void_run", "bars", "rest")
    return all(getattr(a[1], f) == getattr(b[1], f) for f in fields)


KINGS = {"e1": ("K", 0), "e5": ("K", 1)}


# ------------------------------------------------------------------- the key


def test_the_collector_names_speed_whatever_its_value() -> None:
    assert _schema_twenty_eight_uses(gardner({"R": 1})) == {"pieces.speed"}
    assert _schema_twenty_eight_uses(gardner({"R": 0})) == {"pieces.speed"}
    assert _schema_twenty_eight_uses(gardner(None)) == set()


def test_speed_at_schema_twenty_seven_is_refused() -> None:
    spec = gardner({"R": 1})
    spec["schema_version"] = 27
    assert any("pieces.speed" in e and "schema 28" in e for e in validate_spec(spec))


def test_speed_without_rounds_is_refused() -> None:
    spec = gardner({"R": 1})
    del spec["rules"]["simultaneous"]
    assert any("this game has none" in e for e in validate_spec(spec))


@pytest.mark.parametrize(
    ("value", "words"),
    [
        (0, "speed 0 is the default"),
        (-1, "at least 1"),
        (1.5, "at least 1"),
        (True, "at least 1"),
        ("fast", "at least 1"),
    ],
)
def test_a_speed_that_is_not_a_rank_of_at_least_one_is_refused(value: Any, words: str) -> None:
    spec = gardner(None)
    spec["pieces"][1]["speed"] = value
    assert any(words in e for e in validate_spec(spec))


def test_speeds_must_be_ranks_with_none_skipped() -> None:
    errors = validate_spec(gardner({"N": 2}))
    assert any("speeds are ranks" in e and "2 as 1" in e for e in errors)
    # Every type at one speed skips rank 0: one game, one spelling.
    every = gardner({pid: 1 for pid in "PNBRQK"})
    assert any("speeds are ranks" in e for e in validate_spec(every))
    assert validate_spec(gardner({"N": 2, "B": 1})) == []


def test_speed_on_a_portal_is_refused() -> None:
    spec = with_speeds(load("_simultaneous_portals"), {"N": 1})
    portal = next(p for p in spec["pieces"] if p["id"] == "O")
    portal["speed"] = 1
    assert any("on a portal" in e for e in validate_spec(spec))


def test_speed_on_a_wild_piece_is_refused() -> None:
    spec = with_speeds(load("_simultaneous_wild"), {"K": 1})
    wild = spec["rules"]["npcs"][0]["piece"]
    next(p for p in spec["pieces"] if p["id"] == wild)["speed"] = 1
    assert any("on a wild piece" in e for e in validate_spec(spec))


def test_speeds_that_can_never_order_a_round_are_refused() -> None:
    """The only faster type is one neither player can ever hold."""
    spec = with_speeds(GARDNER, {"Z": 1}, PAWN_TWO)
    assert any("never orders a round" in e for e in validate_spec(spec))


def test_beats_on_a_type_with_no_equal_speed_opponent_is_refused() -> None:
    """Two armies strictly ordered (the watering hole's kinds, given speeds): a `beats` is
    read only at a collision, which needs two equal speeds."""
    spec = strictly_ordered()
    spec["pieces"][0]["beats"] = []
    assert any("one speed collide" in e for e in validate_spec(spec))


def strictly_ordered(**simultaneous: Any) -> dict[str, Any]:
    """Two armies of different kinds, every opposing pair of a different speed."""
    steps = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]
    return {
        "schema_version": 28, "name": "ordered", "board": {"width": 4, "height": 4},
        "pieces": [
            {"id": "R", "moves": steps, "royal": True, "speed": 1},
            {"id": "S", "moves": steps, "speed": 3},
            {"id": "r", "moves": steps, "royal": True},
            {"id": "s", "moves": steps, "speed": 2},
        ],
        "setup": {"mirror": False, "placements": [
            {"piece": "R", "player": 0, "squares": ["a1"]},
            {"piece": "S", "player": 0, "squares": ["b1"]},
            {"piece": "r", "player": 1, "squares": ["d4"]},
            {"piece": "s", "player": 1, "squares": ["c4"]},
        ]},
        "rules": {"win": "royal_capture", "simultaneous": dict(simultaneous)},
    }


def test_a_strictly_ordered_game_is_valid() -> None:
    assert validate_spec(strictly_ordered()) == []


@pytest.mark.parametrize("tie", ["initiative", "impossible"])
def test_a_tie_rule_where_no_equal_speeds_meet_is_refused(tie: str) -> None:
    errors = validate_spec(strictly_ordered(tie=tie))
    assert any(f"tie {tie!r} can never be read" in e for e in errors)


def test_the_impossible_tie_is_proved_over_equal_speeds_only() -> None:
    """Gardner's mirrored army ties everywhere; with the rook faster than the king, a pair
    of rook and king still ties at equal speed (each rook meets the other's rook), so the
    proof still fails, but it names an equal-speed pair."""
    errors = validate_spec(gardner({"R": 1}, simultaneous={"tie": "impossible"}))
    pair = next(e for e in errors if "can collide and tie" in e)
    speeds = {"R": 1}
    first, second = pair.split("but a ")[1].split(" of the first player and a ")
    second = second.split(" of the second")[0]
    assert speeds.get(first, 0) == speeds.get(second, 0)


def test_a_swap_rule_where_no_equal_speed_pair_can_swap_is_refused() -> None:
    errors = validate_spec(strictly_ordered(swap="collision"))
    assert any("can never fire" in e and "has its speed" in e for e in errors)


def test_the_speed_table_is_empty_without_speeds_and_per_type_with_them() -> None:
    plain = compile_spec(gardner(None))
    assert plain.simultaneous is not None and plain.simultaneous.speed == ()
    assert "sim_speed" not in canonical_tables(plain)
    fast = compile_spec(gardner({"N": 2, "B": 1}))
    assert fast.simultaneous is not None
    assert dict(zip(fast.piece_ids, fast.simultaneous.speed, strict=False)) == {
        "P": 0, "N": 2, "B": 1, "R": 0, "Q": 0, "K": 0,
    }
    assert canonical_tables(fast)["sim_speed"] == list(fast.simultaneous.speed)


# --------------------------------------------------------------- §2.5's cases


def test_case_a_the_faster_move_takes_the_slower_piece_where_it_stands() -> None:
    pieces = {"a1": ("R", 0), "e1": ("K", 0), "a5": ("K", 1)}
    moves = (("a1", "a5"), ("a5", "b4"))
    engine, after, events = play(gardner({"R": 1}), pieces, *moves)
    assert after.board[sq(engine, "a5")] and engine.spec.owner_of(after.board[sq(engine, "a5")]) == 0
    assert after.board[sq(engine, "b4")] == 0
    assert events == [(EVENT_ORDER, sq(engine, "a5"), 0), (EVENT_UNPLAYED, sq(engine, "a5"), 1)]
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0
    # Turned round: the king leaves first, and the rook's slide lands where it was.
    engine, turned, events = play(gardner({"K": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "b4"), 1)]
    assert engine.spec.type_of(turned.board[sq(engine, "b4")]) == engine.spec.piece_ids.index("K")
    assert engine.spec.type_of(turned.board[sq(engine, "a5")]) == engine.spec.piece_ids.index("R")
    assert engine.terminal(turned) is None
    # Equal: the chase of today, the same board without an order.
    equal = play(gardner({"R": 1, "K": 1}), pieces, *moves)
    assert equal[2] == [] and same_board(equal, (engine, turned, None))
    assert same_board(equal, play(gardner(None), pieces, *moves))


def test_case_b_no_swap_survives_unequal_speeds() -> None:
    pieces = {"b5": ("Q", 0), "e1": ("K", 0), "b4": ("K", 1)}
    moves = (("b5", "b4"), ("b4", "b5"))
    engine, after, events = play(gardner({"Q": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "b4"), 0), (EVENT_UNPLAYED, sq(engine, "b4"), 1)]
    assert engine.spec.owner_of(after.board[sq(engine, "b4")]) == 0
    assert after.board[sq(engine, "b5")] == 0
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0
    # The king faster: it takes the queen, and the queen's word is lost with it.
    engine, turned, events = play(gardner({"K": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "b5"), 1), (EVENT_LOST, sq(engine, "b5"), 0)]
    assert engine.spec.owner_of(turned.board[sq(engine, "b5")]) == 1
    assert turned.board[sq(engine, "b4")] == 0
    assert engine.terminal(turned) is None
    # Equal: the swap of today, both arrive.
    equal = play(gardner({"Q": 1, "K": 1}), pieces, *moves)
    assert equal[2] == []
    assert equal[0].spec.owner_of(equal[1].board[sq(engine, "b4")]) == 0
    assert equal[0].spec.owner_of(equal[1].board[sq(engine, "b5")]) == 1
    assert same_board(equal, play(gardner(None), pieces, *moves))


C_PIECES = {"b1": ("N", 0), "a5": ("Q", 1), "c4": ("P", 1), **KINGS}


@pytest.mark.parametrize("slower", [("a5", "c3"), ("c4", "c3")], ids=["queen", "pawn"])
def test_case_c_the_two_moves_end_on_one_square(slower: tuple[str, str]) -> None:
    engine, after, events = play(gardner({"N": 1}), C_PIECES, ("b1", "c3"), slower)
    c3 = after.board[sq(engine, "c3")]
    if slower[0] == "a5":
        # The knight arrives first and is then a standing target the queen may take.
        assert events == [(EVENT_ORDER, sq(engine, "c3"), 0)]
        assert engine.spec.owner_of(c3) == 1 and engine.spec.piece_ids[engine.spec.type_of(c3)] == "Q"
    else:
        # A push may not land on an occupied square: void, and the knight stands.
        assert events == [(EVENT_ORDER, sq(engine, "c3"), 0), (EVENT_VOID, sq(engine, "c4"), 1)]
        assert engine.spec.owner_of(c3) == 0
        assert after.board[sq(engine, "c4")] != 0
        assert after.void_run == (-1, -1)
    # Turned round: the black piece lands first and the knight takes it there.
    engine, turned, events = play(gardner({"Q": 1, "P": 1}), C_PIECES, ("b1", "c3"), slower)
    assert events == [(EVENT_ORDER, sq(engine, "c3"), 1)]
    c3 = turned.board[sq(engine, "c3")]
    assert engine.spec.owner_of(c3) == 0 and engine.spec.piece_ids[engine.spec.type_of(c3)] == "N"
    # Equal: a collision, and under the default tie both go.
    equal = play(gardner({"N": 1, "Q": 1, "P": 1}), C_PIECES, ("b1", "c3"), slower)
    assert kinds(equal[2]) == [EVENT_COLLISION]
    assert equal[1].board[sq(engine, "c3")] == 0
    assert same_board(equal, play(gardner(None), C_PIECES, ("b1", "c3"), slower))


def test_case_d_the_slower_sliders_path_is_blocked_by_the_faster_arrival() -> None:
    pieces = {"b1": ("N", 0), "a3": ("R", 1), **KINGS}
    moves = (("b1", "c3"), ("a3", "e3"))
    engine, after, events = play(gardner({"N": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "c3"), 0), (EVENT_VOID, sq(engine, "a3"), 1)]
    assert after.board[sq(engine, "a3")] and after.board[sq(engine, "e3")] == 0
    engine, turned, events = play(gardner({"R": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "e3"), 1)]
    assert turned.board[sq(engine, "e3")] and turned.board[sq(engine, "c3")]
    equal = play(gardner({"N": 1, "R": 1}), pieces, *moves)
    assert equal[2] == [] and same_board(equal, (engine, turned, None))


def test_case_e_a_cannons_screen_that_left_voids_its_hop() -> None:
    pieces = {"a1": ("C", 1), "a3": ("N", 0), "a5": ("R", 0), "e1": ("K", 0), "e4": ("K", 1)}
    moves = (("a3", "b5"), ("a1", "a5"))
    engine, after, events = play(gardner({"N": 1}, CANNON), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "b5"), 0), (EVENT_VOID, sq(engine, "a1"), 1)]
    assert after.board[sq(engine, "a5")] and engine.spec.owner_of(after.board[sq(engine, "a5")]) == 0
    # The cannon faster: it hops over the knight still standing and takes the rook.
    engine, turned, events = play(gardner({"C": 1}, CANNON), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "a5"), 1)]
    assert engine.spec.owner_of(turned.board[sq(engine, "a5")]) == 1
    assert turned.board[sq(engine, "b5")]
    # Equal: today's round, made on the board both saw.
    equal = play(gardner({"N": 1, "C": 1}, CANNON), pieces, *moves)
    assert equal[2] == [] and same_board(equal, (engine, turned, None))


def test_case_e_a_capture_only_hop_whose_target_left_is_void() -> None:
    pieces = {"a1": ("C", 1), "a3": ("N", 0), "a5": ("R", 0), "e1": ("K", 0), "e4": ("K", 1)}
    moves = (("a5", "b5"), ("a1", "a5"))
    engine, after, events = play(gardner({"R": 1}, CANNON), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "b5"), 0), (EVENT_VOID, sq(engine, "a1"), 1)]
    engine, turned, events = play(gardner({"C": 1}, CANNON), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "a5"), 1), (EVENT_LOST, sq(engine, "a5"), 0)]
    equal = play(gardner({"R": 1, "C": 1}, CANNON), pieces, *moves)
    assert kinds(equal[2]) == [EVENT_VOID] and same_board(equal, (engine, after, None))


def test_case_e_a_quiet_slide_a_faster_arrival_blocks_is_void() -> None:
    pieces = {"a1": ("C", 1), "b1": ("N", 0), "e1": ("K", 0), "e4": ("K", 1)}
    moves = (("b1", "a3"), ("a1", "a5"))
    engine, after, events = play(gardner({"N": 1}, CANNON), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "a3"), 0), (EVENT_VOID, sq(engine, "a1"), 1)]
    engine, turned, events = play(gardner({"C": 1}, CANNON), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "a5"), 1)]
    assert turned.board[sq(engine, "a5")] and turned.board[sq(engine, "a3")]
    equal = play(gardner({"N": 1, "C": 1}, CANNON), pieces, *moves)
    assert equal[2] == [] and same_board(equal, (engine, turned, None))


def test_case_e_a_lame_leap_whose_leg_a_faster_piece_lands_on_is_void() -> None:
    pieces = {"b5": ("H", 1), "a2": ("N", 0), "e1": ("K", 0), "e4": ("K", 1)}
    moves = (("a2", "b4"), ("b5", "c3"))
    engine, after, events = play(gardner({"N": 1}, HORSE), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "b4"), 0), (EVENT_VOID, sq(engine, "b5"), 1)]
    engine, turned, events = play(gardner({"H": 1}, HORSE), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "c3"), 1)]
    equal = play(gardner({"N": 1, "H": 1}, HORSE), pieces, *moves)
    assert equal[2] == [] and same_board(equal, (engine, turned, None))


@pytest.mark.parametrize("mode", ["both", "capture"])
def test_case_e_a_capture_whose_target_left_lands_or_is_void_by_its_mode(mode: str) -> None:
    """A rook (mode both) lands on the square a faster victim left; a pawn's diagonal
    (capture only) is void, as `docs/reference/rounds.md` §2.4's chase."""
    if mode == "both":
        pieces = {"a1": ("R", 0), "a5": ("N", 1), **KINGS}
        moves = (("a1", "a5"), ("a5", "b3"))
    else:
        pieces = {"c3": ("P", 0), "d4": ("N", 1), **KINGS}
        moves = (("c3", "d4"), ("d4", "b3"))
    engine, after, events = play(gardner({"N": 1}), pieces, *moves)
    origin = moves[0][0]
    if mode == "both":
        assert events == [(EVENT_ORDER, sq(engine, "b3"), 1)]
        assert after.board[sq(engine, "a5")] and after.board[sq(engine, origin)] == 0
    else:
        assert events == [(EVENT_ORDER, sq(engine, "b3"), 1), (EVENT_VOID, sq(engine, origin), 0)]
        assert after.board[sq(engine, origin)]
    equal = play(gardner({"N": 1, "R": 1, "P": 1}), pieces, *moves)
    assert same_board(equal, (engine, after, None))


def test_case_f_a_slower_piece_taken_by_the_faster_move_is_lost() -> None:
    pieces = {"a1": ("R", 0), "a5": ("N", 1), **KINGS}
    engine, after, events = play(gardner({"R": 1}), pieces, ("a1", "a5"), ("a5", "b3"))
    assert events == [(EVENT_ORDER, sq(engine, "a5"), 0), (EVENT_LOST, sq(engine, "a5"), 1)]
    assert after.board[sq(engine, "b3")] == 0


def test_case_f_a_slower_piece_the_faster_blast_removes_is_lost() -> None:
    atomic = load("_simultaneous_atomic")
    pieces = {"b1": ("N", 0), "c3": ("B", 1), "d4": ("R", 1), "e1": ("K", 0), "a5": ("K", 1)}
    moves = (("b1", "c3"), ("d4", "d5"))
    engine, after, events = play(with_speeds(atomic, {"N": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "c3"), 0), (EVENT_LOST, sq(engine, "d4"), 1)]
    assert after.board[sq(engine, "d4")] == after.board[sq(engine, "d5")] == 0
    # Turned round: the rook leaves the blast first.
    engine, turned, events = play(with_speeds(atomic, {"R": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "d5"), 1)]
    assert turned.board[sq(engine, "d5")]
    equal = play(with_speeds(atomic, {"N": 1, "R": 1}), pieces, *moves)
    assert equal[2] == [] and same_board(equal, (engine, turned, None))


def test_case_f_a_slower_piece_taken_and_reborn_is_lost_not_re_read() -> None:
    circe = load("_simultaneous_circe")
    pieces = {"c1": ("Q", 0), "c3": ("R", 1), "e1": ("K", 0), "e4": ("K", 1)}
    engine, after, events = play(with_speeds(circe, {"Q": 1}), pieces, ("c1", "c3"), ("c3", "c5"))
    assert events == [(EVENT_ORDER, sq(engine, "c3"), 0), (EVENT_LOST, sq(engine, "c3"), 1)]
    assert after.board[sq(engine, "c5")] == 0
    reborn = [s for s, (pid, owner) in board(engine, after).items() if (pid, owner) == ("R", 1)]
    assert reborn, "the rook is reborn at home"


CONTACT_STAY = {"position": "stay", "links": [{"mover": "P", "onto": "Q", "onto_to": "R"}], "immune": ["K"]}


def test_case_g_a_faster_contact_that_stays_fires_before_the_slower_piece_leaves() -> None:
    pieces = {"c3": ("P", 0), "d4": ("Q", 1), **KINGS}
    moves = (("c3", "d4"), ("d4", "a4"))
    engine, after, events = play(gardner({"P": 1}, contacts=CONTACT_STAY), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "d4"), 0), (EVENT_LOST, sq(engine, "d4"), 1)]
    assert board(engine, after)["d4"] == ("R", 1)
    # Turned round: the queen leaves first, and the pawn's capture-only word is void.
    engine, turned, events = play(gardner({"Q": 1}, contacts=CONTACT_STAY), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "a4"), 1), (EVENT_VOID, sq(engine, "c3"), 0)]
    equal = play(gardner({"P": 1, "Q": 1}, contacts=CONTACT_STAY), pieces, *moves)
    assert kinds(equal[2]) == [EVENT_VOID] and same_board(equal, (engine, turned, None))


def test_case_g_a_faster_capture_of_a_piece_whose_slower_move_was_a_turn() -> None:
    pieces = {"a1": ("R", 0), "a5": ("F", 1), **KINGS}
    rotate = {"rotate": True}
    turn = ("a5", "a5", rotate)
    engine, after, events = play(gardner({"R": 1}, FACED), pieces, ("a1", "a5"), turn)
    assert events == [(EVENT_ORDER, sq(engine, "a5"), 0), (EVENT_LOST, sq(engine, "a5"), 1)]
    # Turned round: the piece turns first and is then taken all the same.
    engine, turned, events = play(gardner({"F": 1}, FACED), pieces, ("a1", "a5"), turn)
    assert events == [(EVENT_ORDER, sq(engine, "a5"), 1)]
    assert engine.spec.owner_of(turned.board[sq(engine, "a5")]) == 0
    equal = play(gardner({"R": 1, "F": 1}, FACED), pieces, ("a1", "a5"), turn)
    assert equal[2] == [] and same_board(equal, (engine, turned, None))


def test_case_h_a_slower_merge_onto_a_piece_the_faster_move_took_is_re_read() -> None:
    merge = load("_simultaneous_merge")
    pieces = {"c1": ("N", 0), "d3": ("P", 1), "e4": ("P", 1), "e1": ("K", 0), "a5": ("K", 1)}
    moves = (("c1", "d3"), ("e4", "d3"))
    engine, after, events = play(with_speeds(merge, {"N": 1}), pieces, *moves)
    # The pawn's diagonal now meets the knight on d3: a capture, as a pawn (no merge).
    assert events == [(EVENT_ORDER, sq(engine, "d3"), 0)]
    assert board(engine, after)["d3"] == ("P", 1)


def test_case_h_a_slower_promoting_push_onto_a_held_square_is_void() -> None:
    pieces = {"d4": ("P", 0), "c3": ("N", 1), "e1": ("K", 0), "a5": ("K", 1)}
    moves = (("d4", "d5", {"promotion": "Q"}), ("c3", "d5"))
    engine, after, events = play(gardner({"N": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "d5"), 1), (EVENT_VOID, sq(engine, "d4"), 0)]
    assert engine.spec.piece_ids[engine.spec.type_of(after.board[sq(engine, "d4")])] == "P"
    # Turned round: the pawn promotes first, and the knight takes the queen it became.
    engine, turned, events = play(gardner({"P": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "d5"), 0)]
    assert engine.spec.owner_of(turned.board[sq(engine, "d5")]) == 1
    assert turned.halfmove == 0
    equal = play(gardner({"P": 1, "N": 1}), pieces, *moves)
    assert kinds(equal[2]) == [EVENT_COLLISION]
    assert same_board(equal, play(gardner(None), pieces, *moves))


def test_case_h_a_slower_capture_reads_the_type_the_faster_move_promoted_to() -> None:
    pieces = {"d4": ("P", 0), "a5": ("R", 1), "e1": ("K", 0), "a1": ("K", 1)}
    moves = (("d4", "d5", {"promotion": "Q"}), ("a5", "d5"))
    engine, after, events = play(gardner({"P": 1}), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "d5"), 0)]
    d5 = after.board[sq(engine, "d5")]
    assert engine.spec.owner_of(d5) == 1 and engine.spec.piece_ids[engine.spec.type_of(d5)] == "R"
    # A rook that may not take a queen: the word is not offered, void.
    spec = gardner({"P": 1})
    next(p for p in spec["pieces"] if p["id"] == "R")["captures"] = ["P", "N", "B", "R", "K"]
    engine, after, events = play(spec, pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "d5"), 0), (EVENT_VOID, sq(engine, "a5"), 1)]


def test_case_h_a_slower_drop_onto_a_held_square_is_void_and_stays_in_hand() -> None:
    shogi = load("_simultaneous_shogi")
    pieces = {"a1": ("L", 0), "b2": ("G", 0), "c4": ("L", 1)}
    engine = Engine(with_speeds(shogi, {"G": 1}))
    slot = engine.spec.drops.hand_slot(engine.spec.piece_ids.index("C"))
    hand = [0] * len(engine.initial_state().hands[1])
    hand[slot] = 1
    hands = (tuple(0 for _ in hand), tuple(hand))
    moves = (("b2", "b3"), (None, "b3"))
    engine, after, events = play(with_speeds(shogi, {"G": 1}), pieces, *moves, hands=hands)
    assert events == [(EVENT_ORDER, sq(engine, "b3"), 0), (EVENT_VOID, sq(engine, "b3"), 1)]
    assert after.hands[1][slot] == 1
    # Turned round: the chick is dropped first, and the giraffe takes it into White's hand.
    engine, turned, events = play(with_speeds(shogi, {"C": 1}), pieces, *moves, hands=hands)
    assert events == [(EVENT_ORDER, sq(engine, "b3"), 1)]
    assert turned.hands[1][slot] == 0 and turned.hands[0][slot] == 1


# ----------------------------------------------------------- §2.6, the endings


def test_a_slower_capture_of_the_faster_players_royal_never_happens() -> None:
    pieces = {"a1": ("R", 0), "e1": ("K", 0), "a5": ("K", 1), "e5": ("Q", 1)}
    moves = (("a1", "a5"), ("e5", "e1"))
    engine, after, events = play(gardner({"R": 1}), pieces, *moves)
    assert kinds(events) == [EVENT_ORDER, EVENT_UNPLAYED]
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0
    assert after.board[sq(engine, "e1")] and after.board[sq(engine, "e5")]
    # At equal speeds both royals go in one round: a draw.
    engine, equal, _ = play(gardner({"R": 1, "Q": 1}), pieces, *moves)
    outcome = engine.terminal(equal)
    assert outcome is not None and outcome.winner is None


def test_a_faster_blast_that_takes_both_royals_is_a_draw() -> None:
    atomic = load("_simultaneous_atomic")
    pieces = {"b1": ("N", 0), "c3": ("B", 1), "b2": ("K", 0), "d4": ("K", 1), "e4": ("P", 1)}
    engine, after, events = play(with_speeds(atomic, {"N": 1}), pieces, ("b1", "c3"), ("e4", "e3"))
    assert kinds(events) == [EVENT_ORDER, EVENT_UNPLAYED]
    assert after.board[sq(engine, "e4")], "the slower pawn was never played"
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner is None


RACE = {"squares": ["a5", "b5", "c5", "d5", "e5"], "mirror": True, "result": "win"}


@pytest.mark.parametrize(
    ("speeds", "winner"),
    [({"N": 1}, 0), ({"B": 1}, 1), ({"N": 1, "B": 1}, None)],
    ids=["knight faster", "bishop faster", "equal"],
)
def test_a_race_to_the_goals_is_decided_by_speed(speeds: dict[str, int], winner: int | None) -> None:
    pieces = {"d3": ("N", 0), "b2": ("B", 1), "e2": ("K", 0), "a4": ("K", 1)}
    engine, after, events = play(gardner(speeds, reach=RACE), pieces, ("d3", "c5"), ("b2", "a1"))
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == winner and outcome.reason == "reached"
    if winner is not None:
        assert kinds(events) == [EVENT_ORDER, EVENT_UNPLAYED]


def test_reach_safe_is_read_on_the_board_the_faster_move_left() -> None:
    safe = {"piece": "K", "squares": ["a5", "b5", "c5", "d5", "e5"], "mirror": True,
            "result": "win", "safe": True}
    pieces = {"d4": ("K", 0), "a4": ("R", 1), "c2": ("K", 1)}
    moves = (("d4", "d5"), ("a4", "a5"))
    engine, after, events = play(gardner({"K": 1}, reach=safe), pieces, *moves)
    assert kinds(events) == [EVENT_ORDER, EVENT_UNPLAYED]
    outcome = engine.terminal(after)
    assert outcome is not None and outcome.winner == 0
    # At equal speeds the rook arrives too and the king stands attacked: no win.
    engine, equal, _ = play(gardner({"K": 1, "R": 1}, reach=safe), pieces, *moves)
    assert engine.terminal(equal) is None


# ------------------------------------------------------------- §2.8, the bar


def test_no_void_round_between_unequal_speeds() -> None:
    pieces = {"c2": ("P", 0), "d3": ("Z", 1), **KINGS}
    moves = (("c2", "d3"), ("d3", "c2"))
    engine, after, events = play(gardner({"Z": 1}, PAWN_TWO), pieces, *moves)
    assert events == [(EVENT_ORDER, sq(engine, "c2"), 1), (EVENT_LOST, sq(engine, "c2"), 0)]
    assert after.void_run == (-1, -1)
    # Two pawns of one speed: the void round of today.
    equal = gardner({"Z": 1}, PAWN_TWO)
    engine, after, events = play(equal, {"c2": ("P", 0), "d3": ("P", 1), **KINGS}, *moves)
    assert kinds(events) == [EVENT_VOID, EVENT_VOID]
    assert after.void_run == (sq(engine, "c2"), sq(engine, "d3"))


def test_an_ordered_round_resets_a_run_of_one() -> None:
    spec = gardner({"N": 1})
    pieces = {"c2": ("P", 0), "d3": ("P", 1), "b1": ("N", 0), "a5": ("R", 1), **KINGS}
    engine = Engine(spec)
    state = place(engine, pieces)
    first = engine.apply_round(
        state, move(engine, state, 0, "c2", "d3"), move(engine, state, 1, "d3", "c2")
    )
    assert first.void_run == (sq(engine, "c2"), sq(engine, "d3"))
    a, b = move(engine, first, 0, "b1", "a3"), move(engine, first, 1, "a5", "a4")
    after, _, events = engine.apply_round_with_phase(first, a, b)
    assert after == resolve_round(engine, first, a, b)
    assert kinds(events) == [EVENT_ORDER] and after.void_run == (-1, -1)


# ------------------------------------------------------------ perft, frozen


def oracle_perft(engine: Engine, state: GameState, depth: int) -> int:
    first, second = engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1)
    if depth == 1:
        return len(first) * len(second)
    return sum(
        oracle_perft(engine, resolve_round(engine, state, a, b), depth - 1)
        for a in first for b in second
    )


# Frozen after `oracle_perft` agreed (Q943a). Speed changes no legal move at a round's start,
# so depth 1 is Gardner in rounds' 49; depth 2 differs from its 4393
# (`tests/test_schema_twenty_six.py`) because an ordered round leaves another board.
JOINT_PERFT = {"_simultaneous_speed_gardner": (49, 4449)}


@pytest.mark.parametrize("name", sorted(JOINT_PERFT))
def test_the_joint_perft_of_a_speed_game_agrees_with_the_oracle_and_is_frozen(name: str) -> None:
    engine = Engine(load(name))
    state = engine.initial_state()
    counted = (engine.perft(state, 1), engine.perft(state, 2))
    assert counted == (oracle_perft(engine, state, 1), oracle_perft(engine, state, 2))
    assert counted == JOINT_PERFT[name]


# ------------------------------------------------------- inert without speed


def walk_digest(names: list[str], seeds: int = 3, rounds: int = 60) -> str:
    """Random rounds from the start of each spec, every position and event hashed."""
    rows = []
    for name in names:
        engine = Engine(load(name))
        for seed in range(seeds):
            rng = random.Random(seed)
            state = engine.initial_state()
            for _ in range(rounds):
                if engine.terminal(state) is not None:
                    break
                a = rng.choice(engine.legal_moves_for(state, 0))
                b = rng.choice(engine.legal_moves_for(state, 1))
                state, phase, events = engine.apply_round_with_phase(state, a, b)
                rows.append(repr((name, seed, state, phase, events)))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


SPEEDLESS = sorted(
    path.stem for path in HERE.glob("_simultaneous_*.json") if "speed" not in path.stem
)

# Taken with main's package and core (47490df3bfedc24b, before Q943a) and with this one, and
# equal (481 rounds over the twelve speedless round test specs): a round game that declares
# no speed is resolved byte for byte as it was, its events included.
SPEEDLESS_DIGEST = "44198b8c1ac430dcbe89a291045ee12c02f67f0661cf63ad044ccf9eb57cd3fc"


def test_a_round_game_without_speed_plays_byte_for_byte_as_before() -> None:
    assert walk_digest(SPEEDLESS) == SPEEDLESS_DIGEST


# ------------------------------------------------------ Fairy and the twin check


def alternating_twin(spec: dict[str, Any]) -> dict[str, Any]:
    """The same file without `simultaneous`, `beats` and `speed`: an alternating game."""
    twin = copy.deepcopy(spec)
    del twin["rules"]["simultaneous"]
    for piece in twin["pieces"]:
        piece.pop("beats", None)
        piece.pop("speed", None)
    return twin


@pytest.mark.parametrize("name", ["gardner", "shogi", "portals", "wild"])
def test_a_speed_games_moves_are_still_the_alternating_twins(name: str) -> None:
    """Speed changes no legal move at a round's start (§4.6), so the per-side generator of a
    speed game is its twin's, wherever the ordered rounds take the board."""
    spec = load(f"_simultaneous_speed_{name}")
    engine, other = Engine(spec), Engine(alternating_twin(spec))
    rng = random.Random(28)
    state = engine.initial_state()
    for _ in range(30):
        if engine.terminal(state) is not None:
            break
        lists = []
        for player in (0, 1):
            ours = engine.legal_moves_for(state, player)
            alternating = state.with_(
                side_to_move=player, initiative=-1, void_run=(-1, -1), bars=(), rest=(-1, -1)
            )
            if not state.bars:
                assert ours == other.legal_moves(alternating)
            lists.append(ours)
        state = engine.apply_round(state, rng.choice(lists[0]), rng.choice(lists[1]))


def refusals(spec: dict[str, Any]) -> list[str]:
    from chess_variant_sandbox.fairy.variant import Untranslatable, translate

    with pytest.raises(Untranslatable) as refused:
        translate(spec)
    return list(refused.value.reasons)


def test_fairy_stockfish_refuses_speed_by_name() -> None:
    reasons = refusals(load("_simultaneous_speed_gardner"))
    assert any(reason.startswith("pieces[].speed") for reason in reasons)
    assert any(reason.startswith("rules.simultaneous") for reason in reasons)
    assert not any(reason.startswith("pieces[].speed") for reason in refusals(GARDNER))


def test_the_page_names_an_ordered_round_from_the_cores_events() -> None:
    """The round's words read the core's three new events (Q943b draws the two beats)."""
    from chess_variant_sandbox.web.play import round_reveal

    engine = Engine(gardner({"K": 1}))
    state = place(engine, {"b5": ("Q", 0), "e1": ("K", 0), "b4": ("K", 1)})
    moves = (move(engine, state, 0, "b5", "b4"), move(engine, state, 1, "b4", "b5"))
    after, _, events = engine.apply_round_with_phase(state, *moves)
    reveal = round_reveal(engine, state, after, moves, events, ("White", "Black"))
    assert [event["kind"] for event in reveal["events"]] == ["order", "lost"]
    assert reveal["results"] == ["lost", None]
    assert any("was faster: Kb4xb5." in words for words in reveal["words"])
    assert any("took White's queen on b5 before it could move" in words for words in reveal["words"])
    assert not any("swapped squares" in words for words in reveal["words"])
