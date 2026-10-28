"""SR1a: a lame leap's legs, stated (`docs/plan-schema-next.md` §1 rank 1, §2 X1 and X8).

``legs`` lists, per base vector, the squares (steps from the mover) that block the jump,
turned with the vector by the descriptor's symmetry (`legs.stated_legs`). Absent, a
``lame`` is blocked on `compile._lame_path`'s one computed leg, as it always was. On a hex
lattice a ``lame`` must state them, and under ``obstacles.blocks_leaps`` so must a leap
longer than a step. The hand positions below are counted square by square.
"""

from __future__ import annotations

import copy
import itertools
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.bridge import canonical_tables
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.betza import lame_path
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.complexity import expand_symmetry
from chess_variant_sandbox.spec.describe import move_sentence
from chess_variant_sandbox.spec.legs import SYMMETRY_MAPS, stated_legs, turn

ROOT = Path(__file__).resolve().parent.parent

KING = {"id": "K", "name": "king", "value": 0, "royal": True,
        "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}]}
# Janggi's elephant: one step orthogonally, then two diagonally, blocked on both squares.
ELEPHANT = {"kind": "lame", "vectors": [[2, 3]], "symmetry": "dihedral8",
            "legs": [[[0, 1], [1, 2]]]}
HEX_KING = {"id": "K", "name": "king", "value": 0, "royal": True, "moves": [
    {"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate6"}]}
# A hex knight that a piece beside it blocks: (1, 2) passes the neighbour (0, 1).
HEX_KNIGHT = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral12",
              "legs": [[[0, 1]]]}


def probe(pieces: list[dict[str, Any]], rules: dict[str, Any] | None = None,
          width: int = 9, height: int = 10, validate: bool = True) -> dict[str, Any]:
    spec = {
        "name": "lame legs probe",
        "board": {"width": width, "height": height},
        "pieces": [copy.deepcopy(KING), *copy.deepcopy(pieces)],
        "setup": {"mirror": False, "placements": [
            {"piece": "K", "player": 0, "squares": ["a1"]},
            {"piece": "K", "player": 1, "squares": [f"{chr(ord('a') + width - 1)}{height}"]},
        ]},
        "rules": {"win": "checkmate", **(rules or {})},
    }
    if validate:
        assert validate_spec(spec) == []
    return spec


def hex_probe(pieces: list[dict[str, Any]], rules: dict[str, Any] | None = None,
              shape: str | None = None, validate: bool = True) -> dict[str, Any]:
    board: dict[str, Any] = {"width": 7, "height": 7, "lattice": "hex"}
    if shape:
        board["shape"] = shape
    spec = {
        "name": "hex lame legs probe",
        "board": board,
        "pieces": [copy.deepcopy(HEX_KING), *copy.deepcopy(pieces)],
        "setup": {"mirror": False, "orientation": "rotate", "placements": [
            {"piece": "K", "player": 0, "squares": ["d2"]},
            {"piece": "K", "player": 1, "squares": ["d6"]},
        ]},
        "rules": {"win": "checkmate", **(rules or {})},
    }
    if validate:
        assert validate_spec(spec) == []
    return spec


def piece(pid: str, *moves: dict[str, Any]) -> dict[str, Any]:
    return {"id": pid, "name": f"piece {pid}", "value": 3,
            "moves": [copy.deepcopy(m) for m in moves]}


@cache
def _engine(text: str) -> Engine:
    return Engine(json.loads(text))


def movement(spec: dict[str, Any]) -> dict[str, Any]:
    """The compiled tables less the Zobrist keys, which the spec's name seeds."""
    return {k: v for k, v in canonical_tables(compile_spec(spec)).items()
            if not k.startswith("zobrist")}


def engine_of(spec: dict[str, Any]) -> Engine:
    return _engine(json.dumps(spec, sort_keys=True))


def targets(engine: Engine, state: GameState, origin: str) -> set[str]:
    """Where the piece on ``origin`` may go, by the legal list's own words."""
    found = set()
    for text in engine.move_texts(state):
        body = text.lstrip("~")[1:]
        for sep in ("-", "x"):
            if sep in body:
                start, _, rest = body.partition(sep)
                if start == origin:
                    found.add(rest.rstrip("+#=").split("=")[0])
                break
    return found


# ------------------------------------------------------------------- the maps


def test_the_symmetry_maps_are_the_symmetries_expand_symmetry_writes_out() -> None:
    for symmetry, maps in SYMMETRY_MAPS.items():
        for vector in itertools.product(range(-3, 4), repeat=2):
            assert expand_symmetry(vector, symmetry) == {turn(m, vector) for m in maps}


def test_legs_turn_with_their_vector() -> None:
    found, clashes = stated_legs([[2, 3]], [[[0, 1], [1, 2]]], "dihedral8")
    assert clashes == []
    assert found[(2, 3)] == ((0, 1), (1, 2))
    assert found[(-3, 2)] == ((-2, 1), (-1, 0))
    assert found[(3, -2)] == ((1, 0), (2, -1))
    assert len(found) == 8


# ------------------------------------------------------------------- the spelling


def test_janggi_elephant_and_a_hex_lame_knight_validate() -> None:
    probe([piece("E", ELEPHANT)])
    hex_probe([piece("N", HEX_KNIGHT)])
    hex_probe([piece("N", HEX_KNIGHT)], shape="hexagon")


def test_a_hex_lame_with_no_legs_is_still_refused() -> None:
    spec = hex_probe([piece("N", {**HEX_KNIGHT, "legs": None})], validate=False)
    del spec["pieces"][1]["moves"][0]["legs"]
    assert any("must state its legs" in e for e in validate_spec(spec))


@pytest.mark.parametrize(("legs", "words"), [
    ([[[0, 1]], [[1, 0]]], "one entry per vector"),
    ([[]], "non-empty list"),
    ([[[0, 0]]], "non-null"),
    ([[[0, 1], [0, 1]]], "twice"),
    ([[[0, 3]]], "not shorter than its jump"),
    ([[[2, 3]]], "not shorter than its jump"),
    ("x", "one entry per vector"),
])
def test_malformed_legs_are_refused_by_name(legs: Any, words: str) -> None:
    spec = probe([piece("E", {**ELEPHANT, "legs": legs})], validate=False)
    assert any(words in e for e in validate_spec(spec)), validate_spec(spec)


def test_legs_on_a_slide_are_refused() -> None:
    spec = probe([piece("S", {"kind": "slide", "vectors": [[2, 2]], "legs": [[[1, 1]]]})],
                 validate=False)
    assert any("only means something on a 'lame' or a 'leap'" in e
               for e in validate_spec(spec))


def test_legs_two_turns_disagree_on_are_refused() -> None:
    """(2, 2) is its own mirror across the diagonal, which turns a leg (1, 0) into (0, 1)."""
    spec = probe([piece("A", {"kind": "lame", "vectors": [[2, 2]], "symmetry": "dihedral8",
                              "legs": [[[1, 0]]]})], validate=False)
    assert any("two different sets of legs" in e for e in validate_spec(spec))
    # Both of them stated is a set the mirror keeps: the jump is blocked on either.
    probe([piece("A", {"kind": "lame", "vectors": [[2, 2]], "symmetry": "dihedral8",
                       "legs": [[[1, 0], [0, 1]]]})])


def test_legs_on_a_leap_need_blocks_leaps() -> None:
    knight = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8",
              "legs": [[[1, 1]]]}
    spec = probe([piece("N", knight)], validate=False)
    assert any("only rules.obstacles.blocks_leaps" in e for e in validate_spec(spec))
    probe([piece("N", knight)],
          rules={"obstacles": {"squares": ["e5"], "blocks_leaps": True}})


def test_blocks_leaps_on_a_hex_board_wants_every_long_leap_to_state_its_legs() -> None:
    knight = {"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral12"}
    rules = {"obstacles": {"squares": ["c4"], "blocks_leaps": True}}
    for shape in ("hexagon", None):
        spec = hex_probe([piece("N", knight)], rules=rules, shape=shape, validate=False)
        assert any("must state its 'legs'" in e for e in validate_spec(spec)), shape
        hex_probe([piece("N", {**knight, "legs": [[[0, 1]]]})], rules=rules, shape=shape)
        # A lame states its legs anyway, and a one-step leap has none to state.
        hex_probe([piece("N", HEX_KNIGHT)], rules=rules, shape=shape)
    # Without blocks_leaps a hex leap is never blocked and states nothing.
    hex_probe([piece("N", knight)], shape="hexagon")


def test_a_one_step_lame_is_the_lattices_own_step() -> None:
    """Q884 on a hex lattice: [1, -1] is one step and refused; [1, 1] is two and lame."""
    one = hex_probe([piece("N", {"kind": "lame", "vectors": [[1, -1]],
                                 "legs": [[[1, 0]]]})], validate=False)
    assert any("one-square 'lame' leap" in e for e in validate_spec(one))
    hex_probe([piece("N", {"kind": "lame", "vectors": [[1, 1]], "symmetry": "rotate6",
                           "legs": [[[1, 0], [0, 1]]]})])


# ------------------------------------------------------------------- inert when absent


def test_the_computed_leg_stated_is_the_same_table() -> None:
    """Stating `_lame_path`'s leg builds the table the key's absence builds."""
    horse = piece("H", {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8"})
    stated = piece("H", {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8",
                         "legs": [[[0, 1]]]})
    assert spec_hash(probe([horse])) != spec_hash(probe([stated]))
    assert movement(probe([horse])) == movement(probe([stated]))


def test_xiangqi_is_unmoved() -> None:
    spec = json.loads((ROOT / "rulesets" / "xiangqi.json").read_text("utf-8"))
    stated = copy.deepcopy(spec)
    for p in stated["pieces"]:
        for move in p["moves"]:
            if move["kind"] == "lame" and move["vectors"] in ([[1, 2]], [[2, 2]]):
                move["legs"] = [[[0, 1]]] if move["vectors"] == [[1, 2]] else [[[1, 1]]]
    assert validate_spec(stated) == []
    assert movement(spec) == movement(stated)


# ------------------------------------------------------------------- play


def test_janggi_elephant_by_hand() -> None:
    engine = engine_of(probe([piece("E", ELEPHANT)]))
    state = engine.make_state({"a1": ("K", 0), "i10": ("K", 1), "e5": ("E", 0)})
    # From e5, all eight (2, 3) jumps land on the board.
    everywhere = {"g8", "c8", "g2", "c2", "h7", "b7", "h3", "b3"}
    assert targets(engine, state, "e5") == everywhere
    # e6 is the first leg of both upward jumps (g8, c8); f7 the second of g8 alone.
    state = engine.make_state({"a1": ("K", 0), "i10": ("K", 1), "e5": ("E", 0),
                               "e6": ("E", 1)})
    assert targets(engine, state, "e5") == everywhere - {"g8", "c8"}
    state = engine.make_state({"a1": ("K", 0), "i10": ("K", 1), "e5": ("E", 0),
                               "f7": ("E", 1)})
    assert targets(engine, state, "e5") == everywhere - {"g8"}
    # A piece on the computed leg's square for (2, 3), e6, is the only one the old lame
    # read; d7 blocks nothing here, since no jump passes it.
    state = engine.make_state({"a1": ("K", 0), "i10": ("K", 1), "e5": ("E", 0),
                               "g6": ("E", 1)})
    assert targets(engine, state, "e5") == everywhere - {"h7"}


def test_janggi_elephant_checks_through_both_legs() -> None:
    engine = engine_of(probe([piece("E", ELEPHANT)]))
    state = engine.make_state({"e1": ("K", 0), "i10": ("K", 1), "g4": ("E", 1)})
    # g4 to e1 is (-2, -3): legs g3 and f2.
    assert engine.in_check(state, 0)
    for blocker in ("g3", "f2"):
        blocked = engine.make_state({"e1": ("K", 0), "i10": ("K", 1), "g4": ("E", 1),
                                     blocker: ("E", 0)})
        assert not engine.in_check(blocked, 0), blocker


def test_a_horse_blocked_on_the_diagonal_leg() -> None:
    horse = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8",
             "legs": [[[1, 1]]]}
    engine = engine_of(probe([piece("H", horse)]))
    state = engine.make_state({"a1": ("K", 0), "i10": ("K", 1), "e5": ("H", 0),
                               "e6": ("H", 1)})
    assert len(targets(engine, state, "e5")) == 8  # the orthogonal square blocks nothing
    state = engine.make_state({"a1": ("K", 0), "i10": ("K", 1), "e5": ("H", 0),
                               "f6": ("H", 1)})
    # f6 is the diagonal leg of (1, 2) to f7 and of (2, 1) to g6.
    assert targets(engine, state, "e5") == {"d7", "c6", "c4", "d3", "f3", "g4"}


def test_the_second_player_turns_the_legs_too() -> None:
    """The second player's army is the rank reflection, legs and all: its e4 is the leg
    the first player's e6 is, so it blocks the two jumps toward rank 2."""
    engine = engine_of(probe([piece("E", ELEPHANT)]))
    everywhere = {"g8", "c8", "g2", "c2", "h7", "b7", "h3", "b3"}
    state = engine.make_state({"a1": ("K", 0), "i10": ("K", 1), "e5": ("E", 1),
                               "e4": ("E", 0)}, side_to_move=1)
    assert targets(engine, state, "e5") == everywhere - {"g2", "c2"}
    state = engine.make_state({"a1": ("K", 0), "i10": ("K", 1), "e5": ("E", 1),
                               "f3": ("E", 0)}, side_to_move=1)
    assert targets(engine, state, "e5") == everywhere - {"g2"}


def test_a_hex_lame_knight_plays_and_its_neighbour_blocks_it() -> None:
    spec = hex_probe([piece("N", HEX_KNIGHT)], shape="hexagon")
    engine = engine_of(spec)
    alone = engine.make_state({"d2": ("K", 0), "d6": ("K", 1), "d4": ("N", 0)})
    free = targets(engine, alone, "d4")
    assert len(free) == 12  # the centre of a radius-3 hexagon reaches all twelve
    # One neighbour is the leg of two of the twelve jumps, (1, 2) and (-1, 3)'s mirror.
    beside = engine.make_state({"d2": ("K", 0), "d6": ("K", 1), "d4": ("N", 0),
                                "d5": ("N", 1)})
    assert len(free - targets(engine, beside, "d4")) == 2
    state = engine.initial_state()
    assert engine.perft(state, 2) > 0


# ------------------------------------------------------------------- words


def test_the_legs_are_said() -> None:
    sentence = move_sentence(ELEPHANT)
    assert "blocked" in sentence and "(0, 1)" in sentence and "(1, 2)" in sentence


# ------------------------------------------------------------------- Fairy-Stockfish


def chess_with(move: dict[str, Any]) -> dict[str, Any]:
    spec = json.loads((ROOT / "rulesets" / "chess.json").read_text("utf-8"))
    for p in spec["pieces"]:
        if p["id"] == "N":
            p["moves"] = [copy.deepcopy(move)]
    return spec


def test_fairy_lame_path_is_the_one_its_movegen_walks() -> None:
    assert lame_path(1, 2) == ((0, 1),)
    assert lame_path(2, 2) == ((1, 1),)
    assert lame_path(2, 0) == ((1, 0),)
    assert lame_path(2, 3) == ((0, 1), (1, 2))
    assert lame_path(-3, 2) == ((-1, 0), (-2, 1))


def test_fairy_translates_janggi_elephant_and_refuses_every_other_spelling() -> None:
    assert "nZ" in translate(chess_with(ELEPHANT)).config
    for legs in ([[[1, 1], [1, 2]]], [[[0, 1]]], [[[1, 2]]]):
        with pytest.raises(Untranslatable) as refused:
            translate(chess_with({**ELEPHANT, "legs": legs}))
        assert any("legs" in reason for reason in refused.value.reasons)
    # The (2, 3) lame with its one computed leg is not Fairy-Stockfish's nZ either.
    with pytest.raises(Untranslatable) as refused:
        translate(chess_with({k: v for k, v in ELEPHANT.items() if k != "legs"}))
    assert any("nZ" in reason for reason in refused.value.reasons)
    # A horse whose stated leg is the computed one is still the horse.
    assert "horse = n" in translate(chess_with(
        {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8",
         "legs": [[[0, 1]]]})).config
