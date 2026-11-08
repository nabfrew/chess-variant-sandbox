"""SR11a: a turning path on a lame rider (`docs/plan-schema-next.md` §2.13 PA4).

``path: "circle"`` or ``"zigzag"`` on a ``lame`` with a ``range`` other than 1 is a turning
lame rider: each jump of the path is blocked by the legs of its own direction, read from the
square it jumps from (the stated ``legs`` turned with that direction, or `_lame_path`'s one leg
when none are stated), written as modeless squares before its landing as `_lame_ride` writes a
straight lame rider's; a piece on a leg ends that path there. A single lame leap (range 1, the
default) has no second jump to turn and stays refused. The compiler builds each path as a slide
entry of its own (`compile._turning_rides`), so the core is unchanged.

Held here: the spelling and the refusal, hand-worked table entries (the computed leg and
Janggi's elephant's stated legs), a hand-worked board where a piece on a turned jump's leg ends
those paths and not the first jump's, the compiled entries against `path_oracle`'s own walk on
every square for both players, the oracle over seeded games, a hex lame knight turning on a
hexagon, the words, the distance and the Fairy-Stockfish refusal.
"""

from __future__ import annotations

import copy
import json
import random
from functools import cache
from typing import Any

import path_oracle
import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.engine.moves import decode_move
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.fairy.variant import Untranslatable, translate
from chess_variant_sandbox.spec import spec_hash, validate_spec
from chess_variant_sandbox.spec.describe import move_sentence
from chess_variant_sandbox.spec.distance import distance as spec_distance
from chess_variant_sandbox.spec.plain import plain_problem

KING = {"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}
# The knight's lame (the xiangqi horse), ridden round a circle or along a zigzag.
LAME_CIRCLE = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8", "path": "circle",
               "range": 0}
LAME_ZIGZAG = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral8", "path": "zigzag",
               "range": 3}
# Janggi's elephant: (2, 3) blocked on (0, 1) and (1, 2), turned with each direction.
ELEPHANT_CIRCLE = {"kind": "lame", "vectors": [[2, 3]], "symmetry": "dihedral8",
                   "legs": [[[0, 1], [1, 2]]], "path": "circle", "range": 0}
# A hex knight a neighbour blocks, as `tests/test_lame_legs.py` has it.
HEX_KNIGHT_CIRCLE = {"kind": "lame", "vectors": [[1, 2]], "symmetry": "dihedral12",
                     "legs": [[[0, 1]]], "path": "circle", "range": 0}
HEX_ROSE = {"kind": "slide", "vectors": [[1, 2]], "symmetry": "dihedral12", "path": "circle"}

BLOCKER = {"kind": "leap", "vectors": [[0, 1]], "mode": "move"}


def spec_of(move: dict[str, Any], placements: list[dict[str, Any]], width: int = 8,
            height: int = 8, validate: bool = True) -> dict[str, Any]:
    """The rider X, kings K and blockers P (which move a square forward and never capture)."""
    spec = {
        "name": "turning lame rider probe",
        "board": {"width": width, "height": height},
        "pieces": [
            {"id": "X", "name": "rider", "value": 4, "moves": [copy.deepcopy(move)]},
            {"id": "K", "name": "king", "royal": True, "value": 0, "moves": [KING]},
            {"id": "P", "name": "blocker", "value": 1, "moves": [BLOCKER]},
        ],
        "setup": {"mirror": False, "placements": copy.deepcopy(placements)},
        "rules": {"win": "checkmate", "max_plies": 200},
    }
    if validate:
        assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _at(square: str, width: int = 8) -> int:
    return (int(square[1:]) - 1) * width + ord(square[0]) - ord("a")


def _name(square: int, width: int = 8) -> str:
    return f"{chr(ord('a') + square % width)}{square // width + 1}"


@cache
def _engine(text: str) -> Engine:
    return Engine(json.loads(text))


def engine_of(spec: dict[str, Any]) -> Engine:
    return _engine(json.dumps(spec, sort_keys=True))


def targets(spec: dict[str, Any], state: GameState, origin: str) -> set[str]:
    """Where the piece on ``origin`` may go, by the core's pseudo-legal list."""
    width = spec["board"]["width"]
    engine = engine_of(spec)
    return {
        _name(target, width)
        for start, target, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
        if start == _at(origin, width)
    }


def _kings(white: str, black: str) -> list[dict[str, Any]]:
    return [{"piece": "K", "player": 0, "squares": [white]},
            {"piece": "K", "player": 1, "squares": [black]}]


# ------------------------------------------------------------------- the spelling


@pytest.mark.parametrize("move", [LAME_CIRCLE, LAME_ZIGZAG, {**LAME_CIRCLE, "range": 3},
                                  {**LAME_ZIGZAG, "range": 0}, ELEPHANT_CIRCLE])
def test_a_turning_lame_rider_validates(move: dict[str, Any]) -> None:
    spec_of(move, [{"piece": "X", "player": 0, "squares": ["d4"]}, *_kings("a1", "h8")])


@pytest.mark.parametrize("move", [
    {k: v for k, v in LAME_CIRCLE.items() if k != "range"},  # range 1 by default
    {**LAME_CIRCLE, "range": 1},
    {**LAME_ZIGZAG, "range": 1},
])
def test_a_single_lame_leap_still_never_turns(move: dict[str, Any]) -> None:
    spec = spec_of(move, _kings("a1", "h8"), validate=False)
    lines = [line for line in validate_spec(spec)
             if "on a single lame leap (range 1) is one jump" in line]
    assert lines, validate_spec(spec)
    said = plain_problem(lines[0], spec)
    assert said[0].isupper() and said.endswith("."), said


def test_a_lame_paths_directions_must_still_go_round() -> None:
    spec = spec_of({"kind": "lame", "vectors": [[1, 2]], "path": "circle", "range": 0},
                   _kings("a1", "h8"), validate=False)
    assert any("leave a gap of 360 degrees" in line for line in validate_spec(spec))


def test_a_turning_lame_rider_is_a_new_game() -> None:
    placements = _kings("a1", "h8")
    straight = spec_of({k: v for k, v in LAME_CIRCLE.items() if k != "path"}, placements)
    turning = spec_of(LAME_CIRCLE, placements)
    assert spec_hash(straight) != spec_hash(turning)
    assert spec_distance(straight, turning) > 0


# ------------------------------------------------------------- hand-worked tables


def _entries(spec: dict[str, Any], square: str, player: int = 0) -> set[tuple[int, ...]]:
    compiled = compile_spec(spec)
    width = spec["board"]["width"]
    return {tuple(entry[0]) for entry in compiled.slides[player][0][_at(square, width)]}


def test_the_computed_leg_is_read_from_the_square_each_jump_starts_from() -> None:
    """From d4 the circle that starts (2, 1) and turns counter-clockwise: leg e4, landing f5;
    then (1, 2) from f5: leg f6, landing g7; then (-1, 2) from g7 would land off the board."""
    spec = spec_of(LAME_CIRCLE, _kings("a1", "h8"))
    want = tuple(_at(s) for s in ("e4", "f5", "f6", "g7"))
    assert want in _entries(spec, "d4")
    compiled = compile_spec(spec)
    entry = next(e for e in compiled.slides[0][0][_at("d4")] if tuple(e[0]) == want)
    assert entry[1][0] == 0 and entry[1][2] == 0 and entry[1][1] and entry[1][3]


def test_a_zigzag_lame_rider_of_three_jumps() -> None:
    """From c2, (1, 2) then (2, 1) then (1, 2): legs c3, e4, f6 before d4, f5, g7."""
    spec = spec_of(LAME_ZIGZAG, _kings("a8", "h1"))
    want = tuple(_at(s) for s in ("c3", "d4", "e4", "f5", "f6", "g7"))
    assert want in _entries(spec, "c2")
    assert all(len(entry) <= 6 for entry in _entries(spec, "c2"))  # three jumps at most


def test_stated_legs_turn_with_each_direction() -> None:
    """Janggi's elephant circling from a1 on a 9 by 10 board: (3, 2) over b1 and c2 to d3;
    (2, 3) over d4 and e5 to f6; (-2, 3) over e8 and f7 (its legs (-1, 2) and (0, 1), sorted)
    to d9; then (-3, 2) would land off the board."""
    spec = spec_of(ELEPHANT_CIRCLE, _kings("i1", "i10"), width=9, height=10)
    want = tuple(_at(s, 9) for s in ("b1", "c2", "d3", "d4", "e5", "f6", "e8", "f7", "d9"))
    assert want in _entries(spec, "a1")


@pytest.mark.parametrize("move", [LAME_CIRCLE, LAME_ZIGZAG, ELEPHANT_CIRCLE],
                         ids=["circle", "zigzag", "elephant"])
@pytest.mark.parametrize("player", [0, 1])
def test_every_entry_is_the_oracles_walk(move: dict[str, Any], player: int) -> None:
    """On every square of an empty board, the compiled paths are the oracle's walks, legs and
    landings in order, for either player."""
    width, height = (9, 10) if "legs" in move else (8, 8)
    spec = spec_of(move, _kings("a1", "a2"), width=width, height=height)
    compiled = compile_spec(spec)
    for origin in range(width * height):
        found = {tuple(e[0]) for e in compiled.slides[player][0][origin]}
        said = {
            tuple(square for square, _ in walk)
            for walk in path_oracle.walks(move, player, origin, width, height) if walk
        }
        assert found == said, (player, origin)


# ------------------------------------------------------------- a leg ends a path


# A zigzag of two knight jumps from a1 on a five by five board: (2, 1) over b1 to c2 then (1, 2)
# over c3 to d4, or (1, 2) over a2 to b3 then (2, 1) over c3 to d4; (2, 1) then (2, -1) over d2 to
# e1; (1, 2) then (-1, 2) over b4 to a5.
_SMALL = {**LAME_ZIGZAG, "range": 2}


def _small(*blockers: str) -> tuple[dict[str, Any], GameState]:
    placements = [{"piece": "X", "player": 0, "squares": ["a1"]}, *_kings("e3", "c5")]
    placements += [{"piece": "P", "player": 0, "squares": [b]} for b in blockers]
    spec = spec_of(_SMALL, placements, width=5, height=5)
    return spec, engine_of(spec).initial_state()


def test_the_small_board_by_hand() -> None:
    spec, state = _small()
    assert targets(spec, state, "a1") == {"c2", "d4", "b3", "e1", "a5"}


def test_a_piece_on_a_turned_jumps_leg_ends_those_paths_there() -> None:
    """c3 is the leg of the second jump of both paths to d4, and of no first jump."""
    spec, state = _small("c3")
    assert targets(spec, state, "a1") == {"c2", "b3", "e1", "a5"}


def test_a_piece_on_a_first_jumps_leg_ends_its_paths_and_not_the_others() -> None:
    """b1 is the leg of (2, 1) from a1: c2 and e1 go, d4 is still reached through a2, b3, c3."""
    spec, state = _small("b1")
    assert targets(spec, state, "a1") == {"d4", "b3", "a5"}


def test_the_oracle_says_the_same_squares() -> None:
    for blockers in ((), ("c3",), ("b1",), ("a2", "d2")):
        spec, state = _small(*blockers)
        found = {(_at("a1", 5), _at(t, 5)) for t in targets(spec, state, "a1")}
        assert found == path_oracle.path_targets(spec, state.board, 0), blockers


# ------------------------------------------------------------- the oracle over games


def _game_spec() -> dict[str, Any]:
    placements = [
        {"piece": "K", "player": 0, "squares": ["e1"]},
        {"piece": "K", "player": 1, "squares": ["e8"]},
        {"piece": "X", "player": 0, "squares": ["b1", "g1"]},
        {"piece": "X", "player": 1, "squares": ["b8", "g8"]},
        {"piece": "Y", "player": 0, "squares": ["c1"]},
        {"piece": "Y", "player": 1, "squares": ["c8"]},
        {"piece": "P", "player": 0, "squares": ["a2", "c2", "e2", "g2"]},
        {"piece": "P", "player": 1, "squares": ["b7", "d7", "f7", "h7"]},
    ]
    spec = spec_of(LAME_CIRCLE, placements, validate=False)
    spec["pieces"].append({"id": "Y", "name": "zigzag rider", "value": 3,
                           "moves": [copy.deepcopy(LAME_ZIGZAG)]})
    spec["pieces"][2]["moves"] = [{"kind": "leap", "vectors": [[0, 1]], "mode": "move"}]
    spec["setup"]["mirror"] = False
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def _legal_by_oracle(spec: dict[str, Any], engine: Engine, state: GameState) -> set[int]:
    me = state.side_to_move
    keep = set()
    for move in engine.pseudo_legal_moves(state):
        after = engine.apply(state, move)
        royal = path_oracle.royal_square(spec, after.board, me)
        if royal is None or not path_oracle.attacked(spec, after.board, royal, 1 - me):
            keep.add(move)
    return keep


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_core_and_the_oracle_agree_over_a_seeded_game(seed: int) -> None:
    """Both players' turning lame riders, every ply: the core's pseudo-legal moves of the
    riders are the oracle's, and the legal list is the pseudo-legal one filtered by the
    oracle's attack walk (a rider checks through its own path, blocked on its legs)."""
    spec = _game_spec()
    engine = engine_of(spec)
    n_types = len(spec["pieces"])
    riders = {0, 3}
    chooser = random.Random(seed)
    state = engine.initial_state()
    checked = 0
    for _ in range(80):
        found = {
            (origin, target)
            for origin, target, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
            if (state.board[origin] - 1) % n_types in riders
        }
        assert found == path_oracle.path_targets(spec, state.board, state.side_to_move)
        legal = engine.legal_moves(state)
        assert set(legal) == _legal_by_oracle(spec, engine, state)
        checked += 1
        if not legal or engine.terminal(state, legal) is not None:
            break
        state = engine.apply(state, chooser.choice(legal))
    assert checked >= 2


# ------------------------------------------------------------- a hexagon


def _hex(move: dict[str, Any]) -> dict[str, Any]:
    spec = {
        "name": "hex turning lame probe",
        "board": {"width": 7, "height": 7, "lattice": "hex", "shape": "hexagon"},
        "pieces": [
            {"id": "K", "name": "king", "value": 0, "royal": True,
             "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate6"}]},
            {"id": "N", "name": "rider", "value": 3, "moves": [copy.deepcopy(move)]},
        ],
        "setup": {"mirror": False, "orientation": "rotate", "placements": [
            {"piece": "K", "player": 0, "squares": ["d2"]},
            {"piece": "K", "player": 1, "squares": ["d6"]},
        ]},
        "rules": {"win": "checkmate"},
    }
    assert validate_spec(spec) == [], validate_spec(spec)
    return spec


def test_a_hex_lame_knight_with_stated_legs_turns_on_a_hexagon() -> None:
    """Every path is leg, landing, leg, landing, ...; its landings are a prefix of a hex
    rose's path from the same cell (the rose is the same circle with nothing to block it);
    and the game plays."""
    lame = compile_spec(_hex(HEX_KNIGHT_CIRCLE))
    rose = compile_spec(_hex(HEX_ROSE))
    cells = 0
    for origin, entries in enumerate(lame.slides[0][1]):
        roses = [tuple(e[0]) for e in rose.slides[0][1][origin]]
        for squares, modes, _ in entries:
            assert all((m == 0) == (i % 2 == 0) for i, m in enumerate(modes)), modes
            landings = tuple(squares[1::2])
            assert any(path[:len(landings)] == landings for path in roses), (origin, squares)
            cells += 1
    assert cells > 0
    engine = engine_of(_hex(HEX_KNIGHT_CIRCLE))
    state = engine.make_state({"d2": ("K", 0), "d6": ("K", 1), "d4": ("N", 0)})
    assert engine.perft(state, 2) > 0


def test_a_hex_neighbour_on_a_leg_ends_paths() -> None:
    spec = _hex(HEX_KNIGHT_CIRCLE)
    engine = engine_of(spec)
    alone = engine.make_state({"d2": ("K", 0), "d6": ("K", 1), "d4": ("N", 0)})
    beside = engine.make_state({"d2": ("K", 0), "d6": ("K", 1), "d4": ("N", 0),
                                "d5": ("N", 1)})
    origin = alone.board.index(2)  # the first player's rider, the second type
    assert beside.board.index(2) == origin

    def reach(state: GameState) -> set[int]:
        return {t for s, t, _, _ in map(decode_move, engine.pseudo_legal_moves(state))
                if s == origin}

    assert reach(beside) < reach(alone)  # d5 is no landing, only a leg


# ------------------------------------------------------------- words, Fairy-Stockfish


def test_the_words_say_jumps_turning_and_legs() -> None:
    sentence = move_sentence(LAME_CIRCLE)
    assert "jump" in sentence and "circle" in sentence and "passes" in sentence, sentence
    stated = move_sentence(ELEPHANT_CIRCLE)
    assert "(0, 1)" in stated and "(1, 2)" in stated and "circle" in stated, stated
    zigzag = move_sentence(LAME_ZIGZAG)
    assert "zigzag" in zigzag and "three" in zigzag, zigzag


def test_fairy_stockfish_refuses_it_by_name() -> None:
    spec = spec_of(LAME_CIRCLE, [{"piece": "X", "player": 0, "squares": ["d4"]},
                                 *_kings("a1", "h8")])
    with pytest.raises(Untranslatable, match="circle path turns as it goes"):
        translate(spec)
