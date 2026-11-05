"""SR5g (`docs/plan-schema-next.md` §2.10 row P1): portals on a hex lattice.

The ruled reading: a piece comes out one hex king's step beyond the exit end -- the twelve:
six neighbours across an edge, six diagonals through a corner -- in the one of those twelve
directions nearest the angle it travelled in the plane. A move along one of the hex's twelve
lines comes out continuing it; a leap comes out along the nearest line, so the hex knight
comes out diagonally. No vector ties, which the builder asserts rather than breaking ties.
Fold portals need no direction and lift with it.

**The probe** (fed past the validator and the bridge's hold on SR5f's build
`46c83b0585de37ee`, Glinski's hexagon with one pair on d6 and h6): the old sign rule read a
hex vector's sign on the square's eight directions, so a rook along an edge line came out
right (its sign is a hex neighbour), but a bishop on the (-2, 1) diagonal from f5 came out on
g7, the (-1, 1) *neighbour* of h6, not f7 on its own diagonal; and the exit table was eight
wide, naming two directions (+1, +1) and (-1, -1) that are a hex diagonal and leaving four
diagonals unnamed. The core now reads the lattice's direction count (`PortalTables::n_dirs`,
12 on hex) and twelve-bit masks a mode.

The oracle here is a declared test oracle (CLAUDE.md rule 3): it takes the nearest of the
twelve by floating angle in the plane, worked out from the hex geometry itself, never
`compile.portal_direction`, and steps cells by axial arithmetic against the hexagon's bound,
never a compiled table.
"""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.compile import (
    HEX_PORTAL_DIRECTIONS,
    WILD,
    compile_spec,
    hex_line_offset,
    portal_direction,
)
from chess_variant_sandbox.engine.moves import MOVE_PORTAL, decode_move, move_type
from chess_variant_sandbox.spec import load_spec, validate_spec
from chess_variant_sandbox.spec.describe import describe_rules
from chess_variant_sandbox.spec.schema import MAX_BOARD_DIM, hex_distance
from chess_variant_sandbox.viz.wrapped import portal_transit

ROOT = Path(__file__).resolve().parent.parent
GLINSKI = ROOT / "archive" / "hex-glinski-hexagon.json"
W = 11  # Glinski's hexagon is an 11 x 11 rhombus cut to R = 5
R = 5

# The twelve, by the oracle's own reckoning: the six steps and the six diagonals.
TWELVE = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
          (1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]


def _plane(vector: tuple[int, int]) -> tuple[float, float]:
    q, r = vector
    return q + r / 2, r * math.sqrt(3) / 2


def _gap(a: tuple[int, int], b: tuple[int, int]) -> float:
    """The angle between two vectors in the plane, in degrees, in [0, 180]."""
    (ax, ay), (bx, by) = _plane(a), _plane(b)
    return abs(math.degrees(math.atan2(ax * by - ay * bx, ax * bx + ay * by)))


def _nearest(vector: tuple[int, int]) -> tuple[int, int]:
    return min(TWELVE, key=lambda line: _gap(vector, line))


def _cell(name: str) -> tuple[int, int]:
    return ord(name[0]) - ord("a"), int(name[1:]) - 1


def _name(cell: tuple[int, int]) -> str:
    return f"{chr(ord('a') + cell[0])}{cell[1] + 1}"


def _on_board(cell: tuple[int, int]) -> bool:
    q, r = cell
    return 0 <= q < W and 0 <= r < W and hex_distance(q - R, r - R) <= R


def _square(name: str) -> int:
    q, r = _cell(name)
    return r * W + q


def _turned(name: str) -> str:
    """The half turn about the centre: Glinski's second player's image of a cell."""
    q, r = _cell(name)
    return _name((2 * R - q, 2 * R - r))


def hex_portals(kind: str = "momentum") -> dict[str, Any]:
    """Glinski's hexagon with one pair on d6 and h6, each the other's half turn."""
    spec = load_spec(GLINSKI)
    spec["name"] = "glinski with a portal pair"
    pair: dict[str, Any] = {"orange": "d6", "blue": "h6"}
    if kind == "fold":
        pair = {"kind": "fold", **pair}
    if kind == "moved":
        # Either player may step an end as a hex king does (the hexagon's own turns).
        pair = {**pair, "moved_by": {"orange": [0, 1], "blue": [0, 1]},
                "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate6"}]}
    spec["rules"]["portals"] = [pair]
    return spec


# ------------------------------------------------------------------ the twelve lines


def test_the_compilers_twelve_are_the_hex_king_steps_by_angle() -> None:
    assert sorted(HEX_PORTAL_DIRECTIONS) == sorted(TWELVE)
    for index, line in enumerate(HEX_PORTAL_DIRECTIONS):
        x, y = _plane(line)
        assert math.isclose(math.degrees(math.atan2(y, x)) % 360, 30 * index, abs_tol=1e-9)


def test_no_lattice_vector_up_to_the_move_bound_lies_equally_near_two_lines() -> None:
    """Every vector a hex move can make (each component within the largest board) has one
    nearest line, and by a margin far above float noise: none lies 15 degrees off a line."""
    bound = MAX_BOARD_DIM
    closest_to_a_tie = 0.0
    for q in range(-bound, bound + 1):
        for r in range(-bound, bound + 1):
            if (q, r) == (0, 0):
                continue
            gaps = sorted(_gap((q, r), line) for line in TWELVE)
            assert gaps[1] - gaps[0] > 1e-3, ((q, r), gaps[:2])
            _, offset = hex_line_offset(q, r)
            closest_to_a_tie = max(closest_to_a_tie, offset)
            assert HEX_PORTAL_DIRECTIONS[portal_direction(q, r, True)] == _nearest((q, r))
    assert closest_to_a_tie < 0.5 - 1e-3


def test_a_line_continues_and_the_knight_comes_out_diagonally() -> None:
    for line in TWELVE:
        for length in (1, 2, 5):
            vector = (line[0] * length, line[1] * length)
            assert HEX_PORTAL_DIRECTIONS[portal_direction(*vector, True)] == line
    knight = load_spec(GLINSKI)["pieces"][1]
    assert knight["id"] == "N"
    diagonals = set(TWELVE[6:])
    for x, y in knight["moves"][0]["vectors"]:
        assert HEX_PORTAL_DIRECTIONS[portal_direction(x, y, True)] in diagonals


# ------------------------------------------------------------------ transits, both players

# (piece, where it stands, the step it travels, the cell it enters by) for the first player.
CASES = [
    ("R", "d3", (0, 1)),    # an edge line, up the board
    ("R", "a6", (1, 0)),    # an edge line, across it
    ("B", "c5", (1, 1)),    # a diagonal
    ("B", "b7", (2, -1)),   # a diagonal, a long step
    ("B", "f5", (-2, 1)),   # the probe's diagonal
    ("N", "c4", (1, 2)),    # the hex knight
    ("N", "a7", (3, -1)),   # the hex knight, the other side of a diagonal
]


def _expected_exit(origin: str, step: tuple[int, int], into: str, out: str) -> str:
    """The oracle: walk ``step`` from ``origin`` to ``into``, come out of ``out`` along the
    nearest of the twelve to the step."""
    q, r = _cell(origin)
    while (q, r) != _cell(into):
        q, r = q + step[0], r + step[1]
        assert _on_board((q, r)), (origin, step)
    line = _nearest(step)
    exit_cell = (_cell(out)[0] + line[0], _cell(out)[1] + line[1])
    assert _on_board(exit_cell)
    return _name(exit_cell)


@pytest.mark.parametrize("player", [0, 1])
@pytest.mark.parametrize(("pid", "origin", "step"), CASES)
def test_a_transit_comes_out_where_p1_says(
    pid: str, origin: str, step: tuple[int, int], player: int
) -> None:
    spec = hex_portals()
    assert validate_spec(spec) == []
    engine = Engine(spec)
    into, out = ("d6", "h6") if player == 0 else ("h6", "d6")
    if player == 1:
        origin, step = _turned(origin), (-step[0], -step[1])
    expected = _expected_exit(origin, step, into, out)
    state = engine.make_state({
        "d6": ("1o", WILD), "h6": ("1b", WILD), "g1": ("K", 0), "e11": ("K", 1),
        origin: (pid, player),
    }, side_to_move=player)
    transits = {
        target for start, target, kind, _ in map(decode_move, engine.legal_moves(state))
        if start == _square(origin) and kind == MOVE_PORTAL
    }
    assert transits == {_square(expected)}, (origin, step, expected)
    move = next(
        m for m in engine.legal_moves(state)
        if move_type(m) == MOVE_PORTAL and decode_move(m)[0] == _square(origin)
    )
    after = engine.apply(state, move)
    assert after.board[_square(expected)] == state.board[_square(origin)]
    assert after.board[_square(origin)] == 0
    assert engine._compute_hash(after) == after.key


@pytest.mark.parametrize("player", [0, 1])
def test_a_transit_takes_at_the_exit_and_checks_through_it(player: int) -> None:
    """The rook up its edge line comes out on h7 (or, turned, d5); an enemy there is taken,
    an own piece there blocks it, and an enemy king there is in check from the rook."""
    engine = Engine(hex_portals())
    origin, exit_square = ("d3", "h7") if player == 0 else (_turned("d3"), _turned("h7"))
    base = {"d6": ("1o", WILD), "h6": ("1b", WILD), origin: ("R", player)}
    kings = {"g1": ("K", 0), "e11": ("K", 1)}

    def transits(placement: dict[str, tuple[str, int]]) -> set[int]:
        state = engine.make_state(placement, side_to_move=player)
        return {
            target for start, target, kind, _ in map(decode_move, engine.legal_moves(state))
            if start == _square(origin) and kind == MOVE_PORTAL
        }

    assert transits({**base, **kings, exit_square: ("B", 1 - player)}) == {_square(exit_square)}
    assert transits({**base, **kings, exit_square: ("B", player)}) == set()
    own_king = "g1" if player == 0 else "e11"
    checked = engine.make_state(
        {**base, own_king: ("K", player), exit_square: ("K", 1 - player)},
        side_to_move=1 - player,
    )
    assert engine.in_check(checked)


def test_a_fold_pair_plays_on_hex() -> None:
    spec = hex_portals("fold")
    assert validate_spec(spec) == []
    engine = Engine(spec)
    state = engine.make_state(
        {"g1": ("K", 0), "e11": ("K", 1), "d3": ("R", 0)}, folds=["d6", "h6"],
    )
    targets = {
        target for start, target, _, _ in map(decode_move, engine.legal_moves(state))
        if start == _square("d3")
    }
    assert {_square("d6"), _square("h6")} <= targets


# ------------------------------------------------------------------ undo, perft, play


def _walk(engine: Engine, seed: int, plies: int = 80) -> int:
    rng = random.Random(seed)
    state = engine.initial_state()
    played = 0
    for _ in range(plies):
        if engine.terminal(state) is not None:
            break
        moves = engine.legal_moves(state)
        through = [m for m in moves if move_type(m) == MOVE_PORTAL]
        move = rng.choice(through) if through and rng.random() < 0.7 else rng.choice(moves)
        played += int(move in through)
        state = engine.apply(state, move)
        assert engine._compute_hash(state) == state.key
    return played


@pytest.mark.parametrize("kind", ["momentum", "fold", "moved"])
def test_seeded_games_keep_the_key_and_meet_transits(kind: str) -> None:
    engine = Engine(hex_portals(kind))
    played = sum(_walk(engine, seed) for seed in range(4))
    if kind != "fold":
        assert played > 0


def test_perft_matches_an_apply_count_through_a_transit() -> None:
    """Make and undo are exact: perft (which unmakes) agrees with a count built by apply,
    from a position where both players can transit at once."""
    engine = Engine(hex_portals())
    state = engine.make_state({
        "d6": ("1o", WILD), "h6": ("1b", WILD), "g1": ("K", 0), "e11": ("K", 1),
        "d3": ("R", 0), "c4": ("N", 0), _turned("c5"): ("B", 1), _turned("a6"): ("R", 1),
    })

    def count(position: Any, depth: int) -> int:
        if depth == 0:
            return 1
        return sum(count(engine.apply(position, m), depth - 1)
                   for m in engine.legal_moves(position))

    assert engine.perft(state, 3) == count(state, 3)


def test_the_square_lattice_keeps_its_eight() -> None:
    """A square board's tables are the eight-direction reading: every momentum mask within
    sixteen bits and the exit table eight wide (the corpus's tables byte for byte are the
    record's proof, generation 85)."""
    spec = load_spec(ROOT / "archive" / "cylinder-wormhole.json")
    compiled = compile_spec(spec)
    assert compiled.portals is not None
    assert {len(row) for player in compiled.portals.exits for row in player} == {8}
    hexed = compile_spec(hex_portals())
    assert hexed.portals is not None
    assert {len(row) for player in hexed.portals.exits for row in player} == {12}


def test_the_refusal_is_gone() -> None:
    spec = copy.deepcopy(hex_portals())
    assert validate_spec(spec) == []


def test_the_rules_say_where_a_hex_transit_comes_out() -> None:
    said = " ".join(describe_rules(hex_portals()))
    assert "twelve lines" in said and "comes out along the nearer" in said
    assert "twelve lines" not in " ".join(describe_rules(hex_portals("fold")))
    square = load_spec(ROOT / "archive" / "cylinder-wormhole.json")
    assert "twelve lines" not in " ".join(describe_rules(square))


def test_the_page_is_told_a_hex_transits_way_out() -> None:
    """`viz.wrapped.portal_transit` names the way out among the twelve (`exitHex`), so the
    page's words and the trail read the hex table; a square board's transit is as it was."""
    engine = Engine(hex_portals())
    state = engine.make_state({
        "d6": ("1o", WILD), "h6": ("1b", WILD), "g1": ("K", 0), "e11": ("K", 1),
        "c4": ("N", 0),
    })
    move = next(m for m in engine.legal_moves(state) if move_type(m) == MOVE_PORTAL)
    transit = portal_transit(engine, state, move)
    assert transit is not None and transit["exitHex"] is True
    assert HEX_PORTAL_DIRECTIONS[transit["exitDir"]] == _nearest((1, 2)) == (1, 1)
    assert (transit["portalName"], transit["portalOutName"]) == ("d6", "h6")
