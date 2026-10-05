"""The independent hex geometry (`tests/hex_geometry.py`) against its own hand-counted cases.

The oracle is what the compiler's hexagon landing (Q281a2) and the gate's end-to-end
proofs (Q281a4) will be compared with, so it is checked here first, by four things that
do not depend on it being right:

* the tiling it rests on, counted point by point;
* the spec's two pre-registered tables (`docs/game-ideas.md` *Q281 specified*, *What a
  seam does to the tables*), at R = 5 and R = 3, every row and every piece;
* the group facts the spec states: on the torus every side and diagonal visits every
  cell, and a sixth of a turn carries the lattice to itself;
* today's engine, where today's engine can answer: the unjoined hexagon cut by walls, and
  nine rhombus seams, which check the ray walk (stop on a revisited cell, count distinct
  targets) that a hexagon's seams will lean on.
"""

from __future__ import annotations

from typing import Any

import hex_geometry as oracle
import pytest

from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.moves import move_to

RADII = range(1, 8)
PIECE_ORDER = ("rook", "bishop", "queen", "knight", "king")
PAIRS = ("n_s", "ne_sw", "nw_se")
SIDES = ("n", "s", "ne", "sw", "nw", "se")

# The spec's tables, pre-registered before anything was built. Rook, bishop, queen,
# knight, king: a lone piece's distinct targets from every cell, summed.
TABLES: dict[int, dict[str, tuple[int, ...]]] = {
    5: {
        "nothing joined": (2_070, 1_080, 3_150, 720, 900),
        "one pair both ways": (3_276, 5_364, 6_614, 844, 964),
        "one pair one way": (2_708, 3_246, 5_101, 782, 932),
        "n_s and ne_sw both ways": (8_190, 7_232, 8_190, 968, 1_028),
        "the torus": (8_190, 8_190, 8_190, 1_092, 1_092),
    },
    3: {
        "nothing joined": (498, 246, 744, 216, 324),
        "one pair both ways": (774, 1_002, 1_214, 292, 364),
        "one pair one way": (651, 632, 1_044, 254, 344),
        "n_s and ne_sw both ways": (1_332, 1_296, 1_332, 368, 404),
        "the torus": (1_332, 1_332, 1_332, 444, 444),
    },
}

# Each row's settings. "Any of the three" and "any side" are every one of them.
ROWS: dict[str, list[dict[str, str]]] = {
    "nothing joined": [{}],
    "one pair both ways": [{pair: "both"} for pair in PAIRS],
    "one pair one way": [
        {pair: side} for pair in PAIRS for side in oracle.PAIR_SIDES[pair]
    ],
    "n_s and ne_sw both ways": [{"n_s": "both", "ne_sw": "both"}],
    "the torus": [{pair: "both" for pair in PAIRS}],
}


def _centres(radius: int) -> list[tuple[int, int]]:
    """The board's centre and its six copies', as offsets from the centre."""
    a, b = oracle.basis(radius)
    return [(0, 0)] + [
        (i * a[0] + j * b[0], i * a[1] + j * b[1]) for i, j in oracle.SIDE_OF_COPY
    ]


class TestTheTiling:
    @pytest.mark.parametrize("radius", RADII)
    def test_the_six_copies_tile(self, radius: int) -> None:
        """Every point within 2R+1 of the centre lies in exactly one of seven hexagons."""
        reach = 2 * radius + 1
        centres = _centres(radius)
        for q in range(-reach, reach + 1):
            for r in range(-reach, reach + 1):
                if oracle.hexdist(q, r) > reach:
                    continue
                inside = [
                    c for c in centres if oracle.hexdist(q - c[0], r - c[1]) <= radius
                ]
                assert len(inside) == 1, ((q, r), inside)

    @pytest.mark.parametrize("radius", RADII)
    def test_the_cell_count(self, radius: int) -> None:
        assert len(oracle.cells(radius)) == 3 * radius * radius + 3 * radius + 1

    @pytest.mark.parametrize("radius", RADII)
    def test_a_cell_reduces_to_itself(self, radius: int) -> None:
        for cell in oracle.cells(radius):
            assert oracle.reduce(radius, *cell) == ((0, 0), cell)

    @pytest.mark.parametrize("radius", RADII)
    def test_a_sixth_of_a_turn_carries_the_lattice_to_itself(self, radius: int) -> None:
        a, b = oracle.basis(radius)
        assert oracle.turn(a) == b
        assert oracle.turn(b) == (b[0] - a[0], b[1] - a[1])
        for vector in oracle.SIDES + oracle.DIAGONALS + oracle.KNIGHT:
            assert oracle.hexdist(*oracle.turn(vector)) == oracle.hexdist(*vector)


class TestTheWorkedCrossings:
    def test_c11_up_leaves_by_n_and_lands_on_i1(self) -> None:
        step = oracle.hex_step(5, frozenset({"n"}))
        assert oracle.name(step((2, 10), (0, 1)) or (-1, -1)) == "i1"
        assert oracle.hex_step(5, frozenset({"s"}))((2, 10), (0, 1)) is None

    def test_k3_right_leaves_by_se_and_lands_on_a8(self) -> None:
        step = oracle.hex_step(5, frozenset({"se"}))
        assert oracle.name(step((10, 2), (1, 0)) or (-1, -1)) == "a8"
        assert oracle.hex_step(5, frozenset({"nw"}))((10, 2), (1, 0)) is None

    def test_a_step_two_copies_out_is_refused_not_answered(self) -> None:
        """At R = 1 a knight's leap of 3 exceeds R + 1: refusal 7's case."""
        step = oracle.hex_step(1, frozenset(oracle.SIDE_OF_COPY.values()))
        with pytest.raises(ValueError, match="two copies out"):
            for cell in oracle.cells(1):
                for vector in oracle.KNIGHT:
                    step(cell, vector)


class TestThePreRegisteredTables:
    @pytest.mark.parametrize("radius", sorted(TABLES))
    @pytest.mark.parametrize("row", list(ROWS))
    def test_the_row(self, radius: int, row: str) -> None:
        board = oracle.cells(radius)
        for setting in ROWS[row]:
            step = oracle.hex_step(radius, oracle.may_leave(setting))
            counts = tuple(
                oracle.lone_piece_total(step, board, oracle.PIECES[piece])
                for piece in PIECE_ORDER
            )
            assert counts == TABLES[radius][row], setting

    def test_the_torus_is_the_same_from_every_cell_at_r5(self) -> None:
        step = oracle.hex_step(5, oracle.may_leave({pair: "both" for pair in PAIRS}))
        for cell in oracle.cells(5):
            for piece, expected in (("rook", 90), ("bishop", 90), ("queen", 90),
                                    ("knight", 12), ("king", 12)):
                assert len(oracle.targets(step, cell, oracle.PIECES[piece])) == expected

    @pytest.mark.parametrize("pair", PAIRS)
    def test_both_is_the_union_of_the_two_one_way_sides(self, pair: str) -> None:
        low, high = oracle.PAIR_SIDES[pair]
        both = oracle.hex_step(5, oracle.may_leave({pair: "both"}))
        one = oracle.hex_step(5, oracle.may_leave({pair: low}))
        other = oracle.hex_step(5, oracle.may_leave({pair: high}))
        for cell in oracle.cells(5):
            for piece in ("knight", "king"):
                moves = oracle.PIECES[piece]
                assert oracle.targets(both, cell, moves) == (
                    oracle.targets(one, cell, moves) | oracle.targets(other, cell, moves)
                )


class TestTheTorusIsCyclic:
    @pytest.mark.parametrize("radius", RADII)
    def test_every_side_and_diagonal_generates_the_group(self, radius: int) -> None:
        """Stepping one direction from the centre visits all 3R^2+3R+1 cells, then returns."""
        order = 3 * radius * radius + 3 * radius + 1
        centre = (radius, radius)
        for vector in oracle.SIDES + oracle.DIAGONALS:
            here, steps = centre, 0
            while True:
                _, here = oracle.reduce(radius, here[0] + vector[0], here[1] + vector[1])
                steps += 1
                if here == centre or steps > order:
                    break
            assert steps == order, (vector, steps)


def _lone_spec(
    width: int, height: int, piece: str, wrap: dict[str, Any], walls: list[str]
) -> dict[str, Any]:
    """One piece type on a hex-lattice rhombus, for a lone piece of the first player."""
    moves = []
    slides = [list(v) for v, slide in oracle.PIECES[piece] if slide]
    leaps = [list(v) for v, slide in oracle.PIECES[piece] if not slide]
    if slides:
        moves.append({"kind": "slide", "symmetry": "none", "vectors": slides})
    if leaps:
        moves.append({"kind": "leap", "symmetry": "none", "vectors": leaps})
    rules: dict[str, Any] = {"win": "annihilation"}
    if wrap:
        rules["wrap"] = wrap
    if walls:
        rules["obstacles"] = {"squares": walls}
    open_squares = [
        f"{chr(ord('a') + f)}{r + 1}"
        for r in range(height) for f in range(width)
        if f"{chr(ord('a') + f)}{r + 1}" not in walls
    ]
    return {
        "schema_version": 14,
        "name": f"lone {piece}",
        "description": "one piece, counted from every cell",
        "board": {"width": width, "height": height, "lattice": "hex"},
        "pieces": [{"id": "X", "name": piece, "value": 5, "moves": moves}],
        "setup": {"mirror": False, "orientation": "rotate", "placements": [
            {"piece": "X", "player": 0, "squares": [open_squares[0]]},
            {"piece": "X", "player": 1, "squares": [open_squares[-1]]},
        ]},
        "rules": rules,
    }


def _engine_targets(engine: Engine, cell: tuple[int, int]) -> list[int]:
    state = engine.make_state({oracle.name(cell): ("X", 0)})
    moves = engine.legal_moves(state)
    assert engine.perft(state, 1) == len(moves)
    return sorted(move_to(m) for m in moves)


class TestTodaysEngineAgrees:
    @pytest.mark.parametrize("radius", [3, 5])
    @pytest.mark.parametrize("piece", PIECE_ORDER)
    def test_the_unjoined_hexagon_cut_by_walls(self, radius: int, piece: str) -> None:
        size = 2 * radius + 1
        board = oracle.cells(radius)
        walls = sorted(
            oracle.name((f, r)) for r in range(size) for f in range(size)
            if (f, r) not in set(board)
        )
        engine = Engine(_lone_spec(size, size, piece, {}, walls))
        step = oracle.hex_step(radius, frozenset())
        for cell in board:
            expected = sorted(
                oracle.square(t, size) for t in oracle.targets(step, cell, oracle.PIECES[piece])
            )
            assert _engine_targets(engine, cell) == expected, oracle.name(cell)

    # Nine rhombus seams: three settings by the three kinds of line. A 7 by 5 rhombus so
    # the two axes cannot be confused, and one-way edges so a seam read backwards shows.
    @pytest.mark.parametrize(
        ("wrap", "files", "ranks"),
        [
            ({"files": "both"}, (True, True), (False, False)),
            ({"ranks": "top"}, (False, False), (False, True)),
            ({"files": "left", "ranks": "both"}, (True, False), (True, True)),
        ],
        ids=["files both", "ranks top", "files left ranks both"],
    )
    @pytest.mark.parametrize("piece", ["rook", "bishop", "knight"])
    def test_the_nine_rhombus_seams(
        self, wrap: dict[str, str], files: tuple[bool, bool], ranks: tuple[bool, bool],
        piece: str,
    ) -> None:
        width, height = 7, 5
        engine = Engine(_lone_spec(width, height, piece, wrap, []))
        step = oracle.rhombus_step(width, height, files, ranks)
        for r in range(height):
            for f in range(width):
                expected = sorted(
                    oracle.square(t, width)
                    for t in oracle.targets(step, (f, r), oracle.PIECES[piece])
                )
                assert _engine_targets(engine, (f, r)) == expected, oracle.name((f, r))
