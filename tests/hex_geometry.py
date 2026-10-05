"""An independent geometry for the native hexagon, for tests only (Q281a1).

A declared test oracle under CLAUDE.md rule 3: it answers where a step lands, and nothing
outside `tests/` imports it. It exists so the compiler's landing function (Q281a2) has
something to disagree with, so it shares no code with `spec/schema.py` or
`engine/compile.py` beyond the square naming both use (`rank * width + file`).

The method differs from the compiler's on purpose. The compiler tries the six copies across
the hexagon's sides in turn (`docs/game-ideas.md` *Q281 specified*, *What the compiler
resolves it into*). This reduces a point modulo the lattice the copies span, by brute
search over ``i * (2R+1, -R) + j * (R, R+1)`` with ``|i|, |j| <= 3``, and reads the side a
step crossed off which copy the point fell in.

Coordinates are the frame's ``(file, rank)``, counted from 0, and a hexagon of radius R
sits in a frame ``2R + 1`` wide. A cell is on the board when its offset from the centre,
``(file - R, rank - R)``, is at most R steps away. A rhombus (the grid, `wrap.files` and
`wrap.ranks`) has its own landing here too, for the nine cases where today's engine can
check the ray walk.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache

Cell = tuple[int, int]
Vector = tuple[int, int]
# Where one step from a cell lands, or `None` where it leaves the board.
Step = Callable[[Cell, Vector], "Cell | None"]

SIDES: tuple[Vector, ...] = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))
DIAGONALS: tuple[Vector, ...] = ((1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2))
KNIGHT: tuple[Vector, ...] = (
    (1, 2), (2, 1), (3, -1), (3, -2), (2, -3), (1, -3),
    (-1, -2), (-2, -1), (-3, 1), (-3, 2), (-2, 3), (-1, 3),
)

# A piece is a list of (vector, slides) pairs.
Piece = tuple[tuple[Vector, bool], ...]
PIECES: dict[str, Piece] = {
    "rook": tuple((v, True) for v in SIDES),
    "bishop": tuple((v, True) for v in DIAGONALS),
    "queen": tuple((v, True) for v in SIDES + DIAGONALS),
    "knight": tuple((v, False) for v in KNIGHT),
    "king": tuple((v, False) for v in SIDES + DIAGONALS),
}

# The copy a point falls in, in lattice units of (2R+1, -R) and (R, R+1), and the side a
# step crosses to reach it. Written out from the spec's table, not derived from it.
SIDE_OF_COPY: dict[tuple[int, int], str] = {
    (1, 0): "se", (-1, 0): "nw",
    (0, 1): "ne", (0, -1): "sw",
    (-1, 1): "n", (1, -1): "s",
}
PAIR_SIDES: dict[str, tuple[str, str]] = {
    "n_s": ("n", "s"), "ne_sw": ("ne", "sw"), "nw_se": ("nw", "se"),
}


def hexdist(q: int, r: int) -> int:
    return max(abs(q), abs(r), abs(q + r))


def basis(radius: int) -> tuple[Vector, Vector]:
    return (2 * radius + 1, -radius), (radius, radius + 1)


def cells(radius: int) -> list[Cell]:
    """The hexagon's cells, in square order."""
    size = 2 * radius + 1
    return [
        (f, r) for r in range(size) for f in range(size)
        if hexdist(f - radius, r - radius) <= radius
    ]


def square(cell: Cell, width: int) -> int:
    return cell[1] * width + cell[0]


def name(cell: Cell) -> str:
    return f"{chr(ord('a') + cell[0])}{cell[1] + 1}"


@cache
def reduce(radius: int, file: int, rank: int) -> tuple[tuple[int, int], Cell]:
    """The copy a frame point falls in, and the board cell it is a copy of.

    Exactly one lattice point within the search is at most R from the point, because the
    copies tile the plane; more than one is a broken tiling and raises.
    """
    a, b = basis(radius)
    q, r = file - radius, rank - radius
    found = [
        ((i, j), (q - i * a[0] - j * b[0] + radius, r - i * a[1] - j * b[1] + radius))
        for i in range(-3, 4)
        for j in range(-3, 4)
        if hexdist(q - i * a[0] - j * b[0], r - i * a[1] - j * b[1]) <= radius
    ]
    if len(found) != 1:
        raise ValueError(f"({file}, {rank}) at R={radius} reduces to {found}")
    return found[0]


def may_leave(setting: dict[str, str]) -> frozenset[str]:
    """The sides one player may leave by, from ``{pair: value}`` for that player."""
    sides: set[str] = set()
    for pair, value in setting.items():
        if value == "both":
            sides.update(PAIR_SIDES[pair])
        elif value != "none":
            assert value in PAIR_SIDES[pair], (pair, value)
            sides.add(value)
    return frozenset(sides)


def hex_step(radius: int, leave: frozenset[str]) -> Step:
    """One step on a hexagon whose player may leave by the sides in ``leave``."""

    def step(cell: Cell, vector: Vector) -> Cell | None:
        copy, landed = reduce(radius, cell[0] + vector[0], cell[1] + vector[1])
        if copy == (0, 0):
            return landed
        side = SIDE_OF_COPY.get(copy)
        if side is None:
            raise ValueError(f"{vector} from {cell} lands two copies out at R={radius}")
        return landed if side in leave else None

    return step


def rhombus_step(
    width: int, height: int, files: tuple[bool, bool], ranks: tuple[bool, bool]
) -> Step:
    """One step on a width by height grid, each axis's (low, high) edge open or not."""

    def axis(value: int, size: int, open_edges: tuple[bool, bool]) -> int | None:
        if 0 <= value < size:
            return value
        return value % size if open_edges[0 if value < 0 else 1] else None

    def step(cell: Cell, vector: Vector) -> Cell | None:
        f = axis(cell[0] + vector[0], width, files)
        r = axis(cell[1] + vector[1], height, ranks)
        return None if f is None or r is None else (f, r)

    return step


def ray(step: Step, origin: Cell, vector: Vector) -> list[Cell]:
    """The cells a slide passes, stopping off the board or on a cell already visited."""
    seen = {origin}
    out: list[Cell] = []
    here = origin
    while True:
        there = step(here, vector)
        if there is None or there in seen:
            return out
        out.append(there)
        seen.add(there)
        here = there


def targets(step: Step, origin: Cell, piece: Piece) -> set[Cell]:
    """A lone piece's distinct targets from one cell."""
    out: set[Cell] = set()
    for vector, slides in piece:
        if slides:
            out.update(ray(step, origin, vector))
        else:
            there = step(origin, vector)
            if there is not None and there != origin:
                out.add(there)
    return out


def lone_piece_total(step: Step, board: Iterable[Cell], piece: Piece) -> int:
    """A lone piece's distinct targets, summed over every cell it could stand on."""
    return sum(len(targets(step, cell, piece)) for cell in board)


def turn(vector: Vector) -> Vector:
    """A sixth of a turn about the centre, anticlockwise."""
    q, r = vector
    return -r, q + r
