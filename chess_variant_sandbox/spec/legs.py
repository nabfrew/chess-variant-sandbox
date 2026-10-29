"""A jump's stated legs (SR1a, `docs/plan-schema-next.md` §1 rank 1), and the symmetry maps
that turn them.

Kept out of `complexity`, which is in the metrics fingerprint (`measurement.METRICS_SOURCES`):
these are read by the validator, the compiler, the relation and the Fairy translation, none of
which a score reads, so an edit here should not stale a library measurement. The legs'
complexity charge (Q1054) reads the spelling itself, in `complexity._legs_extras`.
"""

from __future__ import annotations

from typing import Any

# Each symmetry's group as linear maps ``(x, y) -> (a x + b y, c x + d y)``, written
# ``(a, b, c, d)``: the same turns and mirrors `expand_symmetry` writes out for one vector
# (`complexity.expand_symmetry`; `tests/test_lame_legs.py` holds the two together), kept as maps so that a lame leap's
# stated legs (SR1a) turn with its vector by the very map that turned it.
_ROTATE4 = ((1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0))
_ROTATE6 = (
    (1, 0, 0, 1), (0, -1, 1, 1), (-1, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, -1), (1, 1, -1, 0),
)
SYMMETRY_MAPS: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "none": ((1, 0, 0, 1),),
    "lateral": ((1, 0, 0, 1), (-1, 0, 0, 1)),
    "rotate4": _ROTATE4,
    "dihedral8": _ROTATE4 + tuple((a, b, -c, -d) for a, b, c, d in _ROTATE4),
    "rotate6": _ROTATE6,
    "dihedral12": _ROTATE6 + tuple((c, d, a, b) for a, b, c, d in _ROTATE6),
}


def turn(matrix: tuple[int, int, int, int], vector: tuple[int, int]) -> tuple[int, int]:
    """``vector`` under one of `SYMMETRY_MAPS`' maps."""
    a, b, c, d = matrix
    x, y = vector
    return a * x + b * y, c * x + d * y


Legs = tuple[tuple[int, int], ...]


def stated_legs(
    vectors: Any, legs: Any, symmetry: str
) -> tuple[dict[tuple[int, int], Legs], list[tuple[int, int]]]:
    """A lame leap's (or a blocked leap's) stated ``legs``, per direction (SR1a).

    ``legs[i]`` are the squares, as steps from the mover, that block ``vectors[i]``; every
    map of ``symmetry`` turns the vector and its legs together. Returns the direction ->
    legs map (each direction's legs sorted, so the table is a function of the spec alone)
    and the directions two maps gave different legs, which the validator refuses: a jump
    whose leg depends on which turn of it was meant has no single reading. Assumes a
    well-formed spelling (the validator's shape checks run first).
    """
    found: dict[tuple[int, int], Legs] = {}
    clashes: list[tuple[int, int]] = []
    for vector, own in zip(vectors, legs, strict=True):
        for matrix in SYMMETRY_MAPS[symmetry]:
            direction = turn(matrix, (vector[0], vector[1]))
            turned = tuple(sorted({turn(matrix, (leg[0], leg[1])) for leg in own}))
            if found.setdefault(direction, turned) != turned and direction not in clashes:
                clashes.append(direction)
    return found, clashes
