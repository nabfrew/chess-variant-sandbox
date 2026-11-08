"""The declared Python oracle for turning paths (T1, `docs/archive/plan-pieces-moves.md` §5.2).

`pieces[].moves[].path`: a slide whose path turns as it goes. ``circle`` starts with any of
the move's directions and takes the next one round at every step, always the same way, either
way; ``zigzag`` alternates its first direction with a neighbour less than half a turn away,
either one. A path ends off the board or before a square it has already been on, its origin
included; each step's landing is looked at, empty to stop on and go on, occupied to end the
path (taken if it is the other side's). ``range`` counts steps, ``min_range`` the nearest it
may stop on.

This module states that again from the spec, by its own means: the directions laid out by
floating angle (`math.atan2`), never `schema.turning_order`, and squares worked out by file
and rank, never through a compiled table. It also walks attack for the pieces the test specs
use (leaps, straight slides and turning ones), so check can be judged without the core. It
covers plain square boards only: no wrap, walls, regions, conditions or facings. CLAUDE.md
rule 3 allows it: it is a declared test oracle.
"""

from __future__ import annotations

import math
from typing import Any

Square = int


def _images(symmetry: str) -> list[Any]:
    """The symmetry's maps of one square, written out by hand."""
    maps = [lambda x, y: (x, y)]
    if symmetry == "lateral":
        maps.append(lambda x, y: (-x, y))
    if symmetry in ("rotate4", "dihedral8"):
        maps += [lambda x, y: (-y, x), lambda x, y: (-x, -y), lambda x, y: (y, -x)]
    if symmetry == "dihedral8":
        maps += [lambda x, y, m=m: (m(x, y)[0], -m(x, y)[1]) for m in list(maps)]
    return maps


def directions(move: dict[str, Any]) -> set[tuple[int, int]]:
    """A descriptor's directions for the first player, its symmetry expanded by hand."""
    symmetry = move.get("symmetry", "none")
    return {m(x, y) for x, y in move["vectors"] for m in _images(symmetry)}


def legs(move: dict[str, Any], side: int) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """SR11a: a lame move's legs per direction, for ``side``: the stated ones carried by the
    same hand-written map as their vector, else the one square a jump passes (a step along
    its longer axis, or the midpoint of an even diagonal, halved toward the mover)."""
    out: dict[tuple[int, int], set[tuple[int, int]]] = {}
    symmetry = move.get("symmetry", "none")
    for index, (x, y) in enumerate(move["vectors"]):
        for m in _images(symmetry):
            dx, dy = m(x, y)
            if "legs" in move:
                own = {m(a, b) for a, b in move["legs"][index]}
            elif abs(dx) == abs(dy):
                own = {(int(dx / 2), int(dy / 2))}
            elif abs(dx) > abs(dy):
                own = {(1 if dx > 0 else -1, 0)}
            else:
                own = {(0, 1 if dy > 0 else -1)}
            out[(dx, dy)] = own
    if side == 1:
        out = {(x, -y): {(a, -b) for a, b in own} for (x, y), own in out.items()}
    return out


def _by_angle(steps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(steps, key=lambda v: math.atan2(v[1], v[0]) % (2 * math.pi))


def _turn(a: tuple[int, int], b: tuple[int, int]) -> float:
    """The counter-clockwise turn from ``a`` to ``b``, in degrees, in [0, 360)."""
    return math.degrees(math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])) % 360


def step_sequences(move: dict[str, Any], side: int, length: int) -> list[list[tuple[int, int]]]:
    """Every sequence of ``length`` steps the move's path may take, for ``side`` (the second
    player's directions are the first's with the rank reflected)."""
    steps = directions(move)
    if side == 1:
        steps = {(x, -y) for x, y in steps}
    ring = _by_angle(steps)
    count = len(ring)
    out = []
    for start in range(count):
        if move.get("path") == "circle":
            out.append([ring[(start + k) % count] for k in range(length)])
            out.append([ring[(start - k) % count] for k in range(length)])
            continue
        ahead, behind = ring[(start + 1) % count], ring[(start - 1) % count]
        if count > 1 and 0 < _turn(ring[start], ahead) < 180:
            out.append([ring[start] if k % 2 == 0 else ahead for k in range(length)])
        if count > 1 and 0 < _turn(behind, ring[start]) < 180:
            out.append([ring[start] if k % 2 == 0 else behind for k in range(length)])
    return out


def paths(move: dict[str, Any], side: int, origin: Square, width: int,
          height: int) -> list[list[Square]]:
    """Every path from ``origin`` on an empty board: the squares in order, until the edge or
    a square already visited (the origin included), capped at ``range`` steps."""
    cap = move.get("range", 0) or width * height
    out = []
    for sequence in step_sequences(move, side, cap):
        f, r = origin % width, origin // width
        seen = {origin}
        squares = []
        for dx, dy in sequence:
            f, r = f + dx, r + dy
            if not (0 <= f < width and 0 <= r < height) or r * width + f in seen:
                break
            seen.add(r * width + f)
            squares.append(r * width + f)
        out.append(squares)
    return out


def walks(move: dict[str, Any], side: int, origin: Square, width: int,
          height: int) -> list[list[tuple[Square, bool]]]:
    """Every path from ``origin`` on an empty board as ``(square, is_landing)`` in the order
    they are looked at. A slide's are its landings; a turning lame rider's (SR11a) put each
    jump's legs, read from the square it jumps from, before its landing. A path ends at the
    edge or at a square already met (the origin, a leg or a landing), whether a landing or a
    leg is what met it, as `compile._lame_ride` ends a straight ride."""
    if move["kind"] != "lame":
        return [[(s, True) for s in squares]
                for squares in paths(move, side, origin, width, height)]
    blocked_by = legs(move, side)
    cap = move.get("range", 1) or width * height
    out = []
    for sequence in step_sequences(move, side, cap):
        f, r = origin % width, origin // width
        seen = {origin}
        walk: list[tuple[Square, bool]] = []
        for dx, dy in sequence:
            land_f, land_r = f + dx, r + dy
            if not (0 <= land_f < width and 0 <= land_r < height):
                break
            target = land_r * width + land_f
            if target in seen:
                break
            here = []
            for a, b in sorted(blocked_by[(dx, dy)]):
                leg_f, leg_r = f + a, r + b
                leg = leg_r * width + leg_f
                if not (0 <= leg_f < width and 0 <= leg_r < height) or leg in seen \
                        or leg == target or leg in here:
                    break
                here.append(leg)
            else:
                seen.update(here)
                seen.add(target)
                walk += [(leg, False) for leg in here] + [(target, True)]
                f, r = land_f, land_r
                continue
            break
        out.append(walk)
    return out


def _owner(board: bytes, square: Square, n_types: int) -> int | None:
    value = board[square]
    return None if value == 0 else (value - 1) // n_types


def path_targets(spec: dict[str, Any], board: bytes, side: int) -> set[tuple[Square, Square]]:
    """Every (origin, target) ``side``'s turning paths may make on ``board``."""
    width, height = spec["board"]["width"], spec["board"]["height"]
    n_types = len(spec["pieces"])
    found: set[tuple[Square, Square]] = set()
    for origin, value in enumerate(board):
        if value == 0 or (value - 1) // n_types != side:
            continue
        for move in spec["pieces"][(value - 1) % n_types]["moves"]:
            if move.get("kind") not in ("slide", "lame") or move.get("path") not in (
                "circle", "zigzag"
            ):
                continue
            mode = move.get("mode", "both")
            least = move.get("min_range", 1)
            for walk in walks(move, side, origin, width, height):
                step = 0
                for square, landing in walk:
                    held = _owner(board, square, n_types)
                    if not landing:
                        if held is not None:
                            break  # SR11a: a piece on a leg ends the path there
                        continue
                    step += 1
                    if step >= least and (
                        (held is None and mode != "capture")
                        or (held is not None and held != side and mode != "move")
                    ):
                        found.add((origin, square))
                    if held is not None:
                        break
    return found


def attacked(spec: dict[str, Any], board: bytes, target: Square, by: int) -> bool:
    """Whether ``by`` attacks ``target``: a leap's capture, a straight slide's or a turning
    path's (a turning lame rider's too, SR11a: a piece on a leg stops it), each walked here
    (no hop, straight lame, sweep or condition: the test specs have none that capture, bar
    the pawn's one-square slides, which never capture)."""
    width, height = spec["board"]["width"], spec["board"]["height"]
    n_types = len(spec["pieces"])
    for origin, value in enumerate(board):
        if value == 0 or (value - 1) // n_types != by:
            continue
        for move in spec["pieces"][(value - 1) % n_types]["moves"]:
            if move.get("mode", "both") == "move" or move.get("condition"):
                continue
            kind = move["kind"]
            if kind in ("slide", "lame") and move.get("path") in ("circle", "zigzag"):
                least = move.get("min_range", 1)
                for walk in walks(move, by, origin, width, height):
                    step = 0
                    for square, landing in walk:
                        step += landing
                        if landing and square == target and step >= least:
                            return True
                        if board[square]:
                            break
                continue
            if kind in ("slide", "leap"):
                lines = []
                reach = 1 if kind == "leap" else (move.get("range", 0) or max(width, height))
                for dx, dy in directions(move):
                    dy = -dy if by == 1 else dy
                    f, r = origin % width, origin // width
                    line = []
                    for _ in range(reach):
                        f, r = f + dx, r + dy
                        if not (0 <= f < width and 0 <= r < height):
                            break
                        line.append(r * width + f)
                    lines.append(line)
            else:
                raise AssertionError(f"the path oracle does not walk a {kind}")
            for line in lines:
                for square in line:
                    if square == target:
                        return True
                    if board[square]:
                        break
    return False


def royal_square(spec: dict[str, Any], board: bytes, side: int) -> Square | None:
    n_types = len(spec["pieces"])
    royal = {i for i, piece in enumerate(spec["pieces"]) if piece.get("royal")}
    for square, value in enumerate(board):
        if value and (value - 1) // n_types == side and (value - 1) % n_types in royal:
            return square
    return None
