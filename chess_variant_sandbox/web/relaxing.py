"""The map's last step before display: each community drawn as one place of its own.

Q1165. `mapping` places games by likeness (t-SNE) and `families` groups them by the mutual
nearest graph (Louvain); the two are different readings of one relation, so a community's
games could be drawn in two or three knots with another community's dots between them.
On the library of 2026-11-08, 26 of the 61 games in a coloured community sat nearer another
coloured community's centre than their own. This module relaxes the fitted picture so
the colours read as places, and keeps what the picture means:

1. **Where a community is** is where most of it was drawn: the geometric median of its
   members' places, which sits in its biggest knot rather than in the gap between two.
   An unusual game -- in no community -- is a group of one and stays where it was drawn.
2. **Inside a community** its games fill a disc around that place, one sunflower spot
   each (spacing `spacing`, the picture's own nearest gap), and each game takes the free
   spot nearest to where it was drawn relative to the place, after its distance from the
   place is replaced by its rank. So a game keeps its bearing from the community's middle,
   the most typical games stay inside and the outliers on the rim.
3. **Communities keep apart**: each disc, and each unusual game, is a circle; overlapping
   circles are pushed apart, the smaller moving more, until a clear lane of `spacing`
   lies between every two, and a community's disc is kept at least its own width from a
   neighbour's middle (`_apart`) so its rim games are nearer their own middle than any
   other. Pushing apart moves the places as little as it has to, so communities alike stay
   neighbours and the picture's long distances are t-SNE's.

`graphing` then pushes any two dots still too near apart, as it did before.

**A pure function of its arguments.** Groups and games are visited in id order and every
step is a fixed number of passes, so the same picture and the same communities give the
same map. Nothing here names a game, and no number is read off a library: the spacing is
the caller's, the passes a bound that is never reached on a hundred games.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

Point = tuple[float, float]

# The golden angle: each sunflower spot turns this far from the last, so spots spread evenly.
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))

# The share of the plane a sunflower's spots fill at `spacing`: a disc of radius
# `DISC * spacing * sqrt(n)` gives each of its n spots about `spacing` to its nearest.
DISC = 0.6

# Bounds on the iterative steps: the geometric median's, and the push apart's passes.
MEDIAN_STEPS = 200
PUSH_PASSES = 2000


def relaxed(
    placed: Mapping[str, Point],
    family_of: Mapping[str, str | None],
    spacing: float,
) -> dict[str, Point]:
    """The picture with each community drawn as one disc and the discs kept apart.

    `placed` is the fitted picture, `family_of` each game's community (`None` for an
    unusual game), and `spacing` the distance the picture keeps between two neighbours.
    """
    ids = sorted(placed)
    groups: dict[str, list[str]] = {}
    for digest in ids:
        family = family_of.get(digest)
        groups.setdefault(digest if family is None else f"family:{family}", []).append(digest)
    names = sorted(groups)
    centres = {name: _median([placed[digest] for digest in groups[name]]) for name in names}
    radii = {name: _radius(len(groups[name]), spacing) for name in names}
    offsets = {
        name: _filled(groups[name], placed, centres[name], radii[name], spacing)
        for name in names
    }
    kept = {name: len(groups[name]) >= 2 for name in names}
    moved = _pushed(names, centres, radii, kept, spacing)
    found: dict[str, Point] = {}
    for name in names:
        cx, cy = moved[name]
        for digest, (dx, dy) in offsets[name].items():
            found[digest] = (cx + dx, cy + dy)
    return found


def strays(
    placed: Mapping[str, Point], family_of: Mapping[str, str | None], least: int = 2
) -> list[str]:
    """The games drawn nearer another community's middle than their own's.

    A community's middle is the mean of its members' places; only communities of `least`
    or more count, and only their members are judged. The map's measure of whether its
    colours are places (Q1165).
    """
    members: dict[str, list[str]] = {}
    for digest in sorted(placed):
        family = family_of.get(digest)
        if family is not None:
            members.setdefault(family, []).append(digest)
    middles = {
        family: (
            sum(placed[d][0] for d in held) / len(held),
            sum(placed[d][1] for d in held) / len(held),
        )
        for family, held in sorted(members.items())
        if len(held) >= least
    }
    found = []
    for family, held in sorted(members.items()):
        if family not in middles:
            continue
        for digest in held:
            own = math.dist(placed[digest], middles[family])
            if any(
                math.dist(placed[digest], middle) < own
                for other, middle in middles.items()
                if other != family
            ):
                found.append(digest)
    return found


def _radius(size: int, spacing: float) -> float:
    """A group's disc: about `spacing` around each of its games, half a gap for one game."""
    if size == 1:
        return spacing / 2
    return DISC * spacing * math.sqrt(size)


def _median(points: Sequence[Point]) -> Point:
    """The geometric median (Weiszfeld), started from the mean, in a fixed number of steps."""
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    if len(points) <= 2:
        return (x, y)
    for _ in range(MEDIAN_STEPS):
        wx = wy = total = 0.0
        for px, py in points:
            d = math.hypot(px - x, py - y)
            if d < 1e-9:
                continue
            wx += px / d
            wy += py / d
            total += 1.0 / d
        if total == 0.0:
            break
        nx, ny = wx / total, wy / total
        if math.hypot(nx - x, ny - y) < 1e-9:
            x, y = nx, ny
            break
        x, y = nx, ny
    return (x, y)


def _filled(
    members: Sequence[str],
    placed: Mapping[str, Point],
    centre: Point,
    radius: float,
    spacing: float,
) -> dict[str, Point]:
    """Each member's offset from its group's middle: a sunflower spot near its bearing.

    A member's target keeps its bearing from the middle and takes, for its distance, its
    rank among the members' distances spread over the disc. Members take spots nearest
    their targets, the innermost first, ties by id.
    """
    n = len(members)
    if n == 1:
        return {members[0]: (0.0, 0.0)}
    cx, cy = centre
    if n == 2:
        # A pair is a line `spacing` long through its middle, along the line it was drawn on.
        first, second = sorted(members)
        dx, dy = placed[second][0] - placed[first][0], placed[second][1] - placed[first][1]
        apart = math.hypot(dx, dy)
        ux, uy = (dx / apart, dy / apart) if apart > 1e-9 else (1.0, 0.0)
        half = spacing / 2
        return {first: (-ux * half, -uy * half), second: (ux * half, uy * half)}
    far = sorted(
        members, key=lambda d: (math.hypot(placed[d][0] - cx, placed[d][1] - cy), d)
    )
    # The sunflower is turned so its outermost spot lies on the outermost member's bearing:
    # the disc keeps the community's orientation rather than the spiral's own.
    outer = far[-1]
    turn = math.atan2(placed[outer][1] - cy, placed[outer][0] - cx) - (n - 1) * GOLDEN
    spots = [
        (
            radius * math.sqrt((k + 0.5) / n) * math.cos(k * GOLDEN + turn),
            radius * math.sqrt((k + 0.5) / n) * math.sin(k * GOLDEN + turn),
        )
        for k in range(n)
    ]
    free = list(range(n))
    found: dict[str, Point] = {}
    for rank, digest in enumerate(far):
        bearing = math.atan2(placed[digest][1] - cy, placed[digest][0] - cx)
        reach = radius * math.sqrt((rank + 0.5) / n)
        target = (reach * math.cos(bearing), reach * math.sin(bearing))
        best = min(free, key=lambda k: (math.dist(spots[k], target), k))
        free.remove(best)
        found[digest] = spots[best]
    return found


def _apart(
    first: str, second: str, radii: Mapping[str, float], kept: Mapping[str, bool], gap: float
) -> float:
    """How far apart two groups' middles must be.

    A lane of `gap` between their circles; and where both are communities, the larger's
    whole radius again plus the lane, so a game on its rim is nearer its own middle than
    the neighbour's (it is at most a radius from its own, at least that plus the lane from
    the other's).
    """
    least = radii[first] + radii[second] + gap
    if kept[first] and kept[second]:
        least = max(least, 2 * max(radii[first], radii[second]) + gap)
    return least


def _pushed(
    names: Sequence[str],
    centres: Mapping[str, Point],
    radii: Mapping[str, float],
    kept: Mapping[str, bool],
    gap: float,
) -> dict[str, Point]:
    """The groups' middles pushed apart until none is nearer another than `_apart` says.

    Each overlap is shared by area, so the smaller circle moves more; pairs are visited in
    id order, pass after pass, until a pass moves nothing.
    """
    spot = {name: [centres[name][0], centres[name][1]] for name in names}
    weight = {name: radii[name] ** 2 for name in names}
    for _ in range(PUSH_PASSES):
        moved = False
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                least = _apart(first, second, radii, kept, gap)
                dx = spot[first][0] - spot[second][0]
                dy = spot[first][1] - spot[second][1]
                d = math.hypot(dx, dy)
                if d >= least - 1e-6:
                    continue
                if d < 1e-9:
                    dx, dy, d = 1.0, 0.0, 1e-9
                ux, uy = dx / d, dy / d
                short = least - d + 1e-6
                share = weight[second] / (weight[first] + weight[second])
                spot[first][0] += ux * short * share
                spot[first][1] += uy * short * share
                spot[second][0] -= ux * short * (1 - share)
                spot[second][1] -= uy * short * (1 - share)
                moved = True
        if not moved:
            break
    return {name: (spot[name][0], spot[name][1]) for name in names}
