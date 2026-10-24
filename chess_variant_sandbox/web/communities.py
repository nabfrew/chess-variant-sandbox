"""Communities in a weighted graph: Louvain modularity, made deterministic.

The map's lines join games each among the other's nearest by likeness, and the families
those lines form are what the map is drawn around. This finds them. It knows nothing
about games: it takes node names and a weight per pair and returns a partition, which
`web/families.py` names and remembers (`docs/archive/plan-map.md` M2).

**The method** is `docs/archive/plan-graph-map.md` §4's, pre-registered there and first run in
`docs/reference/graph-map-probe.py`: weighted Louvain at resolution 1, from `ORDERS` node
orders, the partition with the highest modularity kept, and a community Louvain leaves in
two pieces split into its connected parts, which is the guarantee Leiden's refinement
gives. On the committed corpus it finds 13 communities from 90 lines, the same partition
from 96 of 100 orders.

**Deterministic, whatever order the library arrives in.** The nodes are sorted and every
pair is keyed in sorted order before anything runs, so the input is a function of the
graph and not of how it was listed. Each order is drawn from a `random.Random` of its own,
seeded by its number, never from the module-level generator. When two runs tie on
modularity, the one with fewer communities wins, then the one whose sorted member lists
sort first -- so the answer never depends on which run happened to finish a tie first.

No dependency: the environment has no graph library, and the sixty lines here are ones
this repository controls. They were checked by their modularity and by agreement across
the orders rather than against a reference implementation (§8 of the plan).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping

# How many node orders Louvain is run from, and the modularity resolution. The plan's.
ORDERS = 100
RESOLUTION = 1.0

# A move must gain more than this to be taken, so that two candidates equal up to the
# last bit of a float sum do not decide a community by rounding.
_EPSILON = 1e-12
# Modularity is compared at this many places, for the same reason.
_PLACES = 12

Pair = tuple[str, str]


def partition(
    weights: Mapping[Pair, float],
    orders: int = ORDERS,
    resolution: float = RESOLUTION,
) -> list[list[str]]:
    """The communities of the nodes these weights join, each a sorted list of names.

    Biggest first, then by first member. A node no positive weight touches is in no
    community and is not returned; a graph with no weight has none.
    """
    joined = canonical(weights)
    nodes = sorted({node for pair in joined for node in pair})
    if not nodes:
        return []
    found: dict[tuple[tuple[str, ...], ...], float] = {}
    for seed in range(max(1, orders)):
        parts = louvain(nodes, joined, random.Random(seed), resolution)
        shape = _shape(split_disconnected(parts, joined))
        if shape not in found:
            found[shape] = modularity(shape, joined, resolution)
    best = min(
        found,
        key=lambda shape: (-round(found[shape], _PLACES), len(shape), shape),
    )
    return sorted((list(part) for part in best), key=lambda part: (-len(part), part))


def canonical(weights: Mapping[Pair, float]) -> dict[Pair, float]:
    """The same weights with every pair in sorted order, positive ones only, sorted."""
    held: dict[Pair, float] = defaultdict(float)
    for (first, second), weight in weights.items():
        if first == second or weight <= 0:
            continue
        held[(first, second) if first <= second else (second, first)] += float(weight)
    return {pair: held[pair] for pair in sorted(held)}


def louvain(
    nodes: list[str],
    weights: Mapping[Pair, float],
    rng: random.Random,
    resolution: float = RESOLUTION,
) -> list[list[str]]:
    """One run of weighted Louvain, visiting nodes in the order `rng` shuffles them into.

    Local moves until no node gains by moving, then each community becomes one node and
    the moves start again, until a whole pass moves nothing.
    """
    adjacent: dict[str, dict[str, float]] = {node: {} for node in nodes}
    for (first, second), weight in weights.items():
        adjacent[first][second] = adjacent[first].get(second, 0.0) + weight
        adjacent[second][first] = adjacent[second].get(first, 0.0) + weight
    loops: dict[str, float] = {node: 0.0 for node in nodes}
    members: dict[str, list[str]] = {node: [node] for node in nodes}
    total = sum(weights.values())
    if total <= 0:
        return [[node] for node in nodes]

    while True:
        degree = {node: sum(adjacent[node].values()) + 2 * loops[node] for node in adjacent}
        home = {node: node for node in adjacent}
        held = dict(degree)
        order = list(adjacent)
        rng.shuffle(order)
        moved_any = False
        while True:
            moved = False
            for node in order:
                here = home[node]
                held[here] -= degree[node]
                links: dict[str, float] = defaultdict(float)
                for other, weight in adjacent[node].items():
                    links[home[other]] += weight
                share = resolution * degree[node] / (2 * total)
                best, gain = here, links.get(here, 0.0) - held[here] * share
                for candidate, weight in links.items():
                    value = weight - held[candidate] * share
                    if value > gain + _EPSILON:
                        best, gain = candidate, value
                home[node] = best
                held[best] += degree[node]
                if best != here:
                    moved = moved_any = True
            if not moved:
                break
        if not moved_any:
            break

        # Each community becomes one node; a line inside one becomes its self-loop.
        folded: dict[str, dict[str, float]] = defaultdict(dict)
        folded_loops: dict[str, float] = defaultdict(float)
        folded_members: dict[str, list[str]] = defaultdict(list)
        for node in adjacent:
            mine = home[node]
            folded_members[mine].extend(members[node])
            folded_loops[mine] += loops[node]
            for other, weight in adjacent[node].items():
                theirs = home[other]
                if mine == theirs:
                    folded_loops[mine] += weight / 2  # seen from both ends
                else:
                    folded[mine][theirs] = folded[mine].get(theirs, 0.0) + weight
        for community in folded_members:
            folded.setdefault(community, {})
        adjacent = {node: folded[node] for node in folded_members}
        loops = dict(folded_loops)
        members = dict(folded_members)

    return [sorted(held_members) for held_members in members.values()]


def settle(
    nodes: Iterable[str],
    weights: Mapping[Pair, float],
    start: Iterable[Iterable[str]],
    known: Iterable[str] = (),
    resolution: float = RESOLUTION,
) -> list[list[str]]:
    """Communities settled from a partition already held, moving as few nodes as it can.

    `docs/archive/plan-map.md` §4, M2b. `start` is the partition held (a name in it that is not a
    node is ignored) and `known` every node it placed, together or alone; any other node
    is new. The rule has a dead band, so that a node near a border does not cross it
    whenever a line's weight wavers:

    - **A node stays** in its community while a line joins it to another member there.
    - **A node no line holds** -- new, alone, or cut off from its community -- is taken
      out and placed afresh by Louvain's local move: into the community its lines gain
      most in (the lines into it, less the community's degree times the node's share of
      all degree), or alone when none gains. A *known* node joins a community of two or
      more only through two or more of its lines, so one line that comes and goes moves
      nobody; a new node, which has no place to keep, needs one.

    Nodes are visited in sorted order, pass after pass until one moves nothing (each move
    raises modularity, so the passes end), and the communities are then cut into their
    connected parts. No community is ever merged whole and no random number is drawn.
    """
    adjacent: dict[str, dict[str, float]] = {node: {} for node in sorted(set(nodes))}
    for (first, second), weight in canonical(weights).items():
        if first in adjacent and second in adjacent:
            adjacent[first][second] = weight
            adjacent[second][first] = weight
    order = list(adjacent)
    seen = set(known)

    # A community is a number: a held one its place in `start`, a lone node one past them.
    home: dict[str, int] = {}
    lone = 0
    for part in start:
        for node in sorted(set(part)):
            if node in adjacent:
                if node in home:
                    raise ValueError(f"{node!r} is in two starting communities")
                home[node] = lone
        lone += 1
    for node in order:
        if node not in home:
            home[node] = lone
            lone += 1
    size: dict[int, int] = defaultdict(int)
    for node in order:
        size[home[node]] += 1

    degree = {node: sum(adjacent[node].values()) for node in order}
    total = sum(degree.values()) / 2
    held: dict[int, float] = defaultdict(float)
    for node in order:
        held[home[node]] += degree[node]

    for _ in range(len(order) + 1):
        moved = False
        for node in order:
            here = home[node]
            if any(home[other] == here for other in adjacent[node]):
                continue
            links: dict[int, float] = defaultdict(float)
            count: dict[int, int] = defaultdict(int)
            for other, weight in adjacent[node].items():
                links[home[other]] += weight
                count[home[other]] += 1
            held[here] -= degree[node]
            size[here] -= 1
            if size[here] == 0:
                best, gain = here, 0.0
            else:
                best, gain = lone, 0.0
            share = resolution * degree[node] / (2 * total) if total > 0 else 0.0
            for community in sorted(links):
                if node in seen and size[community] >= 2 and count[community] < 2:
                    continue
                value = links[community] - held[community] * share
                if value > gain + _EPSILON:
                    best, gain = community, value
            if best == lone:
                lone += 1
            home[node] = best
            held[best] += degree[node]
            size[best] += 1
            moved = moved or best != here
        if not moved:
            break

    parts: dict[int, list[str]] = defaultdict(list)
    for node in order:
        parts[home[node]].append(node)
    return split_disconnected(parts.values(), canonical(weights))


def split_disconnected(
    parts: Iterable[list[str]], weights: Mapping[Pair, float]
) -> list[list[str]]:
    """Every community cut into its connected parts, so no community is two islands."""
    joined: dict[str, set[str]] = defaultdict(set)
    for first, second in weights:
        joined[first].add(second)
        joined[second].add(first)
    found: list[list[str]] = []
    for part in parts:
        inside = set(part)
        seen: set[str] = set()
        for start in sorted(part):
            if start in seen:
                continue
            piece = {start}
            queue = [start]
            while queue:
                node = queue.pop()
                for other in joined[node] & inside:
                    if other not in piece:
                        piece.add(other)
                        queue.append(other)
            seen |= piece
            found.append(sorted(piece))
    return found


def modularity(
    parts: Iterable[Iterable[str]],
    weights: Mapping[Pair, float],
    resolution: float = RESOLUTION,
) -> float:
    """Newman's weighted modularity of this partition, at this resolution."""
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    where = {node: number for number, part in enumerate(parts) for node in part}
    degree: dict[str, float] = defaultdict(float)
    inside: dict[int, float] = defaultdict(float)
    for (first, second), weight in weights.items():
        degree[first] += weight
        degree[second] += weight
        if where[first] == where[second]:
            inside[where[first]] += weight
    held: dict[int, float] = defaultdict(float)
    for node in sorted(degree):
        held[where[node]] += degree[node]
    return sum(
        inside[number] / total - resolution * (held[number] / (2 * total)) ** 2
        for number in sorted(held)
    )


def _shape(parts: Iterable[list[str]]) -> tuple[tuple[str, ...], ...]:
    """A partition in one canonical spelling, so two runs that agree compare equal."""
    return tuple(sorted(tuple(sorted(part)) for part in parts))
