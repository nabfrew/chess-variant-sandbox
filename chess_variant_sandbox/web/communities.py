"""Communities in a weighted graph: Louvain modularity, made deterministic.

The map's similarity lines join games that differ by no more than one rule's weight, and
the families those lines form are what the map is drawn around. This finds them. It
knows nothing about games: it takes node names and a weight per pair and returns a
partition, which `web/graphing.py` names and lays out.

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
