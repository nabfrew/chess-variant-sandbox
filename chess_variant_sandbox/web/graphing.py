"""The library as a map: which games are alike, the communities they form, where each sits.

`docs/archive/plan-map.md` §4 (M4) puts the new map on the page. The relation is
`spec/likeness.py`'s (what two games are made of, each part weighed by how rare it is in
the library), the communities and each game's nearest are `web/families.py`'s, and the
picture is `web/mapping.py`'s, remembered from one build to the next in
`runs/graph/map.json`. This module joins them to what a reader is shown -- names, scores,
what differs, lineage -- and hands the page a payload it can draw without computing
anything.

What it decides itself:

1. **Twins merge.** Two specs whose compiled tables are equal, the Zobrist draw apart, are
   one game under two spellings and become one node carrying both names; the map is built
   over the nodes. The library has no such pair today (`wolves` and `wolves-immune` were
   the last, retired at Q876); `tests/test_graphing.py` keeps the merge measured on made
   pairs and on that pair rebuilt.
2. **A line is a pair of mutual neighbours**: each is among the other's `families.MUTUAL`
   nearest. Its width is their likeness, its words are the rare things they share, and it
   carries `relate`'s reading of what differs between them (the rules apart, and each
   difference within `NEAR`) and what the two scores make of it.
3. **Each game's nearest** (`like`) are the games like this one: its few nearest by
   likeness, each with what they share, whether or not a line joins them.
4. **The picture is fitted to the page.** `mapping` places games in t-SNE's own scale;
   here the picture is scaled so the median distance from a game to its nearest drawn
   neighbour is `RING` layout units, and any two dots nearer than `MIN_SEPARATION` are
   pushed apart, so no dot is drawn on another and names have room.
5. **A community drawn in two places is said to be.** t-SNE keeps neighbourhoods, so a
   community whose halves share little but one word can be drawn as two knots. Each
   community carries its drawn `parts` -- its members joined while each is within
   `PART_REACH` of another -- so an outline is drawn round each part, not round the gap.
6. **Lineage is a second kind of line**, an arrow from parent to child at any distance,
   from what the spec, the design run or the site's own column recorded.

**Nothing here re-reads the DSL.** What a rule *means* is `engine/compile.py`'s, what two
specs differ in is `spec/relate.py`'s, how alike they are is `spec/likeness.py`'s, and what
a game scored is the library's, read from the card it already built. No piece name and no
ruleset name appears in it.

**The cost.** Normal forms are about a second for the library and memoised by `relate`; a
cold map (no memory) about five seconds, a kept one a fraction of one (M3's readings in
the plan). The map is built once per library and memory in a process (`_MAPS`), and pairs
read with `relate` and every spec's compile digest survive a restart in
`runs/graph/pairs.json`, so a warm request computes nothing. There is no per-deploy draw:
the operator-sentence sweep this module once ran on a deploy retired with the rule-step
lines it labelled (M4).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chess_variant_sandbox.spec import relate
from chess_variant_sandbox.spec.likeness import LIKENESS_VERSION, Likeness, likeness_of_forms
from chess_variant_sandbox.web import families, mapping

if TYPE_CHECKING:  # a Game is the library's; importing it at runtime would be a cycle
    from chess_variant_sandbox.web.library import Game

# Bumped when a cached reading moves -- the twin key or what a pair record holds -- and
# when the payload changes shape, so a reader cannot take an older shape for this one. A
# cache file stamped with anything else is dropped whole rather than read. `g11` to `g14`
# were the similarity lines, their communities and the layout by community
# (`docs/archive/plan-graph-map.md`). `g15` (M4 of `docs/archive/plan-map.md`): the lines are the
# mutual neighbours by likeness (`neighbours`), the communities are `families`', the
# picture is `mapping`'s; `links`, `similar` and `levels` left the payload, and a pair
# record keeps `relate`'s reading and nothing else.
GRAPH_VERSION = "g15"

# How many rules apart two games may be and still have each difference spelt out on their
# line. Past three, a list of what differs is a ranking of how unlike two games are rather
# than a reading of it, and the line says only how many rules apart they are.
NEAR = 3

# How many of the rare things two games share a line or a nearest game names.
SHARES = 3

# Where the pair cache lives, under the runs directory the caller names. The map's memory
# is beside it (`families.MAP_FILE`).
CACHE_NAME = "graph"
CACHE_FILE = "pairs.json"

# ---------------------------------------------------------------------- the layout
#
# Plain numbers in layout units, one unit being a node's own radius. The page scales them
# into model pixels; nothing here knows a pixel.

NODE_RADIUS = 1.0
# The median distance from a game to its nearest drawn neighbour, once the picture is
# fitted: the picture's natural spacing, a dot and a half's width of air. The payload calls
# it `ring`, the name the first radial layout gave the same step. Wider, and the whole
# library at the width of a page draws its dots a few pixels across (M4's look).
RING = 3.0
# The least distance between two drawn games: the one diameter the page promises, and a
# quarter of one more.
MIN_SEPARATION = 2.5
# How near two games of one community must be drawn to be outlined as one part of it:
# eight of the picture's median nearest gaps. Chosen with M4's reading in view (the plan):
# on the library of 2026-10-20 it parts the two communities t-SNE drew in two places
# (a step of 79 units between its places, and one of 36) and no other, whose widest step
# is 18.
PART_REACH = 8 * RING


@dataclass(frozen=True)
class Node:
    """One game on the map, which may be two spec files.

    `hash` is the representative's mechanics hash and what the page fetches by; `hashes`
    and `names` carry every spelling merged into it.
    """

    hash: str
    name: str
    names: tuple[str, ...]
    hashes: tuple[str, ...]
    source: str
    board: str
    # How many lines reach it: the games it and they are each among the other's nearest.
    degree: int
    # In no community of two or more: nothing in the library is close enough to it.
    unusual: bool
    score: dict[str, Any] | None
    staleness: dict[str, Any] | None
    # `library.exhibit`: the sentence saying this game is kept on purpose for what it
    # shows, or `None`. The panel prints it beside the score, as the card does.
    exhibit: str | None = None
    x: float = 0.0
    y: float = 0.0
    # The id of the community this game is in, or `None` for an unusual game.
    community: str | None = None
    # The games like this one: its nearest by likeness, nearest first, each
    # `{hash, likeness, shares, line}`.
    like: tuple[dict[str, Any], ...] = ()
    # This dot's place between each size key's two ends, 0 to 1, or `None` for *no value
    # under this key*. Filled once the whole library is known, because both ends are the
    # library's own. The page maps the number to a radius and decides nothing else.
    size: dict[str, float | None] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "name": self.name,
            "names": list(self.names),
            "hashes": list(self.hashes),
            "source": self.source,
            "board": self.board,
            "degree": self.degree,
            "unusual": self.unusual,
            "size": dict(self.size),
            "score": self.score,
            "staleness": self.staleness,
            "exhibit": self.exhibit,
            "community": self.community,
            "like": [dict(one, shares=list(one["shares"])) for one in self.like],
            "x": round(self.x, 3),
            "y": round(self.y, 3),
        }


@dataclass
class Line:
    """One line on the map: two games each among the other's nearest.

    `likeness` is how alike the two are, 0 to 1, and is what the line's width draws;
    `shares` the rarest things both have, in words. `units` is how many rules apart they
    are, `relate`'s count, and `differences` each one where there are at most `NEAR`.
    """

    source: str
    target: str
    likeness: float
    shares: tuple[str, ...]
    units: int
    kind: str
    kinds: tuple[str, ...]
    sentence: str
    differences: tuple[dict[str, Any], ...]
    unit: str
    gap: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "likeness": self.likeness,
            "shares": list(self.shares),
            "units": self.units,
            "kind": self.kind,
            "kinds": list(self.kinds),
            "unit": self.unit,
            "sentence": self.sentence,
            "differences": [dict(one) for one in self.differences],
            "gap": self.gap,
        }


@dataclass
class Graph:
    """The payload, and what building it cost.

    `computed` and `compiled` must be zero on a second call: the pairs this build had to
    read with `relate` and the specs it had to compile. Kept on the result rather than
    printed, so a test can assert the cache works instead of timing it.
    """

    nodes: list[Node]
    neighbours: list[Line]
    lineage: list[dict[str, Any]]
    kinds: list[dict[str, Any]]
    bounds: dict[str, float]
    sizes: list[dict[str, Any]] = field(default_factory=list)
    # The further readings of these same nodes a caller can fetch, by name. Empty from a
    # checkout, which has no store to read one out of, and a constant per server rather
    # than per viewer -- so this payload stays one answer for everybody.
    overlays: list[str] = field(default_factory=list)
    scale: dict[str, Any] | None = None
    components: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    communities: list[dict[str, Any]] = field(default_factory=list)
    # The games in no community, the furthest from its nearest game first.
    unusual: list[dict[str, Any]] = field(default_factory=list)
    computed: int = 0
    compiled: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": GRAPH_VERSION,
            "relate": relate.RELATE_VERSION,
            "likeness": LIKENESS_VERSION,
            "map": families.MAP_VERSION,
            "layout": mapping.LAYOUT_VERSION,
            # How many nearest a game offers when a line is drawn: the words a page says
            # about a line name this rather than a number of their own.
            "mutual": families.MUTUAL,
            "nodes": [node.as_dict() for node in self.nodes],
            "neighbours": [line.as_dict() for line in self.neighbours],
            "lineage": [dict(arrow) for arrow in self.lineage],
            "kinds": [dict(kind) for kind in self.kinds],
            "bounds": dict(self.bounds),
            "units": {"nodeRadius": NODE_RADIUS, "ring": RING},
            "sizes": [dict(size) for size in self.sizes],
            "overlays": list(self.overlays),
            "scale": None if self.scale is None else dict(self.scale),
            "components": dict(self.components),
            "counts": dict(self.counts),
            "communities": [
                {
                    **community,
                    "members": list(community["members"]),
                    "names": list(community["names"]),
                    "parts": [list(part) for part in community["parts"]],
                }
                for community in self.communities
            ],
            "unusual": [dict(one) for one in self.unusual],
        }


def graph(
    games: Iterable[Game],
    runs_dir: Path | str = "runs",
    parents: Mapping[str, str] | None = None,
    overlays: Iterable[str] = (),
) -> dict[str, Any]:
    """The whole map as JSON, for `GET /api/graph`."""
    return build(games, runs_dir, parents, overlays).as_dict()


def build(
    games: Iterable[Game],
    runs_dir: Path | str = "runs",
    parents: Mapping[str, str] | None = None,
    overlays: Iterable[str] = (),
) -> Graph:
    """The map over these games.

    `parents` is the third recorded source of lineage -- the site's own `games.parent`
    column, child hash to parent hash. The spec's `provenance.parents` and a design run's
    `lineage.jsonl` arrive on the `Game` already, and a caller with no store passes
    nothing.

    `overlays` names the further readings of these nodes the caller can answer for, and
    is what tells the page whether to ask for one. It is the caller's constant, not a
    property of the library, because what the page may fetch is what the server serves.

    The first build with no `runs/graph/map.json` draws the picture cold and writes that
    file; every build after it settles from it, and one over an unchanged library keeps it.
    """
    listed, cache, groups, drawn = _prepared(games, Path(runs_dir))
    nodes = {group[0].hash: group for group in groups}
    by_hash = {game.hash: group[0].hash for group in groups for game in group}
    found = drawn.map

    lines = [_line(pair, weight, drawn, nodes, cache) for pair, weight in found.lines.items()]
    _fill_gaps(lines, nodes)
    degree: dict[str, int] = defaultdict(int)
    for line in lines:
        degree[line.source] += 1
        degree[line.target] += 1

    built = [
        replace(
            _node(nodes[digest], degree.get(digest, 0), drawn.placed[digest]),
            unusual=found.family_of.get(digest) is None,
            community=found.family_of.get(digest),
            like=_like(digest, drawn),
        )
        for digest in nodes
    ]
    built.sort(key=lambda node: (node.name.lower(), node.hash))
    communities = _communities(drawn, {node.hash: node for node in built})

    # Both ends of every size key are the library's own, so a dot's place between them is
    # known only once every dot is built.
    scale = _scale(built)
    sizes = _sizes(built, scale)
    built = [replace(node, size=_sized(node, scale)) for node in built]

    arrows = _lineage(listed, by_hash, parents or {}, nodes, cache)
    cache.write()
    return Graph(
        nodes=built,
        neighbours=lines,
        lineage=arrows,
        kinds=_kinds(lines),
        bounds=_bounds(built),
        sizes=sizes,
        overlays=list(overlays),
        scale=scale,
        components=_components(lines),
        communities=communities,
        unusual=[
            {"hash": digest, "distance": round(distance, 4)}
            for digest, distance in found.unusual
        ],
        counts={
            "nodes": len(built),
            "specs": len(listed),
            "twins": sum(1 for group in groups if len(group) > 1),
            "lines": len(lines),
            "lineage": len(arrows),
            "communities": len(communities),
            "unusual": len(found.unusual),
        },
        computed=cache.computed,
        compiled=cache.compiled,
    )


def _prepared(
    games: Iterable[Game], runs: Path
) -> tuple[list[Game], _Cache, list[list[Game]], _Drawn]:
    """The games in the map's order, the pair cache, the twin groups and the map over them.

    What `build` and `near` both stand on, so a game's own page and the map read one map.
    """
    listed = sorted(games, key=lambda game: (game.name.lower(), game.hash))
    cache = _Cache(runs / CACHE_NAME / CACHE_FILE)
    cache.read()
    groups = _twin_groups(listed, cache)
    return listed, cache, groups, _drawn([group[0] for group in groups], runs)


# ------------------------------------------------------------------- one meaning of near
#
# M5 of `docs/archive/plan-map.md`. Every list on the site or the local application that names
# games as near or like another reads this, and this reads the map: the same games, the
# same `spec/likeness.py` relation over them and the same order (`Likeness.nearest`), so a
# game's page and its panel on the map open on the same games in the same order, and its
# first `families.MUTUAL` are exactly the panel's *games like this one*. What two games
# differ in stays `relate`'s, said beside each row by the caller; it ranks nothing.


@dataclass(frozen=True)
class Nearness:
    """A game's nearest on the map, most alike first, and where the map puts it.

    `on_map` is whether the game is one of the map's dots (or a second spelling of one);
    a game the map does not draw -- a draft, a candidate -- is ranked against the map's
    games by the same relation with itself added (`near`). `community` is the map's family
    the game is in, `{id, name}`, or `None`: unusual on the map, or not on it. Each row of
    `like` is `{hash, likeness, shares, line}`, the shape of a map node's `like`.
    """

    on_map: bool
    community: dict[str, str] | None
    like: tuple[dict[str, Any], ...]


def near(
    game: Game, games: Iterable[Game], runs_dir: Path | str, few: int
) -> Nearness:
    """The `few` games on the map over `games` most like `game`, most alike first.

    `games` is the map's population as its caller draws it (`App.mapped`), so the page
    and the map cannot be handed two libraries. On the map, the order is the map's own
    relation's (`Likeness.nearest`, which `families` cuts at `MUTUAL` for the panel);
    a second spelling of one of its games reads as that game. Off it, the relation is
    taken over the map's games and this one -- the library's view of rarity, with this
    game counted in it, as it would be the day it joined -- and the map is unchanged.

    Costs nothing new on the map once it is warm (`_MAPS`); a game off it pays one
    relation over the library, about 55 ms at 92 games, kept per game for the process.
    """
    _, cache, groups, drawn = _prepared(games, Path(runs_dir))
    cache.write()
    by_hash = {member.hash: group[0].hash for group in groups for member in group}
    digest = by_hash.get(game.hash)
    if digest is not None:
        mine = drawn.at[digest]
        family = drawn.map.family_of.get(digest)
        named = {one.id: one.name for one in drawn.map.families}
        return Nearness(
            on_map=True,
            community=None if family is None else {"id": family, "name": named[family]},
            like=tuple(
                _row(drawn, drawn.relation, drawn.forms, mine, other)
                for other in drawn.relation.nearest(mine, few)
            ),
        )
    relation, forms = _beside(drawn, game)
    mine = len(drawn.ids)
    return Nearness(
        on_map=False,
        community=None,
        like=tuple(
            _row(drawn, relation, forms, mine, other, said=False)
            for other in relation.nearest(mine, few)
        ),
    )


# How many games off the map keep their relation in one process: a draft's page is read
# again as it is edited, and the library it is ranked against changes rarely.
_BESIDE_KEPT = 64


def _beside(drawn: _Drawn, game: Game) -> tuple[Likeness, tuple[relate.NormalForm, ...]]:
    """The map's relation with one more game added at the end, kept per game."""
    form = relate.normal_form(game.spec)
    held = drawn.beside.get(form.key)
    if held is None:
        forms = (*drawn.forms, form)
        held = (likeness_of_forms(forms), forms)
        if len(drawn.beside) >= _BESIDE_KEPT:
            drawn.beside.pop(next(iter(drawn.beside)))
        drawn.beside[form.key] = held
    return held


def _row(
    drawn: _Drawn,
    relation: Likeness,
    forms: Sequence[relate.NormalForm],
    mine: int,
    other: int,
    said: bool = True,
) -> dict[str, Any]:
    """One game like another: its hash, how alike, what both have, whether a line joins.

    `mine` and `other` index `relation` and `forms`; past the map's own games (a game off
    the map, added at the end) nothing is joined by a line, and what the two share is said
    afresh rather than kept with the map's (`said`).
    """
    digest = drawn.ids[other]
    if said:
        shares = _shared(drawn, (drawn.ids[mine], digest))
    else:
        shares = tuple(families.shared_words(relation, [mine, other], forms, SHARES))
    pair = _key(drawn.ids[mine], digest) if mine < len(drawn.ids) else None
    return {
        "hash": digest,
        "likeness": round(relation.similarity(mine, other), 3),
        "shares": shares,
        # Whether a line joins the two: each is among the other's nearest.
        "line": pair is not None and pair in drawn.map.lines,
    }


# ------------------------------------------------------------------------ twins (S7)


def _twin_groups(games: list[Game], cache: _Cache) -> list[list[Game]]:
    """The games grouped by what they compile to, the Zobrist draw set aside.

    Two specs that produce the same tables are the same game however differently they are
    written, and no comparison of two written descriptions can see it: `wolves` and
    `wolves-immune` (retired at Q876, the library's last such pair) declared one rule two
    ways and sat six units apart under a structural diff. The compiler is the only reading
    that settles it, and this asks it rather than guessing.

    The Zobrist arrays are dropped because they are a draw from a seeded generator, not a
    rule -- two spellings of one game hash their positions differently and play the same
    game. The rest of `bridge.canonical_tables` is what the core walks.

    A group keeps its members in library order and the first is the node's representative,
    so the same corpus always yields the same node.
    """
    groups: dict[str, list[Game]] = defaultdict(list)
    for game in games:
        groups[cache.twin(game)].append(game)
    return sorted(groups.values(), key=lambda group: (group[0].name.lower(), group[0].hash))


# ------------------------------------------------------------------------- the map


@dataclass(frozen=True)
class _Drawn:
    """One library's map, with what the payload reads off it.

    `map` is `mapping.build`'s, `placed` its picture fitted to layout units, and `ids`,
    `forms` and `relation` the games in id order, their normal forms and the likeness over
    them, so a line or a nearest game is said from the same relation the map was built on.
    """

    map: families.FamilyMap
    placed: dict[str, tuple[float, float]]
    ids: tuple[str, ...]
    forms: tuple[relate.NormalForm, ...]
    relation: Likeness
    at: dict[str, int]
    said: dict[tuple[str, ...], tuple[str, ...]] = field(default_factory=dict)
    # A game off the map, ranked against it (`near`): the relation with it added, by its
    # normal-form key.
    beside: dict[str, tuple[Likeness, tuple[relate.NormalForm, ...]]] = field(
        default_factory=dict
    )


# The last map built over each memory file in this process: the library it was built over,
# the memory file as it stood after the build, and the map. A request over the same library
# and the same memory is answered from here; anything else builds again.
_MAPS: dict[str, tuple[str, tuple[int, int] | None, _Drawn]] = {}

# Every draw this process has made, by what a draw is a function of: the library
# (`_library_key`) and the memory it settled from (the file's digest, `None` for a cold
# draw). Exact, since a draw is deterministic in both, and shared by every runs directory,
# so a second server over the same games -- a test's, a scratch tree's -- draws nothing
# again. A few are kept: one per library and memory a process meets.
_DRAWS: dict[tuple[str, str | None], families.FamilyMap] = {}
_DRAWS_KEPT = 8

# Every compile digest this process has worked out, by mechanics hash (`_Cache.twin`): exact
# for the reason the file's is, and shared by every runs directory for `_DRAWS`' reason.
_TWINS: dict[str, str] = {}


def _drawn(representatives: Sequence[Game], runs: Path) -> _Drawn:
    """The map over these games, settled from `runs/graph/map.json` and written back to it.

    The file is the map's memory: the first build with none draws cold, every other
    settles from it, and one over the library it remembers keeps every place. It is written
    only when it changed, and a runs directory that cannot be written leaves the map drawn
    and unremembered rather than failing the request.
    """
    by = {game.hash: game for game in representatives}
    ids = tuple(sorted(by))
    memory = runs / families.MAP_FILE
    key = _library_key([by[digest] for digest in ids])
    where = str(memory.resolve())
    held = _MAPS.get(where)
    if held is not None and held[0] == key and held[1] == _stamp(memory):
        return held[2]
    forms = tuple(relate.normal_form(by[digest].spec) for digest in ids)
    previous = families.read_memory(memory)
    basis = (key, None if previous is None else _content(memory))
    found = _DRAWS.get(basis)
    if found is None:
        found = mapping.build(
            list(ids), list(forms), [by[digest].name for digest in ids], previous
        )
        if len(_DRAWS) >= _DRAWS_KEPT:
            _DRAWS.pop(next(iter(_DRAWS)))
        _DRAWS[basis] = found
    with contextlib.suppress(OSError):
        families.write_memory(memory, found)
    drawn = _Drawn(
        map=found,
        placed=_fitted(found.positions),
        ids=ids,
        forms=forms,
        relation=likeness_of_forms(forms),
        at={digest: index for index, digest in enumerate(ids)},
        # What a pair shares is a function of the library alone, so a map rebuilt over the
        # same library -- its memory moved -- keeps the words already said.
        said=dict(held[2].said) if held is not None and held[0] == key else {},
    )
    _MAPS[where] = (key, _stamp(memory), drawn)
    return drawn


def _library_key(games: Sequence[Game]) -> str:
    """A digest of what a map is built from: each game's hash, name and spec, in id order.

    A name is in it because a community named *near* its centre says the centre's name; a
    spec because a piece's name, outside the mechanics hash, can name a community.
    """
    blob = json.dumps(
        [[game.hash, game.name, game.spec] for game in games],
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _content(path: Path) -> str | None:
    """A digest of the memory file's bytes, or `None` where it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _stamp(path: Path) -> tuple[int, int] | None:
    """The memory file as it stands -- when it was written, and how long it is -- or none."""
    try:
        held = path.stat()
    except OSError:
        return None
    return (held.st_mtime_ns, held.st_size)


def _fitted(positions: Mapping[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """The picture in layout units: its median nearest gap `RING`, no two dots too near.

    Scaling moves no game relative to another; the push apart moves only the few pairs
    t-SNE drew nearer than `MIN_SEPARATION`, a fraction of a unit each, in id order.
    """
    ids = sorted(positions)
    if len(ids) < 2:
        return {digest: (0.0, 0.0) for digest in ids}
    gaps = [
        min(math.dist(positions[digest], positions[other]) for other in ids if other != digest)
        for digest in ids
    ]
    middle = statistics.median(gaps)
    scale = RING / middle if middle > 0 else 1.0
    spot = {
        digest: [positions[digest][0] * scale, positions[digest][1] * scale] for digest in ids
    }
    _separated(ids, spot, MIN_SEPARATION)
    return {digest: (spot[digest][0], spot[digest][1]) for digest in ids}


def _separated(names: list[str], spot: dict[str, list[float]], least: float) -> None:
    """Push every two of these apart until none is nearer than `least`, in id order."""
    for _ in range(500):
        moved = False
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                dx, dy, apart = _towards(spot[first], spot[second])
                if apart >= least - 1e-9:
                    continue
                half = (least - apart) / 2 + 1e-6
                spot[first][0] += dx * half
                spot[first][1] += dy * half
                spot[second][0] -= dx * half
                spot[second][1] -= dy * half
                moved = True
        if not moved:
            return


def _towards(one: list[float], other: list[float]) -> tuple[float, float, float]:
    """The unit step from `other` to `one`, and the distance, never zero."""
    dx, dy = one[0] - other[0], one[1] - other[1]
    apart = math.hypot(dx, dy)
    if apart < 1e-9:
        return 1.0, 0.0, 1e-9
    return dx / apart, dy / apart, apart


def _shared(drawn: _Drawn, games: Sequence[str]) -> tuple[str, ...]:
    """The rarest things these games all have, in words, at most `SHARES`; worked out once."""
    key = tuple(sorted(games))
    held = drawn.said.get(key)
    if held is None:
        members = [drawn.at[digest] for digest in key]
        held = tuple(families.shared_words(drawn.relation, members, drawn.forms, SHARES))
        drawn.said[key] = held
    return held


def _like(digest: str, drawn: _Drawn) -> tuple[dict[str, Any], ...]:
    """This game's nearest by likeness, nearest first, with what each pair shares."""
    mine = drawn.at[digest]
    return tuple(
        _row(drawn, drawn.relation, drawn.forms, mine, drawn.at[other])
        for other in drawn.map.neighbours.get(digest, ())
    )


def _line(
    pair: tuple[str, str],
    weight: float,
    drawn: _Drawn,
    nodes: Mapping[str, list[Game]],
    cache: _Cache,
) -> Line:
    reading = _reading(nodes[pair[0]], nodes[pair[1]], cache)
    differences = tuple(reading.get("differences") or [])
    first = differences[0] if differences else {}
    return Line(
        source=pair[0],
        target=pair[1],
        likeness=round(weight, 3),
        shares=_shared(drawn, pair),
        units=int(reading["units"]),
        # The kind of the first difference, which is the whole kind on a one-rule line and
        # the one a mixed line is filed under. Every kind on it travels too.
        kind=str(first.get("kind", "")),
        kinds=tuple(sorted({str(one["kind"]) for one in differences})),
        unit=str(first.get("unit", "")),
        sentence="; ".join(str(one["sentence"]) for one in differences),
        differences=differences,
    )


def _reading(
    first: Sequence[Game], second: Sequence[Game], cache: _Cache
) -> dict[str, Any]:
    """What two nodes differ in, as `relate` reads it: the shorter of their spellings'.

    A merged node is one game, so a rule declared on five pieces and the same rule
    declared once are one rule, and the node says so wherever it is compared.
    """
    records = [cache.pair(one, other) for one in first for other in second]
    return min(records, key=lambda record: int(record["units"]))


def _key(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


def _ordered(value: Any) -> Any:
    """The same value with every mapping's keys in sorted order, all the way down.

    The cache file is written with sorted keys, so a record read back from it is ordered
    and a record just computed is in whatever order it was built. Parts of a record reach
    the payload as they are, and the payload is compared byte for byte, so the two
    readings have to agree on order as well as on content.
    """
    if isinstance(value, dict):
        return {key: _ordered(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_ordered(one) for one in value]
    return value


# -------------------------------------------------------------------- the communities


def _communities(drawn: _Drawn, at: Mapping[str, Node]) -> list[dict[str, Any]]:
    """Each community with its name, members, centre, drawn parts and folded place.

    The name and the centre are `families`'; `words` is what marks the community out, in
    words, or `None` where nothing does and it is named *near* its centre. `parts` are its
    members as drawn (`_parts`), the part holding the centre first, and `x, y` -- where its
    folded dot opens -- is the middle of that part, so a community drawn in two places
    folds where its most typical game is.
    """
    found: list[dict[str, Any]] = []
    for family in drawn.map.families:
        members = list(family.members)
        words = (
            None if family.word is None
            else families.words_for(
                family.word, [drawn.forms[drawn.at[digest]] for digest in members]
            )
        )
        parts = _parts(members, drawn.placed)
        parts.sort(key=lambda part: (family.centre not in part, -len(part), part))
        home = parts[0]
        found.append({
            "id": family.id,
            "name": family.name,
            "words": words,
            "members": members,
            "names": [at[digest].name for digest in members],
            "centre": family.centre,
            "size": len(members),
            "parts": parts,
            "x": round(sum(at[digest].x for digest in home) / len(home), 3),
            "y": round(sum(at[digest].y for digest in home) / len(home), 3),
        })
    return found


def _parts(members: Sequence[str], placed: Mapping[str, tuple[float, float]]) -> list[list[str]]:
    """These games as drawn: joined while one is within `PART_REACH` of another.

    Single linkage over the fitted picture, in id order: a community t-SNE drew as one knot
    is one part, and one it drew as two knots far apart is two.
    """
    ordered = sorted(members)
    home = {digest: digest for digest in ordered}

    def root(digest: str) -> str:
        while home[digest] != digest:
            home[digest] = home[home[digest]]
            digest = home[digest]
        return digest

    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            if math.dist(placed[first], placed[second]) <= PART_REACH:
                one, other = root(first), root(second)
                if one != other:
                    home[max(one, other)] = min(one, other)
    grouped: dict[str, list[str]] = defaultdict(list)
    for digest in ordered:
        grouped[root(digest)].append(digest)
    return [grouped[name] for name in sorted(grouped)]


# --------------------------------------------------------------------- the scores


def _fill_gaps(lines: Sequence[Line], nodes: Mapping[str, list[Game]]) -> None:
    """Both indexes and the distance between them, on every line where both can be read.

    The numbers are the library's own, off the card each game already carries. A game
    whose measurement is stale shows no score there and gets none here, and two scores
    under different weights versions never share a leaderboard, so a pair straddling a
    version has no gap however fresh both are.

    *Clear* is the gap against two combined standard deviations, the within-run spread
    `canonical.json` stores. Measured over the committed corpus at Q347, that spread is
    the same size as the spread of a game's four runs about their own mean -- median
    0.0114 against 0.0104, larger on 29 of 68 games -- and taking the larger of the two
    per game moves 2 of 125 links across the line. So there is nothing larger to take,
    and `library.neighbours` and this read one spread rather than two.
    """
    for line in lines:
        mine = _card(nodes[line.source])
        theirs = _card(nodes[line.target])
        if mine is None or theirs is None:
            continue
        if mine.get("weights") != theirs.get("weights"):
            continue
        here, there = mine.get("index"), theirs.get("index")
        if here is None or there is None:
            continue
        # Both ends are printed, so both run counts travel, as on `library._gap`.
        seeds, other_seeds = list(mine.get("seeds") or []), list(theirs.get("seeds") or [])
        if not seeds or not other_seeds:
            continue
        spread = (float(mine.get("sd") or 0.0) ** 2 + float(theirs.get("sd") or 0.0) ** 2) ** 0.5
        gap = float(there) - float(here)
        line.gap = {
            "index": float(here),
            "otherIndex": float(there),
            "seeds": seeds,
            "otherSeeds": other_seeds,
            "gap": round(gap, 4),
            "spread": round(spread, 4),
            "clear": abs(gap) > 2 * spread,
            "weights": mine.get("weights"),
            "moved": _moved(nodes[line.source], nodes[line.target]),
        }


# How many of the index's parts a line names. Three is what fits under a sentence in the
# panel, and past the third the changes are the size of the noise on any one of them. How
# many parts there are is the two games' shared weights version's business, and a line
# reads them off the stored scores rather than from a count.
MOVERS = 3


def _moved(source: list[Game], target: list[Game]) -> list[dict[str, Any]]:
    """Which parts of the score differ most between two games, biggest first.

    The index is its version's measurements combined, and *this rule cost 0.09* is a headline
    with no working under it. The working is already stored: `index-<weights>.json` holds
    each component's 0-1 score and `scoring.Weights.combine`'s own contribution for it,
    which is that component's weighted share of the total. Ranking by how much the
    *contribution* moved is therefore the scoring module's own decomposition rather than
    a second one invented here, and what the page prints is the component's own score at
    each end.

    A component only one of the two games carries is left out: an optional measurement
    its corpus could not supply has no value to differ from.
    """
    mine, theirs = _measurements(source), _measurements(target)
    shared = set(mine) & set(theirs)
    if not shared:
        return []
    ranked = sorted(
        shared,
        key=lambda name: (-abs(theirs[name][1] - mine[name][1]), name),
    )
    return [
        {
            "key": name,
            "index": round(mine[name][0], 4),
            "otherIndex": round(theirs[name][0], 4),
        }
        for name in ranked[:MOVERS]
        if theirs[name][0] != mine[name][0]
    ]


def _measurements(group: list[Game]) -> dict[str, tuple[float, float]]:
    """This node's components, each as its own score and its share of the index.

    Both are the scoring pass's, read off the card the library already built. The node's
    payload does not carry them -- a dozen numbers on each of seventy games is most of a
    payload nothing draws -- so they are read here, where a line needs the pair.
    """
    from chess_variant_sandbox.web.library import current_only

    score = current_only(group[0].score) or {}
    scores = score.get("scores") or {}
    shares = score.get("contributions") or {}
    return {
        str(name): (float(value), float(shares.get(name, 0.0)))
        for name, value in scores.items()
        if isinstance(value, (int, float))
    }


def _components(lines: Sequence[Line]) -> dict[str, str]:
    """A readable name for every component a line names, from the glossary.

    `metrics/glossary.py` is where a measurement is named in English, and the map prints
    those words for the same reason the legend prints `KIND_WORDS`: `skill_gradient` is
    an identifier, and a page that spelled it out its own way would be a second opinion
    about what it measures.
    """
    from chess_variant_sandbox.metrics.glossary import title_of

    named = {
        str(part["key"])
        for line in lines
        for part in ((line.gap or {}).get("moved") or [])
    }
    return {name: title_of(name) for name in sorted(named)}


def _scale(nodes: list[Node]) -> dict[str, Any] | None:
    """The range the map's score fill runs over, and how many games are on it.

    A fill needs two ends, and 0 to 1 is not them: sixty-eight of the seventy scored
    games sit between 0.17 and 0.82 with the median at 0.75, so a fill over the whole
    unit would paint the library one colour. The ends are therefore the library's own,
    and the legend prints both numbers so the colour is read against them rather than
    against an imagined scale.

    **One weights version.** Two scores taken under different weights never share a
    leaderboard, so they must not share a ramp either. The scale is the version most of
    the scored games are under, and a game under any other has no place on it -- the page
    draws it hollow and its own card still says what it scored.
    """
    versions: dict[str, list[float]] = defaultdict(list)
    for node in nodes:
        score = node.score or {}
        index = score.get("index")
        if index is None or not score.get("weights"):
            continue
        versions[str(score["weights"])].append(float(index))
    if not versions:
        return None
    version = max(sorted(versions), key=lambda name: len(versions[name]))
    found = versions[version]
    return {
        "low": round(min(found), 4),
        "high": round(max(found), 4),
        "weights": version,
        "scored": len(found),
        "unscored": len(nodes) - len(found),
    }


# --------------------------------------------------------------------- the sizes
#
# What may drive how big a dot is drawn. More than one thing can, so the payload carries
# a menu rather than a rule, and every dot carries its place between the chosen key's two
# ends as a number from 0 to 1. The page maps that number to a radius and works out
# nothing else -- which is the same division the score ramp already runs under.


def _sizes(nodes: list[Node], scale: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The size menu, in menu order, each key with its rule and its two ends.

    `low` and `high` are the real values a dot's smallest and largest are drawn at, and
    the legend prints them in `lowWords` and `highWords` -- the server's own words,
    because a page that formatted them would be a second opinion about what the end is.

    `score` is offered only where there is a ramp to read it against. A checkout with no
    `runs/` has no scale, every dot is hollow, and a menu entry that sized every game the
    same would be a control that does nothing.
    """
    counts = [node.degree for node in nodes]
    low = min(counts) if counts else 0
    full = families.MUTUAL
    found: list[dict[str, Any]] = [{
        "key": "neighbours",
        "label": "how many lines",
        "words": (
            "How many games this one has a line to: each of the two is among the"
            f" other's {full} nearest."
        ),
        "low": low,
        "high": full,
        "lowWords": "no lines" if low == 0 else "1 line" if low == 1 else f"{low} lines",
        "highWords": f"{full} lines",
    }]
    if scale:
        found.append({
            "key": "score",
            "label": "how it scored",
            "words": (
                "How the game scored, against the library's own lowest and highest. A"
                " game with no current score, or one scored under older scoring rules,"
                " is drawn smallest."
            ),
            "low": scale["low"],
            "high": scale["high"],
            "lowWords": f"{scale['low']:.3f}",
            "highWords": f"{scale['high']:.3f}",
        })
    return found


def _sized(node: Node, scale: dict[str, Any] | None) -> dict[str, float | None]:
    """One dot's place between each key's two ends.

    A number from 0 to 1, or `None` for *no value under this key*, which the page draws
    at the smallest size and the legend explains. `neighbours` always has a value, because
    a count of nothing is nought rather than nothing.
    """
    return {
        "neighbours": round(min(1.0, node.degree / families.MUTUAL), 4),
        "score": _score_size(node, scale),
    }


def _score_size(node: Node, scale: dict[str, Any] | None) -> float | None:
    """Where this game sits on the score ramp, or nothing.

    The same three refusals the fill runs under: no scale, no current measurement, and a
    measurement under other weights than the ramp's. Two scores under different weights
    never share a leaderboard, so one of them must not be drawn bigger than the other.
    """
    score = node.score or {}
    index = score.get("index")
    if not scale or index is None or score.get("weights") != scale["weights"]:
        return None
    span = float(scale["high"]) - float(scale["low"])
    if span <= 0:
        return 1.0
    at = (float(index) - float(scale["low"])) / span
    return round(min(1.0, max(0.0, at)), 4)


def _card(group: list[Game]) -> dict[str, Any] | None:
    """This node's score as the library reads it, or nothing.

    Every field is the card's own, under the card's own spelling, so the map and the
    listing cannot come to describe one measurement two ways. A stale measurement is
    `None` at `library.current_only`: the map's ramp is a comparison, and a provisional
    estimate never enters one (`docs/archive/plan-evolving-score.md` §3); its card says it.

    A merged node's score is its representative's. The two spellings are one game but not
    one measurement -- move order follows spelling, so the runs are not interchangeable --
    and picking one file's runs is the only honest answer while `runs/` is keyed by hash.
    """
    game = group[0]
    from chess_variant_sandbox.web.library import current_only

    score = current_only(game.score)
    if not score or score.get("index") is None:
        return None
    return {
        "index": score.get("index"),
        "sd": score.get("sd"),
        "viable": score.get("viable"),
        "weights": score.get("weights_version"),
        # The run count and the verdict travel with the number, as they do on a card: a
        # mean of four runs and a single run are not the same claim.
        "seeds": list(score.get("seeds") or []),
        "runs": [dict(run) for run in (score.get("runs") or [])],
        "totalsUsed": score.get("totals_used"),
        "runVerdict": score.get("verdict") or None,
    }


# ----------------------------------------------------------------------- the lineage


def _lineage(
    games: list[Game],
    by_hash: dict[str, str],
    parents: Mapping[str, str],
    nodes: Mapping[str, list[Game]],
    cache: _Cache,
) -> list[dict[str, Any]]:
    """Parent-to-child arrows, from the three places a parent is ever recorded.

    A published variant names its parent in `provenance.parents`; an evolved one has it in
    the design run's `lineage.jsonl`, which the library has already merged onto the game;
    a game made on the site has it in the `games.parent` column, which the site passes in.
    Only the immediate parent is drawn -- `parents` is a chain, and its later entries are
    grandparents whose arrow the chain already shows.

    An arrow is drawn at any distance, which is the point of having it: a child that adds
    a defining mechanic lands in that mechanic's community, far from its parent, and
    *this game was made from that one* is a fact no comparison can find.

    A parent that is not on the map is dropped rather than invented, and a parent that
    merged into its child's node -- an edit that changed no table -- is dropped too.
    """
    arrows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for game in games:
        declared = dict(game.spec.get("provenance") or {})
        chain = game.lineage.get("parents") or declared.get("parents") or []
        first = chain[0] if chain else None
        digest = (
            first.get("hash") if isinstance(first, dict)
            else first if isinstance(first, str)
            else None
        ) or parents.get(game.hash)
        child, parent = by_hash.get(game.hash), by_hash.get(str(digest))
        if child is None or parent is None or child == parent:
            continue
        if (parent, child) in seen:
            continue
        seen.add((parent, child))
        arrows.append({
            "source": parent,
            "target": child,
            "units": int(_reading(nodes[parent], nodes[child], cache)["units"]),
            "run": str(game.lineage.get("run") or declared.get("run") or ""),
            "generation": game.lineage.get("generation", declared.get("generation")),
        })
    return sorted(arrows, key=lambda arrow: (arrow["source"], arrow["target"]))


# What each kind of rule is, in words a reader who has never seen this project can act
# on. `relate.KINDS` are the schema's own names -- *wild*, *hand*, *valuation* -- and a
# panel that printed them would be the maintainer's shorthand on a public map. The words
# live here rather than on the page because the payload is the map's whole interface, and
# a page that wrote its own would be a second opinion about what a kind is.
#
# Exhaustive over `relate.KINDS`, which a test holds: a kind added to the schema fails
# that test rather than reaching a panel unnamed.
KIND_WORDS: dict[str, str] = {
    "board": "the board",
    "pieces": "the pieces and how they move",
    "setup": "where the pieces start",
    "capture": "who may take whom",
    "ending": "how the game ends",
    "hand": "taken pieces coming back",
    "wild": "pieces nobody owns",
    "turn": "whose turn it is",
    "clock": "when a game is called a draw",
    "valuation": "what a piece is worth",
}


def _kinds(lines: Sequence[Line]) -> list[dict[str, Any]]:
    """Every kind of rule, in `relate`'s order, in words, with how many lines differ in it.

    Every kind is listed whether or not the library has a line of it, so a reader's words
    for a kind never depend on what the library holds.
    """
    counted: dict[str, int] = defaultdict(int)
    for line in lines:
        for kind in line.kinds:
            counted[kind] += 1
    return [
        {"kind": kind, "words": KIND_WORDS.get(kind, kind), "lines": counted.get(kind, 0)}
        for kind in relate.KINDS
    ]


def _bounds(nodes: Sequence[Any]) -> dict[str, float]:
    """The box the map fits in, for the page's `viewBox`, with room for a node's radius."""
    if not nodes:
        return {"minX": 0.0, "minY": 0.0, "maxX": 0.0, "maxY": 0.0, "width": 0.0, "height": 0.0}
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]
    low_x, low_y = min(xs) - NODE_RADIUS, min(ys) - NODE_RADIUS
    high_x, high_y = max(xs) + NODE_RADIUS, max(ys) + NODE_RADIUS
    return {
        "minX": round(low_x, 3),
        "minY": round(low_y, 3),
        "maxX": round(high_x, 3),
        "maxY": round(high_y, 3),
        "width": round(high_x - low_x, 3),
        "height": round(high_y - low_y, 3),
    }


def _node(group: list[Game], degree: int, spot: tuple[float, float]) -> Node:
    from chess_variant_sandbox.web.library import exhibit

    game = group[0]
    board = game.spec.get("board") or {}
    return Node(
        hash=game.hash,
        name=game.name,
        names=tuple(member.name for member in group),
        hashes=tuple(member.hash for member in group),
        source=game.source,
        board=f"{board.get('width', '?')}x{board.get('height', '?')}",
        degree=degree,
        unusual=False,
        score=_card(group),
        staleness=game.staleness,
        exhibit=exhibit(game),
        x=spot[0],
        y=spot[1],
    )


# ------------------------------------------------------------------------- the cache


class _Cache:
    """Every compile digest and every pair this machine has read, across restarts.

    One record per pair of hashes, stamped with `relate`'s version and this module's. A
    file under any other stamp is dropped whole: a reading taken under an older comparison
    is not a reading of the current one, and half a file under each is worse than a
    recompute.

    Written once at the end of a build and only when something was added, atomically, so
    two requests racing leave one whole file rather than half of two.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.specs: dict[str, str] = {}
        self.pairs: dict[str, dict[str, Any]] = {}
        self.computed = 0
        self.compiled = 0
        self._dirty = False

    def read(self) -> None:
        try:
            held = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(held, dict):
            return
        if held.get("relate") != relate.RELATE_VERSION or held.get("graph") != GRAPH_VERSION:
            return
        self.specs = {
            str(key): str(value) for key, value in (held.get("specs") or {}).items()
        }
        self.pairs = {
            str(key): value
            for key, value in (held.get("pairs") or {}).items()
            if isinstance(value, dict) and "units" in value
        }

    def write(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "relate": relate.RELATE_VERSION,
                        "graph": GRAPH_VERSION,
                        "specs": self.specs,
                        "pairs": self.pairs,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError:
            # A runs directory that cannot be written serves the map uncached.
            return
        self._dirty = False

    def twin(self, game: Game) -> str:
        """What this spec compiles to, as a digest, the Zobrist draw set aside.

        Cached on the mechanics hash, which is exact rather than approximate: the tables
        are built from `canonical.MECHANIC_KEYS` and nothing else, so two specs with one
        hash compile to one set of tables. A change to the compiler that moved a table
        would need `GRAPH_VERSION` bumped, which is what that constant is for.
        """
        held = self.specs.get(game.hash)
        if held is not None:
            return held
        held = _TWINS.get(game.hash)
        if held is not None:
            self.specs[game.hash] = held
            self._dirty = True
            return held
        from chess_variant_sandbox.engine.bridge import canonical_tables
        from chess_variant_sandbox.engine.compile import compile_spec

        tables = canonical_tables(compile_spec(game.spec))
        blob = json.dumps(
            {key: value for key, value in tables.items() if not key.startswith("zobrist")},
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: value.hex() if isinstance(value, bytes) else repr(value),
        )
        answer = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        _TWINS[game.hash] = answer
        self.specs[game.hash] = answer
        self.compiled += 1
        self._dirty = True
        return answer

    def pair(self, first: Game, second: Game) -> dict[str, Any]:
        """What these two specs differ in, from the file or from `relate`.

        The differences themselves are kept only for a pair within `NEAR` rules; past it
        the count is all a line says.
        """
        key = ":".join(_key(first.hash, second.hash))
        held = self.pairs.get(key)
        if held is not None:
            return held
        relation = relate.relate(first.spec, second.spec)
        record: dict[str, Any] = {"units": relation.units}
        if relation.units <= NEAR:
            record["differences"] = [one.as_dict() for one in relation.differences]
        self.pairs[key] = _ordered(record)
        self.computed += 1
        self._dirty = True
        return self.pairs[key]
