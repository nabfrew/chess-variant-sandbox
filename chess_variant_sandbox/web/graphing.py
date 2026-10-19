"""The library as a graph: which games are near which, and where each one sits.

`spec/relate.py` answers *what differs between these two games* and knows nothing about a
library. This is the library half. It decides which of the 2,415 pairs are worth drawing,
merges two spellings of one game into one node, lays the result out, and hands the page a
payload it can draw without computing anything.

Eight decisions, each §3's in `docs/archive/plan-game-graph.md`:

1. **Twins merge.** Two specs whose compiled tables are equal, the Zobrist draw apart, are
   one game under two spellings and become one node carrying both names. The reading the
   node keeps is the shorter of the two, because a rule declared the long way is still
   that rule. The library has no such pair today: `wolves` and `wolves-immune` were the
   only one until Q876 retired the second, and `tests/test_graphing.py` keeps the merge
   measured on that pair (rebuilt) and on made ones.
2. **Every one-rule link is drawn.**
3. **A two- or three-rule link is drawn only when it is unbridged** -- when no third game
   splits it into shorter steps -- and it is drawn as a *gap*, because a pair nothing
   stands between marks a game nobody has made yet.
4. **A node left with no link at all gets one faint link to its nearest relative**, by
   `spec/distance.py`'s 55 counted features. The label says how far, because at that range
   the nearest game is not a relative in any useful sense.
5. **The groups are the similarity communities.** What the page outlines and folds is
   `communities`, found from the similarity lines rather than from the rule steps; the
   same-rule cliques this module once built (`clusters`) and the partition folded from
   them (`groups`) left the payload at `g13` (`docs/archive/plan-graph-map.md` S5).
6. **Lineage is a second kind of link**, an arrow from parent to child at any distance,
   from what the spec, the design run or the site's own column recorded.
7. **A one-rule link says which design step makes one game from the other**, where one of
   `loop/operators.py`'s own edits reaches it. The words are the operator's, not this
   module's, and the draws are off the request path -- see *the operator's own sentence*.
8. **A gap says whether the game between its two ends can be built**, which
   `web/designing.py` decides and this module only asks. Ten of the corpus's fifty-five
   gaps differ in declared rules alone, which is what makes one buildable.

**Two measures ride on every pair.** `relate.units` counts the rules that differ and
decides which lines exist at all. `relate.similarity` weighs four tiers of difference --
rules, movement mechanisms, piece identities, starting squares -- into one number, which
ranks the lines and says in words what kind of difference they are. Every link carries it
and the pair cache keeps its counts; `docs/archive/plan-game-graph.md` §9 holds both.

**Nothing here re-reads the DSL.** What a rule *means* is `engine/compile.py`'s, what two
specs differ in is `spec/relate.py`'s, how far apart they are by feature is
`spec/distance.py`'s, and what a game scored is the library's -- read from the card it
already built, never recomputed. This module joins, counts, lays out and caches. No piece
name and no ruleset name appears in it.

**The cost, and where it goes.** One spec compiles in about 45 ms and the corpus in three
seconds; one pair compares in under a millisecond and all 2,415 in 1.4 s. Both survive a
restart in `runs/graph/pairs.json`, keyed by hash and stamped with `relate`'s version and
this module's, so the second call after a cold start computes nothing at all. A new game
costs its own pairs and its own compile.

The operator sweep is the one thing here that costs real time -- twenty seconds on the
committed corpus against the whole build's five -- so it is off by default and no request
ever pays for it. `build(..., sentences=True)` and `python -m chess_variant_sandbox.web.graphing` are
where it is paid, and the sentences it finds land in the same cache file. The game in
between a gap is a fifth of a second over the whole corpus and is worked out on the first
build rather than asked for, and it lands in that file too.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chess_variant_sandbox.spec import relate

if TYPE_CHECKING:  # a Game is the library's; importing it at runtime would be a cycle
    from chess_variant_sandbox.web.library import Game

# Bumped when a cached reading moves -- the twin key or what a pair record holds -- and
# when the payload grows a key, so a reader cannot take an older shape for this one. A
# cache file stamped with anything else is dropped whole rather than read. `g10`: a node
# carries `exhibit`, the sentence a deliberately kept example is labelled with (Q141).
# `g11`: the similarity lines (`similar`, `levels`) and the communities they form
# (`communities`, and `community` on every node), `docs/archive/plan-graph-map.md` S1. `g12`: the
# opening picture laid out by community (S2), so every coordinate moved. `g13`: the
# same-rule `clusters` and the `groups` folded from them leave the payload, because the
# communities are what the page outlines and folds (S5). `g14`: `spec/distance.py` rescales
# uniformly (three new `_SCALES` entries, Q281a3), so a `distance` a pair record cached
# under `g13` no longer matches what `distance()` returns for that pair.
GRAPH_VERSION = "g14"

# How many rules apart two games may be and still be drawn. `web/library.py` holds the
# same number for a game's own page; three is the threshold of
# `docs/archive/plan-game-graph.md` §3, past which a link stops being a reading and
# becomes a ranking of how unlike everything else a game is.
NEAR = 3

# ------------------------------------------------------------ the similarity lines
#
# `docs/archive/plan-graph-map.md` §2 to §4. The index is a distance -- 0 is one game, and a higher
# number a less alike pair -- so a line is drawn with a *bond*, which rises as the index
# falls and counts the index's own tiers of eight: `5 - log8(1 + index)`, 5 for a twin, 2
# at one rule's weight. Line width and the weight communities are found by are both it.

# The cut the map opens on: two games are joined when everything that separates them
# weighs no more than one rule. §3 chose it before the picture was drawn; it sits on a
# cliff, 71 lines at 448 and 203 at 640, and is the last cut before the clutter.
SIMILAR = 512
# The cuts a reader may move between, each with its count in the payload. A row carries
# the first cut that draws it, so showing a cut is comparing two server integers.
LEVELS = (256, 384, 512, 640, 768)
# The bond's top, which is where a twin sits, and the ratio its tiers are counted in.
BOND_TOP = 5.0
# How a community is named. A rule is *its* when at least this share of its members
# involve it and no more than `DISTINCT_OUT` of the other joined games do; with none, it
# is named after its centre game.
DISTINCT_IN = 0.75
DISTINCT_OUT = 0.25

# Where the pair cache lives, under the runs directory the caller names.
CACHE_NAME = "graph"
CACHE_FILE = "pairs.json"

# ---------------------------------------------------------------------- the layout
#
# Plain numbers in layout units, one unit being a node's own radius. The page scales them
# into a `viewBox`; nothing here knows a pixel.

NODE_RADIUS = 1.0
# The length a line of average bond settles at inside a community, which is the force
# layout's natural spacing. The payload calls it `ring`, the name the radial layout it
# replaced gave the same step.
RING = 6.0
# The least distance between two games of one community, which leaves a name's worth of
# air on top of the one diameter the page promises.
MIN_SEPARATION = 4.0
# The least air between two communities, and between the drawing and the far row; and the
# step between two far games in their row.
COMPONENT_GAP = 6.0
ROW_STEP = 5.0
# The force layout's passes, inside a community and between communities. Enough for the
# largest community on the corpus (thirteen games) to stop moving; the whole layout costs
# a few hundredths of a second.
KNOT_PASSES = 300
SPREAD_PASSES = 400
# The layout's own draw, for the only randomness in it: a small nudge off the starting
# circle so that a symmetric start does not stay symmetric. Its own generator, always.
LAYOUT_SEED = 0
# How many turns of each community are tried when facing it towards its neighbours.
TURNS = 24


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
    degree: int
    far: bool
    score: dict[str, Any] | None
    staleness: dict[str, Any] | None
    # `library.exhibit`: the sentence saying this game is kept on purpose for what it
    # shows, or `None`. The panel prints it beside the score, as the card does.
    exhibit: str | None = None
    x: float = 0.0
    y: float = 0.0
    # The id of the community this game is in, or `None` for a game no similarity line
    # reaches at the default cut, which is in no community and sits in the far row.
    community: str | None = None
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
            "far": self.far,
            "size": dict(self.size),
            "score": self.score,
            "staleness": self.staleness,
            "exhibit": self.exhibit,
            "community": self.community,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
        }


@dataclass
class Link:
    """One line on the map.

    `type` is what the line *is*: `rule` for a one-rule link, `gap` for an unbridged pair
    two or three rules apart, `nearest` for the faint fallback a far game gets. `units` is
    the count of rules for the first two and absent for the third, which is a feature
    distance and not a count of anything.

    `similarity` is `{index, counts, words}`, the tiered reading of the same pair. `units`
    says how many steps apart the two games are and decides whether a line exists at all;
    the index ranks the lines and `words` explains the rank.
    """

    source: str
    target: str
    type: str
    units: int | None
    kind: str
    kinds: tuple[str, ...]
    sentence: str
    differences: tuple[dict[str, Any], ...]
    unit: str
    similarity: dict[str, Any]
    distance: float | None = None
    gap: dict[str, Any] | None = None
    reach: dict[str, Any] | None = None
    between: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "units": self.units,
            "kind": self.kind,
            "kinds": list(self.kinds),
            "unit": self.unit,
            "sentence": self.sentence,
            "similarity": dict(self.similarity),
            "differences": [dict(one) for one in self.differences],
            "distance": self.distance,
            "gap": self.gap,
            "reach": None if self.reach is None else dict(self.reach),
            "between": None if self.between is None else dict(self.between),
        }


@dataclass
class Graph:
    """The payload, and what building it cost.

    `computed`, `compiled`, `drawn` and `spanned` are the four things that must be zero on
    a second call: the pairs this build had to compare, the specs it had to compile, the
    operator draws it had to make and the gaps it had to build a game in between. Kept on
    the result rather than printed, so a test can assert the cache works instead of
    timing it.
    """

    nodes: list[Node]
    links: list[Link]
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
    # Every node pair within the last of `LEVELS`, the cuts with their counts, and the
    # communities the default cut's lines form. `docs/archive/plan-graph-map.md` S1.
    similar: list[dict[str, Any]] = field(default_factory=list)
    levels: list[dict[str, Any]] = field(default_factory=list)
    communities: list[dict[str, Any]] = field(default_factory=list)
    computed: int = 0
    compiled: int = 0
    drawn: int = 0
    spanned: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": GRAPH_VERSION,
            "relate": relate.RELATE_VERSION,
            "nodes": [node.as_dict() for node in self.nodes],
            "links": [link.as_dict() for link in self.links],
            "lineage": [dict(arrow) for arrow in self.lineage],
            "kinds": [dict(kind) for kind in self.kinds],
            "bounds": dict(self.bounds),
            "units": {"nodeRadius": NODE_RADIUS, "ring": RING},
            "sizes": [dict(size) for size in self.sizes],
            "overlays": list(self.overlays),
            "scale": None if self.scale is None else dict(self.scale),
            "components": dict(self.components),
            "counts": dict(self.counts),
            "similar": [dict(row) for row in self.similar],
            "levels": [dict(level) for level in self.levels],
            "communities": [
                {**community, "members": list(community["members"]),
                 "names": list(community["names"])}
                for community in self.communities
            ],
        }


def graph(
    games: Iterable[Game],
    runs_dir: Path | str = "runs",
    parents: Mapping[str, str] | None = None,
    sentences: bool = False,
    overlays: Iterable[str] = (),
) -> dict[str, Any]:
    """The whole map as JSON, for `GET /api/graph`."""
    return build(games, runs_dir, parents, sentences, overlays).as_dict()


def build(
    games: Iterable[Game],
    runs_dir: Path | str = "runs",
    parents: Mapping[str, str] | None = None,
    sentences: bool = False,
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

    `sentences` fills the operator sentence on every one-rule link that has none yet,
    which costs twenty seconds on the committed corpus and belongs to whoever can afford
    it -- a deploy step or the module's own command line, never a request. Off, the build
    prints the sentences the cache already holds and draws nothing.
    """
    listed = sorted(games, key=lambda game: (game.name.lower(), game.hash))
    cache = _Cache(Path(runs_dir) / CACHE_NAME / CACHE_FILE)
    cache.read()

    groups = _twin_groups(listed, cache)
    nodes = {group[0].hash: group for group in groups}
    by_hash = {game.hash: group[0].hash for group in groups for game in group}

    readings = _readings(groups, cache)
    ones, gaps = _structural(list(nodes), readings)
    linked = {node for pair in ones + gaps for node in pair}
    fallback = _fallback(groups, [node for node in nodes if node not in linked], cache)

    degree: dict[str, int] = defaultdict(int)
    for first, second in ones + gaps:
        degree[first] += 1
        degree[second] += 1

    if sentences:
        _sweep(nodes, ones, cache)
    links = [_link(pair, "rule", readings, cache) for pair in ones]
    links += [_link(pair, "gap", readings, cache) for pair in gaps]
    links += [_nearest(pair, span, readings) for pair, span in fallback]
    _fill_gaps(links, nodes)
    _fill_between(links, nodes, cache)

    similar = _similar(readings)
    levels = _levels(similar, list(nodes))
    lines = _lines(similar)
    parts = _partition(lines)
    centres = [_centre(part, readings) for part in parts]
    member_of = {
        digest: _community_id(centre)
        for part, centre in zip(parts, centres, strict=True)
        for digest in part
    }

    placed = _layout(list(nodes), lines, parts)
    built = [
        replace(
            _node(nodes[digest], degree.get(digest, 0), digest not in linked, placed[digest]),
            community=member_of.get(digest),
        )
        for digest in nodes
    ]
    built.sort(key=lambda node: (node.name.lower(), node.hash))
    communities = _communities(parts, centres, nodes, built)

    # Both ends of every size key are the library's own, so a dot's place between them is
    # known only once every dot is built.
    scale = _scale(built)
    sizes = _sizes(built, scale)
    built = [replace(node, size=_sized(node, scale)) for node in built]

    arrows = _lineage(listed, by_hash, parents or {}, readings, nodes)
    cache.write()
    return Graph(
        nodes=built,
        links=links,
        lineage=arrows,
        kinds=_kinds(links),
        bounds=_bounds(built),
        sizes=sizes,
        overlays=list(overlays),
        scale=scale,
        components=_components(links),
        similar=similar,
        levels=levels,
        communities=communities,
        counts={
            "nodes": len(built),
            "specs": len(listed),
            "twins": sum(1 for group in groups if len(group) > 1),
            "links": len(links),
            "oneRule": len(ones),
            "gaps": len(gaps),
            "nearest": len(fallback),
            "lineage": len(arrows),
            # One-rule links an operator draw reached, which is what the cache holds
            # rather than what this call did: an unfilled cache reads zero.
            "sentences": sum(1 for link in links if link.reach),
            # Gaps a game can be built between, which is the button the map offers.
            "between": sum(
                1 for link in links if link.between and link.between.get("words")
            ),
            # Nodes no link of any kind touches, which the fallback exists to keep at
            # zero. A library of one game is the one case that cannot be helped.
            "unlinked": len(
                set(nodes) - {end for link in links for end in (link.source, link.target)}
            ),
            # The similarity lines at the default cut, the games none of them reaches,
            # and the communities they form.
            "similar": sum(1 for row in similar if row["index"] <= SIMILAR),
            "alone": next(
                (level["alone"] for level in levels if level["cut"] == SIMILAR), len(nodes)
            ),
            "communities": len(communities),
        },
        computed=cache.computed,
        compiled=cache.compiled,
        drawn=cache.drawn,
        spanned=cache.spanned,
    )


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


# ----------------------------------------------------------- the readings, pair by pair


def _readings(groups: list[list[Game]], cache: _Cache) -> dict[tuple[str, str], Any]:
    """What every pair of *nodes* differs in, as `relate` reads it.

    A merged node takes the shorter of its spellings' readings to each other node, which
    is the whole point of merging: a rule declared on five pieces and the same rule
    declared once are one rule, and the node should say so wherever it is compared.

    The similarity index is chosen the same way and on its own, because the two measures
    rank differently and the node should carry the nearest reading under each.
    """
    found: dict[tuple[str, str], Any] = {}
    for first, second in combinations(groups, 2):
        records = [cache.pair(one, other) for one in first for other in second]
        best = min(records, key=lambda reading: reading["units"])
        nearest = min(records, key=lambda reading: _similarity(reading).index)
        if nearest is not best:
            best = {**best, "similarity": nearest["similarity"]}
        found[_key(first[0].hash, second[0].hash)] = best
    return found


def _similarity(record: Mapping[str, Any]) -> relate.Similarity:
    """One pair record's tier counts, weighed into the index at today's ratio."""
    return relate.Similarity(**record["similarity"])


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


def _units(readings: dict[tuple[str, str], Any], first: str, second: str) -> int:
    return int(readings[_key(first, second)]["units"])


def _differences(
    readings: dict[tuple[str, str], Any], pair: tuple[str, str]
) -> list[dict[str, Any]]:
    """One pair's differences, which a record past `NEAR` rules does not carry."""
    return list(readings[_key(*pair)].get("differences") or [])


# ------------------------------------------------------------- which links are drawn


def _structural(
    nodes: list[str], readings: dict[tuple[str, str], Any]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The one-rule links, and the two- and three-rule pairs nothing stands between.

    **A pair is bridged when a third game splits it into shorter steps** -- when some
    other node is `left` from one end and `right` from the other with both shorter than
    the pair and `left + right` no longer than it. Drawing only the unbridged ones is what
    keeps the picture from collapsing: 242 pairs are two rules apart and most of them are
    two of one hub's neighbours, which the two links through the hub already say.

    A bridge must be a real step, so a middle at zero units cannot bridge anything. After
    the twin merge no two nodes are at zero, and the guard stays because it is a property
    of *bridging* rather than of this corpus.
    """
    ones: list[tuple[str, str]] = []
    wide: list[tuple[str, str]] = []
    for first, second in combinations(sorted(nodes), 2):
        span = _units(readings, first, second)
        if span == 1:
            ones.append((first, second))
        elif 1 < span <= NEAR:
            wide.append((first, second))

    return ones, [pair for pair in wide if not _bridged(pair, nodes, readings)]


def _bridged(
    pair: tuple[str, str], nodes: list[str], readings: dict[tuple[str, str], Any]
) -> bool:
    first, second = pair
    span = _units(readings, first, second)
    for middle in nodes:
        if middle in pair:
            continue
        left = _units(readings, first, middle)
        right = _units(readings, middle, second)
        if 0 < left < span and 0 < right < span and left + right <= span:
            return True
    return False


def _fallback(
    groups: list[list[Game]], alone: list[str], cache: _Cache
) -> list[tuple[tuple[str, str], float]]:
    """One faint link per node that no structural link reaches, to its nearest relative.

    `spec/distance.py` counts 55 features and answers with a number and no words, which is
    exactly what is wanted here: at five rules and up, *which* rules differ is a list
    nobody reads, and the useful thing the map can say is which game is least unlike this
    one. The label carries the rule count too, so the reader can see the link is weak.

    Two far games that pick each other yield one link rather than two, and a library of one
    game has no relative to be near, which is the one case a node is left with no link.
    """
    representatives = {group[0].hash: group[0] for group in groups}
    if len(representatives) < 2:
        return []
    found: dict[tuple[str, str], float] = {}
    for digest in sorted(alone):
        mine = representatives[digest]
        best = min(
            (cache.distance(mine, other), other.name.lower(), other.hash)
            for other in representatives.values()
            if other.hash != digest
        )
        found[_key(digest, best[2])] = best[0]
    return sorted(found.items())


# --------------------------------------------------- the operator's own sentence (S3)
#
# A one-rule link says *these two games differ in the promotion rule*. What it does not
# say is what a designer would **do** to get from one to the other, and the project
# already owns that answer: `loop/operators.py` is thirty-four named local edits, each of
# which writes a sentence about what it just did. So the question is whether one of those
# edits, drawn from this game, lands on that one -- and where it does, the line can carry
# the operator's own words instead of a description of a difference.
#
# The draw is blind. An operator cannot be aimed at a target, so the only way to ask is
# to draw it and look at what came out, which is why the seeds are fixed and the whole
# sweep is a pure function of the corpus and the operator set.

# Seeds of each operator drawn from each game. Measured at Q348: the committed corpus
# reaches 22 links at five seeds and all 23 at ten, and eighty finds nothing
# more. Forty is the plan's number and the headroom is for a library that is
# not this one; the table is G5's, in `docs/archive/plan-game-graph.md` (*the operator
# sentence as built*), and cutting it to ten is where the sweep's cost is if it ever has
# to be paid on a path that cares.
SEEDS = 40


def _sweep(
    nodes: dict[str, list[Game]], ones: list[tuple[str, str]], cache: _Cache
) -> None:
    """Draw every operator from every end of a one-rule link, and keep what lands.

    **Per game, not per pair.** What a seeded draw from one game produces does not depend
    on which neighbour is being asked about, so a game is swept once against all of its
    one-rule neighbours at once. The row in the plan reads *both ends of each link*; on
    the committed corpus that is the same 50,160 draws read as 38 sweeps rather than 112.

    **A match is `relate.identity_key`'s and nothing else.** A child that normalises to
    what the other end normalises to *is* the other game, whatever the two files spell;
    a twin node answers to either of its spellings, because it is one game under both.

    **An operator that raises, declines or hands back a refused child is a non-match.**
    Validation runs only on a child that already matched, which costs nothing and changes
    no answer: an invalid child that matched is dropped just the same.

    **The first match wins, and *first* is a fixed walk** -- the ends of a link in the
    payload's own order (source before target, which is hash order), then the operators
    in `OPERATORS` order, then the seeds from zero. Nothing here reads a clock or a thread
    count.
    """
    import copy

    from chess_variant_sandbox.loop.operators import OPERATORS
    from chess_variant_sandbox.spec import validate_spec

    # Which pairs are still to do, and what each end is looking for. A pair the cache has
    # already answered -- with a sentence or with nothing -- is not drawn again.
    wanted: dict[str, dict[str, str]] = defaultdict(dict)
    for pair in sorted(ones):
        if cache.swept(*pair):
            continue
        for mine, other in (pair, pair[::-1]):
            for member in nodes[other]:
                wanted[mine][relate.identity_key(member.spec)] = other

    found: dict[tuple[str, str], dict[str, Any] | None] = {
        _key(*pair): None for pair in ones if not cache.swept(*pair)
    }
    for digest in sorted(wanted):
        spec = nodes[digest][0].spec
        targets = wanted[digest]
        left = {other for other in targets.values() if found[_key(digest, other)] is None}
        for operator in OPERATORS:
            for seed in range(SEEDS):
                if not left:
                    break
                child = copy.deepcopy(spec)
                cache.drawn += 1
                try:
                    summary = operator.apply(child, random.Random(seed))
                except Exception:  # noqa: BLE001 - a draw that cannot apply is not news
                    continue
                if summary is None:
                    continue
                landed = targets.get(relate.identity_key(child))
                if landed is None or landed not in left or validate_spec(child):
                    continue
                left.discard(landed)
                found[_key(digest, landed)] = {
                    "from": digest, "operator": operator.name, "words": summary,
                }
            if not left:
                break

    for pair, reach in found.items():
        cache.record(pair, reach)


# ------------------------------------------------------------------- building a link


def _link(
    pair: tuple[str, str],
    type_: str,
    readings: dict[tuple[str, str], Any],
    cache: _Cache,
) -> Link:
    differences = tuple(_differences(readings, pair))
    first = differences[0] if differences else {}
    return Link(
        source=pair[0],
        target=pair[1],
        type=type_,
        units=_units(readings, *pair),
        # The kind of the first difference, which is the whole kind on a one-rule link and
        # the one a mixed link is filed under. Every kind on it travels too, so a chip
        # that hides a kind never has to guess from the first.
        kind=str(first.get("kind", "")),
        kinds=tuple(sorted({str(one["kind"]) for one in differences})),
        unit=str(first.get("unit", "")),
        sentence="; ".join(str(one["sentence"]) for one in differences),
        differences=differences,
        similarity=_similarity(readings[_key(*pair)]).as_dict(),
        # Filled on a one-rule link and left empty elsewhere: the sweep draws against a
        # node's one-rule neighbours alone, so a gap never has one to read.
        reach=cache.reach(*pair),
    )


def _nearest(
    pair: tuple[str, str], span: float, readings: dict[tuple[str, str], Any]
) -> Link:
    """The faint link, which says how far rather than what differs.

    Its `units` is the rule count, so the page can say *nearest relative, 21 rules apart*;
    its `differences` are left empty on purpose, because a list of nine rules is not a
    reading anybody acts on. Which game the line points at is `spec/distance.py`'s answer
    and not the index's: `docs/archive/plan-game-graph.md` §9.5's T5 scored both, and a
    variant of the index, against a list of accepted relatives written first, and the
    features won.
    `library.neighbours` gives a far game's page its four by the same measure.
    """
    return Link(
        source=pair[0],
        target=pair[1],
        type="nearest",
        units=_units(readings, *pair),
        kind="",
        kinds=(),
        unit="",
        sentence="nearest relative",
        differences=(),
        similarity=_similarity(readings[_key(*pair)]).as_dict(),
        distance=round(span, 4),
    )


def _fill_gaps(links: list[Link], nodes: dict[str, list[Game]]) -> None:
    """Both indexes and the distance between them, on every link where both can be read.

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
    for link in links:
        mine = _card(nodes[link.source])
        theirs = _card(nodes[link.target])
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
        link.gap = {
            "index": float(here),
            "otherIndex": float(there),
            "seeds": seeds,
            "otherSeeds": other_seeds,
            "gap": round(gap, 4),
            "spread": round(spread, 4),
            "clear": abs(gap) > 2 * spread,
            "weights": mine.get("weights"),
            "moved": _moved(nodes[link.source], nodes[link.target]),
        }


def _fill_between(links: list[Link], nodes: dict[str, list[Game]], cache: _Cache) -> None:
    """On every gap, the game that stands between its two ends -- or why there is none.

    A gap marks a game nobody has made, and where each rule that differs is one the spec
    declares, that game can be built by carrying one of them across. `designing.in_between`
    is the one place that is decided, so the button the map offers and the draft the
    designer opens cannot disagree about which gaps have one.

    The spec it made is dropped here. A payload carrying twenty drafts would be half as
    big again for something nobody has pressed; what travels is which end it is built on,
    which end the rule comes from and that rule's own phrase, and the press asks the
    designer for the draft itself.

    Measured at Q350 on the committed corpus: ten of the fifty-five gaps differ in
    declared rules alone, and the pass costs 0.27 s, nearly all of it compiling the ten
    games it builds. That is more than a warm request should pay, so it goes in the pair
    cache beside the operator sentence.

    **A merged node is asked about through its representative**, which is the spelling the
    page fetches by and the one the endpoint will build from when the button is pressed.
    The drawn line may carry the *other* spelling's shorter reading, so for a twin the
    count on the line and the count this reads can differ by a unit -- in which case this
    offers nothing or offers a longer split, and never something the press cannot make.
    The library's one pair of twins (retired at Q876) had none of its gaps affected, and
    none exists now.
    """
    from chess_variant_sandbox.web import designing

    for link in links:
        if link.type != "gap":
            continue
        held = cache.between(link.source, link.target)
        if held is None:
            ends = (nodes[link.source][0], nodes[link.target][0])
            made = designing.in_between(
                ends[0].spec, ends[1].spec, (ends[0].name, ends[1].name)
            )
            held = (
                {"why": str(made["why"])}
                if "spec" not in made
                else {
                    "from": (link.source, link.target)[int(made["on"])],
                    "taken": (link.source, link.target)[1 - int(made["on"])],
                    "words": str(made["words"]),
                }
            )
            cache.span(link.source, link.target, held)
        link.between = held


# How many of the index's parts a link names. Three is what fits under a sentence in the
# panel, and past the third the changes are the size of the noise on any one of them. How
# many parts there are is the two games' shared weights version's business, and a link
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
    payload nothing draws -- so they are read here, where a link needs the pair.
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


def _components(links: list[Link]) -> dict[str, str]:
    """A readable name for every component a link names, from the glossary.

    `metrics/glossary.py` is where a measurement is named in English, and the map prints
    those words for the same reason the legend prints `KIND_WORDS`: `skill_gradient` is
    an identifier, and a page that spelled it out its own way would be a second opinion
    about what it measures.
    """
    from chess_variant_sandbox.metrics.glossary import title_of

    named = {
        str(part["key"])
        for link in links
        for part in ((link.gap or {}).get("moved") or [])
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

# The link count at which a dot is drawn full size. G3 chose it, and it is kept here so
# the `links` key reproduces G3's radius exactly: a busier game than this draws the same
# largest dot, because the layout promises only one diameter between two dots.
LINKS_FULL = 8


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
    found: list[dict[str, Any]] = [{
        "key": "links",
        "label": "how many links",
        "words": "How many games are within three rules of this one.",
        "low": low,
        "high": LINKS_FULL,
        "lowWords": "no links" if low == 0 else f"{low} links",
        "highWords": f"{LINKS_FULL} or more links",
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
    at the smallest size and the legend explains. `links` always has a value, because a
    count of nothing is nought rather than nothing.
    """
    return {
        "links": round(min(1.0, node.degree / LINKS_FULL), 4),
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
    readings: dict[tuple[str, str], Any],
    nodes: dict[str, list[Game]],
) -> list[dict[str, Any]]:
    """Parent-to-child arrows, from the three places a parent is ever recorded.

    A published variant names its parent in `provenance.parents`; an evolved one has it in
    the design run's `lineage.jsonl`, which the library has already merged onto the game;
    a game made on the site has it in the `games.parent` column, which the site passes in.
    Only the immediate parent is drawn -- `parents` is a chain, and its later entries are
    grandparents whose arrow the chain already shows.

    An arrow is drawn at any rule distance, which is the point of having it: four of the
    ten sit four rules and more from their parent, where no structural link reaches, and
    *this game was made from that one* is a fact no diff can find.

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
            "units": _units(readings, parent, child),
            "run": str(game.lineage.get("run") or declared.get("run") or ""),
            "generation": game.lineage.get("generation", declared.get("generation")),
        })
    return sorted(arrows, key=lambda arrow: (arrow["source"], arrow["target"]))


# ------------------------------------------------- the similarity lines and communities
#
# `docs/archive/plan-graph-map.md` S1. The rule-step lines above say *how many* rules apart two
# games are; these say how alike they are, from the index every pair already carries, so
# nothing here compares a spec. The page draws a row by its `bond` and shows it when its
# `level` is at most the cut the reader chose.


def bond(index: float) -> float:
    """How tightly two games are bound: `5 - log8(1 + index)`, unrounded.

    The index's own tiers, counted from the top: 5 for one game, about 3.9 a piece apart,
    3.0 a mechanism apart, 2.0 one rule apart and 1.0 eight rules apart. Positive for
    every index below 32,767, so every line the map can draw has a width and a weight.
    """
    return BOND_TOP - math.log(1.0 + index) / math.log(relate.SIMILARITY_RATIO)


def _similar(readings: dict[tuple[str, str], Any]) -> list[dict[str, Any]]:
    """One row per node pair whose index is within the last of `LEVELS`, nearest first.

    `level` is the first cut that draws the row, so a page showing a cut shows the rows
    whose level is at most it. `words` is the index's own sentence for the pair.
    """
    rows: list[dict[str, Any]] = []
    for (first, second), record in readings.items():
        reading = _similarity(record)
        index = reading.index
        if index > LEVELS[-1]:
            continue
        rows.append({
            "source": first,
            "target": second,
            "index": index,
            "bond": round(bond(index), 3),
            "level": next(cut for cut in LEVELS if index <= cut),
            "words": reading.sentence,
        })
    return sorted(rows, key=lambda row: (row["index"], row["source"], row["target"]))


def _levels(rows: list[dict[str, Any]], nodes: list[str]) -> list[dict[str, Any]]:
    """Each cut the reader may choose, with how many lines it draws and games it leaves.

    The page prints these counts rather than counting what it drew, so the number beside
    the slider is the server's.
    """
    found: list[dict[str, Any]] = []
    for cut in LEVELS:
        drawn = [row for row in rows if row["level"] <= cut]
        touched = {end for row in drawn for end in (row["source"], row["target"])}
        found.append({
            "cut": cut,
            "lines": len(drawn),
            "alone": sum(1 for node in nodes if node not in touched),
            "default": cut == SIMILAR,
        })
    return found


def _lines(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """The default cut's lines, each by its unrounded bond: what the map is built on."""
    return {
        (row["source"], row["target"]): bond(row["index"])
        for row in rows
        if row["index"] <= SIMILAR
    }


def _partition(lines: Mapping[tuple[str, str], float]) -> list[list[str]]:
    """The communities these lines form, weighted by their bond."""
    from chess_variant_sandbox.web import communities

    return communities.partition(lines)


def _centre(part: list[str], readings: dict[tuple[str, str], Any]) -> str:
    """The member with the least total index to the rest, the hash breaking a tie."""
    return min(
        part,
        key=lambda digest: (
            sum(
                _similarity(readings[_key(digest, other)]).index
                for other in part
                if other != digest
            ),
            digest,
        ),
    )


def _community_id(centre: str) -> str:
    """A community's id, from its centre game, so it never collides with a node's."""
    return f"community-{centre}"


def _communities(
    parts: list[list[str]],
    centres: list[str],
    nodes: dict[str, list[Game]],
    built: list[Node],
) -> list[dict[str, Any]]:
    """Each community with its name, members, centre and where its members average out.

    **The name is a rule where one is the community's own**: a rule at least
    `DISTINCT_IN` of its members involve and at most `DISTINCT_OUT` of the other joined
    games do, in `relate`'s words. Where several are, the one most members hold wins, then
    the one fewest others hold. Where none is -- the chess-like community differs from
    the rest by nothing it has -- it is named *near* its centre game. Two communities
    named by one rule take their centre's name after it.
    """
    at = {node.hash: node for node in built}
    rules = {
        digest: relate.rules_involved(relate.normal_form(nodes[digest][0].spec))
        for part in parts
        for digest in part
    }
    joined = sorted(rules)
    found: list[dict[str, Any]] = []
    for part, centre in zip(parts, centres, strict=True):
        rule = _distinctive(part, joined, rules)
        found.append({
            "id": _community_id(centre),
            "words": None if rule is None else _rule_words(rule),
            "members": list(part),
            "names": [at[digest].name for digest in part],
            "centre": centre,
            "size": len(part),
            "x": round(sum(at[digest].x for digest in part) / len(part), 3),
            "y": round(sum(at[digest].y for digest in part) / len(part), 3),
        })

    alike: dict[str, int] = defaultdict(int)
    for community in found:
        if community["words"]:
            alike[community["words"]] += 1
    for community in found:
        centre_name = at[community["centre"]].name
        words = community["words"]
        community["name"] = (
            f"near {centre_name}" if not words
            else f"{words} ({centre_name})" if alike[words] > 1
            else words
        )
    return found


def _distinctive(
    part: list[str], joined: list[str], rules: Mapping[str, frozenset[str]]
) -> str | None:
    """The rule this community holds and the other joined games mostly do not, or none."""
    inside = set(part)
    outside = [digest for digest in joined if digest not in inside]
    candidates: list[tuple[float, float, str]] = []
    for rule in sorted(set().union(*(rules[digest] for digest in part))):
        held = sum(rule in rules[digest] for digest in part) / len(part)
        others = (
            sum(rule in rules[digest] for digest in outside) / len(outside)
            if outside else 0.0
        )
        if held >= DISTINCT_IN and others <= DISTINCT_OUT:
            candidates.append((-held, others, rule))
    return min(candidates)[2] if candidates else None


def _rule_words(key: str) -> str:
    """One of `relate.rules_involved`'s keys in `relate`'s own words for it.

    Most are `rules` keys and read straight from `relate.RULE_WORDS`. The six declared
    elsewhere are the board's joined edges and walls, a lattice, who may take whom, a spy
    (schema 27) and speeds (schema 28).
    """
    if key == "board.wrap":
        return relate.RULE_WORDS["wrap"]
    if key == "board.walls":
        return relate.RULE_WORDS["obstacles"]
    if key == "piece.captures":
        return KIND_WORDS["capture"]
    if key == "piece.spy":
        return "a spy whose loss turns the game into rounds"
    if key == "piece.speed":
        return "pieces of different speeds, the faster moving first in a round"
    if key.startswith("board.lattice."):
        return f"a {key.removeprefix('board.lattice.')} board"
    return relate.RULE_WORDS.get(key) or f"the {key.replace('_', ' ')} rule"


# What each kind of rule is, in words a reader who has never seen this project can act
# on. `relate.KINDS` are the schema's own names -- *wild*, *hand*, *valuation* -- and a
# legend that printed them would be the maintainer's shorthand on a public map. The words
# live here rather than on the page because the payload is the map's whole interface, and
# a page that wrote its own would be a second opinion about what a kind is.
#
# Exhaustive over `relate.KINDS`, which a test holds: a kind added to the schema fails
# that test rather than reaching a legend unnamed.
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


def _kinds(links: list[Link]) -> list[dict[str, Any]]:
    """The legend: every kind of rule, in `relate`'s order, in words, with what it holds.

    Every kind is listed whether or not the corpus has a link of it, because a chip that
    appears and disappears with the corpus is a control that moves under the reader.
    """
    counted: dict[str, int] = defaultdict(int)
    for link in links:
        for kind in link.kinds:
            counted[kind] += 1
    return [
        {"kind": kind, "words": KIND_WORDS.get(kind, kind), "links": counted.get(kind, 0)}
        for kind in relate.KINDS
    ]


# ------------------------------------------------------------------------ the layout
#
# Computed here and shipped as coordinates, so the page can pin the picture in a test and
# it looks the same on every visit. `docs/archive/plan-graph-map.md` S2 lays it out by community:
# each community is drawn on its own as a knot, by a force layout whose lines pull by
# their bond; the knots are spread over one page, pulled together by the lines between
# them and held apart by their size; each is turned to face its neighbours; and the games
# no line reaches go in the far row underneath.

Point = tuple[float, float]


def _layout(
    nodes: list[str],
    lines: Mapping[tuple[str, str], float],
    parts: list[list[str]],
) -> dict[str, Point]:
    """Where every node sits, in layout units.

    `lines` are the default cut's similarity lines by their bond, and `parts` the
    communities they form. Every walk is in hash order and the one random draw is the
    layout's own generator, so the same library lays out the same way twice, in whatever
    order it arrived.

    **Each community is nearer its own centre than any other's.** Two knots are held
    further apart than twice the reach of the larger, so no member can sit nearer the
    middle of another community than the middle of its own -- which is what makes a
    community a separate knot at the whole-map view rather than a colour on a hairball.
    """
    knots = [_knot(part, lines) for part in parts]
    reach = [max(math.hypot(x, y) for x, y in knot.values()) for knot in knots]
    centres = _spread(reach, _across(parts, lines))
    knots = _turned(knots, centres, lines)

    placed: dict[str, Point] = {}
    for knot, (across, down) in zip(knots, centres, strict=True):
        for digest, (x, y) in knot.items():
            placed[digest] = (across + x, down + y)
    left = min((x for x, _y in placed.values()), default=0.0)
    bottom = max((y for _x, y in placed.values()), default=0.0)
    alone = sorted(node for node in nodes if node not in placed)
    for index, digest in enumerate(alone):
        placed[digest] = (left + index * ROW_STEP, bottom + COMPONENT_GAP + ROW_STEP)
    return placed


def _knot(part: list[str], lines: Mapping[tuple[str, str], float]) -> dict[str, Point]:
    """One community on its own, centred on its members' mean.

    Fruchterman and Reingold's forces: every two members push apart, every line pulls its
    ends together in proportion to its bond against the community's mean bond, so a
    thick line is a short one. The start is a circle in hash order, nudged by the layout's
    own generator, and a last pass holds every two members `MIN_SEPARATION` apart.
    """
    members = sorted(part)
    if len(members) == 1:
        return {members[0]: (0.0, 0.0)}
    inside = set(members)
    held = {
        pair: weight for pair, weight in sorted(lines.items())
        if pair[0] in inside and pair[1] in inside
    }
    mean = sum(held.values()) / len(held) if held else 1.0
    draw = random.Random(LAYOUT_SEED)
    radius = max(RING / 2, RING * len(members) / (2 * math.pi))
    spot: dict[str, list[float]] = {}
    for index, digest in enumerate(members):
        turn = 2 * math.pi * index / len(members)
        spot[digest] = [
            radius * math.cos(turn) + draw.uniform(-0.1, 0.1) * RING,
            radius * math.sin(turn) + draw.uniform(-0.1, 0.1) * RING,
        ]

    heat = RING
    for _ in range(KNOT_PASSES):
        shift = {digest: [0.0, 0.0] for digest in members}
        for first, second in combinations(members, 2):
            dx, dy, apart = _towards(spot[first], spot[second])
            push = RING * RING / apart
            shift[first][0] += dx * push
            shift[first][1] += dy * push
            shift[second][0] -= dx * push
            shift[second][1] -= dy * push
        for (first, second), weight in held.items():
            dx, dy, apart = _towards(spot[first], spot[second])
            pull = apart * apart / RING * (weight / mean)
            shift[first][0] -= dx * pull
            shift[first][1] -= dy * pull
            shift[second][0] += dx * pull
            shift[second][1] += dy * pull
        for digest in members:
            size = math.hypot(*shift[digest])
            if size > 0:
                step = min(size, heat) / size
                spot[digest][0] += shift[digest][0] * step
                spot[digest][1] += shift[digest][1] * step
        heat = max(heat * 0.98, RING / 100)

    _separated(members, spot, MIN_SEPARATION)
    middle_x = sum(spot[digest][0] for digest in members) / len(members)
    middle_y = sum(spot[digest][1] for digest in members) / len(members)
    return {
        digest: (spot[digest][0] - middle_x, spot[digest][1] - middle_y)
        for digest in members
    }


def _towards(one: list[float], other: list[float]) -> tuple[float, float, float]:
    """The unit step from `other` to `one`, and the distance, never zero."""
    dx, dy = one[0] - other[0], one[1] - other[1]
    apart = math.hypot(dx, dy)
    if apart < 1e-9:
        return 1.0, 0.0, 1e-9
    return dx / apart, dy / apart, apart


def _separated(names: list[str], spot: dict[str, list[float]], least: float) -> None:
    """Push every two of these apart until none is nearer than `least`, in hash order."""
    for _ in range(500):
        moved = False
        for first, second in combinations(names, 2):
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


def _across(
    parts: list[list[str]], lines: Mapping[tuple[str, str], float]
) -> dict[tuple[int, int], float]:
    """The total bond of the lines between each two communities, by their places."""
    where = {digest: number for number, part in enumerate(parts) for digest in part}
    found: dict[tuple[int, int], float] = defaultdict(float)
    for (first, second), weight in sorted(lines.items()):
        mine, theirs = where.get(first), where.get(second)
        if mine is None or theirs is None or mine == theirs:
            continue
        found[(min(mine, theirs), max(mine, theirs))] += weight
    return dict(sorted(found.items()))


def _spread(reach: list[float], across: Mapping[tuple[int, int], float]) -> list[Point]:
    """Where each community's middle sits: close to those it has lines to, on one page.

    Communities start on a sunflower spiral, biggest in the middle. Each pass pulls two
    communities with lines between them together, pulls every one gently to the middle
    so the page holds them all, and pushes apart any two nearer than they may be.
    """
    count = len(reach)
    if not count:
        return []
    size = [one + NODE_RADIUS for one in reach]
    step = 2 * sum(size) / count

    def least(first: int, second: int) -> float:
        return max(
            size[first] + size[second] + COMPONENT_GAP,
            2 * max(reach[first], reach[second]) + MIN_SEPARATION,
        )

    spot = [
        [step * math.sqrt(index) * math.cos(index * _GOLDEN),
         step * math.sqrt(index) * math.sin(index * _GOLDEN)]
        for index in range(count)
    ]
    strongest = max(across.values(), default=1.0)
    for _ in range(SPREAD_PASSES):
        for (first, second), weight in across.items():
            dx, dy, apart = _towards(spot[second], spot[first])
            slack = apart - least(first, second)
            if slack <= 0:
                continue
            pull = slack * 0.1 * weight / strongest
            spot[first][0] += dx * pull
            spot[first][1] += dy * pull
            spot[second][0] -= dx * pull
            spot[second][1] -= dy * pull
        for one in spot:
            one[0] *= 0.98
            one[1] *= 0.98
        _cleared(spot, least)
    for _ in range(1000):
        if not _cleared(spot, least):
            break
    return [(one[0], one[1]) for one in spot]


# The golden angle, which spreads a spiral's points evenly round it.
_GOLDEN = math.pi * (3 - math.sqrt(5))


def _cleared(spot: list[list[float]], least: Callable[[int, int], float]) -> bool:
    """One pass pushing apart every two communities nearer than they may be; any moved."""
    moved = False
    for first, second in combinations(range(len(spot)), 2):
        dx, dy, apart = _towards(spot[first], spot[second])
        want = least(first, second)
        if apart >= want - 1e-9:
            continue
        half = (want - apart) / 2 + 1e-6
        spot[first][0] += dx * half
        spot[first][1] += dy * half
        spot[second][0] -= dx * half
        spot[second][1] -= dy * half
        moved = True
    return moved


def _turned(
    knots: list[dict[str, Point]],
    centres: list[Point],
    lines: Mapping[tuple[str, str], float],
) -> list[dict[str, Point]]:
    """Each community turned about its middle so its lines to others are shortest.

    `TURNS` turns are tried, biggest community first and then round again, and a turn is
    taken only when it shortens the total bond-weighted length of the lines leaving the
    community -- so a community with none keeps the turn it was drawn at. Turning a knot
    moves no member nearer another member and does not move its middle.
    """
    knots = [dict(knot) for knot in knots]
    where = {digest: number for number, knot in enumerate(knots) for digest in knot}

    def at(digest: str) -> Point:
        number = where[digest]
        x, y = knots[number][digest]
        return centres[number][0] + x, centres[number][1] + y

    leaving: list[list[tuple[str, str, float]]] = [[] for _ in knots]
    for (first, second), weight in sorted(lines.items()):
        if first in where and second in where and where[first] != where[second]:
            leaving[where[first]].append((first, second, weight))
            leaving[where[second]].append((second, first, weight))

    for _ in range(2):
        for number, knot in enumerate(knots):
            if not leaving[number]:
                continue
            best, cost = knot, _length(knot, centres[number], leaving[number], at)
            for turn in range(1, TURNS):
                angle = 2 * math.pi * turn / TURNS
                cos, sin = math.cos(angle), math.sin(angle)
                tried = {
                    digest: (x * cos - y * sin, x * sin + y * cos)
                    for digest, (x, y) in knot.items()
                }
                length = _length(tried, centres[number], leaving[number], at)
                if length < cost - 1e-9:
                    best, cost = tried, length
            knots[number] = best
    return knots


def _length(
    knot: dict[str, Point],
    centre: Point,
    leaving: list[tuple[str, str, float]],
    at: Callable[[str], Point],
) -> float:
    """The bond-weighted length of the lines leaving one knot drawn this way round."""
    total = 0.0
    for mine, theirs, weight in leaving:
        x, y = knot[mine]
        total += weight * math.dist((centre[0] + x, centre[1] + y), at(theirs))
    return total


def _bounds(nodes: list[Node]) -> dict[str, float]:
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


def _node(
    group: list[Game], degree: int, far: bool, spot: tuple[float, float]
) -> Node:
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
        far=far,
        score=_card(group),
        staleness=game.staleness,
        exhibit=exhibit(game),
        x=spot[0],
        y=spot[1],
    )


# ------------------------------------------------------------------------- the cache


class _Cache:
    """Every pair and every twin key this machine has worked out, across restarts.

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
        self.drawn = 0
        self.spanned = 0
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
            if isinstance(value, dict)
        }

    def write(self) -> None:
        if not self._dirty:
            return
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
        self.specs[game.hash] = answer
        self.compiled += 1
        self._dirty = True
        return answer

    def pair(self, first: Game, second: Game) -> dict[str, Any]:
        """What these two specs differ in, from the file or from `relate`.

        The differences themselves are kept only for a pair near enough to be drawn.
        Past three rules the count is all the map ever asks for, and holding nine
        sentences for each of two thousand pairs would be a megabyte nobody reads.
        """
        key = ":".join(_key(first.hash, second.hash))
        held = self.pairs.get(key)
        if held is not None and "similarity" in held:
            return held
        relation = relate.relate(first.spec, second.spec)
        record: dict[str, Any] = {
            "units": relation.units,
            "similarity": self._similarity(first, second),
        }
        if relation.units <= NEAR:
            record["differences"] = [one.as_dict() for one in relation.differences]
        if held is not None:
            record = {**held, **record}
        self.pairs[key] = _ordered(record)
        self.computed += 1
        self._dirty = True
        return self.pairs[key]

    def _similarity(self, first: Game, second: Game) -> dict[str, int]:
        """The six tier counts for this pair, and zero for two spellings of one game.

        Only the counts are kept. The index is arithmetic over them under
        `relate.SIMILARITY_RATIO`, so a changed ratio re-reads this file rather than
        dropping it.
        """
        if self.specs.get(first.hash, "first") == self.specs.get(second.hash, "second"):
            return relate.Similarity(0, 0, 0, 0, 0, 0).counts
        return relate.similarity(first.spec, second.spec).counts

    def swept(self, first: str, second: str) -> bool:
        """Whether the operator draws for this pair of *nodes* have been made.

        *Made and found nothing* is an answer and is stored as one, because otherwise the
        thirty-three links no operator reaches would be redrawn on every call -- which is
        most of the sweep's cost and all of its waste.
        """
        return "reach" in self.pairs.get(":".join(_key(first, second)), {})

    def reach(self, first: str, second: str) -> dict[str, Any] | None:
        """The operator sentence held for this pair of nodes, if there is one."""
        held = self.pairs.get(":".join(_key(first, second)), {}).get("reach")
        return dict(held) if isinstance(held, dict) else None

    def record(self, pair: tuple[str, str], reach: dict[str, Any] | None) -> None:
        """Keep what a sweep found for one pair, a miss included."""
        self.pairs.setdefault(":".join(pair), {})["reach"] = _ordered(reach)
        self._dirty = True

    def between(self, first: str, second: str) -> dict[str, Any] | None:
        """The game in between this pair of nodes, or the sentence saying there is none.

        Kept for the same reason the operator sentence is: the answer costs a compile of
        the game it would build, which is a fifth of a second over the corpus's gaps, and
        it is the same answer every time. A refusal is stored as an answer too, so the
        forty-five gaps nothing can be built between are not tried again on every call.
        """
        held = self.pairs.get(":".join(_key(first, second)), {}).get("between")
        return dict(held) if isinstance(held, dict) else None

    def span(self, first: str, second: str, made: dict[str, Any]) -> None:
        """Keep what `designing.in_between` said about one pair."""
        self.pairs.setdefault(":".join(_key(first, second)), {})["between"] = _ordered(made)
        self.spanned += 1
        self._dirty = True

    def distance(self, first: Game, second: Game) -> float:
        """`spec/distance.py`'s number for this pair, kept on the pair's own record."""
        from chess_variant_sandbox.spec.distance import distance

        record = self.pair(first, second)
        if "distance" not in record:
            record["distance"] = distance(first.spec, second.spec)
            self._dirty = True
        return float(record["distance"])


# ------------------------------------------------------------------ filling the cache


def _main(argv: list[str] | None = None) -> int:
    """`python -m chess_variant_sandbox.web.graphing [root]` -- fill the cache, sentences and all.

    The sweep is twenty seconds against the rest of the build's five, so no request pays
    for it and something else has to. This is that something: a deploy's load step, or a
    person who wants the sentences on their own map. It writes `runs/graph/pairs.json`
    and nothing else, and a second run has nothing to do.
    """
    import argparse
    import time

    from chess_variant_sandbox.web import library

    parser = argparse.ArgumentParser(description=str(_main.__doc__ or "").splitlines()[0])
    parser.add_argument("root", nargs="?", default=".", type=Path)
    parsed = parser.parse_args(argv)
    root = parsed.root

    games = [
        game
        for game in library.discover(root / "rulesets", root / "runs", root / "archive")
        if game.source in ("committed", "archived")
    ]
    started = time.perf_counter()
    built = build(games, root / "runs", sentences=True)
    seconds = time.perf_counter() - started
    print(
        f"{built.counts['nodes']} games, {built.counts['oneRule']} one-rule links, "
        f"{built.counts['sentences']} of them with an operator sentence "
        f"({built.computed} pairs compared, {built.compiled} specs compiled, "
        f"{built.drawn} draws, {seconds:.1f}s)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the cache filler, not a library path
    raise SystemExit(_main())
