"""The library as a map: which games merge, which are alike, the communities, where each sits.

`tests/test_relate.py` holds what two specs differ in, `tests/test_likeness.py` how alike
two are, `tests/test_families.py` the mutual graph and the communities, and
`tests/test_mapping.py` the picture. This holds what the map's payload makes of them
(`docs/plan-map.md` M4): one node per game however many files spell it, a line between
two games each among the other's nearest, saying what both have and what differs, each
game's nearest, the communities with where they are drawn, the picture in layout units,
and the memory a server keeps of it.

The specs are written here rather than read from `rulesets/`, because what is under test
is the shape of a library and not what any game happens to be. Two classes at the end read
the committed corpus and today's library, and are marked slow: they compile sixty specs
and more.

**What M4 retired, and why no test of it is kept.** The rule-step lines (one rule apart,
the unbridged gaps, the faint nearest-relative line), the operator sentence a deploy-time
sweep drew onto them, *the game in between* a gap, the similarity rows with their cuts and
the communities they formed, and the layout by community with its far row
(`TestWhichLinksAreDrawn`, `TestTheOperatorSentence`, `TestTheNearestRelative`,
`TestTheGroupsAreTheCommunities`, `TestTheGameInBetweenAGap`, `TestTheCacheFiller`,
`TestTheSimilarityIndex`, `TestTheSimilarityLines`, `TestTheLayoutByCommunity` and their
corpus pins) held code that no longer exists; `docs/plan-map.md` §1 and §4 say why each
layer went. What each held that still holds -- twins, lineage, the scores on a line, the
scale, the sizes, the pair cache's versions, the bounds -- is held below, on the new
lines.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from retired_specs import REBUILT, RETIRED_HASHES

from chess_variant_sandbox.spec import relate, validate_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web import families, graphing, library, mapping

# A three-piece game with a promoting walker, so promotion and the ending rules all have
# something to be about. Every spec below is a copy of this with one thing changed.
BASE: dict[str, Any] = {
    "schema_version": 2,
    "name": "base",
    "board": {"width": 6, "height": 6},
    "pieces": [
        {
            "id": "K",
            "name": "chief",
            "royal": True,
            "moves": [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4"}],
        },
        {
            "id": "W",
            "name": "walker",
            "value": 1,
            "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 2, "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1], [-1, 1]], "mode": "capture"},
            ],
        },
        {
            "id": "R",
            "name": "tower",
            "value": 5,
            "moves": [{"kind": "slide", "vectors": [[1, 0]], "symmetry": "rotate4"}],
        },
    ],
    "setup": {
        "mirror": True,
        "placements": [
            {"piece": "K", "squares": ["c1"]},
            {"piece": "R", "squares": ["a1", "f1"]},
            {"piece": "W", "squares": ["a2", "b2", "c2", "d2", "e2", "f2"]},
        ],
    },
    "rules": {
        "win": "checkmate",
        "promotion": {"piece": "W", "rank": 5, "to": ["R"]},
    },
}


def spec(name: str, **rules: Any) -> dict[str, Any]:
    """`BASE` under a new name, with these `rules` keys set."""
    made = copy.deepcopy(BASE)
    made["name"] = name
    made["rules"].update(rules)
    return made


def moved(name: str, piece: str) -> dict[str, Any]:
    """`BASE` with one piece given a longer reach, which is one rule on that piece."""
    made = copy.deepcopy(BASE)
    made["name"] = name
    for one in made["pieces"]:
        if one["id"] == piece:
            one["moves"] = [{"kind": "slide", "vectors": [[1, 1]], "symmetry": "rotate4"}]
    return made


def far(name: str) -> dict[str, Any]:
    """A game nothing in this file is within three rules of."""
    return {
        "schema_version": 2,
        "name": name,
        "board": {"width": 4, "height": 9},
        "pieces": [
            {
                "id": "K",
                "name": "chief",
                "royal": True,
                "moves": [{"kind": "leap", "vectors": [[1, 0]], "symmetry": "rotate4"}],
            },
            {
                "id": "H",
                "name": "horse",
                "value": 3,
                "moves": [{"kind": "leap", "vectors": [[1, 2]], "symmetry": "dihedral8"}],
            },
        ],
        "setup": {
            "mirror": True,
            "placements": [
                {"piece": "K", "squares": ["b1"]},
                {"piece": "H", "squares": ["a1", "d1"]},
            ],
        },
        "rules": {"win": "checkmate", "max_plies": 120, "check_limit": 2},
    }


def walked(name: str, limit: int) -> dict[str, Any]:
    """A game off the second family: the walker's move changed, and a `check_limit`.

    So `walks`, `walks-one` and `walks-two` differ from each other in the same rule the
    `base` family does, which is what makes two groups that would be named alike.
    """
    made = moved(name, "W")
    made["rules"]["check_limit"] = limit
    return made


def game(made: dict[str, Any], source: str = "committed") -> library.Game:
    return library.Game(
        hash=spec_hash(made),
        name=str(made["name"]),
        description="",
        path=Path(f"{made['name']}.json"),
        source=source,
        spec=made,
    )


def built(specs: list[dict[str, Any]], runs: Path, **kwargs: Any) -> graphing.Graph:
    return graphing.build([game(one) for one in specs], runs, **kwargs)


def named(graph: graphing.Graph) -> dict[str, str]:
    """Each node's hash, by the name it carries, so a test can read a link."""
    return {node.name: node.hash for node in graph.nodes}


def line_between(graph: graphing.Graph, first: str, second: str) -> graphing.Line | None:
    """The line between the two games with these names, or `None` where none joins them."""
    ids = named(graph)
    pair = {ids[first], ids[second]}
    for line in graph.neighbours:
        if {line.source, line.target} == pair:
            return line
    return None


def family_map(graph: graphing.Graph, specs: list[dict[str, Any]]) -> families.FamilyMap:
    """`families.build` over the same games, cold: the relation the payload must agree with."""
    by_hash = {spec_hash(one): one for one in specs}
    ids = [node.hash for node in graph.nodes]
    return families.build(
        ids,
        [relate.normal_form(by_hash[digest]) for digest in ids],
        [node.name for node in graph.nodes],
    )


class TestTheSpecsUnderTest:
    def test_every_one_of_them_is_a_game(self) -> None:
        """A spec the validator refuses would make every count below meaningless."""
        for made in (BASE, spec("a", check_limit=3), moved("b", "R"), far("c"), *LIBRARY):
            assert validate_spec(made) == []


def spelt_out(name: str) -> dict[str, Any]:
    """`BASE` with who-may-take-whom written out on every piece.

    Saying *this piece may take all three* is saying what the game does anyway, so the two
    files compile to the same tables. No comparison of two written descriptions can know
    that, and the one twin pair the library had (`wolves` and `wolves-immune`, retired by
    Q876) was this shape.
    """
    made = spec(name)
    made["schema_version"] = 3
    for one in made["pieces"]:
        one["captures"] = [other["id"] for other in made["pieces"]]
    return made


class TestADeliberateExample:
    """Q141: a game kept on purpose for what it shows says so on the map as on its card."""

    def test_a_kept_game_s_dot_carries_its_sentence(self, tmp_path: Path) -> None:
        kept = spec("kept")
        kept["provenance"] = {"exhibit": "Kept on purpose as a test."}
        drafted = spec("drafted", check_limit=3)
        drafted["provenance"] = {"exhibit": "A draft calling itself an example."}
        games = [game(kept, "archived"), game(drafted, "designed"),
                 game(spec("plain", check_limit=4))]
        said = {node["name"]: node["exhibit"]
                for node in graphing.build(games, tmp_path).as_dict()["nodes"]}
        # `library.exhibit` decides, so a draft is not believed about itself.
        assert said == {"kept": "Kept on purpose as a test.", "drafted": None, "plain": None}


class TestTwins:
    """Two files, one game. `docs/archive/plan-game-graph.md` §1 S7."""

    def test_two_spellings_of_one_game_are_one_node(self, tmp_path: Path) -> None:
        """Merged on what the compiler produced, which is the only reading that settles it."""
        plain = spec("plain")
        plain["schema_version"] = 3
        graph = built([plain, spelt_out("spelt"), spec("other", check_limit=3)], tmp_path)
        assert relate.units(plain, spelt_out("spelt")) == 1
        assert graph.counts["specs"] == 3
        assert graph.counts["nodes"] == 2
        assert graph.counts["twins"] == 1
        merged = next(node for node in graph.nodes if len(node.hashes) > 1)
        assert sorted(merged.names) == ["plain", "spelt"]
        assert merged.hash in merged.hashes

    def test_a_merged_node_keeps_the_shorter_of_its_two_readings(
        self, tmp_path: Path
    ) -> None:
        """One spelling is one rule from a third game and the other is two.

        The node is one game, so the line to that game says one rule -- a rule declared
        the long way is still that rule.
        """
        plain = spec("plain")
        plain["schema_version"] = 3
        spelt = spelt_out("spelt")
        other = spec("other", check_limit=3)
        other["schema_version"] = 3
        assert relate.units(plain, other) == 1
        assert relate.units(spelt, other) == 2

        graph = built([plain, spelt, other, far("away")], tmp_path)
        assert graph.counts["nodes"] == 3
        merged = next(node for node in graph.nodes if len(node.hashes) > 1)
        line = next(
            one for one in graph.neighbours
            if {one.source, one.target} == {merged.hash, named(graph)["other"]}
        )
        assert line.units == 1


# Two families and two far games, the old similarity tests' library: the `check_limit`
# family (with the walker changed on two of them), the one round `base`, and two games
# alike only to each other.
LIBRARY = [
    BASE,
    spec("one", check_limit=1),
    spec("two", check_limit=2),
    spec("three", check_limit=3),
    spec("wide", stalemate="loss"),
    spec("wider", stalemate="win"),
    moved("walks", "W"),
    walked("walks-one", 1),
    walked("walks-two", 2),
    moved("towers", "R"),
    far("away"),
    {**far("yonder"), "rules": {"win": "checkmate", "max_plies": 100}},
]


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> graphing.Graph:
    """`LIBRARY`'s map, built once for the module."""
    return built(LIBRARY, tmp_path_factory.mktemp("drawn"))


class TestTheLines:
    """M4 of `docs/plan-map.md`: a line is two games each among the other's nearest.

    The rule-step lines, the gaps and the faint nearest-relative lines this class's
    predecessors held (`TestWhichLinksAreDrawn`, `TestTheNearestRelative`,
    `TestTheSimilarityLines`) left the map with M4; the relation is `spec/likeness.py`'s
    and `tests/test_families.py` holds the mutual graph itself. What is held here is that
    the payload draws that graph and says what each line is.
    """

    def test_the_lines_are_the_mutual_nearest_by_likeness(self, drawn: graphing.Graph) -> None:
        expected = family_map(drawn, LIBRARY).lines
        assert {(line.source, line.target): line.likeness for line in drawn.neighbours} == {
            pair: round(weight, 3) for pair, weight in expected.items()
        }
        assert drawn.counts["lines"] == len(expected) > 0

    def test_a_line_says_what_both_have_in_words(self, drawn: graphing.Graph) -> None:
        """The rarest words both games have, as `families.shared_words` says them."""
        assert any(line.shares for line in drawn.neighbours)
        for line in drawn.neighbours:
            assert len(line.shares) <= graphing.SHARES
            for said in line.shares:
                # Words a reader can read, never a token of `likeness.words`.
                assert ":" not in said and "(" not in said.split()[0], said

    def test_a_line_within_three_rules_spells_out_each_difference(
        self, drawn: graphing.Graph
    ) -> None:
        near = [line for line in drawn.neighbours if 0 < line.units <= graphing.NEAR]
        assert near
        for line in near:
            assert line.differences and line.sentence
            assert line.kind in line.kinds
            assert line.sentence == "; ".join(one["sentence"] for one in line.differences)

    def test_past_three_rules_a_pair_says_only_how_many(self, tmp_path: Path) -> None:
        """A list of nine differences is a ranking, not a reading, so it is not kept."""
        cache = graphing._Cache(tmp_path / graphing.CACHE_FILE)
        record = cache.pair(game(BASE), game(far("away")))
        assert record["units"] > graphing.NEAR
        assert "differences" not in record

    def test_a_line_names_every_kind_on_it_and_not_only_the_first(
        self, tmp_path: Path
    ) -> None:
        graph = built(
            [BASE, spec("two", check_limit=3, must_capture=True), far("away")], tmp_path
        )
        line = line_between(graph, "base", "two")
        assert line is not None
        assert set(line.kinds) == {
            relate.RULE_KINDS["check_limit"], relate.RULE_KINDS["must_capture"]
        }


class TestTheGamesLikeThisOne:
    """Each node's `like`: its nearest by likeness, with what each pair shares (M4)."""

    def test_like_is_each_game_s_nearest_in_the_relation_s_order(
        self, drawn: graphing.Graph
    ) -> None:
        found = family_map(drawn, LIBRARY)
        for node in drawn.nodes:
            assert [row["hash"] for row in node.like] == list(found.neighbours[node.hash])
            assert len(node.like) == min(families.MUTUAL, len(drawn.nodes) - 1)
            likeness = [row["likeness"] for row in node.like]
            assert likeness == sorted(likeness, reverse=True)

    def test_a_row_says_whether_a_line_joins_the_two(self, drawn: graphing.Graph) -> None:
        joined = {(line.source, line.target) for line in drawn.neighbours}
        for node in drawn.nodes:
            for row in node.like:
                pair = tuple(sorted((node.hash, row["hash"])))
                assert row["line"] == (pair in joined)
        assert any(row["line"] for node in drawn.nodes for row in node.like)


class TestTheCommunities:
    """What the page colours, outlines and folds: `families`' communities, with parts."""

    def test_they_partition_every_game_that_is_not_unusual(self, drawn: graphing.Graph) -> None:
        found = family_map(drawn, LIBRARY)
        seen: set[str] = set()
        for community in drawn.communities:
            assert not seen & set(community["members"])
            seen |= set(community["members"])
            assert community["size"] == len(community["members"]) >= 2
        for node in drawn.nodes:
            assert node.community == found.family_of[node.hash]
            assert node.unusual == (node.community is None)
            assert (node.hash in seen) == (not node.unusual)
        assert [one["hash"] for one in drawn.unusual] == [
            digest for digest, _ in found.unusual
        ]
        assert drawn.counts["unusual"] == sum(node.unusual for node in drawn.nodes)

    def test_each_carries_its_name_words_centre_parts_and_folded_place(
        self, drawn: graphing.Graph
    ) -> None:
        at = {node.hash: node for node in drawn.nodes}
        payload = json.loads(json.dumps(drawn.as_dict()))
        assert payload["communities"]
        for community in payload["communities"]:
            assert {"id", "name", "words", "members", "names", "centre", "size", "parts",
                    "x", "y"} <= set(community)
            assert community["names"] == [at[digest].name for digest in community["members"]]
            # Every member in one part, and the centre's part first.
            assert sorted(d for part in community["parts"] for d in part) == sorted(
                community["members"]
            )
            assert community["centre"] in community["parts"][0]
            home = community["parts"][0]
            assert community["x"] == pytest.approx(
                sum(at[digest].x for digest in home) / len(home), abs=1e-3
            )
            # Named by what marks it out, or near its centre.
            if community["words"] is None:
                assert community["name"].startswith("near ")
            else:
                assert community["name"].startswith(community["words"])

    def test_a_community_drawn_in_two_places_has_two_parts(self) -> None:
        """Single linkage over the fitted picture, at `PART_REACH`."""
        reach = graphing.PART_REACH
        placed = {
            "a": (0.0, 0.0), "b": (reach * 0.9, 0.0), "c": (reach * 1.8, 0.0),
            "d": (reach * 5, 0.0), "e": (reach * 5, reach * 0.5),
        }
        assert graphing._parts(list(placed), placed) == [["a", "b", "c"], ["d", "e"]]
        assert graphing._parts(["a", "b"], placed) == [["a", "b"]]


class TestThePicture:
    """M3's picture, fitted to the page: `mapping`'s places in layout units."""

    def test_the_median_nearest_gap_is_the_ring(self) -> None:
        """Scaling moves no game against another; it only says how big the picture is."""
        grid = {f"g{x}{y}": (x * 7.0, y * 7.0) for x in range(4) for y in range(3)}
        fitted = graphing._fitted(grid)
        gaps = sorted(
            min(math.dist(fitted[a], fitted[b]) for b in fitted if b != a) for a in fitted
        )
        assert gaps[len(gaps) // 2] == pytest.approx(graphing.RING)

    def test_no_two_dots_are_drawn_nearer_than_the_least_separation(
        self, drawn: graphing.Graph
    ) -> None:
        placed = {"a": (0.0, 0.0), "b": (0.01, 0.0), "c": (10.0, 0.0), "d": (20.0, 0.0)}
        fitted = graphing._fitted(placed)
        assert math.dist(fitted["a"], fitted["b"]) >= graphing.MIN_SEPARATION - 1e-6
        spots = [(node.x, node.y) for node in drawn.nodes]
        assert min(
            math.dist(one, other)
            for index, one in enumerate(spots) for other in spots[index + 1:]
        ) >= graphing.MIN_SEPARATION - 1e-3

    def test_the_same_library_in_any_order_draws_the_same_payload(
        self, tmp_path: Path
    ) -> None:
        first = built(LIBRARY, tmp_path / "a")
        second = built(list(reversed(LIBRARY)), tmp_path / "b")
        assert json.dumps(first.as_dict()) == json.dumps(second.as_dict())

    def test_the_bounds_hold_every_node(self, drawn: graphing.Graph) -> None:
        bounds = drawn.bounds
        for node in drawn.nodes:
            assert bounds["minX"] <= node.x <= bounds["maxX"]
            assert bounds["minY"] <= node.y <= bounds["maxY"]
        _hold_the_span(bounds, drawn.nodes)

    def test_the_span_is_rounded_once_where_its_ends_round_apart(self) -> None:
        """Q744. Ends at -0.9994 and 10.0004 round to -0.999 and 10.0, 10.999 apart, while
        their span, 10.9998, rounds to 11.0. That pair is what failed on cloud Linux: the
        old check held the width to the rounded ends at a relative 1e-6, about 1e-5 here.
        """
        r = graphing.NODE_RADIUS
        nodes = [_At(0.0006 - (1.0 - r), 0.0), _At(9.0004 + (1.0 - r), 0.0)]
        bounds = graphing._bounds(nodes)
        assert (bounds["minX"], bounds["maxX"], bounds["width"]) == (-0.999, 10.0, 11.0)
        _hold_the_span(bounds, nodes)


class _At:
    """A node as far as `_bounds` reads one: where it is drawn."""

    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


def _hold_the_span(bounds: dict[str, float], nodes: Any) -> None:
    """The width and height are each the box's span, rounded once to the page's three places.

    `_bounds` rounds each of the four ends and the two spans separately, so a span and the
    difference of its rounded ends can be a thousandth apart, and which way a platform's
    floats fall decides whether they are (Q734 §3: the cloud-Linux failure). So a span is
    held exactly to the unrounded box rounded once, and to its rounded ends within that one
    thousandth (the 1e-9 is the float error of two three-place numbers subtracted).
    """
    r = graphing.NODE_RADIUS
    xs, ys = [node.x for node in nodes], [node.y for node in nodes]
    assert bounds["width"] == round((max(xs) + r) - (min(xs) - r), 3)
    assert bounds["height"] == round((max(ys) + r) - (min(ys) - r), 3)
    assert abs(bounds["width"] - (bounds["maxX"] - bounds["minX"])) <= 0.001 + 1e-9
    assert abs(bounds["height"] - (bounds["maxY"] - bounds["minY"])) <= 0.001 + 1e-9


def _memory(runs: Path) -> Path:
    return runs / families.MAP_FILE


class TestTheMemory:
    """`runs/graph/map.json`: the first build draws cold and writes it; later ones settle.

    M4's acceptance for the box: a server with no memory draws the map and leaves one, a
    restarted server settles from it and keeps every place when nothing changed, and the
    file is rewritten only when the map moved.
    """

    def test_the_first_build_writes_the_memory_and_the_next_keeps_it(
        self, tmp_path: Path
    ) -> None:
        first = built(LIBRARY, tmp_path)
        path = _memory(tmp_path)
        held = json.loads(path.read_text(encoding="utf-8"))
        assert held["version"] == families.MAP_VERSION
        assert held["layout"] == mapping.LAYOUT_VERSION
        assert set(held["games"]) == {node.hash for node in first.nodes}
        written = path.read_bytes(), path.stat().st_mtime_ns
        # A restarted server: nothing in this process remembers the map.
        graphing._MAPS.clear()
        again = built(LIBRARY, tmp_path)
        assert (path.read_bytes(), path.stat().st_mtime_ns) == written
        assert {node.hash: (node.x, node.y) for node in again.nodes} == {
            node.hash: (node.x, node.y) for node in first.nodes
        }
        assert again.communities == first.communities

    def test_a_second_request_is_answered_from_the_process(self, tmp_path: Path) -> None:
        built(LIBRARY, tmp_path)
        where = str(_memory(tmp_path).resolve())
        held = graphing._MAPS[where][2]
        built(list(reversed(LIBRARY)), tmp_path)
        assert graphing._MAPS[where][2] is held

    def test_a_new_game_settles_into_the_picture(self, tmp_path: Path) -> None:
        """The others keep their places but for a shuffle; the new one joins the memory."""
        first = built(LIBRARY, tmp_path)
        newcomer = spec("four", check_limit=4)
        second = built([*LIBRARY, newcomer], tmp_path)
        assert spec_hash(newcomer) in json.loads(
            _memory(tmp_path).read_text(encoding="utf-8")
        )["games"]
        before = {node.hash: (node.x, node.y) for node in first.nodes}
        after = {node.hash: (node.x, node.y) for node in second.nodes}
        moved = sorted(math.dist(before[digest], after[digest]) for digest in before)
        assert moved[len(moved) // 2] < 0.05 * first.bounds["width"]

    def test_a_memory_of_another_version_is_drawn_over_cold(self, tmp_path: Path) -> None:
        built(LIBRARY, tmp_path)
        path = _memory(tmp_path)
        held = json.loads(path.read_text(encoding="utf-8"))
        held["version"] = "m0"
        path.write_text(json.dumps(held), encoding="utf-8")
        graphing._MAPS.clear()
        cold = built(LIBRARY, tmp_path / "elsewhere")
        again = built(LIBRARY, tmp_path)
        assert json.loads(path.read_text(encoding="utf-8"))["version"] == families.MAP_VERSION
        assert json.dumps(again.as_dict()) == json.dumps(cold.as_dict())

    def test_a_runs_directory_that_cannot_be_written_still_draws(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_args: Any, **_kwargs: Any) -> bool:
            raise PermissionError("read-only")

        monkeypatch.setattr(families, "write_memory", refuse)
        graph = built(LIBRARY, tmp_path)
        assert graph.counts["nodes"] == len(LIBRARY)
        assert not _memory(tmp_path).exists()


class TestLineage:
    def test_an_arrow_comes_from_each_of_the_three_recorded_sources(
        self, tmp_path: Path
    ) -> None:
        """The spec's own provenance, a design run's archive, and the site's column."""
        parent = spec("parent")
        child = spec("child", check_limit=3)
        child["provenance"] = {"parents": [spec_hash(parent)]}
        evolved = spec("evolved", stalemate="loss")
        stored = spec("stored", max_plies=200)

        games = [game(parent), game(child), game(evolved), game(stored)]
        games[2].lineage = {
            "parents": [{"hash": spec_hash(parent), "name": "parent"}],
            "run": "design-1",
            "generation": 2,
        }
        graph = graphing.build(
            games, tmp_path, parents={spec_hash(stored): spec_hash(parent)}
        )
        ids = {node.name: node.hash for node in graph.nodes}
        drawn = {(arrow["source"], arrow["target"]) for arrow in graph.lineage}
        assert drawn == {
            (ids["parent"], ids["child"]),
            (ids["parent"], ids["evolved"]),
            (ids["parent"], ids["stored"]),
        }
        run = next(a for a in graph.lineage if a["target"] == ids["evolved"])
        assert (run["run"], run["generation"]) == ("design-1", 2)

    def test_an_arrow_is_drawn_however_far_apart_the_two_games_are(
        self, tmp_path: Path
    ) -> None:
        """Which is the point of having it: a child far from its parent keeps its arrow."""
        parent = spec("parent")
        child = far("child")
        child["provenance"] = {"parents": [spec_hash(parent)]}
        graph = graphing.build([game(parent), game(child)], tmp_path)
        assert len(graph.lineage) == 1
        assert graph.lineage[0]["units"] > graphing.NEAR

    def test_a_parent_nobody_has_is_not_invented(self, tmp_path: Path) -> None:
        child = spec("child", check_limit=3)
        child["provenance"] = {"parents": ["0123456789ab"]}
        graph = graphing.build([game(spec("other")), game(child)], tmp_path)
        assert graph.lineage == []

    def test_only_the_immediate_parent_gets_an_arrow(self, tmp_path: Path) -> None:
        """`parents` is a chain, and its later entries are grandparents.

        The arrow from the grandparent to the parent already says what an arrow from the
        grandparent to the child would.
        """
        old = spec("old")
        middle = spec("middle", check_limit=3)
        young = spec("young", check_limit=3, stalemate="loss")
        young["provenance"] = {"parents": [spec_hash(middle), spec_hash(old)]}
        graph = graphing.build(
            [game(old), game(middle), game(young)], tmp_path
        )
        ids = {node.name: node.hash for node in graph.nodes}
        assert [(a["source"], a["target"]) for a in graph.lineage] == [
            (ids["middle"], ids["young"])
        ]


class Scored(library.Game):
    """A game carrying a card handed to it rather than read off disk."""

    held: dict[str, Any] | None = None

    @property
    def score(self) -> dict[str, Any] | None:
        return self.held


def scored(made: dict[str, Any], held: dict[str, Any] | None) -> library.Game:
    one = Scored(
        hash=spec_hash(made),
        name=str(made["name"]),
        description="",
        path=Path("x.json"),
        source="committed",
        spec=made,
    )
    one.held = held
    return one


# One measurement as the library reads it, headline and working both. The three weighted
# shares are the scores under weights of 0.25, 0.078 and 0.125, which is what
# `scoring.Weights.combine` writes into `index-<weights>.json` beside the scores.
CARD: dict[str, Any] = {
    "index": 0.5, "sd": 0.01, "viable": True, "weights_version": "v10",
    "seeds": [0, 1000, 2000, 3000], "totals_used": 4,
    "runs": [{"seed": 0, "total": 0.5}],
    "verdict": {"rule": "vote-tie-refuses", "accept": 4, "refuse": 0},
    "scores": {"skill_gradient": 0.8, "stall_rate": 0.9, "ply_spread": 0.4},
    "contributions": {"skill_gradient": 0.2, "stall_rate": 0.07, "ply_spread": 0.05},
}


class TestTheScoreOnANode:
    """Read from the card the library already built, never worked out again.

    A third game, scored or not, is in each library so that the two being compared
    share something rare: two games alone share only what both have, which weighs
    nothing, and no line joins them.
    """

    def test_a_node_carries_the_run_count_and_the_verdict_beside_the_number(
        self, tmp_path: Path
    ) -> None:
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(far("away"), None),
                scored(spec("other", check_limit=3), {**CARD, "index": 0.7}),
            ],
            tmp_path,
        )
        node = next(one for one in graph.nodes if one.name == "base")
        assert node.score is not None
        assert node.score["seeds"] == [0, 1000, 2000, 3000]
        assert node.score["totalsUsed"] == 4
        assert node.score["runVerdict"]["accept"] == 4
        assert node.score["weights"] == "v10"

    def test_a_line_between_two_read_scores_carries_the_gap(self, tmp_path: Path) -> None:
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(far("away"), None),
                scored(spec("other", check_limit=3), {**CARD, "index": 0.7}),
            ],
            tmp_path,
        )
        gap = _line(graph).gap
        assert gap is not None
        # Signed from the line's own source to its target, which is why the test reads
        # the size rather than the sign.
        assert abs(gap["gap"]) == pytest.approx(0.2)
        assert {gap["index"], gap["otherIndex"]} == {0.5, 0.7}
        assert gap["clear"] is True

    def test_each_end_of_a_gap_carries_the_runs_it_rests_on(self, tmp_path: Path) -> None:
        """Q339. Both numbers are printed, so both counts travel, each with its own end.

        A one-run end and a four-run end are different claims, and the gap line says so
        beside each number rather than letting one count stand for both.
        """
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(far("away"), None),
                scored(spec("other", check_limit=3),
                       {**CARD, "index": 0.7, "seeds": [0], "totals_used": 1}),
            ],
            tmp_path,
        )
        gap = _line(graph).gap
        assert gap is not None
        by_end = {gap["index"]: gap["seeds"], gap["otherIndex"]: gap["otherSeeds"]}
        assert by_end == {0.5: [0, 1000, 2000, 3000], 0.7: [0]}

    def test_an_end_that_names_no_seeds_has_no_gap(self, tmp_path: Path) -> None:
        """A number with no count is not printed, so a difference of it is not either."""
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(far("away"), None),
                scored(spec("other", check_limit=3), {**CARD, "index": 0.7, "seeds": []}),
            ],
            tmp_path,
        )
        assert _line(graph).gap is None

    def test_a_game_whose_score_cannot_be_read_shows_none_and_no_gap(
        self, tmp_path: Path
    ) -> None:
        """`Game.score` is already `None` for a stale measurement. That is the one policy."""
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(far("away"), None),
                scored(spec("other", check_limit=3), None),
            ],
            tmp_path,
        )
        assert [node.score is None for node in graph.nodes].count(True) == 2
        assert _line(graph).gap is None

    def test_a_gap_names_the_measurements_that_moved_most(self, tmp_path: Path) -> None:
        """Ranked by how much each moved the index, not by how much it moved.

        `scoring.Weights.combine` already says what each measurement contributed, and
        that is the ranking: a measurement worth a fifth of the score moving 0.2 matters
        more than one worth a twentieth moving 0.3. Both ends' own numbers travel, so the
        page prints them and works out nothing.
        """
        other = {
            **CARD,
            "index": 0.7,
            "scores": {"skill_gradient": 0.5, "stall_rate": 0.9, "ply_spread": 0.9},
            "contributions": {
                "skill_gradient": 0.125, "stall_rate": 0.07, "ply_spread": 0.1125,
            },
        }
        graph = graphing.build(
            [scored(BASE, dict(CARD)), scored(spec("other", check_limit=3), other),
             scored(far("away"), None)],
            tmp_path,
        )
        gap = _line(graph).gap
        assert gap is not None
        moved = gap["moved"]
        # `ply_spread` moves further -- 0.5 against `skill_gradient`'s 0.3 -- and still
        # comes second, because `skill_gradient` is worth twice as much of the index and
        # the ranking is by what each one did to the total.
        assert [part["key"] for part in moved] == ["skill_gradient", "ply_spread"]
        # Both ends' own numbers, signed from the line's own source as the gap above it
        # is -- which is why this reads the pair rather than which way round it fell.
        assert {moved[0]["index"], moved[0]["otherIndex"]} == {0.8, 0.5}

    def test_a_measurement_only_one_of_them_carries_is_left_out(
        self, tmp_path: Path
    ) -> None:
        """An optional component the other game's corpus could not supply.

        There is nothing to compare it with, and reading its absence as a fall to zero
        would put it at the top of every list it appeared in.
        """
        other = {
            **CARD,
            "index": 0.55,
            "scores": {"skill_gradient": 0.8, "stall_rate": 0.9, "ply_spread": 0.4,
                       "opening_diversity": 0.9},
            "contributions": {"skill_gradient": 0.2, "stall_rate": 0.07,
                              "ply_spread": 0.05, "opening_diversity": 0.06},
        }
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(far("away"), None),
                scored(spec("other", check_limit=3), other),
            ],
            tmp_path,
        )
        gap = _line(graph).gap
        assert gap is not None
        assert gap["moved"] == []

    def test_a_measurement_is_named_in_the_glossary_own_words(
        self, tmp_path: Path
    ) -> None:
        """`skill_gradient` is an identifier, and the map is a page people read.

        The words come from `metrics/glossary.py`, the same place the score panel and the
        sort menu take them from, and they travel in the payload rather than being spelt
        out on the page.
        """
        from chess_variant_sandbox.metrics.glossary import title_of

        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(far("away"), None),
                scored(
                    spec("other", check_limit=3),
                    {**CARD, "index": 0.7,
                     "scores": {**CARD["scores"], "skill_gradient": 0.5}},
                ),
            ],
            tmp_path,
        )
        assert graph.components == {"skill_gradient": title_of("skill_gradient")}
        assert graph.components["skill_gradient"] != "skill_gradient"

    def test_two_weights_versions_never_share_a_gap(self, tmp_path: Path) -> None:
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(far("away"), None),
                scored(
                    spec("other", check_limit=3),
                    {**CARD, "index": 0.7, "weights_version": "v9"},
                ),
            ],
            tmp_path,
        )
        assert _line(graph).gap is None


class TestTheScaleTheFillRunsOver:
    """`docs/archive/plan-game-graph.md` G4: the two ends of the map's score fill."""

    def test_it_is_the_library_own_range_and_not_nought_to_one(
        self, tmp_path: Path
    ) -> None:
        """A corpus packed into the top third of the unit would paint one colour."""
        graph = graphing.build(
            [
                scored(BASE, {**CARD, "index": 0.62}),
                scored(spec("one", check_limit=1), {**CARD, "index": 0.81}),
                scored(spec("two", check_limit=2), {**CARD, "index": 0.74}),
            ],
            tmp_path,
        )
        assert graph.scale == {
            "low": 0.62, "high": 0.81, "weights": "v10", "scored": 3, "unscored": 0,
        }

    def test_a_game_under_other_weights_is_not_on_the_scale(
        self, tmp_path: Path
    ) -> None:
        """Two scores under different weights never share a leaderboard.

        So they must not share a ramp either. The scale is the version most of the scored
        games are under, and the other game counts as one the fill cannot reach.
        """
        graph = graphing.build(
            [
                scored(BASE, {**CARD, "index": 0.62}),
                scored(spec("one", check_limit=1), {**CARD, "index": 0.81}),
                scored(
                    spec("two", check_limit=2),
                    {**CARD, "index": 0.2, "weights_version": "v9"},
                ),
            ],
            tmp_path,
        )
        assert graph.scale is not None
        assert (graph.scale["weights"], graph.scale["low"]) == ("v10", 0.62)
        assert (graph.scale["scored"], graph.scale["unscored"]) == (2, 1)

    def test_a_library_nothing_can_be_read_in_has_no_scale(self, tmp_path: Path) -> None:
        """A fresh checkout with no `runs/`. The map draws, and every dot is hollow."""
        graph = built([BASE, spec("one", check_limit=3)], tmp_path)
        assert graph.scale is None
        assert graph.as_dict()["scale"] is None


def _line(graph: graphing.Graph) -> graphing.Line:
    """The line between `base` and `other`, which every library above draws."""
    line = line_between(graph, "base", "other")
    assert line is not None
    return line


class TestWhatSizesADot:
    """`docs/archive/plan-game-graph.md` §8.3 G11: the size menu and each dot's place on it.

    More than one thing can decide how big a dot is drawn, so the payload carries a menu
    and every node carries a number from 0 to 1 under each key. The page maps that number
    to a radius and decides nothing else, which is the rule the fill already runs under.
    M4 made the first key the count of lines (each of `families.MUTUAL` at most), where it
    was the count of rule-step links within three rules, eight drawing full size.
    """

    def test_the_menu_is_the_two_keys_the_library_can_answer(
        self, tmp_path: Path
    ) -> None:
        """*neighbours* first, because it is the size the map opens on."""
        graph = graphing.build(
            [
                scored(BASE, {**CARD, "index": 0.62}),
                scored(spec("one", check_limit=1), {**CARD, "index": 0.81}),
                scored(far("away"), None),
            ],
            tmp_path,
        )
        assert [size["key"] for size in graph.sizes] == ["neighbours", "score"]
        lines, score = graph.sizes
        assert lines["high"] == families.MUTUAL
        assert lines["highWords"] == f"{families.MUTUAL} lines"
        assert lines["low"] == min(node.degree for node in graph.nodes)
        # The ramp's own two ends, which is what makes a dot's size and its colour one
        # reading of one number rather than two.
        assert (score["low"], score["high"]) == (0.62, 0.81)
        assert (score["lowWords"], score["highWords"]) == ("0.620", "0.810")
        # Every key says what it means in one sentence, and the one some games have no
        # value under says what happens to them.
        assert all(size["words"].endswith(".") for size in graph.sizes)
        assert "is drawn smallest" in score["words"]

    def test_neighbours_is_the_count_of_lines_over_the_most_a_game_can_have(
        self, drawn: graphing.Graph
    ) -> None:
        touching = {node.hash: 0 for node in drawn.nodes}
        for line in drawn.neighbours:
            touching[line.source] += 1
            touching[line.target] += 1
        for node in drawn.nodes:
            assert node.degree == touching[node.hash] <= families.MUTUAL
            assert node.size["neighbours"] == round(node.degree / families.MUTUAL, 4)

    def test_a_game_off_the_ramp_has_no_size_under_score(self, tmp_path: Path) -> None:
        """No value is not nought. The page draws it smallest and the legend says why.

        Three refusals, all the fill's own: never measured, out of date -- which the
        library already reads as no card at all -- and scored under other weights, which
        never shares a leaderboard and so never shares a ramp.
        """
        graph = graphing.build(
            [
                scored(BASE, {**CARD, "index": 0.6}),
                scored(spec("one", check_limit=1), {**CARD, "index": 0.8}),
                scored(spec("two", check_limit=2), None),
                scored(
                    spec("three", check_limit=4),
                    {**CARD, "index": 0.7, "weights_version": "v9"},
                ),
            ],
            tmp_path,
        )
        at = {node.name: node.size["score"] for node in graph.nodes}
        assert (at["base"], at["one"]) == (0.0, 1.0)
        assert at["two"] is None and at["three"] is None
        # And every value the payload carries is on the unit or absent.
        for node in graph.as_dict()["nodes"]:
            for value in node["size"].values():
                assert value is None or 0.0 <= value <= 1.0

    def test_a_library_with_no_ramp_is_not_offered_the_score_key(
        self, drawn: graphing.Graph
    ) -> None:
        """A fresh checkout with no `runs/`: every dot hollow, and one key in the menu.

        A menu entry that sized every game the same would be a control doing nothing.
        """
        assert [size["key"] for size in drawn.sizes] == ["neighbours"]
        assert all(node.size["score"] is None for node in drawn.nodes)
        held = json.loads(json.dumps(drawn.as_dict()))
        assert all(set(node["size"]) == {"neighbours", "score"} for node in held["nodes"])


class TestThePairCache:
    SPECS = [BASE, spec("one", check_limit=1), spec("two", check_limit=2), far("away")]

    def cold(self) -> None:
        """Forget what this process remembers, so only the files can answer."""
        relate._PAIRS.clear()
        relate._normal_cached.cache_clear()
        graphing._MAPS.clear()

    def test_a_second_call_computes_no_pair_and_compiles_no_spec(
        self, tmp_path: Path
    ) -> None:
        first = built(self.SPECS, tmp_path)
        assert first.computed > 0
        assert first.compiled == len(self.SPECS)
        self.cold()
        second = built(self.SPECS, tmp_path)
        assert (second.computed, second.compiled) == (0, 0)
        assert second.as_dict() == first.as_dict()

    def test_a_new_game_costs_its_own_compile_and_the_lines_it_drew(
        self, tmp_path: Path
    ) -> None:
        built(self.SPECS, tmp_path)
        self.cold()
        added = built([*self.SPECS, spec("three", check_limit=3)], tmp_path)
        assert added.compiled == 1
        assert added.computed <= len(added.neighbours) + len(added.lineage)

    def test_a_record_taken_under_another_version_is_not_read(
        self, tmp_path: Path
    ) -> None:
        """A reading under an older comparison is not a reading of this one."""
        built(self.SPECS, tmp_path)
        path = tmp_path / graphing.CACHE_NAME / graphing.CACHE_FILE
        held = json.loads(path.read_text(encoding="utf-8"))
        held["relate"] = "r0"
        path.write_text(json.dumps(held), encoding="utf-8")
        self.cold()
        again = built(self.SPECS, tmp_path)
        assert again.computed > 0

    def test_a_record_taken_under_an_older_graph_version_is_not_read(
        self, tmp_path: Path
    ) -> None:
        """A `g14` file held similarity counts, a distance and the sweep's sentences."""
        built(self.SPECS, tmp_path)
        path = tmp_path / graphing.CACHE_NAME / graphing.CACHE_FILE
        held = json.loads(path.read_text(encoding="utf-8"))
        assert held["graph"] == graphing.GRAPH_VERSION == "g15"
        held["graph"] = "g14"
        path.write_text(json.dumps(held), encoding="utf-8")
        self.cold()
        again = built(self.SPECS, tmp_path)
        assert again.computed > 0 and again.compiled == len(self.SPECS)

    def test_an_unreadable_cache_is_a_recompute_and_not_a_failure(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / graphing.CACHE_NAME / graphing.CACHE_FILE
        path.parent.mkdir(parents=True)
        path.write_text("{ not json", encoding="utf-8")
        assert built(self.SPECS, tmp_path).counts["nodes"] == len(self.SPECS)

    def test_a_cold_build_and_a_warm_one_are_the_same_bytes(self, tmp_path: Path) -> None:
        """Q362 found the first build after a cold cache differing from the next.

        The difference was JSON key order alone, in the parts of a pair record that reach
        the payload as they are -- a record just computed keeps its insertion order and
        one read back from the file is sorted. A reader that caches the payload compares
        it byte for byte, so the order is part of the answer.
        """
        first = built(LIBRARY, tmp_path)
        assert any(line.differences for line in first.neighbours)
        self.cold()
        second = built(LIBRARY, tmp_path)
        assert json.dumps(first.as_dict()) == json.dumps(second.as_dict())


class TestThePayload:
    def test_it_is_json_and_carries_its_versions(self, drawn: graphing.Graph) -> None:
        payload = json.loads(json.dumps(drawn.as_dict()))
        assert payload["version"] == graphing.GRAPH_VERSION
        assert payload["relate"] == relate.RELATE_VERSION
        assert payload["map"] == families.MAP_VERSION
        assert payload["layout"] == mapping.LAYOUT_VERSION
        assert payload["mutual"] == families.MUTUAL
        assert set(payload) >= {
            "nodes", "neighbours", "lineage", "kinds", "bounds", "units", "counts",
            "communities", "unusual",
        }

    def test_the_retired_layers_left_the_payload(self, drawn: graphing.Graph) -> None:
        """M4: the rule-step links, the similarity rows and their cuts, the far row."""
        payload = drawn.as_dict()
        for gone in ("links", "similar", "levels", "clusters", "groups"):
            assert gone not in payload, gone
        for gone in ("oneRule", "gaps", "nearest", "sentences", "between", "similar",
                     "alone", "unlinked"):
            assert gone not in payload["counts"], gone
        assert all("far" not in node for node in payload["nodes"])
        for gone in ("SIMILAR", "LEVELS", "DISTINCT_IN", "DISTINCT_OUT", "SEEDS", "bond",
                     "_sweep", "_main", "_structural", "_fallback", "_layout",
                     "COMPONENT_GAP", "ROW_STEP", "LINKS_FULL"):
            assert not hasattr(graphing, gone), gone

    def test_every_kind_arrives_in_words_a_reader_can_act_on(
        self, drawn: graphing.Graph
    ) -> None:
        """The panel prints `words` and never the schema's own name for a kind.

        *wild*, *hand* and *valuation* are positions in the schema, not English, and the
        map is a public page. The words live here rather than on the page because the
        payload is the map's whole interface -- and they are exhaustive over
        `relate.KINDS`, so a kind added to the schema fails this rather than reaching a
        panel unnamed.
        """
        assert set(graphing.KIND_WORDS) == set(relate.KINDS)
        assert all(words and words != kind for kind, words in graphing.KIND_WORDS.items())
        assert [entry["kind"] for entry in drawn.kinds] == list(relate.KINDS)
        assert all(entry["words"] == graphing.KIND_WORDS[entry["kind"]]
                   for entry in drawn.kinds)


def test_the_map_modules_name_no_piece_and_no_game() -> None:
    """Rule 5, over the code of `graphing` and `communities` with the prose stripped.

    Plan S1's proof: the comments reach for examples from the library, as the rest of
    the module does, and what may not happen is a name reaching the code. Module and
    function docstrings are dropped; every other string is code, and is read.
    """
    import ast

    root = Path(__file__).resolve().parents[1]
    named: set[str] = set()
    for folder in ("rulesets", "archive"):
        for path in sorted((root / folder).glob("*.json")):
            held = json.loads(path.read_text(encoding="utf-8"))
            named.add(path.stem.lower())
            named.add(str(held.get("name", "")).lower())
            named.update(
                str(piece["name"]).lower()
                for piece in held.get("pieces", []) if piece.get("name")
            )
    # `rules.castling` names its two seats in the schema itself, and `pieces[].spy` (schema
    # 27) is a key the code reads by name. A schema key is not a piece name.
    named -= {"king", "rook", "spy", ""}
    for module in ("graphing", "communities", "families"):
        tree = ast.parse(
            (root / f"chess_variant_sandbox/web/{module}.py").read_text(encoding="utf-8")
        )
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (
                isinstance(body, list) and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body[0] = ast.Pass()
        code = ast.unparse(tree).lower()
        found = sorted(
            word for word in named if re.search(rf"\b{re.escape(word)}\b", code)
        )
        assert not found, (module, found)


#: The corpus `docs/archive/plan-game-graph.md` G2 and `plan-graph-map.md` S1 were measured
#: over, by mechanics hash: the 70 committed and archived specs of 2026-09-25 (after Q434
#: retired `tunnel-horizontal`), less what the library has retired since (below) and with two
#: respelt games under their new hash. `TestTheCommittedCorpus` pins the plans' numbers on this
#: set, not on whatever the library holds today, so a game added later (Q281a4's three hex
#: showcases, 2026-09-28, were the first) cannot turn it red; the live library is
#: `TestTheWholeLibrary`'s, which reads its counts from the corpus. A game of this set
#: that is retired or edited in place fails `plan_corpus` by name: that is a re-read of the
#: plans (`plan-graph-map.md` §8), not a re-aim.
#:
#: Re-read at Q878 (2026-10-13), 60 of the 70, every pinned number re-derived on them:
#: Q868's audit retired ten of the seventy with nothing to rebuild them from (`tunnel-vertical`
#: 0268bb0926bf, `onewaytunnel` da7647bf02e1, `sardines-torus` f5585062433b,
#: `torus-extended` fdadd9839d34, `weathervane-torus` 9777cd6c9ed1, `hierarchy-fall-swap`
#: f1fdbb539699, `wolves-rotated` 213f7b6604e4, `-2` 8ec0ce4678e9, `-3` 2e94e1688601, `-4`
#: 87a488975531), and Q869 respelt two in place, so their hashes moved: `crazyhouse-torus`
#: ab865f3f954a -> 9598e603241d, `torus-less-extended` 3194d3a2251b -> 8d9fb1b4c0b4. Two
#: retired games are kept, rebuilt from their twins by `retired_specs` (`REBUILT`): `hex-glinski-
#: promoting` (Q868) and `wolves-immune` (Q876), the corpus's one twin pair, which is what keeps
#: the merge claimed in the plans measured on a real pair and not only on `TestTwins`' made ones.
PLAN_CORPUS = frozenset({
    "03aa7c3f2f18", "0b96463ffba7", "0be72e61ef24", "1442513d38d7", "14a126b38a1a", "151582aed409",
    "15593fc47f6d", "1ba3d61e5961", "2261f8203948", "279f92a97530", "29d5213b02b2", "2eefcd835b3b",
    "313c913a5b3b", "328cab4db5cc", "3c639d277808", "3ec1560124c3", "42bd7d06828d", "482ddc556331",
    "4c1922ffae26", "50a3da8f75e7", "51d9b36b4e93", "54b38fb8cba8", "54c731ca6e95", "57667d7861ad",
    "57c73975c905", "5f9c1bfcf85a", "66b001f1b790", "67270a8805a0", "6bd4d9c5769e", "71912ea8ba7f",
    "78130df01edd", "82a4788c02f0", "85555467a4ca", "8d9fb1b4c0b4", "8fc3939243d5", "93b02b591c87",
    "9574b0bf9095", "9598e603241d", "975dde173f01", "9c26a04f1ca5", "9f538ab85a77", "a2193fb37097",
    "a2f0c65ceedf", "a7b6f22e6b73", "a9306688fe57", "add543cc28c3", "b02bbe8da72c", "b1f536573c22",
    "bb5d97520084", "bdf2801ade24", "c0e0c35f78b1", "c2925e027455", "c6ff883e5919", "ceb4916bdabc",
    "df2fbba0ca76", "e7293070b5be", "eaa48ed29ad7", "fcbb53d53ce4", "fe7c42772c71", "ffc7b604bea0",
})


@pytest.fixture(scope="module")
def library_games() -> list[library.Game]:
    """Every committed and archived ruleset today, which is what the map draws."""
    root = Path(__file__).resolve().parents[1]
    return [
        one
        for one in library.discover(root / "rulesets", root / "runs", root / "archive")
        if one.source in ("committed", "archived")
    ]


@pytest.fixture(scope="module")
def corpus(library_games: list[library.Game]) -> list[library.Game]:
    """The plans' corpus out of today's library: every game of `PLAN_CORPUS`, and only those.

    A member the library retired comes back rebuilt from `tests/retired_specs.py` where that
    module holds a rebuild, and only where it reproduces the member's hash.
    """
    kept = [one for one in library_games if one.hash in PLAN_CORPUS]
    for name, rebuild in REBUILT.items():
        digest = RETIRED_HASHES[name]
        if digest in PLAN_CORPUS:
            made = rebuild()
            assert spec_hash(made) == digest, name
            kept.append(game(made, "archived"))
    missing = PLAN_CORPUS - {one.hash for one in kept}
    assert not missing, (
        f"the plans' corpus lost {sorted(missing)}: a game was retired or its mechanics "
        "edited, so re-read plan-graph-map.md's numbers on what is left (its §8)"
    )
    return kept


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One runs directory for the whole module, so the corpus is compiled once."""
    return tmp_path_factory.mktemp("graph")


@pytest.fixture(scope="module")
def graph(corpus: list[library.Game], cache: Path) -> graphing.Graph:
    return graphing.build(corpus, cache)


@pytest.mark.slow
class TestTheCommittedCorpus:
    """`docs/archive/plan-game-graph.md` §5 G2's corpus, under M4's map.

    Sixty spec files, fifty-nine nodes: what the plans' corpus pins that M4 did not retire
    is the twin merge and the lineage. The one-rule links, gaps, nearest relatives,
    operator sentences, the default cut's lines and the communities they formed were the
    rule-step and similarity layers' numbers and left with them; M4's own reading of the
    library (N1 to N9) is recorded in `docs/plan-map.md` rather than pinned here, because
    it is a reading of the library of the day.
    """

    def test_the_counts_are_the_ones_the_plan_names(self, graph: graphing.Graph) -> None:
        assert graph.counts["specs"] == len(PLAN_CORPUS)
        assert graph.counts["nodes"] == 59
        assert graph.counts["twins"] == 1
        assert graph.counts["lineage"] == 9

    def test_the_one_pair_of_twins_is_one_node_with_the_shorter_reading(
        self, graph: graphing.Graph, corpus: list[library.Game]
    ) -> None:
        """Two files declare one rule two ways and compile to the same tables.

        So they are one node, and a line from it reads the shorter of its two spellings'
        readings -- what neither file says on its own.
        """
        merged = [node for node in graph.nodes if len(node.hashes) > 1]
        assert len(merged) == 1
        specs = {one.hash: one.spec for one in corpus}
        for line in graph.neighbours:
            if merged[0].hash not in (line.source, line.target):
                continue
            other = line.target if line.source == merged[0].hash else line.source
            readings = [relate.units(specs[digest], specs[other])
                        for digest in merged[0].hashes]
            assert line.units == min(readings)

    def test_no_two_nodes_are_drawn_on_top_of_each_other(
        self, graph: graphing.Graph
    ) -> None:
        spots = [(node.x, node.y) for node in graph.nodes]
        closest = min(
            math.dist(one, other)
            for index, one in enumerate(spots)
            for other in spots[index + 1:]
        )
        assert closest >= graphing.MIN_SEPARATION - 1e-3

    def test_the_library_in_reverse_is_the_same_map(
        self, corpus: list[library.Game], cache: Path, graph: graphing.Graph
    ) -> None:
        """Determinism: the order the library arrives in moves nothing, and costs no pair."""
        graphing._MAPS.clear()
        again = graphing.build(list(reversed(corpus)), cache)
        assert (again.computed, again.compiled) == (0, 0)
        assert again.communities == graph.communities
        assert [(n.hash, n.x, n.y) for n in again.nodes] == [
            (n.hash, n.x, n.y) for n in graph.nodes
        ]


@pytest.fixture(scope="module")
def today(
    library_games: list[library.Game], cache: Path, graph: graphing.Graph
) -> graphing.Graph:
    """The map of today's library, after `graph` has compiled the plans' specs."""
    return graphing.build(library_games, cache)


@pytest.mark.slow
class TestTheWholeLibrary:
    """The map of the library as it stands today, whatever it has grown to.

    Nothing here is a number pinned to a date: every count is read from the corpus, so a
    new game moves the map and leaves this green, while a game the map drops, a twin
    counted twice or a game in two communities still fails.
    """

    def test_every_game_is_drawn_once(
        self, today: graphing.Graph, library_games: list[library.Game]
    ) -> None:
        assert today.counts["specs"] == len(library_games)
        drawn = [digest for node in today.nodes for digest in node.hashes]
        assert sorted(drawn) == sorted(one.hash for one in library_games)
        assert today.counts["nodes"] == len(today.nodes)
        assert today.counts["twins"] == sum(
            len(node.hashes) - 1 for node in today.nodes
        )

    def test_the_communities_partition_the_games_that_are_not_unusual(
        self, today: graphing.Graph
    ) -> None:
        sizes = [one["size"] for one in today.communities]
        assert sum(sizes) == sum(1 for node in today.nodes if node.community is not None)
        assert today.counts["unusual"] == sum(node.unusual for node in today.nodes)
        assert {one["hash"] for one in today.unusual} == {
            node.hash for node in today.nodes if node.unusual
        }
