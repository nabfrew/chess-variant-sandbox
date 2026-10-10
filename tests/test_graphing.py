"""The library as a graph: which links are drawn, which games merge, and where they sit.

`tests/test_relate.py` holds what two specs differ in. This holds what a *library* of them
looks like -- one node per game however many files spell it, a link only where a link says
something, a faint one where nothing else reaches, and a coordinate for every node.

The specs are written here rather than read from `rulesets/`, because what is under test
is the shape of a library and not what any game happens to be. One test at the end reads
the committed corpus, because the plan's acceptance is a claim about that corpus, and it
is marked slow: it compiles seventy specs.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.spec import relate, validate_spec
from chess_variant_sandbox.spec.canonical import spec_hash
from chess_variant_sandbox.web import graphing, library

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


def between(graph: graphing.Graph, first: str, second: str) -> graphing.Link | None:
    ids = named(graph)
    pair = {ids[first], ids[second]}
    for link in graph.links:
        if {link.source, link.target} == pair:
            return link
    return None


class TestTheSpecsUnderTest:
    def test_every_one_of_them_is_a_game(self) -> None:
        """A spec the validator refuses would make every count below meaningless."""
        for made in (BASE, spec("a", check_limit=3), moved("b", "R"), far("c")):
            assert validate_spec(made) == []


def spelt_out(name: str) -> dict[str, Any]:
    """`BASE` with who-may-take-whom written out on every piece.

    Saying *this piece may take all three* is saying what the game does anyway, so the two
    files compile to the same tables. No comparison of two written descriptions can know
    that, and the corpus's one pair of twins is this shape.
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

        The node is one game, so it is one rule away -- a rule declared the long way is
        still that rule.
        """
        plain = spec("plain")
        plain["schema_version"] = 3
        spelt = spelt_out("spelt")
        other = spec("other", check_limit=3)
        other["schema_version"] = 3
        assert relate.units(plain, other) == 1
        assert relate.units(spelt, other) == 2

        graph = built([plain, spelt, other], tmp_path)
        assert graph.counts["nodes"] == 2
        merged = next(node for node in graph.nodes if len(node.hashes) > 1)
        link = next(
            one for one in graph.links
            if {one.source, one.target} == {merged.hash, named(graph)["other"]}
        )
        assert link.units == 1


class TestWhichLinksAreDrawn:
    def test_every_one_rule_pair_is_a_link(self, tmp_path: Path) -> None:
        graph = built([BASE, spec("one", check_limit=3)], tmp_path)
        link = between(graph, "base", "one")
        assert link is not None
        assert (link.type, link.units, link.unit) == ("rule", 1, "rules.check_limit")
        assert link.kind == relate.RULE_KINDS["check_limit"]
        assert link.sentence == relate.RULE_WORDS["check_limit"]

    def test_a_pair_a_third_game_splits_is_not_drawn(self, tmp_path: Path) -> None:
        """Two rules apart with a game holding one of them between: no link.

        Both steps are already on the map and the pair adds nothing but a line.
        """
        graph = built(
            [
                BASE,
                spec("middle", check_limit=3),
                spec("both", check_limit=3, stalemate="loss"),
            ],
            tmp_path,
        )
        assert between(graph, "base", "both") is None
        assert graph.counts["gaps"] == 0
        assert graph.counts["oneRule"] == 2

    def test_a_pair_nothing_splits_is_drawn_as_a_gap(self, tmp_path: Path) -> None:
        """Two rules apart with nothing between: a dashed link marking a game nobody made."""
        graph = built(
            [BASE, spec("two", check_limit=3, stalemate="loss")], tmp_path
        )
        link = between(graph, "base", "two")
        assert link is not None
        assert link.type == "gap"
        assert link.units == 2
        assert len(link.differences) == 2
        assert graph.counts["gaps"] == 1

    def test_nothing_past_three_rules_is_drawn_as_a_gap(self, tmp_path: Path) -> None:
        graph = built([BASE, far("away")], tmp_path)
        link = between(graph, "base", "away")
        assert link is not None
        assert link.type == "nearest"
        assert link.units is not None
        assert link.units > graphing.NEAR


class TestTheOperatorSentence:
    """What a one-rule link says about the design step that crosses it.

    `loop/operators.py` is thirty-four named local edits, each of which writes a sentence
    about what it did. Where one of them, drawn from one of these two games, lands on the
    other, the link carries those words. The draws are blind -- an operator cannot be
    aimed -- so the seeds are fixed and the whole sweep is a pure function of the corpus
    and the operator set.

    Three games a stalemate rule apart, because `toggle-a-rule` reaches all three of them
    inside forty seeds and the tests below can then say *this link has a sentence* about
    a library of three rather than about the corpus.
    """

    SPECS = [BASE, spec("loses", stalemate="loss"), spec("wins", stalemate="win")]

    @staticmethod
    def drawing(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        """Sweep with these operators alone.

        Forty seeds of all thirty-four from three games is four thousand draws and most
        of a second, which is the corpus test's business and not the fast suite's. The
        registry is read inside `_sweep`, so narrowing it here narrows what that call
        draws and nothing else.
        """
        from chess_variant_sandbox.loop import operators

        monkeypatch.setattr(
            operators,
            "OPERATORS",
            tuple(one for one in operators.OPERATORS if one.name in names),
        )

    def test_a_link_an_operator_reaches_carries_the_operator_own_words(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.drawing(monkeypatch, "toggle-a-rule")
        graph = built(self.SPECS, tmp_path, sentences=True)
        link = between(graph, "base", "loses")
        assert link is not None and link.reach is not None
        assert link.reach["operator"] == "toggle-a-rule"
        # The words are the operator's own, written when it made the child, and they say
        # something a reading of two files cannot: what somebody would *do*.
        assert link.reach["words"].startswith("a player with no legal move now ")
        assert link.reach["words"] != link.sentence
        assert graph.counts["sentences"] == 3

    def test_the_sentence_says_which_end_it_starts_from(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An edit has a direction and a link has two ends, so the record names one.

        Both ends reach here -- the rule can be set either way round -- and the link's
        own source wins, which is the fixed walk rather than whichever ran first.
        """
        self.drawing(monkeypatch, "toggle-a-rule")
        graph = built(self.SPECS, tmp_path, sentences=True)
        for link in graph.links:
            assert link.reach is not None
            assert link.reach["from"] == link.source

    def test_a_link_no_operator_reaches_carries_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A miss is the ordinary case -- two thirds of the corpus's links -- not news."""
        self.drawing(monkeypatch, "resize-the-board")
        graph = built(self.SPECS, tmp_path, sentences=True)
        assert graph.counts["oneRule"] == 3
        assert graph.counts["sentences"] == 0
        assert all(link.reach is None for link in graph.links)

    def test_an_operator_that_raises_is_a_non_match_and_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A draw that blows up is a draw that did not land, and the page never sees it."""
        from chess_variant_sandbox.loop import operators

        def boom(spec: dict[str, Any], rng: Any) -> str:
            raise ValueError("this operator cannot be drawn")

        monkeypatch.setattr(
            operators, "OPERATORS", (operators.Operator("boom", boom),)
        )
        graph = built(self.SPECS, tmp_path, sentences=True)
        assert graph.counts["sentences"] == 0
        assert graph.counts["oneRule"] == 3

    def test_a_child_the_validator_refuses_is_a_non_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Landing on the other game is not enough; the child has to be a game.

        The normal form does not compare `schema_version`, so this edit lands exactly on
        the neighbour and is still refused, which is the case that tells a match by
        identity apart from a match that is worth a sentence.
        """
        from chess_variant_sandbox.loop import operators

        def refused(made: dict[str, Any], rng: Any) -> str:
            made["rules"]["stalemate"] = "loss"
            made["schema_version"] = 99
            return "set the stalemate rule in a spec nothing can read"

        child = copy.deepcopy(BASE)
        refused(child, None)
        assert validate_spec(child)  # the premise: this child is refused
        assert relate.identity_key(child) == relate.identity_key(self.SPECS[1])

        monkeypatch.setattr(
            operators, "OPERATORS", (operators.Operator("refused", refused),)
        )
        graph = built(self.SPECS, tmp_path, sentences=True)
        assert graph.counts["sentences"] == 0

    def test_a_gap_carries_no_sentence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One operator is one edit, so only a one-rule link is ever drawn against."""
        self.drawing(monkeypatch, "toggle-a-rule")
        graph = built(
            [BASE, spec("two", check_limit=3, stalemate="loss")], tmp_path,
            sentences=True,
        )
        link = between(graph, "base", "two")
        assert link is not None and link.type == "gap"
        assert link.reach is None

    def test_the_draws_run_once_per_pair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Including the misses, which are most of them.

        A miss stored as an answer is what keeps the second call free; a miss stored as
        an absence would redraw every link nothing reaches, for ever.
        """
        self.drawing(monkeypatch, "toggle-a-rule", "resize-the-board")
        first = built(self.SPECS, tmp_path, sentences=True)
        assert first.drawn > 0
        relate._PAIRS.clear()
        relate._normal_cached.cache_clear()
        second = built(self.SPECS, tmp_path, sentences=True)
        assert (second.drawn, second.computed, second.compiled) == (0, 0, 0)
        assert second.as_dict() == first.as_dict()
        # And a build that was not asked to sweep still prints what the sweep found.
        assert built(self.SPECS, tmp_path).as_dict() == first.as_dict()

    def test_a_build_that_was_not_asked_to_sweep_draws_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Which is what keeps twenty seconds off a request.

        The map is served by a handler that cannot afford the sweep, so the default is
        off and the sentences arrive from the cache that something else filled.
        """
        self.drawing(monkeypatch, "toggle-a-rule")
        graph = built(self.SPECS, tmp_path)
        assert graph.drawn == 0
        assert graph.counts["sentences"] == 0
        assert all(link.reach is None for link in graph.links)

    def test_a_new_game_draws_only_for_its_own_links(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.drawing(monkeypatch, "toggle-a-rule")
        built(self.SPECS, tmp_path, sentences=True)
        added = built(
            [*self.SPECS, spec("draws", stalemate="draw", check_limit=2)], tmp_path,
            sentences=True,
        )
        # The new game and the ends it linked to, and nothing that was already answered.
        assert 0 < added.drawn <= 4 * graphing.SEEDS

    def test_the_same_library_draws_the_same_sentences_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fixed seeds, a fixed walk, no clock and no thread count."""
        self.drawing(monkeypatch, "toggle-a-rule")
        one = built(self.SPECS, tmp_path / "one", sentences=True)
        other = built(self.SPECS, tmp_path / "other", sentences=True)
        assert one.as_dict() == other.as_dict()


class TestTheNearestRelative:
    def test_a_game_nothing_reaches_gets_one_faint_link(self, tmp_path: Path) -> None:
        """`spec/distance.py` picks it, and the link says how far in rules as well.

        At that range *which* rules differ is a list nobody reads, so the link carries
        the count and the feature distance and no sentences.
        """
        graph = built([BASE, spec("near", check_limit=3), far("away")], tmp_path)
        faint = [link for link in graph.links if link.type == "nearest"]
        assert len(faint) == 1
        assert faint[0].distance is not None
        assert faint[0].differences == ()
        assert named(graph)["away"] in (faint[0].source, faint[0].target)

    def test_no_node_is_left_without_a_link(self, tmp_path: Path) -> None:
        graph = built([BASE, spec("near", check_limit=3), far("away")], tmp_path)
        joined = {node for link in graph.links for node in (link.source, link.target)}
        assert joined == {node.hash for node in graph.nodes}
        assert graph.counts["unlinked"] == 0

    def test_one_game_on_its_own_has_nothing_to_be_near(self, tmp_path: Path) -> None:
        """The one case a node is left with no link, and it must not raise."""
        graph = built([BASE], tmp_path)
        assert (graph.counts["nodes"], graph.counts["unlinked"]) == (1, 1)
        assert graph.links == []
        assert graph.bounds["width"] == pytest.approx(2 * graphing.NODE_RADIUS)

    def test_two_far_games_that_pick_each_other_share_one_link(
        self, tmp_path: Path
    ) -> None:
        one, other = far("away"), far("elsewhere")
        other["rules"]["max_plies"] = 130
        graph = built([BASE, one, other], tmp_path)
        assert graph.counts["nearest"] == 1
        assert between(graph, "away", "elsewhere") is not None

    def test_a_far_game_page_names_the_game_its_line_is_drawn_to(
        self, tmp_path: Path
    ) -> None:
        """The map's faint line and the page's nearest four are one measure's answer.

        `docs/archive/plan-game-graph.md` §9.5's T5 kept `spec/distance.py` for both. Here
        the count of rules would name another game first: `tall` is `BASE` on the far game's
        board, five rules from `away`, and `near` is seven, but nearer by feature.
        """
        tall = copy.deepcopy(BASE)
        tall["name"] = "tall"
        tall["board"] = {"width": 4, "height": 9}
        tall["setup"]["placements"] = [
            {"piece": "K", "squares": ["b1"]},
            {"piece": "R", "squares": ["a1", "d1"]},
            {"piece": "W", "squares": ["a2", "b2", "c2", "d2"]},
        ]
        games = [game(one) for one in (BASE, spec("near", check_limit=3), tall, far("away"))]
        graph = graphing.build(games, tmp_path)
        ids = named(graph)
        faint = [link for link in graph.links if link.type == "nearest"]
        assert [{link.source, link.target} for link in faint] == [{ids["away"], ids["near"]}]

        listed = library.neighbours(games[-1], games)
        assert listed[0]["hash"] == ids["near"]
        assert listed[0]["distance"] == faint[0].distance
        # What ordering by the count would have put first.
        assert min(listed, key=lambda link: link["units"])["hash"] == ids["tall"]


class TestTheGroupsAreTheCommunities:
    """`docs/archive/plan-graph-map.md` S5: what the page outlines and folds is `communities`.

    The same-rule `clusters` and the `groups` folded from them were the rule steps'
    grouping. The map is drawn by likeness now, so they left the payload at `g13`, and
    nothing builds them: the communities are a partition already, which is what a fold
    needs, and each carries the place its folded dot opens at.
    """

    def test_the_rule_steps_groupings_are_gone_from_the_payload(
        self, tmp_path: Path
    ) -> None:
        graph = built(
            [BASE, spec("one", check_limit=1), spec("two", check_limit=2), far("away")],
            tmp_path,
        )
        payload = json.loads(json.dumps(graph.as_dict()))
        assert graphing.GRAPH_VERSION == "g14"
        assert payload["version"] == "g14"
        assert "clusters" not in payload and "groups" not in payload
        assert "clusters" not in payload["counts"] and "groups" not in payload["counts"]
        assert not hasattr(graph, "clusters") and not hasattr(graph, "groups")
        for gone in ("_clusters", "_cliques", "_groups", "FAR_WORDS"):
            assert not hasattr(graphing, gone), gone

    def test_every_community_is_a_fold_with_a_place_and_a_name(
        self, tmp_path: Path
    ) -> None:
        """What the page folds and labels, each field the server's.

        A community is a partition cell -- no game in two -- and its `x, y` is where its
        folded dot opens, the mean of its members' places, as a group's was.
        """
        graph = built(
            [BASE, spec("one", check_limit=1), spec("two", check_limit=2), far("away")],
            tmp_path,
        )
        payload = json.loads(json.dumps(graph.as_dict()))
        assert payload["communities"]
        seen: set[str] = set()
        at = {node["hash"]: node for node in payload["nodes"]}
        for community in payload["communities"]:
            assert {"id", "name", "members", "names", "size", "x", "y"} <= set(community)
            assert not seen & set(community["members"])
            seen |= set(community["members"])
            assert community["size"] == len(community["members"])
            assert all(at[digest]["community"] == community["id"]
                       for digest in community["members"])


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
        """Which is the point of having it: no structural link reaches that far."""
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
    """Read from the card the library already built, never worked out again."""

    def test_a_node_carries_the_run_count_and_the_verdict_beside_the_number(
        self, tmp_path: Path
    ) -> None:
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(spec("other", check_limit=3), {**CARD, "index": 0.7}),
            ],
            tmp_path,
        )
        node = graph.nodes[0]
        assert node.score is not None
        assert node.score["seeds"] == [0, 1000, 2000, 3000]
        assert node.score["totalsUsed"] == 4
        assert node.score["runVerdict"]["accept"] == 4
        assert node.score["weights"] == "v10"

    def test_a_link_between_two_read_scores_carries_the_gap(self, tmp_path: Path) -> None:
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(spec("other", check_limit=3), {**CARD, "index": 0.7}),
            ],
            tmp_path,
        )
        gap = graph.links[0].gap
        assert gap is not None
        # Signed from the link's own source to its target, which is why the test reads
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
                scored(spec("other", check_limit=3),
                       {**CARD, "index": 0.7, "seeds": [0], "totals_used": 1}),
            ],
            tmp_path,
        )
        link = graph.links[0]
        gap = link.gap
        assert gap is not None
        by_end = {gap["index"]: gap["seeds"], gap["otherIndex"]: gap["otherSeeds"]}
        assert by_end == {0.5: [0, 1000, 2000, 3000], 0.7: [0]}

    def test_an_end_that_names_no_seeds_has_no_gap(self, tmp_path: Path) -> None:
        """A number with no count is not printed, so a difference of it is not either."""
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(spec("other", check_limit=3), {**CARD, "index": 0.7, "seeds": []}),
            ],
            tmp_path,
        )
        assert graph.links[0].gap is None

    def test_a_game_whose_score_cannot_be_read_shows_none_and_no_gap(
        self, tmp_path: Path
    ) -> None:
        """`Game.score` is already `None` for a stale measurement. That is the one policy."""
        graph = graphing.build(
            [
                scored(BASE, dict(CARD)),
                scored(spec("other", check_limit=3), None),
            ],
            tmp_path,
        )
        assert [node.score is None for node in graph.nodes].count(True) == 1
        assert graph.links[0].gap is None

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
            [scored(BASE, dict(CARD)), scored(spec("other", check_limit=3), other)],
            tmp_path,
        )
        gap = graph.links[0].gap
        assert gap is not None
        moved = gap["moved"]
        # `ply_spread` moves further -- 0.5 against `skill_gradient`'s 0.3 -- and still
        # comes second, because `skill_gradient` is worth twice as much of the index and
        # the ranking is by what each one did to the total.
        assert [part["key"] for part in moved] == ["skill_gradient", "ply_spread"]
        # Both ends' own numbers, signed from the link's own source as the gap above it
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
                scored(spec("other", check_limit=3), other),
            ],
            tmp_path,
        )
        gap = graph.links[0].gap
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
                scored(
                    spec("other", check_limit=3),
                    {**CARD, "index": 0.7, "weights_version": "v9"},
                ),
            ],
            tmp_path,
        )
        assert graph.links[0].gap is None


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


class TestWhatSizesADot:
    """`docs/archive/plan-game-graph.md` §8.3 G11: the size menu and each dot's place on it.

    More than one thing can decide how big a dot is drawn, so the payload carries a menu
    and every node carries a number from 0 to 1 under each key. The page maps that number
    to a radius and decides nothing else, which is the rule the fill already runs under.
    """

    SPECS = [BASE, spec("one", check_limit=1), spec("two", check_limit=2)]

    def test_the_menu_is_the_two_keys_the_corpus_can_answer(
        self, tmp_path: Path
    ) -> None:
        """*links* first, because it is the size the map opens on."""
        graph = graphing.build(
            [
                scored(BASE, {**CARD, "index": 0.62}),
                scored(spec("one", check_limit=1), {**CARD, "index": 0.81}),
            ],
            tmp_path,
        )
        assert [size["key"] for size in graph.sizes] == ["links", "score"]
        links, score = graph.sizes
        assert (links["low"], links["high"]) == (1, graphing.LINKS_FULL)
        assert links["highWords"] == "8 or more links"
        # The ramp's own two ends, which is what makes a dot's size and its colour one
        # reading of one number rather than two.
        assert (score["low"], score["high"]) == (0.62, 0.81)
        assert (score["lowWords"], score["highWords"]) == ("0.620", "0.810")
        # Every key says what it means in one sentence, and the one some games have no
        # value under says what happens to them.
        assert all(size["words"].endswith(".") for size in graph.sizes)
        assert "is drawn smallest" in score["words"]

    def test_links_is_the_count_over_eight_and_reproduces_the_maps_own_radius(
        self, tmp_path: Path
    ) -> None:
        """G3's rule, moved to the server unchanged.

        The page draws a radius of 0.5 plus half this number, so a game with eight links
        or more is the largest dot the layout has room for and everything else is a
        straight fraction of the way there.
        """
        graph = built(self.SPECS, tmp_path)
        for node in graph.nodes:
            assert node.degree == 2
            assert node.size["links"] == 0.25
        busy = built([BASE] + [spec(f"n{at}", check_limit=at) for at in range(1, 10)],
                     tmp_path / "busy")
        assert {node.size["links"] for node in busy.nodes} == {1.0}
        assert min(node.degree for node in busy.nodes) == 9

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
        self, tmp_path: Path
    ) -> None:
        """A fresh checkout with no `runs/`: every dot hollow, and one key in the menu.

        A menu entry that sized every game the same would be a control doing nothing.
        """
        graph = built(self.SPECS, tmp_path)
        assert [size["key"] for size in graph.sizes] == ["links"]
        assert all(node.size["score"] is None for node in graph.nodes)

    def test_the_payload_carries_the_menu_and_every_place_on_it(
        self, tmp_path: Path
    ) -> None:
        held = json.loads(json.dumps(built(self.SPECS, tmp_path).as_dict()))
        assert [size["key"] for size in held["sizes"]] == ["links"]
        assert all(set(node["size"]) == {"links", "score"} for node in held["nodes"])


class TestTheLayout:
    SPECS = [
        BASE,
        spec("one", check_limit=1),
        spec("two", check_limit=2),
        spec("three", check_limit=3),
        spec("wide", stalemate="loss"),
        moved("walks", "W"),
        far("away"),
    ]

    def test_the_same_library_lays_out_the_same_way_twice(self, tmp_path: Path) -> None:
        first = built(self.SPECS, tmp_path / "a")
        second = built(list(reversed(self.SPECS)), tmp_path / "b")
        assert {node.hash: (node.x, node.y) for node in first.nodes} == {
            node.hash: (node.x, node.y) for node in second.nodes
        }

    def test_no_two_nodes_are_drawn_on_top_of_each_other(self, tmp_path: Path) -> None:
        graph = built(self.SPECS, tmp_path)
        spots = [(node.x, node.y) for node in graph.nodes]
        closest = min(
            math.dist(one, other)
            for index, one in enumerate(spots)
            for other in spots[index + 1:]
        )
        assert closest >= 2 * graphing.NODE_RADIUS

    def test_the_bounds_hold_every_node(self, tmp_path: Path) -> None:
        graph = built(self.SPECS, tmp_path)
        bounds = graph.bounds
        for node in graph.nodes:
            assert bounds["minX"] <= node.x <= bounds["maxX"]
            assert bounds["minY"] <= node.y <= bounds["maxY"]
        _hold_the_span(bounds, graph.nodes)

    def test_the_span_is_rounded_once_where_its_ends_round_apart(self) -> None:
        """Q744. Ends at -0.9994 and 10.0004 round to -0.999 and 10.0, 10.999 apart, while
        their span, 10.9998, rounds to 11.0. That pair is what failed on cloud Linux: the
        old check held the width to the rounded ends at a relative 1e-6, about 1e-5 here.
        """
        r = graphing.NODE_RADIUS
        nodes = [_At(0.0006 - (1.0 - r), 0.0), _At(9.0004 + (1.0 - r), 0.0)]
        bounds = graphing._bounds(nodes)  # type: ignore[arg-type]
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


class TestThePairCache:
    SPECS = [BASE, spec("one", check_limit=1), spec("two", check_limit=2), far("away")]

    def cold(self) -> None:
        """Forget what this process remembers, so only the file can answer."""
        relate._PAIRS.clear()
        relate._normal_cached.cache_clear()

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

    def test_a_new_game_costs_its_own_pairs_and_no_others(self, tmp_path: Path) -> None:
        built(self.SPECS, tmp_path)
        self.cold()
        added = built([*self.SPECS, spec("three", check_limit=3)], tmp_path)
        assert (added.computed, added.compiled) == (len(self.SPECS), 1)

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
        """A cache written before the index is not a cache of a payload that carries it."""
        built(self.SPECS, tmp_path)
        path = tmp_path / graphing.CACHE_NAME / graphing.CACHE_FILE
        held = json.loads(path.read_text(encoding="utf-8"))
        assert held["graph"] == graphing.GRAPH_VERSION
        held["graph"] = "g7"
        for record in held["pairs"].values():
            record.pop("similarity", None)
        path.write_text(json.dumps(held), encoding="utf-8")
        self.cold()
        again = built(self.SPECS, tmp_path)
        assert again.computed > 0
        assert all(link.similarity["words"] for link in again.links)

    def test_a_distance_cached_under_g13_is_not_read(self, tmp_path: Path) -> None:
        """Q281a3 rescaled `distance()` uniformly; a `g13` record predates that rescale.

        The bump to `GRAPH_VERSION` is what makes this record unreadable -- without it, a
        pair's stale `distance` would sit in the file forever, since nothing else ever
        overwrites an already-present key (`_Cache.distance`).
        """
        first, second = game(far("away")), game(far("elsewhere"))
        path = tmp_path / graphing.CACHE_FILE
        path.write_text(
            json.dumps(
                {
                    "relate": relate.RELATE_VERSION,
                    "graph": "g13",
                    "specs": {},
                    "pairs": {
                        ":".join(sorted((first.hash, second.hash))): {"distance": -1.0}
                    },
                }
            ),
            encoding="utf-8",
        )
        cache = graphing._Cache(path)
        cache.read()
        assert cache.pairs == {}
        assert cache.distance(first, second) != -1.0

    def test_an_unreadable_cache_is_a_recompute_and_not_a_failure(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / graphing.CACHE_NAME / graphing.CACHE_FILE
        path.parent.mkdir(parents=True)
        path.write_text("{ not json", encoding="utf-8")
        assert built(self.SPECS, tmp_path).counts["nodes"] == len(self.SPECS)

    def test_a_cold_build_and_a_warm_one_are_the_same_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Q362 found the first build after a cold cache differing from the next.

        The difference was JSON key order alone, in the parts of a pair record that reach
        the payload as they are -- a record just computed keeps its insertion order and
        one read back from the file is sorted. A reader that caches the payload compares
        it byte for byte, so the order is part of the answer.
        """
        from chess_variant_sandbox.loop import operators

        monkeypatch.setattr(
            operators,
            "OPERATORS",
            tuple(one for one in operators.OPERATORS if one.name == "toggle-a-rule"),
        )
        specs = [
            BASE,
            spec("loses", stalemate="loss"),
            spec("two", check_limit=3, stalemate="win"),
            far("away"),
        ]
        first = built(specs, tmp_path, sentences=True)
        assert any(link.reach for link in first.links)
        assert any(link.between for link in first.links)
        self.cold()
        second = built(specs, tmp_path, sentences=True)
        assert json.dumps(first.as_dict()) == json.dumps(second.as_dict())


class TestTheGameInBetweenAGap:
    """What a dashed line says about the game nobody has made.

    `docs/archive/plan-game-graph.md` G6. Whether one can be built is `web/designing.py`'s
    answer and this module only asks it, once per pair, and carries what came back so the
    page can offer a button without deciding anything.
    """

    def test_a_gap_whose_rules_can_be_carried_across_says_which_and_from_where(
        self, tmp_path: Path
    ) -> None:
        graph = built([BASE, spec("two", check_limit=3, stalemate="loss")], tmp_path)
        link = between(graph, "base", "two")
        assert link is not None and link.between is not None
        assert link.between["words"] in relate.RULE_WORDS.values()
        assert link.between["from"] in (link.source, link.target)
        assert link.between["taken"] in (link.source, link.target)
        assert link.between["from"] != link.between["taken"]
        assert graph.counts["between"] == 1

    def test_a_gap_that_differs_in_more_than_rules_says_why_instead(
        self, tmp_path: Path
    ) -> None:
        """A piece and a rule apart, so nothing here knows what half way is."""
        one = moved("moves", "R")
        one["rules"]["check_limit"] = 3
        graph = built([BASE, one], tmp_path)
        link = between(graph, "base", "moves")
        assert link is not None and link.type == "gap"
        assert link.between is not None
        assert "words" not in link.between
        assert "not only their rules" in link.between["why"]
        assert graph.counts["between"] == 0

    def test_a_one_rule_link_is_not_asked_about(self, tmp_path: Path) -> None:
        """There is nothing between two games one rule apart, so the question is not one."""
        graph = built([BASE, spec("one", check_limit=3)], tmp_path)
        link = between(graph, "base", "one")
        assert link is not None and link.between is None

    def test_the_answer_is_worked_out_once(self, tmp_path: Path) -> None:
        """It costs a compile of the game it builds, which no warm request should pay."""
        specs = [BASE, spec("two", check_limit=3, stalemate="loss")]
        first = built(specs, tmp_path)
        assert first.spanned == 1
        relate._PAIRS.clear()
        relate._normal_cached.cache_clear()
        second = built(specs, tmp_path)
        assert (second.spanned, second.computed, second.compiled) == (0, 0, 0)
        assert second.as_dict() == first.as_dict()

    def test_the_payload_carries_it_on_the_link(self, tmp_path: Path) -> None:
        graph = built([BASE, spec("two", check_limit=3, stalemate="loss")], tmp_path)
        payload = json.loads(json.dumps(graph.as_dict()))
        gap = next(link for link in payload["links"] if link["type"] == "gap")
        assert set(gap["between"]) == {"from", "taken", "words"}
        assert all(link["between"] is None for link in payload["links"]
                   if link["type"] == "rule")

    def test_the_game_it_names_is_one_the_designer_builds(self, tmp_path: Path) -> None:
        """The payload's promise, kept by the one module that decides it.

        The map offers a button and the press asks `designing.in_between` again, so a
        link that says there is a game in between has to be a link that gets one.
        """
        from chess_variant_sandbox.web import designing

        specs = [BASE, spec("two", check_limit=3, stalemate="loss")]
        graph = built(specs, tmp_path)
        link = between(graph, "base", "two")
        assert link is not None and link.between is not None
        by_hash = {spec_hash(one): one for one in specs}
        made = designing.in_between(
            by_hash[link.between["from"]], by_hash[link.between["taken"]]
        )
        assert "spec" in made
        assert made["words"] == link.between["words"]
        assert made["on"] == 0


class TestTheCacheFiller:
    """`python -m chess_variant_sandbox.web.graphing [root]`, which the deploy runs as a oneshot unit.

    The deploy runs it on every deploy and it has no guard of its own, so *skipped when
    the cache is current* has to mean *draws nothing when the cache is current*. What the
    box reads is the line it prints, so the draw count is in that line.
    """

    SPECS = [BASE, spec("loses", stalemate="loss")]

    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A checkout the filler can discover: rulesets/ and nothing else yet."""
        from chess_variant_sandbox.loop import operators

        monkeypatch.setattr(
            operators,
            "OPERATORS",
            tuple(one for one in operators.OPERATORS if one.name == "toggle-a-rule"),
        )
        rulesets = tmp_path / "rulesets"
        rulesets.mkdir()
        for made in self.SPECS:
            (rulesets / f"{made['name']}.json").write_text(
                json.dumps(made), encoding="utf-8"
            )
        return tmp_path

    @staticmethod
    def drew(printed: str) -> int:
        found = re.search(r"(\d+) draws", printed)
        assert found, printed
        return int(found.group(1))

    def test_a_second_run_draws_nothing_and_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = self.root(tmp_path, monkeypatch)
        assert graphing._main([str(root)]) == 0
        first = capsys.readouterr().out
        assert self.drew(first) > 0
        assert "1 of them with an operator sentence" in first

        relate._PAIRS.clear()
        relate._normal_cached.cache_clear()
        assert graphing._main([str(root)]) == 0
        second = capsys.readouterr().out
        assert self.drew(second) == 0
        assert "0 pairs compared, 0 specs compiled" in second
        # And the second run says the same thing about the map as the first.
        assert second.split("(")[0] == first.split("(")[0]

    def test_it_writes_the_cache_the_app_reads_and_nothing_else(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The unit's `ReadWritePaths` is `runs/`, so a second file would fail on the box."""
        root = self.root(tmp_path, monkeypatch)
        graphing._main([str(root)])
        capsys.readouterr()
        assert [path.name for path in (root / "runs").rglob("*") if path.is_file()] == [
            graphing.CACHE_FILE
        ]
        assert (root / "runs" / graphing.CACHE_NAME / graphing.CACHE_FILE).is_file()


class TestTheSimilarityIndex:
    """How alike two games are, beside how many rules apart they are.

    `tests/test_relate.py` holds what the six counts are and what the index weighs them
    at. This holds what a *library* does with them -- every line carries the reading, a
    pair record keeps the counts and not the number, and two spellings of one game read
    as one game. `docs/archive/plan-game-graph.md` §9.5 T2.
    """

    TIERS = ("rules", "settings", "mechanisms", "pieces", "squares", "books")
    SAME = relate.Similarity(0, 0, 0, 0, 0, 0)

    @staticmethod
    def restricted(name: str) -> dict[str, Any]:
        """`BASE` with the walker allowed to take the walker alone.

        It involves *who may take whom* and sets it differently from `spelt_out`, while
        `BASE` does not involve it at all. So all three are one rule apart by `units` and
        two different distances by the index, which is what the merge has to choose over.
        """
        made = spec(name)
        made["schema_version"] = 3
        for one in made["pieces"]:
            if one["id"] == "W":
                one["captures"] = ["W"]
        return made

    def test_a_line_carries_what_relate_reads_for_its_two_games(
        self, tmp_path: Path
    ) -> None:
        one = spec("one", check_limit=3)
        graph = built([BASE, one], tmp_path)
        link = between(graph, "base", "one")
        assert link is not None
        assert link.similarity == relate.similarity(BASE, one).as_dict()
        assert set(link.similarity) == {"index", "counts", "words"}
        assert set(link.similarity["counts"]) == set(self.TIERS)

    def test_every_line_carries_one_however_far_apart_its_ends_are(
        self, tmp_path: Path
    ) -> None:
        """The faint line included: ranking wants the far pairs too."""
        graph = built(
            [
                BASE,
                spec("one", stalemate="loss"),
                spec("two", check_limit=3, stalemate="win"),
                far("away"),
            ],
            tmp_path,
        )
        payload = json.loads(json.dumps(graph.as_dict()))
        assert {link["type"] for link in payload["links"]} >= {"rule", "gap", "nearest"}
        for link in payload["links"]:
            assert set(link["similarity"]) == {"index", "counts", "words"}
            assert link["similarity"]["words"]
            assert link["similarity"]["index"] > 0

    def test_a_pair_record_keeps_the_counts_and_not_the_number(
        self, tmp_path: Path
    ) -> None:
        """The index is arithmetic over the counts under `relate.SIMILARITY_RATIO`.

        Keeping the counts alone means a changed ratio re-reads this file rather than
        dropping it, so nothing stored here depends on what the ratio is.
        """
        built([BASE, spec("one", check_limit=3), far("away")], tmp_path)
        path = tmp_path / graphing.CACHE_NAME / graphing.CACHE_FILE
        held = json.loads(path.read_text(encoding="utf-8"))
        assert held["pairs"]
        for record in held["pairs"].values():
            assert set(record["similarity"]) == set(self.TIERS)

    def test_two_spellings_of_one_game_read_as_one_game(self, tmp_path: Path) -> None:
        """No line is drawn between them -- they are one node -- so the rule lives in the
        pair cache, which is where anything asking about the two finds it."""
        plain = spec("plain")
        plain["schema_version"] = 3
        spelt = spelt_out("spelt")
        assert relate.similarity(plain, spelt).index > 0

        cache = graphing._Cache(tmp_path / graphing.CACHE_FILE)
        first, second = game(plain), game(spelt)
        assert cache.twin(first) == cache.twin(second)
        assert cache.pair(first, second)["similarity"] == self.SAME.counts

    def test_a_merged_node_takes_the_lower_index_to_a_third_game(
        self, tmp_path: Path
    ) -> None:
        """The two measures rank differently, and each picks its own nearest spelling.

        Both spellings are one rule from `other`, so `units` cannot choose between them.
        One of them involves who may take whom and sets it differently, which is half a
        rule; the other does not involve it at all, which is a whole one.
        """
        plain = spec("plain")
        plain["schema_version"] = 3
        spelt = spelt_out("spelt")
        other = self.restricted("other")
        assert relate.units(plain, other) == relate.units(spelt, other) == 1
        assert relate.similarity(plain, other).index > relate.similarity(spelt, other).index

        graph = built([plain, spelt, other], tmp_path)
        merged = next(node for node in graph.nodes if len(node.hashes) > 1)
        link = next(
            one for one in graph.links
            if {one.source, one.target} == {merged.hash, named(graph)["other"]}
        )
        assert link.units == 1
        assert link.similarity == relate.similarity(spelt, other).as_dict()

    def test_a_game_page_and_its_panel_on_the_map_read_one_index(
        self, tmp_path: Path
    ) -> None:
        """§9.5's T3 (`docs/archive/plan-game-graph.md`): one number, one order, two places.

        Four games one rule from `BASE`, named so the alphabet orders them wrongly: a new
        rule, a rule set differently, a piece that moves another way and a changed price.
        The game's page lists them by the index, and each carries the reading the map's
        line between the same two games carries, which is what the panel orders by.
        """
        price = spec("d-price")
        for one in price["pieces"]:
            if one["id"] == "R":
                one["value"] = 4
        games = [
            game(one)
            for one in (
                BASE,
                spec("a-rule", check_limit=3),
                spec("b-setting", promotion={"piece": "W", "rank": 4, "to": ["R"]}),
                moved("c-moved", "R"),
                price,
            )
        ]
        graph = graphing.build(games, tmp_path)
        ids = named(graph)
        lines = {
            (link.target if link.source == ids["base"] else link.source): link
            for link in graph.links
            if ids["base"] in (link.source, link.target)
        }
        listed = library.neighbours(games[0], games)
        assert {link["hash"] for link in listed} == set(lines)
        assert all(link["units"] == 1 for link in listed)
        for link in listed:
            # The page's link also carries the reading in a visitor's words (G40); the
            # reading itself is the map's.
            reading = {key: value for key, value in link["similarity"].items() if key != "site"}
            assert reading == lines[link["hash"]].similarity, link["name"]
        indexes = [link["similarity"]["index"] for link in listed]
        assert indexes == sorted(indexes) and len(set(indexes)) == len(indexes)
        assert [link["name"] for link in listed] == ["d-price", "c-moved", "b-setting", "a-rule"]

    def test_it_says_nothing_about_which_lines_are_drawn(self, tmp_path: Path) -> None:
        """A line exists because two games are a step apart, which the index does not move."""
        specs = [BASE, spec("one", check_limit=3), moved("two", "R"), far("away")]
        graph = built(specs, tmp_path)
        drawn = {(link.source, link.target, link.type) for link in graph.links}
        assert drawn
        assert all(
            link.units == relate.units(
                next(one for one in specs if spec_hash(one) == link.source),
                next(one for one in specs if spec_hash(one) == link.target),
            )
            for link in graph.links
        )


class TestThePayload:
    def test_it_is_json_and_carries_its_two_versions(self, tmp_path: Path) -> None:
        graph = built([BASE, spec("one", check_limit=3), far("away")], tmp_path)
        payload = json.loads(json.dumps(graph.as_dict()))
        assert payload["version"] == graphing.GRAPH_VERSION
        assert payload["relate"] == relate.RELATE_VERSION
        assert set(payload) >= {
            "nodes", "links", "lineage", "kinds", "bounds", "units", "counts",
            "similar", "levels", "communities",
        }

    def test_the_legend_names_every_kind_of_rule_whatever_the_corpus_holds(
        self, tmp_path: Path
    ) -> None:
        """A chip that appears and disappears with the corpus moves under the reader."""
        graph = built([BASE, spec("one", check_limit=3)], tmp_path)
        assert [entry["kind"] for entry in graph.kinds] == list(relate.KINDS)

    def test_every_kind_arrives_in_words_a_reader_can_act_on(
        self, tmp_path: Path
    ) -> None:
        """G3's legend prints `words` and never the schema's own name for a kind.

        *wild*, *hand* and *valuation* are positions in the schema, not English, and the
        map is a public page. The words live here rather than on the page because the
        payload is the map's whole interface -- and they are exhaustive over
        `relate.KINDS`, so a kind added to the schema fails this rather than reaching a
        legend unnamed.
        """
        assert set(graphing.KIND_WORDS) == set(relate.KINDS)
        assert all(words and words != kind for kind, words in graphing.KIND_WORDS.items())
        graph = built([BASE, spec("one", check_limit=3)], tmp_path)
        assert all(entry["words"] == graphing.KIND_WORDS[entry["kind"]]
                   for entry in graph.kinds)

    def test_a_link_names_every_kind_on_it_and_not_only_the_first(
        self, tmp_path: Path
    ) -> None:
        graph = built([BASE, spec("two", check_limit=3, must_capture=True)], tmp_path)
        link = graph.links[0]
        assert link.kind in link.kinds
        assert set(link.kinds) == {
            relate.RULE_KINDS["check_limit"], relate.RULE_KINDS["must_capture"]
        }


class TestTheSimilarityLines:
    """`docs/archive/plan-graph-map.md` S1: every near pair as a row, the cuts, the communities.

    Two families and two far games. The `check_limit` family (with the walker changed on
    two of them) and the one round `base` are joined inside by one setting or one rule;
    the two far games are one rule and a clock apart, which is past the default cut.
    """

    SPECS = [
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

    @pytest.fixture(scope="class")
    @classmethod
    def graph(cls, tmp_path_factory: pytest.TempPathFactory) -> graphing.Graph:
        return built(cls.SPECS, tmp_path_factory.mktemp("similar"))

    def test_the_bond_counts_the_index_tiers_from_the_top(self) -> None:
        assert graphing.bond(0) == pytest.approx(5.0)
        assert graphing.bond(7) == pytest.approx(4.0)
        assert graphing.bond(63) == pytest.approx(3.0)
        assert graphing.bond(511) == pytest.approx(2.0)
        assert graphing.bond(32_766) > 0

    def test_every_pair_within_the_last_cut_is_a_row_and_no_other(
        self, graph: graphing.Graph
    ) -> None:
        index = {
            graphing._key(spec_hash(one), spec_hash(other)): relate.similarity(one, other).index
            for position, one in enumerate(self.SPECS)
            for other in self.SPECS[position + 1:]
        }
        rows = {(row["source"], row["target"]): row for row in graph.similar}
        assert set(rows) == {
            pair for pair, value in index.items() if value <= graphing.LEVELS[-1]
        }
        for pair, row in rows.items():
            assert row["source"] < row["target"]
            assert row["index"] == index[pair]
            assert row["bond"] == round(graphing.bond(row["index"]), 3)
            assert row["level"] == min(cut for cut in graphing.LEVELS if row["index"] <= cut)
            assert row["words"] == relate.Similarity(**_counts(self.SPECS, pair)).sentence

    def test_rows_come_nearest_first(self, graph: graphing.Graph) -> None:
        order = [(row["index"], row["source"], row["target"]) for row in graph.similar]
        assert order == sorted(order)

    def test_a_row_and_a_drawn_line_over_one_pair_read_one_index(
        self, graph: graphing.Graph
    ) -> None:
        rows = {(row["source"], row["target"]): row for row in graph.similar}
        for link in graph.links:
            row = rows.get((link.source, link.target))
            if row is not None:
                assert row["index"] == link.similarity["index"]
                assert row["words"] == link.similarity["words"]

    def test_each_cut_counts_the_rows_it_draws_and_the_games_it_leaves(
        self, graph: graphing.Graph
    ) -> None:
        assert [level["cut"] for level in graph.levels] == list(graphing.LEVELS)
        assert [level["cut"] for level in graph.levels if level["default"]] == [
            graphing.SIMILAR
        ]
        for level in graph.levels:
            drawn = [row for row in graph.similar if row["level"] <= level["cut"]]
            touched = {end for row in drawn for end in (row["source"], row["target"])}
            assert level["lines"] == len(drawn)
            assert level["alone"] == len(graph.nodes) - len(touched)

    def test_the_communities_are_louvain_s_on_the_default_cut(
        self, graph: graphing.Graph
    ) -> None:
        from chess_variant_sandbox.web import communities

        expected = communities.partition({
            (row["source"], row["target"]): graphing.bond(row["index"])
            for row in graph.similar
            if row["index"] <= graphing.SIMILAR
        })
        assert [one["members"] for one in graph.communities] == expected
        assert all(one["size"] == len(one["members"]) for one in graph.communities)

    def test_every_game_a_line_reaches_is_in_exactly_one_community(
        self, graph: graphing.Graph
    ) -> None:
        joined = {
            end
            for row in graph.similar
            if row["index"] <= graphing.SIMILAR
            for end in (row["source"], row["target"])
        }
        where = {node.hash: node.community for node in graph.nodes}
        for community in graph.communities:
            assert all(where[digest] == community["id"] for digest in community["members"])
            assert community["centre"] in community["members"]
        assert {digest for digest, held in where.items() if held} == joined
        assert graph.counts["communities"] == len(graph.communities)
        assert graph.counts["alone"] == len(graph.nodes) - len(joined)

    def test_a_community_is_named_by_its_own_rule_or_near_its_centre(
        self, graph: graphing.Graph
    ) -> None:
        names = sorted(one["name"] for one in graph.communities)
        centre = next(
            one for one in graph.communities if one["words"] is None
        )["centre"]
        assert names == sorted([
            relate.RULE_WORDS["check_limit"],
            f"near {next(node.name for node in graph.nodes if node.hash == centre)}",
        ])

    def test_the_centre_is_the_member_nearest_the_rest(self, graph: graphing.Graph) -> None:
        index = {
            (row["source"], row["target"]): row["index"] for row in graph.similar
        }

        def total(digest: str, members: list[str]) -> float:
            return sum(
                index.get(graphing._key(digest, other), math.inf)
                for other in members if other != digest
            )

        for community in graph.communities:
            members = community["members"]
            assert total(community["centre"], members) == min(
                total(digest, members) for digest in members
            )

    def test_the_same_library_in_reverse_is_the_same_payload(
        self, graph: graphing.Graph, tmp_path: Path
    ) -> None:
        again = built(list(reversed(self.SPECS)), tmp_path)
        assert again.as_dict()["similar"] == graph.as_dict()["similar"]
        assert again.as_dict()["communities"] == graph.as_dict()["communities"]
        assert again.as_dict()["nodes"] == graph.as_dict()["nodes"]

    def test_a_rule_names_a_community_only_where_it_is_the_community_s_own(self) -> None:
        rules = {
            "a": frozenset({"win", "npcs"}),
            "b": frozenset({"win", "npcs"}),
            "c": frozenset({"win", "npcs", "drops"}),
            "d": frozenset({"win"}),
            "e": frozenset({"win", "drops"}),
            "f": frozenset({"win"}),
        }
        joined = sorted(rules)
        # `npcs` is all of a, b and c and none of the rest; `win` is everybody's.
        assert graphing._distinctive(["a", "b", "c"], joined, rules) == "npcs"
        # `drops` is half of c and e's pair, which is not three quarters.
        assert graphing._distinctive(["d", "e"], joined, rules) is None

    def test_a_rule_is_named_in_relate_s_words(self) -> None:
        assert graphing._rule_words("board.wrap") == relate.RULE_WORDS["wrap"]
        assert graphing._rule_words("board.walls") == relate.RULE_WORDS["obstacles"]
        assert graphing._rule_words("piece.captures") == graphing.KIND_WORDS["capture"]
        assert graphing._rule_words("strike") == relate.RULE_WORDS["strike"]
        assert all(
            graphing._rule_words(key) == words for key, words in relate.RULE_WORDS.items()
        )

    def test_the_payload_carries_them(self, graph: graphing.Graph) -> None:
        held = json.loads(json.dumps(graph.as_dict()))
        assert held["version"] == graphing.GRAPH_VERSION
        assert held["similar"][0].keys() == {
            "source", "target", "index", "bond", "level", "words",
        }
        assert held["levels"][0].keys() == {"cut", "lines", "alone", "default"}
        assert held["communities"][0].keys() >= {
            "id", "name", "members", "names", "centre", "size", "x", "y",
        }
        assert all("community" in node for node in held["nodes"])
        # A community's id can never be taken for a game's.
        hashes = {node["hash"] for node in held["nodes"]}
        assert not hashes & {one["id"] for one in held["communities"]}


class TestTheLayoutByCommunity:
    """`docs/archive/plan-graph-map.md` S2: the opening picture, one knot per community.

    The library of `TestTheSimilarityLines`: two communities joined by lines between
    them, and two games no line reaches.
    """

    SPECS = TestTheSimilarityLines.SPECS

    @pytest.fixture(scope="class")
    @classmethod
    def graph(cls, tmp_path_factory: pytest.TempPathFactory) -> graphing.Graph:
        return built(cls.SPECS, tmp_path_factory.mktemp("laid"))

    def test_two_builds_give_identical_coordinates(
        self, graph: graphing.Graph, tmp_path: Path
    ) -> None:
        again = built(list(reversed(self.SPECS)), tmp_path)
        assert [(node.hash, node.x, node.y) for node in again.nodes] == [
            (node.hash, node.x, node.y) for node in graph.nodes
        ]

    def test_it_draws_nothing_from_the_module_level_generator(self, tmp_path: Path) -> None:
        import random

        random.seed(3)
        before = random.getstate()
        built(self.SPECS, tmp_path)
        assert random.getstate() == before

    def test_no_two_dots_are_nearer_than_one_diameter(self, graph: graphing.Graph) -> None:
        spots = [(node.x, node.y) for node in graph.nodes]
        closest = min(
            math.dist(one, other)
            for index, one in enumerate(spots)
            for other in spots[index + 1:]
        )
        assert closest >= 2 * graphing.NODE_RADIUS

    def test_every_game_is_nearer_its_own_community_s_centre_than_any_other(
        self, graph: graphing.Graph
    ) -> None:
        _assert_knots(graph)

    def test_the_games_no_line_reaches_sit_in_the_far_row(
        self, graph: graphing.Graph
    ) -> None:
        """One row under the drawing, from its left edge, a step apart, in hash order."""
        alone = sorted(
            (node for node in graph.nodes if node.community is None),
            key=lambda node: node.hash,
        )
        drawn = [node for node in graph.nodes if node.community is not None]
        assert len(alone) == 2 and drawn
        left = min(node.x for node in drawn)
        row = max(node.y for node in drawn) + graphing.COMPONENT_GAP + graphing.ROW_STEP
        for index, node in enumerate(alone):
            assert (node.x, node.y) == pytest.approx(
                (left + index * graphing.ROW_STEP, row)
            )

    def test_a_community_carries_where_its_members_average_out(
        self, graph: graphing.Graph
    ) -> None:
        at = {node.hash: node for node in graph.nodes}
        for community in graph.communities:
            members = [at[digest] for digest in community["members"]]
            assert community["x"] == pytest.approx(
                sum(node.x for node in members) / len(members), abs=1e-3
            )
            assert community["y"] == pytest.approx(
                sum(node.y for node in members) / len(members), abs=1e-3
            )

    def test_a_library_with_no_line_is_one_row(self, tmp_path: Path) -> None:
        graph = built([BASE, far("away")], tmp_path)
        assert not graph.communities
        assert sorted(node.y for node in graph.nodes)[0] == sorted(
            node.y for node in graph.nodes
        )[-1]


def _assert_knots(graph: graphing.Graph) -> None:
    """Every game in a community is nearer its community's middle than any other's."""
    middles = {}
    for community in graph.communities:
        members = [node for node in graph.nodes if node.community == community["id"]]
        middles[community["id"]] = (
            sum(node.x for node in members) / len(members),
            sum(node.y for node in members) / len(members),
        )
    for node in graph.nodes:
        if node.community is None:
            continue
        mine = math.dist((node.x, node.y), middles[node.community])
        others = [
            math.dist((node.x, node.y), middle)
            for ident, middle in middles.items() if ident != node.community
        ]
        assert all(mine < other for other in others), node.name


def _counts(specs: list[dict[str, Any]], pair: tuple[str, str]) -> dict[str, int]:
    """The tier counts `relate` reads for the two specs with these hashes."""
    by_hash = {spec_hash(one): one for one in specs}
    return relate.similarity(by_hash[pair[0]], by_hash[pair[1]]).counts


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
    # `rules.castling` names its two seats in the schema itself. A schema key is not a
    # piece name.
    named -= {"king", "rook", ""}
    for module in ("graphing", "communities"):
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
#: retired `tunnel-horizontal`). `TestTheCommittedCorpus` pins the plans' numbers on this
#: set, not on whatever the library holds today, so a game added later (Q281a4's three hex
#: showcases, 2026-09-28, were the first) cannot turn it red; the live library is
#: `TestTheWholeLibrary`'s, which reads its counts from the corpus. A game of this set
#: that is retired or edited in place fails `plan_corpus` by name: that is a re-read of the
#: plans (`plan-graph-map.md` §8), not a re-aim.
PLAN_CORPUS = frozenset({
    "0268bb0926bf", "03aa7c3f2f18", "0b96463ffba7", "0be72e61ef24", "1442513d38d7", "14a126b38a1a",
    "151582aed409", "15593fc47f6d", "1ba3d61e5961", "213f7b6604e4", "2261f8203948", "279f92a97530",
    "29d5213b02b2", "2e94e1688601", "2eefcd835b3b", "313c913a5b3b", "3194d3a2251b", "328cab4db5cc",
    "3c639d277808", "3ec1560124c3", "42bd7d06828d", "482ddc556331", "4c1922ffae26", "50a3da8f75e7",
    "51d9b36b4e93", "54b38fb8cba8", "54c731ca6e95", "57667d7861ad", "57c73975c905", "5f9c1bfcf85a",
    "66b001f1b790", "67270a8805a0", "6bd4d9c5769e", "71912ea8ba7f", "78130df01edd", "82a4788c02f0",
    "85555467a4ca", "87a488975531", "8ec0ce4678e9", "8fc3939243d5", "93b02b591c87", "9574b0bf9095",
    "975dde173f01", "9777cd6c9ed1", "9c26a04f1ca5", "9f538ab85a77", "a2193fb37097", "a2f0c65ceedf",
    "a7b6f22e6b73", "a9306688fe57", "ab865f3f954a", "add543cc28c3", "b02bbe8da72c", "b1f536573c22",
    "bb5d97520084", "bdf2801ade24", "c0e0c35f78b1", "c2925e027455", "c6ff883e5919", "ceb4916bdabc",
    "da7647bf02e1", "df2fbba0ca76", "e7293070b5be", "eaa48ed29ad7", "f1fdbb539699", "f5585062433b",
    "fcbb53d53ce4", "fdadd9839d34", "fe7c42772c71", "ffc7b604bea0",
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
    """The plans' corpus out of today's library: every game of `PLAN_CORPUS`, and only those."""
    kept = [one for one in library_games if one.hash in PLAN_CORPUS]
    missing = PLAN_CORPUS - {one.hash for one in kept}
    assert not missing, (
        f"the plans' corpus lost {sorted(missing)}: a game was retired or its mechanics "
        "edited, so re-read plan-graph-map.md's numbers on what is left (its §8)"
    )
    return kept


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One runs directory for the whole module, so the corpus is compared once."""
    return tmp_path_factory.mktemp("graph")


@pytest.fixture(scope="module")
def graph(corpus: list[library.Game], cache: Path) -> graphing.Graph:
    return graphing.build(corpus, cache)


@pytest.fixture(scope="module")
def reached(
    corpus: list[library.Game], cache: Path, graph: graphing.Graph
) -> graphing.Graph:
    """The same map with the operator sweep paid for.

    Eighteen seconds and forty-four thousand draws on this corpus, which is why no
    request makes them and why this fixture depends on `graph`: the pairs are already
    compared by the time it runs, so what it costs is the sweep and nothing else.
    """
    return graphing.build(corpus, cache, sentences=True)


@pytest.mark.slow
class TestTheCommittedCorpus:
    """`docs/archive/plan-game-graph.md` §5 G2's acceptance, on the corpus it was written for.

    Seventy spec files, sixty-nine nodes. Every number here was re-derived at Q329 and
    the plan's row was corrected to match; the figures it was written with came from the
    probe, before G1's folds moved four one-rule readings and before twins merged. Q434
    retired `tunnel-horizontal` (2026-09-25), which took one spec, one node and one gap
    and moved nothing else here. Since Q743 (2026-10-05) the class draws `PLAN_CORPUS`, those
    seventy by hash, so a game the library gains later does not move a number here.
    """

    def test_the_counts_are_the_ones_the_plan_names(self, graph: graphing.Graph) -> None:
        assert graph.counts["specs"] == len(PLAN_CORPUS)
        assert graph.counts["nodes"] == 69
        assert graph.counts["twins"] == 1
        assert graph.counts["oneRule"] == 56
        assert graph.counts["gaps"] == 54
        assert graph.counts["nearest"] == 16
        assert graph.counts["lineage"] == 10
        assert graph.counts["unlinked"] == 0
        # G6: the gaps a game can be built in between, which is 10 of the 54.
        assert graph.counts["between"] == 10

    def test_the_busiest_node_is_a_hub_of_one_rule_links(
        self, graph: graphing.Graph
    ) -> None:
        hub = max(graph.nodes, key=lambda node: node.degree)
        spokes = [
            link for link in graph.links
            if link.type == "rule" and hub.hash in (link.source, link.target)
        ]
        assert len(spokes) == 22

    def test_the_one_pair_of_twins_takes_the_shorter_reading_to_the_hub(
        self, graph: graphing.Graph, corpus: list[library.Game]
    ) -> None:
        """Two files declare one rule two ways and compile to the same tables.

        So they are one node, and the node is one rule from the hub -- which is what the
        shorter of its two readings says and what neither file says on its own.
        """
        merged = [node for node in graph.nodes if len(node.hashes) > 1]
        assert len(merged) == 1
        hub = max(graph.nodes, key=lambda node: node.degree)
        link = next(
            one for one in graph.links
            if {one.source, one.target} == {merged[0].hash, hub.hash}
        )
        assert link.units == 1
        specs = {one.hash: one.spec for one in corpus}
        readings = sorted(
            relate.units(specs[digest], specs[hub.hash]) for digest in merged[0].hashes
        )
        assert readings[0] == 1
        assert readings[-1] > 1

    def test_every_node_carries_a_link(self, graph: graphing.Graph) -> None:
        joined = {node for link in graph.links for node in (link.source, link.target)}
        assert {node.hash for node in graph.nodes} - joined == set()

    def test_the_far_games_are_the_ones_with_a_faint_link(
        self, graph: graphing.Graph
    ) -> None:
        faint = {
            node for link in graph.links if link.type == "nearest"
            for node in (link.source, link.target)
        }
        assert {node.hash for node in graph.nodes if node.far} <= faint
        assert sum(1 for node in graph.nodes if node.far) == 19

    def test_a_third_of_the_one_rule_links_carry_an_operator_sentence(
        self, reached: graphing.Graph
    ) -> None:
        """G5's acceptance. 23 of the 56, which is the figure the probe found too.

        The rest cannot be reached and the reason is structural: sibling games mostly
        differ in a rule's *setting*, and no operator draws a piece that another game
        happens to have. A link with no sentence still says what differs.
        """
        assert reached.counts["sentences"] == 23
        assert reached.counts["oneRule"] == 56
        assert all(
            link.type == "rule" for link in reached.links if link.reach
        )

    def test_the_hub_arrival_variant_reads_as_the_plan_says(
        self, reached: graphing.Graph
    ) -> None:
        """The plan's acceptance example, which is chess and king of the hill.

        Named here because the plan names it: one draw of `win-by-arrival` from the hub
        makes the variant, and the line says so in that operator's own words.
        """
        hub = max(reached.nodes, key=lambda node: node.degree)
        found = [
            link for link in reached.links
            if hub.hash in (link.source, link.target)
            and (link.reach or {}).get("operator") == "win-by-arrival"
        ]
        assert len(found) == 1
        assert found[0].reach is not None
        assert found[0].reach["from"] == hub.hash
        assert found[0].reach["words"] == (
            "the game is now also won by walking the K to the centre"
        )

    def test_the_sweep_runs_once_and_the_map_is_free_afterwards(
        self, reached: graphing.Graph, corpus: list[library.Game], cache: Path
    ) -> None:
        """What makes the sentences affordable at all: they are drawn once, ever."""
        assert reached.drawn > 0
        again = graphing.build(corpus, cache, sentences=True)
        assert (again.drawn, again.computed, again.compiled) == (0, 0, 0)
        assert again.as_dict() == reached.as_dict()

    def test_no_two_nodes_are_drawn_on_top_of_each_other(
        self, graph: graphing.Graph
    ) -> None:
        spots = [(node.x, node.y) for node in graph.nodes]
        closest = min(
            math.dist(one, other)
            for index, one in enumerate(spots)
            for other in spots[index + 1:]
        )
        assert closest >= 2 * graphing.NODE_RADIUS

    # `docs/archive/plan-graph-map.md` S1, on the corpus §3 and §4 were measured over. When the
    # corpus grows these are re-read, not re-aimed (§8).

    def test_the_default_cut_draws_the_lines_the_plan_found(
        self, graph: graphing.Graph
    ) -> None:
        """90 lines at 512 and 17 games with none; the cliff either side of it."""
        assert graph.counts["similar"] == 90
        assert graph.counts["alone"] == 17
        assert {level["cut"]: level["lines"] for level in graph.levels} == {
            256: 49, 384: 54, 512: 90, 640: 203, 768: 231,
        }
        assert {level["cut"]: level["alone"] for level in graph.levels} == {
            256: 28, 384: 25, 512: 17, 640: 13, 768: 10,
        }

    def test_louvain_finds_the_communities_the_plan_found(
        self, graph: graphing.Graph
    ) -> None:
        """13 communities over the 52 joined games, 8 of them of three games or more."""
        sizes = [one["size"] for one in graph.communities]
        assert sizes == [13, 6, 5, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2]
        assert sum(sizes) == 52
        assert len({one["name"] for one in graph.communities}) == 13

    def test_the_games_that_take_without_moving_are_one_community(
        self, graph: graphing.Graph, corpus: list[library.Game]
    ) -> None:
        """The five `strike` games a line reaches; the two it does not are in none."""
        held = self._involving(graph, corpus, "strike")
        joined = {digest: where for digest, where in held.items() if where}
        assert len(joined) == 5
        assert len(set(joined.values())) == 1
        community = self._community(graph, next(iter(joined.values())))
        assert set(community["members"]) == set(joined)
        assert community["name"] == relate.RULE_WORDS["strike"]

    def test_the_joined_edge_boards_are_one_community_of_six(
        self, graph: graphing.Graph, corpus: list[library.Game]
    ) -> None:
        """Three communities are named by joined edges; the biggest is the six boards."""
        held = self._involving(graph, corpus, "board.wrap")
        by_community: dict[str, int] = {}
        for where in held.values():
            if where:
                by_community[where] = by_community.get(where, 0) + 1
        whole = [
            self._community(graph, where)
            for where, count in by_community.items()
            if count == self._community(graph, where)["size"]
        ]
        assert sorted(one["size"] for one in whole) == [2, 3, 6]
        assert all(
            one["name"].startswith(relate.RULE_WORDS["wrap"] + " (") for one in whole
        )

    def test_a_game_with_no_line_is_in_no_community(self, graph: graphing.Graph) -> None:
        assert sum(1 for node in graph.nodes if node.community is None) == 17

    def test_the_communities_are_separate_knots(self, graph: graphing.Graph) -> None:
        """S2: every game nearer its own community's middle than any other's."""
        _assert_knots(graph)

    def test_the_far_row_holds_the_games_no_line_reaches(
        self, graph: graphing.Graph
    ) -> None:
        drawn = max(node.y for node in graph.nodes if node.community is not None)
        row = {node.y for node in graph.nodes if node.community is None}
        assert len(row) == 1
        assert row.pop() == pytest.approx(
            drawn + graphing.COMPONENT_GAP + graphing.ROW_STEP
        )

    def test_the_library_in_reverse_is_the_same_partition(
        self, corpus: list[library.Game], cache: Path, graph: graphing.Graph
    ) -> None:
        """Determinism: the order the library arrives in moves nothing, and costs no pair."""
        again = graphing.build(list(reversed(corpus)), cache)
        assert (again.computed, again.compiled) == (0, 0)
        assert again.communities == graph.communities
        assert again.similar == graph.similar

    @staticmethod
    def _involving(
        graph: graphing.Graph, corpus: list[library.Game], rule: str
    ) -> dict[str, str | None]:
        """Each node whose game involves this rule, with the community it is in."""
        specs = {one.hash: one.spec for one in corpus}
        return {
            node.hash: node.community
            for node in graph.nodes
            if rule in relate.rules_involved(relate.normal_form(specs[node.hash]))
        }

    @staticmethod
    def _community(graph: graphing.Graph, ident: str) -> dict[str, Any]:
        return next(one for one in graph.communities if one["id"] == ident)


@pytest.fixture(scope="module")
def today(
    library_games: list[library.Game], cache: Path, graph: graphing.Graph
) -> graphing.Graph:
    """The map of today's library, after `graph` has compared the plans' pairs."""
    return graphing.build(library_games, cache)


@pytest.mark.slow
class TestTheWholeLibrary:
    """The map of the library as it stands today, whatever it has grown to.

    Nothing here is a number pinned to a date: every count is read from the corpus, so a
    new game moves the map and leaves this green, while a game the map drops, a twin
    counted twice or a game with no line at all still fails.
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

    def test_every_node_carries_a_link_and_a_far_one_a_faint_one(
        self, today: graphing.Graph
    ) -> None:
        joined = {node for link in today.links for node in (link.source, link.target)}
        assert {node.hash for node in today.nodes} <= joined
        faint = {
            node for link in today.links if link.type == "nearest"
            for node in (link.source, link.target)
        }
        assert {node.hash for node in today.nodes if node.far} <= faint

    def test_the_communities_partition_the_joined_games(
        self, today: graphing.Graph
    ) -> None:
        sizes = [one["size"] for one in today.communities]
        assert sum(sizes) == sum(1 for node in today.nodes if node.community is not None)
        assert sum(1 for node in today.nodes if node.community is None) == (
            today.counts["alone"]
        )
        _assert_knots(today)
