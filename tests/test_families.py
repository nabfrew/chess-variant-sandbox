"""The map's families and their memory (`web/families.py`), on libraries made here.

Every library is made in this file from one small game and a few pieces nothing else
has, so which games are kin is known before anything runs. What is under test is the
method -- the mutual graph, settling from a previous map, the names, the memory -- not
where any real game lands; `docs/archive/plan-map.md` (M2) holds the readings on the library.
"""

from __future__ import annotations

import ast
import copy
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from chess_variant_sandbox.spec import relate, validate_spec
from chess_variant_sandbox.web import families

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "chess_variant_sandbox" / "web" / "families.py"

BASE: dict = {
    "schema_version": 3,
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
            "moves": [
                {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
                {"kind": "leap", "vectors": [[1, 1], [-1, 1]], "mode": "capture"},
            ],
        },
        {
            "id": "R",
            "name": "tower",
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
    "rules": {"promotion": {"piece": "W", "rank": 5, "to": ["R"]}},
}


def leaper(pid: str, name: str, vector: list[int]) -> dict:
    return {
        "id": pid,
        "name": name,
        "moves": [{"kind": "leap", "vectors": [vector], "symmetry": "dihedral8"}],
    }


# Pieces nothing else has: each family below is the games that carry one of them.
CAMEL = leaper("C", "camel", [1, 3])
ZEBRA = leaper("Z", "zebra", [2, 3])
GIRAFFE = leaper("G", "giraffe", [1, 4])
OKAPI = leaper("O", "okapi", [3, 4])
LONER = leaper("L", "loner", [2, 5])
SQUARES = ("b1", "d1", "e1")


def game(number: int, *pieces: dict, rules: dict | None = None) -> dict:
    """`BASE` with these pieces added, these rules, and a game length its own.

    The length (`max_plies`) is a word no other game has, so every game is distinct and
    what a family shares is only what it was made to share.
    """
    spec = copy.deepcopy(BASE)
    spec["name"] = f"game {number}"
    spec["rules"]["max_plies"] = 200 + number
    spec["rules"].update(rules or {})
    for piece, square in zip(pieces, SQUARES, strict=False):
        spec["pieces"].append(copy.deepcopy(piece))
        spec["setup"]["placements"].append({"piece": piece["id"], "squares": [square]})
    assert validate_spec(spec) == [], spec["name"]
    return spec


def library() -> dict[str, dict]:
    """Three families of four, one of three by a rule, and one game kin to nobody."""
    specs: dict[str, dict] = {}
    number = 0
    for piece in (CAMEL, ZEBRA, GIRAFFE):
        for _ in range(4):
            number += 1
            specs[f"{piece['name']} {number}"] = game(number, piece)
    # The rule's family: each game has a piece of its own, so what all three share is the
    # rule and an army the size of everybody else's.
    for length in (2, 3, 4):
        number += 1
        own = leaper("U", f"runner {length}", [0, length])
        specs[f"rule {number}"] = game(number, own, rules={"must_capture": True})
    specs["loner"] = game(99, LONER)
    return specs


def build(
    specs: dict[str, dict], previous: families.Memory | None = None, order: list[str] | None = None
) -> families.FamilyMap:
    """The families of these specs, each game's id its name (so a test reads as names)."""
    names = order or list(specs)
    return families.build(
        names, [relate.normal_form(specs[name]) for name in names], names, previous
    )


def kin(found: families.FamilyMap) -> set[frozenset[str]]:
    return {frozenset(family.members) for family in found.families}


def fellows(found: families.FamilyMap, game: str) -> frozenset[str]:
    family = found.family_of[game]
    if family is None:
        return frozenset()
    return frozenset(other for other, held in found.family_of.items() if held == family)


def made(*bags: set[str]) -> families.Likeness:
    """A relation over word sets written here (only `words` is read when naming)."""
    return families.Likeness(
        words=tuple(frozenset(bag) for bag in bags),
        keys=tuple(str(n) for n in range(len(bags))),
        weights={},
        totals=(),
    )


def holders(relation: families.Likeness) -> Counter[str]:
    return Counter(word for bag in relation.words for word in bag)


class TestWhatItFinds:
    def test_the_made_families_are_found_and_the_loner_is_unusual(self) -> None:
        found = build(library())
        assert kin(found) == {
            frozenset({"camel 1", "camel 2", "camel 3", "camel 4"}),
            frozenset({"zebra 5", "zebra 6", "zebra 7", "zebra 8"}),
            frozenset({"giraffe 9", "giraffe 10", "giraffe 11", "giraffe 12"}),
            frozenset({"rule 13", "rule 14", "rule 15"}),
        }
        assert [game for game, _ in found.unusual] == ["loner"]
        assert found.family_of["loner"] is None

    def test_a_mutual_line_joins_two_games_each_among_the_other_s_nearest(self) -> None:
        found = build(library())
        for first, second in found.lines:
            assert second in found.neighbours[first] and first in found.neighbours[second]
            assert first < second
        assert all(0 < weight <= 1 for weight in found.lines.values())

    def test_the_unusual_are_ranked_furthest_first(self) -> None:
        specs = library()
        specs["far"] = game(98, LONER, GIRAFFE)
        found = build(specs)
        distances = [distance for _, distance in found.unusual]
        assert distances == sorted(distances, reverse=True)

    def test_a_library_of_one_or_none_has_no_family(self) -> None:
        one = {"only": game(1, CAMEL)}
        found = build(one)
        assert found.families == () and found.family_of == {"only": None}
        assert families.build([], [], []).families == ()


class TestTheNames:
    def test_a_family_sharing_a_rule_is_named_in_the_rule_s_words(self) -> None:
        found = build(library())
        named = {frozenset(family.members): family for family in found.families}
        ruled = named[frozenset({"rule 13", "rule 14", "rule 15"})]
        assert ruled.word is not None and ruled.word.startswith(("rule:", "set:"))
        assert ruled.name == relate.RULE_WORDS["must_capture"]

    def test_a_family_sharing_a_piece_is_named_by_what_its_games_call_it(self) -> None:
        found = build(library())
        named = {family.name for family in found.families}
        assert {"the camel", "the zebra", "the giraffe"} <= named

    def test_a_piece_everybody_has_made_different_is_named_by_what_was_changed(self) -> None:
        """Three games whose tower may take only the walker: not *the tower*."""
        specs = library()
        for number in (40, 41, 42):
            spec = game(number, leaper("U", f"runner {number}", [number - 37, number - 37]))
            tower = next(piece for piece in spec["pieces"] if piece["id"] == "R")
            tower["captures"] = ["W"]
            assert validate_spec(spec) == []
            specs[f"picky {number}"] = spec
        found = build(specs)
        picky = next(f for f in found.families if "picky 40" in f.members)
        assert set(picky.members) == {"picky 40", "picky 41", "picky 42"}
        assert picky.name == "who may take whom"

    def test_a_word_most_of_the_library_has_names_nothing(self) -> None:
        """Every game has the walker: it marks no family out, though each family has it."""
        assert all(family.name != "the walker" for family in build(library()).families)
        relation = made(*[{"rule:a"}] * 6)
        assert families._word(relation, holders(relation), [0, 1]) is None

    def test_a_word_every_member_has_comes_before_one_that_marks_them_out_more(self) -> None:
        relation = made(
            {"rule:a", "piece:b"}, {"rule:a", "piece:b"}, {"rule:a"},
            {"rule:a"}, {"rule:a"}, {"rule:a"}, set(), set(),
        )
        assert families._word(relation, holders(relation), [0, 1, 2]) == "rule:a"

    def test_a_word_only_one_of_two_has_names_nothing(self) -> None:
        relation = made({"piece:b"}, set(), set(), set())
        assert families._word(relation, holders(relation), [0, 1]) is None

    def test_a_family_no_word_marks_out_is_named_near_its_centre(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(families, "_word", lambda *_: None)
        found = build(library())
        assert found.families
        for family in found.families:
            assert family.word is None and family.name == f"near {family.centre}"

    def test_a_draw_clock_never_names_a_family(self) -> None:
        assert families._clock("set:max_plies=200")
        assert families._clock("rule:repetition_draw")
        assert not families._clock("rule:drops")
        assert not families._clock("piece:(...)")

    def test_no_name_is_a_token(self) -> None:
        """N9: what a reader sees is words, never `rule:`, `<piece>` or a schema key."""
        for found in (build(library()),):
            for family in found.families:
                assert not any(mark in family.name for mark in (":", "<", ">", "_", "="))

    @pytest.mark.parametrize(
        ("word", "said"),
        [
            ("rule:explode_on_capture", relate.RULE_WORDS["explode_on_capture"]),
            ("set:castling.king_steps=2", relate.RULE_WORDS["castling"]),
            ("set:promotion[].to[]=<piece>", relate.RULE_WORDS["promotion"]),
            ("rule:board.wrap", relate.RULE_WORDS["wrap"]),
            ("lattice:hex", "a hex board"),
            ("size:5x6", "a 5 by 6 board"),
            ("army:64", "64 pieces at the start"),
            ("set:win='annihilation'", "annihilation (how the game is won)"),
            ("set:scoring.count='region'", relate.RULE_WORDS["scoring"]),
        ],
    )
    def test_every_kind_of_word_has_words(self, word: str, said: str) -> None:
        assert families.words_for(word, []) == said

    @pytest.mark.parametrize(
        ("wrap", "said"),
        [
            ((("both", "both"), ("both", "both")), "a board whose opposite edges all join"),
            ((("both", "both"), ("none", "none")), "a board with two opposite edges joined"),
            ((("left", "left"), ("none", "none")), "a board with two opposite edges joined, one way"),
        ],
    )
    def test_joined_edges_are_said_by_their_shape(self, wrap: tuple, said: str) -> None:
        assert families._wrap_words(wrap) == said

    def test_an_army_on_every_square_is_said_as_a_full_board(self) -> None:
        spec = game(1, CAMEL, ZEBRA, GIRAFFE)
        spec["board"]["height"] = 4
        spec["rules"]["promotion"]["rank"] = 3
        assert validate_spec(spec) == []
        form = relate.normal_form(spec)
        assert families.words_for(f"army:{len(form.cells)}", [form]) == (
            "every square filled at the start"
        )
        assert families.words_for("army:12", [form]) == "12 pieces at the start"


class TestDeterminism:
    def test_two_builds_give_one_answer(self) -> None:
        assert build(library()).as_dict() == build(library()).as_dict()

    def test_the_order_games_arrive_in_moves_nothing(self) -> None:
        specs = library()
        expected = build(specs).as_dict()
        for seed in range(3):
            order = list(specs)
            random.Random(seed).shuffle(order)
            assert build(specs, order=order).as_dict() == expected

    def test_the_order_games_arrive_in_moves_nothing_when_settled(self) -> None:
        specs = library()
        memory = build(specs).memory()
        specs["camel 20"] = game(20, CAMEL)
        expected = build(specs, memory).as_dict()
        order = sorted(specs, reverse=True)
        assert build(specs, memory, order=order).as_dict() == expected

    def test_a_game_listed_twice_is_refused(self) -> None:
        form = relate.normal_form(game(1, CAMEL))
        with pytest.raises(ValueError, match="twice"):
            families.build(["a", "a"], [form, form], ["a", "a"])


class TestMemory:
    def test_settled_from_itself_a_map_is_unchanged(self) -> None:
        found = build(library())
        again = build(library(), found.memory())
        assert again.as_dict() == found.as_dict() and again.warm and not found.warm

    def test_a_one_rule_variant_of_a_family_moves_nobody_outside_it(self) -> None:
        """The box's acceptance: a new game joins its relatives and nobody else moves."""
        specs = library()
        before = build(specs)
        variant = copy.deepcopy(specs["zebra 6"])
        variant["name"] = "zebra variant"
        variant["rules"]["check_limit"] = 3
        assert validate_spec(variant) == []
        specs["zebra variant"] = variant
        after = build(specs, before.memory())
        zebras = fellows(before, "zebra 6")
        assert after.family_of["zebra variant"] == before.family_of["zebra 6"]
        for name in before.family_of:
            if name not in zebras:
                assert fellows(after, name) == fellows(before, name), name
                assert after.family_of[name] == before.family_of[name], name

    def test_a_game_leaving_moves_nobody_outside_its_family(self) -> None:
        """N4 on a made library: each game left out in turn, nobody else changes kin."""
        specs = library()
        before = build(specs)
        for out in list(specs):
            rest = {name: spec for name, spec in specs.items() if name != out}
            after = build(rest, before.memory())
            for name in rest:
                assert fellows(after, name) - {out} == fellows(before, name) - {out}, (out, name)

    def test_a_family_keeps_its_id_while_it_keeps_most_of_its_members(self) -> None:
        specs = library()
        before = build(specs)
        del specs["camel 1"]
        after = build(specs, before.memory())
        assert after.family_of["camel 2"] == before.family_of["camel 2"]

    def test_a_new_family_takes_a_number_no_family_has_held(self) -> None:
        specs = library()
        first = build(specs)
        giraffes = {name: specs.pop(name) for name in list(specs) if name.startswith("giraffe")}
        second = build(specs, first.memory())
        gone = first.family_of["giraffe 9"]
        assert gone not in set(second.family_of.values())
        specs.update(giraffes)
        third = build(specs, second.memory())
        returned = third.family_of["giraffe 9"]
        assert returned is not None and returned != gone
        assert returned not in set(first.family_of.values())

    def test_a_new_game_starts_alone_and_finds_its_family(self) -> None:
        specs = library()
        before = build(specs)
        specs["camel 30"] = game(30, CAMEL)
        after = build(specs, before.memory())
        assert after.family_of["camel 30"] == before.family_of["camel 1"]

    def test_the_memory_round_trips_through_its_file(self, tmp_path: Path) -> None:
        found = build(library())
        path = tmp_path / families.MAP_FILE
        families.write_memory(path, found)
        held = json.loads(path.read_text(encoding="utf-8"))
        assert held["version"] == families.MAP_VERSION
        memory = families.read_memory(path)
        assert memory is not None
        assert dict(memory.family_of) == found.family_of
        assert memory.next_number == found.next_number
        assert build(library(), memory).as_dict() == found.as_dict()

    def test_no_file_another_version_or_a_broken_file_is_no_memory(self, tmp_path: Path) -> None:
        path = tmp_path / "map.json"
        assert families.read_memory(path) is None
        path.write_text("{not json", encoding="utf-8")
        assert families.read_memory(path) is None
        path.write_text(json.dumps({"version": "m0", "games": {}}), encoding="utf-8")
        assert families.read_memory(path) is None


class TestNoSnapshotConstants:
    """N6: no constant names a game or is read off a library."""

    def test_no_ruleset_name_is_in_the_module(self) -> None:
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        strings = {
            node.value.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        names = {
            json.loads(path.read_text(encoding="utf-8")).get("name", "").lower()
            for path in (ROOT / "rulesets").glob("*.json")
        }
        names.discard("")
        assert not {
            name for name in names for string in strings if len(name) > 3 and name in string
        }

    def test_its_numbers_are_the_plan_s_parameters(self) -> None:
        """The mutual graph's five, `MARKS`' half, and the arithmetic's own 0, 1 and 2."""
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        numbers = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        }
        assert numbers <= {0, 1, 2, 5, 0.5}
