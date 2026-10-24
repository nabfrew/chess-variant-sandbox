"""The map's picture (`web/mapping.py`), on libraries made here.

The libraries are `test_families.py`'s: families of games sharing a piece or a rule that
nothing else has, and one game kin to nobody, so which games belong together is known
before anything runs. What is under test is the method -- a deterministic t-SNE, settled
from the last picture -- not where any real game lands; `docs/archive/plan-map.md` (M3) holds the
readings on the library.
"""

from __future__ import annotations

import ast
import json
import math
import statistics
from pathlib import Path

import pytest
from test_families import CAMEL, LONER, ROOT, game, library

from chess_variant_sandbox.spec import relate
from chess_variant_sandbox.spec.likeness import LIKENESS_VERSION
from chess_variant_sandbox.web import families, mapping

MODULE = ROOT / "chess_variant_sandbox" / "web" / "mapping.py"

Point = tuple[float, float]


def build(
    specs: dict[str, dict],
    previous: families.Memory | None = None,
    order: list[str] | None = None,
) -> families.FamilyMap:
    """The map with its picture, each game's id its name (so a test reads as names)."""
    names = order or list(specs)
    return mapping.build(
        names, [relate.normal_form(specs[name]) for name in names], names, previous
    )


def diameter(points: dict[str, Point]) -> float:
    values = list(points.values())
    return max(math.dist(a, b) for a in values for b in values)


def nearest_drawn(found: families.FamilyMap, game: str) -> str:
    at = found.positions
    return min((g for g in at if g != game), key=lambda g: (math.dist(at[game], at[g]), g))


@pytest.fixture(scope="module")
def cold() -> families.FamilyMap:
    return build(library())


class TestThePicture:
    def test_every_game_has_a_finite_place(self, cold: families.FamilyMap) -> None:
        assert set(cold.positions) == set(cold.family_of)
        assert all(math.isfinite(v) for point in cold.positions.values() for v in point)
        assert cold.layout == mapping.LAYOUT_VERSION

    def test_families_are_drawn_together(self, cold: families.FamilyMap) -> None:
        """N10 on a made library: kin are near, and every game's nearest drawn game is kin."""
        within, between = [], []
        games = sorted(cold.positions)
        for i, one in enumerate(games):
            for other in games[i + 1:]:
                mine, theirs = cold.family_of[one], cold.family_of[other]
                if mine is None or theirs is None:
                    continue
                distance = math.dist(cold.positions[one], cold.positions[other])
                (within if mine == theirs else between).append(distance)
        assert statistics.fmean(within) <= statistics.fmean(between) / 2
        for game_id, family in cold.family_of.items():
            if family is not None:
                assert cold.family_of[nearest_drawn(cold, game_id)] == family, game_id

    def test_no_two_games_share_a_place(self) -> None:
        """Two games with the same words would never part without the start's nudge."""
        specs = library()
        twin = json.loads(json.dumps(specs["camel 1"]))
        twin["name"] = "camel twin"
        specs["camel twin"] = twin
        found = build(specs)
        assert found.positions["camel twin"] != found.positions["camel 1"]

    def test_a_library_of_none_one_or_two(self) -> None:
        assert mapping.place([], []) == {}
        assert mapping.place(["a"], [[0.0]]) == {"a": (0.0, 0.0)}
        two = mapping.place(["a", "b"], [[0.0, 0.5], [0.5, 0.0]])
        assert two["a"] != two["b"]

    def test_distances_must_be_square(self) -> None:
        with pytest.raises(ValueError):
            mapping.place(["a", "b"], [[0.0, 1.0]])


class TestDeterminism:
    def test_two_builds_give_one_picture(self, cold: families.FamilyMap) -> None:
        assert build(library()).positions == cold.positions

    def test_the_order_games_arrive_in_moves_nothing(self, cold: families.FamilyMap) -> None:
        specs = library()
        assert build(specs, order=list(reversed(specs))).positions == cold.positions

    def test_settled_twice_gives_one_picture(self, cold: families.FamilyMap) -> None:
        specs = library()
        specs["camel 30"] = game(30, CAMEL)
        first = build(specs, cold.memory())
        second = build(specs, cold.memory(), order=list(reversed(specs)))
        assert first.positions == second.positions


class TestMemory:
    def test_nothing_changed_nothing_moves(self, cold: families.FamilyMap) -> None:
        assert build(library(), cold.memory()).positions == cold.positions

    def test_a_new_game_lands_by_its_family_and_moves_the_others_little(
        self, cold: families.FamilyMap
    ) -> None:
        specs = library()
        specs["camel 30"] = game(30, CAMEL)
        after = build(specs, cold.memory())
        assert after.family_of[nearest_drawn(after, "camel 30")] == after.family_of["camel 1"]
        across = diameter(cold.positions)
        moved = [math.dist(cold.positions[g], after.positions[g]) / across for g in cold.positions]
        # The plan's N8 bar on the median; the furthest is bounded too.
        assert statistics.median(moved) <= 0.05
        assert max(moved) <= 0.25

    def test_a_game_leaving_moves_the_others_little(self, cold: families.FamilyMap) -> None:
        specs = library()
        del specs["zebra 6"]
        after = build(specs, cold.memory())
        across = diameter(cold.positions)
        moved = [math.dist(cold.positions[g], after.positions[g]) / across for g in after.positions]
        assert statistics.median(moved) <= 0.05
        assert max(moved) <= 0.25

    def test_a_settled_picture_is_not_turned(self, cold: families.FamilyMap) -> None:
        """Procrustes onto the last picture: north stays north, and the scale is kept."""
        specs = library()
        specs["far"] = game(98, LONER, CAMEL)
        after = build(specs, cold.memory())
        kept = [g for g in cold.positions]
        before_spread = statistics.fmean(abs(cold.positions[g][0]) for g in kept)
        after_spread = statistics.fmean(abs(after.positions[g][0]) for g in kept)
        assert after_spread == pytest.approx(before_spread, rel=0.2)
        for g in kept:
            assert math.dist(cold.positions[g], after.positions[g]) < diameter(cold.positions) / 4

    def test_another_layout_version_is_drawn_cold(self, cold: families.FamilyMap) -> None:
        specs = library()
        specs["camel 30"] = game(30, CAMEL)
        fresh = build(specs)
        memory = families.Memory(
            family_of=cold.family_of,
            next_number=cold.next_number,
            positions=cold.positions,
            layout="p0",
            likeness=LIKENESS_VERSION,
        )
        assert build(specs, memory).positions == fresh.positions

    def test_another_likeness_settles_rather_than_keeps(self, cold: families.FamilyMap) -> None:
        memory = families.Memory(
            family_of=cold.family_of,
            next_number=cold.next_number,
            positions=cold.positions,
            layout=mapping.LAYOUT_VERSION,
            likeness="k0",
        )
        settled = build(library(), memory)
        assert settled.positions != cold.positions
        assert set(settled.positions) == set(cold.positions)

    def test_the_picture_round_trips_through_the_memory_file(
        self, cold: families.FamilyMap, tmp_path: Path
    ) -> None:
        path = tmp_path / families.MAP_FILE
        families.write_memory(path, cold)
        held = json.loads(path.read_text(encoding="utf-8"))
        assert held["layout"] == mapping.LAYOUT_VERSION
        assert all(len(row["position"]) == 2 for row in held["games"].values())
        memory = families.read_memory(path)
        assert memory is not None
        assert dict(memory.positions) == cold.positions
        assert memory.layout == mapping.LAYOUT_VERSION
        assert memory.likeness == LIKENESS_VERSION
        assert build(library(), memory).positions == cold.positions

    def test_a_broken_place_is_no_place(self, tmp_path: Path) -> None:
        path = tmp_path / "map.json"
        path.write_text(
            json.dumps(
                {
                    "version": families.MAP_VERSION,
                    "layout": mapping.LAYOUT_VERSION,
                    "games": {
                        "a": {"family": None, "position": [0.0, "north"]},
                        "b": {"family": None, "position": [1.0]},
                        "c": {"family": None, "position": [1.0, 2.0]},
                    },
                }
            ),
            encoding="utf-8",
        )
        memory = families.read_memory(path)
        assert memory is not None
        assert dict(memory.positions) == {"c": (1.0, 2.0)}

    def test_a_map_without_its_picture_writes_no_place(self) -> None:
        found = families.build([], [], [])
        assert "layout" not in found.as_dict()


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

    def test_every_number_is_a_named_parameter_or_the_arithmetic_s_own(self) -> None:
        """A tunable number is a module constant the docstring accounts for; the rest is
        t-SNE's own arithmetic (the gradient's 4, halves, ones and zeros)."""
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        named: set[float] = set()
        loose: set[float] = set()
        for node in tree.body:
            numbers = {
                inner.value
                for inner in ast.walk(node)
                if isinstance(inner, ast.Constant)
                and isinstance(inner.value, (int, float))
                and not isinstance(inner.value, bool)
            }
            is_constant = isinstance(node, (ast.Assign, ast.AnnAssign)) and all(
                isinstance(target, ast.Name) and target.id.isupper()
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
            (named if is_constant else loose).update(numbers)
        assert loose <= {0, 1, 2, 4}
        constants = {
            name for name in dir(mapping) if name.isupper() and not name.startswith("_")
        }
        assert constants == {
            "LAYOUT_VERSION", "PERPLEXITY", "EXAGGERATION", "EXAGGERATED", "MOMENTUM",
            "GAIN_UP", "GAIN_DOWN", "GAIN_FLOOR", "COLD", "WARM", "RATE",
            "ENTROPY_TOLERANCE", "SEARCH", "NUDGE", "START_SCALE", "AXES_TOLERANCE",
            "AXES_LONGEST", "GOLDEN", "LIKENESS_VERSION",
        }

    def test_it_draws_no_random_number(self) -> None:
        source = MODULE.read_text(encoding="utf-8")
        assert "import random" not in source and "random." not in source
