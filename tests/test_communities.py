"""Louvain communities, as `web/communities.py` finds them for the map.

The graphs here are written by hand, because what is under test is the method and its
determinism rather than any library. `tests/test_graphing.py` holds what the method finds
on the committed corpus.
"""

from __future__ import annotations

import ast
import itertools
import random
from pathlib import Path

import pytest

from chess_variant_sandbox.web import communities


def clique(names: str, weight: float = 1.0) -> dict[tuple[str, str], float]:
    return {pair: weight for pair in itertools.combinations(names, 2)}


# Two tight triangles and a square, joined to each other by one weak line each.
BARBELL = {
    **clique("abc", 3.0),
    **clique("def", 3.0),
    **clique("ghij", 2.0),
    ("c", "d"): 0.5,
    ("f", "g"): 0.5,
}


class TestWhatItFinds:
    def test_tight_groups_joined_by_weak_lines_are_separate_communities(self) -> None:
        assert communities.partition(BARBELL) == [
            ["g", "h", "i", "j"], ["a", "b", "c"], ["d", "e", "f"],
        ]

    def test_the_answer_is_biggest_first_and_each_community_sorted(self) -> None:
        found = communities.partition(BARBELL)
        assert [len(part) for part in found] == sorted(
            (len(part) for part in found), reverse=True
        )
        assert all(part == sorted(part) for part in found)

    def test_a_node_no_line_touches_is_in_no_community(self) -> None:
        found = communities.partition({**BARBELL, ("x", "y"): 0.0})
        assert "x" not in {node for part in found for node in part}

    def test_nothing_to_join_is_no_community(self) -> None:
        assert communities.partition({}) == []

    def test_two_islands_are_never_one_community(self) -> None:
        """Leiden's guarantee, applied after Louvain: a community is connected."""
        found = communities.split_disconnected(
            [["a", "b", "c", "d"]], {("a", "b"): 1.0, ("c", "d"): 1.0}
        )
        assert found == [["a", "b"], ["c", "d"]]

    def test_the_partition_it_keeps_is_the_best_of_its_runs(self) -> None:
        weights = communities.canonical(BARBELL)
        best = communities.modularity(communities.partition(BARBELL), weights)
        nodes = sorted({node for pair in weights for node in pair})
        for seed in range(20):
            run = communities.louvain(nodes, weights, random.Random(seed))
            assert communities.modularity(run, weights) <= best + 1e-12

    def test_modularity_is_newman_s(self) -> None:
        """Two disjoint edges, each its own community: 2 x (1/2 - (2/4)^2) = 0.5."""
        weights = {("a", "b"): 1.0, ("c", "d"): 1.0}
        assert communities.modularity([["a", "b"], ["c", "d"]], weights) == pytest.approx(0.5)
        assert communities.modularity([["a", "b", "c", "d"]], weights) == pytest.approx(0.0)


class TestDeterminism:
    def test_the_same_graph_listed_any_way_gives_the_same_partition(self) -> None:
        expected = communities.partition(BARBELL)
        pairs = list(BARBELL.items())
        for seed in range(5):
            random.Random(seed).shuffle(pairs)
            flipped = {
                ((second, first) if index % 2 else (first, second)): weight
                for index, ((first, second), weight) in enumerate(pairs)
            }
            assert communities.partition(flipped) == expected

    def test_it_draws_nothing_from_the_module_level_generator(self) -> None:
        """Every run has a generator of its own; the shared one is never touched."""
        random.seed(1)
        before = random.getstate()
        communities.partition(BARBELL)
        assert random.getstate() == before

    def test_the_module_calls_random_only_through_a_seeded_generator(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "chess_variant_sandbox/web/communities.py"
        ).read_text(encoding="utf-8")
        calls = {
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "random"
        }
        assert calls == {"Random"}

    def test_a_tie_between_runs_is_settled_by_the_partition_not_the_run(self) -> None:
        """A six-cycle of equal lines splits four ways, all equally good.

        Into three pairs two ways or two triples two ways, every one at modularity 1/6,
        and the runs find all four. Which run found which first must not decide the
        answer: the tie goes to fewer communities, then to the split whose sorted member
        lists sort first.
        """
        ring = {
            (first, second): 1.0
            for first, second in zip("abcdef", "bcdefa", strict=True)
        }
        weights = communities.canonical(ring)
        splits = [
            [["a", "b"], ["c", "d"], ["e", "f"]],
            [["a", "f"], ["b", "c"], ["d", "e"]],
            [["a", "b", "c"], ["d", "e", "f"]],
            [["a", "b", "f"], ["c", "d", "e"]],
        ]
        for split in splits:
            assert communities.modularity(split, weights) == pytest.approx(1 / 6)
        assert communities.partition(ring) == [["a", "b", "c"], ["d", "e", "f"]]
