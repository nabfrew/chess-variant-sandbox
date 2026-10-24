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


def settled(
    weights: dict[tuple[str, str], float],
    start: list[list[str]],
    known: str | list[str] = "",
    nodes: str | list[str] = "",
) -> list[list[str]]:
    everyone = set(nodes) | {node for pair in weights for node in pair}
    found = communities.settle(sorted(everyone), weights, start, list(known))
    return sorted(sorted(part) for part in found)


# One heavy triangle and one light pair, with `x` tied to both.
PULLED = {
    **clique("abc", 3.0),
    ("p", "q"): 1.0,
    ("x", "p"): 0.2,
    ("x", "a"): 3.0,
    ("x", "b"): 3.0,
}


class TestSettling:
    """Communities settled from a partition already held (`docs/archive/plan-map.md` §4, M2b)."""

    def test_a_start_that_is_already_best_does_not_move(self) -> None:
        best = communities.partition(BARBELL)
        assert settled(BARBELL, best, "abcdefghij") == sorted(best)

    def test_a_node_stays_while_one_line_holds_it_whatever_pulls_elsewhere(self) -> None:
        """Louvain alone would move `x` to the triangle; a line to `p` keeps it put."""
        start = [["a", "b", "c"], ["p", "q", "x"]]
        assert settled(PULLED, start, "abcpqx") == [["a", "b", "c"], ["p", "q", "x"]]
        cold = communities.partition(PULLED)
        assert ["a", "b", "c", "x"] in sorted(sorted(part) for part in cold)

    def test_a_node_no_line_holds_is_placed_afresh(self) -> None:
        weights = {pair: w for pair, w in PULLED.items() if pair != ("x", "p")}
        start = [["a", "b", "c"], ["p", "q", "x"]]
        assert settled(weights, start, "abcpqx") == [["a", "b", "c", "x"], ["p", "q"]]

    def test_a_known_node_joins_a_community_only_through_two_lines(self) -> None:
        one = {**clique("abc", 3.0), ("x", "a"): 1.0}
        assert settled(one, [["a", "b", "c"]], "abcx") == [["a", "b", "c"], ["x"]]
        two = {**one, ("x", "b"): 1.0}
        assert settled(two, [["a", "b", "c"]], "abcx") == [["a", "b", "c", "x"]]

    def test_a_new_node_joins_through_one_line(self) -> None:
        one = {**clique("abc", 3.0), ("x", "a"): 1.0}
        assert settled(one, [["a", "b", "c"]], "abc") == [["a", "b", "c", "x"]]

    def test_a_known_node_pairs_with_a_lone_node_through_one_line(self) -> None:
        weights = {**clique("abc", 3.0), ("x", "y"): 1.0}
        assert settled(weights, [["a", "b", "c"]], "abcxy") == [
            ["a", "b", "c"], ["x", "y"],
        ]

    def test_a_community_left_in_two_pieces_is_cut_in_two(self) -> None:
        weights = {**clique("abc", 1.0), **clique("def", 1.0)}
        assert settled(weights, [["a", "b", "c", "d", "e", "f"]], "abcdef") == [
            ["a", "b", "c"], ["d", "e", "f"],
        ]

    def test_two_held_communities_are_never_merged_whole(self) -> None:
        """A ring of thirty triangles: folding would merge neighbours, settling never does."""
        triangles = [[f"t{k:02}{corner}" for corner in "abc"] for k in range(30)]
        weights: dict[tuple[str, str], float] = {}
        for k, triangle in enumerate(triangles):
            weights.update(clique(triangle))
            weights[(triangle[2], triangles[(k + 1) % 30][0])] = 1.0
        nodes = [node for triangle in triangles for node in triangle]
        assert settled(weights, triangles, nodes) == sorted(triangles)

    def test_a_node_with_no_line_is_alone(self) -> None:
        assert settled(clique("abc"), [["a", "b", "c", "z"]], "abcz", "abcz") == [
            ["a", "b", "c"], ["z"],
        ]

    def test_a_name_that_is_no_node_is_ignored(self) -> None:
        start = [["a", "b", "c", "zz"], ["d", "e", "f"], ["g", "h", "i", "j"]]
        assert ["a", "b", "c"] in settled(BARBELL, start, "abcdefghij")

    def test_a_node_in_two_starting_communities_is_refused(self) -> None:
        with pytest.raises(ValueError, match="two starting"):
            settled(BARBELL, [["a", "b"], ["b", "c"]])

    def test_settling_reads_no_listing_order_and_draws_no_random_number(self) -> None:
        random.seed(3)
        before = random.getstate()
        start = [["a", "b", "c", "d"], ["e", "f"]]
        first = settled(BARBELL, start, "abcdef")
        assert random.getstate() == before
        listed = dict(reversed([((b, a), w) for (a, b), w in BARBELL.items()]))
        found = communities.settle(
            sorted("jihgfedcba", reverse=True), listed, [["f", "e"], ["d", "c", "b", "a"]], "fedcba"
        )
        assert sorted(sorted(part) for part in found) == first
