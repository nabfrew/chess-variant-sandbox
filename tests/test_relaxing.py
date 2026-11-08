"""The map's relaxation before display (`web/relaxing.py`, Q1165): each community one place.

The measure is `relaxing.strays`: the games drawn nearer another community's middle than
their own's. Before Q1165 the real library drew 26 of the 61 games in coloured communities
(32 of 94 in any community) as strays. The threshold, set before the relaxed number was
seen, is **at most 2** strays on the real library; the fixtures here, built to be fixed,
hold zero.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from chess_variant_sandbox.web import graphing, library, relaxing
from chess_variant_sandbox.web.server import MAPPED

# Pre-registered (Q1165), before the relaxed real library was measured.
REAL_LIBRARY_STRAYS = 2


def _interleaved() -> tuple[dict[str, tuple[float, float]], dict[str, str | None]]:
    """Three communities along a line, each with a member drawn in a neighbour's knot,
    and two unusual games: what t-SNE and the communities disagreeing looks like."""
    placed: dict[str, tuple[float, float]] = {}
    family: dict[str, str | None] = {}
    noise = random.Random(1165)
    for index, name in enumerate(("a", "b", "c")):
        for k in range(6):
            digest = f"{name}{k}"
            placed[digest] = (index * 40.0 + noise.uniform(-6, 6), noise.uniform(-6, 6))
            family[digest] = name
    # Strays: a member of a drawn in c's knot, one of c in b's, one of b in a's.
    placed["a5"] = (82.0, 1.0)
    placed["c5"] = (41.0, -2.0)
    placed["b5"] = (1.0, 2.0)
    placed["u0"] = (20.0, 60.0)
    placed["u1"] = (200.0, 0.0)
    family["u0"] = family["u1"] = None
    return placed, family


def _middle(placed: dict[str, tuple[float, float]], members: list[str]) -> tuple[float, float]:
    return (
        sum(placed[d][0] for d in members) / len(members),
        sum(placed[d][1] for d in members) / len(members),
    )


class TestTheRelaxation:
    def test_the_fixture_has_strays_and_the_relaxed_picture_none(self) -> None:
        placed, family = _interleaved()
        assert len(relaxing.strays(placed, family)) == 3
        assert relaxing.strays(relaxing.relaxed(placed, family, 3.0), family) == []

    def test_the_communities_keep_their_order_and_a_lone_game_far_off_stays(self) -> None:
        """What the picture means survives: where each community is drawn, and an
        unusual game no community crowds."""
        placed, family = _interleaved()
        found = relaxing.relaxed(placed, family, 3.0)
        xs = [
            _middle(found, [d for d in found if family[d] == name])[0] for name in "abc"
        ]
        assert xs == sorted(xs)
        assert found["u1"] == placed["u1"]

    def test_no_two_groups_overlap(self) -> None:
        placed, family = _interleaved()
        found = relaxing.relaxed(placed, family, 3.0)
        for one in "abc":
            for other in "abc":
                if one < other:
                    gap = min(
                        math.dist(found[x], found[y])
                        for x in found if family[x] == one
                        for y in found if family[y] == other
                    )
                    assert gap >= 3.0

    def test_the_same_picture_in_any_order_relaxes_the_same(self) -> None:
        placed, family = _interleaved()
        backwards = dict(reversed(list(placed.items())))
        assert relaxing.relaxed(placed, family, 3.0) == relaxing.relaxed(
            backwards, family, 3.0
        )

    def test_a_pair_keeps_its_bearing(self) -> None:
        placed = {"p": (0.0, 0.0), "q": (0.0, 10.0)}
        family: dict[str, str | None] = {"p": "f", "q": "f"}
        found = relaxing.relaxed(placed, family, 3.0)
        assert found["q"][1] > found["p"][1]
        assert abs(found["q"][0] - found["p"][0]) < 1e-6


def test_the_toy_library_s_map_has_no_strays(tmp_path: Path) -> None:
    from test_graphing import LIBRARY, built

    drawn = built(LIBRARY, tmp_path)
    placed = {node.hash: (node.x, node.y) for node in drawn.nodes}
    family = {node.hash: node.community for node in drawn.nodes}
    assert relaxing.strays(placed, family) == []


@pytest.mark.slow
def test_the_real_library_s_map_has_at_most_the_pre_registered_strays(tmp_path: Path) -> None:
    """The whole committed and archived library, drawn cold (about ten seconds)."""
    root = Path(__file__).resolve().parents[1]
    games = [
        one
        for one in library.discover(root / "rulesets", tmp_path / "runs", root / "archive")
        if one.source in MAPPED
    ]
    drawn = graphing.build(games, tmp_path / "runs")
    placed = {node.hash: (node.x, node.y) for node in drawn.nodes}
    family = {node.hash: node.community for node in drawn.nodes}
    assert len(relaxing.strays(placed, family)) <= REAL_LIBRARY_STRAYS
