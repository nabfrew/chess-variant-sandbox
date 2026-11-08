"""`calibration`'s anchoring half, pinned where `test_band_placed_ends.py` leaves it loose.

`anchor_curves` is how `v3` to `v11` placed their ends and stays so they can be re-derived
(`docs/reference/negatives-widened.py`); `check_ends` is `anchor`'s report since `v12`. Each
test builds its corpus inline and states the ends it expects as literals worked by hand.
"""

from __future__ import annotations

import dataclasses

import pytest

from chess_variant_sandbox.metrics.calibration import (
    EndCheck,
    Measured,
    _clamp,
    anchor_curves,
    check_ends,
)
from chess_variant_sandbox.metrics.scoring import Curve

HUMAN = [10.0, 11.0, 12.0, 13.0, 14.0, 16.0]


def _humans(name: str, values: list[float]) -> list[Measured]:
    return [Measured(f"h{n}", "human", {name: v, "ladder_monotonicity": 1.0})
            for n, v in enumerate(values)]


class TestTheDomain:
    """`DOMAIN` is load-bearing: `_clamp` holds every end `anchor_curves` writes inside it.

    A band 0.4 wide spends a floor of 0.8, so its zero ends land at -0.1 and 1.1 and only
    the domain brings them back; each case below is placed to reach past both bounds.
    """

    SHARE = [0.3, 0.38, 0.46, 0.54, 0.62, 0.7]

    @pytest.mark.parametrize("name", [
        "opening_diversity", "distinct_game_rate", "blunder_recoverability",
        "decision_criticality", "piece_utilisation", "progress_irreversibility",
        "instance_participation", "lead_conversion",
    ])
    def test_a_share_is_held_to_zero_and_one(self, name: str) -> None:
        points, _ = anchor_curves(_humans(name, self.SHARE), {name: {"curve": "band"}})
        assert points == {name: {
            "good_low": 0.3, "good_high": 0.7, "zero_low": 0.0, "zero_high": 1.0,
        }}

    def test_gradient_persistence_is_held_to_zero_and_two_and_a_half(self) -> None:
        name = "gradient_persistence"
        points, _ = anchor_curves(_humans(name, [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]),
                                  {name: {"curve": "band"}})
        # Width 1.5, floor 3.0: unclamped the zero ends would be -1.0 and 3.5.
        assert points == {name: {
            "good_low": 0.5, "good_high": 2.0, "zero_low": 0.0, "zero_high": 2.5,
        }}

    def test_draw_slope_is_held_to_zero_and_two(self) -> None:
        name = "draw_slope"
        points, _ = anchor_curves(_humans(name, [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]),
                                  {name: {"curve": "band"}})
        # Width 1.0, floor 2.0: unclamped the zero ends would be -0.5 and 2.5.
        assert points == {name: {
            "good_low": 0.5, "good_high": 1.5, "zero_low": 0.0, "zero_high": 2.0,
        }}


class TestTheRecords:
    def test_measured_and_endcheck_are_frozen(self) -> None:
        entry = Measured("a", "human", {})
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.label = "worse"  # type: ignore[misc]
        check = EndCheck(component="x", kind="band", ends={}, sources={}, games=0,
                         plateau=0, zero=0, low=None, high=None, past={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            check.games = 1  # type: ignore[misc]

    def test_a_record_with_no_ladder_reads_as_a_clean_one(self) -> None:
        assert Measured("a", "human", {"stall_rate": 0.1}).ladder == 1.0

    def test_an_end_is_written_to_four_places(self) -> None:
        assert _clamp("mean_branching", 0.123456) == 0.1235
        assert _clamp("no_such_component", 1.234567) == 1.2346
        assert _clamp("stall_rate", 1.5) == 1.0


class TestWhoTheAnchorReads:
    def test_it_reads_past_negatives_held_out_ladders_and_absent_components(self) -> None:
        """A negative first, a held-out ladder next, one human at exactly the measurable
        ladder (0.5, kept), and a component nobody measured ahead of one that is."""
        corpus = [
            Measured("w0", "worse", {"mean_branching": 1.0}),
            Measured("crooked", "human", {"mean_branching": 50.0, "ladder_monotonicity": 0.4}),
            *(Measured(f"h{n}", "human",
                       {"mean_branching": v, "ladder_monotonicity": 0.5 if n == 2 else 1.0})
              for n, v in enumerate(HUMAN)),
        ]
        components = {"stall_rate": {"curve": "band"},
                      "mean_branching": {"curve": "higher_better"}}
        points, held = anchor_curves(corpus, components)
        # Width 6: good_at 16 + 0.35*6; the negatives' median 1.0 is below good_at - 12.
        assert points == {"mean_branching": {"good_at": 18.1, "zero_at": 1.0}}
        assert held == {"mean_branching": ["crooked"]}


class TestTheSpanFloors:
    def test_a_band_noise_wider_than_two_widths_sets_the_zero_ends(self) -> None:
        points, _ = anchor_curves(_humans("mean_branching", HUMAN),
                                  {"mean_branching": {"curve": "band"}},
                                  noise={"mean_branching": 5.0})
        # Floor 15 (3 x 5.0) over 12 (2 x width): zero ends 7.5 out, no fallback.
        assert points["mean_branching"] == {
            "good_low": 10.0, "good_high": 16.0, "zero_low": 2.5, "zero_high": 23.5,
        }

    def test_a_component_the_noise_map_does_not_name_spends_two_widths(self) -> None:
        points, _ = anchor_curves(_humans("mean_branching", [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
                                  {"mean_branching": {"curve": "band"}},
                                  noise={"ply_spread": 9.0})
        assert points["mean_branching"] == {
            "good_low": 1.0, "good_high": 1.5, "zero_low": 0.5, "zero_high": 2.0,
        }

    def test_a_band_of_one_value_gets_zero_ends_a_twentieth_out(self) -> None:
        points, _ = anchor_curves(_humans("mean_branching", [5.0] * 6),
                                  {"mean_branching": {"curve": "band"}})
        # The 1e-6 floor rounds away, the zero ends land on the good ones, and the
        # fallback moves each out by max(width, 0.05).
        assert points["mean_branching"] == {
            "good_low": 5.0, "good_high": 5.0, "zero_low": 4.95, "zero_high": 5.05,
        }

    def test_a_band_of_one_value_still_spends_the_minimum_width(self) -> None:
        points, _ = anchor_curves(_humans("mean_branching", [0.00004999] * 6),
                                  {"mean_branching": {"curve": "lower_better"}})
        # Width 1e-6 spends a floor of 2e-6: zero_at 0.00005199, which rounds up.
        assert points["mean_branching"] == {"good_at": 0.0, "zero_at": 0.0001}


class _Weights:
    def __init__(self, curves: dict[str, Curve]) -> None:
        self.curves = curves
        self.sources: dict[str, dict[str, str]] = {}

    def components(self) -> list[str]:
        return sorted(self.curves)


class TestCheckEnds:
    def test_a_skill_derived_curve_reads_only_clean_ladders(self) -> None:
        ladders = {"crooked": 0.9, "edge": 0.99, "clean": 1.0, "half": 0.5, "low": 0.4}
        corpus = [Measured(n, "human", {"skill_gradient": 5.0, "mean_branching": 5.0,
                                        "ladder_monotonicity": ladder})
                  for n, ladder in ladders.items()]
        corpus.append(Measured("unladdered", "human",
                               {"skill_gradient": 5.0, "mean_branching": 5.0}))
        curve = Curve(kind="higher_better", points={"zero_at": 0.0, "good_at": 10.0})
        checks = {c.component: c for c in check_ends(
            corpus, _Weights({"skill_gradient": curve, "mean_branching": curve}))}
        assert (checks["skill_gradient"].games, checks["skill_gradient"].unreadable) == (
            3, ["crooked", "half", "low"])
        assert (checks["mean_branching"].games, checks["mean_branching"].unreadable) == (
            5, ["low"])

    def test_past_names_the_games_strictly_beyond_each_end(self) -> None:
        stall = [0.05, 0.3, 0.5, 0.7, 0.95]
        spread = [0.123456, 0.2, 0.5, 0.8, 0.987654]
        corpus = [Measured(n, "human", {"stall_rate": s, "mean_branching": v,
                                        "ply_spread": v, "ladder_monotonicity": 1.0})
                  for n, s, v in zip("abcde", stall, spread, strict=True)]
        weights = _Weights({
            "stall_rate": Curve(kind="band", points={
                "zero_low": 0.1, "good_low": 0.3, "good_high": 0.7, "zero_high": 0.9}),
            "mean_branching": Curve(kind="higher_better",
                                    points={"zero_at": 0.2, "good_at": 0.8}),
            "ply_spread": Curve(kind="lower_better", points={"good_at": 0.2, "zero_at": 0.8}),
        })
        checks = {c.component: c for c in check_ends(corpus, weights)}
        # A game sitting exactly on an end is not past it.
        assert checks["stall_rate"].past == {
            "zero_low": ["a"], "good_low": ["a"], "good_high": ["e"], "zero_high": ["e"]}
        assert checks["mean_branching"].past == {"zero_at": ["a"], "good_at": ["e"]}
        assert checks["ply_spread"].past == {"good_at": ["a"], "zero_at": ["e"]}
        # n = 5, so k = 1: the extremes, rounded to four places.
        assert (checks["mean_branching"].low, checks["mean_branching"].high) == (0.1235, 0.9877)
        assert (checks["stall_rate"].low, checks["stall_rate"].high) == (0.05, 0.95)
