"""The pure readings of `docs/reference/score-dials-other-ends.py` (R-S13, its record's §1.3 to
§1.6), on hand-built games: no stored run is read."""

from __future__ import annotations

import math

import private_paths
import pytest

SCRIPT = "docs/reference/score-dials-other-ends.py"
ends_script = private_paths.script(SCRIPT, "score_dials_other_ends")
pytestmark = private_paths.needs(SCRIPT)


def _game(name: str, value: float, *, ladder: float = 1.0, gradient_runs: int = 4) -> dict:
    raw = {c: value for c in ends_script.REFERENCE}
    raw["ladder_monotonicity"] = ladder
    runs = [{"raw": {"skill_gradient": value, "ladder_monotonicity": ladder}}
            for _ in range(gradient_runs)]
    return {"name": name, "raw": raw, "runs": runs}


def test_quantiles_take_the_kth_from_each_end() -> None:
    # n = 15: k = 2, the second-lowest and second-highest.
    low, high, n, k = ends_script.quantiles(float(i) for i in range(15))
    assert (low, high, n, k) == (1.0, 13.0, 15, 2)
    # n = 10: k = 1, the extremes.
    assert ends_script.quantiles(float(i) for i in range(10))[:2] == (0.0, 9.0)


def test_derived_ends_follow_v13s_sources() -> None:
    assert ends_script.derive_points("mean_branching", 8.0, 60.0) == {
        "good_low": 8.0, "good_high": 60.0, "zero_high": 120.0}
    assert ends_script.derive_points("piece_utilisation", 0.8, 0.95)["zero_at"] == pytest.approx(0.65)
    assert ends_script.derive_points("skill_gradient", 1.0, 150.0) == {"good_at": 150.0}
    assert ends_script.derive_points("lead_conversion", 0.6, 0.9) == {"good_low": 0.6, "good_high": 0.9}


def test_a1_needs_three_clean_runs() -> None:
    runs = [{"raw": {"skill_gradient": 100.0, "ladder_monotonicity": 1.0}}] * 2
    assert ends_script.a1_gradient(runs) is None
    runs = runs + [{"raw": {"skill_gradient": 130.0, "ladder_monotonicity": 0.99}},
                   {"raw": {"skill_gradient": 999.0, "ladder_monotonicity": 0.9}}]
    assert ends_script.a1_gradient(runs) == pytest.approx(110.0)


def test_too_few_keeps_v13s_ends_and_unreadable_ladders_do_not_count() -> None:
    games = [_game(f"g{i}", 0.1 * (i + 1)) for i in range(9)] + [_game("blind", 0.5, ladder=0.4)]
    ends = ends_script.read_ends(games)
    assert ends["ply_spread"]["points"] is None  # nine readable, under ten


def test_gradient_falls_back_to_the_game_level_rule() -> None:
    games = [_game(f"g{i}", 10.0 * (i + 1), gradient_runs=2) for i in range(12)]
    ends = ends_script.read_ends(games)
    assert ends["skill_gradient"]["rule"].startswith("game-level")
    assert ends["skill_gradient"]["points"] == {"good_at": 110.0}


def test_leave_one_out_decides_too_few_on_the_full_set() -> None:
    games = [_game(f"g{i}", 0.05 * (i + 1)) for i in range(10)]
    rest = games[1:]
    assert ends_script.read_ends(rest)["ply_spread"]["points"] is None
    loo = ends_script.read_ends(rest, decided_on=games)
    assert loo["ply_spread"]["points"] == {"good_at": pytest.approx(0.5)}


def test_alt_weights_change_only_the_chosen_ends() -> None:
    base = ends_script.base_data()
    games = [_game(f"g{i}", 0.05 * (i + 1)) for i in range(12)]
    ends = ends_script.read_ends(games)
    weights = ends_script.alt_weights(base, ends, only=["ply_spread"])
    assert weights.curves["ply_spread"].points["good_at"] == pytest.approx(0.55)
    v13 = base["components"]
    assert weights.curves["mean_branching"].points == {k: float(v) for k, v in
                                                        v13["mean_branching"]["points"].items()}


def test_verdict_bands() -> None:
    assert ends_script.verdict(0.19, 0.44) == "the ends carry the gap"
    assert ends_script.verdict(0.19, 0.37) == "the ends carry the gap"
    assert ends_script.verdict(0.19, 0.25) == "the ends do not carry it"
    assert ends_script.verdict(0.19, 0.15) == "the ends do not carry it, wider"
    assert ends_script.verdict(0.19, 0.30) == "neither: the ends carry part of it"
    assert math.isnan(ends_script.auc([], [1.0]))
    assert ends_script.auc([1.0, 2.0], [1.0]) == pytest.approx(0.75)
