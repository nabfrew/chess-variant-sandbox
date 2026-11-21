"""`docs/reference/blind-rules.py` (BL3): the arithmetic its registered reading rests on
(`blind-rules.md` §6.1), held on hand-built numbers. No stored run is read and no game is played."""

from __future__ import annotations

import private_paths
import pytest

SCRIPT = "docs/reference/blind-rules.py"
record = private_paths.script(SCRIPT, "blind_rules_reading")
pytestmark = private_paths.needs(SCRIPT)


@pytest.mark.parametrize(
    ("value", "verdict"),
    [(0.30, "dispatch"), (0.41, "dispatch"), (0.2999, "waits"), (0.1501, "waits"),
     (0.15, "struck"), (0.0, "struck")],
)
def test_the_bands_are_inclusive_at_both_edges(value: float, verdict: str) -> None:
    """§4: 0.30 or more dispatches stage 5, 0.15 or less strikes it, between it waits."""
    assert record.grade(value) == verdict


def test_the_registration_is_the_plans() -> None:
    assert (record.BEFORE_END, record.MIN_PLIES, record.PREDICTED) == (8, 20, 0.25)
    assert (record.DISPATCH_AT, record.STRUCK_AT) == (0.30, 0.15)
    assert record.LOSS_REASONS == ("checkmate", "royal_captured", "royal_destroyed")


def _game(winner: int, series: list[float]) -> dict:
    return {"winner": winner, "plies": len(series), "material": series}


def test_the_loser_is_read_from_its_own_side_eight_plies_before_the_end() -> None:
    """The stored series is player 0's balance; the loser's side is its sign or its negation,
    read at `material[plies - 8]`, and level counts as level."""
    series = [0.0] * 12 + [-1.0] + [0.0] * 7  # 20 plies: index 12 is 8 before the end
    assert record.level_from_series(_game(winner=1, series=series)) is False  # player 0 lost, behind
    assert record.level_from_series(_game(winner=0, series=series)) is True  # player 1 lost, ahead
    assert record.level_from_series(_game(winner=1, series=[0.0] * 20)) is True


def test_a_series_shorter_than_the_game_is_not_read() -> None:
    assert record.level_from_series({"winner": 0, "plies": 20, "material": [0.0] * 5}) is None


def test_the_wilson_interval_holds_the_share() -> None:
    low, high = record.wilson(130, 849)
    assert low < 130 / 849 < high
    assert record.wilson(0, 0) is None


def test_the_game_bootstrap_is_fixed_by_its_seed() -> None:
    cells = [(3, 20), (8, 48), (0, 10), (12, 30)]
    first = record.cluster_bootstrap(cells)
    assert first == record.cluster_bootstrap(cells)
    assert first is not None and first[0] <= sum(c[0] for c in cells) / sum(c[1] for c in cells) <= first[1]
    assert record.cluster_bootstrap([(0, 0)]) is None
