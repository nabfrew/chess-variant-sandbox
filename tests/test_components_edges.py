"""`metrics/components.py` boundary pins: the edges of the skill-response measures, the
held-type filter and the analysis components.

A second file beside `tests/test_components.py`, which holds the whole-output tests. These
pin one boundary each, on inputs built inline (no `runs/`, no engine, no chess pieces).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from chess_variant_sandbox.metrics import components
from chess_variant_sandbox.metrics.components import (
    _analysis_components,
    _decisive_by_tier,
    _gradient_persistence,
    held_type_indices,
)
from chess_variant_sandbox.tournament.analysis import CorpusAnalysis, GameAnalysis
from chess_variant_sandbox.tournament.ratings import RatingTable, Standing
from chess_variant_sandbox.tournament.record import GameRecord

ROOT = Path(__file__).resolve().parent.parent
TOP = "minimax:nodes=400"


def _record(seed: int, agents: tuple[str, str], winner: int | None) -> GameRecord:
    return GameRecord(
        ruleset_hash="h", ruleset_name="g", seed=seed, agents=agents,
        moves=[seed, seed + 1], winner=winner, reason="annihilation", plies=20, material=[],
    )


def _table(*pairs: tuple[str, float]) -> RatingTable:
    return RatingTable(
        standings=[
            Standing(name=n, rating=r, games=10, wins=5.0, draws=0, as_first=5,
                     first_player_score=0.5)
            for n, r in pairs
        ],
        head_to_head={},
    )


def test_least_squares_slope_centres_the_ratings_before_multiplying() -> None:
    """Three rungs at log2 budgets, ratings near 1000. Centring y and adding the mean are
    equal in exact arithmetic; in floats the centred sum lands on this value."""
    points = [(math.log2(n), y) for n, y in ((100, 1000.0), (200, 1100.0), (400, 1250.0))]
    assert components._least_squares_slope(points) == 124.99999999999994


def test_gradient_persistence_clips_a_falling_top_at_zero() -> None:
    """The bottom half rises 100 a doubling, the top half falls 100: the ratio is clipped
    to 0.0, not read as -1."""
    ratings = _table(("minimax:nodes=100", 0.0), ("minimax:nodes=200", 100.0),
                     ("minimax:nodes=400", 150.0), ("minimax:nodes=800", 50.0))
    assert _gradient_persistence(ratings) == (0.0, {"minimax": 0.0})


def test_decisive_by_tier_on_two_rungs_reads_each_as_its_own_tier() -> None:
    low, high = "minimax:nodes=100", "minimax:nodes=400"
    records = [_record(1, (low, low), 0), _record(2, (high, high), None),
               _record(3, (low, high), 1)]
    assert _decisive_by_tier(records) == (
        {"low": 1.0, "high": 0.0},
        {"kind": "minimax", "low_games": 1, "low_agents": [low], "low_decisive": 1.0,
         "high_games": 1, "high_agents": [high], "high_decisive": 0.0},
    )


def test_decisive_by_tier_on_three_rungs_takes_only_the_top_rung_as_high() -> None:
    """Three rungs split one-and-one: the middle rung is in neither tier, so its game with
    the top rung is not a high-tier game."""
    m100, m400, m1600 = (f"minimax:nodes={n}" for n in (100, 400, 1600))
    records = [_record(1, (m100, m100), None), _record(2, (m400, m1600), 0),
               _record(3, (m1600, m1600), None)]
    assert _decisive_by_tier(records) == (
        {"low": 0.0, "high": 0.0},
        {"kind": "minimax", "low_games": 1, "low_agents": [m100], "low_decisive": 0.0,
         "high_games": 1, "high_agents": [m1600], "high_decisive": 0.0},
    )


def test_held_type_indices_keeps_every_index_of_a_longer_vector_without_portals() -> None:
    """One declared piece and two types, no portal: the extra entry cannot be placed, so
    every index is kept rather than filtered."""
    spec: dict[str, Any] = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    assert len(spec["pieces"]) == 1
    assert held_type_indices(spec, 2) == [0, 1]


def test_analysis_components_one_walked_ply_and_one_legal_move_are_read() -> None:
    games = [GameAnalysis(seed=1, agents=(TOP, TOP), near_best=[1], legal_counts=[1],
                          capture_counts=[1], revisited_plies=0, walked_plies=1)]
    assert _analysis_components(CorpusAnalysis(games=games), {}, {}) == {
        "decision_criticality": 0.0,
        "position_revisit_rate": 0.0,
        "capture_availability": 1.0,
    }
