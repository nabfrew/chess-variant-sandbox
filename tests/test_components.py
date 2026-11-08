"""Pins `metrics/components.py`'s helpers' whole outputs.

Each test hands one helper (or ``compute_components`` itself) an inline input with
six-decimal values where it can, and compares everything the helper returns with a
literal, so a change to any branch, default or boundary of the helper reads as a failure
here rather than surviving behind a test that checked one field. The records and specs
are built inline; ``rulesets/dull.json`` is read, never written.
"""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_variant_sandbox.engine.compile import compile_spec
from chess_variant_sandbox.metrics import components
from chess_variant_sandbox.metrics.components import (
    _PREFIX,
    _analysis_components,
    _budget_tiers,
    _decisive_by_tier,
    _depth_by_agent,
    _draw_slope,
    _entered,
    _extends,
    _gradient_persistence,
    _ladder_monotonicity,
    _lead_changes,
    _lead_conversion,
    _NeighbourTable,
    _normalised_entropy,
    _opening_diversity,
    _search_advantage,
    _seat_readings,
    _seat_split,
    _set_distance,
    _skill_gradient,
    _split_rung,
    _threat_asymmetry,
    _traversal_time,
    _volatility,
    _worst_half_rates,
    compute_components,
    spy_summary,
)
from chess_variant_sandbox.metrics.seat import (
    SEAT_DECIDED_CHANCE,
    TOP_RUNG_FIRST_SEAT_SCORE,
    TOP_RUNG_SEAT_DETAIL,
)
from chess_variant_sandbox.tournament.analysis import (
    BlunderTrial,
    CorpusAnalysis,
    GameAnalysis,
)
from chess_variant_sandbox.tournament.ratings import RatingTable, Standing
from chess_variant_sandbox.tournament.record import GameRecord

ROOT = Path(__file__).resolve().parent.parent


def _record(
    seed: int,
    agents: tuple[str, str],
    winner: int | None,
    moves: list[int] | None = None,
    plies: int = 20,
    material: list[float] | None = None,
    reason: str = "annihilation",
) -> GameRecord:
    return GameRecord(
        ruleset_hash="h",
        ruleset_name="g",
        seed=seed,
        agents=agents,
        moves=list(moves if moves is not None else [seed, seed + 1, seed + 2]),
        winner=winner,
        reason=reason,
        plies=plies,
        material=list(material or []),
    )


def _standing(name: str, rating: float) -> Standing:
    return Standing(
        name=name, rating=rating, games=10, wins=5.0, draws=0, as_first=5,
        first_player_score=0.5,
    )


def _table(*pairs: tuple[str, float], h2h: dict[tuple[str, str], tuple[float, int]] | None = None
           ) -> RatingTable:
    return RatingTable(standings=[_standing(n, r) for n, r in pairs], head_to_head=h2h or {})


TOP = "minimax:nodes=400"
LOW = "minimax:nodes=100"


def _corpus() -> tuple[list[GameRecord], RatingTable]:
    """Self-play at two rungs, a cross pairing both ways, a random game, a greedy rung."""
    records = [
        _record(1, (TOP, TOP), 0, material=[0.0, 1.0, -0.5, 2.0], plies=31),
        _record(2, (TOP, TOP), 0, material=[1.0, 1.0], plies=9),
        _record(3, (TOP, TOP), None, reason="ply_limit", plies=120, material=[0.0, -1.0, 1.0]),
        _record(4, (LOW, TOP), 1, plies=17, material=[0.5, 0.25]),
        _record(5, (TOP, LOW), 0, plies=11, material=[-1.0, 0.0, 3.0]),
        _record(6, ("random", TOP), 1, plies=7),
        _record(7, ("greedy", "greedy"), None, reason="no_progress", plies=60,
                material=[0.0, 0.0]),
        _record(8, ("greedy", LOW), 1, plies=8, material=[0.0, -2.0]),
    ]
    ratings = _table(
        (TOP, 212.345678), (LOW, 101.234567), ("greedy", 20.5), ("random", -50.125),
        h2h={(TOP, "random"): (0.833333, 6), ("random", TOP): (0.1, 4)},
    )
    return records, ratings


def test_compute_components_whole_output_on_a_mixed_corpus() -> None:
    """The balance reads self-play only: four games (three at the top rung, one greedy
    draw) score the first seat 0.75, so its advantage is 0.5; the cross and random games
    would move both."""
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    records, ratings = _corpus()
    result = compute_components(spec, records, ratings)
    assert result.raw == {
        "decisive_rate": 0.7142857142857143,
        "decisive_rate_worst_half": 0.5,
        "distinct_game_rate": 1.0,
        "draw_slope": 1.0,
        "first_player_advantage": 0.5,
        "first_player_score": 0.75,
        "gradient_persistence": 1.0,
        "ladder_monotonicity": 1.0,
        "lead_changes": 16.666666666666668,
        "material_volatility": 1.0595238095238095,
        "mean_branching": 0.0,
        "mean_plies": 36.57142857142857,
        "mean_plies_worst_half": 34.0,
        "opening_diversity": 1.0,
        "piece_types": 1.0,
        "ply_spread": 1.0414094734571269,
        "prefix_divergence": 1.0,
        "quick_kill_rate": 0.42857142857142855,
        "quick_kill_rate_worst_half": 0.5,
        "rating_spread": 262.470678,
        "rule_mass": 11.75,
        "rule_yield": 22.337930042553193,
        "search_advantage": 0.8666665,
        "seat_decided_chance": 0.625,
        "skill_gradient": 55.56,
        "special_rules": 1.0,
        "stall_rate": 0.2857142857142857,
        "stall_rate_worst_half": 0.5,
        "threat_asymmetry": 0.0,
        "top_rung_first_seat_score": 0.7,
        "traversal_time": 4.0,
    }
    diagnostics = dict(result.diagnostics)
    assert diagnostics.pop("complexity")["piece_types"] == 1
    assert diagnostics == {
        "decisive_by_tier": {
            "high_agents": [TOP], "high_decisive": 0.6667, "high_games": 3,
            "kind": "minimax", "low_agents": [LOW], "low_games": 0,
        },
        "depth_by_agent": {},
        "games": 8,
        "graded_games": 7,
        "gradient_persistence_fits": {},
        "ladder_split": {
            "games": {"strong": 5, "weak": 2},
            "rates": {
                "strong": {"decisive_rate": 0.8, "mean_plies": 37.6,
                           "quick_kill_rate": 0.4, "stall_rate": 0.2},
                "weak": {"decisive_rate": 0.5, "mean_plies": 34.0,
                         "quick_kill_rate": 0.5, "stall_rate": 0.5},
            },
            "rung_depth": {},
            "rungs": [0, 100, 400],
            "split_rung": 100,
            "weaker_seat_depth": {"strong": None, "weak": None},
            "worst_half": {"decisive_rate": "weak", "mean_plies": "weak",
                           "quick_kill_rate": "weak", "stall_rate": "weak"},
        },
        "lead_conversion_sample": {"decisive_games": 5, "qualifying_games": 2},
        "outcome_reasons": {"annihilation": 5, "no_progress": 1, "ply_limit": 1},
        "self_play_games": 4,
        "skill_gradient_fits": {"minimax": 55.56},
        "top_rung_seat": {
            "chance": 0.625, "distinct": 4, "distinct_losses": 1, "distinct_wins": 3,
            "draws": 1,
            "first_seat_transcripts": ["af6405d0ef20c951", "c1aaa993ca216753",
                                       "e2353910701f1c2a"],
            "games": 5, "losses": 1, "rungs": [100, 400],
            "transcripts": ["af6405d0ef20c951", "c1aaa993ca216753", "e2353910701f1c2a",
                            "fd3b4f6a25df4569"],
            "wins": 3,
        },
    }


def test_split_rung_on_two_rungs_is_the_lower_and_on_one_is_none() -> None:
    two = [_record(1, ("greedy", TOP), 0), _record(2, (TOP, TOP), 1)]
    assert _split_rung(two) == 0
    assert _split_rung([_record(1, (TOP, TOP), 0)]) is None


def test_seat_readings_on_one_rung_read_nothing_from_no_games() -> None:
    records = [_record(1, (TOP, TOP), 0), _record(2, (TOP, TOP), 1)]
    assert _seat_readings(records) == ({}, {TOP_RUNG_SEAT_DETAIL: {"rungs": [400], "games": 0}})


def test_seat_split_on_a_band_no_game_sits_in_scores_the_seat_level() -> None:
    records = [_record(1, (LOW, TOP), 0), _record(2, (TOP, LOW), 1)]
    raw, detail = _seat_split(records, [1600, 6400], SEAT_DECIDED_CHANCE,
                              TOP_RUNG_FIRST_SEAT_SCORE)
    assert raw == {SEAT_DECIDED_CHANCE: 1.0, TOP_RUNG_FIRST_SEAT_SCORE: 0.5}
    assert detail == {
        "rungs": [1600, 6400], "games": 0, "wins": 0, "draws": 0, "losses": 0,
        "chance": 1.0, "distinct": 0, "distinct_wins": 0, "distinct_losses": 0,
        "transcripts": [], "first_seat_transcripts": [],
    }


def test_opening_diversity_edges() -> None:
    """One game, and three games with one opening, read 0.0 (a positive zero: the
    value is written to JSON); five different openings read exactly 1.0, where the
    unclipped ratio is 1.0000000000000002."""
    assert _opening_diversity([_record(1, (TOP, TOP), 0)]) == 0.0
    same = [_record(seed, (TOP, TOP), 0, moves=[7, 8, 9]) for seed in (1, 2, 3)]
    value = _opening_diversity(same)
    assert value == 0.0 and math.copysign(1.0, value) == 1.0
    assert json.dumps(value) == "0.0"
    distinct = [_record(seed, (TOP, TOP), 0, moves=[seed * 10]) for seed in range(5)]
    assert _opening_diversity(distinct) == 1.0


def test_set_distance() -> None:
    assert _set_distance(frozenset(), frozenset()) == 0.0
    assert _set_distance(frozenset({1, 2}), frozenset({2, 3})) == 0.6666666666666667
    assert _set_distance(frozenset({1}), frozenset({2})) == 1.0


def test_extends_at_its_size_bound() -> None:
    table = _NeighbourTable.build(_PREFIX, [(1,), (2,), (3,), (4,)], None)
    # Every content shared, and the table exactly twice this call's size: grown from.
    assert _extends(table, [(1,), (2,)]) is True
    # Four entries for a call of one: more than twice its size.
    assert _extends(table, [(1,)]) is False
    # Too little shared.
    assert _extends(table, [(1,), (5,), (6,)]) is False


def test_entered_reuses_grows_and_replaces_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(components, "_TABLES", {})
    first, slots = _entered(_PREFIX, [(1,), (2,), (3,), (4,)])
    assert slots == [0, 1, 2, 3]
    assert components._TABLES["prefix_divergence"] == [first]
    # Mostly shared: grown from the first table, which it replaces.
    grown, slots = _entered(_PREFIX, [(1,), (2,), (3,), (5,)])
    assert grown is not first and slots == [0, 1, 2, 4]
    assert grown.contents == [(1,), (2,), (3,), (4,), (5,)]
    assert components._TABLES["prefix_divergence"] == [grown]
    # Held already: the same table, not a new one.
    again, slots = _entered(_PREFIX, [(5,), (1,)])
    assert again is grown and slots == [4, 0]
    assert components._TABLES["prefix_divergence"] == [grown]


def test_entered_survives_its_base_evicted_while_it_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another thread may drop the base table between the two locked sections."""
    monkeypatch.setattr(components, "_TABLES", {})
    _entered(_PREFIX, [(1,), (2,), (3,), (4,)])
    build = _NeighbourTable.build

    def evicting(measure: Any, contents: Any, base: Any) -> Any:
        components._TABLES[measure.name].clear()
        return build(measure, contents, base)

    monkeypatch.setattr(_NeighbourTable, "build", evicting)
    grown, slots = _entered(_PREFIX, [(1,), (2,), (3,), (5,)])
    assert slots == [0, 1, 2, 4]
    assert components._TABLES["prefix_divergence"] == [grown]


def test_lead_changes_counts_crossings_per_hundred_plies() -> None:
    record = _record(1, (TOP, TOP), 0, material=[-0.5, 1.0, 0.0, 2.0, -1.0, 0.25])
    assert _lead_changes(record) == 50.0


def test_volatility() -> None:
    assert _volatility(_record(1, (TOP, TOP), 0, material=[1.5])) == 0.0
    assert _volatility(_record(1, (TOP, TOP), 0, material=[0.0, 3.0])) == 3.0
    assert _volatility(_record(1, (TOP, TOP), 0, material=[0.0, 1.25, -0.5])) == 1.5


def test_skill_gradient_reads_a_one_node_rung() -> None:
    ratings = _table(("minimax:nodes=0", 500.0), ("minimax:nodes=1", 0.0),
                     ("minimax:nodes=4", 100.0), ("random", -20.0))
    assert _skill_gradient(ratings) == (50.0, {"minimax": 50.0})


def test_search_advantage_on_two_standings() -> None:
    ratings = _table((TOP, 100.0), ("random", 0.0),
                     h2h={(TOP, "random"): (0.75, 4), ("random", TOP): (0.125, 4)})
    assert _search_advantage(ratings) == 0.8125


def test_budget_tiers_skips_a_heuristic_and_a_budgetless_name_and_reads_on() -> None:
    names = ["minimax:h=abc,nodes=100", "greedy", "minimax:nodes=1600", "minimax:nodes=400",
             "mcts:nodes=200", "minimax:nodes=0"]
    assert _budget_tiers(names) == {
        "minimax": [(400, "minimax:nodes=400"), (1600, "minimax:nodes=1600")],
        "mcts": [(200, "mcts:nodes=200")],
    }


def test_gradient_persistence_whole_output() -> None:
    ratings = _table(
        ("minimax:nodes=100", 0.0), ("minimax:nodes=200", 100.0),
        ("minimax:nodes=400", 250.0), ("minimax:nodes=800", 423.4),
        ("mcts:nodes=100", 0.0), ("mcts:nodes=200", 50.0), ("mcts:nodes=400", 75.0),
    )
    assert _gradient_persistence(ratings) == (1.117, {"minimax": 1.734, "mcts": 0.5})
    steep = _table(
        ("minimax:nodes=100", 0.0), ("minimax:nodes=200", 100.0),
        ("minimax:nodes=400", 280.0), ("minimax:nodes=800", 460.0),
    )
    assert _gradient_persistence(steep) == (1.8, {"minimax": 1.8})


def test_ladder_monotonicity_skips_a_one_rung_kind_and_reads_each_step() -> None:
    ratings = _table(("alpha:nodes=100", 0.0), ("minimax:nodes=100", 100.0),
                     ("minimax:nodes=400", 200.0), ("minimax:nodes=1600", 150.0))
    assert _ladder_monotonicity(ratings) == 0.5


def test_ladder_monotonicity_with_no_step_reads_a_clean_ladder() -> None:
    assert _ladder_monotonicity(_table(("minimax:nodes=100", 0.0), ("random", 5.0))) == 1.0


def test_decisive_by_tier_reads_same_tier_pairings_only() -> None:
    m100, m400, m1600, m6400 = (f"minimax:nodes={n}" for n in (100, 400, 1600, 6400))
    records = [
        _record(1, (m100, m100), 0),
        _record(2, (m100, m400), None),
        _record(3, (m400, m100), 1),
        _record(4, ("random", m100), 0),
        _record(5, (m1600, m6400), None),
        _record(6, (m6400, m6400), 1),
        _record(7, (m6400, "random"), 0),
    ]
    assert _decisive_by_tier(records) == (
        {"low": 0.6666666666666666, "high": 0.5},
        {
            "kind": "minimax",
            "low_games": 3, "low_agents": [m100, m400], "low_decisive": 0.6667,
            "high_games": 2, "high_agents": [m1600, m6400], "high_decisive": 0.5,
        },
    )


def test_draw_slope() -> None:
    assert _draw_slope({}) == 1.0
    assert _draw_slope({"low": 0.0, "high": 0.5}) == 1.0
    assert _draw_slope({"low": 0.5, "high": 0.6}) == 1.2
    assert _draw_slope({"low": 0.25, "high": 0.6}) == 2.0


def test_normalised_entropy() -> None:
    # Five equal counts: the unclipped ratio is 1.0000000000000002.
    assert _normalised_entropy([3] * 5) == 1.0
    assert _normalised_entropy([0, 0]) == 0.0
    assert _normalised_entropy([4]) == 0.0
    assert _normalised_entropy([1, 3]) == 0.8112781244591328


def test_threat_asymmetry_reads_mirrored_geometry() -> None:
    """A strikes on (1, 1); B strikes on (1, -1) and (-1, 1). B can answer A (A's
    displacement mirrored in y is in B's set), A cannot answer B's (-1, 1): one of the
    two ordered pairs is a one-way threat."""
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    spec["pieces"] = [
        {"id": "A", "name": "a", "value": 1, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            {"kind": "leap", "vectors": [[1, 1]], "mode": "capture"}]},
        {"id": "B", "name": "b", "value": 1, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            {"kind": "leap", "vectors": [[1, -1], [-1, 1]], "mode": "capture"}]},
    ]
    spec["setup"]["placements"] = [{"piece": "A", "squares": ["a1", "b1"]},
                                   {"piece": "B", "squares": ["d1", "e1"]}]
    assert _threat_asymmetry(compile_spec(spec)) == 0.5


def test_traversal_time_with_no_piece_that_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(components, "_displacements", lambda compiled, mask: [set(), set()])
    assert _traversal_time(SimpleNamespace(width=5, height=7)) == 0.0  # type: ignore[arg-type]


def test_spy_summary_whole_output() -> None:
    spies = [
        {"lost_ply_0": 12, "lost_ply_1": None, "lost_how_0": "captured", "lost_how_1": None,
         "reply_turns": 1, "alternating_plies": 9, "announced_rounds": 3, "blind_rounds": 5},
        {"lost_ply_0": 20, "lost_ply_1": 20, "lost_how_0": "exposed",
         "lost_how_1": "captured", "reply_turns": 0, "alternating_plies": 14,
         "announced_rounds": 2, "blind_rounds": 4},
    ]
    assert spy_summary(spies) == {
        "games": 2,
        "spy_lost_share": 1.0,
        "both_lost_share": 0.5,
        "both_at_once": 1,
        "first_loss_median_ply": 16.0,
        "reply_turn_share": 0.5,
        "mean_alternating_plies": 11.5,
        "mean_announced_rounds": 2.5,
        "mean_blind_rounds": 4.5,
        "lost_how": {"captured": 2, "exposed": 1},
    }


def _analysis() -> CorpusAnalysis:
    games = [
        GameAnalysis(seed=1, agents=(TOP, TOP), near_best=[2, 1, 3], legal_counts=[8, 0, 6],
                     moves_by_type=[5, 0, 3], capture_counts=[1, 0, 3], revisited_plies=2,
                     walked_plies=7, starting_instances=6, moved_instances=4,
                     rounds={"rounds": 4, "collisions": 1, "void_moves": 0, "swaps": 2}),
        GameAnalysis(seed=2, agents=(TOP, TOP), near_best=[1], legal_counts=[4],
                     moves_by_type=[1, 0, 191], capture_counts=[2], revisited_plies=1,
                     walked_plies=9, starting_instances=6, moved_instances=5,
                     rounds={"rounds": 6, "collisions": 3}),
    ]
    blunders = [
        BlunderTrial(seed=1, agents=(TOP, TOP), ply=3, original_winner=0,
                     blundering_side=0, new_winner=1),
        BlunderTrial(seed=2, agents=(TOP, TOP), ply=5, original_winner=0,
                     blundering_side=1, new_winner=None),
        BlunderTrial(seed=3, agents=(TOP, TOP), ply=4, original_winner=None,
                     blundering_side=1, new_winner=None),
    ]
    return CorpusAnalysis(games=games, blunders=blunders)


def test_analysis_components_whole_output() -> None:
    """The second game's rounds lack two counts: the tallies read the first game's keys."""
    diagnostics: dict[str, Any] = {}
    raw = _analysis_components(_analysis(), diagnostics, {})
    assert raw == RAW_ANALYSIS
    assert diagnostics == DIAGNOSTICS_ANALYSIS


def test_analysis_components_with_no_piece_moved_reads_no_utilisation() -> None:
    games = [GameAnalysis(seed=1, agents=(TOP, TOP), moves_by_type=[0, 0])]
    diagnostics: dict[str, Any] = {}
    assert _analysis_components(CorpusAnalysis(games=games), diagnostics, {}) == {}
    assert diagnostics == {}
    assert _analysis_components(CorpusAnalysis(), diagnostics, {}) == {}
    assert diagnostics == {}


RAW_ANALYSIS: dict[str, float] = {
    "blunder_recoverability": 0.6666666666666666,
    "capture_availability": 0.375,
    "decision_criticality": 0.6666666666666667,
    "instance_participation": 0.75,
    "piece_utilisation": 0.12264760695797414,
    "position_revisit_rate": 0.1875,
}
DIAGNOSTICS_ANALYSIS: dict[str, Any] = {
    "blunder_changed_result": 0.6667,
    "blunder_trials": 3,
    "criticality_samples": 3,
    "held_piece_types": 3,
    "idle_piece_types": 1,
    "moves_by_type": [6, 0, 194],
    "rounds": {
        "collisions": 4, "collisions_per_round": 0.4, "rounds": 10, "swaps": 2,
        "swaps_per_round": 0.2, "void_moves": 0, "void_moves_per_round": 0.0,
    },
}


def _searched(record: GameRecord, depth_sum: tuple[int, int], searches: tuple[int, int]
              ) -> GameRecord:
    return replace(record, depth_sum=depth_sum, searches=searches)


def _ladder_corpus() -> list[GameRecord]:
    """Three rungs (greedy, 100, 400): three games in each half, thirds in every rate."""
    return [
        _searched(_record(1, (TOP, TOP), 0, plies=11), (20, 24), (7, 7)),
        _searched(_record(2, (LOW, TOP), None, plies=120, reason="ply_limit"), (5, 9), (3, 3)),
        _searched(_record(3, (LOW, LOW), None, plies=40, reason="no_progress"), (4, 6), (3, 3)),
        _searched(_record(4, ("greedy", LOW), 0, plies=9), (0, 4), (0, 3)),
        _record(5, ("greedy", "greedy"), None, plies=60, reason="no_progress"),
        _searched(_record(6, (TOP, "greedy"), 1, plies=11), (14, 0), (6, 0)),
    ]


def test_worst_half_rates_whole_output_on_two_halves_failing_different_gates() -> None:
    """The strong half stalls more and decides less; the weak half dies quicker and
    shorter. Each rate reports its own worse half, and the detail's thirds keep four
    places, the depths three."""
    raw, detail = _worst_half_rates(_ladder_corpus())
    assert raw == {
        "quick_kill_rate_worst_half": 0.6666666666666666,
        "stall_rate_worst_half": 0.6666666666666666,
        "decisive_rate_worst_half": 0.3333333333333333,
        "mean_plies_worst_half": 26.666666666666668,
    }
    assert detail == {
        "split_rung": 100,
        "rungs": [0, 100, 400],
        "games": {"weak": 3, "strong": 3},
        "rates": {
            "weak": {"stall_rate": 0.3333, "decisive_rate": 0.6667, "mean_plies": 26.6667,
                     "quick_kill_rate": 0.6667},
            "strong": {"stall_rate": 0.6667, "decisive_rate": 0.3333, "mean_plies": 57.0,
                       "quick_kill_rate": 0.3333},
        },
        "worst_half": {"quick_kill_rate": "weak", "stall_rate": "strong",
                       "decisive_rate": "strong", "mean_plies": "weak"},
        "rung_depth": {"100": 1.583, "400": 2.913},
        "weaker_seat_depth": {"weak": None, "strong": 2.565},
    }


def test_depth_by_agent_pools_each_agents_searches_and_omits_the_non_searchers() -> None:
    assert _depth_by_agent(_ladder_corpus()) == {
        LOW: {"mean_depth": 1.583, "searches": 12},
        TOP: {"mean_depth": 2.913, "searches": 23},
    }


def test_lead_conversion_reads_past_a_draw_and_a_small_edge() -> None:
    """A draw first, then a game whose edge at a third is under the epsilon and one with no
    material series, then nine with a real edge: seven of them won by the side ahead."""
    records = [
        _record(1, (TOP, TOP), None, material=[2.0, 2.0, 2.0]),
        _record(2, (TOP, TOP), 0, material=[0.0, 0.5, 3.0]),
        _record(3, (TOP, TOP), 0, material=[]),
    ]
    edges = [(1.5, 0), (-2.0, 1), (1.0, 0), (-1.25, 1), (3.0, 0), (-1.0, 1), (2.5, 0),
             (1.5, 1), (-2.0, 0)]
    for seed, (edge, winner) in enumerate(edges, start=10):
        records.append(_record(seed, (TOP, TOP), winner, material=[0.0, edge, 0.0]))
    assert _lead_conversion(records) == (
        0.7777777777777778, {"qualifying_games": 9, "decisive_games": 11},
    )


def test_ladder_monotonicity_reads_a_two_rung_kind_and_a_tie_as_no_rise() -> None:
    ratings = _table(("minimax:nodes=100", 50.0), ("minimax:nodes=400", 50.0),
                     ("mcts:nodes=100", 10.0), ("mcts:nodes=400", 30.5))
    assert _ladder_monotonicity(ratings) == 0.5


def test_threat_asymmetry_with_one_piece_type_has_no_pair_to_read() -> None:
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    spec["pieces"] = [
        {"id": "A", "name": "a", "value": 1, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            {"kind": "leap", "vectors": [[1, 1]], "mode": "capture"}]},
    ]
    spec["setup"]["placements"] = [{"piece": "A", "squares": ["a1", "b1"]}]
    assert _threat_asymmetry(compile_spec(spec)) == 0.0


def test_spy_summary_whole_output_on_three_games_in_thirds() -> None:
    spies = [
        {"lost_ply_0": 12, "lost_ply_1": None, "lost_how_0": "captured", "lost_how_1": None,
         "reply_turns": 1, "alternating_plies": 9, "announced_rounds": 3, "blind_rounds": 5},
        {"lost_ply_0": 20, "lost_ply_1": 20, "lost_how_0": "exposed",
         "lost_how_1": "captured", "reply_turns": 0, "alternating_plies": 14,
         "announced_rounds": 2, "blind_rounds": 4},
        {"lost_ply_0": None, "lost_ply_1": None, "lost_how_0": None, "lost_how_1": None,
         "reply_turns": 1, "alternating_plies": 10, "announced_rounds": 1, "blind_rounds": 7},
    ]
    assert spy_summary(spies) == {
        "games": 3,
        "spy_lost_share": 0.6667,
        "both_lost_share": 0.3333,
        "both_at_once": 1,
        "first_loss_median_ply": 16.0,
        "reply_turn_share": 0.6667,
        "mean_alternating_plies": 11.0,
        "mean_announced_rounds": 2.0,
        "mean_blind_rounds": 5.33,
        "lost_how": {"captured": 2, "exposed": 1},
    }


def test_analysis_components_round_shares_in_thirds() -> None:
    games = [GameAnalysis(seed=1, agents=(TOP, TOP),
                          rounds={"rounds": 3, "collisions": 1, "void_moves": 2, "swaps": 0})]
    diagnostics: dict[str, Any] = {}
    assert _analysis_components(CorpusAnalysis(games=games), diagnostics, {}) == {}
    assert diagnostics == {"rounds": {
        "rounds": 3, "collisions": 1, "void_moves": 2, "swaps": 0,
        "collisions_per_round": 0.3333, "void_moves_per_round": 0.6667,
        "swaps_per_round": 0.0,
    }}


def test_spec_facts_are_frozen() -> None:
    """Memoised and handed to every caller of one spec: a caller must not be able to
    write into the shared copy."""
    facts = components.SpecFacts(complexity=None, threat_asymmetry=0.25,  # type: ignore[arg-type]
                                 traversal_time=4.0)
    with pytest.raises(FrozenInstanceError):
        facts.traversal_time = 1.0  # type: ignore[misc]


def test_outcome_rates_on_no_games_read_zero() -> None:
    assert components._outcome_rates([]) == {
        "stall_rate": 0.0, "decisive_rate": 0.0, "mean_plies": 0.0, "quick_kill_rate": 0.0,
    }


def test_neighbour_table_keeps_its_bases_distances() -> None:
    """A distance the base holds is copied, not recomputed: a doctored one survives."""
    base = _NeighbourTable.build(_PREFIX, [(1,), (2,)], None)
    base.dist[0][1] = base.dist[1][0] = 0.123456
    grown = _NeighbourTable.build(_PREFIX, [(1,), (2,), (3,)], base)
    assert grown.dist == [[0.0, 0.123456, 1.0], [0.123456, 0.0, 1.0], [1.0, 1.0, 0.0]]


def test_traversal_time_divides_the_diameter_by_a_reach_of_one_and_a_half() -> None:
    """One piece stepping one or two squares forward on a five-square board: a mean
    reach of 1.5 across a diameter of 4."""
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    spec["pieces"] = [
        {"id": "A", "name": "a", "value": 1, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            {"kind": "leap", "vectors": [[0, 2]], "mode": "move"}]},
    ]
    spec["setup"]["placements"] = [{"piece": "A", "squares": ["a1", "b1"]}]
    assert _traversal_time(compile_spec(spec)) == 2.6666666666666665


def test_least_squares_slope_on_a_unit_and_a_zero_spread() -> None:
    """Four points, two each at x = 0 and x = 1: the x spread sums to exactly 1."""
    points = [(0.0, 0.0), (0.0, 0.0), (1.0, 2.5), (1.0, 2.5)]
    assert components._least_squares_slope(points) == 2.5
    assert components._least_squares_slope([(3.0, 1.0), (3.0, 4.0)]) is None


def test_analysis_components_on_one_move_and_on_one_held_type() -> None:
    """A single move is enough to read utilisation; one held type reads full marks."""
    one = [GameAnalysis(seed=1, agents=(TOP, TOP), moves_by_type=[1, 0])]
    diagnostics: dict[str, Any] = {}
    assert _analysis_components(CorpusAnalysis(games=one), diagnostics, {}) == {
        "piece_utilisation": 0.0,
    }
    assert diagnostics == {"held_piece_types": 2, "idle_piece_types": 1, "moves_by_type": [1, 0]}
    single = [GameAnalysis(seed=1, agents=(TOP, TOP), moves_by_type=[3])]
    diagnostics = {}
    assert _analysis_components(CorpusAnalysis(games=single), diagnostics, {}) == {
        "piece_utilisation": 1.0,
    }
    assert diagnostics == {"held_piece_types": 1, "idle_piece_types": 0, "moves_by_type": [3]}


def test_spy_summary_phase_means_in_thirds_keep_two_places() -> None:
    spies = [
        {"lost_ply_0": None, "lost_ply_1": None, "lost_how_0": None, "lost_how_1": None,
         "reply_turns": 0, "alternating_plies": alternating, "announced_rounds": announced,
         "blind_rounds": 0}
        for alternating, announced in ((9, 3), (14, 2), (11, 2))
    ]
    summary = spy_summary(spies)
    assert (summary["mean_alternating_plies"], summary["mean_announced_rounds"]) == (11.33, 2.33)


def test_rule_yield_divides_by_a_rule_mass_under_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rule mass of 0.75 still divides: only a mass of zero reads no yield."""
    complexity = SimpleNamespace(rule_mass=0.75, piece_types=1, special_rules=0,
                                 as_dict=lambda: {})
    monkeypatch.setattr(components, "spec_facts", lambda spec: components.SpecFacts(
        complexity=complexity, threat_asymmetry=0.0, traversal_time=0.0))  # type: ignore[arg-type]
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    records = [_record(1, (TOP, TOP), 0), _record(2, (TOP, TOP), 1)]
    result = compute_components(spec, records, _table((TOP, 100.0), (LOW, 1.0)))
    assert (result.raw["rating_spread"], result.raw["rule_yield"]) == (99.0, 132.0)


def test_is_weak_reads_the_kind_before_the_options() -> None:
    assert components._is_weak("random") is True
    assert components._is_weak("random:seed=3") is True
    assert components._is_weak("minimax:random") is False


def test_spread_at_a_mean_of_one_and_of_minus_one() -> None:
    assert components._spread([0.5, 1.5]) == 0.5
    assert components._spread([-0.5, -1.5]) == -0.5
    assert components._spread([0.0, 0.0]) == 0.0


def test_outcome_rates_a_decisive_game_at_the_quick_kill_bound_is_not_quick() -> None:
    records = [_record(1, (TOP, TOP), 0, plies=12), _record(2, (TOP, TOP), 1, plies=11),
               _record(3, (TOP, TOP), None, plies=5, reason="ply_limit")]
    assert components._outcome_rates(records) == {
        "stall_rate": 0.3333333333333333, "decisive_rate": 0.6666666666666666,
        "mean_plies": 9.333333333333334, "quick_kill_rate": 0.3333333333333333,
    }


def test_seat_readings_read_the_top_three_over_a_one_node_rung() -> None:
    """Rungs 1, 100 and 400: the bottom rung searches (one node), so the wide band is read."""
    one = "minimax:nodes=1"
    records = [_record(1, (TOP, TOP), 0), _record(2, (LOW, LOW), 0), _record(3, (one, one), 1),
               _record(4, (TOP, LOW), 0), _record(5, (LOW, one), None, reason="ply_limit")]
    raw, details = _seat_readings(records)
    assert raw == {
        "seat_decided_chance": 0.25, "top_rung_first_seat_score": 1.0,
        "top_three_seat_decided_chance": 0.625, "top_three_first_seat_score": 0.7,
    }
    assert details["top_three_seat"]["rungs"] == [1, 100, 400]


def test_measure_is_frozen() -> None:
    """The three measures are module-level and shared by every call: none may be rewritten."""
    measure = components._Measure("m", components._as_moves, _set_distance)
    with pytest.raises(FrozenInstanceError):
        measure.name = "n"  # type: ignore[misc]


def test_entered_moves_a_table_it_finds_to_the_front(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three disjoint tables; the oldest, found again, becomes the newest."""
    monkeypatch.setattr(components, "_TABLES", {})
    oldest, _ = _entered(_PREFIX, [(1,), (2,)])
    middle, _ = _entered(_PREFIX, [(11,), (12,)])
    newest, _ = _entered(_PREFIX, [(21,), (22,)])
    assert components._TABLES["prefix_divergence"] == [newest, middle, oldest]
    found, slots = _entered(_PREFIX, [(2,), (1,)])
    assert found is oldest and slots == [1, 0]
    assert components._TABLES["prefix_divergence"] == [oldest, newest, middle]


def test_extends_with_half_its_contents_shared() -> None:
    table = _NeighbourTable.build(_PREFIX, [(1,), (2,)], None)
    assert _extends(table, [(1,), (5,)]) is True


def test_irreversibility_is_the_share_of_irreversible_plies() -> None:
    record = replace(_record(1, (TOP, TOP), 0), captures=[1, 0, 0])
    assert components._irreversibility(record) == 0.3333333333333333
    assert components._irreversibility(_record(1, (TOP, TOP), 0)) == 0.0


def test_lead_conversion_reads_a_sample_of_exactly_the_minimum() -> None:
    edges = [(1.5, 0), (-2.0, 1), (1.25, 0), (-1.25, 1), (3.0, 0), (-1.5, 1), (2.5, 1),
             (1.5, 1)]
    records = [_record(seed, (TOP, TOP), winner, material=[0.0, edge, 0.0])
               for seed, (edge, winner) in enumerate(edges, start=1)]
    assert _lead_conversion(records) == (0.75, {"qualifying_games": 8, "decisive_games": 8})


def test_skill_gradient_with_no_fit_reads_zero_and_averages_two_kinds() -> None:
    assert _skill_gradient(_table(("random", 0.0), ("greedy", 10.0))) == (0.0, {})
    ratings = _table(("minimax:nodes=100", 0.0), ("minimax:nodes=400", 300.0),
                     ("mcts:nodes=100", 0.0), ("mcts:nodes=200", 50.0))
    assert _skill_gradient(ratings) == (100.0, {"minimax": 150.0, "mcts": 50.0})


def test_gradient_persistence_skips_a_kind_whose_upper_fit_is_flat() -> None:
    """Two names at the top budget: the upper half's points share one x and fit nothing."""
    ratings = _table(("minimax:nodes=100", 0.0), ("minimax:nodes=400", 100.0),
                     ("minimax:d=1,nodes=400", 200.0))
    assert _gradient_persistence(ratings) == (1.0, {})


def test_traversal_time_reads_on_past_a_piece_type_that_cannot_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(components, "_displacements",
                        lambda compiled, mask: [set(), {(0, 1), (0, 2)}])
    compiled = SimpleNamespace(width=5, height=7, n_squares=0, initial_board=[],
                               encode_piece=lambda player, piece_type: 0)
    assert _traversal_time(compiled) == 4.0  # type: ignore[arg-type]


def test_traversal_time_reads_a_ray_past_its_unusable_first_square() -> None:
    """A step of one, and a slide of two or three whose first square is below its minimum:
    reaches of 1, 2 and 3 across a diameter of 4."""
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    spec["pieces"] = [{"id": "A", "name": "a", "value": 1, "moves": [
        {"kind": "leap", "vectors": [[0, 1]], "mode": "move"},
        {"kind": "slide", "vectors": [[0, 1]], "range": 3, "min_range": 2, "mode": "move"}]}]
    spec["setup"]["placements"] = [{"piece": "A", "squares": ["a1", "b1"]}]
    assert _traversal_time(compile_spec(spec)) == 2.0


def test_analysis_components_one_starting_instance_is_read() -> None:
    games = [GameAnalysis(seed=1, agents=(TOP, TOP), starting_instances=1, moved_instances=1)]
    assert _analysis_components(CorpusAnalysis(games=games), {}, {}) == {
        "instance_participation": 1.0,
    }


def test_lead_conversion_one_short_of_the_minimum_sample_reads_nothing() -> None:
    edges = [(1.5, 0), (-2.0, 1), (1.25, 0), (-1.25, 1), (3.0, 0), (-1.5, 1), (2.5, 1)]
    records = [_record(seed, (TOP, TOP), winner, material=[0.0, edge, 0.0])
               for seed, (edge, winner) in enumerate(edges, start=1)]
    assert _lead_conversion(records) == (None, {"qualifying_games": 7, "decisive_games": 7})


def test_rule_yield_on_a_rule_mass_of_zero_reads_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    complexity = SimpleNamespace(rule_mass=0.0, piece_types=1, special_rules=0,
                                 as_dict=lambda: {})
    monkeypatch.setattr(components, "spec_facts", lambda spec: components.SpecFacts(
        complexity=complexity, threat_asymmetry=0.0, traversal_time=0.0))  # type: ignore[arg-type]
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    records = [_record(1, (TOP, TOP), 0), _record(2, (TOP, TOP), 1)]
    result = compute_components(spec, records, _table((TOP, 100.0), (LOW, 1.0)))
    assert (result.raw["rating_spread"], result.raw["rule_yield"]) == (99.0, 0.0)


def test_prefix_distance_from_an_empty_game_is_zero() -> None:
    assert components._prefix_distance((), (1, 2)) == 0.0
    assert components._prefix_distance((3,), ()) == 0.0


def test_extends_refuses_a_table_three_times_the_calls_size() -> None:
    table = _NeighbourTable.build(_PREFIX, [(1,), (2,), (3,)], None)
    assert _extends(table, [(1,)]) is False


def test_lead_changes_on_a_series_of_exactly_two() -> None:
    """Two non-zero entries that cross: one change in two plies. A zero then a lead: no
    series to read. A one-point lead counts like any other non-zero value."""
    assert _lead_changes(_record(1, (TOP, TOP), 0, material=[1.0, -1.0])) == 50.0
    assert _lead_changes(_record(1, (TOP, TOP), 0, material=[0.0, 2.0])) == 0.0


def test_traversal_time_weights_each_type_by_the_first_seats_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seat 0 holds three of type 0 (reach 1) and one of type 1 (reach 3); seat 1's type-1
    piece is not counted. Reaches 1, 1, 1, 3 average 1.5 across a diameter of 6."""
    monkeypatch.setattr(components, "_displacements",
                        lambda compiled, mask: [{(0, 1)}, {(0, 3)}])
    compiled = SimpleNamespace(
        width=5, height=7, n_squares=5, initial_board=[1, 1, 1, 2, 12],
        encode_piece=lambda player, piece_type: 10 * player + piece_type + 1,
    )
    assert _traversal_time(compiled) == 4.0  # type: ignore[arg-type]


def test_skill_gradient_reads_on_past_a_budgetless_name() -> None:
    ratings = _table(("random", 5.0), ("minimax:nodes=100", 0.0), ("minimax:nodes=400", 300.0))
    assert _skill_gradient(ratings) == (150.0, {"minimax": 150.0})


def test_search_advantage_with_no_random_mover_or_only_one_standing_reads_zero() -> None:
    assert _search_advantage(_table((TOP, 100.0), (LOW, 50.0))) == 0.0
    assert _search_advantage(_table(("random", 0.0))) == 0.0


def test_gradient_persistence_reads_past_a_short_kind_and_clips_at_two_and_a_half() -> None:
    """A two-rung kind first (skipped), then a ladder whose upper slope is three times its
    lower: the ratio 3 is clipped to 2.5."""
    ratings = _table(
        ("mcts:nodes=100", 0.0), ("mcts:nodes=200", 50.0),
        ("minimax:nodes=100", 0.0), ("minimax:nodes=200", 100.0),
        ("minimax:nodes=400", 200.0), ("minimax:nodes=800", 500.0),
    )
    assert _gradient_persistence(ratings) == (2.5, {"minimax": 2.5})


def test_decisive_by_tier_with_one_rung_reads_nothing() -> None:
    records = [_record(1, (TOP, TOP), 0), _record(2, (TOP, TOP), None)]
    assert _decisive_by_tier(records) == ({}, {})


def test_analysis_components_a_single_legal_move_is_a_sample() -> None:
    games = [GameAnalysis(seed=1, agents=(TOP, TOP), near_best=[1], legal_counts=[1])]
    diagnostics: dict[str, Any] = {}
    assert _analysis_components(CorpusAnalysis(games=games), diagnostics, {}) == {
        "decision_criticality": 0.0,
    }
    assert diagnostics == {"criticality_samples": 1}


def test_analysis_components_rounds_with_no_round_count_have_no_shares() -> None:
    games = [GameAnalysis(seed=1, agents=(TOP, TOP), rounds={"collisions": 2, "swaps": 1})]
    diagnostics: dict[str, Any] = {}
    assert _analysis_components(CorpusAnalysis(games=games), diagnostics, {}) == {}
    assert diagnostics == {"rounds": {"collisions": 2, "swaps": 1}}


def test_opening_diversity_reads_six_plies() -> None:
    """Two games that part only at their sixth ply are two openings."""
    games = [_record(1, (TOP, TOP), 0, moves=[1, 2, 3, 4, 5, 6, 9]),
             _record(2, (TOP, TOP), 0, moves=[1, 2, 3, 4, 5, 7, 9])]
    assert _opening_diversity(games) == 1.0


def test_spread_of_one_value_is_zero() -> None:
    assert components._spread([5.0]) == 0.0


def test_traversal_time_counts_a_type_standing_nowhere_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One type-0 piece stands (reach 1); type 1 (reach 3) stands nowhere and counts once:
    reaches 1 and 3 average 2 across a diameter of 6."""
    monkeypatch.setattr(components, "_displacements",
                        lambda compiled, mask: [{(0, 1)}, {(0, 3)}])
    compiled = SimpleNamespace(
        width=5, height=7, n_squares=2, initial_board=[1, 0],
        encode_piece=lambda player, piece_type: 10 * player + piece_type + 1,
    )
    assert _traversal_time(compiled) == 3.0  # type: ignore[arg-type]


def test_skill_gradient_reads_on_past_a_one_rung_kind() -> None:
    ratings = _table(("mcts:nodes=100", 0.0), ("minimax:nodes=100", 0.0),
                     ("minimax:nodes=400", 300.0))
    assert _skill_gradient(ratings) == (150.0, {"minimax": 150.0})


def test_budget_tiers_keeps_a_one_node_budget() -> None:
    assert _budget_tiers(["minimax:nodes=4", "minimax:nodes=1"]) == {
        "minimax": [(1, "minimax:nodes=1"), (4, "minimax:nodes=4")],
    }


def test_gradient_persistence_reads_on_past_a_kind_whose_lower_fit_falls() -> None:
    """mcts's lower slope is negative (skipped); minimax's halves climb 100 then 150."""
    ratings = _table(
        ("mcts:nodes=100", 50.0), ("mcts:nodes=200", 0.0), ("mcts:nodes=400", 100.0),
        ("minimax:nodes=100", 0.0), ("minimax:nodes=200", 100.0),
        ("minimax:nodes=400", 200.0), ("minimax:nodes=800", 350.0),
    )
    assert _gradient_persistence(ratings) == (1.5, {"minimax": 1.5})


def test_analysis_components_moves_only_by_an_unheld_type_count_no_idle_type() -> None:
    """Type B is placed for no player, so the held army is A alone, which never moved."""
    spec = {"pieces": [{"id": "A"}, {"id": "B"}],
            "setup": {"placements": [{"piece": "A", "squares": ["a1"]}]}}
    games = [GameAnalysis(seed=1, agents=(TOP, TOP), moves_by_type=[0, 5])]
    diagnostics: dict[str, Any] = {}
    assert _analysis_components(CorpusAnalysis(games=games), diagnostics, spec) == {
        "piece_utilisation": 1.0,
    }
    assert diagnostics == {"held_piece_types": 1, "idle_piece_types": 0,
                           "moves_by_type": [0, 5]}


def test_spy_summary_median_of_four_first_losses() -> None:
    spies = [
        {"lost_ply_0": ply, "lost_ply_1": None, "lost_how_0": "captured", "lost_how_1": None,
         "reply_turns": 0, "alternating_plies": 0, "announced_rounds": 0, "blind_rounds": 0}
        for ply in (40, 10, 30, 20)
    ]
    assert spy_summary(spies)["first_loss_median_ply"] == 25.0


def test_gradient_persistence_reads_a_lower_slope_under_one_and_keeps_three_places() -> None:
    """Budgets 1, 2, 4, 8 (doublings exact): the lower half climbs 0.75 a doubling, a
    slope under one that still counts, the upper 0.25, so the ratio is a third, kept to
    three places."""
    ratings = _table(("minimax:nodes=1", 0.0), ("minimax:nodes=2", 0.75),
                     ("minimax:nodes=4", 1.5), ("minimax:nodes=8", 1.75))
    assert _gradient_persistence(ratings) == (0.333, {"minimax": 0.333})


def test_analysis_components_an_idle_type_is_under_two_percent_of_the_moves() -> None:
    """One move in a hundred is idle; two in a hundred is exactly the share, and is not."""
    for counts, idle, entropy in (([1, 99], 1, 0.080793), ([2, 98], 0, 0.141441)):
        games = [GameAnalysis(seed=1, agents=(TOP, TOP), moves_by_type=counts)]
        diagnostics: dict[str, Any] = {}
        raw = _analysis_components(CorpusAnalysis(games=games), diagnostics, {})
        assert raw["piece_utilisation"] == pytest.approx(entropy, abs=1e-6)
        assert diagnostics == {"held_piece_types": 2, "idle_piece_types": idle,
                               "moves_by_type": counts}


def test_analysis_components_reads_series_of_unequal_length_to_the_shorter() -> None:
    """A game whose near-best and capture series are shorter than its legal counts (an
    analysis written before a series existed) is read over the plies they share."""
    games = [GameAnalysis(seed=1, agents=(TOP, TOP), near_best=[1], legal_counts=[4, 2],
                          capture_counts=[])]
    diagnostics: dict[str, Any] = {}
    assert _analysis_components(CorpusAnalysis(games=games), diagnostics, {}) == {
        "decision_criticality": 0.75,
    }
    assert diagnostics == {"criticality_samples": 1}


def test_traversal_time_reads_the_first_frame_of_a_facing_piece_on_a_narrow_board() -> None:
    """A four-way facing piece steps 1 or leaps 3 forward on a board two files wide and
    five ranks tall. Facing forward it reaches 1 and 3 (mean 2, diameter 4); turned a
    quarter, its three-square leap would leave the board."""
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    spec["board"] = {"width": 2, "height": 5}
    spec["pieces"] = [
        {"id": "A", "name": "a", "value": 1, "facing": {"axes": 4}, "moves": [
            {"kind": "slide", "vectors": [[0, 1]], "range": 1, "mode": "move"},
            {"kind": "leap", "vectors": [[0, 3]], "mode": "move"}]},
    ]
    spec["setup"]["placements"] = [{"piece": "A", "squares": ["a1", "b1"]}]
    assert _traversal_time(compile_spec(spec)) == 2.0


def test_distinct_game_rate_with_no_moves_played_reads_zero() -> None:
    assert components._distinct_game_rate([]) == 0.0
    assert components._distinct_game_rate([_record(1, (TOP, TOP), 0, moves=[])]) == 0.0
    records = [_record(1, (TOP, TOP), 0, moves=[3, 4]), _record(2, (TOP, TOP), 0, moves=[3, 4]),
               _record(3, (TOP, TOP), 0, moves=[5])]
    assert components._distinct_game_rate(records) == 2 / 3


def test_skill_gradient_reads_on_past_a_heuristic_standing() -> None:
    ratings = _table(("minimax:h=abc,nodes=100", 900.0), ("minimax:nodes=100", 0.0),
                     ("minimax:nodes=400", 300.0))
    assert _skill_gradient(ratings) == (150.0, {"minimax": 150.0})


def test_search_advantage_with_only_weak_standings_reads_zero() -> None:
    ratings = _table(("random", 0.0), ("random:seed=2", 10.0),
                     h2h={("random", "random:seed=2"): (0.75, 4)})
    assert _search_advantage(ratings) == 0.0


def test_opening_diversity_reads_no_further_than_six_plies() -> None:
    """Two games that part only at their seventh ply share one opening."""
    games = [_record(1, (TOP, TOP), 0, moves=[1, 2, 3, 4, 5, 6, 8]),
             _record(2, (TOP, TOP), 0, moves=[1, 2, 3, 4, 5, 6, 9])]
    assert _opening_diversity(games) == 0.0


def test_spec_facts_a_full_cache_evicts_its_oldest_entry_and_keeps_the_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight entries already held: a ninth spec is kept and the first entry goes, so the
    next call for the ninth is answered from the cache, the same object."""
    from collections import OrderedDict

    held: OrderedDict[str, Any] = OrderedDict((f"k{i}", None) for i in range(8))
    monkeypatch.setattr(components, "_FACTS", held)
    spec = json.loads((ROOT / "rulesets" / "dull.json").read_text(encoding="utf-8"))
    first = components.spec_facts(spec)
    assert list(held)[:7] == [f"k{i}" for i in range(1, 8)] and len(held) == 8
    assert components.spec_facts(spec) is first


def test_game_reading_is_frozen() -> None:
    """Kept by record identity and shared by every resample that draws the game."""
    reading = components.GameReading(lead_changes=0.5, volatility=0.25, irreversibility=0.125)
    with pytest.raises(FrozenInstanceError):
        reading.volatility = 1.0  # type: ignore[misc]
