"""Pins on `agents/features.py`'s context and helpers, on small inline specs (Q1152's
survivors): a scoring rule's ceiling at its floors, a score gap with no rule or no colouring,
and an ``own`` extinction rule only one side is subject to."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from test_evaluation_pins import rps, rps_spec

from chess_variant_sandbox.agents.features import (
    MAX_ABS_WEIGHT,
    VOCABULARY,
    FeatureEvaluator,
    _clamp,
    _score_ceiling,
    _score_gap,
    feature_context,
)
from chess_variant_sandbox.agents.heuristics import Heuristic, HeuristicError
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]

STEP = [{"kind": "leap", "vectors": [[1, 0], [1, 1]], "symmetry": "rotate4", "mode": "both"}]


def _evaluator(engine: Engine) -> FeatureEvaluator:
    return FeatureEvaluator.for_engine(engine, (0.0,) * len(VOCABULARY))


def test_a_region_of_one_square_a_side_has_a_ceiling_of_one() -> None:
    engine = rps(scoring={"count": "region", "region": ["c3"]})
    assert feature_context(engine.spec).score_ceiling == 1.0


def test_a_count_of_nothing_is_floored_at_one() -> None:
    engine = rps(scoring={"count": "pieces"})
    context = feature_context(engine.spec)
    assert context.score_ceiling == 6.0
    assert _score_ceiling(engine.spec, context.base, [0, 0, 0]) == 1.0
    assert _score_ceiling(engine.spec, context.base, [0, 2, 0]) == 2.0


def test_no_scoring_rule_and_no_colouring_are_no_gap() -> None:
    plain = rps()
    state = plain.initial_state()
    context = feature_context(plain.spec)
    assert _score_gap(state, context, 0) == 0.0
    assert _score_gap(state, context, 1) == 0.0
    painted = rps(territory={"mode": "permanent"}, scoring={"count": "territory"})
    context = feature_context(painted.spec)
    blank = painted.initial_state().with_(colour=b"")
    assert _score_gap(blank, context, 0) == 0.0
    assert _score_gap(blank, context, 1) == 0.0
    # A colouring with two of player 0's squares and none of player 1's.
    coloured = painted.initial_state().with_(colour=b"\x01\x01" + b"\x00" * 23)
    assert _score_gap(coloured, context, 0) == 2.0


def test_an_own_extinction_rule_only_one_side_is_subject_to_reads_each_side_alone() -> None:
    """Player 0 starts with an x and a y, player 1 with neither: player 1 has nothing to run
    out of, so player 0 losing its x is the whole margin."""
    extra: tuple[dict[str, Any], ...] = (
        {"id": "X", "name": "x", "value": 2, "moves": STEP},
        {"id": "Y", "name": "y", "value": 2, "moves": STEP},
    )
    spec = rps_spec(extra, extinction={"pieces": ["X", "Y"], "per_type": "own"})
    spec["setup"]["placements"] = [
        {"piece": "R", "player": 0, "squares": ["a1"]},
        {"piece": "X", "player": 0, "squares": ["b1"]},
        {"piece": "Y", "player": 0, "squares": ["c1"]},
        {"piece": "R", "player": 1, "squares": ["e5"]},
        {"piece": "P", "player": 1, "squares": ["d5"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = _evaluator(engine)
    assert evaluator.context.extinction_own == (((3, 1), (4, 1)), ())
    state = engine.make_state(
        {"a1": ("R", 0), "c1": ("Y", 0), "e5": ("R", 1), "d5": ("P", 1)}
    )
    index = VOCABULARY.index("extinction_margin")
    assert evaluator.features(engine, state, 0, 0)[index] == -1.0
    assert evaluator.features(engine, state, 1, 0)[index] == 1.0
    # At the opening each of player 0's kinds is all there: one of one is a whole, not half.
    assert evaluator.features(engine, engine.initial_state(), 0, 0)[index] == 0.0


def test_a_points_count_with_no_bounty_on_the_board_has_a_ceiling_of_one() -> None:
    """The bounty is on a player's type, and the wild piece standing at ply zero pays
    nothing: no wild bounty on the board, so the ceiling is its floor."""
    spec = rps_spec(({"id": "W", "name": "wolf", "value": 3, "moves": STEP},),
                    scoring={"count": "points"})
    spec["pieces"][0]["bounty"] = 1
    spec["rules"]["npcs"] = [{"piece": "W", "route": ["a3", "b3", "c3"], "bounty": 0}]
    assert validate_spec(spec) == [], validate_spec(spec)
    assert feature_context(Engine(spec).spec).score_ceiling == 1.0


def test_hand_material_reads_nothing_from_a_state_with_no_hands() -> None:
    """A drops game read off a state that carries no hands (as a hand-built one may): the
    term is zero rather than an index into nothing; with a held paper it is one sixth of the opening's six pieces of material."""
    engine = rps(drops={"from_captures": True})
    evaluator = _evaluator(engine)
    state = engine.initial_state()
    assert evaluator._hand_material(engine, state.with_(hands=()), 0, 0, evaluator.context) == 0.0
    held = state.with_(hands=((0, 1, 0), (0, 0, 0)))
    assert evaluator._hand_material(engine, held, 0, 0, evaluator.context) == 1 / 6


def test_the_feature_clamp_holds_each_end_and_passes_the_middle() -> None:
    assert _clamp(-1.5) == -1.0
    assert _clamp(1.5) == 1.0
    assert _clamp(-0.5) == -0.5


def test_prey_weight_counts_the_pieces_after_a_wild_piece_on_the_first_square() -> None:
    """A wolf on a1 (board order 0), player 0's rock, paper and scissors after it, player 1
    with a rock and a paper. Player 0's shares: rock 0.1 (no enemy scissors, the floor),
    paper and scissors 1/2; player 1's: rock and paper 1/3. Relational 1.1 - 2/3 against
    static 3 - 2, over five pieces of opening material: -17/150."""
    wolf = {"id": "W", "name": "wolf", "value": 3, "moves": STEP}
    spec = rps_spec((wolf,), reach=None)
    spec["rules"]["npcs"] = [{"piece": "W", "route": ["a1", "a2", "a3"]}]
    spec["setup"]["placements"] = [
        {"piece": "R", "player": 0, "squares": ["b1"]},
        {"piece": "P", "player": 0, "squares": ["c1"]},
        {"piece": "S", "player": 0, "squares": ["d1"]},
        {"piece": "R", "player": 1, "squares": ["e5"]},
        {"piece": "P", "player": 1, "squares": ["d5"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = _evaluator(engine)
    state = engine.initial_state()
    assert state.board[0] != 0 and state.board[0] != state.board[1]
    weight = evaluator._prey_weight(engine, state, 0, 0, evaluator.context)
    assert abs(weight - (-17 / 150)) < 1e-12
    assert abs(evaluator._prey_weight(engine, state, 1, 0, evaluator.context) - 17 / 150) < 1e-12


def test_advance_reads_a_board_two_ranks_high() -> None:
    """On 5x2 the span is 1: player 0's paper and scissors on rank 2 have gone all the way,
    its rock on rank 1 none of it, and player 1's rock on its own rank 2 none of it."""
    spec = rps_spec(reach=None)
    spec["board"] = {"width": 5, "height": 2}
    spec["setup"]["placements"] = [
        {"piece": "R", "player": 0, "squares": ["a1"]},
        {"piece": "P", "player": 0, "squares": ["b2"]},
        {"piece": "S", "player": 0, "squares": ["c2"]},
        {"piece": "R", "player": 1, "squares": ["e2"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    assert engine.spec.feature_rank_span == 1
    index = VOCABULARY.index("advance")
    assert _evaluator(engine).features(engine, engine.initial_state(), 0, 0)[index] == 0.5


# ------------------------------------------------------------ Q1153's ninth queue


def test_no_scoring_rule_has_a_ceiling_of_one() -> None:
    assert feature_context(rps().spec).score_ceiling == 1.0


def test_a_points_count_sums_the_bounty_under_wild_pieces_only() -> None:
    """A route whose last square, c1, holds player 0's scissors at ply zero: its bounty is
    no wild piece's, so the ceiling is the bounty under the wolf on a3 alone, 2."""
    spec = rps_spec(({"id": "W", "name": "wolf", "value": 3, "moves": STEP},),
                    scoring={"count": "points"})
    spec["rules"]["npcs"] = [{"piece": "W", "route": ["a3", "b2", "c1"], "bounty": 2}]
    assert validate_spec(spec) == [], validate_spec(spec)
    assert feature_context(Engine(spec).spec).score_ceiling == 2.0


# ------------------------------------------------------------ Q1153's tenth queue


def test_the_contexts_evaluator_is_material_alone() -> None:
    engine = rps()
    base = feature_context(engine.spec).base
    assert base.mobility_weight == 0.0
    state = engine.make_state({"a1": ("R", 0), "e5": ("R", 1)})
    assert base.evaluate(engine, state, 0) == base.material(state, 0)


def test_each_side_races_to_its_own_goal() -> None:
    """e5 for player 0 (square 24), rotated to a1 (square 0) for player 1."""
    assert feature_context(rps().spec).reach_squares == (frozenset({24}), frozenset({0}))


def test_prey_weight_is_nothing_when_one_side_holds_no_prey() -> None:
    """Player 0 has no pieces: player 1 has nothing to win, its share is undefined and the
    derived valuation stands, so the term is zero for each side."""
    engine = rps()
    evaluator = _evaluator(engine)
    state = engine.make_state({"e5": ("R", 1), "d5": ("P", 1)})
    assert evaluator._prey_weight(engine, state, 1, 0, evaluator.context) == 0.0
    assert evaluator._prey_weight(engine, state, 0, 0, evaluator.context) == 0.0


def test_an_own_extinction_count_stops_at_its_sides_last_byte() -> None:
    """Both sides start with one rock, only player 0 with a paper: player 0's count of rocks
    is one, not two, though player 1's rock is the byte just past player 0's block."""
    spec = rps_spec(reach=None, extinction={"pieces": ["R", "P"], "per_type": "own"})
    del spec["setup"]["placements"][4]  # player 1's paper
    assert validate_spec(spec) == [], validate_spec(spec)
    assert feature_context(Engine(spec).spec).extinction_own == (((0, 1), (1, 1)), ((0, 1),))


# ------------------------------------------------------------ Q1153's eleventh queue


def test_a_piece_turned_all_the_way_round_points_minus_one() -> None:
    """Player 0's turnable rock faces backward, player 1's forward: -1 and +1 for player 0,
    over two pieces, is -1."""
    spec = rps_spec(reach=None)
    spec["pieces"][0]["facing"] = {"axes": 4}
    spec["pieces"][0]["moves"] = [{"kind": "leap", "vectors": [[0, 1]], "mode": "both"}]
    spec["setup"]["placements"][0]["facing"] = "backward"
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = _evaluator(engine)
    index = VOCABULARY.index("facing_alignment")
    state = engine.initial_state()
    assert evaluator.features(engine, state, 0, 0)[index] == -1.0
    assert evaluator.features(engine, state, 1, 0)[index] == 1.0


def test_a_weight_past_ten_is_refused_and_ten_is_not() -> None:
    assert MAX_ABS_WEIGHT == 10.0
    Heuristic("edge", "all material", (10.0,) + (0.0,) * (len(VOCABULARY) - 1))
    with pytest.raises(HeuristicError, match="beyond"):
        Heuristic("past", "all material", (10.5,) + (0.0,) * (len(VOCABULARY) - 1))


def test_a_check_limit_of_one_is_read() -> None:
    """One check is the whole limit: a check given and none taken is the whole margin."""
    spec = load_spec(ROOT / "rulesets" / "threecheck.json")
    spec["rules"]["check_limit"] = 1
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = _evaluator(engine)
    state = engine.initial_state().with_(checks=(1, 0))
    assert evaluator._check_margin(engine, state, 0, 0, evaluator.context) == 1.0
    assert evaluator._check_margin(engine, state, 1, 0, evaluator.context) == -1.0


def test_an_own_extinction_rule_only_the_second_side_is_subject_to_reads_each_side_alone() -> None:
    """The mirror of the first-side case: player 1 starts with an x and a y, player 0 with
    neither, and player 1 losing its x is the whole margin."""
    extra: tuple[dict[str, Any], ...] = (
        {"id": "X", "name": "x", "value": 2, "moves": STEP},
        {"id": "Y", "name": "y", "value": 2, "moves": STEP},
    )
    spec = rps_spec(extra, extinction={"pieces": ["X", "Y"], "per_type": "own"})
    spec["setup"]["placements"] = [
        {"piece": "R", "player": 0, "squares": ["a1"]},
        {"piece": "P", "player": 0, "squares": ["b1"]},
        {"piece": "R", "player": 1, "squares": ["e5"]},
        {"piece": "X", "player": 1, "squares": ["d5"]},
        {"piece": "Y", "player": 1, "squares": ["c5"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = _evaluator(engine)
    assert evaluator.context.extinction_own == ((), ((3, 1), (4, 1)))
    state = engine.make_state(
        {"a1": ("R", 0), "b1": ("P", 0), "e5": ("R", 1), "c5": ("Y", 1)}
    )
    index = VOCABULARY.index("extinction_margin")
    assert evaluator.features(engine, state, 1, 0)[index] == -1.0
    assert evaluator.features(engine, state, 0, 0)[index] == 1.0


def test_reach_progress_is_the_cores_arithmetic_to_the_last_bit() -> None:
    """On 4x4 the goal diameter is 3: player 0's rock one step from d4, player 1's two from
    a1. `features.rs` computes each side's nearness as `1 - d/3` and subtracts, which in
    doubles is 0.33333333333333337 -- not the 1/3 that `d1/3 - d0/3` would round to."""
    spec = rps_spec()
    spec["board"] = {"width": 4, "height": 4}
    spec["rules"]["reach"]["squares"] = ["d4"]
    spec["setup"]["placements"] = [
        {"piece": "R", "player": 0, "squares": ["a1"]},
        {"piece": "P", "player": 0, "squares": ["b1"]},
        {"piece": "S", "player": 0, "squares": ["c1"]},
        {"piece": "R", "player": 1, "squares": ["d4"]},
        {"piece": "P", "player": 1, "squares": ["c4"]},
        {"piece": "S", "player": 1, "squares": ["b4"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = _evaluator(engine)
    assert evaluator.context.goal_diameter == 3.0
    state = engine.make_state({"c3": ("R", 0), "b3": ("R", 1)})
    index = VOCABULARY.index("reach_progress")
    assert evaluator.features(engine, state, 0, 0)[index] == 0.33333333333333337


# ------------------------------------------------------------ Q1153's twelfth queue


def test_an_army_worth_nothing_is_normalised_by_one() -> None:
    """Only the two royals stand on the opening board, priced at nothing: the material
    total falls back to exactly one."""
    spec = rps_spec(reach=None, win="royal_capture")
    royal = {"id": "K", "name": "king", "royal": True, "value": 0, "captures": [],
             "moves": STEP}
    spec["pieces"].insert(0, royal)
    spec["setup"]["placements"] = [
        {"piece": "K", "player": 0, "squares": ["e1"]},
        {"piece": "K", "player": 1, "squares": ["a5"]},
    ]
    assert validate_spec(spec) == [], validate_spec(spec)
    assert feature_context(Engine(spec).spec).total_material == 1.0


def test_no_colouring_rule_has_a_territory_threshold_of_zero() -> None:
    context = feature_context(rps().spec)
    assert context.has_territory is False
    assert context.territory_win_at == 0


# ------------------------------------------------------------ Q1153's fourteenth queue


def test_a_feature_context_is_frozen() -> None:
    context = feature_context(rps().spec)
    with pytest.raises(FrozenInstanceError):
        context.n_types = 9  # type: ignore[misc]


def test_a_counted_kind_absent_from_the_opening_is_measured_against_one() -> None:
    """Rocks and x's summed, and nobody starts with an x: player 1 holding one x is a whole
    piece of margin against player 0, -1 and not -1/2."""
    extra: tuple[dict[str, Any], ...] = ({"id": "X", "name": "x", "value": 2, "moves": STEP},)
    spec = rps_spec(extra, reach=None, extinction={"pieces": ["R", "X"]})
    assert validate_spec(spec) == [], validate_spec(spec)
    engine = Engine(spec)
    evaluator = _evaluator(engine)
    assert evaluator.context.extinction_initial == (2, 0)
    state = engine.make_state(
        {"a1": ("R", 0), "b1": ("P", 0), "e5": ("R", 1), "d5": ("X", 1)}
    )
    index = VOCABULARY.index("extinction_margin")
    assert evaluator.features(engine, state, 0, 0)[index] == -1.0
