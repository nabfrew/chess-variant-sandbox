"""Pins on `agents/features.py`'s context and helpers, on small inline specs (Q1152's
survivors): a scoring rule's ceiling at its floors, a score gap with no rule or no colouring,
and an ``own`` extinction rule only one side is subject to."""

from __future__ import annotations

from typing import Any

from test_evaluation_pins import rps, rps_spec

from chess_variant_sandbox.agents.features import (
    VOCABULARY,
    FeatureEvaluator,
    _clamp,
    _score_ceiling,
    _score_gap,
    feature_context,
)
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.spec import validate_spec

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
