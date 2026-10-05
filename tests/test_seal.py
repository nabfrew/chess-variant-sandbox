"""The race term reads who holds the goal (Q708, F1).

`docs/reference/intransitive-quiet-endgame.md` §d has the design and the measurements. A
`reach` square held by a piece its racing side can never capture is *sealed*: no goal
for that side. These hold what the seal reads (the capture relation and nothing else,
on constructed positions in *Intransitive* and `rps-v3`), where it must not fire, the
per-goal table the compiler adds for a side with several goal squares, and that the core
and the Python oracle score the sealed positions the same.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chess_variant_sandbox.agents.evaluation import Evaluator, seal_goals
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.bridge import build_tables
from chess_variant_sandbox.engine.compile import UNREACHABLE
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.analysis import _root_scores, _score_roots

RULESETS = Path(__file__).resolve().parents[1] / "rulesets"


def _engine(name: str) -> Engine:
    return Engine(load_spec(RULESETS / f"{name}.json"))


def _fortress(engine: Engine, placement: dict[str, tuple[str, int]], side: int) -> GameState:
    return engine.make_state(placement, side_to_move=side, moved=list(placement), ply=100)


# Q702's fortress: two rocks each, each side's blocker on its own home corner (the other
# side's goal), each attacker next to the other side's held goal. No rock may take a rock.
FORTRESS = {"a1": ("R", 0), "f6": ("R", 0), "i9": ("R", 1), "b2": ("R", 1)}


def _race(engine: Engine, evaluator: Evaluator, state: GameState, player: int) -> float:
    return evaluator.race_progress(state, player)


def test_both_corners_held_seals_both_goals() -> None:
    engine = _engine("intransitive")
    evaluator = Evaluator.for_engine(engine)
    state = _fortress(engine, FORTRESS, 1)
    assert evaluator.sealed_goals(state) == (1, 1)
    # Both sides as far away as they can be: the race is level, whatever the attackers do.
    assert _race(engine, evaluator, state, 0) == 0.0
    moved = dict(FORTRESS)
    del moved["f6"]
    moved["c3"] = ("R", 0)
    assert _race(engine, evaluator, _fortress(engine, moved, 1), 1) == 0.0


def test_a_blocker_stepping_off_its_corner_opens_the_race() -> None:
    """The defect Q702 found: stepping off cost nothing. Now it costs the other side's race."""
    engine = _engine("intransitive")
    spec = engine.spec
    evaluator = Evaluator.for_engine(engine)
    off = dict(FORTRESS)
    del off["i9"]
    off["h8"] = ("R", 1)
    state = _fortress(engine, off, 0)
    assert evaluator.sealed_goals(state) == (0, 1)
    rock = spec.piece_ids.index("R")
    attacker = spec.goal_distance[0][rock][engine._square_index("f6")]
    diameter = float(spec.goal_diameter)
    # Player 1's own goal stays sealed (diameter); player 0's attacker is `attacker` away.
    expected = (attacker - diameter) / diameter
    assert _race(engine, evaluator, state, 1) == pytest.approx(expected)
    assert _race(engine, evaluator, state, 1) < 0.0
    held = _fortress(engine, FORTRESS, 0)
    assert evaluator.evaluate(engine, state, 1) < evaluator.evaluate(engine, held, 1)


def test_a_goal_held_by_a_piece_that_can_be_captured_is_not_sealed() -> None:
    """The fixture where the rule must not fire: player 0 keeps a paper, which takes rock."""
    engine = _engine("intransitive")
    spec = engine.spec
    evaluator = Evaluator.for_engine(engine)
    placement = dict(FORTRESS)
    placement["d4"] = ("P", 0)
    state = _fortress(engine, placement, 1)
    sealed = evaluator.sealed_goals(state)
    assert sealed == (0, 1)
    # Player 0's distance is the compiled union table's, as before the seal existed.
    nearest = min(
        spec.goal_distance[0][spec.piece_ids.index(piece)][engine._square_index(square)]
        for square, (piece, owner) in placement.items()
        if owner == 0
    )
    assert evaluator._nearest_to_goal(state, 0, -1, sealed[0]) == float(nearest)


def test_a_goal_held_by_ones_own_piece_is_not_sealed() -> None:
    """Player 0's rock on its own goal: it can step off, so nothing is sealed."""
    engine = _engine("intransitive")
    evaluator = Evaluator.for_engine(engine)
    state = _fortress(engine, {"c3": ("R", 0), "i9": ("R", 0), "e5": ("R", 1)}, 1)
    assert evaluator.sealed_goals(state) == (0, 0)


@pytest.mark.parametrize("name", ["chess", "kingofthehill", "racingkings", "dobutsu", "rps-v5"])
def test_no_seal_without_a_priced_race(name: str) -> None:
    """No `reach`, a royal racer, a `safe` arrival or drops: the seal is undefined."""
    assert seal_goals(_engine(name).spec) is None


def test_one_goal_square_a_side_sends_no_per_goal_table() -> None:
    """*Intransitive*'s tables do not move: one goal square is sealed whole or not at all."""
    engine = _engine("intransitive")
    assert engine.spec.goal_distance_each == ()
    assert build_tables(engine.spec)["goal_distance_each"] == []


def test_the_per_goal_table_unions_to_the_goal_table() -> None:
    engine = _engine("rps-v3")
    spec = engine.spec
    goals = seal_goals(spec)
    assert goals is not None and len(goals[0]) == len(goals[1]) == 9
    each = spec.goal_distance_each
    for player in (0, 1):
        assert len(each[player]) == 9
        for piece_type in range(spec.n_types):
            for square in range(spec.n_squares):
                union = spec.goal_distance[player][piece_type][square]
                assert union == min(table[piece_type][square] for table in each[player])


# rps-v3: player 0 races to rank 9 and keeps only rocks; player 1's rocks hold d9, e9, f9.
PARTIAL = {"e8": ("R", 0), "a1": ("R", 0), "d9": ("R", 1), "e9": ("R", 1), "f9": ("R", 1)}


def test_some_goals_sealed_reads_the_nearest_open_one() -> None:
    engine = _engine("rps-v3")
    spec = engine.spec
    evaluator = Evaluator.for_engine(engine)
    state = _fortress(engine, PARTIAL, 1)
    goals = seal_goals(spec)
    assert goals is not None
    held = {engine._square_index(square) for square in ("d9", "e9", "f9")}
    expected_mask = sum(1 << index for index, square in enumerate(goals[0]) if square in held)
    sealed = evaluator.sealed_goals(state)
    assert sealed[0] == expected_mask
    rock = spec.piece_ids.index("R")
    origin = engine._square_index("e8")
    open_goals = [index for index, square in enumerate(goals[0]) if square not in held]
    expected = min(spec.goal_distance_each[0][index][rock][origin] for index in open_goals)
    union = spec.goal_distance[0][rock][origin]
    assert expected != union, "the fixture must make the seal matter"
    distance = evaluator._goal_distance_of(state, 0, 1 << rock, sealed[0])
    assert distance == expected != UNREACHABLE


@pytest.mark.parametrize(
    "name,placement,side",
    [
        ("intransitive", FORTRESS, 1),
        ("intransitive", {**{k: v for k, v in FORTRESS.items() if k != "i9"}, "h8": ("R", 1)}, 0),
        ("intransitive", {**FORTRESS, "d4": ("P", 0)}, 1),
        ("rps-v3", PARTIAL, 0),
    ],
    ids=["both-sealed", "blocker-off", "capturable", "rps-v3-partial"],
)
def test_the_core_and_the_oracle_score_sealed_positions_alike(
    name: str, placement: dict[str, tuple[str, int]], side: int
) -> None:
    """The probe searches through the seal on both sides of the boundary, to the bit.

    The probe has no mobility term, so the oracle is asked without one (as
    `test_immunity.py`'s probe test does).
    """
    engine = _engine(name)
    evaluator = Evaluator.for_engine(engine, mobility_weight=0.0)
    state = _fortress(engine, placement, side)
    legal = engine.legal_moves(state)
    assert len(legal) > 1
    assert _score_roots(engine, evaluator, state, legal) == _root_scores(
        engine, evaluator, state, legal
    )
