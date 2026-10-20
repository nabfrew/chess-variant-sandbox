"""Schema 27 (Q942b): the round players' term for a rested royal under a standing attacker.

`docs/archive/plan-round-chase.md` §3 R1 and the Q942b row of §4.2. Under the rest (schema 27's
`rules.simultaneous.rest`, schema 29's `"rest": 1` on every piece since Q983a) a royal that moved cannot move the next round, so a royal resting where a standing enemy
piece reaches is all but lost, and a depth-one matrix player saw nothing of it. Both sides of
the boundary price it now (`eval.rs::rested_royal_falls`, its oracle
`agents/evaluation.py::rested_royal_falls`, `RESTED_ROYAL_PIECES` each): held to each other on the
four round test specs Q942a added, the test itself on hand-built positions (the lift, a
rested attacker, a second royal), and a `greedy` player that runs its royal to the one square
no standing piece reaches.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_matrix_players import core_round, oracle_round, walk

import chess_variant_sandbox._core as _core
from chess_variant_sandbox.agents import evaluation
from chess_variant_sandbox.agents.evaluation import RESTED_ROYAL_PIECES, rested_royal_falls
from chess_variant_sandbox.engine import Engine, GameState
from chess_variant_sandbox.engine.engine import _square_name
from chess_variant_sandbox.engine.moves import decode_move

HERE = Path(__file__).parent

# Q942a's four round test specs, one switch or more each; only two of them rest.
CHASE_SPECS = ("resting", "collide", "atomic_collide", "scored")


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / f"_simultaneous_{name}.json").read_text(encoding="utf-8"))


def place(
    engine: Engine,
    pieces: dict[str, tuple[str, int]],
    rest: tuple[str | None, str | None] = (None, None),
) -> GameState:
    """A position with these pieces and these rested squares (each resting one round, schema
    29's list), keyed by the slow hash."""
    rested = tuple(() if name is None else ((engine._square_index(name), 1),) for name in rest)
    state = engine.make_state(pieces).with_(rest=rested)
    key = engine._compute_hash(state)
    return state.with_(key=key, repetition=(key,))


def lists(engine: Engine, state: GameState) -> tuple[list[int], list[int]]:
    return (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))


def falls(engine: Engine, state: GameState, side: int) -> bool:
    return rested_royal_falls(engine.spec, state, side, lists(engine, state))


def test_the_price_is_the_cores() -> None:
    assert _core.RESTED_ROYAL_PIECES == RESTED_ROYAL_PIECES
    # Below every finished round in the round search (Q986b: `round_win - ply / 1000`, in
    # mean pieces `ROUND_WIN_PIECES`), so a royal actually taken is still the worse cell.
    assert 0 < RESTED_ROYAL_PIECES < evaluation.ROUND_WIN_PIECES - 1


# --------------------------------------------------------------- the test itself

# Player 1's king rests on b4, which player 0's standing bishop on d2 reaches (c3 empty).
RESTED = {
    "e1": ("K", 0), "d2": ("B", 0), "a1": ("R", 0),
    "b4": ("K", 1), "e4": ("P", 1),
}


def test_a_rested_royal_a_standing_piece_reaches_falls() -> None:
    engine = Engine(load("resting"))
    state = place(engine, RESTED, rest=(None, "b4"))
    assert falls(engine, state, 1) and not falls(engine, state, 0)
    evaluator = evaluation.Evaluator.for_engine(engine)
    both = lists(engine, state)
    counts = (len(both[0]), len(both[1]))
    with_term = evaluator.evaluate_round(engine, state, 0, counts, both)
    without = evaluator.evaluate_round(engine, state.with_(rest=((), ())), 0, counts, both)
    # Q986b: half a round win in this game's mean pieces, not half of `WIN_SCORE`.
    assert evaluator.rested_royal == RESTED_ROYAL_PIECES * 4.65
    assert with_term == without + evaluator.rested_royal
    # The probe's ruler passes no lists: the engine is asked for them, the same number.
    assert evaluator.evaluate_round(engine, state, 0, counts) == with_term


def test_the_attackers_own_rest_and_the_lift_spare_it() -> None:
    engine = Engine(load("resting"))
    # The bishop itself rests: it cannot strike next round, and nothing else reaches b4.
    assert not falls(engine, place(engine, RESTED, rest=("d2", "b4")), 1)
    # A lone king is lifted: its moves are all player 1 has, so it may run.
    lone = {name: piece for name, piece in RESTED.items() if name != "e4"}
    state = place(engine, lone, rest=(None, "b4"))
    assert engine.legal_moves_for(state, 1), "the lift offers the king its moves"
    assert not falls(engine, state, 1)
    # A rested piece that is not the royal is material, not a near-loss.
    assert not falls(engine, place(engine, RESTED, rest=(None, "e4")), 1)


def test_a_side_with_a_second_royal_has_one_to_spare() -> None:
    """Under `royal_capture` a side loses when its last royal goes, so a rested king beside
    a royal knight is a piece at stake, not the game."""
    spec = load("resting")
    for piece in spec["pieces"]:
        if piece["id"] == "N":
            piece["royal"] = True
    engine = Engine(spec)
    assert not falls(engine, place(engine, {**RESTED, "a5": ("N", 1)}, rest=(None, "b4")), 1)
    assert falls(engine, place(engine, RESTED, rest=(None, "b4")), 1)


# --------------------------------------------- the core and the oracle, held together


def _assert_term_rounds_agree(name: str, agents: tuple[str, ...], rounds: int, seed: int) -> None:
    engine = Engine(load(name))
    states = walk(engine, rounds, seed=seed)
    if engine.spec.rest:
        states.append(place(engine, RESTED, rest=(None, "b4")))
    for index, state in enumerate(states):
        for agent in agents:
            for seat in (0, 1):
                core = core_round(engine, agent, state, seat, seed=index)
                assert core == oracle_round(engine, agent, state, seat, seed=index), (
                    name, agent, index, seat,
                )


def test_the_core_and_the_oracle_agree_on_a_resting_spec() -> None:
    _assert_term_rounds_agree("resting", ("greedy", "matrix:nodes=250"), rounds=2, seed=11)


@pytest.mark.slow
@pytest.mark.parametrize("name", CHASE_SPECS)
def test_the_core_and_the_oracle_agree_on_every_chase_spec(name: str) -> None:
    _assert_term_rounds_agree(name, ("greedy", "matrix:nodes=600"), rounds=8, seed=11)


# --------------------------------------------------------- the player it makes

# Player 1's king on a3 is struck where it stands by the rook up the a-file, so it runs. Off
# the file (b2, b3, b4) nothing of player 0's reaches it next round. Up the file, a4 is out of
# the rook's reach this round (the king itself blocks the file), so no collision prices it:
# only the term says that a king resting there is taken by the rook if the rook stood. a2 the
# rook reaches now as well. Before Q942b greedy ran to a4 a quarter of the time.
RUN = {"e1": ("K", 0), "a1": ("R", 0), "a3": ("K", 1), "e4": ("P", 1)}
SAFE, UNSAFE = ("b2", "b3", "b4"), ("a2", "a4")


def _king_weights(engine: Engine, state: GameState, strategy: list[float]) -> dict[str, float]:
    weights: dict[str, float] = {}
    origin = engine._square_index("a3")
    for move, weight in zip(engine.legal_moves_for(state, 1), strategy, strict=True):
        start, target, _, _ = decode_move(move)
        if start == origin:
            weights[_square_name(target, engine.spec.width)] = weight
    return weights


@pytest.mark.parametrize("agent", ["greedy", "matrix:nodes=1600"])
def test_a_player_does_not_run_its_royal_to_a_square_a_standing_piece_reaches(agent: str) -> None:
    engine = Engine(load("resting"))
    state = place(engine, RUN)
    for seed in range(4):
        core = core_round(engine, agent, state, 1, seed=seed)
        assert core == oracle_round(engine, agent, state, 1, seed=seed)
        move, strategy, _, _ = core
        assert move is not None
        _, target, _, _ = decode_move(move)
        assert _square_name(target, engine.spec.width) in SAFE, engine.describe_move(state, move)
        weights = _king_weights(engine, state, strategy)
        assert sum(weights[name] for name in UNSAFE) < 0.001, weights


def test_without_the_term_the_same_player_would_run_up_the_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: the oracle with the price at nought runs to a4 as often as to any safe
    square, so the test above is the term's doing and not the position's."""
    engine = Engine(load("resting"))
    state = place(engine, RUN)
    monkeypatch.setattr(evaluation, "RESTED_ROYAL_PIECES", 0.0)
    _, strategy, _, _ = oracle_round(engine, "greedy", state, 1, seed=0)
    weights = _king_weights(engine, state, strategy)
    assert weights["a4"] > 0.2, weights
