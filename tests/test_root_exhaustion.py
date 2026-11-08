"""Q1105: a search that runs out inside its first ply plays the best root move it scored.

Before Q1105, `agents.rs::search_root` threw away the moves it had scored when the budget
ended partway through depth 1, and `deepen` played `ordered[0]` of the shuffled list: a
random move. Now it keeps the best of the root moves already scored (the first strict
maximum, as a finished iteration does). Deeper iterations are unchanged: a budget that ends
in depth 2 or later plays the last finished iteration's move, as before.

Held here, on gardner (a small board, not chess) and tiny budgets:

- the core (`select_move`, `select_move_depth`) and the Python oracle
  (`MinimaxAgent.select_move`) agree move for move, and both agree with a line-by-line
  oracle of the rule written out below;
- where depth 1 never finishes, the move is the best-scored one and, somewhere, not the
  shuffled list's first: the change is live;
- where depth 1 finishes, the move is the pre-Q1105 player's (`_legacy_move`): nothing that
  completed a ply moves.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents import build_agent
from chess_variant_sandbox.agents.minimax import INFINITY, BudgetExhausted
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.engine import _pack
from chess_variant_sandbox.spec import load_spec
from chess_variant_sandbox.tournament.play import (
    eval_values,
    native_spec,
    play_game,
    play_game_python,
    seat_rng,
)

ROOT = Path(__file__).resolve().parent.parent
GARDNER = ROOT / "rulesets" / "gardner.json"


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine(load_spec(GARDNER))


def _positions(engine: Engine, count: int, seed: int) -> list[Any]:
    """``count`` positions with a choice, along one random line, none of them over."""
    rng = random.Random(seed)
    state = engine.initial_state()
    out: list[Any] = []
    while len(out) < count:
        moves = engine.legal_moves(state)
        if not moves or engine.terminal(state, moves) is not None:
            state = engine.initial_state()
            continue
        if len(moves) > 1:
            out.append(state)
        state = engine.apply(state, rng.choice(moves))
    return out


def _ordered(engine: Engine, state: Any, seed: int) -> list[int]:
    ordered = list(engine.legal_moves(state))
    seat_rng(seed, 0).shuffle(ordered)
    return ordered


def _root(agent: Any, state: Any, moves: list[int], depth: int) -> tuple[int | None, bool]:
    """One root iteration, line for line: the best move scored, and whether it finished."""
    alpha, best, scored = -INFINITY, moves[0], False
    for move in moves:
        child = agent.engine.apply(state, move)
        try:
            if child.side_to_move != state.side_to_move:
                score = -agent._negamax(child, depth - 1, 1, -INFINITY, -alpha)
            else:
                score = agent._negamax(child, depth - 1, 1, alpha, INFINITY)
        except BudgetExhausted:
            return (best if scored else None), False
        scored = True
        if score > alpha:
            alpha, best = score, move
    return best, True


def _move(engine: Engine, player: str, state: Any, seed: int, legacy: bool) -> tuple[int, int]:
    """The deepening, written out: ``(move, depth reached)``; ``legacy`` is the player
    before Q1105, which kept ``ordered[0]`` when depth 1 ran out."""
    agent: Any = build_agent(player, engine)
    agent._begin_search()
    ordered = _ordered(engine, state, seed)
    best, reached = ordered[0], 0
    for depth in range(1, agent.max_depth + 1):
        candidate, finished = _root(agent, state, ordered, depth)
        if not finished:
            if depth == 1 and candidate is not None and not legacy:
                best = candidate
            break
        assert candidate is not None
        best, reached = candidate, depth
        ordered.remove(best)
        ordered.insert(0, best)
        if agent._nodes >= agent.budget:
            break
    return best, reached


def _native(engine: Engine, player: str, state: Any, seed: int) -> tuple[int, int]:
    move, depth = engine.core.select_move_depth(
        _pack(state), native_spec(player, engine.spec.hash), eval_values(engine.spec), seed
    )
    return move, depth


@pytest.mark.parametrize("nodes", [4, 30, 60])
def test_a_first_ply_that_runs_out_plays_the_best_scored_move(engine: Engine, nodes: int) -> None:
    player = f"minimax:nodes={nodes}"
    states = _positions(engine, 10, seed=21)
    seeds = list(range(300, 300 + len(states)))
    ran_out = moved = 0
    for state, seed in zip(states, seeds, strict=True):
        native_move, depth = _native(engine, player, state, seed)
        oracle = build_agent(player, engine).select_move(
            state, engine.legal_moves(state), seat_rng(seed, 0)
        )
        assert native_move == oracle
        assert (native_move, depth) == _move(engine, player, state, seed, legacy=False)
        if depth == 0:
            ran_out += 1
            moved += native_move != _ordered(engine, state, seed)[0]
    # These budgets end inside depth 1 here, and from 30 nodes the best-scored move is
    # sometimes not the shuffled list's first: the rule is live. (Below that, the first
    # child's quiescence eats most of the budget, so few moves are scored.)
    assert ran_out > 0
    if nodes >= 30:
        assert moved > 0


def test_a_whole_game_that_runs_out_agrees_on_both_sides(engine: Engine) -> None:
    """A fixed-seed game at a budget that often ends inside depth 1, core against oracle:
    the moves and every count the record carries."""
    agents = ("minimax:nodes=40", "greedy")
    native = play_game(engine, agents, seed=7)
    in_python = play_game_python(engine, agents, seed=7)
    assert native.moves == in_python.moves
    assert (native.nodes, native.depth_sum, native.searches) == (
        in_python.nodes,
        in_python.depth_sum,
        in_python.searches,
    )
    # Some of the searching seat's moves finished no ply.
    assert native.depth_sum[0] < native.searches[0]


@pytest.mark.parametrize("nodes", [150, 400])
def test_a_search_that_finishes_a_ply_plays_as_before(engine: Engine, nodes: int) -> None:
    player = f"minimax:nodes={nodes}"
    states = _positions(engine, 10, seed=22)
    for index, state in enumerate(states):
        seed = 500 + index
        native_move, depth = _native(engine, player, state, seed)
        assert depth >= 1
        assert native_move == _move(engine, player, state, seed, legacy=True)[0]
