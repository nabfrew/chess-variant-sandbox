"""Q1158: a spy game's announce and answer roots that run out play their best scored move.

Q1105 (`docs/reference/search-depth.md` §5) made `agents.rs::deepen` play the best root move
it had scored when the budget ended inside depth 1, instead of the first of its list. A spy
game's announced round has two roots of its own (`spy_players.rs::announce_minimax` and
`answer_minimax`, oracles `MinimaxAgent.select_announcement` / `select_answer`), which still
kept the first move of their list, in generation order. Now they keep the best they scored
(the first strict maximum, as a finished iteration does); deeper iterations are unchanged.

Held here, on `_spy_small` and tiny budgets, for both roots:

- the core (`Core.select_round`), the Python oracle and a line-by-line rule written out below
  agree on the move, the nodes and the finished depth;
- where depth 1 never finishes, the move is the best scored and, somewhere, not the list's
  first: the change is live (before Q1158 the core played the first, and this fails);
- where depth 1 finishes, the move is the pre-Q1158 player's.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from chess_variant_sandbox.agents import build_agent, native_spec
from chess_variant_sandbox.agents.minimax import INFINITY, BudgetExhausted
from chess_variant_sandbox.engine import Engine
from chess_variant_sandbox.engine.state import GameState
from chess_variant_sandbox.tournament.play import eval_values, play_game, replay, seat_rng

HERE = Path(__file__).parent


@cache
def _engine() -> Engine:
    return Engine(json.loads((HERE / "_spy_small.json").read_text(encoding="utf-8")))


@cache
def _announced() -> tuple[tuple[GameState, int, tuple[list[int], list[int]]], ...]:
    """Announced-round positions along a few greedy games, each with the announcer's seat
    and both lists; only those where the announcer and the answerer both have a choice."""
    engine = _engine()
    out = []
    for seed in range(4):
        for state in replay(engine, play_game(engine, ("greedy", "greedy"), seed))[:-1]:
            phase = engine.phase(state)
            if phase.name != "announced":
                continue
            lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
            if len(lists[0]) > 1 and len(lists[1]) > 1:
                out.append((state, phase.announcer, lists))
    assert len(out) >= 8, len(out)
    return tuple(out[:12])


def _row(agent: Any, state: GameState, announcer: int, moves: list[int], answers: list[int],
         depth: int) -> tuple[int | None, bool]:
    """The announcer's root, line for line: the best move scored, and whether it finished."""
    alpha, best, scored = -INFINITY, moves[0], False
    for move in moves:
        try:
            value = -agent._answer_row(state, announcer, move, answers, depth, 1, -INFINITY, -alpha)
        except BudgetExhausted:
            return (best if scored else None), False
        scored = True
        if value > alpha:
            alpha, best = value, move
    return best, True


def _replies(agent: Any, state: GameState, announcer: int, announced: int, moves: list[int],
             depth: int) -> tuple[int | None, bool]:
    """The answerer's root, line for line."""
    alpha, best, scored = -INFINITY, moves[0], False
    for move in moves:
        pair = (announced, move) if announcer == 0 else (move, announced)
        child = agent.engine.apply_round(state, *pair)
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


def _rule(player: str, state: GameState, seat: int, lists: tuple[list[int], list[int]],
          announced: int | None, legacy: bool) -> tuple[int, int]:
    """The deepening, written out: ``(move, depth reached)``; ``legacy`` is the player
    before Q1158, which kept the list's first move when depth 1 ran out."""
    agent: Any = build_agent(player, _engine())
    agent._begin_search()
    ordered = list(lists[seat])
    best, reached = ordered[0], 0
    for depth in range(1, agent.max_depth + 1):
        if announced is None:
            candidate, finished = _row(agent, state, seat, ordered, lists[1 - seat], depth)
        else:
            candidate, finished = _replies(agent, state, 1 - seat, announced, ordered, depth)
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


def _cases() -> list[tuple[GameState, int, tuple[list[int], list[int]], int | None]]:
    """Each position twice: the announcer asked, and the answerer against every third
    announcement."""
    cases: list[tuple[GameState, int, tuple[list[int], list[int]], int | None]] = []
    for state, announcer, lists in _announced():
        cases.append((state, announcer, lists, None))
        for announced in lists[announcer][::3]:
            cases.append((state, 1 - announcer, lists, announced))
    return cases


def _native(player: str, state: GameState, seat: int, announced: int | None) -> tuple[int, int, int]:
    engine = _engine()
    native = native_spec(player, engine.spec.hash)
    assert native is not None
    move, _, nodes, depth = engine.select_round(
        state, seat, native, eval_values(engine.spec), 7, announced
    )
    assert move is not None
    return move, nodes, depth


def _oracle(player: str, state: GameState, seat: int, lists: tuple[list[int], list[int]],
            announced: int | None) -> tuple[int, int]:
    agent: Any = build_agent(player, _engine())
    rng = seat_rng(7, seat)
    if announced is None:
        move = agent.select_announcement(state, seat, lists, rng)
    else:
        move = agent.select_answer(state, seat, lists, announced, rng)
    return move, agent.stats.nodes


# Where each root runs out here (61 cases, 12 of them the announcer's): the answerer's
# depth 1 costs a node a reply and ends inside it up to about 30 nodes, the announcer's a
# row of replies each and ends inside it up to about 150. Each budget names the roots whose
# best-scored move is, somewhere, not the list's first: the rule is live there.
_LIVE = {10: ("answer",), 30: ("announce", "answer"), 80: ("announce",)}


@pytest.mark.parametrize("nodes", list(_LIVE))
def test_a_first_ply_that_runs_out_plays_the_best_scored_move(nodes: int) -> None:
    player = f"minimax:nodes={nodes}"
    moved = {"announce": 0, "answer": 0}
    for state, seat, lists, announced in _cases():
        root = "announce" if announced is None else "answer"
        if root not in _LIVE[nodes]:
            continue
        move, spent, depth = _native(player, state, seat, announced)
        assert (move, spent) == _oracle(player, state, seat, lists, announced), root
        assert (move, depth) == _rule(player, state, seat, lists, announced, legacy=False), root
        if depth == 0:
            moved[root] += move != lists[seat][0]
    for root in _LIVE[nodes]:
        assert moved[root] > 0, moved


@pytest.mark.parametrize("nodes", [40, 100])
def test_a_search_that_finishes_a_ply_plays_as_before(nodes: int) -> None:
    """Every case that finishes depth 1 plays the pre-Q1158 player's move."""
    player = f"minimax:nodes={nodes}"
    finished = 0
    for state, seat, lists, announced in _cases():
        move, _, depth = _native(player, state, seat, announced)
        if depth == 0:
            continue
        finished += 1
        assert move == _rule(player, state, seat, lists, announced, legacy=True)[0]
    assert finished >= 40, finished
