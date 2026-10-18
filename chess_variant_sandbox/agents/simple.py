"""The bottom two rungs of the strength ladder.

These exist to be beaten. The index's strategic-depth component asks whether thinking
harder reliably wins, and that question needs a floor to measure from: if a random
mover holds its own, the game has no depth to find. Random is also the control that
catches an engine bug masquerading as a strategy -- if a searching agent cannot beat
random, suspect the rules before the search.
"""

from __future__ import annotations

import random
from typing import Any

from chess_variant_sandbox.agents.base import NativeSpec, SearchStats, register, register_native
from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES, Evaluator
from chess_variant_sandbox.agents.matrix import (
    RoundSearch,
    greedy_announcement,
    greedy_answer,
    greedy_round,
    pure,
    refuse_alternating_objective,
)
from chess_variant_sandbox.agents.nets import require_net
from chess_variant_sandbox.agents.objective import objective_for
from chess_variant_sandbox.agents.objective_options import objective_values as _values
from chess_variant_sandbox.agents.objective_options import objective_weights as _objective
from chess_variant_sandbox.engine import Engine, GameState


class RandomAgent:
    """Uniform choice among legal moves."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.name = "random"
        self.stats = SearchStats()

    def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
        self.stats.moves += 1
        self.stats.nodes += len(moves)
        return rng.choice(moves)

    def select_round_move(
        self,
        state: GameState,
        seat: int,
        mine: list[int],
        theirs: list[int],
        rng: random.Random,
    ) -> int:
        """Schema 26: uniform over this seat's own moves, one draw a round."""
        self.stats.moves += 1
        self.stats.nodes += len(mine)
        return rng.choice(mine)

    def select_announcement(
        self, state: GameState, seat: int, lists: tuple[list[int], list[int]], rng: random.Random
    ) -> int:
        """Schema 27 (Q935a): an announced round's first move, uniform over this seat's own."""
        return self.select_round_move(state, seat, lists[seat], lists[1 - seat], rng)

    def select_answer(
        self,
        state: GameState,
        seat: int,
        lists: tuple[list[int], list[int]],
        announced: int,
        rng: random.Random,
    ) -> int:
        """Schema 27 (Q935a): the answer to an announced move, which a random player ignores."""
        return self.select_round_move(state, seat, lists[seat], lists[1 - seat], rng)


class GreedyAgent:
    """One ply of lookahead: take the move with the best immediate evaluation.

    Ties are broken randomly rather than by move order. Without that, a greedy player
    is fully deterministic and every game between two greedy players is the same game,
    which would make the repetitiveness metric measure the agent instead of the rules.
    """

    def __init__(
        self,
        engine: Engine,
        mobility: float = 0.05,
        goal: float = GOAL_PIECES,
        weights: tuple[float, ...] = (),
        values: tuple[float, ...] = (),
        immune: float = IMMUNE_PIECES,
        net: bytes = b"",
    ) -> None:
        self.engine = engine
        self.evaluator = objective_for(engine, mobility, goal, weights, values, immune, net)
        self.name = "greedy"
        self.stats = SearchStats()
        # Schema 26 (Q918b): the round arm's matrix search, and the strategy it last drew from.
        self._rounds: RoundSearch | None = None
        self.strategy: list[float] = []

    def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
        engine = self.engine
        player = state.side_to_move
        self.stats.moves += 1
        self.stats.nodes += len(moves)

        best_score = float("-inf")
        best_moves: list[int] = []
        spy = engine.spy
        for move in moves:
            successor = engine.apply(state, move)
            outcome = engine.terminal(successor)
            if outcome is not None:
                score = 0.0 if outcome.is_draw else (1e9 if outcome.winner == player else -1e9)
            elif spy and engine.is_round(successor):
                # Schema 27 (Q935b): a spy game's turn may leave a round, valued as a round
                # is (`docs/plan-spy.md` §3.5).
                score = self._round_evaluator().evaluate_round(
                    engine, successor, player, *_counted(engine, successor)
                )
            else:
                score = self.evaluator.evaluate(engine, successor, player)
            if score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
        return rng.choice(best_moves)

    def select_round_move(
        self,
        state: GameState,
        seat: int,
        mine: list[int],
        theirs: list[int],
        rng: random.Random,
    ) -> int:
        """Schema 26 (Q918b): the one-round matrix game, solved and drawn from
        (`agents/matrix.py`; the oracle for `rust/src/matrix.rs::greedy_round`)."""
        move, self.strategy = greedy_round(
            self._round_search(), self.stats, state, seat, mine, theirs, rng
        )
        return move

    def select_announcement(
        self, state: GameState, seat: int, lists: tuple[list[int], list[int]], rng: random.Random
    ) -> int:
        """Schema 27 (Q935b, `docs/plan-spy.md` §3.3): the one-round maximin, pure."""
        move = greedy_announcement(self._round_search(), self.stats, state, seat, lists)
        self.strategy = pure(lists[seat], move)
        return move

    def select_answer(
        self,
        state: GameState,
        seat: int,
        lists: tuple[list[int], list[int]],
        announced: int,
        rng: random.Random,
    ) -> int:
        """Schema 27 (Q935b): the best cell of the announced row, pure."""
        move = greedy_answer(self._round_search(), self.stats, state, seat, lists, announced)
        self.strategy = pure(lists[seat], move)
        return move

    def _round_evaluator(self) -> Any:
        from chess_variant_sandbox.agents.nets import NetEvaluator

        assert isinstance(self.evaluator, Evaluator | NetEvaluator), (
            "a round plays the built-in objective or a learned net (Q918g)"
        )
        return self.evaluator

    def _round_search(self) -> RoundSearch:
        if self._rounds is None:
            self._rounds = RoundSearch(self.engine, self._round_evaluator())
        return self._rounds


def _counted(
    engine: Engine, state: GameState
) -> tuple[tuple[int, int], tuple[list[int], list[int]]]:
    """Both players' move counts and lists on a round's board, for ``evaluate_round``."""
    lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
    return (len(lists[0]), len(lists[1])), lists


@register("random")
def _build_random(engine: Engine, options: dict[str, Any]) -> RandomAgent:
    return RandomAgent(engine)


@register_native("random")
def _native_random(options: dict[str, Any], ruleset_hash: str = "") -> NativeSpec:
    return ("random", 0, 0.0, 0, 0, 0.0, 0.0, (), (), 0.0, b"", 0)


@register("greedy")
def _build_greedy(engine: Engine, options: dict[str, Any]) -> GreedyAgent:
    if engine.simultaneous:
        refuse_alternating_objective(options, "greedy", spy=engine.spy)
    return GreedyAgent(
        engine,
        mobility=float(options.get("mobility", 0.05)),
        goal=float(options.get("goal", GOAL_PIECES)),
        weights=_objective(options, engine.spec.hash),
        values=_values(options, engine.spec.hash),
        immune=float(options.get("immune", IMMUNE_PIECES)),
        net=require_net(options, engine.spec.hash),
    )


@register_native("greedy")
def _native_greedy(options: dict[str, Any], ruleset_hash: str = "") -> NativeSpec:
    return (
        "greedy",
        0,
        float(options.get("mobility", 0.05)),
        0,
        0,
        0.0,
        float(options.get("goal", GOAL_PIECES)),
        _objective(options, ruleset_hash),
        _values(options, ruleset_hash),
        float(options.get("immune", IMMUNE_PIECES)),
        require_net(options, ruleset_hash),
        # One ply has no quiescence and no table to keep: the search repairs are
        # minimax's alone (Q502i).
        0,
    )
