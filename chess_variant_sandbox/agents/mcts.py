"""UCT search with random rollouts.

Worth having alongside alpha-beta for a reason specific to this project: the two
disagree in informative ways. Alpha-beta leans on the evaluator, so a ruleset where
material is a bad guide makes it look weak; MCTS leans on rollouts, so a ruleset with
enormous branching or very long games makes *it* look weak. When the index reports
strategic depth, agreement between two agents that fail differently is much better
evidence than either one alone.
"""

from __future__ import annotations

import math
import random
from typing import Any

from chess_variant_sandbox.agents.base import NativeSpec, SearchStats, register, register_native
from chess_variant_sandbox.engine import Engine, GameState, Outcome

EXPLORATION = 1.4


class _Node:
    __slots__ = ("state", "moves", "children", "visits", "wins", "parent", "move", "outcome")

    def __init__(self, state: GameState, moves: list[int], parent: _Node | None, move: int):
        self.state = state
        self.moves = moves
        self.children: list[_Node] = []
        self.visits = 0
        self.wins = 0.0
        self.parent = parent
        self.outcome: Outcome | None = None
        self.move = move

    @property
    def fully_expanded(self) -> bool:
        return len(self.children) == len(self.moves)


class MCTSAgent:
    """Monte Carlo tree search; the node budget is a simulation count."""

    def __init__(
        self,
        engine: Engine,
        nodes: int = 2_000,
        rollout_depth: int = 60,
        exploration: float = EXPLORATION,
    ) -> None:
        self.engine = engine
        self.budget = max(1, int(nodes))
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.name = f"mcts:nodes={self.budget}"
        self.stats = SearchStats()

    def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
        if len(moves) == 1:
            return moves[0]

        root = _Node(state, list(moves), None, -1)
        rng.shuffle(root.moves)
        self.stats.moves += 1

        for _ in range(self.budget):
            node = self._descend(root, rng)
            reward = self._rollout(node, rng)
            self._backpropagate(node, reward)

        self.stats.nodes += self.budget
        best = max(root.children, key=lambda child: child.visits)
        return best.move

    def _descend(self, node: _Node, rng: random.Random) -> _Node:
        engine = self.engine
        while True:
            if node.outcome is not None:
                return node
            if not node.fully_expanded:
                move = node.moves[len(node.children)]
                child_state = engine.apply(node.state, move)
                child_moves = engine.legal_moves(child_state)
                child = _Node(child_state, child_moves, node, move)
                child.outcome = engine.terminal(child_state, child_moves)
                rng.shuffle(child.moves)
                node.children.append(child)
                return child
            if not node.children:
                return node
            node = self._best_child(node)

    def _best_child(self, node: _Node) -> _Node:
        """UCT selection.

        A child's ``wins`` are banked from the perspective of the player to move *at
        that child*, which is usually the opponent of the player choosing here. The
        exploitation term is therefore ``1 - winrate``: picking the child with the
        highest stored score would be picking the move that most helps the opponent.

        "Usually" is the schema-4 caveat. With a turn longer than one move the child can
        belong to the *same* player, and its winrate is then already the right way round.
        Asked of the states rather than of the turn rule, because a ramp means the answer
        differs from one ply to the next of the same game.
        """
        log_visits = math.log(node.visits + 1)
        exploration = self.exploration
        side = node.state.side_to_move

        def key(child: _Node) -> float:
            if not child.visits:
                return float("inf")
            winrate = child.wins / child.visits
            exploit = winrate if child.state.side_to_move == side else 1.0 - winrate
            return exploit + exploration * math.sqrt(log_visits / child.visits)

        return max(node.children, key=key)

    def _rollout(self, node: _Node, rng: random.Random) -> float:
        """Play randomly to a verdict, scored for the player to move at ``node``.

        A rollout that hits the depth cap without finishing is scored as a draw rather
        than by evaluating the position. Reaching for the evaluator here would fold
        alpha-beta's blind spots straight back into MCTS and destroy the independence
        that makes running both agents worthwhile.
        """
        engine = self.engine
        perspective = node.state.side_to_move

        if node.outcome is not None:
            return node.outcome.score_for(perspective)

        state = node.state
        moves = node.moves
        for _ in range(self.rollout_depth):
            outcome = engine.terminal(state, moves)
            if outcome is not None:
                return outcome.score_for(perspective)
            if not moves:
                return 0.5
            state = engine.apply(state, rng.choice(moves))
            moves = engine.legal_moves(state)
        return 0.5

    def _backpropagate(self, node: _Node | None, reward: float) -> None:
        # ``reward`` is from the perspective of the player to move at the leaf, so it
        # flips on every step back towards the root at which the player actually
        # changes -- which with one move per turn is every step, and is what this used
        # to assume outright.
        while node is not None:
            node.visits += 1
            node.wins += reward
            parent = node.parent
            if parent is not None and parent.state.side_to_move != node.state.side_to_move:
                reward = 1.0 - reward
            node = parent


@register("mcts")
def _build_mcts(engine: Engine, options: dict[str, Any]) -> MCTSAgent:
    return MCTSAgent(
        engine,
        nodes=int(options.get("nodes", 2_000)),
        rollout_depth=int(options.get("rollout_depth", 60)),
        exploration=float(options.get("exploration", EXPLORATION)),
    )


@register_native("mcts")
def _native_mcts(options: dict[str, Any], ruleset_hash: str = "") -> NativeSpec:
    return (
        "mcts",
        int(options.get("nodes", 2_000)),
        0.0,
        0,
        int(options.get("rollout_depth", 60)),
        float(options.get("exploration", EXPLORATION)),
        # No evaluator: MCTS scores a leaf by playing it out, so a goal weight would
        # have nothing to weight.
        0.0,
        (),
        (),
        0.0,
        b"",
        0,
    )
