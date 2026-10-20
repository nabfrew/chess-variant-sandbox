"""The computer players for a simultaneous game (schema 26, Q918b): the declared oracle.

`docs/archive/plan-simultaneous.md` §6.2. A round is a **matrix game**: rows are my moves, columns
the opponent's, each cell the position after resolving that pair. A good player in such a
game mixes, so every player above `random` here chooses a probability over its moves and
draws one with its own seat's stream.

* **The solver** is regret matching in its positive-part form (regret matching+): the two
  players' regrets updated in turn from a uniform start, each clamped at zero, for
  `REGRET_ITERATIONS` iterations, and the average strategy weighted by the iteration number.
  No randomness and no linear-programming solver: a fixed sequence of floating-point sums,
  added in the same order as `rust/src/matrix.rs`, so the two agree to the bit. Python's
  `sum` is compensated since 3.12, so every sum here is an explicit loop.
* **`greedy`** on a round game builds the whole matrix one round deep (every pair of legal
  moves, resolved, scored by the evaluator from its own seat), solves it and draws. Nodes are
  cells resolved.
* **`matrix:nodes=N`** deepens: pass 1 is greedy's matrix; each further pass re-values the
  cells by the value of the matrix game one round further on, in order of the previous
  pass's joint probability. One node is one resolved round. A cell whose re-valuation the
  budget cuts off keeps the value it had; the cells re-valued before it keep theirs.

`rust/src/matrix.rs` is the player; this file is its oracle (CLAUDE.md rule 3). It asks the
engine only what the engine answers for everyone: a player's legal moves, the position after
a round and the verdict.
"""

from __future__ import annotations

import math
import random
from typing import Any

from chess_variant_sandbox.agents.base import (
    NativeSpec,
    SearchStats,
    refuse_unknown_options,
    register,
    register_native,
)
from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES, Evaluator, round_win
from chess_variant_sandbox.engine import Engine, GameState, Outcome

# How many iterations of regret matching+ a matrix game is solved with. Chosen by the
# pre-registered test against exact equilibria of small matrices (`docs/archive/plan-simultaneous.md`,
# *Q918b, pre-registered*; `tests/test_matrix_players.py`), never by a game's score. Held to
# `_core.REGRET_ITERATIONS` by a test: changing it is a new player under every round name.
REGRET_ITERATIONS = 100

# The deepest pass `matrix:` may reach, as minimax's `MAX_DEPTH` bounds its iterations. The
# budget ends a search long before this on any board worth playing.
MAX_DEPTH = 64

# The objective options a round player may not take. A heuristic or searched
# objective reads one mover's mobility and was made on alternating play;
# `docs/archive/plan-simultaneous.md` §6.2 measures whether the round players need their own (Q918d).
# A learned net may (Q918g, §6.4): one trained on rounds reads the round's own state. Not in
# a spy game, whose turns and rounds no net has been trained across.
_ALTERNATING_OBJECTIVES = ("h", "values")
# Every option a round player reads; any other is refused (`refuse_unknown_options`).
MATRIX_OPTIONS = ("nodes", "mobility", "max_depth", "goal", "immune", "net")


def solve(
    values: list[float], rows: int, cols: int, iterations: int = REGRET_ITERATIONS
) -> tuple[list[float], list[float]]:
    """The average strategies of regret matching+ on ``values`` (row-major, the row
    player's payoff; the column player's is its negation): ``(rows', cols')``.

    Alternating updates, as CFR+ makes them: the row player answers the column player's
    current strategy, then the column player answers the row player's new one. Each
    player's average weights the strategy it took at iteration ``t`` by ``t``."""
    regret_r = [0.0] * rows
    regret_c = [0.0] * cols
    sum_r = [0.0] * rows
    sum_c = [0.0] * cols
    p = [1.0 / rows] * rows
    q = [1.0 / cols] * cols
    utility_r = [0.0] * rows
    utility_c = [0.0] * cols
    for t in range(1, iterations + 1):
        weight = float(t)
        for i in range(rows):
            total = 0.0
            base = i * cols
            for j in range(cols):
                total += values[base + j] * q[j]
            utility_r[i] = total
        expected = 0.0
        for i in range(rows):
            expected += p[i] * utility_r[i]
        for i in range(rows):
            updated = regret_r[i] + utility_r[i] - expected
            regret_r[i] = updated if updated > 0.0 else 0.0
        p = _normalised(regret_r)
        for i in range(rows):
            sum_r[i] += weight * p[i]
        for j in range(cols):
            total = 0.0
            for i in range(rows):
                total += values[i * cols + j] * p[i]
            utility_c[j] = -total
        expected = 0.0
        for j in range(cols):
            expected += q[j] * utility_c[j]
        for j in range(cols):
            updated = regret_c[j] + utility_c[j] - expected
            regret_c[j] = updated if updated > 0.0 else 0.0
        q = _normalised(regret_c)
        for j in range(cols):
            sum_c[j] += weight * q[j]
    return _normalised(sum_r), _normalised(sum_c)


def _normalised(weights: list[float]) -> list[float]:
    total = 0.0
    for weight in weights:
        total += weight
    if total > 0.0:
        return [weight / total for weight in weights]
    return [1.0 / len(weights)] * len(weights)


def game_value(values: list[float], p: list[float], q: list[float]) -> float:
    """The row player's expected payoff under ``p`` and ``q``, rows outer."""
    cols = len(q)
    total = 0.0
    for i, row_weight in enumerate(p):
        base = i * cols
        for j, col_weight in enumerate(q):
            total += row_weight * col_weight * values[base + j]
    return total


def draw(moves: list[int], strategy: list[float], rng: random.Random) -> int:
    """One move from ``strategy``, by one uniform draw over the cumulative sums in
    generation order (the last move if rounding leaves the draw past the total)."""
    point = rng.random()
    running = 0.0
    for move, weight in zip(moves, strategy, strict=True):
        running += weight
        if point < running:
            return move
    return moves[-1]


def terminal_value(outcome: Outcome, state: GameState, seat: int, win: float) -> float:
    """A finished round's value for ``seat``: ``win`` (``evaluation.round_win``, in the
    evaluator's units), a win sooner worth more by ``ply / 1000``, too little to cross a unit.
    Q986b: ±``WIN_SCORE`` drowned the evaluator in the solve, and a fixed number of units is a
    different number of pieces on every game (`docs/reference/round-hardest-weakness.md`)."""
    if outcome.is_draw:
        return 0.0
    signed = win - state.ply / 1000
    return signed if outcome.winner == seat else -signed


class _Exhausted(Exception):
    pass


class RoundSearch:
    """The matrix search, shared by greedy's round arm and `matrix:`."""

    def __init__(self, engine: Engine, evaluator: Any) -> None:
        self.engine = engine
        # The built-in `Evaluator`, or a learned net's `nets.NetEvaluator` (Q918g).
        self.evaluator = evaluator
        # Q986b: a finished round in this evaluator's units, `Player::round_win`'s product.
        self.round_win = round_win(engine.spec, tuple(evaluator.values))
        self.nodes = 0
        self.budget = 0
        self.bounded = False

    def _pair(self, seat: int, mine: int, theirs: int) -> tuple[int, int]:
        return (mine, theirs) if seat == 0 else (theirs, mine)

    def cells(
        self, state: GameState, seat: int, mine: list[int], theirs: list[int], depth: int
    ) -> list[float]:
        """Every cell's value for ``seat`` at ``depth`` rounds, rows outer."""
        values = []
        for a in mine:
            for b in theirs:
                values.append(self.cell(state, seat, a, b, depth))
        return values

    def cell(self, state: GameState, seat: int, a: int, b: int, depth: int) -> float:
        self.nodes += 1
        if self.bounded and self.nodes >= self.budget:
            raise _Exhausted
        child = self.engine.apply_round(state, *self._pair(seat, a, b))
        return self.value(child, seat, depth - 1)

    def value(self, state: GameState, seat: int, depth: int) -> float:
        engine = self.engine
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        outcome = engine.terminal(state)
        if outcome is not None:
            return terminal_value(outcome, state, seat, self.round_win)
        if depth == 0:
            counts = (len(lists[0]), len(lists[1]))
            return self.evaluator.evaluate_round(engine, state, seat, counts, lists)
        mine, theirs = lists[seat], lists[1 - seat]
        values = self.cells(state, seat, mine, theirs, depth)
        p, q = solve(values, len(mine), len(theirs))
        return game_value(values, p, q)


def round_evaluator(
    engine: Engine, mobility: float, goal: float, immune: float, net: bytes = b""
) -> Any:
    """The round players' objective: the built-in evaluator, or a learned net (Q918g),
    whose ``evaluate_round`` is ``Objective::evaluate_round``'s net arm."""
    if net:
        from chess_variant_sandbox.agents.nets import NetEvaluator

        derived = Evaluator.for_engine(engine, mobility_weight=mobility).values
        return NetEvaluator(engine, net, derived)
    return Evaluator.for_engine(
        engine, mobility_weight=mobility, goal_pieces=goal, immune_pieces=immune
    )


def refuse_alternating_objective(options: dict[str, Any], kind: str, spy: bool = False) -> None:
    """A round player plays the built-in objective only (`docs/archive/plan-simultaneous.md` §6.2), or
    a learned net trained on rounds (Q918g, §6.4); a spy game's players the built-in only."""
    refused = (*_ALTERNATING_OBJECTIVES, "net") if spy else _ALTERNATING_OBJECTIVES
    named = [key for key in refused if options.get(key) is not None]
    if named:
        raise ValueError(
            f"{kind} on a simultaneous game plays the built-in objective only, or a learned "
            f"net trained on rounds; {', '.join(named)} names an objective made for "
            "alternating play" + (" (no net plays a spy game)" if spy else "")
        )


def greedy_round(
    search: RoundSearch,
    stats: SearchStats,
    state: GameState,
    seat: int,
    mine: list[int],
    theirs: list[int],
    rng: random.Random,
) -> tuple[int, list[float]]:
    """Greedy's round arm: the one-round matrix, solved, drawn from."""
    stats.moves += 1
    if len(mine) == 1:
        return mine[0], [1.0]
    search.nodes = 0
    search.bounded = False
    values = search.cells(state, seat, mine, theirs, 1)
    stats.nodes += search.nodes
    p, _ = solve(values, len(mine), len(theirs))
    return draw(mine, p, rng), p


class MatrixAgent:
    """`matrix:nodes=N`: iterative deepening on matrix games (§6.2)."""

    def __init__(
        self,
        engine: Engine,
        nodes: int = 1600,
        mobility: float = 0.05,
        max_depth: int = MAX_DEPTH,
        goal: float = GOAL_PIECES,
        immune: float = IMMUNE_PIECES,
        net: bytes = b"",
    ) -> None:
        if not engine.simultaneous:
            raise ValueError(
                "matrix plays a simultaneous game (rules.simultaneous) only; on a game that "
                "alternates, the searching player is minimax"
            )
        self.engine = engine
        self.nodes = max(1, nodes)
        self.max_depth = max_depth
        self.search = RoundSearch(engine, round_evaluator(engine, mobility, goal, immune, net))
        self.name = f"matrix:nodes={nodes}"
        self.stats = SearchStats()
        self.depth = 0
        self.strategy: list[float] = []

    def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
        raise ValueError("matrix plays rounds; a game that alternates has none")

    def select_round_move(
        self,
        state: GameState,
        seat: int,
        mine: list[int],
        theirs: list[int],
        rng: random.Random,
    ) -> int:
        move, self.strategy, self.depth = matrix_round(
            self.search, self.stats, state, seat, mine, theirs, rng, self.nodes, self.max_depth
        )
        return move


def matrix_round(
    search: RoundSearch,
    stats: SearchStats,
    state: GameState,
    seat: int,
    mine: list[int],
    theirs: list[int],
    rng: random.Random,
    budget: int,
    max_depth: int,
) -> tuple[int, list[float], int]:
    """`matrix:nodes=budget`'s round (§6.2): the move, the strategy it was drawn from and the
    last pass that finished. Also how `minimax:nodes=N` plays a spy game's blind round
    (schema 27, Q935b; `docs/plan-spy.md` §3.3)."""
    stats.moves += 1
    if len(mine) == 1:
        return mine[0], [1.0], 0
    _values, p, _q, done = matrix_search(search, state, seat, mine, theirs, budget, max_depth)
    stats.nodes += search.nodes
    stats.depth_sum += done
    stats.searches += 1
    return draw(mine, p, rng), p, done


def matrix_search(
    search: RoundSearch,
    state: GameState,
    seat: int,
    mine: list[int],
    theirs: list[int],
    budget: int,
    max_depth: int,
) -> tuple[list[float], list[float], list[float], int]:
    """The deepening `matrix:nodes=budget` does at a root, without the draw: the cells'
    values, the strategy pair of the last solve and the last pass that finished
    (`matrix.rs::matrix_search`). ``search.nodes`` is what it spent."""
    search.nodes = 0
    search.budget = budget
    search.bounded = False
    rows, cols = len(mine), len(theirs)
    values = search.cells(state, seat, mine, theirs, 1)
    p, q = solve(values, rows, cols)
    done = 1
    search.bounded = True
    while search.nodes < budget and done < max_depth:
        depth = done + 1
        joint = [p[k // cols] * q[k % cols] for k in range(rows * cols)]
        order = sorted(range(rows * cols), key=lambda k: joint[k], reverse=True)
        complete = True
        for k in order:
            if search.nodes >= budget:
                complete = False
                break
            try:
                values[k] = search.cell(state, seat, mine[k // cols], theirs[k % cols], depth)
            except _Exhausted:
                complete = False
                break
        p, q = solve(values, rows, cols)
        if not complete:
            break
        done = depth
    return values, p, q, done


def round_value_for_first(
    search: RoundSearch,
    state: GameState,
    lists: tuple[list[int], list[int]],
    budget: int | None,
    max_depth: int = MAX_DEPTH,
) -> tuple[float, int]:
    """Q918g: the round player's own value of this round for player 0 and its last finished
    pass -- `matrix.rs::round_value_for_first`, the oracle for `Core.round_values`. ``budget``
    ``None`` is greedy (the one-round matrix); a number is `matrix:nodes=budget`."""
    mine, theirs = lists
    if budget is None:
        search.nodes = 0
        search.bounded = False
        values = search.cells(state, 0, mine, theirs, 1)
        p, q = solve(values, len(mine), len(theirs))
        return game_value(values, p, q), 1
    values, p, q, done = matrix_search(search, state, 0, mine, theirs, budget, max_depth)
    return game_value(values, p, q), done


def score(units: float, scale: float) -> float:
    """Q918i: built-in units as player 0's expected score, ``sigmoid(units / scale)``, written
    as ``training.data.squash`` and `matrix.rs::score` write it, so the three agree to the bit."""
    return 0.5 * (1.0 + math.tanh(units / (2.0 * scale)))


def round_score_for_first(
    search: RoundSearch,
    state: GameState,
    lists: tuple[list[int], list[int]],
    scale: float,
) -> tuple[float, int]:
    """Q918i: the one-round matrix's value for player 0 on expected score, the oracle for
    `Core.round_values(..., score_scale=scale)` (`matrix.rs::round_score_for_first`).

    Each cell is player 0's ``score(evaluate_round, scale)`` after the round, or 1, 1/2 or 0
    where the round ends the game; the matrix is solved by the same regret matching and its
    value is returned as it is, already a score. Only the one-round teacher has it."""
    engine = search.engine
    mine, theirs = lists
    values = []
    for a in mine:
        for b in theirs:
            child = engine.apply_round(state, a, b)
            after = (engine.legal_moves_for(child, 0), engine.legal_moves_for(child, 1))
            outcome = engine.terminal(child)
            if outcome is not None:
                values.append(0.5 if outcome.is_draw else (1.0 if outcome.winner == 0 else 0.0))
            else:
                counts = (len(after[0]), len(after[1]))
                units = search.evaluator.evaluate_round(engine, child, 0, counts, after)
                values.append(score(units, scale))
    p, q = solve(values, len(mine), len(theirs))
    return game_value(values, p, q), 1


def pure(moves: list[int], chosen: int) -> list[float]:
    """A pure choice as a strategy over ``moves``: one on ``chosen``, nothing elsewhere."""
    return [1.0 if move == chosen else 0.0 for move in moves]


def greedy_announcement(
    search: RoundSearch,
    stats: SearchStats,
    state: GameState,
    announcer: int,
    lists: tuple[list[int], list[int]],
) -> int:
    """Schema 27 (Q935b, `docs/plan-spy.md` §3.3): greedy as the announcer, the one-round
    maximin over the whole matrix, every cell from the announcer's seat (a node each), the
    first move of the best worst case. The oracle for `spy_players.rs::announce_greedy`."""
    mine, theirs = lists[announcer], lists[1 - announcer]
    stats.moves += 1
    if len(mine) == 1:
        return mine[0]
    search.nodes = 0
    search.bounded = False
    values = search.cells(state, announcer, mine, theirs, 1)
    stats.nodes += search.nodes
    cols = len(theirs)
    best, best_value = 0, -float("inf")
    for row in range(len(mine)):
        worst = float("inf")
        for value in values[row * cols:(row + 1) * cols]:
            if value < worst:
                worst = value
        if worst > best_value:
            best, best_value = row, worst
    return mine[best]


def greedy_answer(
    search: RoundSearch,
    stats: SearchStats,
    state: GameState,
    answerer: int,
    lists: tuple[list[int], list[int]],
    announced: int,
) -> int:
    """Schema 27 (Q935b): greedy as the answerer, the announced row only, a cell each of its
    own moves (a node each), the first best. The oracle for `spy_players.rs::answer_greedy`."""
    mine = lists[answerer]
    stats.moves += 1
    if len(mine) == 1:
        return mine[0]
    search.nodes = 0
    search.bounded = False
    best, best_value = mine[0], -float("inf")
    for move in mine:
        value = search.cell(state, answerer, move, announced, 1)
        if value > best_value:
            best, best_value = move, value
    stats.nodes += search.nodes
    return best


# Schema 27 (Q935b, `docs/plan-spy.md` §3.3): one name per budget on a spy game, whose
# `minimax:nodes=N` plays its blind rounds as `matrix:nodes=N` would.
SPY_REFUSAL = (
    "a spy game (pieces[].spy) is played by random, greedy and minimax:nodes=N, which play "
    "every phase (minimax plays the blind rounds as matrix would); matrix and mcts are refused"
)


@register("matrix")
def _build_matrix(engine: Engine, options: dict[str, Any]) -> MatrixAgent:
    from chess_variant_sandbox.agents.nets import require_net

    refuse_alternating_objective(options, "matrix")
    refuse_unknown_options(options, MATRIX_OPTIONS, "matrix")
    if engine.spy:
        raise ValueError(SPY_REFUSAL)
    return MatrixAgent(
        engine,
        nodes=int(options.get("nodes", 1600)),
        mobility=float(options.get("mobility", 0.05)),
        max_depth=int(options.get("max_depth", MAX_DEPTH)),
        goal=float(options.get("goal", GOAL_PIECES)),
        immune=float(options.get("immune", IMMUNE_PIECES)),
        net=require_net(options, engine.spec.hash),
    )


@register_native("matrix")
def _native_matrix(options: dict[str, Any], ruleset_hash: str = "") -> NativeSpec:
    from chess_variant_sandbox.agents.nets import require_net

    refuse_alternating_objective(options, "matrix")
    refuse_unknown_options(options, MATRIX_OPTIONS, "matrix")
    return (
        "matrix",
        int(options.get("nodes", 1600)),
        float(options.get("mobility", 0.05)),
        int(options.get("max_depth", MAX_DEPTH)),
        0,
        0.0,
        float(options.get("goal", GOAL_PIECES)),
        (),
        (),
        float(options.get("immune", IMMUNE_PIECES)),
        # Q918g: a learned net trained on rounds, or nothing (the built-in objective).
        require_net(options, ruleset_hash),
        0,
    )
