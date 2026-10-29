"""Alpha-beta search with iterative deepening under a node budget.

Iterative deepening is not an optimisation here, it is what makes a *node* budget
usable at all. A fixed-depth search cannot honour "you may look at 20,000 positions"
-- it either finishes early or blows through. Deepening one ply at a time means the
search always has a completed, usable answer in hand when the budget runs out, so two
agents given different budgets are genuinely the same player thinking for different
lengths of time. The strategic-depth metric compares exactly those pairs.
"""

from __future__ import annotations

import random
from typing import Any

from chess_variant_sandbox.agents.base import (
    NativeSpec,
    SearchStats,
    refuse_unknown_options,
    register,
    register_native,
)
from chess_variant_sandbox.agents.evaluation import GOAL_PIECES, IMMUNE_PIECES, WIN_SCORE, Evaluator
from chess_variant_sandbox.agents.matrix import (
    RoundSearch,
    matrix_round,
    pure,
    refuse_alternating_objective,
)
from chess_variant_sandbox.agents.nets import require_net
from chess_variant_sandbox.agents.objective import objective_for
from chess_variant_sandbox.agents.objective_options import objective_values as _values
from chess_variant_sandbox.agents.objective_options import objective_weights as _objective
from chess_variant_sandbox.engine import Engine, GameState, Outcome
from chess_variant_sandbox.engine.moves import (
    MOVE_CASTLE,
    MOVE_DROP,
    MOVE_DUCK,
    MOVE_EN_PASSANT,
    MOVE_FOLD,
    MOVE_PORTAL,
    MOVE_PROMOTION,
    MOVE_ROTATE,
    MOVE_SWEEP,
    decode_move,
    move_from,
    move_to,
    move_type,
)
from chess_variant_sandbox.engine.state import SpyPhase

INFINITY = float("inf")
MAX_DEPTH = 64

# How deep a quiescence search may go past the horizon before it settles for the standing
# evaluation. Bounded because a game where every move is a capture -- antichess, under
# ``must_capture`` -- would otherwise let it run to the node budget on every leaf, which
# is a full-width search wearing a different name.
QUIESCENCE_DEPTH = 4

# How many plies from the root the killer table has room for. ``MAX_DEPTH`` is 64 and a
# quiescence search runs past it, so this is the sum with a little room.
MAX_PLY = 96

# What a stored value means. A search inside a window only ever proves a *bound* for
# every move except the best, so a table that records the number without recording which
# kind of number it is hands back a bound as though it were a value.
EXACT, LOWER, UPPER = 0, 1, 2
# The core's table key is a u64 and wraps; Python's integers do not, so the mixing is
# masked back down. Without it the two searches would key the same position differently
# and collide in different places.
_U64 = (1 << 64) - 1

# Q502i: the generic search repairs (`docs/reference/general-player.md` §2 F), one bit
# each, the mirror of `agents.rs`. Each is spelled by its own option, ``keep=1``,
# ``threats=1``, ``loud_drops=1``; an unspelled one takes its default below, and a default
# is sent explicitly so the core holds none of its own.
#
# * ``keep``: the transposition table survives into this player's next search; entries
#   older than the previous search are dropped.
# * ``threats``: quiescence also chases a quiet move after which the engine's terminal
#   check says the mover has won, or, on the first quiescence ply only, one that leaves
#   the other side's royal attacked.
# * ``loud_drops``: on the first quiescence ply a drop is chased as a capture is.
KEEP_TABLE, THREATS, LOUD_DROPS = 1, 2, 4
REPAIR_OPTIONS = {"keep": KEEP_TABLE, "threats": THREATS, "loud_drops": LOUD_DROPS}
# Which repairs a player that spells none of them searches with: the ones that won their
# pre-registered duel (general-player.md §4.3 E9, graded in §6.7). ``threats`` won and is
# on; ``keep`` and ``loud_drops`` lost and are off, spelled only by their options. Moving
# this is a new player at every budget and owes the library a replay; held to
# ``_core.DEFAULT_REPAIRS`` by a test, so it moves the build fingerprint too. Today's
# player before Q502i is ``threats=0``.
DEFAULT_REPAIRS = THREATS

# Every option a minimax player reads; any other is refused (`refuse_unknown_options`).
OPTIONS = (
    "nodes", "mobility", "max_depth", "goal", "immune", "h", "values", "net",
    *REPAIR_OPTIONS,
)


def repairs_of(options: dict[str, Any]) -> int:
    """The repair bits these options spell, each unspelled one at its default."""
    repairs = 0
    for option, bit in REPAIR_OPTIONS.items():
        spelled = options.get(option)
        on = bool(DEFAULT_REPAIRS & bit) if spelled is None else _flag(option, spelled)
        repairs |= bit if on else 0
    return repairs


def _flag(option: str, value: Any) -> bool:
    if value in (0, 1, False, True):
        return bool(value)
    raise ValueError(f"{option}={value!r}: a search repair is spelled 0 or 1")


class BudgetExhausted(Exception):
    """Raised mid-search to abandon an incomplete iteration."""


class MinimaxAgent:
    """Negamax with alpha-beta pruning, a transposition table, and capture ordering."""

    # Schema 27 (Q935b): its round arm plays a spy game's blind rounds only; a game played in
    # rounds from the start refuses it (`tournament/play.py::_play_rounds_python`).
    plays_rounds = False

    def __init__(
        self,
        engine: Engine,
        nodes: int = 20_000,
        mobility: float = 0.05,
        max_depth: int = MAX_DEPTH,
        goal: float = GOAL_PIECES,
        weights: tuple[float, ...] = (),
        values: tuple[float, ...] = (),
        immune: float = IMMUNE_PIECES,
        net: bytes = b"",
        repairs: int = DEFAULT_REPAIRS,
    ) -> None:
        self.engine = engine
        self.evaluator = objective_for(engine, mobility, goal, weights, values, immune, net)
        self._squares = engine.spec.n_squares
        self._killers: list[list[int]] = [[-1, -1] for _ in range(MAX_PLY)]
        self._history: dict[int, int] = {}
        self.budget = max(1, int(nodes))
        self.max_depth = max_depth
        self.name = f"minimax:nodes={self.budget}"
        self.stats = SearchStats()
        self._nodes = 0
        self._table: dict[int, tuple[int, float, int, int]] = {}
        self.repairs = repairs
        # Which search this is, for ``keep``'s aging; the core's is a wrapping u16 and
        # only equality with this search and the previous one is ever asked.
        self._generation = 0
        # Schema 27 (Q935b): a spy game's tree crosses into rounds (`docs/plan-spy.md` §3.3);
        # its blind rounds are played by the matrix search, built when first asked.
        self._spy = engine.spy
        self._rounds: RoundSearch | None = None
        self.strategy: list[float] = []

    def select_move(self, state: GameState, moves: list[int], rng: random.Random) -> int:
        if len(moves) == 1:
            return moves[0]

        self._begin_search()

        # Shuffle once so that equal-scoring moves are not resolved by generation order.
        # Otherwise every game between two minimax agents of the same budget is
        # identical, and the repetitiveness metric ends up measuring the search.
        ordered = list(moves)
        rng.shuffle(ordered)
        best_move = ordered[0]

        reached = 0
        for depth in range(1, self.max_depth + 1):
            try:
                candidate = self._search_root(state, ordered, depth)
            except BudgetExhausted:
                break
            best_move = candidate
            reached = depth
            # Put the previous best first: alpha-beta prunes far harder with a good
            # first guess, so the next iteration is cheaper than this one was.
            ordered.remove(best_move)
            ordered.insert(0, best_move)
            if self._nodes >= self.budget:
                break

        self._end_search(reached)
        return best_move

    def _begin_search(self) -> None:
        self._nodes = 0
        self._generation = (self._generation + 1) & 0xFFFF
        if self.repairs & KEEP_TABLE:
            now, previous = self._generation, (self._generation - 1) & 0xFFFF
            self._table = {
                key: entry for key, entry in self._table.items() if entry[3] in (now, previous)
            }
        else:
            self._table.clear()
        for slot in self._killers:
            slot[0] = slot[1] = -1
        self._history.clear()
        self.stats.moves += 1

    def _end_search(self, reached: int) -> None:
        self.stats.nodes += self._nodes
        self.stats.depth_sum += reached
        self.stats.searches += 1

    # ------------------------------------------------------------ a spy game's rounds
    #
    # Schema 27 (Q935b, `docs/plan-spy.md` §3): the oracle for `rust/src/spy_players.rs`.
    # An announced round is two plies of the tree, the announcer's writing nothing to the
    # board and the answerer's resolving the round; neither mixes and neither draws.

    def select_announcement(
        self, state: GameState, seat: int, lists: tuple[list[int], list[int]], rng: random.Random
    ) -> int:
        """The announcer's move: the maximin by iterative deepening, pure (§3.2)."""
        mine = lists[seat]
        if len(mine) == 1:
            self.strategy = [1.0]
            return mine[0]
        self._begin_search()
        ordered, answers = list(mine), lists[1 - seat]
        best_move, reached = ordered[0], 0
        for depth in range(1, self.max_depth + 1):
            try:
                best_move = self._announce_root(state, seat, ordered, answers, depth)
            except BudgetExhausted:
                break
            reached = depth
            ordered.remove(best_move)
            ordered.insert(0, best_move)
            if self._nodes >= self.budget:
                break
        self._end_search(reached)
        self.strategy = pure(mine, best_move)
        return best_move

    def select_answer(
        self,
        state: GameState,
        seat: int,
        lists: tuple[list[int], list[int]],
        announced: int,
        rng: random.Random,
    ) -> int:
        """The answerer's move: the announced row searched alone, pure (§3.2)."""
        mine = lists[seat]
        if len(mine) == 1:
            self.strategy = [1.0]
            return mine[0]
        self._begin_search()
        ordered = list(mine)
        best_move, reached = ordered[0], 0
        for depth in range(1, self.max_depth + 1):
            try:
                best_move = self._answer_root(state, 1 - seat, announced, ordered, depth)
            except BudgetExhausted:
                break
            reached = depth
            ordered.remove(best_move)
            ordered.insert(0, best_move)
            if self._nodes >= self.budget:
                break
        self._end_search(reached)
        self.strategy = pure(mine, best_move)
        return best_move

    def select_round_move(
        self,
        state: GameState,
        seat: int,
        mine: list[int],
        theirs: list[int],
        rng: random.Random,
    ) -> int:
        """A spy game's blind round, played as `matrix:nodes=N` plays it (§3.3)."""
        if not self._spy:
            raise ValueError(
                "a simultaneous game (rules.simultaneous) is played by random, greedy and "
                "matrix:nodes=N; minimax and mcts search one mover at a time"
            )
        if self._rounds is None:
            assert isinstance(self.evaluator, Evaluator), "a round plays the built-in objective"
            self._rounds = RoundSearch(self.engine, self.evaluator)
        move, self.strategy, _ = matrix_round(
            self._rounds, self.stats, state, seat, mine, theirs, rng, self.budget, self.max_depth
        )
        return move

    def _announce_root(
        self, state: GameState, announcer: int, moves: list[int], answers: list[int], depth: int
    ) -> int:
        alpha = -INFINITY
        best_move = moves[0]
        for move in moves:
            value = -self._answer_row(state, announcer, move, answers, depth, 1, -INFINITY, -alpha)
            if value > alpha:
                alpha, best_move = value, move
        return best_move

    def _answer_root(
        self, state: GameState, announcer: int, announced: int, moves: list[int], depth: int
    ) -> int:
        alpha = -INFINITY
        best_move = moves[0]
        for move in moves:
            child = self.engine.apply_round(state, *_pair(announcer, announced, move))
            score = self._negamax(child, depth - 1, 1, alpha, INFINITY)
            if score > alpha:
                alpha, best_move = score, move
        return best_move

    def _round_leaf(self, state: GameState) -> float:
        outcome = self.engine.terminal_prefix(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)
        return self._round_leaf_lists(state)

    def _round_leaf_lists(self, state: GameState) -> float:
        engine = self.engine
        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        outcome = engine.terminal(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)
        return self._evaluate_round(state, lists)

    def _evaluate_round(self, state: GameState, lists: tuple[list[int], list[int]]) -> float:
        assert isinstance(self.evaluator, Evaluator), "a round plays the built-in objective"
        counts = (len(lists[0]), len(lists[1]))
        return self.evaluator.evaluate_round(self.engine, state, state.side_to_move, counts, lists)

    def _round_node(
        self, state: GameState, phase: SpyPhase, depth: int, ply: int, alpha: float, beta: float
    ) -> float:
        """A round inside the tree: a blind one a leaf, an announced one two plies."""
        engine = self.engine
        outcome = engine.terminal_prefix(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)
        if phase.name != "announced":
            return self._round_leaf_lists(state)
        announcer = phase.announcer

        key = self._table_key(state)
        cached = self._table.get(key)
        if cached is not None and cached[0] >= depth:
            _, value, bound, _ = cached
            if (
                bound == EXACT
                or (bound == LOWER and value >= beta)
                or (bound == UPPER and value <= alpha)
            ):
                return value

        lists = (engine.legal_moves_for(state, 0), engine.legal_moves_for(state, 1))
        outcome = engine.terminal(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)
        if depth <= 0:
            return self._evaluate_round(state, lists)
        announced = list(lists[announcer])
        answers = lists[1 - announcer]
        self._order(state, announced, ply)

        opened_at = alpha
        announcer_beta = -alpha
        announcer_alpha = -beta
        announcer_best = -INFINITY
        for move in announced:
            quiet = not self._is_loud(state, move)
            score = -self._answer_row(
                state, announcer, move, answers, depth, ply + 1, -announcer_beta, -announcer_alpha
            )
            if score > announcer_best:
                announcer_best = score
            if announcer_best > announcer_alpha:
                announcer_alpha = announcer_best
            if announcer_alpha >= announcer_beta:
                self._remember(move, quiet, depth, ply)
                break

        best = -announcer_best
        if best <= opened_at:
            bound = UPPER
        elif best >= beta:
            bound = LOWER
        else:
            bound = EXACT
        self._table[key] = (depth, best, bound, self._generation)
        return best

    def _answer_row(
        self,
        state: GameState,
        announcer: int,
        announced: int,
        answers: list[int],
        depth: int,
        ply: int,
        alpha: float,
        beta: float,
    ) -> float:
        """The answerer's ply: the best of its moves against ``announced``, ordered at
        ``ply``; the side does not change across a round, so a child's value passes."""
        moves = list(answers)
        self._order(state, moves, ply)
        best = -INFINITY
        for move in moves:
            quiet = not self._is_loud(state, move)
            child = self.engine.apply_round(state, *_pair(announcer, announced, move))
            score = self._negamax(child, depth - 1, ply + 1, alpha, beta)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._remember(move, quiet, depth, ply)
                break
        return best

    def _search_root(self, state: GameState, moves: list[int], depth: int) -> int:
        engine = self.engine
        alpha = -INFINITY
        best_move = moves[0]
        for move in moves:
            # See ``_negamax``: the negation and the window flip are only right when the
            # side actually changed, which a turn longer than one move makes conditional.
            child = engine.apply(state, move)
            if child.side_to_move != state.side_to_move:
                score = -self._negamax(child, depth - 1, 1, -INFINITY, -alpha)
            else:
                score = self._negamax(child, depth - 1, 1, alpha, INFINITY)
            if score > alpha:
                alpha, best_move = score, move
        return best_move

    def _negamax(
        self, state: GameState, depth: int, ply: int, alpha: float, beta: float
    ) -> float:
        self._nodes += 1
        if self._nodes >= self.budget:
            raise BudgetExhausted

        engine = self.engine
        if self._spy:
            phase = engine.phase(state)
            if phase.is_round:
                return self._round_node(state, phase, depth, ply, alpha, beta)
        # Asked before the table, not after, and no longer only on the rulesets whose
        # endings the evaluator reads. A cached value is a value for a position still
        # being played; a position that has already ended has an outcome instead. Two
        # positions sharing a key can differ in ply, in the no-progress clock and in how
        # often the position has repeated -- none of which the key carries -- so this was
        # never a territorial problem. Ungating it moves fifteen committed rulesets,
        # which is why it waited for this phase and its campaign.
        outcome = engine.terminal_prefix(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)

        key = self._table_key(state)
        cached = self._table.get(key)
        if cached is not None and cached[0] >= depth:
            # A stored number is only usable as an answer when the search that produced
            # it proved the right kind of thing. This used to return every entry as
            # though it were exact, which hands back a beta cutoff's lower bound as a
            # value -- and a lower bound is exactly what a cutoff refuses to establish.
            _, value, bound, _ = cached
            if (
                bound == EXACT
                or (bound == LOWER and value >= beta)
                or (bound == UPPER and value <= alpha)
            ):
                return value

        moves = engine.legal_moves(state)
        outcome = engine.terminal(state, moves)
        if outcome is not None:
            return self._terminal_value(outcome, state)

        if depth <= 0:
            # Not the standing evaluation. Iterative deepening stops on whichever depth
            # the budget allowed, which is as often odd as even, so a leaf is as likely
            # as not to be in the middle of an exchange -- and material scored there is a
            # number about a position nobody would ever stop in.
            return self._quiesce(state, QUIESCENCE_DEPTH, ply, alpha, beta)

        self._order(state, moves, ply)

        opened_at = alpha
        best = -INFINITY
        for move in moves:
            # Negamax negates the child and flips the window because the side is assumed
            # to alternate. Once a turn can be longer than one move it does not always,
            # and when the same player moves again the child's value is already from
            # this player's point of view -- so value and window both pass through.
            #
            # Asked as "did the side change" rather than "is there a turn rule", because
            # that is the only form that cannot drift out of step with the engine: a ramp
            # means the answer differs between one ply and the next.
            quiet = not self._is_loud(state, move)
            child = engine.apply(state, move)
            if child.side_to_move != state.side_to_move:
                score = -self._negamax(child, depth - 1, ply + 1, -beta, -alpha)
            else:
                score = self._negamax(child, depth - 1, ply + 1, alpha, beta)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._remember(move, quiet, depth, ply)
                break

        if best <= opened_at:
            bound = UPPER
        elif best >= beta:
            bound = LOWER
        else:
            bound = EXACT
        self._table[key] = (depth, best, bound, self._generation)
        return best

    def _quiesce(
        self, state: GameState, depth: int, ply: int, alpha: float, beta: float
    ) -> float:
        """Search on past the horizon until nothing is hanging.

        The standing evaluation is a lower bound on what the side to move can get, since
        it can decline to do anything loud -- that is the "stand pat", and it is what
        makes this a search over a *subset* of moves rather than a broken one.
        """
        self._nodes += 1
        if self._nodes >= self.budget:
            raise BudgetExhausted

        engine = self.engine
        # Schema 27: quiescence stops at a round (a capture in a round can be dodged).
        if self._spy and engine.is_round(state):
            return self._round_leaf(state)
        outcome = engine.terminal_prefix(state)
        if outcome is not None:
            return self._terminal_value(outcome, state)

        moves = engine.legal_moves(state)
        outcome = engine.terminal(state, moves)
        if outcome is not None:
            return self._terminal_value(outcome, state)

        stand_pat = self.evaluator.evaluate(engine, state, state.side_to_move, len(moves))
        if depth <= 0 or stand_pat >= beta:
            return stand_pat

        # Keep the loud moves, in generation order, so the two implementations settle
        # ties the same way.
        horizon = depth == QUIESCENCE_DEPTH
        loud = [
            move
            for move in moves
            if self._is_loud(state, move) or self._chases(state, move, horizon)
        ]
        self._order(state, loud, ply)

        best = stand_pat
        alpha = max(alpha, stand_pat)
        for move in loud:
            child = engine.apply(state, move)
            if child.side_to_move != state.side_to_move:
                score = -self._quiesce(child, depth - 1, ply + 1, -beta, -alpha)
            else:
                score = self._quiesce(child, depth - 1, ply + 1, alpha, beta)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def _is_loud(self, state: GameState, move: int) -> bool:
        """Whether a move changes material.

        Read off the encoding and the board rather than off any piece's name: a move onto
        an occupied square, an en-passant capture, which lands on an empty one, and a
        promotion, which adds material without taking any.

        **Two kinds end on an occupied square and take nothing.** A rotation ends where
        it began, so its target holds the piece that is turning; a file-based castling
        can send the king to its own rook's square. A quiescence search that chased
        either would never come to rest.
        """
        _, target, kind, promoted = decode_move(move)
        # Schema 22: a fold moved onto a piece takes nothing either.
        if kind not in (MOVE_ROTATE, MOVE_CASTLE, MOVE_FOLD, MOVE_DUCK) and state.board[target]:
            return True
        # Schema 19: a transit that promotes at its exit, as `agents.rs::is_loud` reads it.
        return kind in (MOVE_EN_PASSANT, MOVE_PROMOTION, MOVE_SWEEP) or (
            kind == MOVE_PORTAL and promoted != 0
        )

    def _chases(self, state: GameState, move: int, horizon: bool) -> bool:
        """Q502i: whether quiescence chases this quiet move under the player's repairs.

        The mirror of `agents.rs::chases`. ``horizon`` is the first quiescence ply.
        """
        if self.repairs & LOUD_DROPS and horizon and move_type(move) == MOVE_DROP:
            return True
        if not self.repairs & THREATS:
            return False
        engine = self.engine
        mover = state.side_to_move
        child = engine.apply(state, move)
        outcome = engine.terminal_prefix(child)
        if outcome is not None and not outcome.is_draw and outcome.winner == mover:
            return True
        return horizon and engine.in_check(child, 1 - mover)

    def _order(self, state: GameState, moves: list[int], ply: int) -> None:
        """Captures by what they take, then killers, then history.

        Captures-only ordering is close to no ordering at all in a game where captures
        are rare, and that is most of the corpus -- it is why a branching factor of 41
        bought only depth 2. Sorted on a tuple rather than a blended number: a single
        score would have to pick scale factors, and two orderings that disagree about a
        tie are two different games. Python's sort is stable and Rust's `sort_by` is too,
        so equal keys keep generation order on both sides.
        """
        board = state.board
        killers = self._killers[min(ply, MAX_PLY - 1)]
        squares = self._squares

        def rank(move: int) -> tuple[float, int, int]:
            origin, target, _, _ = decode_move(move)
            killer = 2 if move == killers[0] else (1 if move == killers[1] else 0)
            return (
                -self._capture_gain(board, move),
                -killer,
                -self._history.get(origin * squares + target, 0),
            )

        moves.sort(key=rank)

    def _remember(self, move: int, quiet: bool, depth: int, ply: int) -> None:
        """Record that this move caused a cutoff, so the next search tries it sooner."""
        slot = self._killers[min(ply, MAX_PLY - 1)]
        if quiet and slot[0] != move:
            slot[1] = slot[0]
            slot[0] = move
        origin, target, _, _ = decode_move(move)
        weight = max(depth, 0) ** 2
        index = origin * self._squares + target
        self._history[index] = self._history.get(index, 0) + weight

    @staticmethod
    def _terminal_value(outcome: Outcome, state: GameState) -> float:
        """A finished position's value, from the side to move's point of view.

        Discounted by ply so a forced win found sooner is preferred and a loss is
        delayed. Without it the search is indifferent between mate now and mate in six,
        and games drift instead of finishing.
        """
        if outcome.is_draw:
            return 0.0
        signed = WIN_SCORE - state.ply
        return signed if outcome.winner == state.side_to_move else -signed

    def _table_key(self, state: GameState) -> int:
        """The key this search's transposition table is filed under.

        Not ``state.key``. That key deliberately omits a *permanent* colouring, and the
        argument for the omission is sound for the rule it was made for: a permanent
        claim is irreversible, so it resets the no-progress clock, so it clears the
        repetition window, so two positions the repetition rule ever compares always
        share a colouring. A search tree is not a repetition window -- two branches
        routinely reach the same placement having claimed different squares, and once
        the evaluator reads territory those are two positions at different distances
        from a win. The mixing matches ``agents.rs::table_key`` exactly, so the two
        searches collide in the same places as well as agreeing in the rest.
        """
        if self.engine.spec.territory is None:
            return state.key
        mine = state.colour.count(1) * 0x9E3779B97F4A7C15
        theirs = state.colour.count(2) * 0xC2B2AE3D27D4EB4F
        return state.key ^ (mine & _U64) ^ (theirs & _U64)

    def _capture_gain(self, board: bytes, move: int) -> float:
        """Order captures first, most valuable victim first. Cheap and generic.

        Same trap as `_is_loud`, and the mirror of `agents.rs::capture_gain`. A
        rotation's target is its own square and a file-based castling's can be its own
        rook's, so reading the victim off the target orders both ahead of every quiet
        move by the value of the mover's own piece. That is not a tie-break difference --
        on `weathervane` most of the move list is rotations, so the two searches would
        cut off in different places and play different games under one name.
        """
        if move_type(move) in (MOVE_ROTATE, MOVE_CASTLE, MOVE_FOLD, MOVE_DUCK):
            return 0.0
        spec = self.engine.spec
        values = self.evaluator.values
        victim = board[move_to(move)]
        gain = values[spec.type_of(victim)] if victim else 0.0
        # Schema 13: a sweep's path victims are summed onto the target's, in ray order, as
        # `agents.rs::capture_gain` sums them.
        if move_type(move) == MOVE_SWEEP:
            for square in spec.sweep_path(board, move_from(move), move_to(move)):
                if board[square]:
                    gain += values[spec.type_of(board[square])]
        return gain


def _pair(announcer: int, announced: int, answer: int) -> tuple[int, int]:
    """A round's two moves in seat order, ``announced`` the announcer's."""
    return (announced, answer) if announcer == 0 else (answer, announced)


@register("minimax")
def _build_minimax(engine: Engine, options: dict[str, Any]) -> MinimaxAgent:
    refuse_unknown_options(options, OPTIONS, "minimax")
    if engine.spy:
        # Schema 27 (Q935b): a spy game is played on the built-in objective only, as a
        # round game is (`docs/plan-spy.md` §3.3).
        refuse_alternating_objective(options, "minimax", spy=True)
    return MinimaxAgent(
        engine,
        nodes=int(options.get("nodes", 20_000)),
        mobility=float(options.get("mobility", 0.05)),
        max_depth=int(options.get("max_depth", MAX_DEPTH)),
        goal=float(options.get("goal", GOAL_PIECES)),
        weights=_objective(options, engine.spec.hash),
        values=_values(options, engine.spec.hash),
        immune=float(options.get("immune", IMMUNE_PIECES)),
        net=require_net(options, engine.spec.hash),
        repairs=repairs_of(options),
    )


@register_native("minimax")
def _native_minimax(options: dict[str, Any], ruleset_hash: str = "") -> NativeSpec:
    refuse_unknown_options(options, OPTIONS, "minimax")
    return (
        "minimax",
        int(options.get("nodes", 20_000)),
        float(options.get("mobility", 0.05)),
        int(options.get("max_depth", MAX_DEPTH)),
        0,
        0.0,
        float(options.get("goal", GOAL_PIECES)),
        _objective(options, ruleset_hash),
        _values(options, ruleset_hash),
        float(options.get("immune", IMMUNE_PIECES)),
        require_net(options, ruleset_hash),
        repairs_of(options),
    )
